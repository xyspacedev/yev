"""skill_atlas: Claude-made decisions about public agent skills, read from a local skill-atlas checkout.

Three families, all built from verdicts a model wrote in skill-atlas (never from its TF-IDF
classifier, `scope_pred.jsonl` or dedupe groups):

- `skill_scope`   portable / vendor / internal / unclear, from `pipeline/out/scope/scope-*.jsonl`.
- `skill_judge`   which of two skills in a judged group is better, from `judge/judge*.jsonl`.
- `skill_filing`  which org-chart leaf a skill belongs in, from the Opus label passes.

Model provenance (skill-atlas docs): the README and the explorer credit the judge, the fallback
(`fb*`) passes and the thin-department pass to Opus and the seed to Sonnet; the herd pass is
unrecorded (ROADMAP's calibration chose Sonnet for it). Two batches the ROADMAP and the relabel
plan explicitly record as Sonnet dispatches are excluded: `judge4` and `labels_fb3`. The scope
pass's model is not recorded anywhere; it is kept (the brief asks for the family) and marked
`model: "unrecorded"` in metadata so it can be filtered later.

Skill text is data: states carry the skill's name, description and body, whitespace-normalised
and truncated, with nothing added.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from yev import licences
from yev.schema import Decision, Option, write_jsonl

NAME = "skill_atlas"
DEFAULT_ROOT = Path(os.environ.get("YEV_SKILL_ATLAS", "../skill-atlas"))

SPDX_KEEP = ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC")

# Label files whose rows count as Opus decisions, in precedence order (later wins per skill).
OPUS_LABEL_FILES = ("labels_fb.jsonl", "labels_fb2.jsonl", "labels_fb4.jsonl", "labels_fb5.jsonl", "labels_thin.jsonl")
OPUS_LABEL_METHODS = frozenset({"fallback", "thin"})
# Judge prefixes in precedence order (first complete verdict wins). `judge4` was a Sonnet dispatch.
JUDGE_PREFIXES = ("judge", "judge7", "judge6", "judge5")
EXCLUDED_JUDGE_PREFIXES = ("judge4",)

# Herd flags that drop a skill: safety/injection flags, plus low-quality hard flags. Soft metadata
# (`internal`, `likely-*`, `variant`, `no-license`, `stale`) does not; the licence rule covers
# unlicensed repos. Any skill listed in flags.jsonl is always dropped.
SAFETY_HERD_FLAGS = frozenset({"prompt-injection", "guardrail-bypass", "harmful-content", "credential-exposure"})
QUALITY_HERD_FLAGS = frozenset({"example-path", "placeholder", "fixture", "benchmark", "no-description"})
DROP_HERD_FLAGS = SAFETY_HERD_FLAGS | QUALITY_HERD_FLAGS

SCOPE_PER_GOLD_CAP = 1500
SCOPE_CAP = 4000
FILING_CAP = 4000
JUDGE_CAP = 2500
JUDGE_MIN_GAP = 15
JUDGE_MAX_PAIRS = 3
FILING_TARGET = 5

SCOPE_OPTIONS = [
    Option("portable", "Portable: a general-purpose skill that any project or company could adopt."),
    Option("vendor", "Vendor: about using a public product, library or platform; useful to anyone who uses it."),
    Option("internal", "Internal: only meaningful inside its home repository or organisation."),
    Option("unclear", "Unclear: the skill's scope cannot be told from its text."),
]
SCOPE_ALIASES = {"general": "portable", "generic": "portable", "reusable": "portable", "public": "portable",
                 "product": "vendor", "library": "vendor", "platform": "vendor", "tool": "vendor",
                 "repo-internal": "internal", "repo_internal": "internal", "project-internal": "internal",
                 "private": "internal", "unknown": "unclear", "ambiguous": "unclear"}
SCOPE_QUESTION = ("Is this agent skill a portable general-purpose skill, tied to a vendor's product, "
                  "internal to one organisation, or unclear?")

JUDGE_QUESTION = ("Which of the two skills would better help an agent do the job it claims? Judge on trigger "
                  "clarity, instruction quality, specificity, safety, progressive disclosure and maintenance.")
JUDGE_OPTIONS = [Option("a", "Skill A is the better skill."), Option("b", "Skill B is the better skill.")]

FILING_QUESTION = "Which category of the company org chart does this agent skill belong in?"


@dataclass
class Stats:
    candidates: int = 0
    dropped_provenance: int = 0
    dropped_flags: int = 0
    dropped_licence: int = 0
    dropped_no_text: int = 0
    dropped_other: int = 0
    kept: int = 0
    reasons: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "reasons"}
        if self.reasons:
            d["other_reasons"] = dict(sorted(self.reasons.items()))
        return d


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def skill_id(repo: str, path: str) -> str:
    return hashlib.sha1(f"{repo}\n{path}".encode("utf-8")).hexdigest()[:16]


def truncate(text: str, cap: int) -> str:
    """If longer than cap, cut at a word boundary and append ' …' so the result is at most cap chars."""
    if len(text) <= cap:
        return text
    head = text[: cap - 2]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 0 else head).rstrip() + " …"


def render_skill(skill: dict, cap: int) -> str:
    name = re.sub(r"\s+", " ", skill.get("name") or "").strip()
    desc = re.sub(r"\s+", " ", skill.get("description") or "").strip()
    body = re.sub(r"\s+", " ", skill.get("body") or "").strip()
    return truncate(f"Skill: {name}\nDescription: {desc}\nBody: {body}", cap)


def _key(leaf: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")


class Ontology:
    def __init__(self, root: Path):
        data = json.loads((root / "pipeline" / "ontology.json").read_text(encoding="utf-8"))
        self.version = data.get("version")
        self.leaves: dict[str, dict] = {}
        self.team_leaves: dict[str, list[str]] = defaultdict(list)
        self.dept_leaves: dict[str, list[str]] = defaultdict(list)
        for dept in data.get("departments", []) + data.get("bins", []):
            for team in dept.get("teams", []):
                for cat in team.get("categories", []):
                    self.leaves[cat["id"]] = {"name": cat["name"], "description": cat.get("description", ""),
                                              "team": team["id"], "dept": dept["id"]}
                    self.team_leaves[team["id"]].append(cat["id"])
                    self.dept_leaves[dept["id"]].append(cat["id"])
        mig_path = root / "pipeline" / "ontology_migrations.json"
        self.leaf_map = json.loads(mig_path.read_text(encoding="utf-8")).get("leaf_map", {}) if mig_path.exists() else {}

    def resolve(self, leaf: str | None) -> str | None:
        """Follow merges to a current leaf; split parents and unknown ids resolve to None."""
        seen: set[str] = set()
        while leaf and leaf in self.leaf_map and leaf not in seen:
            seen.add(leaf)
            leaf = self.leaf_map[leaf]
        return leaf if leaf in self.leaves else None

    def option(self, leaf: str) -> Option:
        info = self.leaves[leaf]
        return Option(_key(leaf), f"{info['name']}: {info['description']}".strip())


class Atlas:
    """Loads skill-atlas outputs and applies the licence and flag rules."""

    def __init__(self, root: Path):
        self.root = root
        self.out = root / "pipeline" / "out"
        self.herd = {r["skill_id"]: r for r in _read_jsonl(self.out / "herd.jsonl") if r.get("skill_id")}
        self.flagged = {r["skill_id"] for r in _read_jsonl(self.out / "flags.jsonl") if r.get("skill_id")}
        self._text: dict[str, dict] = {}

    def licence(self, sid: str) -> str | None:
        """SPDX id if the skill's repo licence is on the keep-list and yev allows it, else None."""
        spdx = (self.herd.get(sid) or {}).get("license")
        if spdx in SPDX_KEEP and licences.is_allowed(spdx.lower()):
            return spdx
        return None

    def is_flagged(self, sid: str) -> bool:
        herd_flags = set((self.herd.get(sid) or {}).get("flags") or [])
        return sid in self.flagged or bool(herd_flags & DROP_HERD_FLAGS)

    def screen(self, sid: str) -> str | None:
        """None if usable, else the drop reason ('flags' or 'licence')."""
        if self.is_flagged(sid):
            return "flags"
        if self.licence(sid) is None:
            return "licence"
        return None

    def load_text(self, ids: set[str]) -> None:
        """Fill name/description/body/repo/path for `ids` from corpus.parquet, else batch inputs."""
        need = set(ids) - set(self._text)
        if not need:
            return
        parquet = self.out / "corpus.parquet"
        if parquet.exists():
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(parquet)
            for batch in pf.iter_batches(batch_size=2048, columns=["repo", "path", "name", "description", "body"]):
                cols = batch.to_pydict()
                for i, (repo, path) in enumerate(zip(cols["repo"], cols["path"])):
                    sid = skill_id(repo, path)
                    if sid in need:
                        self._text[sid] = {"repo": repo, "path": path, "name": cols["name"][i] or "",
                                           "description": cols["description"][i] or "", "body": cols["body"][i] or ""}
            need -= set(self._text)
        if need:
            for p in sorted(glob.glob(str(self.out / "batches" / "*.input.jsonl"))):
                for r in _read_jsonl(Path(p)):
                    sid = r.get("skill_id")
                    if sid in need and r.get("repo") and r.get("path"):
                        self._text[sid] = {"repo": r["repo"], "path": r["path"], "name": r.get("name") or "",
                                           "description": r.get("description") or "", "body": r.get("body_head") or ""}
                        need.discard(sid)

    def text(self, sid: str) -> dict | None:
        return self._text.get(sid)

    def attribution(self, sid: str) -> dict:
        t = self._text[sid]
        return {"repo": t["repo"], "path": t["path"], "skill_id": sid, "licence": self.licence(sid)}


def _rng(seed: int, *parts: str) -> random.Random:
    return random.Random(":".join([str(seed), NAME, *parts]))


def _decision(id: str, state: str, question: str, options: list[Option], gold: str, family: str,
              licence: str, metadata: dict) -> Decision:
    return Decision(id=f"{NAME}:{id}", type="choice", state=state, question=question, options=options, gold=gold,
                    family=family, source=NAME, licence=licence.lower(), metadata=metadata).validate()


# ---------------------------------------------------------------- scope

def scope_verdicts(atlas: Atlas) -> dict[str, dict]:
    """skill_id -> {scope, file}; later files win."""
    got: dict[str, dict] = {}
    for p in sorted(glob.glob(str(atlas.out / "scope" / "scope-*.jsonl"))):
        for r in _read_jsonl(Path(p)):
            sid, scope = r.get("skill_id"), str(r.get("scope") or "").strip().lower()
            scope = SCOPE_ALIASES.get(scope, scope)
            if sid and scope in {o.key for o in SCOPE_OPTIONS}:
                got[sid] = {"scope": scope, "file": Path(p).name}
    return got


def build_scope(atlas: Atlas, seed: int) -> tuple[list[Decision], Stats]:
    stats = Stats()
    verdicts = scope_verdicts(atlas)
    usable = []
    for sid in sorted(verdicts):
        stats.candidates += 1
        reason = atlas.screen(sid)
        if reason:
            setattr(stats, f"dropped_{reason}", getattr(stats, f"dropped_{reason}") + 1)
        else:
            usable.append(sid)
    atlas.load_text(set(usable))
    out = []
    for sid in usable:
        if atlas.text(sid) is None:
            stats.dropped_no_text += 1
            continue
        v = verdicts[sid]
        lic = atlas.licence(sid)
        meta = {**atlas.attribution(sid), "stage": "s13_scope", "method": "scope", "model": "unrecorded",
                "batch": v["file"]}
        out.append(_decision(f"scope:{sid}", render_skill(atlas.text(sid), SCOPE_CAP), SCOPE_QUESTION,
                             list(SCOPE_OPTIONS), v["scope"], "skill_scope", lic, meta))
    by_gold: dict[str, list[Decision]] = defaultdict(list)
    for d in out:
        by_gold[d.gold].append(d)
    capped = []
    for gold in sorted(by_gold):
        rows = by_gold[gold]
        if len(rows) > SCOPE_PER_GOLD_CAP:
            stats.reasons[f"capped_{gold}"] = len(rows) - SCOPE_PER_GOLD_CAP
            stats.dropped_other += len(rows) - SCOPE_PER_GOLD_CAP
            rows = _rng(seed, "scope_cap", gold).sample(rows, SCOPE_PER_GOLD_CAP)
        capped.extend(rows)
    out = sorted(capped, key=lambda d: d.id)
    stats.kept = len(out)
    return out, stats


# ---------------------------------------------------------------- judge

def judge_verdicts(atlas: Atlas) -> tuple[dict[str, tuple[str, list[dict]]], int]:
    """group_id -> (prefix, ranking) from the first complete verdict in JUDGE_PREFIXES order.

    Also returns how many groups had a complete verdict only from an excluded (non-Opus) prefix.
    """
    expected = {g["group_id"]: set(g.get("members") or []) for g in _read_jsonl(atlas.out / "judge_groups.jsonl")}
    by_prefix: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for p in sorted(glob.glob(str(atlas.out / "judge" / "judge*.jsonl"))):
        prefix = Path(p).name.rsplit("-", 1)[0]
        for o in _read_jsonl(Path(p)):
            gid = o.get("group_id")
            if gid not in expected:
                continue
            ranking = [r for r in o.get("ranking") or [] if isinstance(r, dict)
                       and r.get("skill_id") in expected[gid] and isinstance(r.get("overall"), (int, float))]
            if {r["skill_id"] for r in ranking} == expected[gid]:
                by_prefix[prefix][gid] = ranking
    got: dict[str, tuple[str, list[dict]]] = {}
    for gid in sorted(expected):
        for prefix in JUDGE_PREFIXES:
            if gid in by_prefix.get(prefix, {}):
                got[gid] = (prefix, by_prefix[prefix][gid])
                break
    excluded_only = sum(1 for gid in expected if gid not in got
                        and any(gid in by_prefix.get(p, {}) for p in EXCLUDED_JUDGE_PREFIXES))
    return got, excluded_only


def _pairs(ranking: list[dict]) -> list[tuple[dict, dict, float]]:
    """All (hi, lo, gap) with gap >= JUDGE_MIN_GAP, largest gaps first, deterministic ties."""
    rows = sorted(ranking, key=lambda r: r["skill_id"])
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            gap = abs(a["overall"] - b["overall"])
            if gap >= JUDGE_MIN_GAP:
                hi, lo = (a, b) if a["overall"] > b["overall"] else (b, a)
                pairs.append((hi, lo, gap))
    pairs.sort(key=lambda t: (-t[2], t[0]["skill_id"], t[1]["skill_id"]))
    return pairs


def build_judge(atlas: Atlas, seed: int) -> tuple[list[Decision], Stats]:
    stats = Stats()
    verdicts, excluded_only = judge_verdicts(atlas)
    stats.dropped_provenance = excluded_only
    chosen: list[tuple[str, str, dict, dict, float]] = []
    seen_pairs: set[tuple[str, str]] = set()
    screened: dict[str, str | None] = {}
    for gid, (prefix, ranking) in verdicts.items():
        n = 0
        for hi, lo, gap in _pairs(ranking):
            if n >= JUDGE_MAX_PAIRS:
                break
            stats.candidates += 1
            key = tuple(sorted((hi["skill_id"], lo["skill_id"])))
            if key in seen_pairs:
                stats.dropped_other += 1
                stats.reasons["duplicate_pair"] += 1
                continue
            reasons = [screened.setdefault(s, atlas.screen(s)) for s in key]
            reason = "flags" if "flags" in reasons else "licence" if "licence" in reasons else None
            if reason:
                setattr(stats, f"dropped_{reason}", getattr(stats, f"dropped_{reason}") + 1)
                continue
            seen_pairs.add(key)
            chosen.append((gid, prefix, hi, lo, gap))
            n += 1
    atlas.load_text({s for _, _, hi, lo, _ in chosen for s in (hi["skill_id"], lo["skill_id"])})
    out = []
    for gid, prefix, hi, lo, gap in chosen:
        if atlas.text(hi["skill_id"]) is None or atlas.text(lo["skill_id"]) is None:
            stats.dropped_no_text += 1
            continue
        rng = _rng(seed, "judge", gid, hi["skill_id"], lo["skill_id"])
        a, b = (hi, lo) if rng.random() < 0.5 else (lo, hi)
        gold = "a" if a is hi else "b"
        state = (f"=== Skill A ===\n{render_skill(atlas.text(a['skill_id']), JUDGE_CAP)}\n=== End of Skill A ===\n\n"
                 f"=== Skill B ===\n{render_skill(atlas.text(b['skill_id']), JUDGE_CAP)}\n=== End of Skill B ===")
        lics = {atlas.licence(a["skill_id"]), atlas.licence(b["skill_id"])}
        # Both permissive; the row carries the one with more conditions, and metadata carries each.
        lic = "Apache-2.0" if "Apache-2.0" in lics else sorted(lics)[0]
        skills = [{"slot": slot, **atlas.attribution(r["skill_id"]), "overall": r["overall"]}
                  for slot, r in (("A", a), ("B", b))]
        meta = {"repo": [s["repo"] for s in skills], "path": [s["path"] for s in skills],
                "skill_id": [s["skill_id"] for s in skills], "licence": [s["licence"] for s in skills],
                "skills": skills, "group_id": gid, "gap": gap, "stage": "judge", "method": prefix, "model": "opus"}
        pair_id = "~".join(sorted((a["skill_id"], b["skill_id"])))
        out.append(_decision(f"judge:{pair_id}", state, JUDGE_QUESTION, list(JUDGE_OPTIONS), gold,
                             "skill_judge", lic, meta))
    stats.kept = len(out)
    return out, stats


# ---------------------------------------------------------------- filing

def opus_labels(atlas: Atlas) -> tuple[dict[str, dict], int]:
    """skill_id -> label row from the Opus label files (later file wins); plus non-Opus label skills."""
    got: dict[str, dict] = {}
    for name in OPUS_LABEL_FILES:
        for r in _read_jsonl(atlas.out / name):
            if r.get("skill_id") and r.get("method") in OPUS_LABEL_METHODS:
                got[r["skill_id"]] = {**r, "_file": name}
    others: set[str] = set()
    for p in sorted(glob.glob(str(atlas.out / "labels*.jsonl"))):
        if Path(p).name in OPUS_LABEL_FILES:
            continue
        for r in _read_jsonl(Path(p)):
            sid = r.get("skill_id")
            if sid and sid not in got:
                others.add(sid)
    return got, len(others)


def build_filing(atlas: Atlas, seed: int) -> tuple[list[Decision], Stats]:
    stats = Stats()
    onto = Ontology(atlas.root)
    labels, non_opus = opus_labels(atlas)
    stats.dropped_provenance = non_opus
    audit = {r["skill_id"]: onto.resolve(r.get("leaf")) for r in _read_jsonl(atlas.out / "labels_audit.jsonl")
             if r.get("skill_id")}
    usable = []
    for sid in sorted(labels):
        stats.candidates += 1
        r = labels[sid]
        leaf = onto.resolve(r.get("leaf"))
        if leaf is None:
            stats.dropped_other += 1
            stats.reasons["leaf_not_in_ontology"] += 1
            continue
        confirmed = audit.get(sid) == leaf
        if r.get("confidence") != "high" and not confirmed:
            stats.dropped_other += 1
            stats.reasons["not_high_or_audit_confirmed"] += 1
            continue
        reason = atlas.screen(sid)
        if reason:
            setattr(stats, f"dropped_{reason}", getattr(stats, f"dropped_{reason}") + 1)
            continue
        usable.append((sid, leaf, confirmed))
    atlas.load_text({sid for sid, _, _ in usable})
    out = []
    for sid, leaf, confirmed in usable:
        if atlas.text(sid) is None:
            stats.dropped_no_text += 1
            continue
        r = labels[sid]
        rng = _rng(seed, "filing", sid)
        chosen = [leaf]
        secondary = onto.resolve(r.get("secondary"))
        if secondary and secondary != leaf:
            chosen.append(secondary)
        info = onto.leaves[leaf]
        for pool in (onto.team_leaves[info["team"]], onto.dept_leaves[info["dept"]]):
            cands = sorted(set(pool) - set(chosen))
            rng.shuffle(cands)
            chosen += cands[: max(0, FILING_TARGET - len(chosen))]
        if len(chosen) < 4:
            stats.dropped_other += 1
            stats.reasons["too_few_options"] += 1
            continue
        chosen = chosen[:6]
        rng.shuffle(chosen)
        meta = {**atlas.attribution(sid), "stage": "fallback" if r["method"] == "fallback" else "s14_thin",
                "method": r["method"], "label_file": r["_file"], "model": "opus",
                "confidence": r.get("confidence"), "audit_confirmed": confirmed,
                "ontology_version": r.get("ontology_version"), "options_ontology_version": onto.version,
                "recorded_leaf": r.get("leaf"), "recorded_secondary": r.get("secondary")}
        out.append(_decision(f"filing:{sid}", render_skill(atlas.text(sid), FILING_CAP), FILING_QUESTION,
                             [onto.option(x) for x in chosen], _key(leaf), "skill_filing",
                             atlas.licence(sid), meta))
    stats.kept = len(out)
    return out, stats


# ---------------------------------------------------------------- entry point

FAMILIES = (("skill_scope", build_scope), ("skill_judge", build_judge), ("skill_filing", build_filing))


def build_all(root: Path | str = DEFAULT_ROOT, seed: int = 0) -> tuple[dict[str, list[Decision]], dict] | None:
    """Build all three families. Returns None (and prints why) when the skill-atlas root is missing."""
    root = Path(root)
    if not (root / "pipeline" / "out").is_dir():
        print(f"{NAME}: skipped, no skill-atlas checkout at {root}")
        return None
    atlas = Atlas(root)
    families: dict[str, list[Decision]] = {}
    stats: dict[str, dict] = {}
    for name, fn in FAMILIES:
        decisions, st = fn(atlas, seed)
        families[name] = decisions
        stats[name] = st.as_dict()
    return families, stats


def attribution_rows(families: dict[str, list[Decision]]) -> list[dict]:
    seen: dict[str, dict] = {}
    for decisions in families.values():
        for d in decisions:
            m = d.metadata or {}
            for s in m.get("skills") or [{k: m[k] for k in ("repo", "path", "licence", "skill_id")}]:
                seen[s["skill_id"]] = {"repo": s["repo"], "path": s["path"], "licence": s["licence"],
                                       "skill_id": s["skill_id"]}
    return [seen[k] for k in sorted(seen)]


ATTRIBUTION_FILE = "ATTRIBUTION.json"


def write_attribution(out: Path | str, rows: list[dict]) -> Path:
    """Write the attribution rows as one JSON array (not *.jsonl, so Decision globs never read it)."""
    path = Path(out) / ATTRIBUTION_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def run(out: Path | str, root: Path | str = DEFAULT_ROOT, seed: int = 0) -> dict:
    """Write one JSONL per family plus ATTRIBUTION.json into `out`; return stats (empty when skipped)."""
    built = build_all(root, seed)
    if built is None:
        return {"skipped": True, "reason": f"no skill-atlas checkout at {root}"}
    families, stats = built
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, decisions in families.items():
        stats[name]["written"] = write_jsonl(out / f"{name}.jsonl", decisions)
    rows = attribution_rows(families)
    write_attribution(out, rows)
    stats["attribution_skills"] = len(rows)
    return stats
