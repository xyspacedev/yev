"""Planned mode for the Opus synthetic pipeline: code fixes every item, Opus writes only the state.

A PlannedFamily's `plan` function builds Groups. A Group is one scene with one or more states (contrastive groups
have 2+ states that differ only where their briefs differ); every state carries the questions it must support,
each with options and a gold fixed in code. Writers get the scene, each state's brief and the questions with their
options, never the golds, and return one line per state: {"gid", "sid", "state"}.

Run layout (same as a document run, so check-prepare, check-score, nodoc-* and diff-* run unchanged):
  batches/planned-batch-XXX.json       the plan, golds included (the writer prompt tells writers not to open it)
  batches/planned-batch-XXX.prompt.md  the writer prompt; names the output file
  written/planned-batch-XXX.jsonl      writer output (the batch's "output" field, relative to the run)
  ingested.jsonl, reports/ingest.json, reports/ingest-dropped.json   from synth planned-ingest
check-score keeps groups with keep_groups (not documents.keep_docs) and writes final/synthetic_docs.jsonl.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from yev.generators.common import SYNTH_LICENCE
from yev.generators.llm import documents
from yev.generators.llm.families import DOMAINS
from yev.schema import Decision, Option, SchemaError

SOURCE = "synthetic_planned"
MODE = "planned"
BATCH_PREFIX = "planned-batch-"
KEEP_SINGLE_SHARE = documents.KEEP_DOC_SHARE


@dataclass(frozen=True)
class QSpec:
    name: str                      # unique within a state, snake_case
    type: str                      # "choice" | "noul" | "score"
    question: str                  # our own wording
    options: tuple[Option, ...]    # score: in scale order; noul: yes/no
    gold: str
    skill: str                     # free-text tag for reports, e.g. "injection_obeyed"


@dataclass(frozen=True)
class StateSpec:
    sid: str                       # unique within the group, snake_case, e.g. "obey"
    brief: str                     # facts the written state must make true (plain English, our words)
    questions: tuple[QSpec, ...]


@dataclass(frozen=True)
class Group:
    gid: str                       # unique within the family run
    scene: str                     # shared setting for every state: domain, people, tools, format notes
    states: tuple[StateSpec, ...]


@dataclass(frozen=True)
class PlannedFamily:
    name: str
    description: str               # writer-facing, our words, no benchmark names
    state_format: str              # writer-facing format rules (e.g. a [USER]/[AGENT]/[TOOL] log)
    tokens: tuple[int, int]        # per-state length range, checked at ingest
    per_batch: int                 # groups per writer batch
    plan: Callable[[random.Random, int, str], Group]   # (rng, index, domain) -> Group


def is_planned_spec(spec: dict) -> bool:
    return spec.get("mode") == MODE


# --- planning -------------------------------------------------------------------------------------

def _check_group(family: str, g: Group, seen_gids: set[str]) -> None:
    """Raise ValueError when a planned group could not become valid Decisions."""
    where = f"{family}:{g.gid}"
    if not g.gid or g.gid in seen_gids:
        raise ValueError(f"{where}: empty or duplicate gid")
    seen_gids.add(g.gid)
    if not g.states:
        raise ValueError(f"{where}: no states")
    sids = [s.sid for s in g.states]
    if len(set(sids)) != len(sids):
        raise ValueError(f"{where}: duplicate sid in {sids}")
    for s in g.states:
        if not s.questions:
            raise ValueError(f"{where}:{s.sid}: no questions")
        names = [q.name for q in s.questions]
        if len(set(names)) != len(names):
            raise ValueError(f"{where}:{s.sid}: duplicate question name in {names}")
        for q in s.questions:
            try:
                Decision(id=f"{where}:{s.sid}:{q.name}", type=q.type, state=s.brief, question=q.question,
                         options=list(q.options), gold=q.gold, family=family, source=SOURCE,
                         licence=SYNTH_LICENCE).validate()
            except SchemaError as e:
                raise ValueError(f"bad planned question: {e}") from e


def plan_batches(family: PlannedFamily, n_groups: int, seed: int) -> list[dict]:
    """n_groups groups from family.plan, per_batch to a writer batch. Each group gets a domain from DOMAINS."""
    if family.per_batch < 1:
        raise ValueError("per_batch must be at least 1")
    rng = random.Random(f"planned:{family.name}:{seed}")
    groups: list[Group] = []
    seen: set[str] = set()
    for i in range(n_groups):
        g = family.plan(rng, i, rng.choice(DOMAINS))
        _check_group(family.name, g, seen)
        groups.append(g)
    batches = []
    for k in range(0, len(groups), family.per_batch):
        batch_id = f"{BATCH_PREFIX}{k // family.per_batch:03d}"
        batches.append({
            "batch_id": batch_id,
            "family": family.name,
            "mode": MODE,
            "description": family.description,
            "state_format": family.state_format,
            "tokens": list(family.tokens),
            "output": f"written/{batch_id}.jsonl",
            "groups": [asdict(g) for g in groups[k:k + family.per_batch]],
        })
    return json.loads(json.dumps(batches))  # tuples -> lists, so a plan equals its JSON round trip


# --- writer prompt --------------------------------------------------------------------------------

def _words(tokens: int) -> int:
    words = tokens * documents.WORDS_PER_TOKEN
    return max(1, int(round(words, -1) if words >= 100 else round(words)))


def _question_lines(q: dict) -> str:
    opts = "\n".join(f"    - `{o['key']}`: {o['description']}" for o in q["options"])
    return f"  - ({q['type']}) {q['question']}\n{opts}"


def writer_prompt(batch: dict, output_path: str) -> str:
    """Scene, briefs and questions with options for every state of the batch. Never contains a gold."""
    lo, hi = batch["tokens"]
    n_states = sum(len(g["states"]) for g in batch["groups"])
    parts = []
    for g in batch["groups"]:
        lines = [f"### Group `{g['gid']}`", "", f"Scene: {g['scene']}", ""]
        for s in g["states"]:
            lines += [f"#### State `{s['sid']}` (group `{g['gid']}`)", "",
                      f"Brief: {s['brief']}", "",
                      f"Length: {lo:,} to {hi:,} tokens (about {_words(lo):,} to {_words(hi):,} words).", "",
                      "A reader of this state must be able to answer these questions from it alone:", ""]
            lines += [_question_lines(q) for q in s["questions"]]
            lines.append("")
        parts.append("\n".join(lines))
    groups = "\n".join(parts)
    example = json.dumps({"gid": batch["groups"][0]["gid"], "sid": batch["groups"][0]["states"][0]["sid"],
                          "state": "The whole state text, with real line breaks written as \\n."})
    return f"""# Write {n_states} states for {len(batch['groups'])} planned groups

You are writing training data for a small decision model. Each item shows the model a state (a text it must read),
one question about it and a list of options; the model must pick the single right option. The questions and
options are already fixed. Your only job is to write the state texts so that the facts in each brief hold.

## What these states are
{batch['description']}

## Format of a state
{batch['state_format']}

## Rules
- Make every fact in a state's brief true and visible in that state. A careful reader must be able to find each one
  in the text; nothing the brief requires may be left implied or only hinted at.
- Never state the verdict. Do not write what the right option is, and do not use label words such as "unsafe",
  "violation" or "correct answer". Show what happened; the reader judges it.
- All names are ones you invent: people, companies, products, tools, accounts, ticket numbers. Do not reuse examples
  you remember from public datasets or benchmarks, and do not name or quote any benchmark.
- Leave no arithmetic for the reader: if a brief involves an amount, a count, a date or a duration, state the
  figure the question needs directly.
- The states of one group share the scene: the same setting, people, tools and wording, and differ only where their
  briefs differ. Write the shared parts the same way in every state of the group.
- Each state stands alone: it must not mention the other states, the brief, the questions or these instructions.
- Write in the realistic voice of the setting, with plausible detail, inside the length range given for each state.
- Do not open any other file or directory; everything you need is in this prompt.

## Groups

{groups}
## Output format
Write exactly {n_states} lines to `{output_path}` using the Write tool: one JSON object per line, one line per state,
no prose, no code fences. Each object has exactly the keys `gid`, `sid` and `state`, for example:

{example}

Use the group and state ids exactly as given above, once each. Do not write questions or options.

When the file is written, reply with only the path and the number of lines written.
"""


# --- ingest ---------------------------------------------------------------------------------------

def _rows_for(batch: dict, group: dict, state_spec: dict, state: str, run_name: str,
              est: int | None) -> list[Decision]:
    family = batch["family"]
    cid = f"plan:{run_name}:{family}:{group['gid']}"
    total = sum(len(s["questions"]) for s in group["states"])
    out = []
    for q in state_spec["questions"]:
        out.append(Decision(
            id=f"{cid}:{state_spec['sid']}:{q['name']}", type=q["type"], state=state, question=q["question"],
            options=[Option(o["key"], o["description"]) for o in q["options"]], gold=q["gold"], family=family,
            source=SOURCE, licence=SYNTH_LICENCE, cluster_id=cid,
            metadata={"family_mode": MODE, "gid": group["gid"], "sid": state_spec["sid"], "skill": q["skill"],
                      "question_name": q["name"], "est_tokens": est, "questions_written": total,
                      "states_planned": len(group["states"])},
        ).validate())
    return out


def plan_rows(batch: dict, state: str | None = None, run_name: str = "plan") -> list[Decision]:
    """Decisions straight from the plan, with each state's text replaced by `state` (default: scene + brief).
    For checking planner text (scenes, briefs, questions, options) before any writer runs."""
    rows = []
    for g in batch["groups"]:
        for s in g["states"]:
            rows += _rows_for(batch, g, s, state if state is not None else f"{g['scene']}\n{s['brief']}",
                              run_name, None)
    return rows


def ingest_planned(run_dir: Path | str, count: Callable[[str], int] | None = None,
                   method: str | None = None) -> tuple[list[Decision], dict]:
    """Rows for every written state of a planned run, and a report with each drop reason counted.
    Drops: bad_shape, unknown_state (gid/sid not in the batch), duplicate_state (first kept), empty_state,
    too_short / too_long (outside the batch's token range)."""
    run = Path(run_dir)
    if count is None:
        count, method = documents.load_doc_counter()
    method = method or "custom"
    run_name = run.resolve().name
    stats: Counter = Counter()
    dropped: list[dict] = []
    rows: list[Decision] = []
    seen: set[tuple[str, str]] = set()
    for spec_path in sorted((run / "batches").glob("*.json")):
        batch = json.loads(spec_path.read_text())
        if not is_planned_spec(batch):
            continue
        plan = {(g["gid"], s["sid"]): (g, s) for g in batch["groups"] for s in g["states"]}
        stats["states_planned"] += len(plan)
        stats["groups_planned"] += len(batch["groups"])
        written = run / batch.get("output", f"written/{batch['batch_id']}.jsonl")
        if not written.exists():
            stats["missing_file"] += 1
            continue
        lo, hi = batch["tokens"]
        for obj in documents._objects(written.read_text(encoding="utf-8"), stats):
            stats["states_seen"] += 1

            def drop(why: str, **extra) -> None:
                stats[why] += 1
                dropped.append({"batch": batch["batch_id"], "reason": why, **extra})

            if not isinstance(obj, dict) or not isinstance(obj.get("state", ""), str) \
                    or "gid" not in obj or "sid" not in obj:
                drop("bad_shape")
                continue
            gid, sid = str(obj["gid"]).strip(), str(obj["sid"]).strip()
            if (gid, sid) not in plan:
                drop("unknown_state", gid=gid, sid=sid)
                continue
            if (gid, sid) in seen:
                drop("duplicate_state", gid=gid, sid=sid)
                continue
            seen.add((gid, sid))
            state = str(obj.get("state") or "").strip()
            if not state:
                drop("empty_state", gid=gid, sid=sid)
                continue
            est = count(state)
            if est < lo or est > hi:
                drop("too_short" if est < lo else "too_long", gid=gid, sid=sid, est_tokens=est)
                continue
            g, s = plan[(gid, sid)]
            got = _rows_for(batch, g, s, state, run_name, est)
            stats["states_kept"] += 1
            rows.extend(got)
    groups_kept = Counter(d.cluster_id for d in rows)
    stats["states_missing"] = stats["states_planned"] - stats["states_kept"]
    stats["groups_with_rows"] = len(groups_kept)
    stats["rows"] = len(rows)
    report = {**dict(stats), "token_count": method, "dropped": dropped}
    return rows, report


# --- check scoring --------------------------------------------------------------------------------

def keep_groups(survivors: list[Decision], ingested: list[Decision]) -> list[Decision]:
    """Contrastive groups (2+ planned states) keep nothing unless every ingested question survived the check;
    single-state groups keep their survivors if they are at least KEEP_SINGLE_SHARE (70%) of the questions
    planned for the group (metadata questions_written)."""
    n_ingested = Counter(d.cluster_id for d in ingested)
    info = {d.cluster_id: d.metadata or {} for d in ingested}
    alive = Counter(d.cluster_id for d in survivors)

    def states(cid: str) -> int:
        m = info.get(cid, {})
        return m.get("states_planned") or len({x.metadata.get("sid") for x in ingested if x.cluster_id == cid})

    keep = set()
    for cid in alive:
        if states(cid) >= 2:
            ok = alive[cid] == n_ingested[cid]
        else:
            written = info.get(cid, {}).get("questions_written") or n_ingested[cid]
            ok = alive[cid] >= KEEP_SINGLE_SHARE * written - 1e-9
        if ok:
            keep.add(cid)
    return [d for d in survivors if d.cluster_id in keep]


def score_planned_answers(ingested: list[Decision], key: dict, answers: list[dict],
                          survivors_out: list | None = None) -> tuple[list[Decision], Counter]:
    """Blind-check votes per question, then keep_groups."""
    from yev.generators.llm.check import score_answers

    survivors, stats = score_answers(ingested, key, answers, cluster_filter=None)
    if survivors_out is not None:
        survivors_out.extend(survivors)
    kept = keep_groups(survivors, ingested)
    stats["questions_survived"] = len(survivors)
    stats["groups_dropped"] = len({d.cluster_id for d in survivors} - {d.cluster_id for d in kept})
    stats["groups_kept"] = len({d.cluster_id for d in kept})
    stats["kept"] = len(kept)
    return kept, stats


def planned_pilot_report(specs: list[dict], ingest: dict, ingested: list[Decision], survivors: list[Decision],
                         kept: list[Decision]) -> dict:
    """Survival from plan to final, by family and skill. final_questions / final_docs keep the document-run names
    so the later steps can update them in place."""
    def by(rows, field):
        return Counter((d.metadata or {}).get(field) if field != "family" else d.family for d in rows)

    families = sorted({b["family"] for b in specs})
    skills = sorted({documents.skill_of(d) for d in ingested})
    return {
        "mode": MODE,
        "planned_groups": sum(len(b["groups"]) for b in specs),
        "planned_states": sum(len(g["states"]) for b in specs for g in b["groups"]),
        "ingested_states": ingest.get("states_kept", 0),
        "ingest_drop_reasons": {k: ingest[k] for k in ("bad_shape", "unknown_state", "duplicate_state",
                                                       "empty_state", "too_short", "too_long", "missing_file",
                                                       "bad_json") if ingest.get(k)},
        "questions_ingested": len(ingested),
        "questions_survived_check": len(survivors),
        "question_survival_rate": round(len(survivors) / max(len(ingested), 1), 4),
        "ingested_groups": len({d.cluster_id for d in ingested}),
        "final_docs": len({d.cluster_id for d in kept}),
        "final_questions": len(kept),
        "survival_by_family": {f: {"ingested": by(ingested, "family")[f], "survived_check": by(survivors, "family")[f],
                                   "final": by(kept, "family")[f]} for f in families},
        "survival_by_skill": {k: {"ingested": sum(documents.skill_of(d) == k for d in ingested),
                                  "survived_check": sum(documents.skill_of(d) == k for d in survivors),
                                  "final": sum(documents.skill_of(d) == k for d in kept)} for k in skills},
        "gold_balance": {t: dict(Counter(d.gold for d in kept if d.type == t)) for t in ("choice", "noul", "score")},
        "token_count_method": ingest.get("token_count"),
    }
