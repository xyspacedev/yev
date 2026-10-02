"""Fingerprints of every benchmark we report, so no training row overlaps any of their items.

Each benchmark keeps its states' 8-gram hashes (8-byte blake2b, same tokenisation as the DecideBench
filter), each with the indices of the items it came from. A row overlaps benchmark B when it shares
ITEM_MIN_SHARED distinct 8-grams with one item of B (DecideBench: a single 8-gram, main spec section 2).
`yev bench fingerprint` builds the file once; `yev filter --benchmarks` reads it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from yev.filters.contamination import NGRAM, decision_text, ngrams, normalize_tokens
from yev.schema import Decision

HASH_NAME = "blake2b-8"
# Boilerplate stoplist: an 8-gram found in this many distinct benchmarks, or in this many items of one
# benchmark, is stock phrasing ("is there anything else i can assist you with"), not item content.
STOP_MIN_BENCHMARKS = 2
STOP_MIN_ITEMS = 20
# Items of up to SHORT_EXACT_MAX tokens also keep a hash of their whole normalised text: a row whose
# normalised state equals one is dropped. Items shorter than NGRAM tokens (no 8-grams) of at least
# SHORT_CONTAINED_MIN tokens also match inside longer states as a whole-token run.
SHORT_EXACT_MAX = 9
SHORT_CONTAINED_MIN = 6
# The exact check runs on the whole state, each line, each line after a leading "<label>:" of up to
# 4 words ("Customer message: ...", "User: ..."), and each double-quoted span (straight or curly).
LABEL_PREFIX_RE = re.compile(r"^\s*(?:[^\s:]+\s+){0,3}[^\s:]+\s*:\s*(.*)$")
# Line, label-stripped and quoted spans count only from this many tokens: one- and two-token lines such
# as "Regards," or "false" equal CLINC150 items but are not overlaps. The whole state has no floor.
SPAN_MIN_TOKENS = 3
QUOTED_RE = re.compile(r'"([^"\n]+)"|\u201c([^\u201d\n]+)\u201d')
# A row overlaps a benchmark when it shares this many distinct 8-grams with one single item of it, or
# all of that item's kept 8-grams when it has fewer (items of 8-9 tokens, or mostly stoplisted) ...
ITEM_MIN_SHARED = 3
# ... except for these benchmarks, where one shared 8-gram is enough (main spec section 2). They are
# also exempt from the stoplist, so overlaps_any alone is a complete DecideBench check.
STRICT_BENCHMARKS = frozenset({"decidebench"})

# The 38 benchmarks that count in Decision Index 0.2.1, with the pinned source ids from the kit's
# docs/suite.md (GitHub repos, Hub datasets, or the download host) plus Hub mirrors of the same data.
# Full table with revisions and files: docs/research/decision-index-0.2.1-sources.md.
DECISION_INDEX_SOURCES: dict[str, tuple[str, ...]] = {
    "BFCL": ("gorilla-llm/gorilla", "gorilla-llm/Berkeley-Function-Calling-Leaderboard"),
    "ToolRet": ("mangopy/ToolRet-Queries", "mangopy/ToolRet-Tools", "mangopy/tool-retrieval-benchmark"),
    "API-Bank": ("AlibabaResearch/DAMO-ConvAI", "liminghao1630/API-Bank"),
    "Home appliances": ("apolinario/decision-index",),
    "When2Call": ("nvidia/When2Call",),
    "BANKING77": ("PolyAI-LDN/task-specific-datasets", "PolyAI/banking77", "legacy-datasets/banking77"),
    "CLINC150+OOS": ("clinc/oos-eval", "clinc/clinc_oos"),
    "BRIGHT": ("xlangai/BRIGHT",),
    "Amazon ESCI": ("amazon-science/esci-data",),
    "PhishNChips": ("AreLit/PhishNChips", "anisselbd/jev-phishing-bench"),
    "HoVer": ("hover-nlp/hover",),
    "ContractNLI": ("stanfordnlp/contract-nli",),
    "ANLI": ("facebook/anli",),
    "WinoGrande": ("allenai/winogrande",),
    "HellaSwag": ("Rowan/hellaswag",),
    "ACOS": ("NUSTM/ACOS",),
    "FinEntity": ("yixuantt/FinEntity",),
    "iSarcasmEval": ("iabufarha/iSarcasmEval",),
    "VAST": ("emilyallaway/zero-shot-stance",),
    "NLI4CT": ("ai-systems/Task-2-SemEval-2024",),
    "RAGTruth": ("ParticleMedia/RAGTruth",),
    "GPQA Diamond": ("idavidrein/gpqa",),
    "GSM8K": ("openai/gsm8k", "gsm8k"),
    "ChessBench": ("google-deepmind/searchless_chess",),
    "MuSR": ("Zayne-sprague/MuSR", "TAUR-Lab/MuSR"),
    "SATA-Bench": ("sata-bench/sata-bench",),
    "CRUXEval": ("facebookresearch/cruxeval", "cruxeval-org/cruxeval"),
    "CLadder": ("causalNLP/cladder",),
    "HLE": ("cais/hle",),
    "MMLU-Pro": ("TIGER-Lab/MMLU-Pro",),
    "BBH": ("suzgunmirac/BIG-Bench-Hard", "lukaemon/bbh"),
    "BPoMP": ("zenodo:7299879",),
    "Humicroedit": ("cs.rochester.edu/u/nhossain/semeval-2020-task-7",),
    "POP909-CL": ("AndyWeasley2004/POP909-CL-Dataset",),
    "cfcolor": ("dgp.toronto.edu/~donovan/cfcolor",),
    "ForecastBench": ("forecastingresearch/forecastbench-datasets",),
    "Habermas Machine": ("google-deepmind/habermas_machine",),
    "New Yorker": ("jmhessel/newyorker_caption_contest",),
}

# Shown on the board but not counted in the index; still never a training source.
DECISION_INDEX_SHOWN_SOURCES: dict[str, tuple[str, ...]] = {
    "RouterBench": ("withmartian/routerbench",),
    "SGD": ("google-research-datasets/dstc8-schema-guided-dialogue", "google-research-datasets/schema_guided_dstc8"),
    "MMLU": ("cais/mmlu",),
    "ARC-Easy": ("allenai/ai2_arc",),
    "ARC-Challenge": ("allenai/ai2_arc",),
    "SimpleBench": ("simple-bench/SimpleBench",),
}


def decision_index_ids() -> set[str]:
    """Every source id (lower-cased) behind a Decision Index 0.2.1 benchmark, counted or shown."""
    return {i.lower() for ids in (*DECISION_INDEX_SOURCES.values(), *DECISION_INDEX_SHOWN_SOURCES.values()) for i in ids}


def gram_hash(gram: tuple[str, ...]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(gram).encode(), digest_size=8).digest(), "big")


def text_hashes(text: str) -> set[int]:
    return {gram_hash(g) for g in ngrams(normalize_tokens(text), NGRAM)}


def _run_hashes(tokens: list[str], n: int) -> set[int]:
    return {gram_hash(tuple(tokens[i : i + n])) for i in range(len(tokens) - n + 1)}


@dataclass
class BenchmarkFingerprints:
    """Per benchmark: 8-gram hash -> indices of the items holding it, in source order."""

    grams: dict[str, dict[int, list[int]]]
    items: dict[str, int]
    thresholds: dict[str, int]
    # benchmark -> kept 8-gram count of each item (per-item threshold = min(threshold, count))
    item_grams: dict[str, list[int]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    # benchmark -> token count (1..SHORT_EXACT_MAX) -> hashes of the whole normalised text of its short items
    short: dict[str, dict[int, set[int]]] = field(default_factory=dict)
    _index: dict | None = field(default=None, init=False, repr=False, compare=False)

    def names(self) -> list[str]:
        return list(self.grams)

    def index(self) -> dict[int, list[tuple[int, list[int]]]]:
        """hash -> [(benchmark position, item indices)] across all benchmarks."""
        if self._index is None:
            idx: dict[int, list[tuple[int, list[int]]]] = {}
            for b, owners in enumerate(self.grams.values()):
                for h, its in owners.items():
                    idx.setdefault(h, []).append((b, its))
            self._index = idx
        return self._index

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        benchmarks = {}
        for n, owners in self.grams.items():
            hs = sorted(owners)
            benchmarks[n] = {
                "items": self.items[n],
                "threshold": self.thresholds[n],
                "item_grams": self.item_grams[n],
                "hashes": hs,
                # parallel to hashes: one item index, or a list when the gram is in several items
                "owners": [owners[h][0] if len(owners[h]) == 1 else owners[h] for h in hs],
                "short": {str(k): sorted(v) for k, v in sorted(self.short.get(n, {}).items())},
            }
        data = {"ngram": NGRAM, "hash": HASH_NAME, "meta": self.meta, "benchmarks": benchmarks}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path | str) -> "BenchmarkFingerprints":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("ngram") != NGRAM or data.get("hash") != HASH_NAME or "item_grams" not in next(
                iter(data.get("benchmarks", {}).values()), {"item_grams": []}):
            raise ValueError(f"{path}: expected item-level {NGRAM}-gram {HASH_NAME} fingerprints; "
                             "rebuild with yev bench fingerprint")
        b = data["benchmarks"]
        return cls(
            grams={n: {h: (o if isinstance(o, list) else [o]) for h, o in zip(v["hashes"], v["owners"])}
                   for n, v in b.items()},
            items={n: v["items"] for n, v in b.items()},
            thresholds={n: v["threshold"] for n, v in b.items()},
            item_grams={n: v["item_grams"] for n, v in b.items()},
            meta=data.get("meta") or {},
            short={n: {int(k): set(h) for k, h in (v.get("short") or {}).items()} for n, v in b.items()},
        )


def build_benchmark_fingerprints(sources: dict[str, list[str]], meta: dict | None = None) -> BenchmarkFingerprints:
    """Item-level 8-gram hashes per benchmark minus the boilerplate stoplist, plus short-item hashes."""
    owners_by: dict[str, dict[int, list[int]]] = {}
    short: dict[str, dict[int, set[int]]] = {}
    items: dict[str, int] = {}
    for name, texts in sources.items():
        owners: dict[int, list[int]] = {}
        sh: dict[int, set[int]] = {}
        for i, t in enumerate(texts):
            tokens = normalize_tokens(t)
            if 0 < len(tokens) <= SHORT_EXACT_MAX:
                sh.setdefault(len(tokens), set()).add(gram_hash(tuple(tokens)))
            for h in {gram_hash(g) for g in ngrams(tokens, NGRAM)}:
                owners.setdefault(h, []).append(i)
        owners_by[name], short[name], items[name] = owners, sh, len(texts)
    n_benchmarks: Counter = Counter(h for owners in owners_by.values() for h in owners)
    grams: dict[str, dict[int, list[int]]] = {}
    dropped: dict[str, int] = {}
    item_grams: dict[str, list[int]] = {}
    rescued: dict[str, int] = {}
    for name, owners in owners_by.items():
        if name in STRICT_BENCHMARKS:
            grams[name] = owners
        else:
            grams[name] = {h: its for h, its in owners.items()
                           if len(its) < STOP_MIN_ITEMS and n_benchmarks[h] < STOP_MIN_BENCHMARKS}
        dropped[name] = len(owners) - len(grams[name])
        # An item that would lose every gram to the stoplist keeps them all, for itself only.
        had = [0] * items[name]
        for its in owners.values():
            for i in its:
                had[i] += 1
        kept = [0] * items[name]
        for its in grams[name].values():
            for i in its:
                kept[i] += 1
        lost_all = {i for i in range(items[name]) if had[i] and not kept[i]}
        for h, its in owners.items():
            mine = [i for i in its if i in lost_all]
            if mine:
                grams[name].setdefault(h, [])
                grams[name][h] = sorted(set(grams[name][h]) | set(mine))
        rescued[name] = len(lost_all)
        per_item = [0] * items[name]
        for its in grams[name].values():
            for i in its:
                per_item[i] += 1
        item_grams[name] = per_item
    thresholds = {n: 1 if n in STRICT_BENCHMARKS else ITEM_MIN_SHARED for n in sources}
    meta = dict(meta or {})
    meta["item_min_shared"] = ITEM_MIN_SHARED
    meta["thresholds"] = thresholds
    meta["stoplist"] = {"min_benchmarks": STOP_MIN_BENCHMARKS, "min_items": STOP_MIN_ITEMS,
                        "exempt": sorted(STRICT_BENCHMARKS), "dropped": dropped, "rescued_items": rescued}
    meta["short_items"] = {"exact_max_tokens": SHORT_EXACT_MAX, "contained_min_tokens": SHORT_CONTAINED_MIN,
                           "counts": {n: sum(len(v) for v in sh.values()) for n, sh in short.items()}}
    return BenchmarkFingerprints(grams=grams, items=items, thresholds=thresholds, item_grams=item_grams,
                                 meta=meta, short=short)


def _exact_candidates(state: str) -> set[tuple[int, int]]:
    """(token count, hash) of every short span the exact check compares with short items."""
    out = set()
    tokens = normalize_tokens(state)
    if 0 < len(tokens) <= SHORT_EXACT_MAX:
        out.add((len(tokens), gram_hash(tuple(tokens))))
    texts = []
    for line in state.splitlines():
        texts.append(line)
        m = LABEL_PREFIX_RE.match(line)
        if m:
            texts.append(m.group(1))
    texts += [a or b for a, b in QUOTED_RE.findall(state)]
    for t in texts:
        tokens = normalize_tokens(t)
        if SPAN_MIN_TOKENS <= len(tokens) <= SHORT_EXACT_MAX:
            out.add((len(tokens), gram_hash(tuple(tokens))))
    return out


def overlaps_any(d: Decision, bfp: BenchmarkFingerprints) -> str | None:
    """Name of the first benchmark (in source order) the row overlaps, else None.

    The row overlaps benchmark B when its text (state, question and options) shares, with one single
    item of B, at least min(B's threshold, that item's kept 8-gram count) distinct 8-grams
    (ITEM_MIN_SHARED; 1 for DecideBench); or when an item of B of at most SHORT_EXACT_MAX tokens
    equals the normalised state, one of its lines (with or without a leading "<label>:" of up to 4
    words) or a double-quoted span in it (spans of SPAN_MIN_TOKENS or more); or contains an item of SHORT_CONTAINED_MIN..NGRAM-1 tokens as a
    whole-token run.
    """
    names = bfp.names()
    thr = [bfp.thresholds[n] for n in names]
    item_grams = [bfp.item_grams[n] for n in names]
    hits: set[int] = set()
    counts: dict[int, Counter] = {}
    idx = bfp.index()
    for h in text_hashes(decision_text(d)):
        for b, its in idx.get(h, ()):
            if b in hits:
                continue
            if thr[b] <= 1:
                hits.add(b)
                continue
            c = counts.setdefault(b, Counter())
            per_item = item_grams[b]
            for i in its:
                c[i] += 1
                if c[i] >= min(thr[b], per_item[i]):
                    hits.add(b)
                    break
    if any(bfp.short.values()):
        state = normalize_tokens(d.state)
        wholes = _exact_candidates(d.state)
        runs = {n: _run_hashes(state, n) for n in range(SHORT_CONTAINED_MIN, NGRAM)}
        for b, name in enumerate(names):
            by_len = bfp.short.get(name, {})
            if any(h in by_len.get(n, ()) for n, h in wholes):
                hits.add(b)
            elif any(not runs[n].isdisjoint(hs) for n, hs in by_len.items() if SHORT_CONTAINED_MIN <= n < NGRAM):
                hits.add(b)
    return names[min(hits)] if hits else None


# --- Sources read by `yev bench fingerprint` ------------------------------------------------------

def chat_row_sources(bench_dir: Path | str) -> dict[str, list[str]]:
    """Item states of every `<bench_dir>/<name>/rows*.chat.jsonl`, one per row id, keyed by directory name."""
    from yev.bench.common import item_state, read_rows

    out: dict[str, list[str]] = {}
    for d in sorted(p for p in Path(bench_dir).iterdir() if p.is_dir()):
        files = sorted(d.glob("rows*.chat.jsonl"))
        if not files:
            continue
        states: dict[str, str] = {}
        for f in files:
            for r in read_rows(f):
                states.setdefault(r["id"], item_state(r))
        out[d.name] = list(states.values())
    return out


def flatten_strings(obj) -> list[str]:
    """Every string value in a JSON object, depth first (keys are skipped)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in flatten_strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in flatten_strings(v)]
    return []


# Pinned revisions (HF Hub, read 2026-10-02). See docs/research/decision-index-0.2.1-sources.md.
WORKFLOWEVALS = {
    "workflowevals_invoice_processing": ("typesafe/evalsafe-invoice-processing", "6beeb2d2acd65c086c835022f5f4d7434114cafc"),
    "workflowevals_customer_service": ("typesafe/evalsafe-customer-service", "b1342f5a704587dbc465867c38c2694348ff86e4"),
    "workflowevals_agent_trace_observability": ("typesafe/evalsafe-agent-trace-observability", "8635540973910a92465fe2bc53e195375aa6e1a8"),
    "workflowevals_security_incidents": ("typesafe/evalsafe-security-incidents", "fbe1ea5c69cf494157fd23f2002a0d9d9a418443"),
}
WORKFLOWEVALS_FILE = "data/cases.parquet"

# Decision Index test sets whose source dataset is in our pool (plan 5 Ruling 1).
DI_TEST_SETS = {
    "di_banking77_test": ("legacy-datasets/banking77", "f54121560de48f2852f90be299010d1d6dc612ec",
                          ["data/test-00000-of-00001.parquet"]),
    "di_clinc150_test": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec",
                         ["plus/test-00000-of-00001.parquet"]),
    "di_arc_test": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453",
                    ["ARC-Easy/test-00000-of-00001.parquet", "ARC-Challenge/test-00000-of-00001.parquet"]),
    "di_gsm8k_test": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866",
                      ["main/test-00000-of-00001.parquet"]),
    "di_finentity": ("yixuantt/FinEntity", "e86e5451a1fce4731fe07c710649a229d92e5cab", ["FinEntity.json"]),
}


def _download(repo: str, revision: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _di_texts(name: str, path: Path) -> list[str]:
    if name == "di_finentity":
        return [r["content"] for r in json.loads(path.read_text(encoding="utf-8"))]
    import pandas as pd

    df = pd.read_parquet(path)
    if name == "di_arc_test":
        return ["\n".join([q, *c["text"]]) for q, c in zip(df["question"], df["choices"])]
    if name == "di_gsm8k_test":
        return df["question"].tolist()
    return df["text"].tolist()


def hub_sources() -> dict[str, list[str]]:
    """WorkflowEvals cases and the targeted Decision Index test sets, from pinned Hub revisions (network)."""
    import pandas as pd

    out: dict[str, list[str]] = {}
    for name, (repo, rev) in WORKFLOWEVALS.items():
        df = pd.read_parquet(_download(repo, rev, WORKFLOWEVALS_FILE), columns=["input_json"])
        out[name] = ["\n".join(flatten_strings(json.loads(s))) for s in df["input_json"]]
    for name, (repo, rev, files) in DI_TEST_SETS.items():
        out[name] = [t for f in files for t in _di_texts(name, _download(repo, rev, f))]
    return out


def hub_meta() -> dict:
    return {name: {"repo": repo, "revision": rev, "files": [WORKFLOWEVALS_FILE]} for name, (repo, rev) in WORKFLOWEVALS.items()} | {
        name: {"repo": repo, "revision": rev, "files": files} for name, (repo, rev, files) in DI_TEST_SETS.items()}
