"""Fingerprints of every benchmark we report, so no training row shares a normalised 8-gram with any of them.

Each benchmark keeps a set of 8-byte blake2b hashes of its states' 8-grams (same tokenisation as the
DecideBench filter). `yev bench fingerprint` builds the file once; `yev filter --benchmarks` reads it.
"""

from __future__ import annotations

import hashlib
import json
import os
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
# Items shorter than NGRAM tokens are kept whole; from this length they also match inside longer states.
SHORT_CONTAINED_MIN = 4

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
    """8-gram hash sets per benchmark, in source order (the first overlapping one is reported)."""

    hashes: dict[str, set[int]]
    items: dict[str, int]
    meta: dict = field(default_factory=dict)
    # benchmark -> token count (1..NGRAM-1) -> hashes of the whole normalised text of its short items
    short: dict[str, dict[int, set[int]]] = field(default_factory=dict)
    _first: dict[int, str] | None = field(default=None, init=False, repr=False, compare=False)

    def names(self) -> list[str]:
        return list(self.hashes)

    def first_owner(self) -> dict[int, str]:
        if self._first is None:
            first: dict[int, str] = {}
            for name, hs in self.hashes.items():
                for h in hs:
                    first.setdefault(h, name)
            self._first = first
        return self._first

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "ngram": NGRAM,
            "hash": HASH_NAME,
            "meta": self.meta,
            "benchmarks": {n: {"items": self.items[n], "hashes": sorted(hs),
                               "short": {str(k): sorted(v) for k, v in sorted(self.short.get(n, {}).items())}}
                           for n, hs in self.hashes.items()},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path | str) -> "BenchmarkFingerprints":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("ngram") != NGRAM or data.get("hash") != HASH_NAME:
            raise ValueError(f"{path}: expected {NGRAM}-gram {HASH_NAME} fingerprints, got "
                             f"{data.get('ngram')}-gram {data.get('hash')}")
        b = data["benchmarks"]
        return cls(hashes={n: set(v["hashes"]) for n, v in b.items()},
                   items={n: v["items"] for n, v in b.items()}, meta=data.get("meta") or {},
                   short={n: {int(k): set(h) for k, h in (v.get("short") or {}).items()} for n, v in b.items()})


def build_benchmark_fingerprints(sources: dict[str, list[str]], meta: dict | None = None) -> BenchmarkFingerprints:
    """8-gram hashes per benchmark minus the boilerplate stoplist, plus whole-text hashes of short items."""
    item_counts: dict[str, Counter] = {}
    short: dict[str, dict[int, set[int]]] = {}
    items: dict[str, int] = {}
    for name, texts in sources.items():
        c: Counter = Counter()
        sh: dict[int, set[int]] = {}
        for t in texts:
            tokens = normalize_tokens(t)
            if 0 < len(tokens) < NGRAM:
                sh.setdefault(len(tokens), set()).add(gram_hash(tuple(tokens)))
            c.update({gram_hash(g) for g in ngrams(tokens, NGRAM)})
        item_counts[name], short[name], items[name] = c, sh, len(texts)
    n_benchmarks: Counter = Counter(h for c in item_counts.values() for h in c)
    hashes: dict[str, set[int]] = {}
    dropped: dict[str, int] = {}
    for name, c in item_counts.items():
        keep = {h for h, k in c.items() if k < STOP_MIN_ITEMS and n_benchmarks[h] < STOP_MIN_BENCHMARKS}
        hashes[name], dropped[name] = keep, len(c) - len(keep)
    meta = dict(meta or {})
    meta["stoplist"] = {"min_benchmarks": STOP_MIN_BENCHMARKS, "min_items": STOP_MIN_ITEMS, "dropped": dropped}
    meta["short_items"] = {"contained_min_tokens": SHORT_CONTAINED_MIN,
                           "counts": {n: sum(len(v) for v in sh.values()) for n, sh in short.items()}}
    return BenchmarkFingerprints(hashes=hashes, items=items, meta=meta, short=short)


def overlaps_any(d: Decision, bfp: BenchmarkFingerprints) -> str | None:
    """Name of the first benchmark (in source order) the row overlaps, else None.

    Overlap is a shared normalised 8-gram (state, question and options), or a short benchmark item
    (< 8 tokens) equal to the normalised state, or contained in it as a whole-token run when the item
    has at least SHORT_CONTAINED_MIN tokens.
    """
    owner = bfp.first_owner()
    hits = {owner[h] for h in text_hashes(decision_text(d)) if h in owner}
    if any(bfp.short.values()):
        state = normalize_tokens(d.state)
        whole = gram_hash(tuple(state)) if 0 < len(state) < NGRAM else None
        runs = {n: _run_hashes(state, n) for n in range(SHORT_CONTAINED_MIN, NGRAM)}
        for name, by_len in bfp.short.items():
            if whole is not None and whole in by_len.get(len(state), ()):
                hits.add(name)
                continue
            if any(not runs[n].isdisjoint(hs) for n, hs in by_len.items() if n >= SHORT_CONTAINED_MIN):
                hits.add(name)
    if not hits:
        return None
    order = {n: i for i, n in enumerate(bfp.hashes)}
    return min(hits, key=order.__getitem__)


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
