"""DecideBench: the one-time final test (spec §2 rules 1 and 5).

`build` is the only code path that reads the test set. It replicates the official harness
(github.com/choyiny/decidebench, `decidebench/prompts.py` and `decidebench/fewshot.py`, read at commit
HARNESS_COMMIT; nothing from that checkout is vendored):

- the user turn is compact JSON `{"state","question","options":[{"label","key","description"}]}` with
  `separators=(",", ":")`, options in file order, letters A.. in that order;
- "with examples" (the leaderboard "Accuracy" column, chat-model path): the item's template examples
  (one solved example per option, from HF `data/examples.jsonl`, which is the harness's
  `examples/*.jsonl` concatenated in sorted file order) shuffled with `random.Random(item_id)`, each
  given as an earlier user turn (same JSON) and an assistant turn (its gold letter);
- "zero-shot": no examples.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from jeff import decidebench as db
from jeff.bench import score as S
from jeff.bench.common import BENCH_SPLIT, META_FILE, write_json, write_rows
from jeff.format import LETTERS, SYSTEM_PROMPT

NAME = "decidebench"
HARNESS_REPO = "choyiny/decidebench"
HARNESS_COMMIT = "975e87bc815f3c7d8034b95bece6fd259eea168c"
VARIANTS = {"examples": "rows_examples.chat.jsonl", "zeroshot": "rows_zeroshot.chat.jsonl"}
REFERENCE = {
    "source": "https://huggingface.co/spaces/choyiny/decidebench-leaderboard (v1.0, measured 2026-09-28/29)",
    "examples": {
        "imajev-4b": {"accuracy": 0.950, "pair_accuracy": 0.905},
        "TEV (Tev1-4B-experimental)": {"accuracy": 0.9275, "pair_accuracy": 0.860},
        "JEV (AI Space)": {"accuracy": 0.980, "pair_accuracy": 0.960},
        "Qwen3-8B, no thinking": {"accuracy": 0.905, "pair_accuracy": 0.815},
    },
    "zeroshot": {
        "TEV (Tev1-4B-experimental)": {"accuracy": 0.900},
        "JEV (AI Space)": {"accuracy": 0.9825},
    },
}


def user_message(state: str, question: str, options: list[dict]) -> str:
    return json.dumps(
        {"state": state, "question": question,
         "options": [{"label": LETTERS[i], "key": o["key"], "description": o["description"]}
                     for i, o in enumerate(options)]},
        ensure_ascii=False, separators=(",", ":"),
    )


def template_key(item: dict) -> tuple:
    return (item["category"], item["question"], tuple((o["key"], o["description"]) for o in item["options"]))


def example_pool(examples: list[dict]) -> dict[tuple, list[dict]]:
    """Examples grouped by template, in file order (the harness's `default_pool`)."""
    pool: dict[tuple, list[dict]] = {}
    for e in examples:
        pool.setdefault(template_key(e), []).append(e)
    return pool


def shots_for(item: dict, pool: dict[tuple, list[dict]]) -> list[dict]:
    """The harness's `fewshot.for_item`: the template's examples in a per-item seeded order."""
    shots = list(pool.get(template_key(item), []))
    random.Random(item["id"]).shuffle(shots)
    return shots


def to_row(item: dict, shots: list[dict]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in shots:
        messages.append({"role": "user", "content": user_message(ex["state"], ex["question"], ex["options"])})
        messages.append({"role": "assistant", "content": LETTERS[[o["key"] for o in ex["options"]].index(ex["gold"])]})
    messages.append({"role": "user", "content": user_message(item["state"], item["question"], item["options"])})
    keys = [o["key"] for o in item["options"]]
    letters = {LETTERS[i]: k for i, k in enumerate(keys)}
    return {
        "id": f"decidebench:{item['id']}",
        "messages": messages,
        "answer": LETTERS[keys.index(item["gold"])],
        "letters": letters,
        "target": {L: float(k == item["gold"]) for L, k in letters.items()},
        "weight": 1.0,
        "type": "choice",
        "family": item["category"],
        "source": "decidebench",
        "cluster_id": item["pair_id"],
        "edit_type": None,
        "split": BENCH_SPLIT,
        "bench": {"pair_id": item["pair_id"], "difficulty": item["difficulty"], "n_examples": len(shots)},
    }


def convert(test_items: list[dict], examples: list[dict]) -> dict[str, list[dict]]:
    pool = example_pool(examples)
    with_ex, zero = [], []
    for item in test_items:
        shots = shots_for(item, pool)
        if sorted(e["gold"] for e in shots) != sorted(o["key"] for o in item["options"]):
            raise ValueError(f"{item['id']}: examples cover {sorted(e['gold'] for e in shots)}, "
                             f"options are {sorted(o['key'] for o in item['options'])}")
        with_ex.append(to_row(item, shots))
        zero.append(to_row(item, []))
    return {"examples": with_ex, "zeroshot": zero}


def _examples_raw() -> list[dict]:
    """HF `data/examples.jsonl` as raw dicts in file order (Decision options keep that order)."""
    return [{"id": d.id.split(":", 1)[1], "category": d.family, "state": d.state, "question": d.question,
             "options": [{"key": o.key, "description": o.description} for o in d.options], "gold": d.gold}
            for d in db.load_examples()]


def build(out: Path) -> dict:
    test_items = db.load_test_raw()  # the single sanctioned read of the test set
    examples = _examples_raw()
    rows = convert(test_items, examples)
    counts = {v: write_rows(out / VARIANTS[v], rows[v]) for v in VARIANTS}
    meta = {
        "name": NAME,
        "source": f"https://huggingface.co/datasets/{db.REPO_ID}",
        "revision": db.REVISION,
        "harness": f"https://github.com/{HARNESS_REPO}/tree/{HARNESS_COMMIT} (prompts.py, fewshot.py read, not vendored)",
        "licence": "CC-BY-4.0 (data), MIT (harness)",
        "n": len(test_items),
        "files": {VARIANTS[v]: counts[v] for v in VARIANTS},
        "notes": [
            "Options in file order, letters A.. in that order (no shuffling).",
            "User turns are compact JSON (separators=(',', ':')), exactly as the harness's build_user_message; "
            "training rows used json.dumps default separators.",
            "rows_examples: one solved example per option from HF data/examples.jsonl, order "
            "random.Random(item_id).shuffle over the template's examples in file order, as earlier "
            "user/assistant turns (harness chat-model path, leaderboard 'Accuracy' column).",
            "rows_zeroshot: no examples (leaderboard 'Zero-shot accuracy' column).",
        ],
    }
    write_json(out / META_FILE, meta)
    return meta


def _pair(p: dict) -> str:
    return p["bench"].get("pair_id") or p["cluster_id"]


def _macro(preds: list[dict]) -> dict:
    """Harness `label_metrics` + `macro`: one-vs-rest P/R/F1 per (family, label), mean over labels then families."""
    counts: dict[tuple[str, str], list[int]] = {}
    for p in preds:
        fam = p["family"]
        c = counts.setdefault((fam, p["gold"]), [0, 0, 0])
        if p["pred"] == p["gold"]:
            c[0] += 1
        else:
            c[2] += 1
            counts.setdefault((fam, p["pred"]), [0, 0, 0])[1] += 1
    per: dict[str, list[tuple[float, float, float]]] = {}
    for (fam, _), (tp, fp, fn) in counts.items():
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per.setdefault(fam, []).append((pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0))
    fams = [tuple(sum(x[i] for x in v) / len(v) for i in range(3)) for v in per.values()]
    if not fams:
        return {"macro_p": None, "macro_r": None, "macro_f1": None}
    return {k: sum(f[i] for f in fams) / len(fams) for i, k in enumerate(("macro_p", "macro_r", "macro_f1"))}


def metrics(preds: list[dict], variant: str | None = None) -> dict:
    pair_acc, n_pairs = S.pair_accuracy(preds, _pair)
    hard = [p for p in preds if p["bench"].get("difficulty") == "hard"]
    hard_pair_acc, n_hard_pairs = S.pair_accuracy(hard, _pair)
    rep = {
        **S.common(preds),
        "ci95": S.pair_bootstrap_ci(preds, _pair),
        "pair_accuracy": pair_acc,
        "n_pairs": n_pairs,
        "by_family": S.by_field(preds, lambda p: p["family"]),
        "hard": {"n": len(hard), "accuracy": S.accuracy(hard), "pair_accuracy": hard_pair_acc, "n_pairs": n_hard_pairs},
        **_macro(preds),
        "variant": variant,
        "reference": REFERENCE,
    }
    return rep
