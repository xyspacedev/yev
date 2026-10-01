"""JevBench public items (github.com/fstandhartinger/jevbench, datasets/public/*.jsonl, MIT).

Items are natively Choice/Noul/Score: `question.instructions` is the question, `question.criteria` the options.
Choice: options in `labels` order, description = criteria[label]. Noul: labels are no/yes with criteria
true/false; our Noul is (yes, no) with yes = criteria["true"]. Score: options in scale order (`labels`,
criteria is a list aligned with it), gold = str(expected). `probability` items carry `gold_probs`,
kept as the soft target; accuracy is argmax against `expected`, as JevBench scores them.
"""
from __future__ import annotations

import json
from pathlib import Path

from jeff.bench import score as S
from jeff.bench.common import META_FILE, ROWS_FILE, chat_row, github_raw, write_json, write_rows
from jeff.schema import Decision, Option

NAME = "jevbench"
REPO = "fstandhartinger/jevbench"
COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"
TIERS = ("original", "easy", "hard")
REFERENCE = {
    "source": "jevbench results/v1.4.2.2/jevbench-v1.4.2.2-results.json, `public_accuracy` over the 231 public items",
    "public_accuracy": {
        "Imajev-4B": 0.8615, "Jev 1.13.0 (TypeSafe)": 0.8658, "Plumb-4B": 0.8961, "JevK5 v0.2.0": 0.8528,
        "decider-4b v2": 0.8355, "Raw Qwen3-4B-Instruct-2507 logits": 0.6970, "GPT-6 Luna (medium)": 0.9957,
    },
    "Imajev-4B_hard_public": 0.7207,
    "note": "The official JevBench Score needs sealed items and the speed/cost protocol; only public accuracy compares.",
}


def _key(label: str) -> str:
    return label.lower()  # a few size labels are upper case (S, M, L, XL); our keys are lower snake case


def to_decision(item: dict, tier: str) -> Decision:
    q = item["question"]
    kind, crit, labels = q["type"], q["criteria"], item["labels"]
    if kind == "noul":
        if sorted(labels) != ["no", "yes"] or sorted(crit) != ["false", "true"]:
            raise ValueError(f"{item['id']}: unexpected noul labels {labels} / criteria {sorted(crit)}")
        options = [Option("yes", crit["true"]), Option("no", crit["false"])]
        gold = str(item["expected"])
    elif kind == "score":
        if len(crit) != len(labels):
            raise ValueError(f"{item['id']}: {len(labels)} labels but {len(crit)} criteria")
        options = [Option(_key(lab), c) for lab, c in zip(labels, crit)]
        gold = _key(str(item["expected"]))
    elif kind == "choice":
        options = [Option(_key(lab), crit[lab]) for lab in labels]
        gold = _key(str(item["expected"]))
    else:
        raise ValueError(f"{item['id']}: unknown question type {kind!r}")
    soft = None
    gp = (item.get("provenance") or {}).get("gold_probs")
    if gp:
        soft = {_key(k): float(v) for k, v in gp.items()}
        total = sum(soft.values())
        soft = {k: v / total for k, v in soft.items()}
    state = item["state"]
    if not isinstance(state, str):  # some hard items carry a JSON-object state
        state = json.dumps(state, ensure_ascii=False)
    return Decision(
        id=f"jevbench:{item['id']}", type=kind, state=state, question=q["instructions"],
        options=options, gold=gold, family=item["family"], source=NAME, licence="mit", split="dev",
        cluster_id=item.get("group"), soft_gold=soft,
    ).validate()


def convert(items_by_tier: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for tier in TIERS:
        for item in items_by_tier.get(tier, []):
            rows.append(chat_row(to_decision(item, tier), {"tier": tier, "group": item.get("group")}))
    return rows


def build(out: Path) -> dict:
    items = {}
    for tier in TIERS:
        raw = github_raw(REPO, COMMIT, f"datasets/public/{tier}.jsonl").decode("utf-8")
        items[tier] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows = convert(items)
    n = write_rows(out / ROWS_FILE, rows)
    meta = {
        "name": NAME, "source": f"https://github.com/{REPO}/tree/{COMMIT}/datasets/public", "revision": COMMIT,
        "licence": "MIT", "n": n, "n_by_tier": {t: len(items[t]) for t in TIERS},
        "notes": [
            "Options in the benchmark's `labels` order (no shuffling); Score options in scale order.",
            "Noul: (yes, no) with yes = criteria['true'], no = criteria['false'].",
            "Upper-case labels (S, M, L, XL) lower-cased to fit our key schema.",
            "probability-family items carry gold_probs as the soft target; accuracy is argmax vs `expected`.",
            "Rendered with jeff.format.user_turn (training format).",
            f"{sum(not isinstance(i['state'], str) for t in TIERS for i in items[t])} items have a JSON-object "
            "state; it is serialised with json.dumps (ensure_ascii=False) into the state string.",
        ],
    }
    write_json(out / META_FILE, meta)
    return meta


def metrics(preds: list[dict], variant: str | None = None) -> dict:
    pair_acc, n_pairs = S.pair_accuracy(preds, lambda p: p["bench"].get("group"))
    return {
        **S.common(preds),
        "by_tier": S.by_field(preds, lambda p: p["bench"].get("tier")),
        "by_family": S.by_field(preds, lambda p: p["family"]),
        "by_type": S.by_field(preds, lambda p: p["type"]),
        "pair_accuracy": pair_acc,
        "n_pairs": n_pairs,
        "reference": REFERENCE,
    }
