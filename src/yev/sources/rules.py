"""Rule-application adapters: Eikos decisions (already typed) and RuleTaker."""

from __future__ import annotations

import random

from yev.schema import Decision, Option
from yev.sources.base import humanize, slug


def convert_eikos(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    if row["lang"] != "English" or row["split"] != "train":
        return []
    qtype = row["question_type"]
    raw_labels = [o["label"] for o in row["options"]]
    keys = [slug(label) for label in raw_labels]
    if any(not k for k in keys) or len(set(keys)) != len(keys):
        return []
    if row["expected"] not in raw_labels:
        return []
    gold = keys[raw_labels.index(row["expected"])]
    options = [
        Option(k, (o.get("description") or "").strip() or humanize(o["label"]))
        for k, o in zip(keys, row["options"])
    ]
    soft = None
    probs = row.get("target_probs")
    if probs and len(probs) == len(keys) and sum(probs) > 0:
        total = sum(probs)
        soft = {k: p / total for k, p in zip(keys, probs)}
    if qtype == "noul":
        if set(keys) != {"yes", "no"}:
            return []
        options.sort(key=lambda o: o.key != "yes")
    return [
        Decision(
            id=str(row["id"]),
            type=qtype,
            state=row["state"],
            question=row["instructions"],
            options=options,
            gold=gold,
            family=f"eikos_{row['family']}",
            source="",
            licence="",
            soft_gold=soft,
        )
    ]


RULETAKER_OPTIONS = [
    Option("yes", "The facts and rules show the statement is true."),
    Option("no", "The facts and rules do not show the statement is true."),
]


def convert_ruletaker(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    if row["label"] not in ("entailment", "not entailment"):
        return []
    return [
        Decision(
            id=str(i),
            type="noul",
            state=f"Facts and rules:\n{row['context'].strip()}",
            question=f'Based only on these facts and rules, is this statement true? "{row["question"].strip()}"',
            options=list(RULETAKER_OPTIONS),
            gold="yes" if row["label"] == "entailment" else "no",
            family="rules",
            source="",
            licence="",
        )
    ]
