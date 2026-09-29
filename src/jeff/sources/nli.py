"""Claim-vs-evidence adapters. All map to one three-way option set."""

from __future__ import annotations

import random

from jeff.schema import Decision, Option

EVIDENCE_OPTIONS = {
    "supported": "The evidence shows the claim is true.",
    "refuted": "The evidence shows the claim is false.",
    "not_enough_info": "The evidence does not settle whether the claim is true or false.",
}

FRAMINGS = [
    ("Claim: {h}\nEvidence: {p}", "What does the evidence say about the claim?"),
    ("Premise: {p}\nHypothesis: {h}", "Given the premise, is the hypothesis true, false, or undetermined?"),
]

NLI_LABELS = {"entailment": "supported", "contradiction": "refuted", "neutral": "not_enough_info"}
VITAMINC_LABELS = {"SUPPORTS": "supported", "REFUTES": "refuted", "NOT ENOUGH INFO": "not_enough_info"}


def evidence_decision(
    *, id: str, premise: str, hypothesis: str, gold: str, rng: random.Random, cluster_id: str | None = None
) -> Decision:
    state_tpl, question = rng.choice(FRAMINGS)
    keys = list(EVIDENCE_OPTIONS)
    rng.shuffle(keys)
    return Decision(
        id=id,
        type="choice",
        state=state_tpl.format(p=premise.strip(), h=hypothesis.strip()),
        question=question,
        options=[Option(k, EVIDENCE_OPTIONS[k]) for k in keys],
        gold=gold,
        family="evidence",
        source="",
        licence="",
        cluster_id=cluster_id,
    )


def _blank(*texts: str | None) -> bool:
    return any(not (t or "").strip() for t in texts)


def _nli(id: str, premise, hypothesis, gold, rng, cluster_id=None) -> list[Decision]:
    if gold is None or _blank(premise, hypothesis):
        return []
    return [evidence_decision(id=id, premise=premise, hypothesis=hypothesis, gold=gold, rng=rng, cluster_id=cluster_id)]


def convert_vitaminc(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    return _nli(
        str(row["unique_id"]), row["evidence"], row["claim"], VITAMINC_LABELS.get(row["label"]), rng,
        cluster_id=str(row["case_id"]),
    )


def convert_wanli(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    return _nli(str(row["id"]), row["premise"], row["hypothesis"], NLI_LABELS.get(row["gold"]), rng)


def convert_multi_nli(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    return _nli(str(row["pairID"]), row["premise"], row["hypothesis"], NLI_LABELS.get(row["label"]), rng)


def convert_snli_cf(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    return _nli(str(row["idx"]), row["premise"], row["hypothesis"], NLI_LABELS.get(row["label"]), rng)


def convert_negation(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    if _blank(row["anchor"], row["entailment"], row["negative"]):
        return []
    return [
        evidence_decision(id=f"{i}a", premise=row["anchor"], hypothesis=row["entailment"], gold="supported",
                          rng=rng, cluster_id=str(i)),
        evidence_decision(id=f"{i}b", premise=row["anchor"], hypothesis=row["negative"], gold="refuted",
                          rng=rng, cluster_id=str(i)),
    ]
