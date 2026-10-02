"""General replay: BoolQ (Noul) and ARC / CommonsenseQA multiple choice."""

from __future__ import annotations

import random

from yev.schema import Decision, Option
from yev.sources.base import slug

BOOLQ_OPTIONS = [
    Option("yes", "The passage shows the answer is yes."),
    Option("no", "The passage shows the answer is no."),
]


def convert_boolq(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    question, passage = (row["question"] or "").strip(), (row["passage"] or "").strip()
    if not question or not passage:
        return []
    return [
        Decision(
            id=str(i),
            type="noul",
            state=f"Passage: {passage}",
            question=question[0].upper() + question[1:] + "?",
            options=list(BOOLQ_OPTIONS),
            gold="yes" if row["answer"] else "no",
            family="replay",
            source="",
            licence="",
        )
    ]


def convert_multiple_choice(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    question = (row["question"] or "").strip()
    raw_labels, texts = row["choices"]["label"], row["choices"]["text"]
    if not question or not row["answerKey"] or row["answerKey"] not in raw_labels:
        return []
    return [
        Decision(
            id=str(row["id"]),
            type="choice",
            state=f"Question: {question}",
            question="Which answer is correct?",
            options=[Option(slug(lab), text.strip()) for lab, text in zip(raw_labels, texts)],
            gold=slug(row["answerKey"]),
            family="replay",
            source="",
            licence="",
        )
    ]
