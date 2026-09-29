"""Intent (Banking77, CLINC150) and emotion (GoEmotions) adapters."""

from __future__ import annotations

import random

from jeff.schema import Decision, Option
from jeff.sources.base import Converter, humanize, pick_distractors, slug

NONE_OPTION = Option("none_of_these", "None of the other options matches the request.")
NONE_RATE = 0.3  # share of in-scope items that also list "none of these"


def _intent_option(label: str) -> Option:
    return Option(slug(label), f"The request is about: {humanize(label).lower()}.")


def intent_converter(text_col: str, label_col: str, oos_label: str | None = None) -> Converter:
    def convert(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
        text = (row[text_col] or "").strip()
        label = row[label_col]
        if not text or label is None:
            return []
        in_scope = [lab for lab in labels if lab != oos_label]
        if label == oos_label:
            chosen = rng.sample(in_scope, rng.randint(2, 5))
            options = [_intent_option(lab) for lab in chosen] + [NONE_OPTION]
            gold = NONE_OPTION.key
        else:
            with_none = oos_label is not None and rng.random() < NONE_RATE
            chosen = pick_distractors(label, in_scope, rng, k_max=5 if with_none else 6)
            options = [_intent_option(lab) for lab in chosen] + ([NONE_OPTION] if with_none else [])
            gold = slug(label)
        rng.shuffle(options)
        return [
            Decision(
                id=str(i),
                type="choice",
                state=f"Customer message: {text}",
                question="Which intent best matches the customer's message?",
                options=options,
                gold=gold,
                family="intent",
                source="",
                licence="",
            )
        ]

    return convert


def _emotion_option(label: str) -> Option:
    if label == "neutral":
        return Option("neutral", "The writer expresses no particular emotion.")
    return Option(slug(label), f"The writer mainly expresses {label}.")


def convert_go_emotions(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    text = (row["text"] or "").strip()
    if not text or len(row["labels"]) != 1:
        return []
    [label] = row["labels"]
    chosen = pick_distractors(label, labels, rng)
    return [
        Decision(
            id=str(row["id"]),
            type="choice",
            state=f"Text: {text}",
            question="Which emotion does the text mainly express?",
            options=[_emotion_option(lab) for lab in chosen],
            gold=slug(label),
            family="sentiment",
            source="",
            licence="",
        )
    ]
