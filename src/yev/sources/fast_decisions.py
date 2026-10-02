"""fastino/fast-decisions (Apache-2.0): published development split, one config per domain.

Trained on at the user's request; Fastino's test split is private, so this does
not contaminate their benchmark.
"""

from __future__ import annotations

import random

from yev.schema import Decision, Option
from yev.sources.base import humanize, pick_distractors, slug

FAST_DECISIONS_CONFIGS = (
    "agent_handoff", "banking_intent", "benefits_request", "clinic_request", "document_type",
    "email_triage", "news_topic", "paper_field", "product_feedback", "restaurant_review",
    "review_sentiment", "screen_tags", "sports_recap", "support_intent", "support_topic",
    "ticket_route", "travel_request",
)

ORDERED_SCALES = frozenset(
    {
        ("negative", "neutral", "positive"),
        ("low", "normal", "high", "critical"),
    }
)


def _decision(id: str, type: str, state: str, question: str, options: list[Option], gold: str) -> Decision:
    return Decision(
        id=id, type=type, state=state, question=question, options=options, gold=gold,
        family="fast_decisions", source="", licence="",
    )


def _yes_no(subject: str) -> list[Option]:
    return [Option("yes", f"Yes: {subject} applies."), Option("no", f"No: {subject} does not apply.")]


def _head(i: int, text: str, head: dict, rng: random.Random) -> list[Decision]:
    task, labels, gold = head["task"], head["labels"], head["true_label"]
    name = humanize(task).lower()
    # Guard against slug collisions that would produce duplicate keys
    if len({slug(label) for label in labels}) != len(labels):
        return []
    if head["multi_label"]:
        return [
            _decision(
                f"{i}:{task}:{slug(label)}", "noul", text,
                f"Does the {name} of this input include {humanize(label).lower()}?",
                _yes_no(humanize(label).lower()), "yes" if label in gold else "no",
            )
            for label in labels
        ]
    if len(gold) != 1 or gold[0] not in labels:
        return []
    [g] = gold
    id = f"{i}:{task}"
    if sorted(label.lower() for label in labels) == ["no", "yes"]:
        return [_decision(id, "noul", text, f"For this input: {name}?", _yes_no(name), g.lower())]
    if tuple(labels) in ORDERED_SCALES:
        options = [Option(slug(label), f"The {name} is {humanize(label).lower()}.") for label in labels]
        return [_decision(id, "score", text, f"What is the {name} of this input?", options, slug(g))]
    chosen = pick_distractors(g, labels, rng)
    options = [Option(slug(label), f"The {name} is {humanize(label).lower()}.") for label in chosen]
    return [_decision(id, "choice", text, f"What is the {name} of this input?", options, slug(g))]


def convert_fast_decisions(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    text = (row.get("input") or "").strip()
    if not text:
        return []
    out: list[Decision] = []
    for head in row["output"]["classifications"]:
        out.extend(_head(i, text, head, rng))
    return out
