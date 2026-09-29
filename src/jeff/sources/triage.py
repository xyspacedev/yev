"""Ticket triage adapters: priority and severity (Score), team routing (Choice)."""

from __future__ import annotations

import random

from jeff.schema import Decision, Option
from jeff.sources.base import pick_distractors, slug

HELP_DESK_SCALE = [
    ("low", "Low priority: can wait; little impact."),
    ("medium", "Medium priority: should be handled in the normal queue."),
    ("high", "High priority: significant impact; handle soon."),
    ("highest", "Highest priority: major impact; handle before other work."),
    ("blocker", "Blocker: work or service is stopped until this is fixed."),
]
CVSS_SCALE = [
    ("low", "Low severity: limited impact and hard to exploit."),
    ("medium", "Medium severity: real impact under some conditions."),
    ("high", "High severity: serious impact, readily exploitable."),
    ("critical", "Critical severity: severe impact, easily exploited, often remotely."),
]


def _score(id: str, state: str, question: str, scale, gold: str) -> Decision:
    return Decision(
        id=id, type="score", state=state, question=question,
        options=[Option(k, d) for k, d in scale], gold=gold,
        family="triage", source="", licence="",
    )


def convert_help_desk(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    text = (row["text"] or "").strip()
    gold = slug(row["issue_priority"] or "")
    if not text or gold not in dict(HELP_DESK_SCALE):
        return []
    return [_score(str(row["issue_id"]), f"Ticket: {text}", "What priority should this ticket get?", HELP_DESK_SCALE, gold)]


def convert_cvss(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    text = " ".join(t.strip() for t in (row["title"] or "", row["description"] or "") if t.strip())
    gold = row["severity_band"]
    if not text or gold not in dict(CVSS_SCALE):
        return []
    return [_score(str(row["id"]), f"Vulnerability report: {text}", "How severe is this vulnerability?", CVSS_SCALE, gold)]


def convert_it_support(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    text = (row["text"] or "").strip()
    if not text or row["label"] is None:
        return []
    chosen = pick_distractors(row["label"], labels, rng)
    return [
        Decision(
            id=str(i),
            type="choice",
            state=f"Ticket: {text}",
            question="Which team should handle this ticket?",
            options=[Option(slug(lab), f"Route to the {lab} team.") for lab in chosen],
            gold=slug(row["label"]),
            family="triage",
            source="",
            licence="",
        )
    ]
