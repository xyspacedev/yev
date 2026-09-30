"""Shared helpers for synthetic cluster generators (spec §4.3)."""

from __future__ import annotations

import difflib
import re
from collections import defaultdict

from jeff.schema import Decision, Option

EDIT_TYPES = (
    "negation", "threshold", "date", "entity_swap", "quantifier",
    "exception", "unit", "policy_edit", "injection",
)
EDIT_TYPE_DEFINITIONS = {
    "negation": "something is made true/false: 'is' vs 'is not', 'lost' vs 'kept', 'has' vs 'lacks'",
    "threshold": "a number moves across a limit: $500 vs $501, 99 vs 100 users",
    "date": "a date or duration moves across a window: day 30 vs day 31, before vs after a deadline",
    "entity_swap": "one thing is replaced by another of the same kind: production vs staging, internal vs external",
    "quantifier": "how many is changed: all vs some, every vs one, none vs any",
    "exception": "an exception or special condition is added or removed: 'final sale', 'has an approved ticket'",
    "unit": "a unit or scale changes: hours vs days, MB vs GB, per month vs per year",
    "policy_edit": "the input text is identical but one option's rule text changed",
    "injection": "the input gains a sentence that tries to instruct the reader, and nothing else changes",
}
EDIT_TIE_BREAKS = (
    "Tie-breaks: date: the changed value is a calendar date or an elapsed time/duration; "
    "unit: only the unit word changes (hours -> days); threshold: any other number changes; "
    "exception: a special-case clause is added or removed; "
    "negation: a fact flips true/false without adding a clause."
)
NUMERIC_GROUP = frozenset({"threshold", "date", "unit"})
SYNTH_LICENCE = "cc-by-4.0"
MAX_EDIT_TOKENS = 15

_TOKEN = re.compile(r"\w+|[^\w\s]")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_edit_size(a: str, b: str) -> int:
    """Tokens changed between a and b: for each differing span, the larger side's length."""
    ta, tb = tokens(a), tokens(b)
    size = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb, autojunk=False).get_opcodes():
        if tag != "equal":
            size += max(i2 - i1, j2 - j1)
    return size


def make_cluster(
    *,
    cluster_id: str,
    family: str,
    source: str,
    qtype: str,
    question: str,
    members: list[tuple[str, list[Option], str, str | None]],
) -> list[Decision]:
    return [
        Decision(
            id=f"{cluster_id}:{k}",
            type=qtype,
            state=state,
            question=question,
            options=list(options),
            gold=gold,
            family=family,
            source=source,
            licence=SYNTH_LICENCE,
            cluster_id=cluster_id,
            edit_type=edit_type,
        )
        for k, (state, options, gold, edit_type) in enumerate(members)
    ]


def keep_valid_clusters(decisions: list[Decision]) -> list[Decision]:
    """Keep only rows whose cluster still has at least two different golds."""
    golds: dict[str, set[str]] = defaultdict(set)
    for d in decisions:
        golds[d.cluster_id].add(d.gold)
    return [d for d in decisions if d.cluster_id is not None and len(golds[d.cluster_id]) >= 2]


def spread_into_parts(items: list, cluster_of: list[str], part_size: int, rng) -> list[list]:
    """Split items into parts so that no part holds two items of the same cluster."""
    import math

    groups: dict[str, list] = defaultdict(list)
    for item, c in zip(items, cluster_of):
        groups[c].append(item)
    n_parts = max(math.ceil(len(items) / part_size), max((len(g) for g in groups.values()), default=1), 1)
    parts: list[list] = [[] for _ in range(n_parts)]
    cursor = 0
    for c in sorted(groups):
        offset = cursor
        cursor += len(groups[c])
        for j, item in enumerate(groups[c]):
            parts[(offset + j) % n_parts].append(item)
    for part in parts:
        rng.shuffle(part)
    return [p for p in parts if p]
