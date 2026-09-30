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
