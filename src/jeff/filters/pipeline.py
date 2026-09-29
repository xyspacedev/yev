"""Run the filters in spec order and count what each one removed, per source."""

from __future__ import annotations

from collections import Counter, defaultdict

from jeff import licences
from jeff.filters.contamination import EmbeddingFilter, Fingerprints, has_canary, overlaps
from jeff.filters.dedupe import dedupe_key
from jeff.schema import Decision

REPORT_KEYS = ("in", "licence", "dev_split", "canary", "ngram", "embedding", "duplicate", "out")


class ContaminationError(RuntimeError):
    """A row that should have been filtered reached the output."""


def run_filters(
    decisions: list[Decision], fp: Fingerprints, embed: EmbeddingFilter | None, batch: int = 512
) -> tuple[list[Decision], dict[str, dict[str, int]]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    survivors: list[Decision] = []
    for d in decisions:
        c = counts[d.source]
        c["in"] += 1
        if not licences.is_allowed(d.licence):
            c["licence"] += 1
        elif d.split == "dev":
            c["dev_split"] += 1
        elif has_canary(d):
            c["canary"] += 1
        elif overlaps(d, fp):
            c["ngram"] += 1
        else:
            survivors.append(d)

    if embed is not None:
        flagged: list[bool] = []
        for start in range(0, len(survivors), batch):
            flagged += embed.too_similar([d.state for d in survivors[start : start + batch]])
        kept_after_embed = []
        for d, bad in zip(survivors, flagged):
            if bad:
                counts[d.source]["embedding"] += 1
            else:
                kept_after_embed.append(d)
        survivors = kept_after_embed

    seen: set[str] = set()
    kept: list[Decision] = []
    for d in survivors:
        key = dedupe_key(d)
        if key in seen:
            counts[d.source]["duplicate"] += 1
            continue
        seen.add(key)
        kept.append(d)
        counts[d.source]["out"] += 1

    report = {src: {k: c.get(k, 0) for k in REPORT_KEYS} for src, c in counts.items()}
    return kept, report


def assert_clean(decisions: list[Decision], fp: Fingerprints) -> None:
    for d in decisions:
        if has_canary(d) or overlaps(d, fp):
            raise ContaminationError(f"{d.id} overlaps DecideBench")
