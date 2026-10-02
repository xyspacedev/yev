"""Run the filters in spec order and count what each one removed, per source."""

from __future__ import annotations

from collections import Counter, defaultdict

from yev import licences
from yev.filters.contamination import EmbeddingFilter, Fingerprints, has_canary, overlaps
from yev.filters.dedupe import dedupe_key
from yev.schema import Decision
from yev.sources.base import SourceSpec
from yev.sources.registry import LOCAL_SOURCES

REASON_ORDER = ("licence", "dev_split", "canary", "ngram", "embedding")
REPORT_KEYS = ("in", "licence", "dev_split", "canary", "ngram", "embedding", "duplicate", "out")


class ContaminationError(RuntimeError):
    """A row that should have been filtered reached the output."""


def _licence_ok(d: Decision, specs: dict[str, SourceSpec], cache: dict[str, bool]) -> bool:
    if d.source in LOCAL_SOURCES:
        # Local sources carry a licence per row (e.g. each skill's repo licence), not one per spec.
        return licences.is_allowed(d.licence)
    spec = specs.get(d.source)
    if spec is None or d.source == "decidebench" or d.licence != spec.licence:
        return False
    if d.source not in cache:
        try:
            licences.check(spec.hf_id, spec.licence, config=spec.config, split=spec.split, data_files=spec.data_files)
            cache[d.source] = True
        except licences.LicenceError:
            cache[d.source] = False
    return cache[d.source]


def run_filters(
    decisions: list[Decision],
    fp: Fingerprints,
    embed: EmbeddingFilter | None,
    specs: dict[str, SourceSpec],
    batch: int = 512,
) -> tuple[list[Decision], dict[str, dict[str, int]]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    survivors: list[Decision] = []
    licence_cache: dict[str, bool] = {}
    # cluster_id -> the earliest stage (by REASON_ORDER) that dropped a member
    dropped: dict[str, str] = {}

    def drop(d: Decision, reason: str) -> None:
        counts[d.source][reason] += 1
        if d.cluster_id is not None:
            prev = dropped.get(d.cluster_id)
            if prev is None or REASON_ORDER.index(reason) < REASON_ORDER.index(prev):
                dropped[d.cluster_id] = reason

    for d in decisions:
        counts[d.source]["in"] += 1
        if not _licence_ok(d, specs, licence_cache):
            drop(d, "licence")
        elif d.split == "dev":
            drop(d, "dev_split")
        elif has_canary(d):
            drop(d, "canary")
        elif overlaps(d, fp):
            drop(d, "ngram")
        else:
            survivors.append(d)

    if embed is not None:
        flagged: list[bool] = []
        for start in range(0, len(survivors), batch):
            flagged += embed.too_similar([d.state for d in survivors[start : start + batch]])
        kept_after_embed = []
        for d, bad in zip(survivors, flagged, strict=True):
            if bad:
                drop(d, "embedding")
            else:
                kept_after_embed.append(d)
        survivors = kept_after_embed

    # Clusters stay whole: a cluster that lost a member to any filter loses all of them.
    remaining: list[Decision] = []
    for d in survivors:
        if d.cluster_id is not None and d.cluster_id in dropped:
            counts[d.source][dropped[d.cluster_id]] += 1
        else:
            remaining.append(d)
    survivors = remaining

    # Dedupe across clusters only: rows sharing a non-null cluster_id are never duplicates of each other.
    seen: dict[str, list[str | None]] = {}
    kept: list[Decision] = []
    dup_clusters: set[str] = set()
    for d in survivors:
        key = dedupe_key(d)
        if any(c is None or c != d.cluster_id for c in seen.get(key, [])):
            counts[d.source]["duplicate"] += 1
            if d.cluster_id is not None:
                dup_clusters.add(d.cluster_id)
            continue
        seen.setdefault(key, []).append(d.cluster_id)
        kept.append(d)
        counts[d.source]["out"] += 1

    if dup_clusters:
        remaining = []
        for d in kept:
            if d.cluster_id in dup_clusters:
                counts[d.source]["out"] -= 1
                counts[d.source]["duplicate"] += 1
            else:
                remaining.append(d)
        kept = remaining

    report = {src: {k: c.get(k, 0) for k in REPORT_KEYS} for src, c in counts.items()}
    return kept, report


def assert_clean(decisions: list[Decision], fp: Fingerprints) -> None:
    for d in decisions:
        if has_canary(d) or overlaps(d, fp):
            raise ContaminationError(f"{d.id} overlaps DecideBench")
