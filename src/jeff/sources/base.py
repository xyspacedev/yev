"""Shared machinery for public-dataset adapters.

A converter turns one dataset row into zero or more Decisions. It sets a local
`id` and leaves `source` and `licence` empty; `build` fills them from the spec.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator

from jeff import licences
from jeff.schema import Decision, SchemaError

MAX_STATE_CHARS = 6000

Converter = Callable[[dict, int, random.Random, list[str]], list[Decision]]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    hf_id: str
    config: str | None
    split: str
    licence: str
    convert: Converter
    pool_size: int
    label_column: str | None = None
    max_scan: int | None = None


@dataclass
class BuildStats:
    scanned: int = 0
    converted: int = 0
    skipped: int = 0
    kept: int = 0


def humanize(label: str) -> str:
    return re.sub(r"[_\-]+", " ", label).strip().capitalize()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def pick_distractors(gold: str, labels: list[str], rng: random.Random, k_min: int = 3, k_max: int = 6) -> list[str]:
    others = [label for label in labels if label != gold]
    k_hi = min(k_max, len(others) + 1)
    k_lo = min(k_min, k_hi)
    k = rng.randint(k_lo, k_hi)
    chosen = rng.sample(others, k - 1) + [gold]
    rng.shuffle(chosen)
    return chosen


def sample_pool(decisions: Iterable[Decision], n: int, seed: int) -> list[Decision]:
    """Sample ~n decisions, balanced across gold labels, keeping each cluster whole."""
    units: dict[str, list[Decision]] = {}
    for d in decisions:
        units.setdefault(d.cluster_id or d.id, []).append(d)
    by_gold: dict[str, list[list[Decision]]] = defaultdict(list)
    for unit in units.values():
        by_gold[unit[0].gold].append(unit)
    rng = random.Random(f"{seed}:sample")
    buckets = [by_gold[g] for g in sorted(by_gold)]
    for bucket in buckets:
        rng.shuffle(bucket)
    out: list[Decision] = []
    i = 0
    while len(out) < n and any(buckets):
        bucket = buckets[i % len(buckets)]
        if bucket:
            out.extend(bucket.pop())
        i += 1
    return out


def _label_universe(rows: list[dict], column: str) -> list[str]:
    labels: set[str] = set()
    for row in rows:
        value = row.get(column)
        if isinstance(value, list):
            labels.update(v for v in value if v)
        elif value:
            labels.add(value)
    return sorted(labels)


def build(spec: SourceSpec, rows: Iterable[dict], seed: int = 0) -> tuple[list[Decision], BuildStats]:
    licences.check(spec.hf_id, spec.licence, config=spec.config, split=spec.split)
    rows = list(rows)
    labels = _label_universe(rows, spec.label_column) if spec.label_column else []
    rng = random.Random(f"{seed}:{spec.name}")
    stats = BuildStats()
    converted: list[Decision] = []
    for i, row in enumerate(rows):
        stats.scanned += 1
        produced = spec.convert(row, i, rng, labels)
        if not produced:
            stats.skipped += 1
            continue
        for d in produced:
            d = replace(
                d,
                id=f"{spec.name}:{d.id}",
                source=spec.name,
                licence=spec.licence,
                cluster_id=f"{spec.name}:{d.cluster_id}" if d.cluster_id else None,
            )
            if len(d.state) > MAX_STATE_CHARS:
                stats.skipped += 1
                continue
            try:
                converted.append(d.validate())
            except SchemaError:
                stats.skipped += 1
    stats.converted = len(converted)
    kept = sample_pool(converted, spec.pool_size, seed)
    stats.kept = len(kept)
    return kept, stats


def fetch(spec: SourceSpec) -> Iterator[dict]:
    """Stream the spec's train split, with ClassLabel ints mapped to their names."""
    from datasets import ClassLabel, load_dataset

    split = spec.split if spec.max_scan is None else f"{spec.split}[:{spec.max_scan}]"
    ds = load_dataset(spec.hf_id, spec.config, split=split)
    scalar = {c: f.names for c, f in ds.features.items() if isinstance(f, ClassLabel)}
    listed = {
        c: f.feature.names
        for c, f in ds.features.items()
        if isinstance(getattr(f, "feature", None), ClassLabel)
    }
    for row in ds:
        for col, names in scalar.items():
            row[col] = names[row[col]] if row[col] is not None and row[col] >= 0 else None
        for col, names in listed.items():
            row[col] = [names[v] for v in row[col]]
        yield row
