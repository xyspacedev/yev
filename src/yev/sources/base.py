"""Shared machinery for public-dataset adapters.

A converter turns one dataset row into zero or more Decisions. It sets a local
`id` and leaves `source` and `licence` empty; `build` fills them from the spec.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Iterable, Iterator

from yev import licences
from yev.schema import Decision, SchemaError

MAX_STATE_CHARS = 6000
# Chat-template tokens around the system and user turns, plus slack for the tokenizer joining the
# separately counted state and the rest of the user turn.
PROMPT_OVERHEAD_TOKENS = 24

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
    revision: str | None = None
    data_files: str | None = None
    # Regroups the raw rows before conversion (e.g. one record per document from per-question rows).
    prepare: Callable[[list[dict]], list[dict]] | None = None
    max_state_chars: int = MAX_STATE_CHARS
    # (min state tokens, max full-prompt tokens), counted with the real tokenizer; rows outside are dropped.
    token_window: tuple[int, int] | None = None


@dataclass
class BuildStats:
    scanned: int = 0
    converted: int = 0
    skipped: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = asdict(self)
        if not out["dropped"]:
            del out["dropped"]
        return out


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


def sample_natural(decisions: Iterable[Decision], n: int, seed: int) -> list[Decision]:
    """Sample whole units (cluster or row) at random until ~n rows, keeping natural label priors."""
    units: dict[str, list[Decision]] = {}
    for d in decisions:
        units.setdefault(d.cluster_id or d.id, []).append(d)
    order = sorted(units)
    random.Random(f"{seed}:natural").shuffle(order)
    out: list[Decision] = []
    for key in order:
        if len(out) >= n:
            break
        out.extend(units[key])
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


def prompt_tokens(d: Decision, count: Callable[[str], int], cache: dict[str, int] | None = None) -> int:
    """Tokens of the rendered prompt (system turn plus the user turn's JSON), without few-shot examples."""
    from yev.format import SYSTEM_PROMPT, user_turn

    cache = {} if cache is None else cache
    key = "json:" + d.state
    if key not in cache:
        cache[key] = count(json.dumps(d.state, ensure_ascii=False))
    rest = user_turn(replace(d, state=""), d.options)
    return count(SYSTEM_PROMPT) + cache[key] + count(rest) + PROMPT_OVERHEAD_TOKENS


def _token_filter(decisions: list[Decision], window: tuple[int, int], count: Callable[[str], int],
                  stats: BuildStats) -> list[Decision]:
    lo, hi = window
    cache: dict[str, int] = {}
    bad_clusters: set[str] = set()
    kept: list[Decision] = []
    for d in decisions:
        if d.state not in cache:
            cache[d.state] = count(d.state)
        reason = None
        if cache[d.state] < lo:
            reason = "too_short"
        elif prompt_tokens(d, count, cache) > hi:
            reason = "too_long"
        if reason:
            stats.dropped[reason] = stats.dropped.get(reason, 0) + 1
            stats.skipped += 1
            if d.cluster_id:
                bad_clusters.add(d.cluster_id)
            continue
        kept.append(d)
    out = [d for d in kept if d.cluster_id not in bad_clusters]
    stats.skipped += len(kept) - len(out)
    return out


def build(
    spec: SourceSpec, rows: Iterable[dict], seed: int = 0, *, natural: bool = False, pool_scale: float = 1.0,
    count: Callable[[str], int] | None = None,
) -> tuple[list[Decision], BuildStats]:
    """`count` measures tokens for `spec.token_window`; by default the real Qwen3.5 tokenizer (no heuristic)."""
    licences.check(spec.hf_id, spec.licence, config=spec.config, split=spec.split, data_files=spec.data_files)
    rows = list(rows)
    if spec.prepare:
        rows = spec.prepare(rows)
    labels = _label_universe(rows, spec.label_column) if spec.label_column else []
    rng = random.Random(f"{seed}:{spec.name}")
    stats = BuildStats()
    converted: list[Decision] = []
    skipped_clusters: set[str] = set()
    used_ids: set[str] = set()
    for i, row in enumerate(rows):
        stats.scanned += 1
        produced = spec.convert(row, i, rng, labels)
        if not produced:
            stats.skipped += 1
            continue
        for d in produced:
            new_id = f"{spec.name}:{d.id}"
            n = 1
            while new_id in used_ids:
                n += 1
                new_id = f"{spec.name}:{d.id}#{n}"
            used_ids.add(new_id)
            d = replace(
                d,
                id=new_id,
                source=spec.name,
                licence=spec.licence,
                cluster_id=f"{spec.name}:{d.cluster_id}" if d.cluster_id else None,
            )
            if len(d.state) > spec.max_state_chars:
                stats.skipped += 1
                if d.cluster_id:
                    skipped_clusters.add(d.cluster_id)
                continue
            try:
                converted.append(d.validate())
            except SchemaError:
                stats.skipped += 1
                if d.cluster_id:
                    skipped_clusters.add(d.cluster_id)
    # Drop all decisions in clusters that had any member skipped
    filtered = [d for d in converted if d.cluster_id not in skipped_clusters]
    stats.skipped += len(converted) - len(filtered)
    converted = filtered
    if spec.token_window:
        if count is None:
            from yev.generators.padding import load_token_counter

            count = load_token_counter()
        converted = _token_filter(converted, spec.token_window, count, stats)
    stats.converted = len(converted)
    target = max(1, int(spec.pool_size * pool_scale))
    kept = (sample_natural if natural else sample_pool)(converted, target, seed)
    stats.kept = len(kept)
    return kept, stats


def _iter_rows(spec: SourceSpec) -> Iterator[dict]:
    """Load the dataset and yield rows with ClassLabel ints mapped to names."""
    from datasets import ClassLabel, load_dataset

    ds = load_dataset(
        spec.hf_id,
        spec.config,
        split=spec.split,
        revision=spec.revision,
        data_files={"train": spec.data_files} if spec.data_files else None,
    )
    if spec.max_scan is not None:
        # A random subset, not the head: heads are ordered (e.g. RuleTaker depth) and biased.
        ds = ds.shuffle(seed=0).select(range(min(spec.max_scan, len(ds))))
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
            row[col] = [names[v] for v in row[col] if v >= 0]
        yield row


def fetch(spec: SourceSpec) -> Iterator[dict]:
    """Load the spec's train split with licence guard, yielding rows with ClassLabel ints mapped to names."""
    licences.check(spec.hf_id, spec.licence, config=spec.config, split=spec.split, data_files=spec.data_files)
    return _iter_rows(spec)
