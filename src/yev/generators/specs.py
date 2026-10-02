"""Registered specs for our own synthetic sources, so `yev filter` accepts their rows."""

from __future__ import annotations

from yev.generators.common import SYNTH_LICENCE
from yev.sources.base import SourceSpec


def _not_downloaded(row, i, rng, labels):
    return []


SYNTHETIC_SPECS = [
    SourceSpec("synthetic_rules", "yev/synthetic", "rules", "train", SYNTH_LICENCE, _not_downloaded, 0),
    SourceSpec("synthetic_opus", "yev/synthetic", "opus", "train", SYNTH_LICENCE, _not_downloaded, 0),
]
