"""Registered specs for our own synthetic sources, so `jeff filter` accepts their rows."""

from __future__ import annotations

from jeff.generators.common import SYNTH_LICENCE
from jeff.sources.base import SourceSpec


def _not_downloaded(row, i, rng, labels):
    return []


SYNTHETIC_SPECS = [
    SourceSpec("synthetic_rules", "jeff/synthetic", "rules", "train", SYNTH_LICENCE, _not_downloaded, 0),
    SourceSpec("synthetic_opus", "jeff/synthetic", "opus", "train", SYNTH_LICENCE, _not_downloaded, 0),
]
