"""Plan-6 rule generators never share text with a reported benchmark (data/bench/fingerprints.json)."""

from pathlib import Path

import pytest

from yev.cli import RULE_GENERATORS

FINGERPRINTS = Path(__file__).resolve().parents[2] / "data" / "bench" / "fingerprints.json"


@pytest.fixture(scope="module")
def bench_fp():
    if not FINGERPRINTS.is_file():
        pytest.skip(f"{FINGERPRINTS} not found")
    from yev.filters.benchmarks import BenchmarkFingerprints

    return BenchmarkFingerprints.load(FINGERPRINTS)


@pytest.mark.parametrize("name", ["policy_counts", "payment_terms", "invoice_action", "agent_triples"])
def test_rule_generator_has_no_benchmark_overlap(name, bench_fp):
    from yev.filters.benchmarks import overlaps_any

    rows = RULE_GENERATORS[name](300, 0)
    hits = [(d.id, h) for d in rows if (h := overlaps_any(d, bench_fp)) is not None]
    assert not hits, (len(hits), hits[:5])
