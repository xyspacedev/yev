import random

import pytest

from jeff.licences import LicenceError
from jeff.schema import Decision, Option
from jeff.sources.base import (
    MAX_STATE_CHARS,
    SourceSpec,
    build,
    fetch,
    humanize,
    pick_distractors,
    sample_pool,
    slug,
)


def dec(i, gold="a", cluster=None, state="s") -> Decision:
    return Decision(
        id=str(i), type="choice", state=state, question="q",
        options=[Option("a", "A"), Option("b", "B")], gold=gold,
        family="f", source="", licence="", cluster_id=cluster,
    )


def spec(convert, **kw) -> SourceSpec:
    base = dict(name="toy", hf_id="toy/toy", config=None, split="train", licence="mit",
                convert=convert, pool_size=100)
    base.update(kw)
    return SourceSpec(**base)


def test_humanize_and_slug():
    assert humanize("activate_my_card") == "Activate my card"
    assert slug("Claim Correction-API!") == "claim_correction_api"
    assert slug("  ") == ""


def test_pick_distractors_contains_gold_and_respects_bounds():
    rng = random.Random(0)
    labels = [f"l{i}" for i in range(20)]
    for _ in range(50):
        out = pick_distractors("l3", labels, rng)
        assert "l3" in out and 3 <= len(out) <= 6 and len(set(out)) == len(out)


def test_pick_distractors_with_tiny_label_set():
    out = pick_distractors("x", ["x", "y"], random.Random(0))
    assert sorted(out) == ["x", "y"]


def test_sample_pool_balances_golds_and_keeps_clusters_whole():
    ds = [dec(i, gold="a") for i in range(90)] + [dec(100 + i, gold="b") for i in range(10)]
    ds += [dec(200, gold="a", cluster="c1"), dec(201, gold="b", cluster="c1")]
    out = sample_pool(ds, 20, seed=0)
    golds = [d.gold for d in out]
    assert golds.count("b") >= 9
    ids = {d.id for d in out}
    assert ("200" in ids) == ("201" in ids)


def test_sample_pool_is_deterministic():
    ds = [dec(i, gold="ab"[i % 2]) for i in range(50)]
    assert [d.id for d in sample_pool(ds, 10, 1)] == [d.id for d in sample_pool(ds, 10, 1)]


def test_build_prefixes_ids_and_fills_provenance():
    def convert(row, i, rng, labels):
        return [dec(row["n"], cluster="k")]

    out, stats = build(spec(convert), [{"n": 1}, {"n": 2}])
    assert {d.id for d in out} == {"toy:1", "toy:2"}
    assert {d.source for d in out} == {"toy"} and {d.licence for d in out} == {"mit"}
    assert {d.cluster_id for d in out} == {"toy:k"}
    assert (stats.scanned, stats.converted, stats.skipped, stats.kept) == (2, 2, 0, 2)


def test_build_skips_empty_conversions_invalid_rows_and_oversized_states():
    def convert(row, i, rng, labels):
        if row["kind"] == "empty":
            return []
        if row["kind"] == "invalid":
            return [dec(i, gold="zzz")]
        if row["kind"] == "huge":
            return [dec(i, state="x" * (MAX_STATE_CHARS + 1))]
        return [dec(i)]

    rows = [{"kind": k} for k in ["ok", "empty", "invalid", "huge"]]
    out, stats = build(spec(convert), rows)
    assert [d.id for d in out] == ["toy:0"]
    assert stats.skipped == 3


def test_build_passes_label_universe():
    seen = {}

    def convert(row, i, rng, labels):
        seen["labels"] = labels
        return []

    build(spec(convert, label_column="lab"), [{"lab": "b"}, {"lab": ["a", "c"]}, {"lab": None}])
    assert seen["labels"] == ["a", "b", "c"]


def test_build_refuses_forbidden_sources():
    with pytest.raises(LicenceError):
        build(spec(lambda *a: [], hf_id="facebook/anli"), [])
    with pytest.raises(LicenceError):
        build(spec(lambda *a: [], split="test"), [])


def test_build_drops_entire_cluster_if_any_member_is_skipped():
    def convert(row, i, rng, labels):
        if i == 0:
            return [dec(i, cluster="k", state="x" * (MAX_STATE_CHARS + 1))]
        return [dec(i, cluster="k")]

    rows = [{"_": 0}, {"_": 1}]
    out, stats = build(spec(convert), rows)
    assert len(out) == 0
    assert stats.scanned == 2
    assert stats.skipped == 2


def test_fetch_guards_eagerly_before_network():
    with pytest.raises(LicenceError):
        fetch(spec(lambda *a: [], hf_id="toy/toy", split="test"))
