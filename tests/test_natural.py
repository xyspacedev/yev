from collections import Counter

from jeff import cli
from jeff.schema import Decision, Option
from jeff.sources.base import SourceSpec, build, sample_natural


def dec(i, gold, cluster=None):
    return Decision(id=str(i), type="choice", state=f"s{i}", question="q?",
                    options=[Option("a", "A."), Option("b", "B.")], gold=gold,
                    family="f", source="", licence="", cluster_id=cluster)


def test_sample_natural_keeps_priors_and_clusters():
    ds = [dec(i, "a") for i in range(900)] + [dec(1000 + i, "b") for i in range(100)]
    ds += [dec(5000, "a", "c1"), dec(5001, "b", "c1")]
    out = sample_natural(ds, 300, seed=0)
    share_b = Counter(d.gold for d in out)["b"] / len(out)
    assert 0.03 < share_b < 0.2
    ids = {d.id for d in out}
    assert ("5000" in ids) == ("5001" in ids)
    assert [d.id for d in out] == [d.id for d in sample_natural(ds, 300, seed=0)]


def test_build_natural_and_pool_scale():
    rows = [{"g": "a"}] * 90 + [{"g": "b"}] * 10

    def convert(row, i, rng, labels):
        return [dec(i, row["g"])]

    spec = SourceSpec("toy", "toy/toy", None, "train", "mit", convert, 50)
    balanced, _ = build(spec, rows)
    natural, _ = build(spec, rows, natural=True)
    scaled, _ = build(spec, rows, natural=True, pool_scale=0.2)
    assert Counter(d.gold for d in balanced)["b"] == 10
    assert Counter(d.gold for d in natural)["b"] < 10
    assert len(scaled) == 10


def test_cli_flags_parse():
    args = cli.make_parser().parse_args(["build-public", "--out", "x", "--natural", "--pool-scale", "0.1"])
    assert args.natural is True and args.pool_scale == 0.1
