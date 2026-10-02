from collections import Counter, defaultdict
from fractions import Fraction

from yev.generators import severity
from yev.generators.common import EDIT_TYPES, MAX_EDIT_TOKENS, token_edit_size
from yev.generators.scales import FAMILY, KINDS, SOURCE, generate


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def test_numbered_keys_ascending_and_descriptions_kept():
    ds = generate(200, seed=3)
    sev = [d for d in ds if d.metadata["kind"] == "severity"]
    assert sev
    descs = {o.description for d in severity.generate(400, 0) for o in d.options}
    prefixes = {desc.split(":")[0].lower() for desc in descs}
    assert prefixes == set(severity.SCALE)
    for d in sev:
        assert [o.key for o in d.options] == [str(i) for i in range(len(severity.SCALE))]
        # Descriptions are kept verbatim and still name the level, in ascending order.
        assert [o.description.split(":")[0].lower() for o in d.options] == list(severity.SCALE)
        assert d.metadata["orig_gold"] == severity.SCALE[int(d.gold)]
    for d in ds:
        d.validate()
        assert d.type == "score" and d.family == FAMILY and d.source == SOURCE
        assert [o.key for o in d.options] == [str(i) for i in range(len(d.options))]
        assert 3 <= len(d.options) <= 11


def expected(d) -> str:
    m = d.metadata
    if m["kind"] == "severity":
        return str(severity.SCALE.index(m["orig_gold"]))
    value = {"risk": lambda: Fraction(m["likelihood"] * m["impact"]),
             "priority": lambda: Fraction(m["days_overdue"] + m["per_team"] * m["blocked_teams"]),
             "percent_complete": lambda: Fraction(100 * m["done"], m["total"])}[m["kind"]]()
    return str(sum(1 for c in m["cuts"] if value >= c))


def test_scales_labels_computed_and_levels():
    ds = generate(300, seed=5)
    kinds = Counter(d.metadata["kind"] for d in ds)
    assert set(kinds) == set(KINDS)
    levels = Counter()
    for d in ds:
        assert d.gold == expected(d), d.id
        if d.metadata["kind"] != "severity":
            assert 3 <= len(d.options) <= 7
            assert len(d.options) == len(d.metadata["cuts"]) + 1
            levels[len(d.options)] += 1
    assert set(levels) == {3, 4, 5, 6, 7}


def test_scales_clusters_contrast():
    ds = generate(200, seed=1)
    for cid, rows in by_cluster(ds).items():
        assert len({r.gold for r in rows}) >= 2, cid
        base = rows[0]
        assert base.edit_type is None
        for r in rows[1:]:
            assert r.edit_type in EDIT_TYPES
            if r.edit_type != "policy_edit":
                assert token_edit_size(base.state, r.state) <= MAX_EDIT_TOKENS, (r.id, base.state, r.state)


def test_scales_deterministic():
    assert [d.to_dict() for d in generate(20, 9)] == [d.to_dict() for d in generate(20, 9)]
