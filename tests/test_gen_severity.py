from collections import defaultdict

from jeff.generators.common import EDIT_TYPES, token_edit_size
from jeff.generators.severity import SCALE, SeverityCase, SeverityPolicy, generate, severity_decision

P = SeverityPolicy(medium=10, high=100, critical=1000)


def test_severity_rules():
    assert severity_decision(P, SeverityCase("staging", 5000, True, False)) == "low"
    assert severity_decision(P, SeverityCase("production", 3, True, True)) == "critical"
    assert severity_decision(P, SeverityCase("production", 1000, False, False)) == "critical"
    assert severity_decision(P, SeverityCase("production", 1000, False, True)) == "high"
    assert severity_decision(P, SeverityCase("production", 100, False, False)) == "high"
    assert severity_decision(P, SeverityCase("production", 100, False, True)) == "medium"
    assert severity_decision(P, SeverityCase("production", 99, False, False)) == "medium"
    assert severity_decision(P, SeverityCase("production", 9, False, False)) == "low"


def test_generated_score_clusters_keep_scale_order():
    ds = generate(60, seed=4)
    clusters = defaultdict(list)
    for d in ds:
        d.validate()
        assert d.type == "score" and d.keys == list(SCALE) and d.family == "triage"
        clusters[d.cluster_id].append(d)
    for members in clusters.values():
        assert len({d.gold for d in members}) >= 2
        for d in members[1:]:
            if d.edit_type != "policy_edit":
                assert token_edit_size(members[0].state, d.state) <= 15


def _by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def test_policy_edit_changes_only_options_by_a_small_amount():
    seen = 0
    for members in _by_cluster(generate(200, seed=5)).values():
        base = members[0]
        for d in members[1:]:
            if d.edit_type == "policy_edit":
                seen += 1
                assert d.state == base.state
                total = sum(token_edit_size(a.description, b.description) for a, b in zip(base.options, d.options))
                assert total <= 15
    assert seen > 0


def test_base_states_are_diverse():
    bases = [m[0].state for m in _by_cluster(generate(200, seed=1)).values()]
    assert len(set(bases)) >= 190


def test_severity_metadata_and_option_sharing():
    for members in _by_cluster(generate(60, seed=4)).values():
        base = members[0]
        assert base.edit_type is None
        for d in members:
            assert d.source == "synthetic_rules" and d.licence == "cc-by-4.0"
        for d in members[1:]:
            assert d.edit_type in EDIT_TYPES
            if d.edit_type != "policy_edit":
                assert d.options == base.options


def test_base_golds_are_diverse_and_verbatim_users_do_not_leak_the_label():
    import re
    from collections import Counter

    clusters = _by_cluster(generate(1000, seed=11))
    base_golds = Counter(m[0].gold for m in clusters.values())
    assert sum(1 for v in base_golds.values() if v / 1000 >= 0.15) >= 3, base_golds
    verbatim = Counter()
    for d in (d for m in clusters.values() for d in m):
        users = re.search(r"About (\d+) users", d.state).group(1)
        if any(re.search(rf"(?<!\d){users}(?!\d)", o.description) for o in d.options):
            verbatim[d.gold] += 1
    assert verbatim and max(verbatim.values()) / sum(verbatim.values()) <= 0.6, verbatim


def test_threshold_variants_cross_the_boundary_by_one_or_land_on_it():
    import re

    crossing = 0
    for members in _by_cluster(generate(300, seed=12)).values():
        base = members[0]
        bu = int(re.search(r"About (\d+) users", base.state).group(1))
        for d in members[1:]:
            if d.edit_type == "threshold":
                du = int(re.search(r"About (\d+) users", d.state).group(1))
                crossing += d.gold != base.gold
                assert du != bu
    assert crossing > 0
