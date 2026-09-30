from collections import defaultdict

from jeff.generators.common import token_edit_size
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
