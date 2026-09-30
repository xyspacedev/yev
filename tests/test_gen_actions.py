from collections import defaultdict

from jeff.generators.actions import ActionCase, ActionPolicy, action_decision, generate
from jeff.generators.common import token_edit_size

P = ActionPolicy(limit=500)


def test_action_decision_rules():
    assert action_decision(P, ActionCase("data", env="production", op="update")) == "escalate"
    assert action_decision(P, ActionCase("data", env="staging", op="update")) == "allow"
    assert action_decision(P, ActionCase("data", env="production", op="read")) == "allow"
    assert action_decision(P, ActionCase("data", env="production", op="delete")) == "deny"
    assert action_decision(P, ActionCase("data", env="production", op="delete", ticket=True)) == "escalate"
    assert action_decision(P, ActionCase("payment", amount=500)) == "allow"
    assert action_decision(P, ActionCase("payment", amount=501)) == "escalate"
    assert action_decision(P, ActionCase("payment", amount=2501)) == "deny"
    assert action_decision(P, ActionCase("message")) == "allow"
    assert action_decision(P, ActionCase("message", external=True)) == "escalate"
    assert action_decision(P, ActionCase("message", external=True, sensitive=True)) == "deny"


def test_generated_clusters_valid_and_small_edits():
    ds = generate(90, seed=3)
    assert [d.id for d in ds] == [d.id for d in generate(90, seed=3)]
    clusters = defaultdict(list)
    for d in ds:
        d.validate()
        assert set(d.keys) == {"allow", "escalate", "deny"} and d.family == "action_review"
        clusters[d.cluster_id].append(d)
    assert len(clusters) == 90
    for members in clusters.values():
        assert len({d.gold for d in members}) >= 2
        base = members[0]
        for d in members[1:]:
            if d.edit_type == "policy_edit":
                assert d.state == base.state
            else:
                assert token_edit_size(base.state, d.state) <= 15, (base.state, d.state)
