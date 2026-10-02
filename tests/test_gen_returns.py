import re
from collections import defaultdict
from datetime import datetime

from yev.generators.common import EDIT_TYPES, token_edit_size
from yev.generators.returns import ReturnCase, ReturnPolicy, generate, return_decision

P = ReturnPolicy(window=30, grace=14, opened_ok=False)


def test_return_decision_boundaries():
    assert return_decision(P, ReturnCase(30, False, False)) == "full_refund"
    assert return_decision(P, ReturnCase(31, False, False)) == "store_credit"
    assert return_decision(P, ReturnCase(44, False, False)) == "store_credit"
    assert return_decision(P, ReturnCase(45, False, False)) == "decline"
    assert return_decision(P, ReturnCase(10, True, False)) == "store_credit"
    assert return_decision(ReturnPolicy(30, 14, True), ReturnCase(10, True, False)) == "full_refund"
    assert return_decision(P, ReturnCase(1, False, True)) == "decline"


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def test_generated_clusters_are_valid_contrastive_and_deterministic():
    ds = generate(60, seed=1)
    assert [d.id for d in ds] == [d.id for d in generate(60, seed=1)]
    clusters = by_cluster(ds)
    assert len(clusters) == 60
    for members in clusters.values():
        for d in members:
            d.validate()
            assert d.family == "returns_policy" and d.source == "synthetic_rules"
        assert len({d.gold for d in members}) >= 2
        base = members[0]
        assert base.edit_type is None
        for d in members[1:]:
            assert d.edit_type in EDIT_TYPES
            assert d.question == base.question
            if d.edit_type == "policy_edit":
                assert d.state == base.state and d.options != base.options
            else:
                assert d.options == base.options
                assert token_edit_size(base.state, d.state) <= 15


def test_day_count_matches_printed_dates():
    for d in generate(40, seed=2):
        dates = re.findall(r"\d{1,2} [A-Z][a-z]+ \d{4}", d.state)
        n = int(re.search(r"(\d+) days after delivery", d.state).group(1))
        first, second = (datetime.strptime(x, "%d %B %Y") for x in dates)
        assert abs((second - first).days) == n


def test_policy_edit_changes_only_options_by_a_small_amount():
    seen = 0
    for members in by_cluster(generate(200, seed=5)).values():
        base = members[0]
        for d in members[1:]:
            if d.edit_type == "policy_edit":
                seen += 1
                assert d.state == base.state
                total = sum(token_edit_size(a.description, b.description) for a, b in zip(base.options, d.options))
                assert total <= 15
    assert seen > 0


def test_base_states_are_diverse():
    bases = [m[0].state for m in by_cluster(generate(200, seed=1)).values()]
    assert len(set(bases)) >= 190


def test_day_count_shifts_are_tagged_date_not_threshold():
    seen = 0
    for members in by_cluster(generate(100, seed=6)).values():
        for d in members[1:]:
            assert d.edit_type != "threshold"
            seen += d.edit_type == "date"
    assert seen > 0
