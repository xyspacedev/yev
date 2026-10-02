import re
from collections import Counter, defaultdict
from datetime import date

from yev.generators.common import EDIT_TYPES, MAX_EDIT_TOKENS, token_edit_size
from yev.generators.dates import fmt
from yev.generators.returns_v2 import generate


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def expected_gold(m: dict) -> str:
    """Independent re-statement of the policy rule, from the row's metadata."""
    anchor = date.fromisoformat(m["delivered"] if m["anchor"] == "delivery" else m["purchased"])
    days = (date.fromisoformat(m["requested"]) - anchor).days
    if m["final_sale"]:
        return "deny"
    if days <= m["window"] and (m["opened_ok"] or not m["opened"]):
        return "refund"
    if m["credit_days"] is not None and days <= m["credit_days"]:
        return "store_credit"
    return "deny"


def test_returns_v2_state_never_states_day_count():
    for d in generate(300, seed=3):
        assert not re.search(r"\d+ days after", d.state), d.state
        m = d.metadata
        anchor = date.fromisoformat(m["delivered"] if m["anchor"] == "delivery" else m["purchased"])
        n = (date.fromisoformat(m["requested"]) - anchor).days
        assert not re.search(rf"\b{n}\s*(days|-day|day)\b", d.state) or n in (m["window"], m["credit_days"]), d.state


def test_returns_v2_gold_matches_rule():
    ds = generate(500, seed=11)
    assert len(by_cluster(ds)) >= 450
    for d in ds:
        d.validate()
        m = d.metadata
        assert d.gold == expected_gold(m), (d.id, m)
        # the dates the rule uses are printed in the state, in the row's chosen styles
        assert fmt(date.fromisoformat(m["delivered"]), m["styles"]["delivered"]) in d.state
        assert fmt(date.fromisoformat(m["requested"]), m["styles"]["requested"]) in d.state
        if m["anchor"] == "purchase":
            assert fmt(date.fromisoformat(m["purchased"]), m["styles"]["purchased"]) in d.state
        assert f"{m['window']} days" in d.state or f"{m['window']}-day" in d.state


def test_returns_v2_no_store_credit_option_when_policy_lacks_it():
    seen = Counter()
    for d in generate(300, seed=4):
        has_tier = d.metadata["credit_days"] is not None
        seen[has_tier] += 1
        if has_tier:
            assert d.keys == ["refund", "store_credit", "deny"]
        else:
            assert d.keys == ["refund", "deny"]
            assert d.gold != "store_credit"
    assert seen[True] > 50 and seen[False] > 50


def test_returns_v2_clusters_are_contrastive():
    ds = generate(300, seed=5)
    for members in by_cluster(ds).values():
        assert len({d.gold for d in members}) >= 2
        base = members[0]
        assert base.edit_type is None
        for d in members:
            assert d.family == "returns_v2" and d.source == "rules:returns_v2"
            assert d.question == base.question and d.options == base.options
        for d in members[1:]:
            assert d.edit_type in EDIT_TYPES
            assert d.state != base.state
            assert token_edit_size(base.state, d.state) <= MAX_EDIT_TOKENS


def test_returns_v2_boundary_coverage():
    ds = generate(500, seed=6)
    windows = Counter(d.metadata["window"] for d in ds)
    assert sum(v for w, v in windows.items() if w >= 30) > 0.5 * len(ds)
    spans = Counter()
    for d in ds:
        m = d.metadata
        anchor = date.fromisoformat(m["delivered"] if m["anchor"] == "delivery" else m["purchased"])
        req = date.fromisoformat(m["requested"])
        if anchor.year < req.year:
            spans["year_end"] += 1
        if anchor.month != req.month:
            spans["month_end"] += 1
        for y in range(anchor.year, req.year + 1):
            leap = y % 4 == 0
            feb_end = date(y, 2, 29 if leap else 28)
            if anchor < feb_end < req or feb_end == req:
                spans["feb_leap" if leap else "feb_nonleap"] += 1
        spans["edge_exact" if (req - anchor).days in (m["window"], m["credit_days"]) else "edge_other"] += 1
        if m["styles"]["delivered"] == m["styles"]["requested"] == "iso":
            spans["all_iso"] += 1
        spans[m["anchor"]] += 1
    for k in ("year_end", "month_end", "feb_leap", "feb_nonleap", "edge_exact", "edge_other", "all_iso", "purchase"):
        assert spans[k] >= 20, (k, spans)
    golds = Counter(d.gold for d in ds)
    assert all(golds[g] > 0.1 * len(ds) for g in ("refund", "store_credit", "deny")), golds


def test_generators_deterministic():
    a, b = generate(40, seed=1), generate(40, seed=1)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert [x.state for x in a] != [x.state for x in generate(40, seed=2)]
    assert all(x.id.startswith("returns_v2:1:") for x in a)
