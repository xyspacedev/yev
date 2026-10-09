import random
import re
from collections import Counter, defaultdict
from dataclasses import replace

import pytest

from yev.generators.common import SYNTH_LICENCE
from yev.generators.invoice_actions import (
    ACTIONS, CLEAN_CASE, INVOICE_ACTION_FAMILY, PAYMENT_TERMS_FAMILY, SOURCE, TERMS_KEYS, VARIANTS, decide,
    generate_invoice_action, generate_payment_terms, render_state,
)


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


# ------------------------------------------------------------------------------------------------ payment_terms

def terms_of(d) -> str:
    return re.search(r"^Payment terms: (.+)$", d.state, re.M).group(1)


PLAIN_RE = re.compile(r"^(net \d+|due on receipt|payable on receipt)\.?$", re.I)


def test_payment_terms_rows_valid():
    ds = generate_payment_terms(300, seed=1)
    for d in ds:
        d.validate()
        assert d.type == "choice" and d.family == PAYMENT_TERMS_FAMILY and d.source == SOURCE
        assert d.licence == SYNTH_LICENCE
        assert sorted(d.keys) == sorted(TERMS_KEYS)
        days = d.metadata["discount_days"]
        assert d.gold == ("no_discount" if days is None else f"within_{days}_days")
        if days is not None:
            # The window is written in the terms string; nothing needs computing.
            assert re.search(rf"\b{days}\b", terms_of(d)), terms_of(d)
    assert len({tuple(d.keys) for d in ds}) > 1  # some rows shuffle the option order


def test_payment_terms_plain_net_is_no_discount():
    ds = generate_payment_terms(600, seed=2)
    plain = [d for d in ds if PLAIN_RE.match(terms_of(d))]
    assert len(plain) > 50
    assert all(d.gold == "no_discount" for d in plain)
    assert all(d.gold == "no_discount" for d in ds if d.metadata["terms_kind"] in ("net", "receipt"))
    # Distractor numbers equal to option windows appear in no-discount rows too.
    tricky = [d for d in ds if d.gold == "no_discount" and re.search(r"\b(7|10|15|20|30)\b", terms_of(d))]
    assert len(tricky) > 50


def test_payment_terms_share():
    ds = generate_payment_terms(700, seed=3)
    assert len(ds) >= 2000
    none = sum(d.gold == "no_discount" for d in ds) / len(ds)
    assert abs(none - 0.70) <= 0.05, none
    days = Counter(d.gold for d in ds if d.gold != "no_discount")
    assert set(days) == {f"within_{n}_days" for n in (7, 10, 15, 20, 30)}


def test_payment_terms_length():
    est = [len(d.state + d.question) / 4 for d in generate_payment_terms(300, seed=4)]
    assert min(est) >= 50 and max(est) <= 300, (min(est), max(est))


# ----------------------------------------------------------------------------------------------- invoice_action

def facts(d) -> dict[str, str]:
    """The printed checks and answered facts, keyed by name (first word of the value)."""
    out = {}
    for line in d.state.splitlines():
        m = re.match(r"^(?:- )?([a-z_]+): (\S+)", line)
        if m:
            out[m.group(1)] = m.group(2).rstrip(".,;")
    return out


# The routing tables the brief asks for, restated here so a wrong table in the module fails.
TIER_ACTIONS = {
    "small_company": {"bank": "hold_verify_bank_details", "scope": "dispute_lines",
                      "completion": "hold_request_documents", "price": "project_owner_review"},
    "enterprise": {"bank": "verify_vendor_out_of_band", "scope": "procurement_review",
                   "completion": "project_owner_review", "price": "procurement_review"},
    "high_volume_retailer": {"bank": "hold_verify_bank_details", "scope": "short_pay",
                             "completion": "hold_request_documents", "price": "short_pay"},
}


def route(variant: str, f: dict[str, str]) -> str:
    t = TIER_ACTIONS[variant]
    if f["sender"] == "impostor":
        return "fraud_review"
    if f["duplicate_flag"] == "exact":
        return "reject_as_duplicate"
    if f["duplicate_flag"] == "possible":
        return "duplicate_review"
    if f["bank_change_flag"] == "yes":
        return t["bank"]
    if f["owner_declined"] == "yes" or f["po_approved"] == "no":
        return "reject_as_unauthorized"
    if f["unapproved_extra_lines"] != "none":
        return t["scope"]
    if f["missing_records_lines"] != "none":
        return t["completion"]
    if f["totals_match"] == "no":
        return "request_corrected_invoice"
    if f["po_variance_flag"] == "yes":
        return t["price"]
    band = f["approval_band"]
    if band == "approval":
        return "human_approval"
    if band == "escalation":
        return "approval_escalation"
    assert band == "auto_pay"
    if variant == "small_company":
        return "pay"
    if variant == "enterprise":
        return "schedule_payment"
    return "pay" if f["early_payment_discount"] == "available" else "schedule_payment"


CHECK_NAMES = ("sender", "duplicate_flag", "bank_change_flag", "owner_declined", "po_approved",
               "unapproved_extra_lines", "missing_records_lines", "totals_match", "po_variance_flag",
               "early_payment_discount", "approval_band")


def test_invoice_action_rows_follow_printed_checks():
    ds = generate_invoice_action(800, seed=1)
    assert len(ACTIONS) == 16
    for d in ds:
        d.validate()
        assert d.type == "choice" and d.family == INVOICE_ACTION_FAMILY and d.source == SOURCE
        assert 2 <= len(d.options) <= 10 and set(d.keys) <= set(ACTIONS)
        f = facts(d)
        assert set(CHECK_NAMES) <= set(f), (d.id, set(CHECK_NAMES) - set(f))
        v = d.metadata["variant"]
        assert VARIANTS[v].label in d.state
        assert d.gold == route(v, f), (d.id, d.state)
    assert set(d.gold for d in ds) == set(ACTIONS)


def test_invoice_action_prints_bands_and_tolerance():
    ds = generate_invoice_action(200, seed=2)
    for d in ds:
        p = VARIANTS[d.metadata["variant"]]
        assert f"{p.tolerance_pct}%" in d.state
        assert f"${p.auto_pay_limit:,}" in d.state and f"${p.approval_limit:,}" in d.state
        # Every computed flag shows its numbers: the variance line prints the delta in money and percent.
        var = re.search(r"^po_variance_flag: (yes|no) \(.*?([+-]\$[\d,]+\.\d\d).*?([+-]\d+\.\d)%", d.state, re.M)
        assert var, d.state


def test_invoice_action_states_are_consistent():
    ds = generate_invoice_action(400, seed=3)
    for d in ds:
        f, s = facts(d), d.state
        assert (f["bank_change_flag"] == "yes") == ("account details differ" in s)
        assert (f["owner_declined"] == "yes") == ("withholds approval" in s)
        assert (f["sender"] == "impostor") == ("does not match" in s)
        # The variance flag agrees with the printed percent and tolerance.
        m = re.search(r"^po_variance_flag: (yes|no) .*?([+-]\d+\.\d)% .*?up to (\d+(?:\.\d)?)%", s, re.M)
        assert (m.group(1) == "yes") == (float(m.group(2)) > float(m.group(3))), m.group(0)


def case_state(variant: str, case) -> str:
    return render_state(variant, case, random.Random(0))


EDGES = [
    # (higher-tier override, lower-tier override, expected gold under small_company)
    ({"impostor": True}, {"duplicate": "exact"}, "fraud_review"),
    ({"duplicate": "exact"}, {"bank_change": True}, "reject_as_duplicate"),
    ({"duplicate": "possible"}, {"bank_change": True}, "duplicate_review"),
    ({"bank_change": True}, {"owner_declined": True}, "hold_verify_bank_details"),
    ({"owner_declined": True}, {"extra_lines": (4,)}, "reject_as_unauthorized"),
    ({"po_approved": False}, {"extra_lines": (3,)}, "reject_as_unauthorized"),
    ({"extra_lines": (2, 5)}, {"missing_lines": (1,)}, "dispute_lines"),
    ({"missing_lines": (3,)}, {"totals_match": False}, "hold_request_documents"),
    ({"totals_match": False}, {"variance_pct10": 45}, "request_corrected_invoice"),
    ({"variance_pct10": 21}, {}, "project_owner_review"),
]


@pytest.mark.parametrize("high,low,gold", EDGES)
def test_invoice_action_priority_table(high, low, gold):
    both = replace(CLEAN_CASE, **{**low, **high})
    assert decide("small_company", both) == gold
    assert route("small_company", facts(type("D", (), {"state": case_state("small_company", both)}))) == gold
    # Dropping the higher flag hands the decision to the lower one.
    lower = replace(CLEAN_CASE, **low)
    assert decide("small_company", lower) != gold or not low


def test_invoice_action_band_and_clean_edges():
    small = VARIANTS["small_company"]
    at_limit = replace(CLEAN_CASE, po_cents=small.auto_pay_limit * 100, variance_pct10=0)
    assert decide("small_company", at_limit) == "pay"  # exactly at the auto-pay limit still auto-pays
    over = replace(at_limit, po_cents=small.auto_pay_limit * 100 + 1000)
    assert decide("small_company", over) == "human_approval"
    top = replace(at_limit, po_cents=small.approval_limit * 100 + 1000)
    assert decide("small_company", top) == "approval_escalation"
    at_tol = replace(CLEAN_CASE, variance_pct10=int(small.tolerance_pct * 10))
    assert decide("small_company", at_tol) == "pay"  # exactly at the tolerance is accepted
    disc = replace(CLEAN_CASE, discount_available=True)
    assert decide("high_volume_retailer", disc) == "pay"
    assert decide("high_volume_retailer", CLEAN_CASE) == "schedule_payment"
    assert decide("enterprise", disc) == "schedule_payment"


def test_invoice_action_policy_variant_changes_gold():
    scope = replace(CLEAN_CASE, extra_lines=(4,))
    assert {v: decide(v, scope) for v in VARIANTS} == {
        "small_company": "dispute_lines", "enterprise": "procurement_review", "high_volume_retailer": "short_pay"}
    ds = generate_invoice_action(800, seed=4)
    pairs = same = 0
    for rows in by_cluster(ds).values():
        base = rows[0]
        for r in rows[1:]:
            if r.edit_type != "entity_swap":
                continue
            assert r.metadata["variant"] != base.metadata["variant"]
            fb, fr = facts(base), facts(r)
            if all(fb[k] == fr[k] for k in CHECK_NAMES):
                pairs += 1
                same += base.gold == r.gold
    assert pairs > 300
    # The same printed flags give a different gold under another policy in a large share of pairs.
    assert (pairs - same) / pairs >= 0.35, (pairs, same)


def test_invoice_action_clean_share():
    ds = generate_invoice_action(800, seed=5)
    assert len(ds) >= 2000
    clean = [d for d in ds if d.gold in ("pay", "schedule_payment")]
    share = len(clean) / len(ds)
    assert abs(share - 0.25) <= 0.05, share
    for d in clean:
        f = facts(d)
        assert f["sender"] == "known_contact" and f["duplicate_flag"] == "no" and f["approval_band"] == "auto_pay"
        assert f["unapproved_extra_lines"] == f["missing_records_lines"] == "none"
    golds = Counter(d.gold for d in ds)
    assert max(golds.values()) / len(ds) < 0.2  # no other action dominates


def test_invoice_action_length():
    est = [len(d.state + d.question) / 4 for d in generate_invoice_action(300, seed=6)]
    assert min(est) >= 300 and max(est) <= 1500, (min(est), max(est))


def test_invoice_action_clusters():
    ds = generate_invoice_action(300, seed=7)
    for cid, rows in by_cluster(ds).items():
        assert re.fullmatch(rf"{INVOICE_ACTION_FAMILY}:7:\d+", cid)
        assert len({r.gold for r in rows}) >= 2
        assert len({tuple(r.keys) for r in rows}) == 1  # one option set per cluster, holding every gold


def test_deterministic():
    for gen, fam in ((generate_payment_terms, PAYMENT_TERMS_FAMILY), (generate_invoice_action, INVOICE_ACTION_FAMILY)):
        a, b = gen(12, seed=3), gen(12, seed=3)
        assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
        assert [x.state for x in a] != [x.state for x in gen(12, seed=4)]
        assert all(x.id.startswith(f"{fam}:3:") for x in a)


def test_registered_in_rule_generators():
    from yev.cli import RULE_GENERATORS
    from yev.filters.pipeline import _licence_ok
    from yev.generators.specs import SYNTHETIC_SPECS

    assert RULE_GENERATORS["payment_terms"] is generate_payment_terms
    assert RULE_GENERATORS["invoice_action"] is generate_invoice_action
    specs = {s.name: s for s in SYNTHETIC_SPECS}
    assert all(_licence_ok(d, specs, {}) for d in generate_invoice_action(2, 0) + generate_payment_terms(2, 0))


def test_invoice_action_printed_routing_rules_match_table():
    """The routing rules printed in the state name the same actions the gold comes from."""
    printed = {"bank": r"If bank_change_flag is yes: (\w+)\.", "scope": r"If unapproved_extra_lines lists any line: (\w+)\.",
               "completion": r"If missing_records_lines lists any line: (\w+)\.",
               "price": r"If po_variance_flag is yes: (\w+)\."}
    clean = {"small_company": "Otherwise: pay.", "enterprise": "Otherwise: schedule_payment.",
             "high_volume_retailer": "Otherwise: pay if early_payment_discount is available, otherwise schedule_payment."}
    for d in generate_invoice_action(100, seed=8):
        v = d.metadata["variant"]
        for tier, pat in printed.items():
            assert re.search(pat, d.state).group(1) == TIER_ACTIONS[v][tier], (v, tier)
        assert clean[v] in d.state
        rules = re.findall(r"^  (\d+)\. If (.+?): (\w+)\.$", d.state, re.M)
        assert [int(n) for n, _, _ in rules] == list(range(1, 12))  # first match wins, in the priority order
