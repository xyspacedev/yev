import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from yev.generators.common import EDIT_TYPES, MAX_EDIT_TOKENS, token_edit_size
from yev.generators.many_options import NONE_KEY, TAXONOMIES, generate, reframe_intents, sibling_ok
from yev.schema import Decision, Option


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def test_many_options_counts_between_7_and_10():
    ds = generate(300, seed=2)
    counts = Counter(len(d.options) for d in ds)
    assert set(counts) <= set(range(7, 11)) and len(counts) == 4
    for d in ds:
        d.validate()
        assert d.type == "choice" and d.family == "many_options" and d.source == "synthetic_rules"
        assert NONE_KEY in d.keys
    assert {d.metadata["taxonomy"] for d in ds} == set(TAXONOMIES)


def _matches(state: str, taxonomy: str) -> list[str]:
    found = []
    for cat in TAXONOMIES[taxonomy]:
        if any(re.search(rf"\b{re.escape(k)}\b", state, re.IGNORECASE) for k in cat.keywords):
            found.append(cat.key)
    return found


def test_many_options_gold_by_rule():
    ds = generate(400, seed=4)
    golds = Counter()
    for d in ds:
        found = _matches(d.state, d.metadata["taxonomy"])
        assert len(found) <= 1, (d.id, found, d.state)
        expect = found[0] if found and found[0] in d.keys else NONE_KEY
        assert d.gold == expect, (d.id, d.state)
        assert d.metadata["category"] == (found[0] if found else None)
        if d.gold != NONE_KEY:
            # The keyword that decides the row is printed in the gold option's rule.
            desc = next(o.description for o in d.options if o.key == d.gold)
            assert d.metadata["keyword"].lower() in desc.lower()
        golds[d.gold == NONE_KEY] += 1
    assert 0.1 < golds[True] / len(ds) < 0.4


def test_many_options_clusters_flip_with_small_edits():
    ds = generate(200, seed=6)
    for cid, rows in by_cluster(ds).items():
        assert len({r.gold for r in rows}) >= 2, cid
        base = rows[0]
        for r in rows[1:]:
            assert r.edit_type in EDIT_TYPES
            if r.edit_type == "policy_edit":
                assert r.state == base.state
            else:
                assert token_edit_size(base.state, r.state) <= MAX_EDIT_TOKENS


def test_many_options_states_short():
    for d in generate(100, seed=8):
        sentences = [s for s in re.split(r"(?<=[.?!])\s+", d.state.strip()) if s]
        assert 1 <= len(sentences) <= 4, d.state


def _intent_row(i: int, gold: str, labels: list[str], source="banking77", licence="cc-by-4.0", split="pool"):
    opts = [Option(lab, f"The request is about: {lab.replace('_', ' ')}.") for lab in labels]
    return Decision(id=f"{source}:{i}", type="choice", state=f"Customer message: message {i}",
                    question="Which intent best matches the customer's message?", options=opts, gold=gold,
                    family="intent", source=source, licence=licence, split=split)


LABELS = ["card_arrival", "card_delivery_estimate", "lost_or_stolen_card", "activate_my_card", "card_linking",
          "top_up_failed", "top_up_reverted", "pending_top_up", "exchange_rate", "age_limit", "change_pin",
          "transfer_timing", "declined_transfer", "cash_withdrawal_charge", "receiving_money",
          "fiat_currency_support", "apr", "interest_rate", "passcode_forgotten", "terminate_account",
          "edit_personal_details", "verify_my_identity", "why_verify_identity", "beneficiary_not_allowed"]
CLINC = ["accept_reservations", "weather", "timer", "alarm", "spelling", "definition", "calories", "nutrition_info",
         "tell_joke", "fun_fact", "uber", "car_rental", "flip_coin", "roll_dice"]


def _pool(source="banking77", licence="cc-by-4.0", labels=LABELS):
    rows = []
    for i, gold in enumerate(labels):
        others = [lab for lab in labels if lab != gold][i % 5: i % 5 + 3]
        rows.append(_intent_row(i, gold, [gold] + others, source, licence))
    return rows


def test_reframe_keeps_gold_and_licence():
    rows = _pool() + _pool("clinc150", "cc-by-3.0", CLINC)
    out = reframe_intents(rows, range(7, 11), seed=0)
    assert len(out) == len(rows)
    orig = {r.id: r for r in rows}
    sizes = Counter()
    for d in out:
        d.validate()
        o = orig[d.metadata["orig_id"]]
        assert d.gold == o.gold and d.licence == o.licence and d.source == o.source
        assert d.state == o.state and d.family == "many_options_intent" and d.split == "pool"
        assert 7 <= len(d.options) <= 10
        # Candidates come from the same label space (source); original options that pass the rule are kept.
        space = {opt.key for r in rows if r.source == o.source for opt in r.options}
        assert set(d.keys) <= space
        assert {k for k in o.keys if k == o.gold or sibling_ok(o.gold, k)} <= set(d.keys)
        sizes[len(d.options)] += 1
    assert len(sizes) >= 2


def _pool_rows():
    from yev.schema import read_jsonl

    paths = [Path("data/stage0-pool/banking77.jsonl"), Path("data/stage0-pool/clinc150.jsonl")]
    if not all(p.exists() for p in paths):
        pytest.skip("intent pool not built")
    return [d for p in paths for d in read_jsonl(p)]


def _content_words(key):
    from yev.generators.many_options import _label_words

    return _label_words(key)


def test_reframe_siblings_share_no_content_word_with_gold():
    out = reframe_intents(_pool() + _pool("clinc150", "cc-by-3.0", CLINC), range(7, 11), seed=4)
    real = reframe_intents(_pool_rows()[::7], range(7, 11), seed=4)
    for d in out + real:
        if d.gold == NONE_KEY:
            continue
        for k in d.keys:
            if k not in (d.gold, NONE_KEY):
                assert not (_content_words(k) & _content_words(d.gold)), (d.id, d.gold, k)
    # plural and inflection folding
    assert _content_words("accept_reservations") & _content_words("cancel_reservation")
    assert _content_words("transaction_charged_twice") & _content_words("exchange_charge")


def test_reframe_never_pairs_near_duplicates():
    from yev.generators.many_options import NEAR_DUPLICATE_LABELS

    flagged = [("receiving_money", "fiat_currency_support"), ("disposable_card_limits", "get_disposable_virtual_card"),
               ("card_delivery_estimate", "card_arrival"), ("transfer_fee_charged", "card_payment_fee_charged"),
               ("apr", "interest_rate"), ("verify_my_identity", "why_verify_identity")]
    assert all(frozenset(p) in NEAR_DUPLICATE_LABELS for p in flagged)
    for d in reframe_intents(_pool() + _pool_rows(), range(7, 11), seed=5):
        if d.gold == NONE_KEY:
            continue
        for k in d.keys:
            assert frozenset((d.gold, k)) not in NEAR_DUPLICATE_LABELS, (d.id, d.gold, k)


def test_reframe_never_uses_test_split_rows():
    rows = _pool()
    bad = _intent_row(999, "age_limit", ["age_limit", "change_pin", "card_arrival"], split="dev")
    with pytest.raises(ValueError):
        reframe_intents(rows + [bad], range(7, 11), seed=0)
    test_src = _intent_row(998, "age_limit", ["age_limit", "change_pin", "card_arrival"], source="di_banking77_test")
    with pytest.raises(ValueError):
        reframe_intents(rows + [test_src], range(7, 11), seed=0)
    tagged = _intent_row(997, "age_limit", ["age_limit", "change_pin", "card_arrival"])
    tagged.metadata = {"hf_split": "test"}
    with pytest.raises(ValueError):
        reframe_intents(rows + [tagged], range(7, 11), seed=0)


def test_reframe_deterministic():
    a = reframe_intents(_pool(), range(7, 11), seed=3)
    b = reframe_intents(_pool(), range(7, 11), seed=3)
    assert [d.to_dict() for d in a] == [d.to_dict() for d in b]


def test_every_template_and_trigger_classifies_to_its_category():
    from yev.generators.many_options import DEPTS, MONTH_NAMES, NAMES, SPECS, classify

    slots = {"name": "X", "ref": "1", "dept": "Y", "day": "1 Z", "year": 2020, "pages": 2, "amount": "$1.00"}
    for name, cats in TAXONOMIES.items():
        for word in NAMES + DEPTS + MONTH_NAMES:
            assert classify(word, name) is None, word
        for tpl in SPECS[name].templates:
            assert classify(tpl.format(trigger="", **slots), name) is None
            for cat in cats:
                for phrase, kw in cat.triggers:
                    assert classify(tpl.format(trigger=phrase, **slots), name) == (cat.key, kw), phrase
            for phrase in SPECS[name].none_triggers:
                assert classify(tpl.format(trigger=phrase, **slots), name) is None, phrase


def test_every_cluster_has_a_none_member():
    for cid, rows in by_cluster(generate(200, seed=10)).items():
        assert any(r.gold == NONE_KEY for r in rows), cid


def test_nested_groups_listed_whole():
    from yev.generators.many_options import NESTED_GROUPS

    assert {"contract", "nda"} in [set(g) for g in NESTED_GROUPS["document_type"]]
    assert {"travel", "lodging"} <= next(set(g) for g in NESTED_GROUPS["expense_category"] if "travel" in g)
    assert {"lost_device", "physical"} in [set(g) for g in NESTED_GROUPS["incident_category"]]
    for name, groups in NESTED_GROUPS.items():
        keys = {c.key for c in TAXONOMIES[name]}
        assert all(g <= keys for g in groups)
    for d in generate(400, seed=12):
        listed = set(d.keys)
        for g in NESTED_GROUPS[d.metadata["taxonomy"]]:
            assert g <= listed or not (g & listed), (d.id, sorted(g), sorted(listed))
