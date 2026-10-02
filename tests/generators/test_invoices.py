from collections import Counter, defaultdict
from datetime import date, timedelta
from fractions import Fraction

from yev.generators.common import SYNTH_LICENCE
from yev.generators.dates import fmt
from yev.generators.invoices import OUTCOMES, Terms, discount_deadline, due_date, generate


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


# ---------------------------------------------------------------- independent re-statement of the rules


def _roll(d: date, hol: set) -> date:
    while d.weekday() >= 5 or d in hol:
        d += timedelta(days=1)
    return d


def _eom(d: date) -> date:
    nxt = date(d.year + d.month // 12, d.month % 12 + 1, 1)
    return nxt - timedelta(days=1)


def _anchor(m: dict) -> date:
    return date.fromisoformat(m["invoice_date"] if m["anchor"] == "invoice" else m["received_date"])


def _due(m: dict) -> date:
    t, a = m["terms"], _anchor(m)
    start = _eom(a) if t["kind"] == "eom" else a
    return _roll(start + timedelta(days=t["net"]), {date.fromisoformat(h) for h in m["holidays"]})


def _round_half_up(x: Fraction) -> int:
    return int((x * 2 + 1) // 2)


def _tax(m: dict) -> int:
    rate = Fraction(m["tax_rate"]) / 100
    base = sum(ln["qty"] * ln["price"] for ln in m["lines"] if ln["cat"] not in m["exempt_cats"])
    if m["freight_taxable"]:
        base += m["freight"]
    return _round_half_up(base * rate)


def _bad_line(m: dict, ln: dict) -> bool:
    limit = ln["ordered"] if m["receipt_missing"] else sum(ln["receipts"])
    over_price = ln["price"] * 100 > ln["po_price"] * (100 + m["tolerance_pct"])
    return ln["qty"] > limit or over_price


def _three_way(m: dict) -> str:
    if m["invoice_no"] in m["ledger_invoice_nos"]:
        return "duplicate"
    if m["receipt_missing"]:
        return "missing_receipt"
    if any(ln["qty"] > sum(ln["receipts"]) for ln in m["lines"]):
        return "quantity_variance"
    if any(ln["price"] * 100 > ln["po_price"] * (100 + m["tolerance_pct"]) for ln in m["lines"]):
        return "price_variance"
    return "match"


def _key_with(d, text: str) -> str:
    [key] = [o.key for o in d.options if o.description == text]
    return key


def expected(d) -> str:
    m = d.metadata
    kind = m["kind"]
    subtotal = sum(ln["qty"] * ln["price"] for ln in m["lines"])
    if kind == "totals_match":
        ok = m["printed_subtotal"] == subtotal and m["printed_total"] == subtotal + m["freight"] + m["printed_tax"]
        return "yes" if ok else "no"
    if kind == "tax_correct":
        return "yes" if m["printed_tax"] == _tax(m) else "no"
    if kind == "three_way":
        return _three_way(m)
    if kind == "due_date":
        return _key_with(d, fmt(_due(m), m["option_style"]))
    if kind == "discount":
        t = m["terms"]
        if t["kind"] != "discount":
            return "no"
        return "yes" if date.fromisoformat(m["pay_date"]) <= _anchor(m) + timedelta(days=t["disc_days"]) else "no"
    if kind == "approval":
        return str(sum(1 for c in m["approval_cuts"] if m["printed_total"] >= c))
    if kind == "wrong_line":
        bad = [ln["no"] for ln in m["lines"] if _bad_line(m, ln)]
        assert len(bad) <= 1, bad
        return f"line_{bad[0]}" if bad else "none"
    if kind == "bank_match":
        return "yes" if m["invoice_bank"] == m["master_bank"] else "no"
    if kind == "tax_amount":
        cents = _tax(m)
        return _key_with(d, f"${cents // 100:,}.{cents % 100:02d}")
    if kind == "received_qty":
        [ln] = [ln for ln in m["lines"] if ln["sku"] == m["target_sku"]]
        n = sum(ln["receipts"])
        return _key_with(d, f"{n} unit{'s' if n != 1 else ''}")
    raise AssertionError(kind)


# ---------------------------------------------------------------- tests


def test_invoice_labels_match_injected_faults():
    ds = generate(150, seed=5)
    kinds = Counter()
    golds: dict[str, Counter] = defaultdict(Counter)
    for d in ds:
        d.validate()
        assert d.gold == expected(d), (d.id, d.metadata["kind"])
        kinds[d.metadata["kind"]] += 1
        golds[d.metadata["kind"]][d.gold] += 1
    assert len(kinds) == 10 and min(kinds.values()) >= 50, kinds
    for k in ("totals_match", "tax_correct", "discount", "bank_match"):
        assert min(golds[k]["yes"], golds[k]["no"]) >= 0.1 * kinds[k], (k, golds[k])
    assert set(golds["three_way"]) == set(OUTCOMES), golds["three_way"]
    assert set(golds["approval"]) == {"0", "1", "2", "3"}, golds["approval"]
    assert golds["wrong_line"]["none"] < 0.7 * kinds["wrong_line"]

    for rows in by_cluster(ds).values():
        f = rows[0].metadata["faults"]
        gold = {d.metadata["kind"]: d.gold for d in rows}
        outcome = ("duplicate" if f["duplicate"] else "missing_receipt" if f["missing_receipt"]
                   else {"qty": "quantity_variance", "price": "price_variance", None: "match"}[f["line"]])
        if "three_way" in gold:
            assert gold["three_way"] == outcome, (f, gold)
        if "wrong_line" in gold:
            assert gold["wrong_line"] == (f"line_{f['line_no']}" if f["line"] else "none"), (f, gold)
        if "tax_correct" in gold:
            assert gold["tax_correct"] == ("no" if f["tax"] else "yes"), (f, gold)
        if "totals_match" in gold:
            assert gold["totals_match"] == ("no" if f["totals"] else "yes"), (f, gold)
        if "bank_match" in gold:
            assert gold["bank_match"] == ("no" if f["bank"] else "yes"), (f, gold)


def test_document_questions_share_state_and_cluster():
    ds = generate(60, seed=2)
    clusters = by_cluster(ds)
    assert sorted(clusters, key=lambda c: int(c.rsplit(":", 1)[1])) == [f"invoices:2:{k}" for k in range(60)]
    for cid, rows in clusters.items():
        assert 6 <= len(rows) <= 10, (cid, len(rows))
        assert len({d.state for d in rows}) == 1
        assert len({d.metadata["kind"] for d in rows}) == len(rows)
        for d in rows:
            assert d.family == "invoice_ledger" and d.source == "synthetic_rules" and d.licence == SYNTH_LICENCE
            assert d.edit_type is None and d.id.startswith(cid + ":")
            assert len(d.options) <= 10
            kind = d.metadata["kind"]
            if kind == "wrong_line":
                assert d.type == "choice" and 5 <= len(d.options) <= 10 and d.keys[-1] == "none"
            elif kind == "approval":
                assert d.type == "score" and d.keys == ["0", "1", "2", "3"]
            elif kind == "three_way":
                assert d.type == "choice" and d.keys == list(OUTCOMES)
            elif kind in ("due_date", "tax_amount", "received_qty"):
                assert d.type == "choice" and 4 <= len(d.options) <= 6
            else:
                assert d.type == "noul"


def test_document_length_range():
    ds = generate(200, seed=9)
    toks = [rows[0].metadata["approx_tokens"] for rows in by_cluster(ds).values()]
    for rows in by_cluster(ds).values():
        assert rows[0].metadata["approx_tokens"] == round(len(rows[0].state) / 4)
    inside = sum(1 for t in toks if 2000 <= t <= 8000)
    assert inside >= 0.95 * len(toks), sorted(toks)


def test_due_date_uses_terms_and_business_days():
    hol = (date(2025, 4, 4),)
    # Net 30 from Fri 10 Jan 2025 lands on Sun 9 Feb -> Mon 10 Feb.
    assert due_date(Terms("net", 30), date(2025, 1, 10), ()) == date(2025, 2, 10)
    assert discount_deadline(Terms("net", 30), date(2025, 1, 10)) is None
    # 2/10 Net 30 from Wed 5 Mar 2025: the discount deadline (Sat 15 Mar) never moves; the net date
    # (Fri 4 Apr) is a holiday and moves to Mon 7 Apr.
    t = Terms("discount", 30, disc_pct=2, disc_days=10)
    assert discount_deadline(t, date(2025, 3, 5)) == date(2025, 3, 15)
    assert due_date(t, date(2025, 3, 5), hol) == date(2025, 4, 7)
    assert due_date(t, date(2025, 3, 5), ()) == date(2025, 4, 4)
    # Net 30 EOM: count from the last day of the anchor's month (leap year: 31 Jan 2024 + 30 = Fri 1 Mar).
    assert due_date(Terms("eom", 30), date(2024, 1, 17), ()) == date(2024, 3, 1)
    # Net 15 EOM from 2 May 2025: 31 May + 15 = Sun 15 Jun -> Mon 16 Jun.
    assert due_date(Terms("eom", 15), date(2025, 5, 2), ()) == date(2025, 6, 16)
    # And the generated packets use the same rule (checked independently in expected()).
    seen = Counter()
    for d in generate(80, seed=4):
        if d.metadata["kind"] == "due_date":
            seen[d.metadata["terms"]["kind"]] += 1
            assert d.gold == expected(d)
    assert set(seen) == {"net", "discount", "eom"}, seen


def test_deterministic():
    a, b = generate(12, seed=3), generate(12, seed=3)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert [x.state for x in a] != [x.state for x in generate(12, seed=4)]
    assert all(x.id.startswith("invoices:3:") for x in a)


def test_registered_and_passes_licence_stage():
    from yev.cli import RULE_GENERATORS
    from yev.filters.pipeline import _licence_ok
    from yev.generators.specs import SYNTHETIC_SPECS

    assert RULE_GENERATORS["invoice_ledger"] is generate
    specs = {s.name: s for s in SYNTHETIC_SPECS}
    assert all(_licence_ok(d, specs, {}) for d in generate(3, seed=0))
