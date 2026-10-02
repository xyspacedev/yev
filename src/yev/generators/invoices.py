"""Long accounts-payable packets with several computed questions each (Stage 1, family 3).

Each document is one review packet: the buyer's AP policy excerpt (tax rate, matching tolerance,
payment-term definitions, company holidays, approval bands, duplicate and bank rules), the vendor
master record, a purchase order, zero to two goods receipts, the supplier invoice (5-60 lines), the
vendor's payment history, an email thread and a routing slip with the scheduled payment date. It
renders to roughly 2-8k tokens with tables in markdown or fixed-width style.

Every packet carries 6-10 questions over the identical state, sharing one cluster_id. These are not
contrastive clusters (no edit_type, no two-gold requirement); the cluster id only keeps a document's
questions together through filtering and splitting. All labels are computed from the structured data,
into which faults are injected at fixed rates: a wrong quantity or price on one line, a duplicate
invoice number, a missing receipt, a wrong tax amount, an arithmetic slip in the totals and a changed
remit-to bank account. Money is held in integer cents and rates as Fractions, so labels are exact.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from fractions import Fraction

from yev.generators.common import SYNTH_LICENCE
from yev.generators.dates import STYLES, fmt, is_business_day, last_day_of_month
from yev.schema import Decision, Option

SOURCE = "synthetic_rules"
FAMILY = "invoice_ledger"
SLASH_NOTE = "Slash dates are written day/month/year."
KINDS = ("totals_match", "tax_correct", "three_way", "due_date", "discount", "approval", "wrong_line",
         "bank_match", "tax_amount", "received_qty")
OUTCOMES = ("match", "price_variance", "quantity_variance", "missing_receipt", "duplicate")
NOUL_OPTIONS = [Option("yes", "Yes."), Option("no", "No.")]

# Fault rates (per document).
P_LINE_QTY, P_LINE_PRICE = 0.25, 0.25
P_DUPLICATE, P_MISSING_RECEIPT = 0.12, 0.12
P_TAX, P_TOTALS, P_BANK = 0.4, 0.3, 0.25
P_MISSED_DISCOUNT = 0.3  # of packets with discount terms

TAX_RATES = ("5", "6", "6.25", "7", "7.5", "8", "8.25", "8.875", "9.5", "10", "13")
TOLERANCES = (0, 1, 2, 2, 3, 5, 5)
APPROVAL_BASES = ((500, 2500, 10000), (1000, 5000, 25000), (2000, 15000, 75000), (2500, 10000, 50000),
                  (5000, 25000, 100000), (10000, 50000, 250000), (20000, 75000, 200000), (25000, 100000, 500000),
                  (50000, 150000, 500000), (100000, 250000, 1000000))  # dollars
APPROVERS = ("an AP clerk", "the budget-holding manager", "the finance director", "the CFO")

VENDORS = ["Calder Industrial Supply", "Brookline Fasteners Ltd", "Westmere Packaging Co.", "Halvorsen Electrical",
           "Pinecrest Lab Services", "Orchard Valley Office Products", "Tamsin Tool & Die", "Ridgeback Safety Gear",
           "Marlowe Chemical Distributors", "Northfold Facilities Services", "Granite Peak Hydraulics",
           "Bellweather Print & Label", "Sundial Bearings Inc.", "Quillon Abrasives", "Harrowgate Pump Works"]
BUYERS = ["Ashcombe Manufacturing", "Delacroix Foods", "Fernhill Medical Devices", "Kestrel Logistics",
          "Lindqvist Robotics", "Oakhurst Brewing Company", "Sorrel Biotech", "Thornbury Plastics",
          "Vantor Aerospace Components", "Wexley Cold Storage"]
STREETS = ["Foundry Road", "Mill Lane", "Harbour Way", "Station Street", "Quarry Drive", "Commerce Park",
           "Kingfisher Avenue", "Riverside Trade Estate"]
TOWNS = ["Easton", "Millbrook", "Port Carrow", "Hollins", "Westfield", "Ardley", "Brackwater", "Selby Cross"]
PEOPLE = ["Priya Natarajan", "Tom Okafor", "Lena Vasquez", "Marcus Bell", "Hana Sato", "Ruth Adeyemi",
          "Diego Ferreira", "Agnes Kowalski", "Sam Whitlock", "Nadia Haddad", "Owen Price", "Ines Moreau"]

# (description, category, unit price range in cents, quantity range, variants)
ITEMS = [
    ("Nitrile gloves, box of 100", "SUP", (650, 1800), (10, 200), ("size S", "size M", "size L", "size XL")),
    ("Hex bolt, zinc plated, box of 50", "MAT", (900, 4200), (5, 120), ("M6 x 30", "M8 x 40", "M10 x 50", "M12 x 60")),
    ("Corrugated shipping carton", "PKG", (85, 640), (100, 2000), ("300 x 200 x 150", "400 x 300 x 250", "600 x 400 x 400")),
    ("Stretch wrap film roll", "PKG", (1400, 3800), (6, 96), ("17 micron", "23 micron", "machine grade")),
    ("Safety glasses, anti-fog", "SUP", (420, 1650), (12, 240), ("clear lens", "tinted lens", "over-spec")),
    ("Hydraulic hose assembly", "MAT", (2800, 16500), (2, 40), ("1/4 in, 1 m", "3/8 in, 2 m", "1/2 in, 3 m")),
    ("Cordless drill kit", "EQP", (9900, 32900), (1, 12), ("18 V", "20 V brushless")),
    ("Pallet jack", "EQP", (29000, 74000), (1, 6), ("2,500 kg", "3,000 kg low-profile")),
    ("Cable tie pack of 1,000", "ELC", (700, 2600), (5, 80), ("100 mm", "200 mm", "300 mm UV black")),
    ("LED high-bay fixture", "ELC", (8800, 26500), (2, 40), ("100 W", "150 W", "200 W")),
    ("Copper wire spool", "ELC", (4200, 21000), (2, 30), ("1.5 mm2", "2.5 mm2", "4 mm2")),
    ("Printer toner cartridge", "SUP", (3900, 15500), (2, 40), ("black", "cyan", "magenta", "yellow")),
    ("Copy paper, case of 5 reams", "SUP", (2100, 4600), (5, 100), ("A4 80 gsm", "letter 20 lb")),
    ("Stainless steel sheet", "MAT", (6400, 38000), (2, 60), ("1 mm, 1 x 2 m", "2 mm, 1 x 2 m", "3 mm, 1.25 x 2.5 m")),
    ("Aluminium angle bar", "MAT", (1200, 7400), (10, 150), ("25 x 25 mm", "40 x 40 mm", "50 x 50 mm")),
    ("Bearing, deep groove", "MAT", (350, 4800), (10, 300), ("6204-2RS", "6205-2RS", "6306-ZZ")),
    ("Respirator cartridge pair", "SUP", (1100, 3300), (10, 120), ("organic vapour", "P100", "multi-gas")),
    ("Floor marking tape", "SUP", (1500, 4900), (6, 60), ("yellow 50 mm", "white 75 mm", "hazard stripe")),
    ("Industrial degreaser, 20 L", "SUP", (4300, 12900), (2, 30), ("citrus", "solvent-free")),
    ("Label roll, thermal", "PKG", (900, 3600), (10, 200), ("100 x 150 mm", "50 x 25 mm", "102 x 76 mm")),
    ("Torque wrench", "EQP", (8500, 41000), (1, 10), ("3/8 in drive", "1/2 in drive")),
    ("Laboratory pipette tips, rack of 96", "SUP", (450, 1900), (20, 400), ("10 ul", "200 ul", "1000 ul")),
    ("Air filter cartridge", "MAT", (1800, 9600), (4, 60), ("G4 panel", "F7 bag", "HEPA H13")),
    ("Steel shelving bay", "EQP", (14500, 48000), (1, 20), ("5 shelf", "6 shelf heavy duty")),
    ("Pneumatic fitting set", "MAT", (2200, 8800), (4, 50), ("6 mm push-in", "8 mm push-in", "1/4 BSP")),
    ("On-site installation labour, per hour", "SVC", (6500, 14500), (4, 80), ("standard rate", "weekend rate")),
    ("Calibration service, per instrument", "SVC", (4500, 22000), (2, 40), ("pressure gauge", "torque tool", "balance")),
    ("Preventive maintenance visit", "SVC", (18000, 65000), (1, 12), ("quarterly", "annual")),
    ("Equipment rental, per day", "SVC", (9500, 42000), (1, 30), ("scissor lift", "forklift", "generator")),
    ("Waste collection, per lift", "SVC", (7500, 26000), (2, 24), ("general waste", "recycling", "hazardous")),
]
CATEGORY_NAMES = {"SUP": "consumable supplies", "MAT": "materials and components", "PKG": "packaging",
                  "EQP": "equipment", "ELC": "electrical", "SVC": "services"}


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


def round_half_up(x: Fraction) -> int:
    return int((x * 2 + 1) // 2)


# ---------------------------------------------------------------- data


@dataclass(frozen=True)
class Terms:
    kind: str  # "net", "discount" (d/D net N) or "eom" (net N counted from the end of the month)
    net: int
    disc_pct: int | None = None
    disc_days: int | None = None

    def label(self) -> str:
        if self.kind == "discount":
            return f"{self.disc_pct}/{self.disc_days} Net {self.net}"
        if self.kind == "eom":
            return f"Net {self.net} EOM"
        return f"Net {self.net}"

    def meta(self) -> dict:
        return {"kind": self.kind, "net": self.net, "disc_pct": self.disc_pct, "disc_days": self.disc_days}


def roll_forward(d: date, holidays) -> date:
    while not is_business_day(d, holidays):
        d += timedelta(days=1)
    return d


def unrolled_due(terms: Terms, anchor: date) -> date:
    start = last_day_of_month(anchor.year, anchor.month) if terms.kind == "eom" else anchor
    return start + timedelta(days=terms.net)


def due_date(terms: Terms, anchor: date, holidays) -> date:
    """Net due date: N days after the anchor (or after its month end for EOM), moved to the next business day."""
    return roll_forward(unrolled_due(terms, anchor), holidays)


def discount_deadline(terms: Terms, anchor: date) -> date | None:
    """Last day the early-payment discount may be taken; it never moves for weekends or holidays."""
    if terms.kind != "discount":
        return None
    return anchor + timedelta(days=terms.disc_days)


@dataclass(frozen=True)
class Line:
    no: int  # invoice line number
    po_line: int
    sku: str
    desc: str
    cat: str
    ordered: int
    po_price: int  # cents
    receipts: tuple[int, ...]  # quantity in each goods receipt (empty when no receipt was posted)
    qty: int  # billed quantity
    price: int  # billed unit price, cents

    @property
    def received(self) -> int:
        return sum(self.receipts)

    @property
    def amount(self) -> int:
        return self.qty * self.price

    def meta(self) -> dict:
        return {"no": self.no, "sku": self.sku, "cat": self.cat, "ordered": self.ordered,
                "po_price": self.po_price, "receipts": list(self.receipts), "qty": self.qty, "price": self.price}


@dataclass(frozen=True)
class Packet:
    lines: tuple[Line, ...]
    receipt_missing: bool
    tolerance_pct: int
    tax_rate: str  # percent, as a decimal string
    exempt_cats: tuple[str, ...]
    freight: int
    freight_taxable: bool
    printed_subtotal: int
    printed_tax: int
    printed_total: int
    invoice_no: str
    ledger: tuple[tuple[str, date, int, date], ...]  # (invoice no, invoice date, amount, paid on)
    terms: Terms
    invoice_terms: Terms  # terms printed on the invoice; the vendor master's terms govern
    anchor: str  # "invoice" (invoice date) or "received" (AP received stamp)
    invoice_date: date
    received_date: date
    holidays: tuple[date, ...]
    pay_date: date
    discount_claimed: bool  # the routing slip deducts the early-payment discount
    approval_cuts: tuple[int, ...]  # cents
    master_bank: str
    invoice_bank: str
    faults: dict


def subtotal(p: Packet) -> int:
    return sum(ln.amount for ln in p.lines)


def tax_base(p: Packet, exempt=None, freight_taxable=None) -> int:
    exempt = p.exempt_cats if exempt is None else exempt
    ft = p.freight_taxable if freight_taxable is None else freight_taxable
    return sum(ln.amount for ln in p.lines if ln.cat not in exempt) + (p.freight if ft else 0)


def tax_at(base: int, rate: str) -> int:
    return round_half_up(base * Fraction(rate) / 100)


def correct_tax(p: Packet) -> int:
    return tax_at(tax_base(p), p.tax_rate)


def anchor_date(p: Packet) -> date:
    return p.invoice_date if p.anchor == "invoice" else p.received_date


def price_over(p: Packet, ln: Line) -> bool:
    return ln.price * 100 > ln.po_price * (100 + p.tolerance_pct)


def qty_over(p: Packet, ln: Line) -> bool:
    return ln.qty > (ln.ordered if p.receipt_missing else ln.received)


def wrong_lines(p: Packet) -> list[int]:
    return [ln.no for ln in p.lines if price_over(p, ln) or qty_over(p, ln)]


def three_way(p: Packet) -> str:
    if p.invoice_no in {x[0] for x in p.ledger}:
        return "duplicate"
    if p.receipt_missing:
        return "missing_receipt"
    if any(qty_over(p, ln) for ln in p.lines):
        return "quantity_variance"
    if any(price_over(p, ln) for ln in p.lines):
        return "price_variance"
    return "match"


def totals_ok(p: Packet) -> bool:
    return p.printed_subtotal == subtotal(p) and p.printed_total == p.printed_subtotal + p.freight + p.printed_tax


def discount_ok(p: Packet) -> bool:
    dl = discount_deadline(p.terms, anchor_date(p))
    return dl is not None and p.pay_date <= dl


def approval_level(total: int, cuts) -> int:
    return sum(1 for c in cuts if total >= c)


# ---------------------------------------------------------------- sampling


def random_day(rng: random.Random) -> date:
    r = rng.random()
    if r < 0.25:  # late in a month, so terms cross a month end
        return last_day_of_month(rng.randint(2023, 2027), rng.randint(1, 12)) - timedelta(days=rng.randint(0, 6))
    if r < 0.4:
        return date(rng.randint(2023, 2027), rng.choice((1, 2, 11, 12)), rng.randint(1, 28))
    return date(2023, 1, 1) + timedelta(days=rng.randrange(5 * 365))


def sample_terms(rng: random.Random) -> Terms:
    r = rng.random()
    if r < 0.3:
        return Terms("net", rng.choice((15, 30, 30, 30, 45, 60)))
    if r < 0.75:
        return Terms("discount", rng.choice((30, 30, 45, 60)), disc_pct=rng.choice((1, 2, 2, 3)),
                     disc_days=rng.choice((10, 10, 15, 20)))
    return Terms("eom", rng.choice((15, 30, 30, 45, 60)))


def sample_lines(rng: random.Random, n: int, receipts: int, tol: int) -> list[Line]:
    pool = [(desc, cat, pr, qr, v) for desc, cat, pr, qr, vs in ITEMS for v in vs]
    picks = rng.sample(pool, n) if n <= len(pool) else [rng.choice(pool) for _ in range(n)]
    skus = rng.sample(range(10000, 99999), n)
    po_order = list(range(n))
    if rng.random() < 0.5:
        rng.shuffle(po_order)
    lines = []
    for i, ((desc, cat, (plo, phi), (qlo, qhi), var), sku) in enumerate(zip(picks, skus)):
        ordered = rng.randint(qlo, qhi)
        po_price = rng.randint(plo, phi)
        if po_price >= 200 and rng.random() < 0.5:
            po_price = po_price // 100 * 100  # whole dollars, so a tolerance edge can be hit exactly
        if receipts == 0:
            recs: tuple[int, ...] = ()
            qty = ordered
        else:
            got = ordered if rng.random() < 0.75 or ordered < 2 else rng.randint(1, ordered - 1)
            if receipts == 2 and got >= 2 and rng.random() < 0.6:
                first = rng.randint(1, got - 1)
                recs = (first, got - first)
            elif receipts == 2:
                recs = (got, 0) if rng.random() < 0.5 else (0, got)
            else:
                recs = (got,)
            qty = got
        lines.append(Line(no=i + 1, po_line=10 * (po_order.index(i) + 1), sku=f"{cat}-{sku}",
                          desc=f"{desc}, {var}", cat=cat, ordered=ordered, po_price=po_price,
                          receipts=recs, qty=qty, price=po_price))
    # Distractors: prices that moved but stay inside tolerance (sometimes exactly on it), or went down.
    for j in rng.sample(range(n), min(n, rng.randint(0, 3))):
        ln = lines[j]
        edge = ln.po_price * (100 + tol) // 100
        if tol and edge > ln.po_price and rng.random() < 0.7:
            price = edge if rng.random() < 0.5 else rng.randint(ln.po_price + 1, edge)
        else:
            price = max(1, ln.po_price - rng.randint(1, max(1, ln.po_price // 20)))
        lines[j] = replace(ln, price=price)
    return lines


def inject_line_fault(rng: random.Random, lines: list[Line], kind: str, tol: int, missing: bool) -> int:
    if kind == "qty":
        j = rng.randrange(len(lines))
        ln = lines[j]
        if ln.received < ln.ordered and rng.random() < 0.6:
            qty = ln.ordered  # billed for the full order while only part has arrived
        else:
            qty = ln.received + rng.randint(1, max(1, ln.received // 5))
        lines[j] = replace(ln, qty=qty)
    else:
        j = rng.randrange(len(lines))
        ln = lines[j]
        edge = ln.po_price * (100 + tol) // 100
        bump = 1 if rng.random() < 0.3 else rng.randint(1, max(1, ln.po_price // 8))
        lines[j] = replace(ln, price=edge + bump)
    return lines[j].no


def transpose_digits(rng: random.Random, cents: int) -> int:
    s = str(cents)
    spots = [i for i in range(len(s) - 1) if s[i] != s[i + 1]]
    if not spots:
        return cents + 100 * rng.choice((10, 100))
    i = rng.choice(spots)
    return int(s[:i] + s[i + 1] + s[i] + s[i + 2:])


def near_miss_number(rng: random.Random, inv: str) -> str:
    prefix, digits = inv.rsplit("-", 1)
    for _ in range(10):
        i = rng.randrange(len(digits) - 1)
        alt = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2:]
        if alt != digits:
            return f"{prefix}-{alt}"
    return f"{prefix}-{int(digits) + 1}"


def sample_packet(rng: random.Random) -> Packet:
    n = rng.randint(5, 60)
    tol = rng.choice(TOLERANCES)
    missing = rng.random() < P_MISSING_RECEIPT
    n_receipts = 0 if missing else (2 if rng.random() < 0.35 else 1)
    lines = sample_lines(rng, n, n_receipts, tol)

    r = rng.random()
    line_fault = "qty" if r < P_LINE_QTY else "price" if r < P_LINE_QTY + P_LINE_PRICE else None
    if missing and line_fault == "qty":
        line_fault = None
    line_no = inject_line_fault(rng, lines, line_fault, tol, missing) if line_fault else None

    freight = 0 if rng.random() < 0.35 else rng.randint(15, 900) * 100 + rng.choice((0, 0, 50, rng.randint(1, 99)))
    tax_rate = rng.choice(TAX_RATES)
    exempt = ("SVC",) if rng.random() < 0.5 else ()
    if rng.random() < 0.2:
        exempt += ("PKG",)
    freight_taxable = rng.random() < 0.4
    base = Packet(lines=tuple(lines), receipt_missing=missing, tolerance_pct=tol, tax_rate=tax_rate,
                  exempt_cats=exempt, freight=freight, freight_taxable=freight_taxable, printed_subtotal=0,
                  printed_tax=0, printed_total=0, invoice_no="", ledger=(), terms=Terms("net", 30),
                  invoice_terms=Terms("net", 30), anchor="invoice", invoice_date=date(2024, 1, 1),
                  received_date=date(2024, 1, 1), holidays=(), pay_date=date(2024, 1, 1), discount_claimed=False, approval_cuts=(),
                  master_bank="", invoice_bank="", faults={})

    # Tax, possibly wrong.
    good_tax = correct_tax(base)
    tax_fault = None
    printed_tax = good_tax
    if rng.random() < P_TAX:
        choices = ["rate", "rate"]
        if freight:
            choices.append("freight")
        if any(ln.cat in exempt for ln in lines):
            choices += ["exempt", "exempt"]
        tax_fault = rng.choice(choices)
        if tax_fault == "rate":
            printed_tax = tax_at(tax_base(base), rng.choice([t for t in TAX_RATES if t != tax_rate]))
        elif tax_fault == "freight":
            printed_tax = tax_at(tax_base(base, freight_taxable=not freight_taxable), tax_rate)
        else:
            printed_tax = tax_at(tax_base(base, exempt=()), tax_rate)
        if printed_tax == good_tax:
            tax_fault = None
    wrong_rate = None
    if tax_fault == "rate":
        wrong_rate = next(t for t in TAX_RATES if tax_at(tax_base(base), t) == printed_tax)

    # Totals, possibly wrong.
    sub = subtotal(base)
    printed_sub, totals_fault = sub, None
    printed_total = sub + freight + printed_tax
    if rng.random() < P_TOTALS:
        totals_fault = rng.choice(("subtotal", "total"))
        if totals_fault == "subtotal":
            ln = rng.choice(lines)
            printed_sub = sub - ln.amount if rng.random() < 0.6 else sub + ln.amount
            printed_total = printed_sub + freight + printed_tax
        else:
            printed_total = (transpose_digits(rng, printed_total) if rng.random() < 0.6
                             else printed_total + rng.choice((-1, 1)) * rng.choice((100, 1000, 10000)))
        if printed_sub == sub and printed_total == sub + freight + printed_tax:
            totals_fault = None

    # Invoice number and payment history.
    prefix = rng.choice(("INV", "SI", "IN", "BIL"))
    inv_no = f"{prefix}-{rng.randint(10000, 999999)}"
    duplicate = rng.random() < P_DUPLICATE
    inv_date = random_day(rng)
    received = inv_date + timedelta(days=rng.randint(1, 12))
    ledger_nos = set()
    ledger = []
    for _ in range(rng.randint(3, 8)):
        no = f"{prefix}-{int(inv_no.split('-')[1]) - rng.randint(5, 4000)}"
        if no in ledger_nos or no == inv_no:
            continue
        ledger_nos.add(no)
        d0 = inv_date - timedelta(days=rng.randint(70, 360))
        ledger.append((no, d0, rng.randint(20_000, 5_000_000), d0 + timedelta(days=rng.randint(10, 60))))
    if duplicate:
        d0 = inv_date - timedelta(days=rng.randint(0, 40)) if rng.random() < 0.5 else inv_date
        amount = printed_total if rng.random() < 0.6 else printed_total + rng.randint(-50_000, 50_000)
        paid = d0 + timedelta(days=rng.randint(1, max(1, (received - d0).days)))
        ledger.append((inv_no, d0, max(100, amount), paid))
    elif rng.random() < 0.6:
        alt = near_miss_number(rng, inv_no)
        if alt not in ledger_nos:
            d0 = inv_date - timedelta(days=rng.randint(70, 150))
            ledger.append((alt, d0, rng.randint(20_000, 5_000_000), d0 + timedelta(days=rng.randint(10, 60))))
    ledger.sort(key=lambda x: x[1])

    # Terms and dates.
    terms = sample_terms(rng)
    invoice_terms = terms if rng.random() < 0.75 else sample_terms(rng)
    anchor = "invoice" if rng.random() < 0.6 else "received"
    a = inv_date if anchor == "invoice" else received
    raw = unrolled_due(terms, a)
    holidays = set()
    if raw.weekday() < 5 and rng.random() < 0.35:
        holidays.add(raw)
        if rng.random() < 0.3 and (raw + timedelta(days=1)).weekday() < 5:
            holidays.add(raw + timedelta(days=1))
    while len(holidays) < rng.randint(2, 5):
        holidays.add(raw + timedelta(days=rng.randint(-80, 80)))
    holidays_t = tuple(sorted(holidays))
    dl = discount_deadline(terms, a)
    claimed = False
    if dl is not None and rng.random() < P_MISSED_DISCOUNT:
        pay, claimed = dl + timedelta(days=1), True  # missed discount: one day late, yet the slip deducts it
    elif dl is not None:
        pay = dl + timedelta(days=rng.choice((-6, -4, -3, -2, -1, -1, 0, 0, 0, 2)))
        claimed = pay <= dl and rng.random() < 0.7
    else:
        pay = due_date(terms, a, holidays_t) - timedelta(days=rng.randint(0, 25))
    pay = max(pay, received + timedelta(days=3))
    missed_discount = claimed and pay > dl

    # Approval bands: aim the printed total at a random level.
    target = rng.randrange(4)
    cands = [tuple(100 * c for c in b) for b in APPROVAL_BASES]
    hits = [c for c in cands if approval_level(printed_total, c) == target]
    cuts = rng.choice(hits) if hits else rng.choice(cands)

    master_bank = f"{rng.randint(0, 9999):04d}"
    bank_fault = rng.random() < P_BANK
    invoice_bank = master_bank
    if bank_fault:
        while invoice_bank == master_bank:
            invoice_bank = f"{rng.randint(0, 9999):04d}"

    faults = {"line": line_fault, "line_no": line_no, "duplicate": duplicate, "missing_receipt": missing,
              "tax": tax_fault, "wrong_rate": wrong_rate, "totals": totals_fault, "bank": bank_fault,
              "missed_discount": missed_discount}
    return replace(base, printed_subtotal=printed_sub, printed_tax=printed_tax, printed_total=printed_total,
                   invoice_no=inv_no, ledger=tuple(ledger), terms=terms, invoice_terms=invoice_terms,
                   anchor=anchor, invoice_date=inv_date, received_date=received, holidays=holidays_t,
                   pay_date=pay, discount_claimed=claimed, approval_cuts=cuts, master_bank=master_bank, invoice_bank=invoice_bank,
                   faults=faults)


# ---------------------------------------------------------------- rendering


def table(headers: list[str], rows: list[list[str]], style: str, right: set[int]) -> str:
    if style == "markdown":
        out = ["| " + " | ".join(headers) + " |",
               "|" + "|".join("---:" if i in right else "---" for i in range(len(headers))) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def cell(i: int, s: str) -> str:
        return s.rjust(widths[i]) if i in right else s.ljust(widths[i])

    out = ["  ".join(cell(i, h) for i, h in enumerate(headers)).rstrip(),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(cell(i, s) for i, s in enumerate(r)).rstrip() for r in rows]
    return "\n".join(out)


@dataclass(frozen=True)
class Look:
    """Everything about a packet's text that is not label-bearing."""

    vendor: str
    vendor_id: str
    buyer: str
    po_no: str
    packet_no: str
    grn_nos: tuple[str, ...]
    doc_style: str
    mail_style: str
    table_style: str
    vendor_rep: str
    buyer_rep: str
    ap_clerk: str
    address: str
    tax_id: str
    cost_centre: str
    policy_variant: int
    policy_last: bool
    show_rate: bool
    show_cat_names: bool
    option_style: str  # date style of the due-date question's options


def render_policy(lk: Look, p: Packet) -> str:
    D = lambda d: fmt(d, lk.doc_style)  # noqa: E731
    v = lk.policy_variant
    anchor_txt = ("the invoice date printed on the invoice" if p.anchor == "invoice"
                  else "the date accounts payable received the invoice (the AP received stamp)")
    hol = ", ".join(D(h) for h in p.holidays)
    ex = [CATEGORY_NAMES[c] + f" ({c})" for c in p.exempt_cats]
    if ex:
        exempt_txt = ("Lines in these categories carry no sales tax: " + "; ".join(ex) + ". "
                      "Every other category is taxable.")
    else:
        exempt_txt = "Every line category is taxable, services included."
    freight_txt = ("Freight charges are taxable at the same rate." if p.freight_taxable
                   else "Freight charges are not taxable.")
    tol = p.tolerance_pct
    tol_txt = ("The billed unit price must not exceed the PO unit price at all."
               if tol == 0 else
               f"The billed unit price may exceed the PO unit price by up to {tol}% of the PO price "
               f"(exactly {tol}% is still acceptable); anything higher is a price variance. A lower price is acceptable.")
    lo = [money(c) for c in p.approval_cuts]
    bands = (f"- Level 0: invoice total below {lo[0]}: {APPROVERS[0]} may approve.\n"
             f"- Level 1: from {lo[0]} up to, but not including, {lo[1]}: {APPROVERS[1]}.\n"
             f"- Level 2: from {lo[1]} up to, but not including, {lo[2]}: {APPROVERS[2]}.\n"
             f"- Level 3: {lo[2]} or more: {APPROVERS[3]}.")
    head = (f"{lk.buyer}: accounts payable desk rules (extract, rev. {v + 3})" if v == 0
            else f"Extract from the {lk.buyer} AP handbook, sections relevant to supplier invoices")
    parts = [
        head,
        "",
        "A. Payment terms",
        f"A1. The payment terms recorded in the vendor master record apply. If the invoice prints different "
        f"terms, the vendor master terms still govern.",
        f"A2. Terms are counted from {anchor_txt}. The day after that date counts as day 1.",
        "A3. \"Net N\" means the invoice is due on day N.",
        "A4. \"d/D Net N\" means the payer may deduct d% of the invoice total if payment is released on or before "
        "day D; otherwise the full amount is due on day N.",
        "A5. \"Net N EOM\" means counting starts from the last day of the calendar month containing the start date: "
        "the first day of the following month is day 1 and the invoice is due on day N.",
        "A6. If a net due date falls on a Saturday, a Sunday or a company holiday, the due date becomes the next "
        "day that is none of these. The early-payment discount deadline is never moved.",
        f"A7. Company holidays in this period: {hol}. Vendor or bank closures listed elsewhere do not count.",
        "",
        "B. Sales tax",
        f"B1. The applicable sales tax rate is {p.tax_rate}%. {exempt_txt}",
        f"B2. {freight_txt}",
        "B3. Tax is computed once on the sum of the taxable line amounts (plus freight when freight is taxable), "
        "never on the printed subtotal, and rounded to the nearest cent; an exact half cent rounds up.",
        "",
        "C. Matching",
        "C1. Each invoice line is matched to the purchase order line with the same SKU and to the quantity received "
        "for that SKU, adding up every posted goods receipt.",
        "C2. A line is a quantity variance when its billed quantity is greater than the quantity received so far. "
        "When no goods receipt has been posted at all, a line is in exception if it bills more than the quantity "
        "ordered.",
        f"C3. {tol_txt} Price changes are agreed only by a PO amendment; an email does not change the PO price.",
        "C4. The overall match result is decided in this order: first duplicate (the invoice number already "
        "appears in the vendor's payment history, whatever the amount or date), then missing receipt (no goods "
        "receipt posted against the PO; delivery notes or emails do not count), then quantity variance, then "
        "price variance. If none applies, the result is match.",
        "C5. The invoice arithmetic is checked as printed: the subtotal must equal the sum of the line amounts and "
        "the total must equal subtotal plus freight plus the printed tax.",
        "",
        "D. Approval",
        "D1. The approval level depends on the invoice total exactly as printed on the invoice, tax and freight "
        "included:",
        bands,
        "",
        "E. Bank details",
        "E1. Pay only to the remit-to account held in the vendor master record. A request to change bank details "
        "takes effect only after the vendor master has been updated; until then an invoice showing another account "
        "does not match.",
    ]
    return "\n".join(parts)


def render_vendor(lk: Look, p: Packet, since: date) -> str:
    rows = [("Vendor ID", lk.vendor_id), ("Legal name", lk.vendor), ("Address", lk.address),
            ("Tax registration", lk.tax_id), ("Payment terms", p.terms.label()),
            ("Remit-to account", f"ACH, account ending {p.master_bank}"), ("Currency", "USD"),
            ("Vendor since", fmt(since, lk.doc_style)), ("Primary contact", lk.vendor_rep), ("Status", "Active")]
    return "VENDOR MASTER RECORD\n" + table(["Field", "Value"], [list(r) for r in rows], lk.table_style, set())


def render_po(lk: Look, p: Packet, po_date: date) -> str:
    rows = []
    for ln in sorted(p.lines, key=lambda x: x.po_line):
        cat = f"{ln.cat}" if not lk.show_cat_names else f"{ln.cat} ({CATEGORY_NAMES[ln.cat]})"
        rows.append([str(ln.po_line), ln.sku, ln.desc, cat, str(ln.ordered), money(ln.po_price),
                     money(ln.ordered * ln.po_price)])
    total = sum(ln.ordered * ln.po_price for ln in p.lines)
    head = (f"PURCHASE ORDER {lk.po_no}\nIssued by: {lk.buyer}, purchasing ({lk.buyer_rep})\n"
            f"PO date: {fmt(po_date, lk.doc_style)}\nVendor: {lk.vendor} ({lk.vendor_id})\n"
            f"Cost centre: {lk.cost_centre}\n")
    return (head + table(["PO line", "SKU", "Description", "Cat", "Qty ordered", "Unit price", "Line total"],
                         rows, lk.table_style, {0, 4, 5, 6})
            + f"\nPO value before tax and freight: {money(total)}")


def render_receipts(lk: Look, p: Packet, dates: tuple[date, ...]) -> str:
    if p.receipt_missing:
        return (f"GOODS RECEIPTS AGAINST {lk.po_no}\n(none on file: the warehouse system shows zero postings for "
                f"this order as of the packet date)")
    blocks = [f"GOODS RECEIPTS AGAINST {lk.po_no}"]
    for k, (grn, d) in enumerate(zip(lk.grn_nos, dates)):
        rows = [[ln.sku, ln.desc, str(ln.receipts[k])] for ln in p.lines if ln.receipts[k] > 0]
        blocks.append(f"Receipt {grn}, posted {fmt(d, lk.doc_style)}, received by stores\n"
                      + table(["SKU", "Description", "Qty received"], rows, lk.table_style, {2}))
    return "\n\n".join(blocks)


def render_invoice(lk: Look, p: Packet) -> str:
    D = lambda d: fmt(d, lk.doc_style)  # noqa: E731
    rows = [[str(ln.no), ln.sku, ln.desc, ln.cat, str(ln.qty), money(ln.price), money(ln.amount)] for ln in p.lines]
    rate = p.faults["wrong_rate"] or p.tax_rate
    tax_label = f"Sales tax ({rate}%)" if lk.show_rate else "Sales tax"
    head = (f"SUPPLIER INVOICE\nFrom: {lk.vendor}, {lk.address}\nBill to: {lk.buyer}, Accounts Payable\n"
            f"Invoice number: {p.invoice_no}\nInvoice date: {D(p.invoice_date)}\n"
            f"AP received stamp: {D(p.received_date)}\nCustomer PO: {lk.po_no}\n"
            f"Terms: {p.invoice_terms.label()}\nRemit to: ACH, account ending {p.invoice_bank}\n")
    foot = [f"Subtotal: {money(p.printed_subtotal)}", f"Freight: {money(p.freight)}",
            f"{tax_label}: {money(p.printed_tax)}", f"Invoice total: {money(p.printed_total)}"]
    return (head + table(["Line", "SKU", "Description", "Cat", "Qty", "Unit price", "Amount"], rows,
                         lk.table_style, {0, 4, 5, 6}) + "\n" + "\n".join(foot))


def render_ledger(lk: Look, p: Packet) -> str:
    D = lambda d: fmt(d, lk.doc_style)  # noqa: E731
    rows = [[no, D(d0), money(amt), D(paid)] for no, d0, amt, paid in p.ledger]
    return (f"PAYMENT HISTORY: {lk.vendor} ({lk.vendor_id}), invoices paid in the last 12 months\n"
            + table(["Invoice no.", "Invoice date", "Amount", "Paid on"], rows, lk.table_style, {2}))


def render_mail(lk: Look, msgs: list[tuple[str, str, date, str, str]]) -> str:
    out = ["CORRESPONDENCE (oldest first)"]
    for sender, to, d, subj, body in msgs:
        out.append(f"From: {sender}\nTo: {to}\nDate: {fmt(d, lk.mail_style)}\nSubject: {subj}\n\n{body}")
    return "\n\n---\n\n".join(out)


def build_mail(rng: random.Random, lk: Look, p: Packet, receipt_dates) -> list[tuple]:
    M = lambda d: fmt(d, lk.mail_style)  # noqa: E731
    v_addr = f"{lk.vendor_rep} <{lk.vendor_rep.split()[0].lower()}@{lk.vendor.split()[0].lower()}.example>"
    b_addr = f"{lk.buyer_rep} (purchasing, {lk.buyer})"
    ap = f"{lk.ap_clerk} (accounts payable, {lk.buyer})"
    t = p.invoice_date
    msgs = [(v_addr, ap, t, f"Invoice {p.invoice_no} for {lk.po_no}",
             f"Hello,\n\nPlease find invoice {p.invoice_no} attached for the goods and services supplied under "
             f"{lk.po_no}. It covers {len(p.lines)} lines. Let me know if anything needs clarifying.\n\n"
             f"Kind regards,\n{lk.vendor_rep}")]
    partial = [ln for ln in p.lines if not p.receipt_missing and ln.received < ln.ordered]
    if partial:
        ln = rng.choice(partial)
        msgs.append((v_addr, b_addr, t - timedelta(days=rng.randint(2, 9)), f"Back-order notice, {lk.po_no}",
                     f"Hi {lk.buyer_rep.split()[0]},\n\nA heads-up that {ln.desc} (SKU {ln.sku}) shipped short. The "
                     f"balance is on back-order with our supplier and will follow in a separate delivery.\n\n"
                     f"{lk.vendor_rep}"))
    if p.receipt_missing:
        d = t - timedelta(days=rng.randint(1, 6))
        msgs.append((v_addr, ap, t + timedelta(days=rng.randint(1, 5)), f"RE: Invoice {p.invoice_no}",
                     f"Our carrier's proof of delivery shows the full order arrived at your dock on {M(d)}, signed "
                     f"for by your dock staff. Could you confirm the invoice is approved for payment?\n\n"
                     f"{lk.vendor_rep}"))
    if p.faults["line"] == "price" or rng.random() < 0.25:
        msgs.append((v_addr, b_addr, t - timedelta(days=rng.randint(10, 40)), "Price list update",
                     f"Dear customer,\n\nOur raw material costs have risen and an updated price list takes effect "
                     f"from {M(t - timedelta(days=rng.randint(1, 9)))}. Revised prices will be applied to new "
                     f"invoices from that date.\n\nSales administration, {lk.vendor}"))
    if p.faults["bank"] or rng.random() < 0.15:
        new = p.invoice_bank if p.faults["bank"] else f"{rng.randint(0, 9999):04d}"
        msgs.append((v_addr, ap, t - timedelta(days=rng.randint(1, 20)), "Change of bank details",
                     f"Hello,\n\nPlease note we have moved our banking. Kindly update your records so that future "
                     f"payments go to the account ending {new}. A signed letter on our letterhead will follow.\n\n"
                     f"Finance team, {lk.vendor}"))
    if rng.random() < 0.6:
        closed = t + timedelta(days=rng.randint(3, 50))
        msgs.append((v_addr, ap, t - timedelta(days=rng.randint(0, 15)), "Office closure",
                     f"Our offices and warehouse will be closed on {M(closed)} for a staff training day. Orders "
                     f"placed that day will be processed the following business day.\n\n{lk.vendor}"))
    if p.ledger and rng.random() < 0.6:
        no, d0, amt, paid = rng.choice(p.ledger)
        msgs.append((v_addr, ap, t - timedelta(days=rng.randint(5, 30)), f"Remittance query {no}",
                     f"Hi,\n\nCould you send the remittance advice for invoice {no} ({money(amt)})? Our ledger "
                     f"shows it as settled on {M(paid)} but the reference was missing.\n\n{lk.vendor_rep}"))
        msgs.append((ap, v_addr, t - timedelta(days=rng.randint(0, 4)), f"RE: Remittance query {no}",
                     f"Hello {lk.vendor_rep.split()[0]},\n\nRemittance attached for {no}. For future invoices "
                     f"please always quote the PO number.\n\nThanks,\n{lk.ap_clerk}"))
    if rng.random() < 0.5:
        other = rng.choice(PEOPLE)
        msgs.append((b_addr, ap, t + timedelta(days=rng.randint(0, 6)), f"Delivery for {lk.po_no}",
                     f"{lk.ap_clerk.split()[0]},\n\nFYI {other} has taken over as site contact for this order while "
                     f"I am on leave. Please copy them on any queries about the delivery.\n\n{lk.buyer_rep}"))
    if rng.random() < 0.5:
        msgs.append((v_addr, ap, p.received_date + timedelta(days=rng.randint(1, 7)),
                     f"Payment status, {p.invoice_no}",
                     f"Good morning,\n\nJust checking that invoice {p.invoice_no} has been received and is in your "
                     f"payment cycle. Our credit control team asked me to follow up.\n\nBest,\n{lk.vendor_rep}"))
        msgs.append((ap, v_addr, p.received_date + timedelta(days=rng.randint(8, 10)),
                     f"RE: Payment status, {p.invoice_no}",
                     f"Hi {lk.vendor_rep.split()[0]},\n\nConfirmed received. It is going through our usual review "
                     f"and you will get a remittance advice when it is paid.\n\n{lk.ap_clerk}"))
    msgs.sort(key=lambda m: m[2])
    return msgs


def render_slip(lk: Look, p: Packet, prepared: date) -> str:
    D = lambda d: fmt(d, lk.doc_style)  # noqa: E731
    disc = ""
    if p.discount_claimed:
        cut = round_half_up(Fraction(p.printed_total * p.terms.disc_pct, 100))
        disc = (f"Early-payment discount: claimed, {p.terms.disc_pct}% = {money(cut)}; amount to release "
                f"{money(p.printed_total - cut)}\n")
    elif p.terms.kind == "discount":
        disc = f"Early-payment discount: not claimed; amount to release {money(p.printed_total)}\n"
    return (f"ROUTING SLIP\nPacket: {lk.packet_no}\nPrepared by: {lk.ap_clerk}, {D(prepared)}\n"
            f"Scheduled payment run for this invoice: {D(p.pay_date)}\n{disc}Cost centre: {lk.cost_centre}\n"
            f"Reviewer notes: complete the checks below before release.")


def render_packet(rng: random.Random, lk: Look, p: Packet) -> str:
    po_date = p.invoice_date - timedelta(days=rng.randint(20, 75))
    rdates: tuple[date, ...] = ()
    if not p.receipt_missing:
        n_rec = len(p.lines[0].receipts)
        r1 = po_date + timedelta(days=rng.randint(5, 15))
        rdates = (r1,) if n_rec == 1 else (r1, r1 + timedelta(days=rng.randint(3, 12)))
    since = p.invoice_date - timedelta(days=rng.randint(400, 3000))
    prepared = p.received_date + timedelta(days=rng.randint(0, 2))
    sections = [
        f"ACCOUNTS PAYABLE REVIEW PACKET: {lk.buyer}\nPacket {lk.packet_no}, vendor {lk.vendor}",
        render_vendor(lk, p, since),
        render_po(lk, p, po_date),
        render_receipts(lk, p, rdates),
        render_invoice(lk, p),
        render_ledger(lk, p),
        render_mail(lk, build_mail(rng, lk, p, rdates)),
        render_slip(lk, p, prepared),
    ]
    policy = render_policy(lk, p)
    if lk.policy_last:
        sections.append(policy)
    else:
        sections.insert(1, policy)
    text = "\n\n==========\n\n".join(sections)
    if "slash_dmy" in (lk.doc_style, lk.mail_style, lk.option_style):
        text += f"\n\n{SLASH_NOTE}"
    return text


def sample_look(rng: random.Random, p: Packet) -> Look:
    vendor = rng.choice(VENDORS)
    people = rng.sample(PEOPLE, 3)
    n_rec = 0 if p.receipt_missing else len(p.lines[0].receipts)
    return Look(
        vendor=vendor, vendor_id=f"V{rng.randint(10000, 99999)}", buyer=rng.choice(BUYERS),
        po_no=f"PO-{rng.randint(400000, 899999)}", packet_no=f"APR-{rng.randint(1000, 99999)}",
        grn_nos=tuple(f"GRN-{rng.randint(100000, 999999)}" for _ in range(n_rec)),
        doc_style=rng.choices(STYLES, (4, 2, 2, 1))[0], mail_style=rng.choices(STYLES, (2, 3, 3, 1))[0],
        table_style=rng.choice(("markdown", "fixed")), vendor_rep=people[0], buyer_rep=people[1], ap_clerk=people[2],
        address=f"{rng.randint(2, 480)} {rng.choice(STREETS)}, {rng.choice(TOWNS)}",
        tax_id=f"TR-{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
        cost_centre=f"CC-{rng.randint(100, 999)}", policy_variant=rng.randrange(2),
        policy_last=rng.random() < 0.3, show_rate=rng.random() < 0.6, show_cat_names=rng.random() < 0.3,
        option_style=rng.choices(STYLES, (4, 2, 2, 1))[0],
    )


# ---------------------------------------------------------------- questions


def packet_meta(p: Packet, approx_tokens: int) -> dict:
    return {
        "approx_tokens": approx_tokens,
        "lines": [ln.meta() for ln in p.lines], "receipt_missing": p.receipt_missing,
        "tolerance_pct": p.tolerance_pct, "tax_rate": p.tax_rate, "exempt_cats": list(p.exempt_cats),
        "freight": p.freight, "freight_taxable": p.freight_taxable,
        "printed_subtotal": p.printed_subtotal, "printed_tax": p.printed_tax, "printed_total": p.printed_total,
        "invoice_no": p.invoice_no, "ledger_invoice_nos": [x[0] for x in p.ledger],
        "terms": p.terms.meta(), "invoice_terms": p.invoice_terms.meta(), "anchor": p.anchor,
        "invoice_date": p.invoice_date.isoformat(), "received_date": p.received_date.isoformat(),
        "holidays": [h.isoformat() for h in p.holidays], "pay_date": p.pay_date.isoformat(), "discount_claimed": p.discount_claimed,
        "approval_cuts": list(p.approval_cuts), "master_bank": p.master_bank, "invoice_bank": p.invoice_bank,
        "faults": dict(p.faults),
    }


def _pick_options(gold, distractors, k: int, rng: random.Random) -> list:
    """Gold plus k-1 distractors (taken in priority order), with gold at a random rank among the sorted values."""
    below, above = [], []
    for x in distractors:
        side = below if x < gold else above if x > gold else None
        if side is not None and x not in side:
            side.append(x)
    lo, hi = max(0, k - 1 - len(above)), min(k - 1, len(below))
    pos = rng.randint(lo, hi)
    return sorted([gold] + below[:pos] + above[:k - 1 - pos])


def q_noul(question: str, gold: bool):
    return "noul", question, NOUL_OPTIONS, "yes" if gold else "no", {}


def q_due_date(rng: random.Random, p: Packet, style: str):
    a = anchor_date(p)
    other = p.received_date if p.anchor == "invoice" else p.invoice_date
    gold = due_date(p.terms, a, p.holidays)
    raw = unrolled_due(p.terms, a)
    prev = raw
    while not is_business_day(prev, p.holidays):
        prev -= timedelta(days=1)
    distractors = [raw, prev, due_date(p.terms, other, p.holidays)]
    if p.terms.kind == "eom":
        distractors.append(roll_forward(a + timedelta(days=p.terms.net), p.holidays))
    if p.terms.kind == "discount":
        distractors.append(discount_deadline(p.terms, a))
    if p.invoice_terms != p.terms:
        distractors.append(due_date(p.invoice_terms, a, p.holidays))
    distractors += [gold + timedelta(days=s * n) for n in (1, 2, 3, 4, 7) for s in (1, -1)]
    dates = _pick_options(gold, distractors, rng.randint(4, 6), rng)
    opts = [Option(f"date_{i + 1}", fmt(d, style)) for i, d in enumerate(dates)]
    question = rng.choice(("On what date is the full (net) payment of this invoice due?",
                           "Under the AP rules in the packet, which date is the invoice's net due date?"))
    return "choice", question, opts, f"date_{dates.index(gold) + 1}", {"option_style": style}


def q_tax_amount(rng: random.Random, p: Packet):
    gold = correct_tax(p)
    base = tax_base(p)
    distractors = [p.printed_tax, tax_at(tax_base(p, exempt=()), p.tax_rate),
                   tax_at(tax_base(p, freight_taxable=not p.freight_taxable), p.tax_rate),
                   int(base * Fraction(p.tax_rate) / 100),
                   tax_at(base, rng.choice([t for t in TAX_RATES if t != p.tax_rate])),
                   *(gold + s * n for n in (1, 100, 1000, 10, 5000) for s in (1, -1))]
    vals = _pick_options(gold, [x for x in distractors if x >= 0], rng.randint(4, 6), rng)
    opts = [Option(f"amount_{i + 1}", money(v)) for i, v in enumerate(vals)]
    question = rng.choice(("What should the sales tax on this invoice be under the company's tax rules?",
                           "Which sales tax amount is correct for this invoice?"))
    return "choice", question, opts, f"amount_{vals.index(gold) + 1}", {}


def q_received_qty(rng: random.Random, p: Packet):
    multi = [ln for ln in p.lines if sum(1 for r in ln.receipts if r) > 1 or ln.received < ln.ordered]
    ln = rng.choice(multi) if multi and rng.random() < 0.7 else rng.choice(p.lines)
    gold = ln.received
    distractors = [ln.ordered, ln.qty] + [r for r in ln.receipts if r] + [gold + s * n for n in (1, 2, 3, 5, 10) for s in (1, -1)]
    vals = _pick_options(gold, [x for x in distractors if x >= 0], rng.randint(4, 6), rng)
    unit = lambda n: f"{n} unit{'s' if n != 1 else ''}"  # noqa: E731
    opts = [Option(f"qty_{i + 1}", unit(v)) for i, v in enumerate(vals)]
    question = (f"How many units of SKU {ln.sku} ({ln.desc}) have been received against the purchase order, "
                f"counting every posted goods receipt?")
    return "choice", question, opts, f"qty_{vals.index(gold) + 1}", {"target_sku": ln.sku}


def q_wrong_line(rng: random.Random, p: Packet):
    bad = wrong_lines(p)
    assert len(bad) <= 1, bad
    m = rng.randint(4, min(9, len(p.lines)))
    tricky = [ln.no for ln in p.lines if ln.price != ln.po_price or ln.received < ln.ordered
              or sum(1 for r in ln.receipts if r) > 1]
    chosen = list(bad)
    for no in rng.sample(tricky, len(tricky)) + rng.sample([ln.no for ln in p.lines], len(p.lines)):
        if len(chosen) >= m:
            break
        if no not in chosen:
            chosen.append(no)
    chosen.sort()
    by_no = {ln.no: ln for ln in p.lines}
    opts = [Option(f"line_{no}", f"Invoice line {no}: {by_no[no].sku}, {by_no[no].desc}") for no in chosen]
    opts.append(Option("none", "None of these lines is in exception."))
    question = rng.choice(("Which invoice line, if any, is in exception under matching rules C2 and C3 (quantity "
                           "or unit price)?",
                           "Under rules C1-C3, which of these invoice lines is in exception, if any?"))
    return "choice", question, opts, f"line_{bad[0]}" if bad else "none", {}


def q_three_way(rng: random.Random, p: Packet):
    opts = [Option("match", "Match: no exception; the invoice can proceed."),
            Option("price_variance", "Price variance: a billed unit price is above the PO price beyond tolerance."),
            Option("quantity_variance", "Quantity variance: a line bills more than has been received."),
            Option("missing_receipt", "Missing receipt: no goods receipt has been posted against the PO."),
            Option("duplicate", "Duplicate: this invoice number has already been paid.")]
    question = rng.choice(("What is the overall result of matching this invoice against the PO, the goods "
                           "receipts and the payment history?",
                           "Applying rule C4, what is the overall match result for this invoice?"))
    return "choice", question, opts, three_way(p), {}


def q_approval(rng: random.Random, p: Packet):
    lo = [money(c) for c in p.approval_cuts]
    opts = [Option("0", f"Level 0 ({APPROVERS[0]}): total below {lo[0]}"),
            Option("1", f"Level 1 ({APPROVERS[1]}): {lo[0]} up to, but not including, {lo[1]}"),
            Option("2", f"Level 2 ({APPROVERS[2]}): {lo[1]} up to, but not including, {lo[2]}"),
            Option("3", f"Level 3 ({APPROVERS[3]}): {lo[2]} or more")]
    question = rng.choice(("Which approval level does this invoice need?",
                           "Under section D, what approval level applies to this invoice?"))
    return "score", question, opts, str(approval_level(p.printed_total, p.approval_cuts)), {}


def build_question(rng: random.Random, kind: str, p: Packet, lk: Look):
    if kind == "totals_match":
        return q_noul(rng.choice(("Is the invoice arithmetic correct as printed (subtotal equals the sum of the "
                                  "line amounts, and total equals subtotal plus freight plus printed tax)?",
                                  "Do the invoice's printed subtotal and total add up under rule C5?")), totals_ok(p))
    if kind == "tax_correct":
        return q_noul(rng.choice(("Is the sales tax amount printed on the invoice correct?",
                                  "Has the vendor charged the right amount of sales tax under section B?")),
                      p.printed_tax == correct_tax(p))
    if kind == "discount":
        return q_noul(rng.choice(("If the invoice is paid on the scheduled payment run date, may the early-payment "
                                  "discount be taken?",
                                  "Can an early-payment discount be deducted when this invoice is paid on the "
                                  "date in the routing slip?")), discount_ok(p))
    if kind == "bank_match":
        return q_noul(rng.choice(("Does the remit-to account on the invoice match the vendor master record?",
                                  "Under rule E1, may payment go to the bank account printed on the invoice?")),
                      p.invoice_bank == p.master_bank)
    if kind == "due_date":
        return q_due_date(rng, p, lk.option_style)
    return {"tax_amount": q_tax_amount, "received_qty": q_received_qty,
            "wrong_line": q_wrong_line, "three_way": q_three_way, "approval": q_approval}[kind](rng, p)


def generate_document(rng: random.Random, cluster_id: str) -> list[Decision]:
    p = sample_packet(rng)
    lk = sample_look(rng, p)
    state = render_packet(rng, lk, p)
    base_meta = packet_meta(p, round(len(state) / 4))
    # No receipt -> nothing to count; a duplicate invoice is never paid, so the discount question would be moot.
    kinds = [k for k in KINDS if not (k == "received_qty" and p.receipt_missing)
             and not (k == "discount" and p.faults["duplicate"])]
    chosen = rng.sample(kinds, rng.randint(6, min(10, len(kinds))))
    out = []
    for j, kind in enumerate(chosen):
        qtype, question, opts, gold, extra = build_question(rng, kind, p, lk)
        out.append(Decision(id=f"{cluster_id}:{j}", type=qtype, state=state, question=question, options=list(opts),
                            gold=gold, family=FAMILY, source=SOURCE, licence=SYNTH_LICENCE, cluster_id=cluster_id,
                            edit_type=None, metadata={"kind": kind, **base_meta, **extra}))
    return out


def generate(n_docs: int, seed: int) -> list[Decision]:
    rng = random.Random(f"invoice_ledger:{seed}")
    out: list[Decision] = []
    for k in range(n_docs):
        out.extend(generate_document(rng, f"invoices:{seed}:{k}"))
    return out
