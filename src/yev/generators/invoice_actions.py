"""Accounts-payable rule families (Plan 6): payment_terms and invoice_action.

payment_terms (choice): a short invoice header whose payment-terms string comes from our own vocabulary (net N,
due on receipt, end of month, N days from receipt, "2/10 net 30"-style strings, a percentage off if paid within
D days, late-fee clauses). The question asks which early-payment discount window the terms give: 7, 10, 15, 20
or 30 days, or none. The window is written in the string; nothing needs computing. 70% of rows are none, and
every plain net-N and due-on-receipt string is none. Late fees and plain due dates are never a discount.

invoice_action (choice): a reviewer summary of already-answered facts plus a block of precomputed checks, under
one of three AP policies (small company, enterprise, high-volume retailer) that print their price tolerance,
auto-pay limit, approval limit and numbered routing rules. The gold is the first routing rule that matches,
in the fixed priority order fraud > duplicate > bank > owner-declined/unauthorised > scope > completion >
price variance (billing error, then PO variance) > approval band > clean; the policy decides which action some
tiers map to, so the same flags give different golds across policies. Every computed flag is printed with its
numbers, so the reader compares nothing. A row lists 10 of the 16 actions, always including the gold.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from yev.generators.common import make_cluster
from yev.generators.llm.targets.agent_trace import person
from yev.schema import Decision, Option

SOURCE = "synthetic_rules"
PAYMENT_TERMS_FAMILY = "payment_terms"
INVOICE_ACTION_FAMILY = "invoice_action"

VENDORS = ("Halberd Office Supply", "Glenmoor Catering", "Pinecrest IT Services", "Sorrel Print Works",
           "Ironbridge Couriers", "Bluefen Cleaning Co.", "Ashcombe Consulting", "Redwing Facilities",
           "Larchfield Packaging", "Moonstone Media", "Kestrel Hardware", "Willowmere Staffing")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS)} 2026"


def _money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


# ------------------------------------------------------------------------------------------------ payment_terms

DISCOUNT_DAYS = (7, 10, 15, 20, 30)
TERMS_KEYS = tuple(f"within_{d}_days" for d in DISCOUNT_DAYS) + ("no_discount",)
TERMS_OPTIONS = tuple(
    [Option(f"within_{d}_days", f"{d} days: paying within {d} days earns the early-payment discount.")
     for d in DISCOUNT_DAYS]
    + [Option("no_discount", "No early-payment discount: the terms give no reduction for paying early.")])
TERMS_QUESTIONS = (
    "Within how many days must this invoice be paid to earn an early-payment discount, if the terms offer one?",
    "What early-payment discount window do these payment terms give, if any?",
    "Reading only the payment terms, how long does the buyer have to take a discount for paying early, if any?",
)
NET_DAYS = (7, 10, 14, 15, 20, 30, 45, 60, 90)
RATES = ("1", "1.5", "2", "3")


def _none_terms(rng: random.Random, kind: str, net: int) -> str:
    lf = rng.choice(("1", "1.5", "2"))
    if kind == "net":
        return rng.choice((f"Net {net}", f"Net {net}", f"Net {net} days", f"net {net}"))
    if kind == "receipt":
        return rng.choice(("Due on receipt", "Payable on receipt", "Due upon receipt of invoice"))
    if kind == "eom":
        return rng.choice(("End of month", "Due at the end of the month", "Net EOM",
                           "Payment due by the last day of the invoice month"))
    if kind == "from_receipt":
        return rng.choice((f"{net} days from receipt of invoice", f"Payable {net} days after the invoice is received"))
    if kind == "late_fee":
        late = rng.choice((7, 10, 15, 20, 30))
        return rng.choice((f"Net {net}; a {lf}% late fee applies to balances unpaid after {net} days",
                           f"Net {net}. Overdue balances are charged {lf}% per month",
                           f"Due in {net} days; payments more than {late} days late incur a "
                           f"${rng.choice((25, 40, 50))} fee"))
    # explicit_none
    return rng.choice((f"Net {net}. No discount is offered for early payment.",
                       f"Due within {net} days; early-payment discounts are not available."))


def _discount_terms(rng: random.Random, days: int, net: int) -> str:
    p = rng.choice(RATES)
    base = rng.choice((
        f"{p}/{days} net {net}", f"{p}/{days}, n/{net}", f"{p}% {days} days, net {net}",
        f"{p}% discount if paid within {days} days; otherwise the full amount is due in {net} days",
        f"Net {net}. Take {p}% off when payment is made within {days} days of the invoice date",
        f"Pay within {days} days and deduct {p}%; balance due net {net}",
    ))
    if rng.random() < 0.25:
        base += f"; a {rng.choice(('1', '1.5', '2'))}% late fee applies after {net} days"
    return base


NONE_KINDS = ("net", "receipt", "eom", "from_receipt", "late_fee", "explicit_none")


def _terms_state(ctx: dict, terms: str) -> str:
    return (f"Invoice {ctx['inv']} from {ctx['vendor']}\n"
            f"Invoice date: {ctx['date']}\n"
            f"Description: {ctx['desc']}\n"
            f"Amount due: {ctx['amount']}\n"
            f"Payment terms: {terms}\n"
            f"Remit to: the bank account held on the vendor record.")


TERMS_DESCS = ("monthly office cleaning", "printer toner and paper", "catering for the spring offsite",
               "courier deliveries for August", "laptop repairs", "consulting hours for the data migration",
               "packaging materials", "window repairs at the depot")


def generate_terms_cluster(rng: random.Random, cluster_id: str) -> list[Decision]:
    ctx = {"inv": f"INV-{rng.randint(10000, 99999)}", "vendor": rng.choice(VENDORS), "date": _date(rng),
           "desc": rng.choice(TERMS_DESCS), "amount": _money(rng.randint(5000, 2_500_000))}
    question = TERMS_QUESTIONS[0] if rng.random() < 0.5 else rng.choice(TERMS_QUESTIONS[1:])
    shuffle = rng.random() < 0.3
    options = list(TERMS_OPTIONS)
    if shuffle:
        rng.shuffle(options)
    rows: list[tuple[str, str, int | None, str | None]] = []  # terms, kind, days, edit_type
    if rng.random() < 0.9:
        days = rng.choice(DISCOUNT_DAYS)
        net = rng.choice([n for n in (30, 45, 60) if n > days])
        rows.append((_discount_terms(rng, days, net), "discount", days, None))
        rows.append((rng.choice((f"Net {net}", f"net {net}")), "net", None, "exception"))
        kind = rng.choice([k for k in NONE_KINDS if k != "net"])
        rows.append((_none_terms(rng, kind, rng.choice(NET_DAYS)), kind, None, "exception"))
    else:
        for i, kind in enumerate(rng.sample(NONE_KINDS, 3)):
            rows.append((_none_terms(rng, kind, rng.choice(NET_DAYS)), kind, None, None if i == 0 else "entity_swap"))
    members = [(_terms_state(ctx, t), options, "no_discount" if d is None else f"within_{d}_days", e)
               for t, _, d, e in rows]
    out = make_cluster(cluster_id=cluster_id, family=PAYMENT_TERMS_FAMILY, source=SOURCE, qtype="choice",
                       question=question, members=members)
    for dec, (t, kind, d, _) in zip(out, rows):
        dec.metadata = {"terms_kind": kind, "discount_days": d, "terms": t, "shuffled_options": shuffle}
    return out


def generate_payment_terms(n_clusters: int, seed: int) -> list[Decision]:
    rng = random.Random(f"{PAYMENT_TERMS_FAMILY}:{seed}")
    out: list[Decision] = []
    for k in range(n_clusters):
        out.extend(generate_terms_cluster(rng, f"{PAYMENT_TERMS_FAMILY}:{seed}:{k}"))
    return out


# ----------------------------------------------------------------------------------------------- invoice_action

ACTION_OPTIONS: tuple[Option, ...] = (
    Option("pay", "Pay the invoice in full now."),
    Option("schedule_payment", "Queue the full amount for the next scheduled payment run."),
    Option("short_pay", "Pay only the part that is in order and hold back the rest."),
    Option("dispute_lines", "Raise a dispute with the vendor over particular lines."),
    Option("request_corrected_invoice", "Ask the vendor to reissue the invoice with correct figures."),
    Option("hold_request_documents", "Put the invoice on hold and ask for the missing delivery records."),
    Option("procurement_review", "Send the invoice to procurement to check against the purchase order."),
    Option("project_owner_review", "Send the invoice to the project owner to look over."),
    Option("duplicate_review", "Hold the invoice while a reviewer checks whether it repeats an earlier one."),
    Option("reject_as_duplicate", "Reject the invoice as a copy of one already paid."),
    Option("reject_as_unauthorized", "Reject the invoice because the purchase was never authorised."),
    Option("fraud_review", "Freeze the invoice and refer it to the fraud team."),
    Option("verify_vendor_out_of_band", "Phone the vendor on the number held on file to confirm the request."),
    Option("hold_verify_bank_details", "Hold payment until the new bank details are confirmed with the vendor."),
    Option("approval_escalation", "Escalate the invoice to the finance director for sign-off."),
    Option("human_approval", "Send the invoice to a manager for sign-off."),
)
ACTIONS: tuple[str, ...] = tuple(o.key for o in ACTION_OPTIONS)
N_OPTIONS = 10
ACTION_QUESTIONS = (
    "Following this company's routing rules, what is the primary action for this invoice?",
    "Which single action should accounts payable take on this invoice, going by the routing rules above?",
    "Under the routing rules in this policy, which primary action does this invoice get?",
)


@dataclass(frozen=True)
class APPolicy:
    name: str
    label: str
    blurb: str
    tolerance_pct: int  # an invoice total up to this far above its PO is accepted
    auto_pay_limit: int  # dollars; at or under needs no sign-off
    approval_limit: int  # dollars; above the auto-pay limit and at or under this needs a manager
    bank: str
    scope: str
    completion: str
    price: str
    clean: str  # pay | schedule_payment | discount (pay when an early-payment discount is available)


VARIANTS: dict[str, APPolicy] = {
    "small_company": APPolicy(
        "small_company", "Small company AP policy", "a fourteen-person design studio that pays most bills by hand",
        2, 2_500, 10_000, bank="hold_verify_bank_details", scope="dispute_lines",
        completion="hold_request_documents", price="project_owner_review", clean="pay"),
    "enterprise": APPolicy(
        "enterprise", "Enterprise AP policy", "a multinational group with a central payables team",
        5, 25_000, 250_000, bank="verify_vendor_out_of_band", scope="procurement_review",
        completion="project_owner_review", price="procurement_review", clean="schedule_payment"),
    "high_volume_retailer": APPolicy(
        "high_volume_retailer", "High-volume retailer AP policy", "a chain of 300 stores paying thousands of "
        "supplier invoices a week", 3, 50_000, 500_000, bank="hold_verify_bank_details", scope="short_pay",
        completion="hold_request_documents", price="short_pay", clean="discount"),
}


@dataclass(frozen=True)
class InvoiceCase:
    impostor: bool = False
    duplicate: str = "none"  # none | possible | exact
    bank_change: bool = False
    owner_declined: bool = False
    po_approved: bool = True
    extra_lines: tuple[int, ...] = ()
    missing_lines: tuple[int, ...] = ()
    totals_match: bool = True
    variance_pct10: int = 0  # invoice total vs PO, in tenths of a percent
    po_cents: int = 180_000  # a multiple of 1,000 cents, so the variance is a whole number of cents
    discount_available: bool = False
    n_lines: int = 6


CLEAN_CASE = InvoiceCase()


def total_cents(c: InvoiceCase) -> int:
    return c.po_cents + c.po_cents * c.variance_pct10 // 1000


def variance_flag(p: APPolicy, c: InvoiceCase) -> bool:
    return c.variance_pct10 > p.tolerance_pct * 10


def band(p: APPolicy, c: InvoiceCase) -> str:
    t = total_cents(c)
    if t <= p.auto_pay_limit * 100:
        return "auto_pay"
    return "approval" if t <= p.approval_limit * 100 else "escalation"


def decide(variant: str, c: InvoiceCase) -> str:
    """The priority table: the first matching tier decides the one primary action."""
    p = VARIANTS[variant]
    if c.impostor:
        return "fraud_review"
    if c.duplicate == "exact":
        return "reject_as_duplicate"
    if c.duplicate == "possible":
        return "duplicate_review"
    if c.bank_change:
        return p.bank
    if c.owner_declined or not c.po_approved:
        return "reject_as_unauthorized"
    if c.extra_lines:
        return p.scope
    if c.missing_lines:
        return p.completion
    if not c.totals_match:
        return "request_corrected_invoice"
    if variance_flag(p, c):
        return p.price
    b = band(p, c)
    if b == "escalation":
        return "approval_escalation"
    if b == "approval":
        return "human_approval"
    if p.clean == "discount":
        return "pay" if c.discount_available else "schedule_payment"
    return p.clean


def routing_rules(p: APPolicy) -> list[str]:
    clean = ("pay if early_payment_discount is available, otherwise schedule_payment" if p.clean == "discount"
             else p.clean)
    return [
        "If the sender is an impostor: fraud_review.",
        "If duplicate_flag is exact: reject_as_duplicate.",
        "If duplicate_flag is possible: duplicate_review.",
        f"If bank_change_flag is yes: {p.bank}.",
        "If owner_declined is yes, or po_approved is no: reject_as_unauthorized.",
        f"If unapproved_extra_lines lists any line: {p.scope}.",
        f"If missing_records_lines lists any line: {p.completion}.",
        "If totals_match is no: request_corrected_invoice.",
        f"If po_variance_flag is yes: {p.price}.",
        "If approval_band is escalation: approval_escalation.",
        "If approval_band is approval: human_approval.",
        f"Otherwise: {clean}.",
    ]


CATEGORIES: dict[str, tuple[str, ...]] = {
    "office supplies": ("printer paper, 40 boxes", "toner cartridges", "desk chairs", "whiteboard markers",
                        "filing cabinets", "monitor stands", "label printer", "storage crates"),
    "catering": ("lunch for the board meeting", "coffee service, weekly", "staff party buffet",
                 "fruit deliveries", "water cooler refills", "breakfast for the training day", "cake order",
                 "event crockery hire"),
    "IT services": ("laptop repairs", "network switch install", "helpdesk hours", "backup licence renewal",
                    "cabling for the new office", "firewall configuration", "server rack rental",
                    "security patching"),
    "courier": ("same-day deliveries", "pallet collection", "international parcels", "weekend collection",
                "fuel surcharge", "express courier surcharge", "returns pickup", "document courier"),
    "facilities": ("window cleaning", "boiler service", "lighting repairs", "carpet cleaning",
                   "pest control visit", "lift inspection", "roof gutter clearing", "lock replacement"),
}


def _lines_text(lines: list[str], picks: tuple[int, ...], why: str) -> str:
    if not picks:
        return "none"
    return ", ".join(f"line {i} ({lines[i - 1]}, {why})" for i in picks)


def _ctx(rng: random.Random) -> dict:
    vendor = rng.choice(VENDORS)
    cat = rng.choice(sorted(CATEGORIES))
    domain = vendor.lower().split()[0] + ".com"
    contact = person(rng)
    return {
        "vendor": vendor, "category": cat, "lines": rng.sample(CATEGORIES[cat], len(CATEGORIES[cat])),
        "inv": f"INV-{rng.randint(10000, 99999)}", "prior_inv": f"INV-{rng.randint(10000, 99999)}",
        "po": f"PO-{rng.randint(1000, 9999)}", "domain": domain, "contact": contact,
        "email": f"{contact.split()[0].lower()}@{domain}",
        "fake_domain": domain.replace(".com", rng.choice(("-billing.net", "-payments.co", "-invoices.org"))),
        "owner": person(rng), "approver": person(rng), "date": _date(rng), "paid_date": _date(rng),
        "acct_old": (old := rng.randint(1000, 9999)), "acct_new": rng.choice([n for n in range(1000, 10000) if n != old]),
        "diff_cents": rng.randint(100, 50_000) * rng.choice((1, -1)), "disc": rng.choice(("2/10 net 30", "1/15 net 45")),
    }


def render_state(variant: str, c: InvoiceCase, rng: random.Random) -> str:
    p = VARIANTS[variant]
    x = _ctx(rng)
    lines = x["lines"][:c.n_lines]
    t, po = total_cents(c), c.po_cents
    delta = t - po
    lines_sum = t + x["diff_cents"] if t + x["diff_cents"] > 0 else t - x["diff_cents"]  # never equal to t
    sign = "+" if delta >= 0 else "-"
    pct = f"{sign}{abs(c.variance_pct10) // 10}.{abs(c.variance_pct10) % 10}"
    rules = routing_rules(p)
    out = [
        f"Accounts payable review: invoice {x['inv']} from {x['vendor']} ({x['category']}), dated {x['date']}, "
        f"against purchase order {x['po']}.",
        "",
        f"{p.label} ({p.blurb}):",
        f"- Price tolerance: an invoice total up to {p.tolerance_pct}% above its PO is accepted (exactly "
        f"{p.tolerance_pct}% is fine); more than {p.tolerance_pct}% above is a price variance.",
        f"- Auto-pay limit: ${p.auto_pay_limit:,}. An invoice total at or under it needs no sign-off.",
        f"- Approval limit: ${p.approval_limit:,}. Totals above ${p.auto_pay_limit:,} and at or under "
        f"${p.approval_limit:,} need a manager's sign-off; totals above ${p.approval_limit:,} go to the finance "
        f"director.",
        "- Routing rules: go down the list and apply the first rule that matches; it names the primary action.",
        *[f"  {i}. {r}" for i, r in enumerate(rules, 1)],
        "",
        f"Invoice lines ({len(lines)}): " + "; ".join(f"{i}. {d}" for i, d in enumerate(lines, 1)) + ".",
        "",
        "Reviewer summary (questions already answered):",
        (f"- sender: impostor ({x['contact']} <billing@{x['fake_domain']}>; the email domain does not match the "
         f"vendor's {x['domain']})" if c.impostor else
         f"- sender: known_contact ({x['contact']} <{x['email']}>, listed on the vendor record)"),
        (f"- Project owner's comment: {x['owner']} withholds approval and asks that the invoice not be paid."
         if c.owner_declined else
         # Approving "as billed" would contradict unapproved extra lines or an unapproved PO.
         f"- Project owner's comment: {x['owner']} has not commented on the invoice yet." if not c.po_approved else
         f"- Project owner's comment: {x['owner']} confirms the work listed on the PO was done." if c.extra_lines
         else f"- Project owner's comment: {x['owner']} approves the invoice as billed."),
        {"none": "- Earlier invoices: nothing else from this vendor for the same work.",
         "possible": f"- Earlier invoices: {x['prior_inv']} from the same vendor is for the same amount under a "
                     f"different invoice number; whether it is the same work is still open.",
         "exact": f"- Earlier invoices: {x['inv']} itself, same number and same amount, was already paid on "
                  f"{x['paid_date']}."}[c.duplicate],
        "",
        "Checks (computed by the AP system):",
        {"none": "duplicate_flag: no",
         "possible": f"duplicate_flag: possible ({x['prior_inv']}: same vendor and amount, different number)",
         "exact": f"duplicate_flag: exact (same number and amount as a paid invoice, paid {x['paid_date']})"}[
            c.duplicate],
        (f"bank_change_flag: yes (the remit-to account details differ from the vendor record: ending "
         f"{x['acct_new']} on the invoice, {x['acct_old']} on file; not yet confirmed)" if c.bank_change else
         "bank_change_flag: no (the remit-to account is the one on the vendor record)"),
        f"owner_declined: {'yes' if c.owner_declined else 'no'}",
        (f"po_approved: yes ({x['po']} approved by {x['approver']})" if c.po_approved else
         f"po_approved: no ({x['po']} was raised but never approved)"),
        f"unapproved_extra_lines: {_lines_text(lines, c.extra_lines, 'not on the PO and not approved')}",
        f"missing_records_lines: {_lines_text(lines, c.missing_lines, 'no delivery record')}",
        ("totals_match: yes (the line amounts add up to the invoice total)" if c.totals_match else
         f"totals_match: no (the line amounts add up to {_money(lines_sum)}; the invoice total says "
         f"{_money(t)})"),
        (f"po_variance_flag: {'yes' if variance_flag(p, c) else 'no'} (invoice total {_money(t)} vs PO "
         f"{_money(po)}: {sign}{_money(abs(delta))}, {pct}% against the PO; this policy accepts up to "
         f"{p.tolerance_pct}% above the PO)"),
        (f"early_payment_discount: available ({x['disc']}, still inside the window)" if c.discount_available
         else "early_payment_discount: none"),
        {"auto_pay": f"approval_band: auto_pay (total {_money(t)} is at or under the ${p.auto_pay_limit:,} "
                     f"auto-pay limit)",
         "approval": f"approval_band: approval (total {_money(t)} is above the ${p.auto_pay_limit:,} auto-pay "
                     f"limit and at or under the ${p.approval_limit:,} approval limit)",
         "escalation": f"approval_band: escalation (total {_money(t)} is above the ${p.approval_limit:,} "
                       f"approval limit)"}[band(p, c)],
    ]
    return "\n".join(out)


TIERS = ("fraud", "dup_exact", "dup_possible", "bank", "unauth_owner", "unauth_po", "scope", "completion",
         "totals", "price", "approval", "escalation", "clean")
TIER_WEIGHTS = (6, 4, 5, 7, 4, 4, 8, 8, 6, 9, 6, 4, 44)
FLAG_TIERS = TIERS[:10]
LOWER_FLAG_P = 0.25


def _line_picks(rng: random.Random, n: int, avoid: tuple[int, ...] = ()) -> tuple[int, ...]:
    free = [i for i in range(1, n + 1) if i not in avoid]
    return tuple(sorted(rng.sample(free, rng.randint(1, 2))))


def _set_tier(rng: random.Random, c: InvoiceCase, tier: str, p: APPolicy) -> InvoiceCase:
    if tier == "fraud":
        return replace(c, impostor=True)
    if tier == "dup_exact":
        return replace(c, duplicate="exact")
    if tier == "dup_possible":
        return replace(c, duplicate="possible")
    if tier == "bank":
        return replace(c, bank_change=True)
    if tier == "unauth_owner":
        return replace(c, owner_declined=True)
    if tier == "unauth_po":
        return replace(c, po_approved=False)
    if tier == "scope":
        return replace(c, extra_lines=_line_picks(rng, c.n_lines, c.missing_lines))
    if tier == "completion":
        return replace(c, missing_lines=_line_picks(rng, c.n_lines, c.extra_lines))
    if tier == "totals":
        return replace(c, totals_match=False)
    if tier == "price":
        tol10 = p.tolerance_pct * 10
        return replace(c, variance_pct10=tol10 + (1 if rng.random() < 0.3 else rng.randint(2, 40)))
    return c


def _with_band(rng: random.Random, c: InvoiceCase, p: APPolicy, want: str) -> InvoiceCase:
    """Pick the PO amount so that the invoice total lands in the wanted band of policy p."""
    lo, hi = {"auto_pay": (20_000, p.auto_pay_limit * 100), "approval": (p.auto_pay_limit * 100 + 1, p.approval_limit * 100),
              "escalation": (p.approval_limit * 100 + 1, p.approval_limit * 300)}[want]
    if want == "auto_pay" and c.variance_pct10 == 0 and rng.random() < 0.1:
        return replace(c, po_cents=p.auto_pay_limit * 100)  # exactly at the limit
    for _ in range(50):
        target = rng.randint(lo, hi)
        po = max(1000, round(target * 1000 / (1000 + c.variance_pct10) / 1000) * 1000)
        cand = replace(c, po_cents=po)
        if band(p, cand) == want:
            return cand
    raise AssertionError("no PO amount fits the band")  # pragma: no cover


def sample_case(rng: random.Random, variant: str) -> tuple[InvoiceCase, str]:
    p = VARIANTS[variant]
    tier = rng.choices(TIERS, TIER_WEIGHTS)[0]
    c = InvoiceCase(n_lines=rng.randint(4, 8), discount_available=rng.random() < 0.35,
                    variance_pct10=rng.randint(-30, 20))  # never above the lowest tolerance (2%)
    if tier in FLAG_TIERS:
        c = _set_tier(rng, c, tier, p)
        for lower in FLAG_TIERS[FLAG_TIERS.index(tier) + 1:]:
            if lower in ("dup_possible", "unauth_po") and tier in ("dup_exact", "unauth_owner"):
                continue  # same tier, other spelling
            if rng.random() < LOWER_FLAG_P:
                c = _set_tier(rng, c, lower, p)
        want = rng.choice(("auto_pay", "auto_pay", "approval", "escalation"))
    else:
        want = {"approval": "approval", "escalation": "escalation", "clean": "auto_pay"}[tier]
    return _with_band(rng, c, p, want), tier


def _clear_top(p: APPolicy, c: InvoiceCase) -> InvoiceCase | None:
    """The case with its highest raised flag cleared, or None when no flag is raised."""
    if c.impostor:
        return replace(c, impostor=False)
    if c.duplicate != "none":
        return replace(c, duplicate="none")
    if c.bank_change:
        return replace(c, bank_change=False)
    if c.owner_declined or not c.po_approved:
        return replace(c, owner_declined=False, po_approved=True)
    if c.extra_lines:
        return replace(c, extra_lines=())
    if c.missing_lines:
        return replace(c, missing_lines=())
    if not c.totals_match:
        return replace(c, totals_match=True)
    if variance_flag(p, c):
        return replace(c, variance_pct10=p.tolerance_pct * 10)  # exactly at the tolerance: accepted
    return None


def _flip(rng: random.Random, c: InvoiceCase, variant: str) -> InvoiceCase:
    """The deciding flag cleared (and the next, while the gold has not moved), or for clean and band rows one
    flag raised."""
    p = VARIANTS[variant]
    gold = decide(variant, c)
    new = _clear_top(p, c)
    while new is not None and decide(variant, new) == gold:
        nxt = _clear_top(p, new)
        new = None if nxt == new else nxt  # no progress: fall through to raising a flag
    if new is not None:
        return new
    for _ in range(20):
        new = _set_tier(rng, c, rng.choice(FLAG_TIERS), p)
        if decide(variant, new) != gold:
            return new
    return replace(c, impostor=True)


def generate_action_cluster(rng: random.Random, cluster_id: str) -> list[Decision]:
    v = rng.choice(sorted(VARIANTS))
    case, tier = sample_case(rng, v)
    others = [w for w in sorted(VARIANTS) if w != v]
    differ = [w for w in others if decide(w, case) != decide(v, case)]
    w = rng.choice(differ) if differ and rng.random() < 0.8 else rng.choice(others)
    flipped = _flip(rng, case, v)
    plan = [(v, case, None), (w, case, "entity_swap"), (v, flipped, "negation")]
    golds = [decide(var, c) for var, c, _ in plan]
    pool = [a for a in ACTIONS if a not in golds]
    keep = set(golds) | set(rng.sample(pool, N_OPTIONS - len(set(golds))))
    options = [o for o in ACTION_OPTIONS if o.key in keep]
    question = ACTION_QUESTIONS[0] if rng.random() < 0.5 else rng.choice(ACTION_QUESTIONS[1:])
    ctx_seed = rng.randrange(1 << 30)
    members = [(render_state(var, c, random.Random(ctx_seed)), options, g, e)
               for (var, c, e), g in zip(plan, golds)]
    out = make_cluster(cluster_id=cluster_id, family=INVOICE_ACTION_FAMILY, source=SOURCE, qtype="choice",
                       question=question, members=members)
    for d, (var, c, _) in zip(out, plan):
        d.metadata = {"variant": var, "base_tier": tier, "band": band(VARIANTS[var], c),
                      "variance_pct10": c.variance_pct10}
    return out


def generate_invoice_action(n_clusters: int, seed: int) -> list[Decision]:
    rng = random.Random(f"{INVOICE_ACTION_FAMILY}:{seed}")
    out: list[Decision] = []
    for k in range(n_clusters):
        out.extend(generate_action_cluster(rng, f"{INVOICE_ACTION_FAMILY}:{seed}:{k}"))
    return out


def generate(n_clusters: int, seed: int) -> list[Decision]:
    """Both families of this module (the CLI registers each one separately)."""
    return generate_payment_terms(n_clusters, seed) + generate_invoice_action(n_clusters, seed)
