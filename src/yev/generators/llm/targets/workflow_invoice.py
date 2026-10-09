"""invoice_line planner (Plan 6, Group D): one review packet per group, six questions about two of its lines.

Code builds the packet's tables (invoice header and lines, purchase order items with their open point, change
orders, evidence records, prior invoices) and the messages' facts; the writer copies the tables and writes the
messages out. For each of two asked lines (never the same item, and neither line's answers resting on the other):
- scope: how the line stands against the order. priced_unit (a priced item, half the time under different
  wording), component_also_billed (a part another item's rate includes, with that item billed on a third line),
  extra_approved (an approved change order adds it), extra_unapproved (not in the order and nothing approves it;
  half are a similar-sounding extra the order's text rules out, and some have a pending change request or a vendor
  email saying the work was done), not_ours (another organisation's site), unsure (the order's open point is about
  exactly this item).
- completion: what our records show for the service period. confirmed / less / not_done come from our single
  record in the line's unit and period; unsure means two such records of ours disagree; vendor_only means only the
  vendor's paperwork has a figure for the period; none means no record at all for the period.
- evidence: whether a named record is our own count for that line, in its unit and for the service period. The
  negatives are another unit, another period, a running total, the vendor's figure or another line's record.
Every figure a question needs is printed in a row; nothing has to be added up or converted.
"""

from __future__ import annotations

import random

from yev.generators.llm.planned import Group, QSpec, StateSpec
from yev.generators.llm.targets.agent_trace import an, person
from yev.generators.llm.targets.workflow_scenes import (
    COVER_NOTES, MONTHS, OTHER_ORGS, OWNER_ROLES, PAYMENT_TERMS, QUIET_MESSAGES, SECOND_OWN_SOURCE, STREETS,
    TRADES, UNIT_KINDS, VENDOR_SUFFIXES, VENDOR_WORDS,
)
from yev.schema import Option

__all__ = ["SCOPE_OPTIONS", "COMPLETION_OPTIONS", "EVIDENCE_OPTIONS", "ordered", "pick", "plan_invoice"]

SCOPE_SHARES: dict[str, float] = {"priced_unit": 0.55, "component_also_billed": 0.07, "extra_approved": 0.07,
                                  "extra_unapproved": 0.15, "not_ours": 0.08, "unsure": 0.08}
COMPLETION_SHARES: dict[str, float] = {"confirmed": 0.50, "less": 0.15, "not_done": 0.08, "vendor_only": 0.08,
                                       "none": 0.12, "unsure": 0.07}
# Another organisation's line can only be silent in our records or carry the vendor's word; the other scopes take
# the rest of the completion shares so the overall shares stay as above.
NOT_OURS_COMPLETION: dict[str, float] = {"none": 0.6, "vendor_only": 0.4}
_P_NOT_OURS = SCOPE_SHARES["not_ours"]
OTHER_COMPLETION: dict[str, float] = {
    k: (v - _P_NOT_OURS * NOT_OURS_COMPLETION.get(k, 0.0)) / (1 - _P_NOT_OURS) for k, v in COMPLETION_SHARES.items()
}
COUNTING = ("confirmed", "less", "unsure")   # completions with a record of ours in the line's unit and period
YES_GIVEN_COUNTING = 0.40 / sum(COMPLETION_SHARES[k] for k in COUNTING)
# Evidence negatives each completion can carry without the decoy changing what our records show.
DECOYS: dict[str, tuple[str, ...]] = {
    "confirmed": ("other_unit", "other_period", "other_line", "cumulative", "vendor"),
    "less": ("other_unit", "other_period", "other_line", "cumulative", "vendor"),
    "unsure": ("other_unit", "other_period", "other_line", "cumulative", "vendor"),
    "not_done": ("other_period", "other_line", "vendor"),
    "vendor_only": ("vendor", "other_period", "other_line"),
    "none": ("other_period", "other_line"),
}
NOT_OURS_DECOYS: dict[str, tuple[str, ...]] = {"vendor_only": ("vendor", "other_line"), "none": ("other_line",)}
# An extra approved this month has no earlier deliveries, so it carries no earlier-period or running-total record.
NO_HISTORY = ("other_period", "cumulative")
ALIAS_SHARE = 0.5       # asked priced lines worded differently from the order
NEAR_SHARE = 0.5        # extras that sound like a priced item the order's text rules out
UNAPPROVED_FLAVOURS = ("silent", "pending", "told")
IRRELEVANT_CO_SHARE = 0.4
CANONICAL_SHARE = 0.5

SCOPE_OPTIONS: tuple[Option, ...] = (
    Option("priced_unit", "The line bills one of the purchase order's priced items, at that item's rate, for our "
                          "site, even if the invoice words it differently."),
    Option("component_also_billed", "The line bills something the order already includes within the rate of "
                                    "another item, and that item is billed on this invoice too."),
    Option("extra_approved", "The line bills work or goods the order does not include, and an approved change "
                             "order of ours adds it."),
    Option("extra_unapproved", "The line bills work or goods the order does not include or rules out, and nothing "
                               "approved on our side adds it."),
    Option("not_ours", "The line is for a site or organisation other than ours."),
    Option("unsure", "The order marks this very item as an open point still to be agreed, so the packet cannot "
                     "settle it."),
)
COMPLETION_OPTIONS: tuple[Option, ...] = (
    Option("confirmed", "Our own record of this line, in its unit and for the service period, shows exactly the "
                        "quantity billed."),
    Option("less", "Our own record of this line, in its unit and for the service period, shows more than zero but "
                   "fewer than billed."),
    Option("not_done", "Our own record of this line, in its unit and for the service period, shows zero."),
    Option("vendor_only", "Only the vendor's own paperwork gives a figure for this line in the service period; "
                          "none of our records does."),
    Option("none", "No record at all, ours or the vendor's, gives a figure for this line in the service period."),
    Option("unsure", "Two of our own records count this line in its unit for the service period, and their "
                     "figures differ."),
)
EVIDENCE_OPTIONS: tuple[Option, Option] = (
    Option("yes", "Yes: it is our own record of that line, in the line's unit, for the service period."),
    Option("no", "No: it is the vendor's, or about another line, or in another unit, or for another period or a "
                 "longer span."),
)
SCOPE_QUESTIONS: tuple[str, ...] = (
    "How does invoice line {n} stand against the purchase order?",
    "Checked against the purchase order and the change orders, which of these is invoice line {n}?",
    "Set against what we ordered, how should invoice line {n} be classed?",
)
COMPLETION_QUESTIONS: tuple[str, ...] = (
    "What do the evidence records show about delivery of invoice line {n} in the service period?",
    "Going by the evidence records, how far was invoice line {n} delivered in the service period?",
    "For invoice line {n}, which of these matches what the evidence records hold for the service period?",
)
EVIDENCE_QUESTIONS: tuple[str, ...] = (
    "Is evidence record {e} our own count of what was delivered on invoice line {n}, in that line's unit and for "
    "the service period?",
    "Does evidence record {e} give our own figure for what invoice line {n} delivered, in the line's unit and for "
    "the service period?",
    "Is evidence record {e} a record of ours that counts invoice line {n} in the line's own unit for the service "
    "period?",
)
NOT_DONE_NOTES = {"goods": "nothing from this vendor arrived for this item in the period",
                  "service": "the booked work did not take place; {why}"}
NOT_DONE_WHY = ("the site was closed for building works", "nobody from the vendor attended",
                "the work was postponed at the vendor's request")


def pick(rng: random.Random, wordings: tuple[str, ...]) -> str:
    """The first wording half the time, else one of the others."""
    return wordings[0] if rng.random() < 0.5 else rng.choice(wordings[1:])


def ordered(rng: random.Random, options: tuple[Option, ...]) -> tuple[Option, ...]:
    """The canonical order half the time, else a shuffle (letter-position balance)."""
    if rng.random() < CANONICAL_SHARE:
        return options
    out = list(options)
    rng.shuffle(out)
    return tuple(out)


def _draw(rng: random.Random, shares: dict[str, float]) -> str:
    return rng.choices(list(shares), weights=list(shares.values()))[0]


def _money(rng: random.Random, span: tuple[int, int]) -> float:
    return rng.randint(*span) + rng.choice((0.0, 0.25, 0.5, 0.75))


def _fmt(x: float) -> str:
    return f"{x:,.2f}"


def _unit(kind: str, sg: str | None, pl: str | None) -> tuple[str, str]:
    return (sg, pl) if kind == "goods" else (UNIT_KINDS[kind]["sg"], UNIT_KINDS[kind]["pl"])


def _title(domain: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in domain.split())


def _scopes(rng: random.Random) -> tuple[str, str]:
    """Scopes for the two asked lines; one trade has a single part and a single open point to give."""
    while True:
        a, b = _draw(rng, SCOPE_SHARES), _draw(rng, SCOPE_SHARES)
        if not (a == b and a in ("component_also_billed", "unsure")):
            return a, b


def _completion(rng: random.Random, scope: str) -> str:
    return _draw(rng, NOT_OURS_COMPLETION if scope == "not_ours" else OTHER_COMPLETION)


def plan_invoice(rng: random.Random, index: int, domain: str) -> Group:
    trade = rng.choice(TRADES)
    items = trade["items"]
    m = rng.randrange(3, 12)                     # April to December, so earlier months and a running total exist
    month, period, prev = MONTHS[m], f"{MONTHS[m]} 2026", f"{MONTHS[m - 1]} 2026"
    vendor = f"{rng.choice(VENDOR_WORDS)} {rng.choice(VENDOR_SUFFIXES)}"
    clerk, owner, role = person(rng), person(rng), rng.choice(OWNER_ROLES)
    contact = person(rng)
    org = f"{_title(domain)} ({rng.choice(STREETS).split()[0]} office)"
    our_street, far_street = rng.sample(STREETS, 2)
    our_site = f"{rng.randint(2, 180)} {our_street}"
    other_site = f"{rng.choice(OTHER_ORGS)}, {rng.randint(2, 180)} {far_street}"
    rates = [_money(rng, it[5]) for it in items]
    project, po_no = f"PRJ-{rng.randint(1000, 9999)}", f"PO-{rng.randint(10000, 99999)}"

    scopes = _scopes(rng)
    free = list(range(len(items)))
    rng.shuffle(free)
    part_parent = trade["part"][0]
    if "component_also_billed" in scopes:
        free.remove(part_parent)
    extras = [("near", trade["near"][2:])] + [("plain", e) for e in trade["extras"]]
    near_free = True
    plain = [e for e in extras if e[0] == "plain"]
    rng.shuffle(plain)

    def item_line(k: int, site: str, alias: bool, role_: str, scope: str) -> dict:
        po, al, kind, sg, pl, _, qspan = items[k]
        sg, pl = _unit(kind, sg, pl)
        text = al if alias else po
        ref = f"PO item {k + 1}: {po}" if site == our_site else f"{text} (delivered to {site})"
        return {"text": text, "site": site, "kind": kind, "sg": sg, "pl": pl, "qty": rng.randint(*qspan),
                "rate": rates[k], "ref": ref, "role": role_, "scope": scope, "qspan": qspan}

    def other_line(text: str, kind: str, sg, pl, rspan, qspan, role_: str, scope: str) -> dict:
        sg, pl = _unit(kind, sg, pl)
        return {"text": text, "site": our_site, "kind": kind, "sg": sg, "pl": pl, "qty": rng.randint(*qspan),
                "rate": _money(rng, rspan), "ref": text, "role": role_, "scope": scope, "qspan": qspan}

    asked, used_extras, flavours = [], [], {}
    for role_, scope in zip("ab", scopes):
        if scope == "priced_unit":
            line = item_line(free.pop(), our_site, rng.random() < ALIAS_SHARE, role_, scope)
        elif scope == "not_ours":
            line = item_line(free.pop(), other_site, rng.random() < 0.5, role_, scope)
        elif scope == "component_also_billed":
            text, sg, pl, rspan, qspan = trade["part"][2:]
            line = other_line(text, "goods", sg, pl, rspan, qspan, role_=role_, scope=scope)
        elif scope == "unsure":
            line = other_line(*trade["open"][2:], role_=role_, scope=scope)
        else:
            if near_free and rng.random() < NEAR_SHARE:
                near_free, spec = False, trade["near"][2:]
            else:
                spec = plain.pop()[1]
            line = other_line(*spec, role_=role_, scope=scope)
            used_extras.append(line["text"])
            if scope == "extra_unapproved":
                flavours[role_] = rng.choice(UNAPPROVED_FLAVOURS)
        line["completion"] = _completion(rng, scope)
        if line["completion"] in ("less", "unsure"):
            line["qty"] = max(line["qty"], 2)
        asked.append(line)

    fillers = []
    if "component_also_billed" in scopes:
        fillers.append(item_line(part_parent, our_site, rng.random() < 0.5, "filler", "priced_unit"))
    n_fill = min(rng.randint(2, 4), len(free) + len(fillers))
    while len(fillers) < n_fill:
        fillers.append(item_line(free.pop(), our_site, rng.random() < 0.5, "filler", "priced_unit"))
    for f in fillers:
        f["completion"] = "confirmed"
    lines = asked + fillers
    rng.shuffle(lines)
    for no, line in enumerate(lines, 1):
        line["no"] = no

    # --- evidence records
    records: list[dict] = []

    def rec(line: dict, ours: bool, fig: int, unit: str | None = None, period_: str = period,
            src: str | None = None, note: str | None = None) -> dict:
        kinds = UNIT_KINDS[line["kind"]]
        r = {"ours": ours, "src": src or (kinds["src"] if ours else kinds["vendor"]), "ref": line["ref"],
             "fig": fig, "unit": unit or line["pl"], "period": period_, "note": note}
        records.append(r)
        return r

    for f in fillers:
        f["counting"] = [rec(f, True, f["qty"])]
    targets = {}
    for line in asked:
        c, qty = line["completion"], line["qty"]
        line["counting"] = []
        vendor_rec = None
        if c == "confirmed":
            line["counting"] = [rec(line, True, qty)]
        elif c == "less":
            line["counting"] = [rec(line, True, rng.randint(max(1, qty // 3), qty - 1))]
        elif c == "unsure":
            line["counting"] = [rec(line, True, qty),
                                rec(line, True, rng.randint(max(1, qty // 3), qty - 1), src=SECOND_OWN_SOURCE)]
        elif c == "not_done":
            note = NOT_DONE_NOTES["goods" if line["kind"] == "goods" else "service"].format(
                why=rng.choice(NOT_DONE_WHY))
            rec(line, True, 0, note=note)
        elif c == "vendor_only":
            vendor_rec = rec(line, False, qty)
        if line["counting"] and rng.random() < YES_GIVEN_COUNTING:
            targets[line["role"]] = (rng.choice(line["counting"]), "yes")
            continue
        pool = NOT_OURS_DECOYS[c] if line["scope"] == "not_ours" else DECOYS[c]
        if line["scope"] == "extra_approved":
            pool = tuple(x for x in pool if x not in NO_HISTORY)
        flavour = rng.choice(pool)
        lo, hi = line["qspan"]
        if flavour == "vendor":
            target = vendor_rec or rec(line, False, qty)
        elif flavour == "other_line":
            target = rng.choice(fillers)["counting"][0]
        elif flavour == "other_period":
            target = rec(line, True, rng.randint(max(1, lo), max(1, hi)), period_=prev)
        elif flavour == "other_unit":
            alt_sg, alt_pl, alt_src = UNIT_KINDS[line["kind"]]["alt"]
            target = rec(line, True, rng.randint(1, max(2, qty)), unit=alt_pl, src=alt_src)
        else:   # cumulative
            target = rec(line, True, qty + rng.randint(qty, 4 * qty),
                         period_=f"January to {month} 2026 (running total since the order began)")
        targets[line["role"]] = (target, f"no_{flavour}")
    if rng.random() < 0.5:   # an earlier month's record for one billed item, which no question is about
        f = rng.choice(fillers)
        rec(f, True, rng.randint(max(1, f["qspan"][0]), max(1, f["qspan"][1])), period_=prev)
    rng.shuffle(records)
    for i, r in enumerate(records, 1):
        r["id"] = f"E{i}"

    def record_row(r: dict) -> str:
        keeper = "our team" if r["ours"] else f"{vendor} (vendor)"
        row = (f"{r['id']} | Kept by: {keeper} | Source: {r['src']} | Item: {r['ref']} | Figure: {r['fig']} | "
               f"Unit: {r['unit']} | Period: {r['period']}")
        return row + (f" | Note: {r['note']}" if r["note"] else "")

    # --- order, change orders, messages
    def po_text(k: int) -> str:
        text = items[k][0]
        if k == trade["part"][0]:
            text += f", with {trade['part'][1]} included in the rate"
        if k == trade["near"][0]:
            text += f"; not covered by this item: {trade['near'][1]}"
        return text

    po_rows = [f"Item {k + 1} | {po_text(k)} | Unit: {_unit(it[2], it[3], it[4])[1]} | Rate per unit: "
               f"{_fmt(rates[k])}" for k, it in enumerate(items)]
    po_rows.append(f"Open point on item {trade['open'][0] + 1}: whether this order covers {trade['open'][1]} is "
                   f"still to be agreed in writing by {owner}, the {role}.")
    changes = []
    co_no = rng.randint(100, 900)
    approved_on = f"{rng.randint(2, 14)} {month} 2026"
    for line in asked:
        if line["scope"] == "extra_approved":
            changes.append(f"CO-{co_no + len(changes)} | Adds: {line['text']} | Unit: {line['pl']} | Rate per unit: "
                           f"{_fmt(line['rate'])} | Approved by: {owner}, {role}, on {approved_on}")
        elif flavours.get(line["role"]) == "pending":
            changes.append(f"CR-{co_no + len(changes)} | Raised by: {vendor} | Adds: {line['text']} | Unit: "
                           f"{line['pl']} | Rate per unit: {_fmt(line['rate'])} | Status: awaiting our decision")
    unused = [e[1] for e in extras if e[1][0] not in used_extras]
    if unused and rng.random() < IRRELEVANT_CO_SHARE:
        text, kind, sg, pl, rspan, _ = rng.choice(unused)
        changes.append(f"CO-{co_no + len(changes)} | Adds: {text} | Unit: {_unit(kind, sg, pl)[1]} | Rate per unit: "
                       f"{_fmt(_money(rng, rspan))} | Approved by: {owner}, {role}, on {approved_on}")
    rng.shuffle(changes)

    messages = [f"An email from {contact} at {vendor} that {rng.choice(COVER_NOTES)}."]
    for line in asked:
        if flavours.get(line["role"]) == "told":
            messages.append(f"A second email from {contact} at {vendor} saying their team also carried out "
                            f"\"{line['text']}\" while on site this month and has billed it on this invoice.")
    messages += [f"{m[:1].upper()}{m[1:]}." for m in rng.sample(QUIET_MESSAGES, rng.randint(1, 2))]
    rng.shuffle(messages)

    inv = rng.randint(10000, 99999)
    priors = [f"INV-{inv - 37 * j} | Period: {MONTHS[m - j]} 2026 | Total: {_fmt(_money(rng, (900, 6000)))} | "
              f"Status: paid" for j in range(1, rng.randint(2, 4))]
    header = [f"Invoice number: INV-{inv}", f"Vendor: {vendor}", f"Bill to: {org}, project {project}",
              f"Our site: {our_site}", f"Purchase order: {po_no}", f"Invoice date: {rng.randint(26, 28)} {month} 2026",
              f"Service period: {period}", f"Payment terms: {rng.choice(PAYMENT_TERMS)}"]
    line_rows = [f"Line {ln['no']} | {ln['text']} | Site: {ln['site']} | Quantity: {ln['qty']} | Unit: {ln['pl']} | "
                 f"Rate per unit: {_fmt(ln['rate'])} | Amount: {_fmt(ln['qty'] * ln['rate'])}"
                 for ln in sorted(lines, key=lambda x: x["no"])]

    def block(heading: str, rows: list[str]) -> str:
        return f"{heading} (copy as given):\n" + "\n".join(f"- {r}" for r in rows)

    brief = "\n\n".join([
        block("Invoice header", header),
        block("Invoice lines", line_rows),
        block("Purchase order items", po_rows),
        block("Change orders and change requests", changes or ["none on file"]),
        block("Evidence records", [record_row(r) for r in records]),
        block("Prior invoices from this vendor", priors),
        "Messages (write each one out in full with its sender and date, keeping these facts):\n"
        + "\n".join(f"{i}. {msg}" for i, msg in enumerate(messages, 1)),
        "The packet holds no order items, change orders, approvals, evidence records or messages beyond these, and "
        "nothing in it comments on whether a line is in order, delivered or due for payment.",
    ])
    scene = (f"{an(domain, start=True)}. {clerk}, a member of the {domain}'s accounts payable team, checks "
             f"supplier invoices before they are paid. The state is one review packet from the invoice queue: "
             f"invoice INV-{inv} from {vendor}, a {trade['trade']} supplier, billed to {org} under purchase order "
             f"{po_no}, together with the order's priced items, change orders, evidence records, earlier invoices "
             f"and messages.")

    questions = []
    for line in sorted(asked, key=lambda x: x["no"]):
        n = line["no"]
        target, ev_skill = targets[line["role"]]
        questions += [
            QSpec(name=f"line{n}_scope", type="choice", question=pick(rng, SCOPE_QUESTIONS).format(n=n),
                  options=ordered(rng, SCOPE_OPTIONS), gold=line["scope"], skill=f"invoice_scope_{line['scope']}"),
            QSpec(name=f"line{n}_completion", type="choice", question=pick(rng, COMPLETION_QUESTIONS).format(n=n),
                  options=ordered(rng, COMPLETION_OPTIONS), gold=line["completion"],
                  skill=f"invoice_completion_{line['completion']}"),
            QSpec(name=f"line{n}_evidence", type="noul",
                  question=pick(rng, EVIDENCE_QUESTIONS).format(n=n, e=target["id"]), options=EVIDENCE_OPTIONS,
                  gold="yes" if ev_skill == "yes" else "no", skill=f"invoice_evidence_{ev_skill}"),
        ]
    return Group(gid=f"inv{index:05d}", scene=scene,
                 states=(StateSpec(sid="packet", brief=brief, questions=tuple(questions)),))
