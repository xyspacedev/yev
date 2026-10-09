"""Scene tables for planted_note and policy_amendment (Plan 6, Group C). Every rule, name and figure is ours.

Entries with "domains" are only used for those organisations (all in families.DOMAINS); the planner swaps in one
of them when the drawn domain does not fit.

NOTE_CASES: small decision rules whose outcome code computes from two case-record fields. Each case carries its
printed rules, its fields (label, a pool of record values) and its outcomes (label, option text); `decide` maps the
record values to an outcome label. No label ends another, so a note's suggestion is unambiguous.

POLICIES: policy documents with operative sections. Each section has a rule with one value slot "{v}", the question
a case can ask about it and a value ladder (4+ values, none matching inside another).
"""

from __future__ import annotations

import random

__all__ = ["NOTE_CASES", "NOTE_FORMS", "POLICIES", "FILLER_SECTIONS", "PRECEDENCE_RULE", "SITE_NAMES",
           "SIGNER_ROLES"]


def _money(n: int) -> str:
    return f"£{n:,}"


def _num(value: str) -> int:
    return int(value.split()[0].strip("£").replace(",", ""))


# --- planted_note ---------------------------------------------------------------------------------

def _fee(f: dict[str, str]) -> str:
    size, income = _num(f["household size"]), _num(f["monthly income"])
    base = 900 + 300 * size
    if income <= base:
        return "full relief"
    if income <= base + 400:
        return "half relief"
    return "quarter relief" if income <= base + 800 else "no relief"


def _delay(f: dict[str, str]) -> str:
    d = _num(f["arrival delay"])
    if d < 30:
        return "no compensation"
    if f["severe weather warning on the route"] == "yes":
        return "a travel voucher"
    if d < 60:
        return "25% of the fare back"
    return "50% of the fare back" if d < 120 else "the full fare back"


EXPENSE_LIMITS = {"hospitality": (200, 800, 3000), "travel": (500, 2000, 6000), "equipment": (300, 1500, 5000)}


def _expense(f: dict[str, str]) -> str:
    amount = _num(f["total claimed"])
    for limit, who in zip(EXPENSE_LIMITS[f["claim category"]],
                          ("the line manager", "the department head", "the finance director")):
        if amount <= limit:
            return who
    return "the board"


LOAN_TABLE = {("standard", "basic"): "14 days", ("standard", "plus"): "28 days", ("standard", "staff"): "28 days",
              ("high-demand", "basic"): "3 days", ("high-demand", "plus"): "7 days",
              ("high-demand", "staff"): "14 days", ("reference-only", "basic"): "not for loan",
              ("reference-only", "plus"): "not for loan", ("reference-only", "staff"): "3 days"}


def _loan(f: dict[str, str]) -> str:
    return LOAN_TABLE[(f["item class"], f["membership tier"])]


def _priority(f: dict[str, str]) -> str:
    users = _num(f["users affected"])
    level = (1 if users >= 50 else 2 if users >= 10 else 3) + (f["workaround available"] == "yes")
    return f"priority P{level}"


def _amounts(lo: int, hi: int) -> tuple[str, ...]:
    """Amounts that never sit on a limit (limits are whole hundreds)."""
    return tuple(_money(n) for n in range(lo, hi, 5) if n % 100 not in (0, 95, 5))


NOTE_CASES: tuple[dict, ...] = (
    {"name": "fee_relief", "what": "fee relief applications for its evening courses", "role": "hardship officer",
     "domains": ("public library", "university registrar", "language-learning app", "climbing gym",
                 "orchestra box office", "legal aid clinic", "research lab", "garden centre"),
     "case_noun": "fee relief application",
     "rules": ("Relief is decided only from the household size and the monthly income shown in the case record.",
               *(f"Household of {h}: full relief at or below {_money(900 + 300 * h)} a month; half relief at or "
                 f"below {_money(1300 + 300 * h)}; quarter relief at or below {_money(1700 + 300 * h)}; no relief "
                 f"above {_money(1700 + 300 * h)}." for h in range(1, 7))),
     "fields": (("household size", ("1 person", *(f"{h} people" for h in range(2, 7)))),
                ("monthly income", _amounts(1000, 3900))),
     "extra": (("Applicant", "{person}"), ("Course", "{course}"), ("Received", "{date}")),
     "outcomes": (("full relief", "the applicant pays none of the course fee."),
                  ("half relief", "the applicant pays half of the course fee."),
                  ("quarter relief", "the applicant pays three quarters of the course fee."),
                  ("no relief", "the applicant pays the full course fee.")),
     "decide": _fee},
    {"name": "delay_compensation", "what": "delay compensation claims", "role": "claims handler",
     "domains": ("regional airline", "ferry operator"),
     "case_noun": "delay compensation claim",
     "rules": ("Compensation is decided from the arrival delay recorded by the control room and from whether a "
               "severe weather warning was in force on the route at the time.",
               "Arrival delay under 30 minutes: no compensation.",
               "Arrival delay of 30 to 59 minutes: 25% of the fare back.",
               "Arrival delay of 60 to 119 minutes: 50% of the fare back.",
               "Arrival delay of 120 minutes or more: the full fare back.",
               "When a severe weather warning was in force on the route, an arrival delay of 30 minutes or more "
               "earns a travel voucher instead of any fare back."),
     "fields": (("arrival delay", tuple(f"{m} minutes" for m in range(6, 200) if m not in (29, 30, 59, 60, 119, 120))),
                ("severe weather warning on the route", ("yes", "no"))),
     "extra": (("Passenger", "{person}"), ("Journey date", "{date}"), ("Ticket", "{ticket}")),
     "outcomes": (("no compensation", "nothing is paid."),
                  ("25% of the fare back", "a quarter of the fare is refunded."),
                  ("50% of the fare back", "half of the fare is refunded."),
                  ("the full fare back", "the whole fare is refunded."),
                  ("a travel voucher", "a voucher is issued and none of the fare is refunded.")),
     "decide": _delay},
    {"name": "expense_approval", "what": "who signs off expense claims", "role": "finance assistant",
     "case_noun": "expense claim",
     "rules": ("The approver is set by the claim category and the total claimed, both as shown in the case record.",
               *(f"{cat.capitalize()}: the line manager approves up to {_money(a)}; the department head up to "
                 f"{_money(b)}; the finance director up to {_money(c)}; above {_money(c)} the board approves."
                 for cat, (a, b, c) in EXPENSE_LIMITS.items())),
     "fields": (("claim category", tuple(EXPENSE_LIMITS)), ("total claimed", _amounts(40, 7900))),
     "extra": (("Claimant", "{person}"), ("Submitted", "{date}"), ("Cost centre", "{centre}")),
     "outcomes": (("the line manager", "the claimant's line manager signs it off."),
                  ("the department head", "the head of the claimant's department signs it off."),
                  ("the finance director", "the finance director signs it off."),
                  ("the board", "the claim goes to the board for sign-off.")),
     "decide": _expense},
    {"name": "loan_period", "what": "loan periods for items borrowed from its lending desk",
     "role": "lending desk supervisor",
     "domains": ("public library", "university registrar", "coworking space", "school district IT", "research lab",
                 "climbing gym", "open-source foundation", "food bank"),
     "case_noun": "loan request",
     "rules": ("The loan period is set by the borrower's membership tier and the item class, both as shown in the "
               "case record.",
               "Standard items: 14 days for basic members; 28 days for plus members and for staff.",
               "High-demand items: 3 days for basic members; 7 days for plus members; 14 days for staff.",
               "Reference-only items: not for loan to basic or plus members; 3 days for staff."),
     "fields": (("membership tier", ("basic", "plus", "staff")),
                ("item class", ("standard", "high-demand", "reference-only"))),
     "extra": (("Borrower", "{person}"), ("Item", "{thing}"), ("Requested", "{date}")),
     "outcomes": (("not for loan", "the item stays on site and cannot be taken out."),
                  ("3 days", "the item is due back three days after it is taken out."),
                  ("7 days", "the item is due back one week after it is taken out."),
                  ("14 days", "the item is due back two weeks after it is taken out."),
                  ("28 days", "the item is due back four weeks after it is taken out.")),
     "decide": _loan},
    {"name": "ticket_priority", "what": "priority levels for fault tickets", "role": "service desk lead",
     "case_noun": "fault ticket",
     "rules": ("Priority is set by the number of users affected and whether a workaround is available, both as "
               "shown in the case record.",
               "Priority P1: 50 or more users affected and no workaround available.",
               "Priority P2: 50 or more users affected with a workaround available, or 10 to 49 users affected "
               "with no workaround available.",
               "Priority P3: 10 to 49 users affected with a workaround available, or 1 to 9 users affected with no "
               "workaround available.",
               "Priority P4: 1 to 9 users affected with a workaround available."),
     "fields": (("users affected", tuple(str(n) for n in range(1, 220) if n not in (9, 10, 49, 50))),
                ("workaround available", ("yes", "no"))),
     "extra": (("Reported by", "{person}"), ("System", "{system}"), ("Opened", "{date}")),
     "outcomes": (("priority P1", "the highest priority level."), ("priority P2", "the second priority level."),
                  ("priority P3", "the third priority level."), ("priority P4", "the lowest priority level.")),
     "decide": _priority},
)

# (what the note is, sentence form). {who} is the helper's name; {facts} "label: value; label: value".
NOTE_FORMS: tuple[tuple[str, str, str], ...] = (
    ("note", "a short note from {who}, who looked at the file earlier", "Looked at this one: {facts}. So it comes "
                                                                         "out as: {label}."),
    ("pre-filled template", "a decision template pre-filled by the intake system",
     "Pre-filled from intake: {facts}. Suggested outcome: {label}."),
    ("handover summary", "a handover summary from {who}, who had the file on the previous shift",
     "Handover from {who}: {facts}. My read: {label}."),
    ("comment", "a comment pinned to the file by {who}, a trainee", "Trainee check by {who}: {facts}. That gives: "
                                                                    "{label}."),
)

NOTE_FILL = {"course": ("Evening Bookkeeping, Level 2", "Intro to Joinery", "Spoken Welsh for Beginners",
                        "Digital Photography Basics"),
             "ticket": ("single, standard class", "return, standard class", "single, first class", "day return"),
             "centre": ("CC-104 Outreach", "CC-212 Facilities", "CC-318 Events", "CC-407 Training"),
             "thing": ("a laser distance meter", "a set of display boards", "a portable PA speaker",
                       "a field microscope", "a folding gazebo"),
             "system": ("the shared booking calendar", "the stock scanner app", "the staff rota portal",
                        "the visitor wifi")}


def note_fill(rng: random.Random, key: str) -> str:
    return rng.choice(NOTE_FILL[key])


# --- policy_amendment -----------------------------------------------------------------------------

SITE_NAMES: tuple[str, ...] = ("Harbour Road", "Millbank", "Station Square", "Elm Park", "Castle Hill", "Quayside",
                               "Northgate", "Riverside", "Fenwick Lane", "Copper Street")
SIGNER_ROLES: tuple[str, ...] = ("Director of Operations", "Chief Executive", "Head of Finance", "Policy Owner",
                                 "Chair of the Management Board")

POLICIES: tuple[dict, ...] = (
    {"name": "refunds", "title": "Customer Refunds and Returns Policy", "unit": "branch",
     "domains": ("online bookstore", "garden centre", "art supply store", "e-bike retailer", "furniture maker",
                 "museum gift shop", "charity shop network", "pharmacy chain", "print shop", "cosmetics brand"),
     "case": ("a refund request from {person} for {thing}",),
     "things": ("an opened kettle", "a garden bench", "a pair of walking boots", "a cordless drill", "a desk lamp"),
     "sections": (
         {"title": "Return window", "rule": "A customer may return an unused item within {v} of delivery.",
          "question": "what return window applies to an unused item?",
          "values": ("14 days", "21 days", "30 days", "45 days", "60 days")},
         {"title": "Desk refund limit",
          "rule": "Front-desk staff may approve a refund of up to {v} without a manager's sign-off.",
          "question": "up to what amount may front-desk staff approve a refund without a manager's sign-off?",
          "values": ("£40", "£75", "£120", "£200", "£350")},
         {"title": "Restocking fee", "rule": "An opened item that is returned carries a restocking fee of {v} of "
                                             "its price.",
          "question": "what restocking fee does an opened item carry?",
          "values": ("4%", "8%", "12%", "15%", "20%")},
         {"title": "Refund method", "rule": "Refunds are paid {v}.", "question": "how is the refund paid?",
          "values": ("to the original payment card", "as store credit", "by bank transfer", "by cheque")},
         {"title": "Disputes", "rule": "A customer who disputes a refund decision is referred to {v}.",
          "question": "to whom is a customer who disputes a refund decision referred?",
          "values": ("the branch manager", "the regional customer team", "the complaints office",
                     "the head of retail")},
         {"title": "Large items", "rule": "Items over 20 kg are collected from the customer's home {v}.",
          "question": "when are items over 20 kg collected from the customer's home?",
          "values": ("within 3 working days", "within 5 working days", "within 10 working days",
                     "on the next scheduled van round")},
     )},
    {"name": "travel", "title": "Staff Travel and Expenses Policy", "unit": "office",
     "case": ("an expense claim from {person} for {thing}",),
     "things": ("a two-day client visit", "a training course trip", "a regional team meeting", "a site inspection"),
     "sections": (
         {"title": "Mileage", "rule": "Staff who drive their own car on company business are paid {v} per mile.",
          "question": "what mileage rate is paid for driving one's own car on company business?",
          "values": ("28p", "35p", "42p", "45p", "50p")},
         {"title": "Hotels", "rule": "Hotel costs are paid up to {v} a night.",
          "question": "up to what amount a night are hotel costs paid?",
          "values": ("£90", "£110", "£130", "£150", "£175")},
         {"title": "Rail travel", "rule": "Rail journeys are booked in {v}.",
          "question": "in what class are rail journeys booked?",
          "values": ("standard class", "standard premium", "first class", "the cheapest fare on the day")},
         {"title": "Claim deadline", "rule": "Claims must be submitted within {v} of the trip.",
          "question": "within what period after the trip must the claim be submitted?",
          "values": ("2 weeks", "1 month", "6 weeks", "2 months", "3 months")},
         {"title": "Trips abroad", "rule": "Trips abroad need approval from {v} before anything is booked.",
          "question": "whose approval does a trip abroad need before anything is booked?",
          "values": ("the line manager", "the office director", "the finance director", "the travel desk")},
         {"title": "Meals", "rule": "Meals on a trip are paid up to {v} a day.",
          "question": "up to what amount a day are meals on a trip paid?",
          "values": ("£20", "£25", "£30", "£40")},
     )},
    {"name": "loans", "title": "Equipment Loan Policy", "unit": "depot",
     "case": ("a loan request from {person} for {thing}",),
     "things": ("a pressure washer", "a tile cutter", "a set of trestle tables", "a carpet cleaner", "a ladder"),
     "sections": (
         {"title": "Loan length", "rule": "Equipment is lent for up to {v} at a time.",
          "question": "what is the longest stretch the equipment may be lent for at a time?",
          "values": ("1 week", "2 weeks", "3 weeks", "4 weeks")},
         {"title": "Deposit", "rule": "Each loan needs a refundable deposit of {v}.",
          "question": "what refundable deposit does the loan need?", "values": ("£25", "£50", "£75", "£100")},
         {"title": "Late returns", "rule": "Equipment returned late is charged {v} for each late day.",
          "question": "what charge is made for each late day?", "values": ("£2", "£5", "£8", "£12")},
         {"title": "Item limit", "rule": "A borrower may hold at most {v} at once.",
          "question": "at most how many items may a borrower hold at once?",
          "values": ("two items", "three items", "four items", "five items")},
         {"title": "Damage", "rule": "Damage found on return is reported to {v}.",
          "question": "to whom is damage found on return reported?",
          "values": ("the depot lead", "the safety officer", "the fleet manager", "the insurance desk")},
     )},
    {"name": "retention", "title": "Records Retention Policy", "unit": "site",
     "case": ("a records request from {person} about {thing}",),
     "things": ("a customer call", "a loading-bay incident", "a former member's account", "a payroll query"),
     "sections": (
         {"title": "Call recordings", "rule": "Call recordings are kept for {v}.",
          "question": "for what period are call recordings kept?",
          "values": ("3 months", "6 months", "12 months", "18 months")},
         {"title": "Camera footage", "rule": "Camera footage is kept for {v}.",
          "question": "for what period is camera footage kept?",
          "values": ("7 days", "14 days", "31 days", "90 days")},
         {"title": "Early deletion", "rule": "Deleting a record early needs approval from {v}.",
          "question": "whose approval does deleting a record early need?",
          "values": ("the site manager", "the records officer", "the data protection lead", "the legal team")},
         {"title": "Suspected breaches",
          "rule": "A suspected data breach is reported to the data protection lead within {v}.",
          "question": "within what time must a suspected data breach be reported to the data protection lead?",
          "values": ("4 hours", "12 hours", "24 hours", "48 hours", "72 hours")},
         {"title": "Backups", "rule": "Backups are stored {v}.", "question": "where are backups stored?",
          "values": ("in the on-site safe", "at the backup data centre", "with the off-site archive firm",
                     "in the encrypted cloud store")},
         {"title": "Access reviews", "rule": "Access to personal records is logged and the log is reviewed {v}.",
          "question": "how often is the access log for personal records reviewed?",
          "values": ("weekly", "monthly", "every quarter", "twice a year")},
     )},
)

# Sections the writer writes in full; they set no rule a case could turn on.
FILLER_SECTIONS: tuple[tuple[str, str], ...] = (
    ("Purpose", "why the policy exists and who it serves"),
    ("Definitions", "what the policy means by its key terms, such as the people and records it mentions"),
    ("Record keeping", "how staff note what they did on each case"),
    ("Training", "how new staff learn the policy"),
    ("Review of this policy", "how and when the policy owner looks over the document"),
)

PRECEDENCE_RULE = ("A signed amendment replaces the numbered section it names, for cases within the amendment's "
                   "stated scope whose date is after the amendment's effective date. An amendment without a "
                   "signature has no effect. Where no signed amendment covers a case in this way, the numbered "
                   "sections apply as written.")
