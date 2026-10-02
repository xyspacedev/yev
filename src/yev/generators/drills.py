"""Date and number drills with computed labels (Stage 1, family 2).

Five question kinds, each as small contrastive clusters whose members sit one step either side of a
boundary: deadline met? (noul), which date is the deadline? (choice of 4-6 dates), business days
between two dates (choice of counts), a sum against a threshold (noul), and a percentage-change band
(score with numbered levels "0".."n-1").
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from fractions import Fraction

from yev.generators.common import keep_valid_clusters, make_cluster
from yev.generators.dates import STYLES, add_business_days, business_days, fmt, last_day_of_month
from yev.schema import Decision, Option

SOURCE = "synthetic_rules"
FAMILY = "date_number_drills"
KINDS = ("deadline_met", "deadline_choice", "business_days", "sum_threshold", "pct_change")
SLASH_NOTE = "Slash dates are written day/month/year."
BUSINESS_NOTE = "Business days are Monday to Friday, excluding public holidays."
# Every state with a day window says how the days are counted, with this phrase.
COUNT_PHRASE = "counts as day 1"
NOUL_OPTIONS = [Option("yes", "Yes."), Option("no", "No.")]


def random_day(rng: random.Random) -> date:
    """Mostly anywhere in 2023-2028, often near a month end, the year end or the end of February."""
    r = rng.random()
    if r < 0.2:
        return date(rng.randint(2023, 2027), 12, rng.randint(10, 31))
    if r < 0.4:
        return date(rng.choice((2023, 2024, 2025, 2026, 2027, 2028)), 2, rng.randint(5, 28))
    if r < 0.6:
        return last_day_of_month(rng.randint(2023, 2028), rng.randint(1, 12)) - timedelta(days=rng.randint(0, 12))
    return date(2023, 1, 1) + timedelta(days=rng.randrange(6 * 365))


def pick_style(rng: random.Random) -> str:
    return rng.choices(STYLES, (4, 2, 2, 1))[0]


def with_slash_note(text: str, styles) -> str:
    return f"{text}\n{SLASH_NOTE}" if "slash_dmy" in styles else text


def money(cents: int) -> str:
    return f"${cents // 100:,}.{cents % 100:02d}"


@dataclass(frozen=True)
class Deadline:
    start: date
    n: int
    unit: str  # "calendar" or "business"
    holidays: tuple[date, ...]

    def due(self) -> date:
        if self.unit == "calendar":
            return self.start + timedelta(days=self.n)
        return add_business_days(self.start, self.n, self.holidays)

    def meta(self) -> dict:
        return {"start": self.start.isoformat(), "n": self.n, "unit": self.unit,
                "holidays": [h.isoformat() for h in self.holidays]}


def sample_deadline(rng: random.Random) -> Deadline:
    start = random_day(rng)
    if rng.random() < 0.5:
        return Deadline(start, rng.choice((7, 10, 14, 15, 21, 28, 30, 30, 45, 60, 90)), "calendar", ())
    n = rng.choice((3, 5, 5, 7, 10, 10, 15, 20))
    holidays: tuple[date, ...] = ()
    if rng.random() < 0.5:
        span = [start + timedelta(days=k) for k in range(1, n + 3)]
        holidays = (rng.choice([d for d in span if d.weekday() < 5]),)
    return Deadline(start, n, "business", holidays)


def holiday_sentence(holidays, style: str) -> str:
    return "".join(f" {fmt(h, style)} was a public holiday." for h in holidays)


def unit_phrase(d: Deadline) -> str:
    return f"{d.n} {d.unit} day{'s' if d.n != 1 else ''}"


def count_rule(d: Deadline, anchor: str) -> str:
    first = "the day after" if d.unit == "calendar" else "the first business day after"
    return f"{first} {anchor} {COUNT_PHRASE}"


def swap_unit(d: Deadline) -> Deadline:
    return replace(d, unit="business" if d.unit == "calendar" else "calendar")


# ---------------------------------------------------------------- kind 1: deadline met?

EVENTS = [
    ("The invoice was issued", "payment is due", "Payment cleared", "the invoice date"),
    ("The contract was signed", "the first deliverable is due", "The deliverable was submitted", "signing"),
    ("The claim was filed", "the insurer must decide", "The decision letter was sent", "the filing date"),
    ("The notice was received", "a written response is due", "The response was posted", "receipt of the notice"),
    ("The order was placed", "the shipment must leave the warehouse", "The parcel was dispatched", "the order date"),
    ("The inspection was booked", "the inspection report is due", "The report was uploaded", "the booking date"),
]


def render_deadline_met(tpl: int, ev: tuple, ref: str, other: date, styles: dict, d: Deadline, done: date) -> str:
    st = styles
    S, X, O = fmt(d.start, st["start"]), fmt(done, st["done"]), fmt(other, st["other"])
    hol = holiday_sentence(d.holidays, st["start"])
    note = f" {BUSINESS_NOTE}" if d.unit == "business" else ""
    event, rule, finished, anchor = ev
    if tpl == 0:
        text = (f"{event} on {S}. Under the terms, {rule} within {unit_phrase(d)} of that date, and "
                f"{count_rule(d, 'that date')}.{note}{hol} "
                f"{finished} on {X}. A reminder had been sent on {O}.")
    elif tpl == 1:
        text = (f"Case {ref} timeline:\n- {O}: account details updated\n- {S}: {event.lower()}\n- {X}: "
                f"{finished.lower()}\nRule: {rule} no later than {unit_phrase(d)} after {anchor}, "
                f"and {count_rule(d, anchor)}."
                f"{note}{hol}")
    else:
        text = (f"Ops log {ref}. {finished} on {X}. {event} on {S}, and {rule} within {unit_phrase(d)}; "
                f"{count_rule(d, 'that date')}.{note}{hol} The file was last audited on {O}.")
    return with_slash_note(text, st.values())


def deadline_met_cluster(rng: random.Random, cid: str) -> list[Decision]:
    d = sample_deadline(rng)
    ev, tpl, ref = rng.choice(EVENTS), rng.randrange(3), f"{rng.randint(1000, 99999)}"
    styles = {k: pick_style(rng) for k in ("start", "done", "other")}
    other = d.start - timedelta(days=rng.randint(3, 60))
    due = d.due()
    offsets = [0, 1, -1]
    base_off = rng.choice((0, 0, 1))
    flip = 1 - base_off  # 0 -> 1 (late), 1 -> 0 (on time)
    rows = [(d, due + timedelta(days=base_off), None), (d, due + timedelta(days=flip), "date")]
    optional = [(d, due + timedelta(days=o), "date") for o in offsets if o not in (base_off, flip)]
    optional.append((replace(d, n=d.n - 1), due + timedelta(days=base_off), "date"))
    optional.append((swap_unit(d), due + timedelta(days=base_off), "unit"))
    if d.holidays:
        optional.append((replace(d, holidays=()), due + timedelta(days=base_off), "exception"))
    rows += rng.sample(optional, rng.randint(1, 2))
    question = rng.choice(("Was the deadline met?", "Was this done on time under the stated rule?"))
    members = [(render_deadline_met(tpl, ev, ref, other, styles, dd, done), NOUL_OPTIONS,
                "yes" if done <= dd.due() else "no", edit) for dd, done, edit in rows]
    metas = [{"kind": "deadline_met", **dd.meta(), "done": done.isoformat()} for dd, done, _ in rows]
    return _cluster(cid, "noul", question, members, metas)


# ---------------------------------------------------------------- kind 2: which date is the deadline?


def add_months(d: date, months: int) -> date:
    """Same day of the month, `months` later, clamped to the month's last day (the "30 days = 1 month" slip)."""
    y, m = divmod(d.month - 1 + months, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, last_day_of_month(y, m).day))


def render_deadline_choice(tpl: int, ref: str, side: date, styles: dict, d: Deadline) -> str:
    st = styles
    S, Y = fmt(d.start, st["start"]), fmt(side, st["side"])
    hol = holiday_sentence(d.holidays, st["start"])
    note = f" {BUSINESS_NOTE}" if d.unit == "business" else ""
    if tpl == 0:
        text = (f"Invoice {ref} is dated {S} and reached accounts payable on {Y}. Payment is due {unit_phrase(d)} "
                f"from the invoice date, and {count_rule(d, 'the invoice date')}.{note}{hol}")
    elif tpl == 1:
        text = (f"Purchase order {ref}: goods accepted on {S}. The supplier's paperwork arrived on {Y}. The buyer "
                f"must pay within {unit_phrase(d)} of acceptance; {count_rule(d, 'acceptance')}.{note}{hol}")
    else:
        text = (f"A formal notice was served on {S}, and a copy was emailed on {Y}. The recipient must reply "
                f"within {unit_phrase(d)} of service, and {count_rule(d, 'service')}.{note}{hol}")
    return with_slash_note(text, st.values())


def deadline_choice_cluster(rng: random.Random, cid: str) -> list[Decision]:
    d = sample_deadline(rng)
    tpl, ref = rng.randrange(3), f"{rng.randint(100, 99999)}"
    styles = {k: pick_style(rng) for k in ("start", "side", "options")}
    side = d.start + timedelta(days=rng.randint(1, 4))
    variants = [(replace(d, start=d.start + timedelta(days=1)), "date"), (replace(d, n=d.n + 1), "date"),
                (replace(d, n=d.n - 1), "date"), (swap_unit(d), "unit")]
    rows = [(d, None)] + rng.sample(variants, 2)
    golds = {dd.due() for dd, _ in rows}
    due = d.due()
    distractors = [due - timedelta(days=1), due + timedelta(days=1), swap_unit(d).due(),
                   replace(d, start=side).due(), due + timedelta(days=2), due - timedelta(days=2)]
    if d.unit == "calendar" and d.n % 30 == 0:
        distractors.insert(0, add_months(d.start, d.n // 30))
    k = rng.randint(4, 6)
    dates = set(golds)
    for x in distractors:
        if len(dates) >= k:
            break
        dates.add(x)
    ordered = sorted(dates)
    opts = [Option(f"date_{i + 1}", fmt(x, styles["options"])) for i, x in enumerate(ordered)]
    key_of = {x: f"date_{i + 1}" for i, x in enumerate(ordered)}
    question = rng.choice(("Which date is the deadline?", "On which date does the deadline fall?"))
    members = [(render_deadline_choice(tpl, ref, side, styles, dd), opts, key_of[dd.due()], edit) for dd, edit in rows]
    metas = [{"kind": "deadline_choice", **dd.meta(), "option_style": styles["options"]} for dd, _ in rows]
    return _cluster(cid, "choice", question, members, metas)


# ---------------------------------------------------------------- kind 3: business days between two dates


def day_text(d: date, style: str, weekday: bool) -> str:
    return f"{d:%A}, {fmt(d, style)}" if weekday else fmt(d, style)


def render_business_days(tpl: int, ref: str, styles: dict, weekday: bool, a: date, b: date, holidays) -> str:
    A, B = day_text(a, styles["a"], weekday), day_text(b, styles["b"], weekday)
    hol = ("".join(f" {fmt(h, styles['a'])} was a public holiday." for h in holidays)
           or " There were no public holidays in this period.")
    if tpl == 0:
        text = (f"Ticket {ref} was opened on {A} and resolved on {B}.{hol} Business days are weekdays after the "
                f"opening date, up to and including the resolution date, skipping public holidays; the first "
                f"business day after opening {COUNT_PHRASE}.")
    elif tpl == 1:
        text = (f"Shipment {ref} was handed to the carrier on {A} and delivered on {B}.{hol} Transit time is "
                f"counted in business days (Monday to Friday, holidays excluded), not counting the hand-over day: "
                f"the first business day after hand-over {COUNT_PHRASE}.")
    else:
        text = (f"Contractor timesheet {ref}: the engagement ran from the day after {A} through {B}.{hol} The "
                f"contractor bills every weekday in that period except public holidays; the first billable weekday "
                f"{COUNT_PHRASE}.")
    return with_slash_note(text, styles.values())


def business_days_cluster(rng: random.Random, cid: str) -> list[Decision]:
    a = random_day(rng)
    b = a + timedelta(days=rng.randint(4, 30))
    inside = [a + timedelta(days=k) for k in range(1, (b - a).days + 1) if (a + timedelta(days=k)).weekday() < 5]
    hol0 = (rng.choice(inside),) if rng.random() < 0.35 else ()
    tpl, ref, weekday = rng.randrange(3), f"{rng.randint(1000, 99999)}", rng.random() < 0.5
    styles = {"a": pick_style(rng), "b": pick_style(rng)}
    if hol0:
        flip = (a, b, (), "exception")
    else:
        flip = (a, b, (rng.choice(inside),), "exception")
    next_b = b + timedelta(days=1)
    while next_b.weekday() >= 5:
        next_b += timedelta(days=1)
    prev_a = a - timedelta(days=1)
    while prev_a.weekday() >= 5:
        prev_a -= timedelta(days=1)
    rows = [(a, b, hol0, None), flip] + rng.sample([(a, next_b, hol0, "date"), (prev_a, b, hol0, "date")], 1)
    counts = [business_days(x, y, h) for x, y, h, _ in rows]
    lo, hi = min(counts), max(counts)
    k = rng.randint(max(4, hi - lo + 1), 6)
    while hi - lo + 1 < k:
        if lo > 0 and rng.random() < 0.5:
            lo -= 1
        else:
            hi += 1
    unit = lambda n: f"{n} business day{'s' if n != 1 else ''}"  # noqa: E731
    opts = [Option(f"count_{i + 1}", unit(n)) for i, n in enumerate(range(lo, hi + 1))]
    question = rng.choice(("How many business days does this period cover?",
                           "How many business days are counted?"))
    members = [(render_business_days(tpl, ref, styles, weekday, x, y, h), opts, f"count_{n - lo + 1}", edit)
               for (x, y, h, edit), n in zip(rows, counts)]
    metas = [{"kind": "business_days", "start": x.isoformat(), "end": y.isoformat(),
              "holidays": [d.isoformat() for d in h]} for x, y, h, _ in rows]
    return _cluster(cid, "choice", question, members, metas)


# ---------------------------------------------------------------- kind 4: sum against a threshold


@dataclass(frozen=True)
class SumDomain:
    items: tuple[str, ...]
    thresholds: tuple[int, ...]  # in minor units
    step: int  # smallest change, in minor units
    show: object  # minor units -> text
    intro: str
    rule_strict: str
    rule_inclusive: str
    excluded: str
    question: str


def _decimal(scale: int, suffix: str):
    return lambda v: f"{v // scale:,}.{v % scale:0{len(str(scale)) - 1}d} {suffix}"


DOMAINS = (
    SumDomain(("hotel", "flights", "taxis", "meals", "conference fee", "train tickets", "parking", "printing"),
              (50000, 100000, 150000, 250000), 1, money, "Expense claim {ref}:",
              "Claims above {t} need director approval.", "Claims of {t} or more need director approval.",
              "refunded by the vendor, not part of this claim", "Does this claim need director approval?"),
    SumDomain(("suitcase", "duffel bag", "ski bag", "box of books", "golf bag"), (230, 320, 460), 1,
              _decimal(10, "kg"), "Checked luggage for booking {ref}:",
              "The checked allowance is {t} in total; anything heavier is charged.",
              "A surcharge applies once the checked total reaches {t}.",
              "moved to the cabin, not checked", "Will the passenger pay a baggage surcharge?"),
    SumDomain(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"), (4000, 4500), 25,
              _decimal(100, "h"), "Timesheet {ref}, hours worked:",
              "Overtime is paid only for weeks above {t}.", "Overtime is paid for weeks of {t} or more.",
              "logged in error and voided", "Is overtime paid for this week?"),
    SumDomain(("phone", "tablet", "laptop hotspot", "smartwatch"), (500, 1000, 2000), 1, _decimal(10, "GB"),
              "Shared data plan {ref}, usage this cycle:",
              "The plan includes {t}; extra charges start only above that.",
              "Extra charges start as soon as usage reaches {t}.",
              "zero-rated streaming, not counted", "Will this cycle incur extra data charges?"),
)


def split_total(rng: random.Random, total: int, k: int, step: int) -> list[int]:
    units = total // step
    cuts = sorted(rng.sample(range(1, units), k - 1))
    parts = [b - a for a, b in zip([0] + cuts, cuts + [units])]
    return [p * step for p in parts]


def render_sum(dom: SumDomain, ref: str, labels, amounts, counted, strict: bool, threshold: int, when: str) -> str:
    lines = [dom.intro.format(ref=ref)]
    for name, v, c in zip(labels, amounts, counted):
        lines.append(f"- {name}: {dom.show(v)}" + ("" if c else f" ({dom.excluded})"))
    rule = (dom.rule_strict if strict else dom.rule_inclusive).format(t=dom.show(threshold))
    lines.append(f"Submitted {when}. {rule}")
    return "\n".join(lines)


def sum_threshold_cluster(rng: random.Random, cid: str) -> list[Decision]:
    dom = rng.choice(DOMAINS)
    k = rng.randint(3, min(6, len(dom.items)))
    labels = rng.sample(dom.items, k) if dom.items[0] != "Monday" else list(dom.items[:k])
    threshold = rng.choice(dom.thresholds)
    amounts = split_total(rng, threshold, k, dom.step)
    strict = rng.random() < 0.5
    ref, when = f"{rng.randint(1000, 99999)}", fmt(random_day(rng), pick_style(rng))
    when_note = ["slash_dmy"] if "/" in when else []
    counted = [True] * k
    extra_amount = rng.randint(1, max(2, threshold // (4 * dom.step))) * dom.step
    unused = [x for x in dom.items if x not in labels]
    if unused and rng.random() < 0.4:  # a distractor line that is listed but not counted
        labels, amounts, counted = labels + [rng.choice(unused)], amounts + [extra_amount], counted + [False]
    i = rng.randrange(k)

    def bump(delta: int) -> list[int]:
        return [v + delta if j == i else v for j, v in enumerate(amounts)]

    flip_delta = dom.step if strict else -dom.step
    rows = [(amounts, counted, threshold, None), (bump(flip_delta), counted, threshold, "threshold")]
    optional = [(bump(-flip_delta), counted, threshold, "threshold"),
                (amounts, counted, threshold - flip_delta, "threshold"),
                (amounts, [c and j != i for j, c in enumerate(counted)], threshold, "negation")]
    rows += rng.sample(optional, rng.randint(1, 2))
    question = dom.question

    def gold(am, co, t) -> str:
        total = sum(v for v, c in zip(am, co) if c)
        return "yes" if (total > t if strict else total >= t) else "no"

    members = [(with_slash_note(render_sum(dom, ref, labels, am, co, strict, t, when), when_note), NOUL_OPTIONS,
                gold(am, co, t), edit) for am, co, t, edit in rows]
    metas = [{"kind": "sum_threshold", "amounts": list(am), "counted": list(co), "threshold": t, "strict": strict}
             for am, co, t, _ in rows]
    return _cluster(cid, "noul", question, members, metas)


# ---------------------------------------------------------------- kind 5: percentage change band

BAND_SETS = ((-10, 0, 10), (-20, -5, 5, 20), (0, 10, 25, 50), (-50, -25, 0, 25, 50), (-15, 0, 15, 30))
METRICS = ("monthly active users", "support tickets", "website sessions", "orders shipped", "new sign-ups",
           "returns processed")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


def band_options(cuts) -> list[Option]:
    def pct(c):
        return f"{c:+d}%" if c else "0%"

    opts = [Option("0", f"A change below {pct(cuts[0])}")]
    for i in range(1, len(cuts)):
        opts.append(Option(str(i), f"A change from {pct(cuts[i - 1])} up to, but not including, {pct(cuts[i])}"))
    opts.append(Option(str(len(cuts)), f"A change of {pct(cuts[-1])} or more"))
    return opts


def band_of(cuts, old: int, new: int) -> str:
    change = Fraction(new - old, old) * 100
    return str(sum(1 for c in cuts if change >= c))


def render_pct(tpl: int, metric: str, team: str, months, old: int, new: int, before: int, after: int) -> str:
    m0, m1, m2, m3 = months
    if tpl == 0:
        return (f"{metric.capitalize()} for the {team} team: {old:,} in {m1} and {new:,} in {m2}. "
                f"For comparison, {m0} had {before:,}.")
    if tpl == 1:
        return (f"{metric.capitalize()} by month ({team})\n{m0}: {before:,}\n{m1}: {old:,}\n{m2}: {new:,}\n"
                f"{m3} (forecast): {after:,}")
    return (f"From the {team} monthly report: {metric} came in at {old:,} for {m1}; the {m2} figure is {new:,}. "
            f"The target for {m3} is {after:,}.")


def pct_change_cluster(rng: random.Random, cid: str) -> list[Decision]:
    cuts = rng.choice(BAND_SETS)
    old = 100 * rng.randint(3, 120)
    cut = rng.choice(cuts)
    new = old * (100 + cut) // 100
    metric, team, tpl = rng.choice(METRICS), rng.choice(("EMEA", "APAC", "retail", "mobile", "partner")), rng.randrange(3)
    start = rng.randrange(12)
    months = tuple(MONTHS[(start + j) % 12] for j in range(4))
    before, after = (int(old * rng.uniform(0.6, 1.4)) for _ in range(2))
    rows = [(old, new, None), (old, new - 1, "threshold")]
    rows += rng.sample([(old, new + 1, "threshold"), (old + 1, new, "threshold"), (old - 1, new, "threshold")],
                       rng.randint(1, 2))
    question = f"Which band does the percentage change in {metric} from {months[1]} to {months[2]} fall into?"
    opts = band_options(cuts)
    members = [(render_pct(tpl, metric, team, months, o, n, before, after), opts, band_of(cuts, o, n), edit)
               for o, n, edit in rows]
    metas = [{"kind": "pct_change", "old": o, "new": n, "cuts": list(cuts)} for o, n, _ in rows]
    return _cluster(cid, "score", question, members, metas)


# ---------------------------------------------------------------- assembly


def _cluster(cid: str, qtype: str, question: str, members, metas) -> list[Decision]:
    out = make_cluster(cluster_id=cid, family=FAMILY, source=SOURCE, qtype=qtype, question=question, members=members)
    for d, m in zip(out, metas):
        d.metadata = m
    return out


BUILDERS = {
    "deadline_met": deadline_met_cluster,
    "deadline_choice": deadline_choice_cluster,
    "business_days": business_days_cluster,
    "sum_threshold": sum_threshold_cluster,
    "pct_change": pct_change_cluster,
}


def generate(n_clusters: int, seed: int) -> list[Decision]:
    rng = random.Random(f"date_number_drills:{seed}")
    out: list[Decision] = []
    for k in range(n_clusters):
        kind = KINDS[k % len(KINDS)] if k < len(KINDS) else rng.choice(KINDS)
        out.extend(BUILDERS[kind](rng, f"date_number_drills:{seed}:{k}"))
    return keep_valid_clusters(out)
