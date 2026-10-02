"""Returns-policy clusters where the policy text and the dates sit in the state (Stage 1, family 1).

Stage 0's returns data always printed "N days after delivery", so the model never had to count days
itself. Here the state holds only the policy text and calendar dates, in mixed styles, and the label is
computed from the dates. Windows lean long (>= 30 days), some policies have no store-credit tier, some
count from purchase rather than delivery, and the request date is placed exactly on the window edge
(± 1 day), across month ends, across Dec -> Jan, and across Feb 28/29 in leap and non-leap years.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

from yev.generators.common import keep_valid_clusters, make_cluster
from yev.generators.dates import STYLES, elapsed_days, fmt, last_day_of_month
from yev.schema import Decision, Option

SOURCE = "synthetic_rules"
FAMILY = "returns_v2"
QUESTIONS = (
    "Following the returns policy in the text, how should this return request be resolved?",
    "Under the stated policy, what outcome does this return get?",
    "Which outcome does the quoted returns policy give for this request?",
)
SHOPS = ["Harbor & Pine", "Northgate Supply", "Kiln Street Home", "Brightfield Outdoors", "Tallow & Wick",
         "Quarry Lane Goods", "Juniper Row", "Saltmarsh Trading", "Copper Fern", "Lantern Bay Outfitters",
         "Mossgrove", "Redwick Electronics"]
PRODUCTS = ["wool blanket", "espresso grinder", "pair of trail shoes", "desk lamp", "bluetooth speaker",
            "cast-iron skillet", "yoga mat", "mechanical keyboard", "rain jacket", "stand mixer",
            "set of bed sheets", "robot vacuum", "camping stove", "pair of headphones", "office chair"]
WINDOWS = (14, 21, 30, 30, 30, 45, 45, 60, 60, 60, 90, 90, 100, 120, 180, 365)
CREDIT_EXTRA = (14, 15, 30, 30, 60, 90)
EDGE_PLACES = ("month_end", "year_end", "feb_leap", "feb_nonleap", "random")
SLASH_NOTE = "Slash dates are written day/month/year."


@dataclass(frozen=True)
class Policy:
    window: int  # days from the anchor date that qualify for a full refund
    credit_days: int | None  # days from the anchor date that qualify for store credit; None = no credit tier
    opened_ok: bool  # whether opened items can still get a full refund
    anchor: str  # "delivery" or "purchase": the date the windows count from


@dataclass(frozen=True)
class Case:
    purchased: date
    delivered: date
    requested: date
    opened: bool
    final_sale: bool


def anchor_date(p: Policy, c: Case) -> date:
    return c.delivered if p.anchor == "delivery" else c.purchased


def decide(p: Policy, c: Case) -> str:
    if c.final_sale:
        return "deny"
    days = elapsed_days(anchor_date(p, c), c.requested)
    if days <= p.window and (p.opened_ok or not c.opened):
        return "refund"
    if p.credit_days is not None and days <= p.credit_days:
        return "store_credit"
    return "deny"


def options(p: Policy) -> list[Option]:
    opts = [Option("refund", "Refund: accept the return and refund the purchase price.")]
    if p.credit_days is not None:
        opts.append(Option("store_credit", "Store credit: accept the return for store credit, not a refund."))
    opts.append(Option("deny", "Deny: refuse the return."))
    return opts


# ---------------------------------------------------------------- rendering


def render_policy(tpl: int, shop: str, p: Policy) -> str:
    a, w, c = p.anchor, p.window, p.credit_days
    if tpl == 0:
        opened = ", opened or not" if p.opened_ok else ", provided they are unopened"
        credit = (f"Requests that miss the refund terms but are made within {c} days of {a} get store credit instead."
                  if c else "Anything outside the refund terms is declined; we do not issue store credit.")
        return (f"{shop} returns policy\n- Items returned within {w} days of {a} get a full refund{opened}.\n"
                f"- {credit}\n- Day 1 is the day after {a}.\n- Nothing marked final sale can come back, whether for a refund or an exchange.")
    if tpl == 1:
        opened = ", even if you have opened it" if p.opened_ok else " as long as it has not been opened"
        credit = (f"If a return misses the refund terms, store credit is available as long as it is requested "
                  f"no later than {c} days from {a}." if c
                  else "There is no store-credit option: returns that miss the refund terms are refused.")
        return (f"Return terms at {shop}: you can send an item back for a full refund for up to {w} days from "
                f"{a}{opened} (day 1 is the day after {a}). {credit} Final-sale purchases are not eligible for any return.")
    if tpl == 2:
        opened = " (opened items are fine)" if p.opened_ok else " (unopened items only)"
        credit = (f"Up to {c} days from {a}, we can still give you store credit." if c
                  else "Then we cannot accept the return, and we do not offer store credit.")
        return (f"{shop} FAQ: Returns\nQ: How long do I have to return something?\nA: {w} days from {a} for a "
                f"full refund{opened}. Day 1 is the day after {a}.\nQ: What if I miss that?\nA: {credit}\n"
                f"Q: Are sale items covered?\nA: Not if they were marked final sale.")
    opened = ", opened items allowed" if p.opened_ok else ", item must be unopened"
    credit = (f"Store-credit window = {c} days from {a}, used when the refund rule fails." if c
              else "No store credit; if the refund rule fails, deny.")
    return (f"Agent guidance ({shop}): refund window = {w} days from {a}{opened}; day 1 = the day after {a}. "
            f"{credit} Final sale = no return.")


@dataclass(frozen=True)
class Scene:
    """Everything about a cluster's text that stays fixed across its members."""

    shop: str
    product: str
    order_no: str
    policy_tpl: int
    ticket_tpl: int
    policy_first: bool
    styles: dict
    aside: date  # an unrelated date (a price-match question, a newsletter sign-up)
    shipped_lag: int
    question: str


def render_ticket(s: Scene, c: Case) -> str:
    st = s.styles
    P, D, R = fmt(c.purchased, st["purchased"]), fmt(c.delivered, st["delivered"]), fmt(c.requested, st["requested"])
    S = fmt(c.purchased + timedelta(days=s.shipped_lag), st["shipped"])
    X = fmt(s.aside, st["aside"])
    cond = ("The box has been opened and the item used." if c.opened
            else "The item is unopened and still sealed.")
    sale = "It was bought as a final-sale clearance item." if c.final_sale else "It was bought at the regular price."
    if s.ticket_tpl == 0:
        text = (f"Order #{s.order_no}: {s.product}\nPurchased: {P}\nShipped: {S}\nDelivered: {D}\n"
                f"Return requested: {R}\n{cond} {sale}")
    elif s.ticket_tpl == 1:
        text = (f"Message received {R} about order {s.order_no}: the customer wants to return the {s.product}. "
                f"Our records show the order was purchased on {P} and delivered on {D}. {cond} {sale} "
                f"They also asked whether a price match from {X} still applies.")
    elif s.ticket_tpl == 2:
        text = (f"Today is {R}. A customer wants to return a {s.product} (order {s.order_no}). It was bought on "
                f"{P}, shipped on {S}, and arrived on {D}. {cond} {sale}")
    else:
        text = (f"Return ticket RT-{s.order_no}\nItem: {s.product}\nPurchase date: {P}\nCourier delivery scan: {D}\n"
                f"Newsletter sign-up: {X}\nRequest logged: {R}\nNotes: {cond} {sale}")
    if "slash_dmy" in st.values():
        text += f"\n{SLASH_NOTE}"
    return text


def render_state(s: Scene, p: Policy, c: Case) -> str:
    policy, ticket = render_policy(s.policy_tpl, s.shop, p), render_ticket(s, c)
    return f"{policy}\n\n{ticket}" if s.policy_first else f"{ticket}\n\nRelevant policy:\n{policy}"


def metadata(s: Scene, p: Policy, c: Case) -> dict:
    return {
        "window": p.window, "credit_days": p.credit_days, "opened_ok": p.opened_ok, "anchor": p.anchor,
        "purchased": c.purchased.isoformat(), "delivered": c.delivered.isoformat(),
        "requested": c.requested.isoformat(), "opened": c.opened, "final_sale": c.final_sale,
        "styles": dict(s.styles),
    }


# ---------------------------------------------------------------- sampling


def edge_date(rng: random.Random, place: str) -> date:
    """A calendar date near a hard boundary: the last day the window allows lands here."""
    if place == "month_end":
        return last_day_of_month(rng.randint(2023, 2028), rng.randint(1, 12)) + timedelta(days=rng.randint(-1, 1))
    if place == "year_end":
        return date(rng.randint(2023, 2027), 12, 31) + timedelta(days=rng.randint(-1, 2))
    if place == "feb_leap":
        return date(rng.choice((2024, 2028)), 2, 28) + timedelta(days=rng.randint(0, 2))
    if place == "feb_nonleap":
        return date(rng.choice((2023, 2025, 2026, 2027)), 2, 27) + timedelta(days=rng.randint(0, 2))
    return date(2023, 1, 1) + timedelta(days=rng.randrange(6 * 365))


def sample_styles(rng: random.Random) -> dict:
    fields = ("purchased", "delivered", "requested", "shipped", "aside")
    if rng.random() < 0.35:
        return {f: "iso" for f in fields}
    weights = (3, 2, 2, 1)
    return {f: rng.choices(STYLES, weights)[0] for f in fields}


def generate_cluster(rng: random.Random, cluster_id: str) -> list[Decision]:
    window = rng.choice(WINDOWS)
    credit = window + rng.choice(CREDIT_EXTRA) if rng.random() < 0.55 else None
    p = Policy(window=window, credit_days=credit, opened_ok=rng.random() < 0.3,
               anchor="purchase" if rng.random() < 0.25 else "delivery")
    on_credit_edge = credit is not None and rng.random() < 0.45
    edge = credit if on_credit_edge else window
    deadline = edge_date(rng, rng.choice(EDGE_PLACES))
    start, lag = deadline - timedelta(days=edge), rng.randint(1, 9)
    purchased, delivered = (start - timedelta(days=lag), start) if p.anchor == "delivery" else (start, start + timedelta(days=lag))
    scene = Scene(
        shop=rng.choice(SHOPS), product=rng.choice(PRODUCTS), order_no=str(rng.randint(10_000, 9_999_999)),
        policy_tpl=rng.randrange(4), ticket_tpl=rng.randrange(4), policy_first=rng.random() < 0.7,
        styles=sample_styles(rng), aside=purchased - timedelta(days=rng.randint(20, 400)),
        shipped_lag=rng.randint(0, lag - 1) if lag > 1 else 0, question=rng.choice(QUESTIONS),
    )
    base = Case(purchased=purchased, delivered=delivered, requested=deadline, opened=False, final_sale=False)

    variants: list[tuple[Policy, Case, str]] = [(p, replace(base, requested=deadline + timedelta(days=1)), "date")]
    optional = [
        (p, replace(base, requested=deadline - timedelta(days=1)), "date"),
        (p, replace(base, final_sale=True), "exception"),
        (replace(p, credit_days=credit - 1) if on_credit_edge else replace(p, window=window - 1), base, "date"),
    ]
    if not on_credit_edge:
        optional.append((p, replace(base, opened=True), "negation"))
    if p.anchor == "delivery":
        optional.append((replace(p, anchor="purchase"), base, "entity_swap"))
    variants += rng.sample(optional, rng.randint(2, 3))

    opts = options(p)
    rows = [(p, base, None)] + variants
    members = [(render_state(scene, vp, vc), opts, decide(vp, vc), edit) for vp, vc, edit in rows]
    out = make_cluster(cluster_id=cluster_id, family=FAMILY, source=SOURCE, qtype="choice",
                       question=scene.question, members=members)
    for d, (vp, vc, _) in zip(out, rows):
        d.metadata = metadata(scene, vp, vc)
    return out


def generate(n_clusters: int, seed: int) -> list[Decision]:
    rng = random.Random(f"returns_v2:{seed}")
    out: list[Decision] = []
    for k in range(n_clusters):
        out.extend(generate_cluster(rng, f"returns_v2:{seed}:{k}"))
    return keep_valid_clusters(out)
