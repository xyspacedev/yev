"""Rule-based returns-policy clusters: labels are computed, values sit on the boundaries."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

from jeff.generators.common import make_cluster
from jeff.schema import Decision, Option

SOURCE = "synthetic_rules"
FAMILY = "returns_policy"
QUESTION = "Which outcome does the shop's returns policy give this request?"
SHOPS = ["Harbor & Pine", "Northgate Supply", "Kiln Street Home", "Brightfield Outdoors", "Tallow & Wick",
         "Quarry Lane Goods", "Juniper Row", "Saltmarsh Trading"]
PRODUCTS = ["wool blanket", "espresso grinder", "pair of trail shoes", "desk lamp", "bluetooth speaker",
            "cast-iron skillet", "yoga mat", "mechanical keyboard", "rain jacket", "stand mixer"]


@dataclass(frozen=True)
class ReturnPolicy:
    window: int  # days after delivery that qualify for a full refund
    grace: int  # further days that qualify for store credit
    opened_ok: bool  # whether opened items can still get a full refund


@dataclass(frozen=True)
class ReturnCase:
    days: int  # days between delivery and the return request
    opened: bool
    final_sale: bool


def return_decision(p: ReturnPolicy, c: ReturnCase) -> str:
    if c.final_sale:
        return "decline"
    if c.days <= p.window and (p.opened_ok or not c.opened):
        return "full_refund"
    if c.days <= p.window + p.grace:
        return "store_credit"
    return "decline"


def policy_options(p: ReturnPolicy) -> list[Option]:
    condition = "" if p.opened_ok else " and the item is unopened"
    return [
        Option("full_refund", f"Full refund: asked within {p.window} days of delivery{condition}; final-sale items excluded."),
        Option("store_credit", f"Store credit: no refund applies, but asked within {p.window + p.grace} days of delivery; final-sale items excluded."),
        Option("decline", f"Decline: the item was sold as final sale, or the request came more than {p.window + p.grace} days after delivery."),
    ]


def _fmt(d: date) -> str:
    return f"{d.day} {d:%B %Y}"


def render_state(template: int, shop: str, product: str, delivered: date, c: ReturnCase) -> str:
    requested = delivered + timedelta(days=c.days)
    condition = "opened and used" if c.opened else "unopened, still in its original packaging"
    sale = "It was bought in a clearance sale marked final sale." if c.final_sale else "It was bought at full price."
    if template == 0:
        return (f"{shop} order: {product}. Delivered {_fmt(delivered)}. Return requested {_fmt(requested)}, "
                f"{c.days} days after delivery. The item is {condition}. {sale}")
    return (f"Return request at {shop} for a {product}, {condition}. {sale} It arrived on {_fmt(delivered)} and the "
            f"customer asked to send it back on {_fmt(requested)} ({c.days} days after delivery).")


def generate_cluster(rng: random.Random, cluster_id: str) -> list[Decision]:
    p = ReturnPolicy(window=rng.randint(7, 90), grace=rng.randint(3, 30),
                     opened_ok=rng.random() < 0.3)
    shop, product = rng.choice(SHOPS), rng.choice(PRODUCTS)
    delivered = date(2025, 1, 1) + timedelta(days=rng.randrange(330))
    template = rng.randrange(2)
    on_refund_edge = rng.random() < 0.6
    base = ReturnCase(days=p.window if on_refund_edge else p.window + p.grace, opened=False, final_sale=False)
    required = (replace(base, days=base.days + 1), "date")
    optional = [
        (replace(base, days=base.days - 1), "date"),
        (replace(base, opened=True), "negation"),
        (replace(base, final_sale=True), "exception"),
    ]
    chosen = [required] + rng.sample(optional, rng.randint(1, 2))
    options = policy_options(p)
    base_state = render_state(template, shop, product, delivered, base)
    members = [(base_state, options, return_decision(p, base), None)]
    for case, edit in chosen:
        members.append((render_state(template, shop, product, delivered, case), options, return_decision(p, case), edit))
    if rng.random() < 0.3:
        p2 = replace(p, window=p.window - 1) if on_refund_edge else replace(p, grace=p.grace - 1)
        members.append((base_state, policy_options(p2), return_decision(p2, base), "policy_edit"))
    return make_cluster(cluster_id=cluster_id, family=FAMILY, source=SOURCE, qtype="choice",
                        question=QUESTION, members=members)


def generate(n: int, seed: int) -> list[Decision]:
    rng = random.Random(f"returns:{seed}")
    out: list[Decision] = []
    for k in range(n):
        out.extend(generate_cluster(rng, f"returns:{seed}:{k}"))
    return out
