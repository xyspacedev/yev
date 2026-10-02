"""Rule-based bug-severity clusters on an ordered four-point scale."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from yev.generators.common import make_cluster
from yev.schema import Decision, Option

SOURCE = "synthetic_rules"
FAMILY = "triage"
QUESTION = "What severity should this report get?"
SCALE = ("low", "medium", "high", "critical")
TEAMS = ["payments", "mobile", "search", "checkout", "notifications", "admin console"]
SYMPTOMS = ["the save button does nothing", "pages time out after login", "totals show the wrong currency",
            "uploads stall at 99%", "push alerts arrive twice", "exports come back empty"]


@dataclass(frozen=True)
class SeverityPolicy:
    medium: int
    high: int
    critical: int


@dataclass(frozen=True)
class SeverityCase:
    env: str
    users: int
    data_loss: bool
    workaround: bool


def severity_decision(p: SeverityPolicy, c: SeverityCase) -> str:
    if c.env != "production":
        return "low"
    if c.data_loss or (c.users >= p.critical and not c.workaround):
        return "critical"
    if c.users >= p.critical or (c.users >= p.high and not c.workaround):
        return "high"
    if c.users >= p.high or c.users >= p.medium:
        return "medium"
    return "low"


def policy_options(p: SeverityPolicy) -> list[Option]:
    return [
        Option("low", f"Low: outside production, or fewer than {p.medium} users affected with no data lost."),
        Option("medium", f"Medium: {p.medium}+ users affected, or {p.high}+ users when a workaround exists."),
        Option("high", f"High: {p.high}+ users with no workaround, or {p.critical}+ users when a workaround exists."),
        Option("critical", f"Critical: any data lost, or {p.critical}+ users affected with no workaround."),
    ]


def render_state(team: str, symptom: str, c: SeverityCase) -> str:
    loss = "Some saved records were lost." if c.data_loss else "No data was lost."
    workaround = "Support has a workaround: retry from the web app." if c.workaround else "There is no known workaround."
    return f"Bug in {team} ({c.env}): {symptom}. About {c.users} users are affected. {loss} {workaround}"


def _base_case(rng: random.Random, p: SeverityPolicy) -> tuple[SeverityCase, str, int]:
    """Pick which threshold the base sits on. Returns (case, threshold field, users - threshold)."""
    field = rng.choice(["medium", "high", "critical"])
    boundary = getattr(p, field)
    if field == "medium":
        # Below the medium threshold half the time, so "low" bases exist; the workaround is irrelevant there.
        offset = -1 if rng.random() < 0.5 else 0
        return SeverityCase("production", boundary + offset, False, True), field, offset
    workaround = rng.random() < 0.5
    offset = 0
    if rng.random() < 0.2:  # well inside the band: >= 10 users from every threshold
        ceiling = {"high": p.critical, "critical": p.critical + 500}[field]
        offset = rng.randint(10, max(10, min(ceiling - boundary - 10, 300)))
    return SeverityCase("production", boundary + offset, False, workaround), field, offset


def generate_cluster(rng: random.Random, cluster_id: str) -> list[Decision]:
    p = SeverityPolicy(medium=rng.randint(3, 40), high=rng.randint(60, 900),
                       critical=rng.randint(1000, 9000))
    team, symptom = rng.choice(TEAMS), rng.choice(SYMPTOMS)
    base, field, offset = _base_case(rng, p)
    boundary = getattr(p, field)
    if offset > 0:  # well inside: the threshold variants move the count across the nearest threshold
        required = (replace(base, users=boundary - 1), "threshold")
        users_up = (replace(base, users=boundary), "threshold")
    else:
        required = (replace(base, users=base.users - 1), "threshold")
        users_up = (replace(base, users=base.users + 1), "threshold")
    optional = [users_up, (replace(base, workaround=not base.workaround), "negation"),
                (replace(base, env="staging"), "entity_swap"), (replace(base, data_loss=True), "negation")]
    options = policy_options(p)
    base_gold = severity_decision(p, base)
    while True:
        chosen = [required] + rng.sample(optional, rng.randint(1, 2))
        if len({base_gold} | {severity_decision(p, c) for c, _ in chosen}) >= 2:
            break
    base_state = render_state(team, symptom, base)
    members = [(base_state, options, base_gold, None)]
    for case, edit in chosen:
        members.append((render_state(team, symptom, case), options, severity_decision(p, case), edit))
    if rng.random() < 0.3:
        # Move the threshold the base sits on by one: below-medium bases move down, others up.
        shift = -1 if (field == "medium" and offset < 0) else 1
        p2 = replace(p, **{field: boundary + shift})
        members.append((base_state, policy_options(p2), severity_decision(p2, base), "policy_edit"))
    return make_cluster(cluster_id=cluster_id, family=FAMILY, source=SOURCE, qtype="score",
                        question=QUESTION, members=members)


def generate(n: int, seed: int) -> list[Decision]:
    rng = random.Random(f"severity:{seed}")
    out: list[Decision] = []
    for k in range(n):
        out.extend(generate_cluster(rng, f"severity:{seed}:{k}"))
    return out
