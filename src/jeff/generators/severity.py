"""Rule-based bug-severity clusters on an ordered four-point scale."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from jeff.generators.common import make_cluster
from jeff.schema import Decision, Option

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


def generate_cluster(rng: random.Random, cluster_id: str) -> list[Decision]:
    p = SeverityPolicy(medium=rng.randint(3, 40), high=rng.randint(60, 900),
                       critical=rng.randint(1000, 9000))
    team, symptom = rng.choice(TEAMS), rng.choice(SYMPTOMS)
    base = SeverityCase("production", p.high, data_loss=False, workaround=False)
    required = (replace(base, users=p.high - 1), "threshold")
    optional = [(replace(base, workaround=True), "negation"), (replace(base, env="staging"), "entity_swap"),
                (replace(base, data_loss=True), "negation")]
    chosen = [required] + rng.sample(optional, rng.randint(1, 2))
    options = policy_options(p)
    base_state = render_state(team, symptom, base)
    members = [(base_state, options, severity_decision(p, base), None)]
    for case, edit in chosen:
        members.append((render_state(team, symptom, case), options, severity_decision(p, case), edit))
    if rng.random() < 0.3:
        p2 = replace(p, high=p.high + 1)
        members.append((base_state, policy_options(p2), severity_decision(p2, base), "policy_edit"))
    return make_cluster(cluster_id=cluster_id, family=FAMILY, source=SOURCE, qtype="score",
                        question=QUESTION, members=members)


def generate(n: int, seed: int) -> list[Decision]:
    rng = random.Random(f"severity:{seed}")
    out: list[Decision] = []
    for k in range(n):
        out.extend(generate_cluster(rng, f"severity:{seed}:{k}"))
    return out
