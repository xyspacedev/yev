"""Score questions on numbered scales, keys "0".."n-1" in ascending order (Stage 1, family 4).

Stage 0 served WorkflowEvals score questions with numeric keys, while only a few percent of the
training score rows used them. This family supplies them in four kinds:

- severity: the `severity` clusters (bug reports on a four-point scale) re-rendered with keys
  "0".."3", option descriptions kept word for word;
- risk: likelihood x impact (each rated 1-5) against 3-7 score bands;
- priority: days overdue plus points per blocked team against 3-7 point bands;
- percent_complete: done/total tasks against 3-7 percentage bands.

Bands are half-open and stated in the option text; the base row sits on a band edge or one below it,
and every cluster has a member that moves the value across that edge.
"""

from __future__ import annotations

import random
from fractions import Fraction

from yev.generators import severity
from yev.generators.common import keep_valid_clusters, make_cluster
from yev.schema import Decision, Option

SOURCE = "synthetic_rules"
FAMILY = "numbered_scales"
KINDS = ("severity", "risk", "priority", "percent_complete")
KIND_WEIGHTS = (4, 2, 2, 2)
POLICY_EDIT_RATE = 0.3


def _cluster(cid: str, question: str, members, metas) -> list[Decision]:
    out = make_cluster(cluster_id=cid, family=FAMILY, source=SOURCE, qtype="score", question=question,
                       members=members)
    for d, m in zip(out, metas):
        d.metadata = m
    return out


def level_of(cuts, value) -> str:
    """Index of the half-open band [cuts[i-1], cuts[i]) holding value, as a numbered key."""
    return str(sum(1 for c in cuts if value >= c))


def _other_cuts(rng: random.Random, pool, keep: int, n_levels: int) -> list[int]:
    others = [c for c in pool if c != keep]
    return sorted(rng.sample(others, n_levels - 2) + [keep])


# ---------------------------------------------------------------- severity, renumbered


def renumber(rows: list[Decision], cid: str) -> list[Decision]:
    """Severity rows with keys low..critical replaced by "0".."3"; descriptions and golds carried over."""
    key_of = {name: str(i) for i, name in enumerate(severity.SCALE)}
    out = []
    for j, r in enumerate(rows):
        assert [o.key for o in r.options] == list(severity.SCALE), r.keys
        out.append(Decision(
            id=f"{cid}:{j}", type="score", state=r.state, question=r.question,
            options=[Option(key_of[o.key], o.description) for o in r.options], gold=key_of[r.gold],
            family=FAMILY, source=SOURCE, licence=r.licence, cluster_id=cid, edit_type=r.edit_type,
            metadata={"kind": "severity", "orig_gold": r.gold, "orig_id": r.id},
        ))
    return out


def severity_cluster(rng: random.Random, cid: str) -> list[Decision]:
    return renumber(severity.generate_cluster(rng, f"severity:{cid}"), cid)


# ---------------------------------------------------------------- risk: likelihood x impact

RISK_PRODUCTS = sorted({a * b for a in range(1, 6) for b in range(1, 6)} - {1})
HAZARDS = ["the main payment vendor suffers a day-long outage", "a key supplier misses a delivery date",
           "the shared drive fills up during month-end close", "a contractor leaves before the handover",
           "the backup job silently stops running", "a data-centre cooling unit fails",
           "a customer contract renewal slips by a quarter", "the release pipeline is blocked by a licence audit"]
AREAS = ["finance", "operations", "IT", "procurement", "facilities", "sales", "engineering", "legal"]


def risk_options(cuts) -> list[Option]:
    n = len(cuts) + 1
    opts = [Option("0", f"Level 0 (lowest): a risk score below {cuts[0]}.")]
    for i in range(1, n - 1):
        lo, hi = cuts[i - 1], cuts[i] - 1
        span = f"of {lo}" if lo == hi else f"from {lo} to {hi}"
        opts.append(Option(str(i), f"Level {i}: a risk score {span}."))
    opts.append(Option(str(n - 1), f"Level {n - 1} (highest): a risk score of {cuts[-1]} or more."))
    return opts


def render_risk(tpl: int, ref: str, area: str, hazard: str, prev: int, lk: int, im: int) -> str:
    rule = "The risk score is likelihood multiplied by impact."
    if tpl == 0:
        return (f"Risk register entry {ref} ({area}): {hazard}. Likelihood is rated {lk} out of 5 and impact "
                f"{im} out of 5. {rule} At the previous review the score was {prev}.")
    if tpl == 1:
        return (f"Risk {ref}, owned by {area}\nScenario: {hazard}\nLikelihood (1-5): {lk}\nImpact (1-5): {im}\n"
                f"Score last quarter: {prev}\n{rule}")
    return (f"The {area} team logged risk {ref}: {hazard}. They rate impact at {im} and likelihood at {lk}, both "
            f"on a 1-5 scale. {rule}")


def risk_cluster(rng: random.Random, cid: str) -> list[Decision]:
    while True:
        lk, im = rng.randint(1, 5), rng.randint(1, 5)
        moves = [(dl, di) for dl, di in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if 1 <= lk + dl <= 5 and 1 <= im + di <= 5 and (lk + dl) * (im + di) != lk * im]
        if moves:
            break
    dl, di = rng.choice(moves)
    edge = max(lk * im, (lk + dl) * (im + di))
    n_levels = rng.randint(3, 7)
    cuts = _other_cuts(rng, RISK_PRODUCTS, edge, n_levels)
    tpl, ref, area, hazard = rng.randrange(3), f"R-{rng.randint(10, 999)}", rng.choice(AREAS), rng.choice(HAZARDS)
    prev = rng.choice(RISK_PRODUCTS)
    rows = [(lk, im, cuts, None), (lk + dl, im + di, cuts, "threshold")]
    others = [(lk + a, im + b, cuts, "threshold") for a, b in moves if (a, b) != (dl, di)]
    rows += rng.sample(others, min(len(others), rng.randint(1, 2)))
    if rng.random() < POLICY_EDIT_RATE:
        # Move the edge to the next reachable product above the base, or down onto the base itself, so
        # every band still holds a score that can occur.
        above = [p for p in RISK_PRODUCTS if p > edge]
        shifted = (above[0] if above else None) if lk * im >= edge else lk * im
        moved = sorted([c for c in cuts if c != edge] + [shifted]) if shifted else cuts
        if (shifted and shifted not in cuts and shifted > 1
                and level_of(moved, lk * im) != level_of(cuts, lk * im)):
            rows.append((lk, im, moved, "policy_edit"))
    question = rng.choice(("Which risk level does this entry fall into?", "What risk level should this get?"))
    members = [(render_risk(tpl, ref, area, hazard, prev, a, b), risk_options(c), level_of(c, a * b), e)
               for a, b, c, e in rows]
    metas = [{"kind": "risk", "likelihood": a, "impact": b, "cuts": list(c)} for a, b, c, _ in rows]
    return _cluster(cid, question, members, metas)


# ---------------------------------------------------------------- priority points

TEAMS = ["billing", "data platform", "mobile", "identity", "warehouse systems", "analytics", "partner API",
         "design systems"]
TASKS = ["the invoice export fix", "the database index migration", "the SSO certificate rotation",
         "the barcode scanner update", "the quarterly report rebuild", "the rate-limit change",
         "the onboarding email rewrite", "the search ranking patch"]


def priority_options(cuts) -> list[Option]:
    n = len(cuts) + 1
    opts = [Option("0", f"Priority 0 (lowest): fewer than {cuts[0]} points.")]
    for i in range(1, n - 1):
        lo, hi = cuts[i - 1], cuts[i] - 1
        span = f"exactly {lo} points" if lo == hi else f"{lo} to {hi} points"
        opts.append(Option(str(i), f"Priority {i}: {span}."))
    opts.append(Option(str(n - 1), f"Priority {n - 1} (highest): {cuts[-1]} or more points."))
    return opts


def _teams(b: int) -> str:
    return f"{b} other team" if b == 1 else f"{b} other teams"


def render_priority(tpl: int, ref: str, team: str, task: str, per: int, days: int, blocked: int, est: int) -> str:
    rule = f"Priority points = days overdue + {per} for each blocked team."
    if tpl == 0:
        return (f"Ticket {ref} ({task}) from the {team} team is {days} days past its due date and is blocking "
                f"{_teams(blocked)}. {rule} The fix is estimated at {est} hours of work.")
    if tpl == 1:
        return (f"Backlog item {ref}: {task}\nOwner: {team}\nDays overdue: {days}\nTeams blocked by it: {blocked}\n"
                f"Estimate: {est} hours\n{rule}")
    return (f"{task.capitalize()} ({ref}, {team}) is now blocking {_teams(blocked)} and has been overdue for "
            f"{days} days. {rule}")


def priority_cluster(rng: random.Random, cid: str) -> list[Decision]:
    per = rng.choice((2, 3, 5, 10))
    days, blocked = rng.randint(3, 30), rng.randint(1, 5)
    score = days + per * blocked
    at_edge = rng.random() < 0.5
    edge = score if at_edge else score + 1
    n_levels = rng.randint(3, 7)
    cuts = _other_cuts(rng, range(2, days + per * 6 + 15), edge, n_levels)
    tpl, ref, team, task = rng.randrange(3), f"OPS-{rng.randint(100, 9999)}", rng.choice(TEAMS), rng.choice(TASKS)
    est = rng.randint(2, 40)
    step = -1 if at_edge else 1
    rows = [(days, blocked, cuts, None), (days + step, blocked, cuts, "date")]
    optional = [(days - step, blocked, cuts, "date"), (days, blocked + step, cuts, "threshold")]
    rows += rng.sample(optional, rng.randint(1, 2))
    if rng.random() < POLICY_EDIT_RATE:
        shifted = edge + 1 if at_edge else edge - 1
        if shifted not in cuts and shifted > 1:
            rows.append((days, blocked, sorted([c for c in cuts if c != edge] + [shifted]), "policy_edit"))
    question = rng.choice(("Which priority level should this ticket get?", "What priority does this item get?"))
    members = [(render_priority(tpl, ref, team, task, per, d, b, est), priority_options(c),
                level_of(c, d + per * b), e) for d, b, c, e in rows]
    metas = [{"kind": "priority", "days_overdue": d, "blocked_teams": b, "per_team": per, "cuts": list(c)}
             for d, b, c, _ in rows]
    return _cluster(cid, question, members, metas)


# ---------------------------------------------------------------- percent complete

PCT_CUTS = (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90)
TOTALS = (20, 40, 60, 80, 100, 120, 200)
PROJECTS = ["the warehouse move", "the CRM migration", "the office fit-out", "the audit preparation",
            "the website redesign", "the payroll switchover", "the safety training rollout", "the data clean-up"]
UNITS = [("tasks", "done"), ("checklist items", "checked off"), ("modules", "completed"), ("tickets", "closed")]


def pct_options(cuts) -> list[Option]:
    n = len(cuts) + 1
    opts = [Option("0", f"Band 0: less than {cuts[0]}% complete.")]
    for i in range(1, n - 1):
        opts.append(Option(str(i), f"Band {i}: from {cuts[i - 1]}% up to, but not including, {cuts[i]}% complete."))
    opts.append(Option(str(n - 1), f"Band {n - 1}: {cuts[-1]}% complete or more."))
    return opts


def render_pct(tpl: int, project: str, unit: tuple[str, str], people: int, done: int, total: int) -> str:
    noun, verb = unit
    if tpl == 0:
        return f"Status of {project}: {done} of {total} {noun} are {verb}. {people} people are assigned to it."
    if tpl == 1:
        return (f"Weekly update on {project}. The plan has {total} {noun}; {done} are {verb} so far. "
                f"Headcount on the project: {people}.")
    return f"Progress tracker for {project}\n{noun.capitalize()} {verb}: {done}\nTotal {noun}: {total}\nStaff: {people}"


def percent_cluster(rng: random.Random, cid: str) -> list[Decision]:
    total = rng.choice(TOTALS)
    edge = rng.choice([c for c in PCT_CUTS if c * total % 100 == 0])
    at_edge = rng.random() < 0.5
    done = edge * total // 100 - (0 if at_edge else 1)
    n_levels = rng.randint(3, 7)
    cuts = _other_cuts(rng, PCT_CUTS, edge, n_levels)
    tpl, project, unit, people = rng.randrange(3), rng.choice(PROJECTS), rng.choice(UNITS), rng.randint(2, 15)
    step = -1 if at_edge else 1
    rows = [(done, total, cuts, None), (done + step, total, cuts, "threshold")]
    optional = [(done - step, total, cuts, "threshold"), (done, total - step, cuts, "threshold")]
    rows += rng.sample(optional, rng.randint(1, 2))
    question = rng.choice(("Which completion band is this project in?", "How far along is this, by band?"))
    members = [(render_pct(tpl, project, unit, people, d, t), pct_options(c), level_of(c, Fraction(100 * d, t)), e)
               for d, t, c, e in rows]
    metas = [{"kind": "percent_complete", "done": d, "total": t, "cuts": list(c)} for d, t, c, _ in rows]
    return _cluster(cid, question, members, metas)


# ---------------------------------------------------------------- assembly

BUILDERS = {"severity": severity_cluster, "risk": risk_cluster, "priority": priority_cluster,
            "percent_complete": percent_cluster}


def generate(n_clusters: int, seed: int) -> list[Decision]:
    rng = random.Random(f"numbered_scales:{seed}")
    out: list[Decision] = []
    for k in range(n_clusters):
        kind = KINDS[k] if k < len(KINDS) else rng.choices(KINDS, KIND_WEIGHTS)[0]
        out.extend(BUILDERS[kind](rng, f"numbered_scales:{seed}:{k}"))
    return keep_valid_clusters(out)
