"""Workflow-decision planned families (Plan 6, Group D): invoice_line, cs_scales_actions and security_context.

- invoice_line: one review packet, six questions about two of its lines (targets.workflow_invoice).
- cs_scales_actions: one support chat and three questions: urgency and frustration on anchored 0-4 scales (levels
  flat; 30% of groups come as near-neighbour pairs, two groups with the same scene whose briefs differ in one cue
  by one level) and the next step, which the team rules printed in the state decide (three rule variants; the facts
  of every chat give a different step under at least one other variant).
- security_context: an alert, 9 to 15 context records and the team's playbook. `covered` has one record that meets
  every part of the playbook's test; `fault` is the same record with exactly one field changed. Two questions: is
  the activity accounted for (noul), and the playbook's next step for (accounted for?, severity band).
Scene tables live in targets.workflow_scenes.
"""

from __future__ import annotations

import random

from yev.generators.llm.planned import Group, PlannedFamily, QSpec, StateSpec
from yev.generators.llm.targets import vocab
from yev.generators.llm.targets.agent_trace import an, per_batch, person
from yev.generators.llm.targets.workflow_invoice import ordered, pick, plan_invoice
from yev.generators.llm.targets.workflow_scenes import (
    CS_ACTION_PHRASES, CS_CONDITIONS, CS_EVENTS, CS_HARMS, CS_OWED, CS_POLICIES, CS_PROBLEMS, FRUSTRATION_CUES,
    MONTHS, NTH, SEC_ACTION_PHRASES, SEC_BANDS, SEC_DEFINITION, SEC_KINDS, SEC_PLACES, SEC_PLAYBOOKS, SEC_PURPOSES,
    SEC_STATUS_FAULT, SEC_STATUS_OK, SEC_TEAMS, URGENCY_CUES, URGENCY_DAYS,
)
from yev.schema import Option

__all__ = ["URGENCY_OPTIONS", "FRUSTRATION_OPTIONS", "ACTION_OPTIONS", "ACCOUNTED_OPTIONS", "STEP_OPTIONS",
           "cs_step", "INVOICE_LINE", "CS_SCALES_ACTIONS", "SECURITY_CONTEXT"]

# --- cs_scales_actions ----------------------------------------------------------------------------

URGENCY_OPTIONS: tuple[Option, ...] = (
    Option("0", "0, none: the customer says nothing depends on when this is sorted."),
    Option("1", "1, low: the customer names a time limit more than seven days away."),
    Option("2", "2, moderate: the customer names a time limit three to seven days away."),
    Option("3", "3, high: the customer needs it today or by tomorrow."),
    Option("4", "4, critical: the customer is losing something right now, for as long as the problem lasts."),
)
FRUSTRATION_OPTIONS: tuple[Option, ...] = (
    Option("0", "0, none: calm and friendly, no complaint about the service, first contact about the problem."),
    Option("1", "1, slight: polite, one remark that the problem is a nuisance, first contact about it."),
    Option("2", "2, clear: says outright they are unhappy; this is their second contact about the problem."),
    Option("3", "3, strong: third or later contact, sharp wording, patience running out, but no threat to "
                "leave."),
    Option("4", "4, extreme: angry writing and a threat to cancel or take their business elsewhere."),
)
ACTION_OPTIONS: tuple[Option, ...] = (
    Option("flag_for_review", "Flag the ticket so that a supervisor reviews it."),
    Option("set_pending", "Set the ticket to pending until the customer replies with what is missing."),
    Option("handoff_human", "Hand the conversation over to a person on the support team."),
    Option("raise_priority", "Raise the ticket's priority in the queue."),
    Option("reply_close", "Reply to the customer and close the ticket."),
)
URGENCY_QUESTIONS: tuple[str, ...] = (
    "How urgent is the customer's problem, going by its timing?",
    "On this scale, how pressing is the customer's problem in time?",
    "Judged by when it has to be fixed, how urgent is this customer's problem?",
)
FRUSTRATION_QUESTIONS: tuple[str, ...] = (
    "How frustrated is the customer in this chat?",
    "On this scale, how much frustration does the customer show?",
    "Going by what the customer writes and how often they have been in touch, how frustrated are they?",
)
ACTION_QUESTIONS: tuple[str, ...] = (
    "Under the team rules, which step should the triage desk take now that the chat has ended?",
    "Applying the team rules to this chat, what is the next step?",
    "Which of these steps do the team rules call for at the end of this chat?",
)
PAIR_SHARE = 0.30                               # groups that belong to a near-neighbour pair
CS_TOKENS = (500, 3000)
CS_PER_BATCH = per_batch(1, CS_TOKENS)
# Chance a new group starts a pair (it then plans its partner at the next index). A pair never starts on the last
# slot of a writer batch, so both members always land in the same batch; the other slots make up for it.
PAIR_START = PAIR_SHARE / (2 - PAIR_SHARE) * CS_PER_BATCH / (CS_PER_BATCH - 1)
FLAG_SHARE = 0.3                                # chance each yes/no fact of the chat holds
_PAIR_ATTR = "_yev_cs_pair"                     # the partner waiting for the next index, kept on the run's rng
_PAIR_LEVELS = (1 / 8, 2 / 8, 2 / 8, 2 / 8, 1 / 8)   # level shares of a varied scale in pairs (k, k+1), k uniform
# Shares for groups outside pairs, so each scale ends flat at 20% per level: a pair varies one scale (half the time
# each), the other scale is flat.
SINGLE_LEVELS: tuple[float, ...] = tuple(
    (0.2 - PAIR_SHARE * (0.5 * p + 0.5 * 0.2)) / (1 - PAIR_SHARE) for p in _PAIR_LEVELS)


def cs_step(policy: str, facts: dict[str, bool]) -> str:
    rules, fallback = CS_POLICIES[policy]
    return next((action for cond, action in rules if facts[cond]), fallback)


def _facts(u: int, f: int, flags: dict[str, bool]) -> dict[str, bool]:
    return {**flags, "u2": u >= 2, "u3": u >= 3, "u4": u == 4, "f3": f >= 3, "f4": f == 4}


def _discriminating(u: int, f: int, flags: dict[str, bool]) -> bool:
    return len({cs_step(p, _facts(u, f, flags)) for p in CS_POLICIES}) >= 2


def _cs_rules(policy: str) -> str:
    rules, fallback = CS_POLICIES[policy]
    lines = ["Work down this list and take the first step whose condition holds."]
    lines += [f"{i}. If {CS_CONDITIONS[c]}, {CS_ACTION_PHRASES[a]}." for i, (c, a) in enumerate(rules, 1)]
    lines.append(f"{len(rules) + 1}. If none of the above holds, {CS_ACTION_PHRASES[fallback]}.")
    return "\n".join(lines)


def _cs_plan(rng: random.Random, domain: str) -> dict:
    """Everything one chat shares with a pair partner, plus its urgency and frustration levels."""
    return {"domain": domain, "assistant": rng.choice(vocab.ASSISTANT_NAMES), "customer": person(rng),
            "problem": rng.choice(CS_PROBLEMS), "event": rng.choice(CS_EVENTS), "harm": rng.choice(CS_HARMS),
            "nth": rng.choice(NTH), "owed": rng.choice(CS_OWED), "ticket": rng.randint(10000, 99999),
            "amount": f"{rng.randint(12, 240)}.{rng.randint(10, 99)}",
            "policy": rng.choice(sorted(CS_POLICIES)), "cue": rng.randrange(2),
            "days": {lv: rng.randint(*span) for lv, span in URGENCY_DAYS.items()}}


def _cs_group(rng: random.Random, index: int, p: dict, u: int, f: int, flags: dict[str, bool], skill: str) -> Group:
    u_cue = URGENCY_CUES[u][p["cue"] % len(URGENCY_CUES[u])].format(
        n=p["days"].get(u, 0), event=p["event"], harm=p["harm"])
    f_cue = FRUSTRATION_CUES[f][p["cue"] % len(FRUSTRATION_CUES[f])].format(nth=p["nth"])
    person_s = ("Partway through, the customer asks to speak to a member of staff instead of the assistant"
                if flags["person"] else "The customer never asks to speak to a member of staff")
    owes_s = (f"The assistant asks for {p['owed']}; the customer says they do not have it to hand and will send it "
              f"later, and the chat ends before they do" if flags["owes"]
              else "The customer gives every detail the assistant asks for during the chat")
    unknown_s = (f"The customer also mentions a payment of {p['amount']} on their statement that they do not "
                 f"recognise" if flags["unknown"] else "The customer mentions no payment they do not recognise")
    brief = f"Timing: the customer {u_cue}. Mood: the customer {f_cue}. {person_s}. {owes_s}. {unknown_s}."
    scene = (f"{an(p['domain'], start=True)}. {p['assistant']} is the AI assistant on the {p['domain']}'s support "
             f"chat. When a chat ends, the triage desk picks the next step for the ticket from the team rules "
             f"below.\n\nTeam rules (copy word for word, in this order):\n{_cs_rules(p['policy'])}\n\n"
             f"Ticket {p['ticket']}: the customer, {p['customer']}, writes in about {p['problem']}. The "
             f"assistant answers helpfully but cannot fix the problem inside the chat.")
    gold = cs_step(p["policy"], _facts(u, f, flags))
    questions = (
        QSpec(name="urgency", type="score", question=pick(rng, URGENCY_QUESTIONS), options=URGENCY_OPTIONS,
              gold=str(u), skill=f"cs_urgency{skill}"),
        QSpec(name="frustration", type="score", question=pick(rng, FRUSTRATION_QUESTIONS),
              options=FRUSTRATION_OPTIONS, gold=str(f), skill=f"cs_frustration{skill}"),
        QSpec(name="next_action", type="choice", question=pick(rng, ACTION_QUESTIONS),
              options=ordered(rng, ACTION_OPTIONS), gold=gold, skill=f"cs_action_{p['policy']}"),
    )
    return Group(gid=f"csa{index:05d}", scene=scene, states=(StateSpec(sid="chat", brief=brief, questions=questions),))


def _flags(rng: random.Random) -> dict[str, bool]:
    return {k: rng.random() < FLAG_SHARE for k in ("person", "owes", "unknown")}


def plan_cs(rng: random.Random, index: int, domain: str) -> Group:
    waiting = getattr(rng, _PAIR_ATTR, None)
    if waiting is not None and waiting[0] == index:
        delattr(rng, _PAIR_ATTR)
        _, p, u, f, flags = waiting
        return _cs_group(rng, index, p, u, f, flags, "_pair")
    p = _cs_plan(rng, domain)
    if index % CS_PER_BATCH != CS_PER_BATCH - 1 and rng.random() < PAIR_START:
        k = rng.randrange(4)
        flat = rng.randrange(5)
        lv = [k, k + 1]
        rng.shuffle(lv)
        if rng.random() < 0.5:
            pairs = [(lv[0], flat), (lv[1], flat)]
        else:
            pairs = [(flat, lv[0]), (flat, lv[1])]
        flags = _flags(rng)
        while not all(_discriminating(u, f, flags) for u, f in pairs):
            flags = _flags(rng)
        setattr(rng, _PAIR_ATTR, (index + 1, p, *pairs[1], flags))
        return _cs_group(rng, index, p, *pairs[0], flags, "_pair")
    u, f = rng.choices(range(5), SINGLE_LEVELS)[0], rng.choices(range(5), SINGLE_LEVELS)[0]
    while not _discriminating(u, f, flags := _flags(rng)):
        pass
    return _cs_group(rng, index, p, u, f, flags, "")


# --- security_context -----------------------------------------------------------------------------

ACCOUNTED_OPTIONS: tuple[Option, Option] = (
    Option("yes", "Yes: one record meets every part of the playbook's test for this activity."),
    Option("no", "No: no single record meets every part of the playbook's test for this activity."),
)
STEP_OPTIONS: tuple[Option, ...] = (
    Option("close_explained", "Close the alert and take no further action."),
    Option("notify_user", "Contact the user named in the alert and ask them to confirm the activity."),
    Option("open_investigation", "Hand the alert to an analyst as a new investigation case."),
    Option("contain_now", "Lock the account or isolate the host straight away."),
)
ACCOUNTED_QUESTIONS: tuple[str, ...] = (
    "Do the context records fully account for this activity?",
    "Measured against the playbook's test, is the activity in this alert accounted for by the context records?",
    "Does any one context record cover the alerted activity in every respect the playbook asks for?",
)
STEP_QUESTIONS: tuple[str, ...] = (
    "Under the team's playbook, what is the next step for this alert?",
    "Which step does the playbook call for with this alert?",
    "Following the playbook, how should this alert be handled now?",
)
SEC_ROLES = ("Asset register", "Staff directory")


def _username(full: str) -> str:
    first, last = full.split(" ", 1)
    return f"{first[0].lower()}.{last.lower().replace('-', '')}"


def _place(rng: random.Random, kind: str, avoid: set[str], keep: bool = True) -> str:
    """A place of the kind not in `avoid`; added to `avoid` when `keep`."""
    while True:
        p = rng.choice(SEC_PLACES[kind]).format(n=rng.randint(1, 9))
        if p not in avoid:
            if keep:
                avoid.add(p)
            return p


def _near_place(rng: random.Random, kind: str, place: str, avoid: set[str]) -> str:
    """Another place that looks like `place`: the same pattern with another number, or another country."""
    if kind == "travel":
        return _place(rng, kind, avoid)
    for _ in range(50):
        alt = rng.choice([t for t in SEC_PLACES[kind] if t.format(n=0)[:4] == place[:4]] or SEC_PLACES[kind])
        alt = alt.format(n=rng.randint(1, 9))
        if alt not in avoid:
            avoid.add(alt)
            return alt
    return _place(rng, kind, avoid)


def _when(rng: random.Random, kind: dict, month: str, day: int, hh: int, mm: int, fault: str | None) -> str:
    """The key record's time field: covering the activity, or (closed_before / badge_absent) not."""
    date = f"{day} {month} 2026"
    if kind["kind"] == "change":
        if fault == "closed_before":
            end = rng.randint(max(1, hh - 3), hh - 1)
            return f"{date} {end - rng.randint(1, 2):02d}:00 to {end:02d}:00"
        return f"{date} {hh - rng.randint(1, 2):02d}:00 to {hh + rng.randint(1, 3):02d}:00"
    if kind["kind"] == "badge":
        if fault == "badge_absent":
            return f"no badge entry recorded on {date}"
        t_in, t_out = hh * 60 + mm - rng.randint(15, 60), hh * 60 + mm + rng.randint(20, 80)
        return f"badged in {t_in // 60:02d}:{t_in % 60:02d}, badged out {t_out // 60:02d}:{t_out % 60:02d} on {date}"
    if fault == "closed_before":
        end = day - rng.randint(1, 3)
        return f"{end - rng.randint(2, 5)} {month} 2026 to {end} {month} 2026"
    return f"{day - rng.randint(1, 5)} {month} 2026 to {day + rng.randint(1, 6)} {month} 2026"


def _record(rid: str, kind: dict, num: int, user: str, place: str, when: str, status: str | None,
            purpose: str | None) -> str:
    parts = [rid, kind["label"], f"{kind['prefix']}-{num}", f"{kind['user']}: {user}", f"{kind['place']}: {place}",
             f"{kind['time']}: {when}"]
    if kind["status"]:
        parts.append(f"{kind['status']}: {status}")
    if purpose:
        parts.append(f"Purpose: {purpose}")
    return " | ".join(parts)


def _distractor(rng: random.Random, kind: dict, rid: str, user: str, place: str, month: str) -> str:
    day = rng.randint(7, 22)
    when = _when(rng, kind, month, day, rng.randint(12, 20), rng.randint(0, 59), None)
    status = rng.choice(("approved", "approved", "closed", "cancelled"))
    purpose = rng.choice(SEC_PURPOSES[kind["kind"]]) if kind["kind"] in SEC_PURPOSES else None
    return _record(rid, kind, rng.randint(1000, 9999), user, place, when, status, purpose)


def plan_security(rng: random.Random, index: int, domain: str) -> Group:
    kind = rng.choice(SEC_KINDS)
    fault = rng.choice(kind["faults"])
    band = rng.choice(SEC_BANDS)
    book = rng.choice(sorted(SEC_PLAYBOOKS))
    analyst, full = person(rng), person(rng)
    user = _username(full)
    names = {user}
    others = []
    while len(others) < 16:
        u = _username(person(rng))
        if u not in names:
            names.add(u)
            others.append(u)
    used: set[str] = set()
    place = _place(rng, kind["kind"], used)
    month = rng.choice(MONTHS)
    day = rng.randint(12, 20)
    hh = rng.randint(20, 22) if kind["kind"] == "badge" else rng.randint(12, 17)
    mm = rng.randint(0, 29) if kind["kind"] == "badge" else rng.randint(0, 59)
    team = rng.choice(SEC_TEAMS)

    n_rec = rng.randint(9, 15)
    k = rng.randrange(n_rec)
    key_id = f"R{k + 1}"
    num = rng.randint(1000, 9999)
    purpose = rng.choice(SEC_PURPOSES[kind["kind"]]) if kind["kind"] in SEC_PURPOSES else None
    good = dict(user=user, place=place, when=_when(rng, kind, month, day, hh, mm, None), status=SEC_STATUS_OK)
    bad = dict(good)
    if fault in SEC_STATUS_FAULT:
        bad["status"] = SEC_STATUS_FAULT[fault]
    elif fault in ("closed_before", "badge_absent"):
        bad["when"] = _when(rng, kind, month, day, hh, mm, fault)
    elif fault == "other_user":
        bad["user"] = others.pop()
    else:
        bad["place"] = _near_place(rng, kind["kind"], place, used)
    covered = _record(key_id, kind, num, purpose=purpose, **good)
    faulty = _record(key_id, kind, num, purpose=purpose, **bad)

    rows = []
    extras = [r for r in SEC_ROLES if kind["kind"] in ("change", "access") or r == "Staff directory"]
    extras = [r for r in extras if rng.random() < 0.5]
    for i in range(n_rec):
        rid = f"R{i + 1}"
        if i == k:
            rows.append(f"{rid} | (as given in the brief)")
        elif extras and rng.random() < 0.3:
            role = extras.pop()
            if role == "Asset register":
                rows.append(f"{rid} | Asset register | AST-{rng.randint(100, 999)} | {kind['place']}: {place} | "
                            f"Owner team: {team} | Primary admin: {user}")
            else:
                rows.append(f"{rid} | Staff directory | User: {user} | Name: {full} | Team: {team}")
        else:
            other_kind = kind if rng.random() < 0.6 else rng.choice(SEC_KINDS)
            spot = _place(rng, other_kind["kind"], used, keep=False)
            rows.append(_distractor(rng, other_kind, rid, others.pop(), spot, month))
    playbook = [SEC_DEFINITION] + [
        f"- {'Accounted for' if acc else 'Not accounted for'}, band {b}: "
        f"{SEC_ACTION_PHRASES[SEC_PLAYBOOKS[book][(acc, b)]]}." for acc in (True, False) for b in SEC_BANDS]
    alert = [f"Rule: {kind['rule']}", f"Severity band: {band}", f"User: {user} ({full}), {team}",
             f"{kind['alert_place']}: {place}", f"Time of activity: {day} {month} 2026 {hh:02d}:{mm:02d}",
             f"Activity: {kind['activity'].format(user=user, place=place)}"]
    scene = (f"{an(domain, start=True)}. {analyst} works on the {domain}'s security operations desk and handles "
             f"each alert with the context records the alert console attaches to it and the team's playbook.\n\n"
             "Alert (copy as given):\n" + "\n".join(f"- {a}" for a in alert) + "\n\n"
             f"Context records (copy as given, in this order; {key_id} is given in the brief):\n"
             + "\n".join(f"- {r}" for r in rows) + "\n\n"
             "Playbook (copy word for word, in this order):\n" + "\n".join(playbook))
    q1 = pick(rng, ACCOUNTED_QUESTIONS)
    q2 = pick(rng, STEP_QUESTIONS)
    opts = ordered(rng, STEP_OPTIONS)
    states = []
    for sid, row, acc in (("covered", covered, True), ("fault", faulty, False)):
        skill = f"security_{kind['kind']}_{'covered' if acc else fault}"
        states.append(StateSpec(sid=sid, brief=f"Record (copy as given, in its place in the list):\n- {row}",
                                questions=(
            QSpec(name="accounted_for", type="noul", question=q1, options=ACCOUNTED_OPTIONS,
                  gold="yes" if acc else "no", skill=skill),
            QSpec(name="next_step", type="choice", question=q2, options=opts,
                  gold=SEC_PLAYBOOKS[book][(acc, band)], skill=f"security_step_{book}"),
        )))
    rng.shuffle(states)
    return Group(gid=f"sec{index:05d}", scene=scene, states=tuple(states))


# --- families -------------------------------------------------------------------------------------

INVOICE_FORMAT = (
    'A supplier invoice review packet in plain text, in this order, each part under its heading on a line of its '
    'own: "Invoice:" with the header rows and then the invoice line rows; "Purchase order:" with the paragraphs of '
    "ordinary order terms the brief's length plan asks for (none of them about which work is included, rates or "
    'what was delivered), then the priced item rows and the open point; "Change orders and change requests:"; "Evidence records:"; "Prior invoices:"; "Messages:" '
    "with each message written out in full, with sender and date. Every row the brief gives is copied exactly, one "
    'per line starting "- ", with every field, figure, unit and period unchanged. The packet adds no figure, record, '
    "approval or comment on the lines beyond what the brief gives."
)
CS_FORMAT = (
    'Plain text in two parts, each under its heading on a line of its own: "Team rules:" with the scene\'s rules '
    'copied word for word in the given order, then "Chat:" with 6 to 20 turns, each starting "Customer:" or '
    '"Assistant:". The chat shows every fact of the brief through what the customer writes; nobody in it rates the '
    "customer's urgency or mood, and nobody decides the next step for the ticket."
)
SECURITY_FORMAT = (
    'Plain text in three parts, each under its heading on a line of its own: "Alert:" with the alert rows; "Context '
    'records:" with the records in the given order, one per line starting "- ", the brief\'s record in its place, '
    'every field copied exactly; "Playbook:" with the playbook copied word for word. A line or two of console '
    "header text may come first. Nothing in the state says whether the activity is accounted for or what to do."
)

INVOICE_LINE = PlannedFamily(
    name="invoice_line",
    description="Accounts payable review packets: a supplier invoice with its lines, our purchase order with its "
                "priced items, change orders, evidence records from our systems and the vendor's, earlier "
                "invoices and messages. Each packet is the one the clerk reads before deciding how to treat two of "
                "the invoice lines.",
    state_format=INVOICE_FORMAT, tokens=(2000, 8000), per_batch=per_batch(1, (2000, 8000)), plan=plan_invoice,
)
CS_SCALES_ACTIONS = PlannedFamily(
    name="cs_scales_actions",
    description="Customer support chats between a customer and an AI assistant, with the team's rules for the "
                "next step on the ticket. The customer's timing, mood and requests come out in what they write.",
    state_format=CS_FORMAT, tokens=CS_TOKENS, per_batch=CS_PER_BATCH, plan=plan_cs,
)
SECURITY_CONTEXT = PlannedFamily(
    name="security_context",
    description="Security alert console views: one alert, the context records attached to it (change tickets, "
                "travel requests, badge logs, access approvals, asset and staff entries) and the team's playbook.",
    state_format=SECURITY_FORMAT, tokens=(600, 3000), per_batch=per_batch(2, (600, 3000)), plan=plan_security,
)
