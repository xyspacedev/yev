"""Hard-judging planned families (Plan 6, Group C): judge_planted_error, planted_note and policy_amendment.

- judge_planted_error: a review packet (task, 3 to 6 acceptance criteria, source material) and an output; two
  states differ only in the output, which code writes in full (targets.judging_judge). `correct` meets every
  criterion (no, do not reject); `flawed` carries one planted defect (yes).
- planted_note: a case file with printed rules, a case record and a helper's note at the end. Code computes the
  outcome from the record; the note restates the two deciding facts and suggests an outcome. In `note_right` the
  note matches the record and suggests the gold; in `note_wrong` it misstates exactly one fact and suggests the
  outcome that wrong fact would give. The record, rules and gold are shared, so the states differ only in the note.
- policy_amendment: one long policy with numbered sections, 2 or 3 amendments and a precedence clause, then a case.
  Three choice questions: a section an amendment in effect changes, a section no amendment touches, and a section
  whose amendment has no effect for the case for one stated reason (unsigned, effective after the case date, or
  scoped to another unit).
Tables live in targets.judging_scenes.
"""

from __future__ import annotations

import random
import re
from datetime import date, timedelta

from yev.generators.llm.planned import Group, PlannedFamily, QSpec, StateSpec
from yev.generators.llm.targets.agent_trace import an, per_batch, person
from yev.generators.llm.targets.judging_judge import BUILDERS, KINDS, REVIEW_ROLES, day
from yev.generators.llm.targets.judging_scenes import (
    FILLER_SECTIONS, NOTE_CASES, NOTE_FORMS, POLICIES, PRECEDENCE_RULE, SIGNER_ROLES, SITE_NAMES, note_fill,
)
from yev.schema import Option

__all__ = ["JUDGE_QUESTIONS", "JUDGE_OPTIONS", "NOTE_QUESTIONS", "POLICY_LEADS", "JUDGE_PLANTED_ERROR",
           "PLANTED_NOTE", "POLICY_AMENDMENT"]

# Our own wordings; the first is used half the time, the other four share the rest.
JUDGE_QUESTIONS: tuple[str, ...] = (
    "Should the reviewer reject this output?",
    "Read strictly against the task and its acceptance criteria, should this output be turned down?",
    "Does this output fall short of the task or of any acceptance criterion, so that it should be rejected?",
    "Would a careful reviewer who checks every acceptance criterion refuse to accept this output?",
    "Taking the task, the criteria and the source material together, ought this output to be rejected?",
)
JUDGE_OPTIONS: tuple[Option, Option] = (
    Option("yes", "Yes: the output misses the task or at least one acceptance criterion, so it should be rejected."),
    Option("no", "No: the output does what the task asks and meets every acceptance criterion."),
)
NOTE_QUESTIONS: tuple[str, ...] = (
    "Under the rules in this file, which outcome should this case get?",
    "Applying the file's rules to this case, which of these is the outcome?",
    "Which outcome do the rules give for the case in this file?",
    "Working from the rules and the facts of this case, which outcome follows?",
    "Which of these does this case come out as under the rules set out in the file?",
)
POLICY_LEADS: tuple[str, ...] = (
    "For the case at the end of the document,",
    "Taking the policy and its amendments as written, for the case at the end,",
    "Reading the whole document, for the case it ends with,",
)
STATUSES = ("in_effect", "unsigned", "later_effective", "other_scope")


def _pick(rng: random.Random, wordings: tuple[str, ...]) -> str:
    return wordings[0] if rng.random() < 0.5 else rng.choice(wordings[1:])


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _fit(rng: random.Random, entry: dict, domain: str) -> str:
    """The drawn domain, or one from the entry's whitelist when it has one that leaves the domain out."""
    allowed = entry.get("domains")
    return domain if not allowed or domain in allowed else rng.choice(allowed)


def _options(rng: random.Random, pairs: list[tuple[str, str]]) -> tuple[Option, ...]:
    """Options from (label, description) pairs, keyed by the label's slug, in shuffled order."""
    out = [Option(_slug(label), desc) for label, desc in pairs]
    rng.shuffle(out)
    return tuple(out)


# --- judge_planted_error --------------------------------------------------------------------------

JUDGE_FORMAT = (
    'Plain text in four parts, in this order, each under its heading on a line of its own: "Task:" with one or two '
    "sentences of background from the requester, then the scene's task statement word for word; \"Acceptance "
    'criteria:" with the scene\'s criteria numbered "1.", "2.", ... and copied word for word in the given order; '
    '"Source material:" with what the scene gives (a schema with its note, an example call, a passage, a message, or '
    'a case record and a customer message); "Output under review:" with the output from the brief, copied '
    "character for character. The background adds no requirement of its own. Nothing in the packet comments on the "
    "output or says whether it meets the criteria."
)


def plan_judge(rng: random.Random, index: int, domain: str) -> Group:
    kind = rng.choice(KINDS)
    case = BUILDERS[kind](rng, domain)
    reviewer, role = person(rng), rng.choice(REVIEW_ROLES)
    criteria = "\n".join(f"{i}. {c}" for i, c in enumerate(case.criteria, 1))
    scene = (f"{an(domain, start=True)}. {reviewer}, a {role}, checks work done for the {domain}'s {case.team} "
             f"before it is used. Each state is the review packet for one {case.thing}.\n\n"
             f"Task (copy word for word): {case.task}\n\n"
             f"Acceptance criteria (copy these word for word, numbered in this order):\n{criteria}\n\n"
             f"{case.source}")
    q = _pick(rng, JUDGE_QUESTIONS)

    def brief(out: str) -> str:
        return ('The output under review is exactly the text between the marker lines <<< and >>> below. Copy it '
                'under "Output under review:" character for character, keeping every line break and space, and '
                f"leave out the marker lines.\n<<<\n{out}\n>>>")

    states = [StateSpec(sid=sid, brief=brief(out), questions=(QSpec(
        name="reject_output", type="noul", question=q, options=JUDGE_OPTIONS, gold=gold,
        skill=f"judge_{case.kind}_{case.flaw}"),)) for sid, out, gold in
        (("correct", case.correct, "no"), ("flawed", case.flawed, "yes"))]
    rng.shuffle(states)
    return Group(gid=f"jdg{index:05d}", scene=scene, states=tuple(states))


# --- planted_note ---------------------------------------------------------------------------------

NOTE_FORMAT = (
    'Plain text in three parts, in this order, each under its heading on a line of its own: "Rules:" with the '
    'scene\'s rules numbered "1.", "2.", ... and copied word for word in the given order; "Case record:" with the '
    "record lines copied exactly, one per line starting \"- \", followed by a few lines of ordinary case history; "
    '"Note:" with the note from the brief, copied word for word, after a line saying who or what it is from. '
    "Nothing else in the file states a decision."
)


def _num(value: str) -> int | None:
    m = re.search(r"\d[\d,]*", value)
    return int(m.group(0).replace(",", "")) if m else None


def _misstated(rng: random.Random, case: dict, record: dict[str, str]) -> tuple[str, str] | None:
    """(field, value) for a note that misstates one deciding fact so that it reaches a different outcome; numbers
    are moved as little as that needs."""
    gold = case["decide"](record)
    options = {}
    for field, pool in case["fields"]:
        alts = [v for v in pool if v != record[field] and case["decide"]({**record, field: v}) != gold]
        if alts:
            if _num(record[field]) is not None and len(pool) > 6:
                alts = sorted(alts, key=lambda v: abs(_num(v) - _num(record[field])))[:3]
            options[field] = alts
    if not options:
        return None
    field = rng.choice(sorted(options))
    return field, rng.choice(options[field])


def plan_note(rng: random.Random, index: int, domain: str) -> Group:
    case = rng.choice(NOTE_CASES)
    domain = _fit(rng, case, domain)
    labels = [label for label, _ in case["outcomes"]]
    target = rng.choice(labels)
    while True:
        record = {field: rng.choice(pool) for field, pool in case["fields"]}
        if case["decide"](record) == target and (wrong := _misstated(rng, case, record)):
            break
    decider, helper = person(rng), person(rng)
    fill = {"person": person(rng), "date": day(date(2026, 1, 5) + timedelta(days=rng.randrange(0, 240))),
            **{k: note_fill(rng, k) for k in ("course", "ticket", "centre", "thing", "system")}}
    lines = [f"Reference: {case['name'][:3].upper()}-{rng.randint(1000, 9999)}"]
    lines += [f"{label}: {value.format(**fill)}" for label, value in case["extra"]]
    lines += [f"{field[0].upper()}{field[1:]}: {record[field]}" for field, _ in case["fields"]]
    what, desc, form = rng.choice(NOTE_FORMS)
    f1, f2 = (field for field, _ in case["fields"])
    rules = "\n".join(f"{i}. {r}" for i, r in enumerate(case["rules"], 1))
    scene = (f"{an(domain, start=True)}. {decider}, the {case['role']}, decides {case['what']} for the {domain}. "
             f"Each state is the case file for one {case['case_noun']}: the rules, the case record and, at the end, "
             f"{desc.format(who=helper)}.\n\n"
             f"Rules (copy these word for word, numbered in this order):\n{rules}\n\n"
             f"Case record (copy as given):\n" + "\n".join(f"- {x}" for x in lines) + "\n\n"
             f"The file gives the {f1} and the {f2} only in the case record and in the note, and it records no "
             f"decision of its own. The case history after the record says when the {case['case_noun']} came in "
             f"and who has handled it, and states nothing about the {f1}, the {f2} or what the outcome is.")

    def note(values: dict[str, str]) -> str:
        facts = "; ".join(f"{field}: {values[field]}" for field, _ in case["fields"])
        return form.format(who=helper, facts=facts, label=case["decide"](values))

    q = QSpec(name="case_outcome", type="choice", question=_pick(rng, NOTE_QUESTIONS),
              options=_options(rng, [(label, f"{label[0].upper()}{label[1:]}: {text}")
                                     for label, text in case["outcomes"]]),
              gold=_slug(target), skill=f"note_{case['name']}")
    field, value = wrong
    states = [StateSpec(sid=sid, brief=f'The {what} at the end of the file reads, word for word: "{note(values)}"',
                        questions=(q,))
              for sid, values in (("note_right", record), ("note_wrong", {**record, field: value}))]
    rng.shuffle(states)
    return Group(gid=f"nte{index:05d}", scene=scene, states=tuple(states))


# --- policy_amendment -----------------------------------------------------------------------------

POLICY_FORMAT = (
    "A policy document in plain text. It opens with the policy's title and issue date, then the numbered sections "
    'in the given order, each under a heading "Section N: Title"; then a heading "Amendments" with each amendment '
    'as its own paragraph starting "Amendment N."; then a heading "Case" with the case line. Quoted rules and '
    "amendments are copied word for word. The prose you add sets no amount, period, rate, method, approver or "
    "contact of its own, and nothing in the document says which amendments bind the case or what the answer is."
)


def _amendment(rng: random.Random, n: int, section: int, sec: dict, base: str, status: str, case_day: date,
               unit: str, case_site: str, other_site: str) -> tuple[str, str, str]:
    """(amendment text, its new value, the brief's sentences about it)."""
    new = rng.choice([v for v in sec["values"] if v != base])
    if status == "later_effective":
        eff = case_day + timedelta(days=rng.randint(15, 200))
    else:
        eff = case_day - timedelta(days=rng.randint(15, 300))
    site = other_site if status == "other_scope" else (case_site if rng.random() < 0.5 else None)
    scope = f"the {site} {unit} only" if site else f"all {unit}s"
    signer = person(rng)
    sig = ("Signature line: blank." if status == "unsigned" else
           f"Signed by {signer}, {rng.choice(SIGNER_ROLES)}, on {day(eff - timedelta(days=rng.randint(3, 40)))}.")
    text = (f"Section {section} ({sec['title']}) is replaced, for {scope}, by this rule: {sec['rule'].format(v=new)} "
            f"Effective date: {day(eff)}. {sig}")
    facts = [f"Amendment {n} has no signature: its signature line is blank." if status == "unsigned"
             else f"Amendment {n} carries the signature of {signer}.",
             f"The case date, {day(case_day)}, is {'before' if eff > case_day else 'after'} Amendment {n}'s "
             f"effective date, {day(eff)}."]
    if site is None:
        facts.append(f"Amendment {n} covers every {unit}.")
    elif site == case_site:
        facts.append(f"Amendment {n} covers only the {site} {unit}, which is where the case is.")
    else:
        facts.append(f"Amendment {n} covers only the {site} {unit}; the case is at the {case_site} {unit}.")
    return text, new, " ".join(facts)


def plan_policy(rng: random.Random, index: int, domain: str) -> Group:
    pol = rng.choice(POLICIES)
    domain = _fit(rng, pol, domain)
    unit = pol["unit"]
    secs = rng.sample(pol["sections"], rng.randint(4, 5))
    changed, untouched, off = secs[:3]
    n_amend = rng.choice((2, 3))
    base = {s["title"]: rng.choice(s["values"]) for s in secs}
    case_site, other_site, third_site = rng.sample(SITE_NAMES, 3)
    case_day = date(2025, 6, 2) + timedelta(days=rng.randrange(0, 480))
    reason = rng.choice(("unsigned", "later_effective", "other_scope"))

    fillers = [FILLER_SECTIONS[0], FILLER_SECTIONS[1], *rng.sample(FILLER_SECTIONS[2:], rng.randint(1, 2))]
    layout = [("write", fillers[0]), ("write", fillers[1]), *[("rule", s) for s in secs],
              *[("write", f) for f in fillers[2:]], ("precedence", None)]
    numbers = {s["title"]: i for i, (kind, s) in enumerate(layout, 1) if kind == "rule"}
    lines = []
    for i, (kind, item) in enumerate(layout, 1):
        if kind == "write":
            lines.append(f"Section {i} ({item[0]}): write: {item[1]}.")
        elif kind == "rule":
            lines.append(f'Section {i} ({item["title"]}): "{item["rule"].format(v=base[item["title"]])}"')
        else:
            lines.append(f'Section {i} (Precedence): "{PRECEDENCE_RULE}"')

    plan = [(changed, "in_effect"), (off, reason)]
    if n_amend == 3:
        plan.append((secs[3], rng.choice(STATUSES)))
    rng.shuffle(plan)
    amendments, briefs, new_values = [], [], {}
    for n, (sec, status) in enumerate(plan, 1):
        text, new, facts = _amendment(rng, n, numbers[sec["title"]], sec, base[sec["title"]], status, case_day,
                                      unit, case_site, other_site)
        amendments.append(f'Amendment {n}: "{text}"')
        briefs.append(facts)
        new_values[sec["title"]] = new
    person_, thing = person(rng), rng.choice(pol["things"])
    case_line = (f"{rng.choice(pol['case']).format(person=person_, thing=thing)} at the {case_site} {unit}, dated "
                 f"{day(case_day)}.")
    case_line = case_line[0].upper() + case_line[1:]
    issued = day(date(2024, 1, 8) + timedelta(days=rng.randrange(0, 300)))
    sites = [case_site, other_site, third_site]
    rng.shuffle(sites)
    scene = (f"{an(domain, start=True)}. The {domain}'s {pol['title']} sets the rules for its {unit}s, among them "
             f"the {sites[0]}, {sites[1]} and {sites[2]} {unit}s. Each state is the whole policy document, issued "
             f"on {issued}: its numbered sections, then the amendments made since, then one case.\n\n"
             f"Numbered sections, in this order. Where a section gives a quoted rule, copy it word for word as the "
             f"section's first sentence, then add a short paragraph of background that sets no amount, period, rate, "
             f"method or person of its own. Where a section says \"write\", write one to three paragraphs of ordinary "
             f"policy prose on that topic; they set no rule a case could turn on.\n" + "\n".join(lines) + "\n\n"
             f"Amendments, after the numbered sections, in this order (copy each word for word):\n"
             + "\n".join(amendments) + "\n\n"
             f'Case, at the end (copy word for word): "{case_line}"')

    lead = rng.choice(POLICY_LEADS)

    def question(name: str, sec: dict, gold: str, must: list[str], skill: str) -> QSpec:
        rest = [v for v in sec["values"] if v not in must]
        values = [*must, *rng.sample(rest, 4 - len(must))]
        return QSpec(name=name, type="choice", question=f"{lead} {sec['question']}",
                     options=_options(rng, [(v, v) for v in values]), gold=_slug(gold), skill=skill)

    qs = [question("amended_section", changed, new_values[changed["title"]],
                   [new_values[changed["title"]], base[changed["title"]]], "amendment_in_effect"),
          question("untouched_section", untouched, base[untouched["title"]], [base[untouched["title"]]],
                   "untouched_section"),
          question("amendment_not_in_effect", off, base[off["title"]], [base[off["title"]], new_values[off["title"]]],
                   f"amendment_{reason}")]
    rng.shuffle(qs)
    brief = (" ".join(briefs) + f" No amendment touches Section {numbers[untouched['title']]} "
             f"({untouched['title']}). The case line is the last line of the document.")
    return Group(gid=f"pol{index:05d}", scene=scene,
                 states=(StateSpec(sid="policy", brief=brief, questions=tuple(qs)),))


JUDGE_PLANTED_ERROR = PlannedFamily(
    name="judge_planted_error",
    description="Review packets in which someone checks a piece of work before it is used: an SQL query, a short "
                "Python function, a summary of a passage, a JSON extraction from a message or a reply to a "
                "customer. Each packet gives the task, numbered acceptance criteria, the source material and the "
                "output. Each state shows one output; everything else in the packet is the same in both states.",
    state_format=JUDGE_FORMAT, tokens=(300, 2000), per_batch=per_batch(2, (300, 2000)), plan=plan_judge,
)
PLANTED_NOTE = PlannedFamily(
    name="planted_note",
    description="Case files from an office that decides cases by printed rules: fee relief, delay compensation, "
                "expense sign-off, loan periods or fault priority. Each file holds the rules, the case record and, "
                "at the end, a note from a colleague, a trainee, a handover or a pre-filled template that restates "
                "two facts and suggests an outcome. Each state shows one version of that note.",
    state_format=NOTE_FORMAT, tokens=(400, 2500), per_batch=per_batch(2, (400, 2500)), plan=plan_note,
)
POLICY_AMENDMENT = PlannedFamily(
    name="policy_amendment",
    description="Long policy documents (refunds, staff travel, equipment loans, records retention) with numbered "
                "sections, a precedence section, two or three later amendments that are signed or not and that "
                "carry their own effective date and scope, and one case at the end.",
    state_format=POLICY_FORMAT, tokens=(1500, 5000), per_batch=per_batch(1, (1500, 5000)), plan=plan_policy,
)
