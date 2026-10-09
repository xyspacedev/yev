import json
import re
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta

from yev.generators.llm.planned import PlannedFamily, plan_batches, writer_prompt
from yev.generators.llm.targets import PLANNED_FAMILIES
from yev.generators.llm.targets import judging
from yev.generators.llm.targets import judging_judge as jj
from yev.generators.llm.targets import judging_scenes as scenes

N = 500
BIG = 2000   # coherence tests sample at least this many groups per family
FAMILIES = ("judge_planted_error", "planted_note", "policy_amendment")
# Benchmark wordings, hand-written from the gap reports. They guard against copying and must never be questions.
BANNED_QUESTIONS = (
    "Does the response fully and correctly satisfy every explicit requirement in the request?",
    "Does the response fully and correctly satisfy the request?",
    "Apply the controlling document and select the single authorized outcome.",
)
# Words a writer would copy into a state as a verdict; briefs give facts only.
LABEL_WORDS = ("reject", "accept", "flaw", "mistake", "wrong", "correct", "incorrect", "error", "invalid",
               "violat", "contradict", "in force", "binding", "not apply", "applies", "supersed", "overrid",
               "outcome is right", "should be")


def _groups(name, n=N, seed=0):
    return [g for b in plan_batches(PLANNED_FAMILIES[name], n, seed) for g in b["groups"]]


def _norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def _block(text, heading):
    """The lines under `heading` up to the next blank line."""
    return text.split(heading, 1)[1].split("\n\n", 1)[0].strip("\n").splitlines()


def _numbered(text, heading):
    lines = _block(text, heading)
    for i, line in enumerate(lines, 1):
        assert line.startswith(f"{i}. "), (i, line)
    return [line.split(". ", 1)[1] for line in lines]


def _output(brief):
    return brief.split("<<<\n", 1)[1].rsplit("\n>>>", 1)[0]


def _day(s):
    return datetime.strptime(s, "%d %B %Y").date()


# --- registry and shapes --------------------------------------------------------------------------

def test_families_registered_with_table_sizes():
    want = {"judge_planted_error": ((300, 2000), 10), "planted_note": ((400, 2500), 8),
            "policy_amendment": ((1500, 5000), 7)}
    for name, (tokens, per_batch) in want.items():
        fam = PLANNED_FAMILIES[name]
        assert isinstance(fam, PlannedFamily) and fam.name == name
        assert fam.tokens == tokens and fam.per_batch == per_batch, name
        # per_batch = max(1, 24000 // (states_per_group x mid_tokens))
        states = len(_groups(name, 1)[0]["states"])
        assert per_batch == max(1, 24000 // (states * sum(tokens) // 2))
    assert judging.JUDGE_PLANTED_ERROR is PLANNED_FAMILIES["judge_planted_error"]
    assert judging.PLANTED_NOTE is PLANNED_FAMILIES["planted_note"]
    assert judging.POLICY_AMENDMENT is PLANNED_FAMILIES["policy_amendment"]


def test_family_label_balance():
    rows = [(s["sid"], q) for g in _groups("judge_planted_error") for s in g["states"] for q in s["questions"]]
    assert len(rows) == 2 * N and all(q["type"] == "noul" for _, q in rows)
    assert all(q["gold"] == {"correct": "no", "flawed": "yes"}[sid] for sid, q in rows)
    assert sum(q["gold"] == "yes" for _, q in rows) / len(rows) == 0.5
    qs = Counter(q["question"] for _, q in rows)
    assert set(qs) == set(judging.JUDGE_QUESTIONS) and len(judging.JUDGE_QUESTIONS) == 5
    kinds = Counter(q["skill"].split("_")[1] for sid, q in rows if sid == "flawed")
    assert set(kinds) == {"sql", "function", "summary", "extraction", "reply"}
    assert all(abs(c / N - 0.2) <= 0.05 for c in kinds.values()), kinds


def test_group_shapes():
    for g in _groups("planted_note", 200):
        assert sorted(s["sid"] for s in g["states"]) == ["note_right", "note_wrong"]
        q = [s["questions"] for s in g["states"]]
        assert all(len(x) == 1 and x[0]["type"] == "choice" and 4 <= len(x[0]["options"]) <= 6 for x in q)
        assert q[0][0]["options"] == q[1][0]["options"] and q[0][0]["gold"] == q[1][0]["gold"]
    for g in _groups("policy_amendment", 200):
        assert len(g["states"]) == 1
        assert all(q["type"] == "choice" for q in g["states"][0]["questions"])


# --- judge_planted_error coherence ----------------------------------------------------------------

def _criteria(scene):
    crit = _numbered(scene, "Acceptance criteria (copy these word for word, numbered in this order):\n")
    assert 3 <= len(crit) <= 6, crit
    return crit


def _sql_check(scene, correct, flawed, flaw):
    crit = " ".join(_criteria(scene))
    schema = "\n".join(_block(scene, "Schema (copy as given):\n"))
    tables = re.findall(r"CREATE TABLE (\w+) \(([^;]+)\);", schema)
    assert len(tables) == 2, schema
    (parent, pcols), (child, ccols) = [(t, [c.strip().split()[0] for c in cols.split(",")]) for t, cols in tables]
    pk, pname, pgroup = pcols
    ck, fk, cdate, cval = ccols
    assert fk == pk and f"{fk} INTEGER REFERENCES {parent}({pk})" in schema
    g = re.search(rf"whose {pgroup} is '([^']+)'", crit).group(1)
    s, e = (_day(x) for x in re.search(r"dated from (\d+ \w+ \d{4}) to (\d+ \w+ \d{4}), both days included",
                                         crit).groups())
    total = re.search(rf"in this order: {pname}, then (\w+)\.", crit).group(1)
    assert "from largest to smallest" in crit
    db = sqlite3.connect(":memory:")
    db.executescript(schema)
    others = "OTHER"
    for i in range(1, 7):
        db.execute(f"INSERT INTO {parent} VALUES (?, ?, ?)", (i, f"name{i}", g if i <= 4 else others))
    days = [s - timedelta(days=1), s, s + (e - s) // 2, e, e + timedelta(days=1)]
    cid = 1
    for p in range(1, 7):
        for d in days:
            db.execute(f"INSERT INTO {child} VALUES (?, ?, ?, ?)", (cid + 20, p, d.isoformat(), 2 ** cid))
            cid += 1
    for k in range(1, 7):   # children whose own id equals a parent id, so a join on the wrong key finds rows
        db.execute(f"INSERT INTO {child} VALUES (?, ?, ?, ?)", (k, 7 - k if k != 3 else 3, s.isoformat(),
                                                                  2 ** (40 + k)))
    want = {}
    for p, d, v in db.execute(f"SELECT {fk}, {cdate}, {cval} FROM {child}"):
        if p <= 4 and s.isoformat() <= d <= e.isoformat():
            want[p] = want.get(p, 0) + v
    want = sorted(((f"name{p}", t) for p, t in want.items()), key=lambda x: -x[1])
    cur = db.execute(correct)
    assert [c[0] for c in cur.description] == [pname, total]
    assert cur.fetchall() == want
    cur = db.execute(flawed)
    got, cols = cur.fetchall(), [c[0] for c in cur.description]
    if flaw == "wrong_shape":
        assert len(cols) == 3 and sorted((r[1], r[2]) for r in got) == sorted(want)
    elif flaw == "criterion":
        assert cols == [pname, total] and got == sorted(want, key=lambda x: x[1])
    else:
        assert cols == [pname, total] and got != want and len(got) > 0
        if flaw == "join_key":
            assert f"ON {child[0] if child[0] != parent[0] else child[:2]}.{ck} =" in flawed
        else:
            assert flaw == "off_by_one"
    return [flawed.count("\n"), correct.count("\n")]


def _ref_function(args):
    if args == "prefix, n":
        return lambda prefix, n: [f"{prefix}-{i:03d}" for i in range(1, n + 1)]
    if args == "values, low, high":
        return lambda values, low, high: [v for v in values if low <= v <= high]
    assert args == "text, n"
    return lambda text, n: text.splitlines()[-n:]


def _inputs(args):
    if args == "prefix, n":
        return [("QX", n) for n in (0, 1, 2, 7, 12, 150)]
    if args == "values, low, high":
        return [([4, 9, 2, 9, 15, 3], 3, 9), ([5, 1, 8, 5], 5, 8), ([], 1, 2), ([7, 7], 7, 7), ([20, 10, 5], 1, 30)]
    return [("a\nb\nc\nd", 2), ("one\ntwo\n", 3), ("x", 1), ("p\nq\nr", 3), ("l1\nl2\nl3\nl4\nl5", 4)]


def _function_check(scene, correct, flawed, flaw):
    crit = _criteria(scene)
    name, args = re.search(r"The function is called (\w+) and takes these arguments, in this order: ([\w, ]+)\.",
                           " ".join(crit)).groups()
    ref = _ref_function(args)
    example = _block(scene, "Example (copy as given):\n")[0]
    call, result = example.split(" returns ")
    assert call.startswith(name + "(")
    assert eval(call.replace(name, "ref", 1), {"ref": ref}) == eval(result)   # noqa: S307
    for code in (correct, flawed):
        assert "import" not in code and code.startswith(f"def {name}({args}):")
    ns_c, ns_f = {}, {}
    exec(correct, ns_c)   # noqa: S102
    exec(flawed, ns_f)    # noqa: S102
    differs = False
    for inp in _inputs(args):
        copy = json.loads(json.dumps(inp))
        assert ns_c[name](*copy) == ref(*inp) and copy == list(inp)
        got = ns_f[name](*json.loads(json.dumps(inp)))
        differs |= got != ref(*inp)
        if flaw == "wrong_shape":
            assert not isinstance(got, list)
        elif flaw == "off_by_one":
            assert isinstance(got, list)
    assert differs


def _quoted(lines):
    return [x for line in lines for x in re.findall(r'"([^"]+)"', line)]


def _sentences(text):
    return [s for s in re.split(r"(?<=\.) (?=[A-Z])", text.strip()) if s]


def _summary_check(scene, correct, flawed, flaw):
    crit = _criteria(scene)
    k = int(re.search(r"at most (\d) sentences long", " ".join(crit)).group(1))
    facts = _block(scene, "with every name, date, time and figure exactly as given:\n")
    silent = re.search(r"It says nothing at all about ([^,]+), and", scene).group(1)
    words = set(re.findall(r"[a-z]+", " ".join(facts).lower()))
    need = _quoted(crit)
    assert need and all(x in correct for x in need) and all(x in " ".join(facts) for x in need)
    cs, fs = _sentences(correct), _sentences(flawed)
    assert len(cs) <= k and all(c.endswith(".") for c in cs)
    assert set(re.findall(r"[a-z]+", correct.lower())) <= words      # every word of the summary is in the facts
    if flaw == "unsupported_claim":
        extra = set(re.findall(r"[a-z]+", flawed.lower())) - words
        triggers = {w for topics in jj.SILENT.values() if silent in topics for w in topics[silent][1]}
        assert extra & triggers and len(fs) <= k and all(x in flawed for x in need), (extra, silent)
    elif flaw == "dropped_field":
        assert set(fs) < set(cs) and any(x not in flawed for x in need)
    else:
        assert flaw == "criterion" and len(fs) == k + 1 and set(cs) < set(fs) and len(cs) == k
        assert set(re.findall(r"[a-z]+", flawed.lower())) <= words


def _extraction_check(scene, correct, flawed, flaw):
    crit = " ".join(_criteria(scene))
    keys = re.findall(r'"(\w+)"', re.search(r"has exactly the keys ([^.]+)\.", crit).group(1))
    lines = _block(scene, "with every name, date and figure exactly as given:\n")
    want = []
    for line in lines:
        vals = re.findall(r"\[([^\]]+)\]", line)
        assert len(vals) == len(keys), line
        want.append({k: (int(v) if v.isdigit() else float(v) if re.fullmatch(r"\d+\.\d+", v) else v)
                     for k, v in zip(keys, vals)})
    assert json.loads(correct) == want
    got = json.loads(flawed)
    if flaw == "dropped_field":
        assert isinstance(got, list) and len(got) == len(want)
        diff = [(a, b) for a, b in zip(got, want) if a != b]
        assert len(diff) == 1 and set(diff[0][0]) < set(diff[0][1]) and len(diff[0][1]) - len(diff[0][0]) == 1
    elif flaw == "wrong_shape":
        if isinstance(got, dict):
            assert list(got.values()) == [want]
        else:
            diff = [(a, b) for a, b in zip(got, want) if a != b]
            assert len(diff) == 1
            (k,) = [k for k in keys if diff[0][0][k] != diff[0][1][k]]
            assert diff[0][0][k] == str(diff[0][1][k]) and re.search(r"not (a string|strings)", crit)
    else:
        assert flaw == "unsupported_claim" and isinstance(got, list) and len(got) == len(want)
        diff = [(a, b) for a, b in zip(got, want) if a != b]
        assert len(diff) == 1 and set(diff[0][0]) == set(diff[0][1])
        (k,) = [k for k in keys if diff[0][0][k] != diff[0][1][k]]
        assert type(diff[0][0][k]) is type(diff[0][1][k])
        given = [re.findall(r"\[([^\]]+)\]", line)[keys.index(k)] for line in lines]
        assert str(diff[0][0][k]) not in given


def _reply_check(scene, correct, flawed, flaw):
    crit = _criteria(scene)
    record = _block(scene, "Case record (copy as given):\n")
    need = _quoted(crit)
    assert all(x in correct for x in need)
    assert all(x in " ".join(record) for x in need)
    cs = [x for x in re.split(r"(?<=[.,])\s+|\n+", correct) if x]
    fs = [x for x in re.split(r"(?<=[.,])\s+|\n+", flawed) if x]
    added = [x for x in fs if x not in cs]
    rec = " ".join(record).lower()
    for word in ("voucher", "discount", "code", "%", "working days", "dispatched", "on its way"):
        assert word not in correct.lower()
    if flaw == "dropped_field":
        missing = [x for x in need if x not in flawed]
        assert len(missing) == 1 and len(fs) == len(cs)
    elif flaw == "criterion":
        assert len(added) == 1 and any(w in added[0].lower() for w in ("discount", "voucher", "credit"))
        assert any("no voucher, discount or credit" in c for c in crit) and all(x in flawed for x in need)
        assert "vouchers or discounts: none approved" in rec
    else:
        assert flaw == "unsupported_claim" and len(added) == 1 and all(x in flawed for x in need)
        claim = added[0].lower()
        assert ("working days" in claim and "refund date: not yet set" in rec) or \
            (("dispatched" in claim or "on its way" in claim) and "replacement: none" in rec), claim
        assert any("the case record does not show" in c for c in crit)


CHECKS = {"sql": _sql_check, "function": _function_check, "summary": _summary_check,
          "extraction": _extraction_check, "reply": _reply_check}
FLAWS = {"sql": {"join_key", "off_by_one", "criterion", "wrong_shape"},
         "function": {"off_by_one", "wrong_shape", "criterion"},
         "summary": {"unsupported_claim", "dropped_field", "criterion"},
         "extraction": {"dropped_field", "wrong_shape", "unsupported_claim"},
         "reply": {"unsupported_claim", "dropped_field", "criterion"}}


def test_judge_flaw_checkable_from_state():
    seen = Counter()
    for g in _groups("judge_planted_error", BIG, seed=3):
        st = {s["sid"]: s for s in g["states"]}
        assert set(st) == {"correct", "flawed"}
        _, kind, flaw = st["flawed"]["questions"][0]["skill"].split("_", 2)
        assert flaw in FLAWS[kind], (kind, flaw)
        seen[(kind, flaw)] += 1
        correct, flawed = _output(st["correct"]["brief"]), _output(st["flawed"]["brief"])
        assert correct != flawed
        # the two briefs differ only in the output they carry
        assert st["correct"]["brief"].replace(correct, "") == st["flawed"]["brief"].replace(flawed, "")
        assert "Task (copy word for word): " in g["scene"]
        CHECKS[kind](g["scene"], correct, flawed, flaw)
    for kind, flaws in FLAWS.items():
        for flaw in flaws:
            assert seen[(kind, flaw)] >= 40, (kind, flaw, seen)


# --- planted_note coherence -----------------------------------------------------------------------

def _fee(f):
    size, income = int(f["household size"].split()[0]), int(f["monthly income"].strip("£").replace(",", ""))
    base = 900 + 300 * size
    for limit, out in ((base, "full relief"), (base + 400, "half relief"), (base + 800, "quarter relief")):
        if income <= limit:
            return out
    return "no relief"


def _delay(f):
    d, warn = int(f["arrival delay"].split()[0]), f["severe weather warning on the route"] == "yes"
    if d < 30:
        return "no compensation"
    if warn:
        return "a travel voucher"
    return "25% of the fare back" if d < 60 else "50% of the fare back" if d < 120 else "the full fare back"


def _expense(f):
    cat, amount = f["claim category"], int(f["total claimed"].strip("£").replace(",", ""))
    limits = {"hospitality": (200, 800, 3000), "travel": (500, 2000, 6000), "equipment": (300, 1500, 5000)}[cat]
    for limit, out in zip(limits, ("the line manager", "the department head", "the finance director")):
        if amount <= limit:
            return out
    return "the board"


def _loan(f):
    table = {"standard": ("14 days", "28 days", "28 days"), "high-demand": ("3 days", "7 days", "14 days"),
             "reference-only": ("not for loan", "not for loan", "3 days")}
    return table[f["item class"]][("basic", "plus", "staff").index(f["membership tier"])]


def _priority(f):
    users, work = int(f["users affected"]), f["workaround available"] == "yes"
    band = 0 if users >= 50 else 1 if users >= 10 else 2
    return f"Priority P{band + 1 + work}"


DECIDE = {"fee_relief": (_fee, ("household size", "monthly income")),
          "delay_compensation": (_delay, ("arrival delay", "severe weather warning on the route")),
          "expense_approval": (_expense, ("claim category", "total claimed")),
          "loan_period": (_loan, ("membership tier", "item class")),
          "ticket_priority": (_priority, ("users affected", "workaround available"))}


def _note_case(g):
    template = g["states"][0]["questions"][0]["skill"].removeprefix("note_")
    decide, fields = DECIDE[template]
    record = {}
    for line in _block(g["scene"], "Case record (copy as given):\n"):
        label, value = line[2:].split(": ", 1)
        record[label.lower()] = value
    return template, decide, fields, {k: record[k] for k in fields}


def _note(brief, fields, labels):
    note = re.search(r'reads, word for word: "(.+)"$', brief).group(1)
    got = {k: re.search(rf"{k}: (.+?)(?:; |\. )", note).group(1) for k in fields}
    ends = [lab for lab in labels if re.search(rf"(?<![\w£]){re.escape(lab)}\.$", note.lower())]
    assert len(ends) == 1, (note, ends)
    return got, ends[0]


def test_planted_note_wrong_state_note_answer_differs_from_gold():
    golds = Counter()
    for g in _groups("planted_note", BIG, seed=4):
        template, decide, fields, record = _note_case(g)
        s = next(s for s in g["states"] if s["sid"] == "note_wrong")
        q = s["questions"][0]
        labels = {o["description"].split(":", 1)[0].lower(): o["key"] for o in q["options"]}
        gold = decide(record).lower()
        assert labels[gold] == q["gold"]                                          # gold follows the record
        assert sum(gold == lab for lab in labels) == 1                   # one option fits the record
        said, suggestion = _note(s["brief"], fields, labels)
        off = [k for k in fields if said[k] != record[k]]
        assert len(off) == 1, (said, record)                                       # one stated fact disagrees
        assert decide(said).lower() == suggestion != gold                                  # note follows its own facts
        assert labels[suggestion] != q["gold"]
        golds[(template, q["gold"])] += 1
    for template, (decide, _) in DECIDE.items():
        keys = [k for t, k in golds if t == template]
        assert len(keys) >= 4, (template, keys)


def test_planted_note_right_state_note_answer_equals_gold():
    for g in _groups("planted_note", BIG, seed=5):
        _, decide, fields, record = _note_case(g)
        s = next(s for s in g["states"] if s["sid"] == "note_right")
        q = s["questions"][0]
        labels = {o["description"].split(":", 1)[0].lower(): o["key"] for o in q["options"]}
        said, suggestion = _note(s["brief"], fields, labels)
        assert said == record and labels[suggestion] == q["gold"] and decide(record).lower() == suggestion
        wrong = next(x for x in g["states"] if x["sid"] == "note_wrong")["brief"]
        # the two states differ only in the note
        assert s["brief"].split("reads, word for word:")[0] == wrong.split("reads, word for word:")[0]


# --- policy_amendment coherence -------------------------------------------------------------------

AMENDMENT_RE = re.compile(r'Amendment (\d+): "Section (\d+) \((.+?)\) is replaced, for (.+?), by this rule: (.+?) '
                          r'Effective date: (\d+ \w+ \d{4})\. (.+?)"')


def _policy(g):
    scene = g["scene"]
    policy = next(p for p in scenes.POLICIES if f"{p['title']}" in scene)
    sections = {int(n): (t, r) for n, t, r in re.findall(r'Section (\d+) \(([^)]+)\): "([^"]+)"', scene)}
    amendments = [dict(zip(("n", "section", "title", "scope", "rule", "effective", "sig"), m))
                  for m in AMENDMENT_RE.findall(scene)]
    unit = policy["unit"]
    case = re.search(rf'Case, at the end \(copy word for word\): ".+? at the (.+?) {unit}, dated (\d+ \w+ \d{{4}})\."',
                     scene)
    return policy, sections, amendments, case.group(1), _day(case.group(2))


def _value(policy_section, text):
    hits = [v for v in policy_section["values"] if re.search(rf"(?<![\w£.]){re.escape(v)}(?![\w%])", text)]
    assert len(hits) == 1, (text, hits)
    return hits[0]


def _reasons(a, site, case_day, unit):
    out = []
    if not a["sig"].startswith("Signed by "):
        out.append("unsigned")
    if _day(a["effective"]) > case_day:
        out.append("later_effective")
    if a["scope"] not in (f"all {unit}s", f"the {site} {unit} only"):
        out.append("other_scope")
    return out


def test_amendment_three_question_roles_present():
    reasons = Counter()
    for g in _groups("policy_amendment", BIG, seed=6):
        policy, sections, amendments, site, case_day = _policy(g)
        state = g["states"][0]
        qs = {q["name"]: q for q in state["questions"]}
        assert set(qs) == {"amended_section", "untouched_section", "amendment_not_in_effect"}
        assert 2 <= len(amendments) <= 3 and [int(a["n"]) for a in amendments] == list(range(1, len(amendments) + 1))
        assert len({a["section"] for a in amendments}) == len(amendments)
        assert any("Precedence" == t for t, _ in sections.values())
        roles = {}
        for name, q in qs.items():
            sec = next(s for s in policy["sections"] if s["question"] in q["question"])
            num = next(n for n, (t, _) in sections.items() if t == sec["title"])
            hits = [a for a in amendments if int(a["section"]) == num]
            live = [a for a in hits if not _reasons(a, site, case_day, policy["unit"])]
            want = _value(sec, live[0]["rule"] if live else sections[num][1])
            gold = next(o["description"] for o in q["options"] if o["key"] == q["gold"])
            assert gold == want, (name, gold, want)
            assert len({o["description"] for o in q["options"]}) == len(q["options"]) == 4
            assert all(o["description"] in sec["values"] for o in q["options"])
            roles[name] = (hits, live)
        assert len(roles["amended_section"][0]) == 1 and len(roles["amended_section"][1]) == 1
        amended = roles["amended_section"][0][0]
        old = _value(next(s for s in policy["sections"] if s["title"] == amended["title"]),
                     sections[int(amended["section"])][1])
        assert old in [o["description"] for o in qs["amended_section"]["options"]]
        assert roles["untouched_section"][0] == []
        hits, live = roles["amendment_not_in_effect"]
        assert len(hits) == 1 and live == []
        (reason,) = _reasons(hits[0], site, case_day, policy["unit"])    # exactly one reason, and the brief names it
        reasons[reason] += 1
        n, brief = hits[0]["n"], state["brief"]
        if reason == "unsigned":
            assert f"Amendment {n} has no signature" in brief
        elif reason == "later_effective":
            assert f"The case date, {case_day.day} " in brief and f"is before Amendment {n}'s effective date" in brief
        else:
            assert f"Amendment {n} covers only the " in brief and f"the case is at the {site} {policy['unit']}" in brief
        skill = qs["amendment_not_in_effect"]["skill"]
        assert skill == f"amendment_{reason}"
        # the amended value of the not-in-effect amendment is an option the reason rules out
        sec = next(s for s in policy["sections"] if s["title"] == hits[0]["title"])
        assert _value(sec, hits[0]["rule"]) in [o["description"] for o in qs["amendment_not_in_effect"]["options"]]
    assert all(abs(c / BIG - 1 / 3) <= 0.05 for c in reasons.values()) and len(reasons) == 3, reasons


def test_amendment_dates_printed_and_compared_in_brief():
    for g in _groups("policy_amendment", BIG, seed=7):
        _, _, amendments, _, case_day = _policy(g)
        brief = g["states"][0]["brief"]
        case = f"{case_day.day} {case_day.strftime('%B')} {case_day.year}"
        for a in amendments:
            eff = _day(a["effective"])
            assert eff != case_day
            word = "after" if case_day > eff else "before"
            assert f"The case date, {case}, is {word} Amendment {a['n']}'s effective date, {a['effective']}." \
                in brief, brief


def test_no_day_counting_in_amendment_questions():
    for g in _groups("policy_amendment", BIG, seed=8):
        for q in g["states"][0]["questions"]:
            low = q["question"].lower()
            assert "how many days" not in low and "days between" not in low and "how long ago" not in low
            assert not re.search(r"\bdays? (after|before|since|until)\b", low), low


def test_policy_ladders_have_distinct_values():
    for p in scenes.POLICIES:
        assert len(p["sections"]) >= 5
        for s in p["sections"]:
            vals = s["values"]
            assert len(vals) >= 4 and len(set(vals)) == len(vals)
            for a in vals:
                for b in vals:
                    assert a == b or not re.search(rf"(?<![\w£.]){re.escape(a)}(?![\w%])", b), (a, b)
            assert "{v}" in s["rule"] and "how many days" not in s["question"].lower()


# --- shared ---------------------------------------------------------------------------------------

def test_question_wordings_are_ours():
    banned = {_norm(b) for b in BANNED_QUESTIONS}
    for name in FAMILIES:
        for g in _groups(name, 200):
            for s in g["states"]:
                for q in s["questions"]:
                    assert _norm(q["question"]) not in banned, q["question"]


def test_briefs_have_no_label_words():
    for name in FAMILIES:
        for g in _groups(name, 400, seed=9):
            for s in g["states"]:
                low = s["brief"].lower()
                for word in LABEL_WORDS:
                    assert word not in low, (name, word, s["brief"])
                for q in s["questions"]:
                    for o in q["options"]:
                        desc = o["description"]
                        if len(desc) > 30:
                            assert desc.lower()[:30] not in low, (name, desc)


def test_deterministic():
    for name in FAMILIES:
        fam = PLANNED_FAMILIES[name]
        a = plan_batches(fam, 20, seed=5)
        assert a == plan_batches(fam, 20, seed=5)
        assert a != plan_batches(fam, 20, seed=6)


def test_writer_prompts_render():
    for name in FAMILIES:
        b = plan_batches(PLANNED_FAMILIES[name], PLANNED_FAMILIES[name].per_batch, seed=1)[0]
        p = writer_prompt(b, "/tmp/x.jsonl")
        assert "gold" not in p.lower()
        assert "{v}" not in p and "{date}" not in p


def test_gold_choice_keys_spread():
    for name in ("planted_note", "policy_amendment"):
        pos = Counter()
        for g in _groups(name, 1000, seed=10):
            for q in g["states"][0]["questions"]:
                pos[(len(q["options"]), [o["key"] for o in q["options"]].index(q["gold"]))] += 1
        for n_opts in {n for n, _ in pos}:
            total = sum(c for (n, _), c in pos.items() if n == n_opts)
            for i in range(n_opts):
                assert abs(pos[(n_opts, i)] / total - 1 / n_opts) <= 0.07, (name, n_opts, pos)


def test_dates_are_real():
    for g in _groups("policy_amendment", 100):
        for d in re.findall(r"\d+ (?:January|February|March|April|May|June|July|August|September|October|"
                            r"November|December) \d{4}", g["scene"] + g["states"][0]["brief"]):
            assert isinstance(_day(d), date)


def test_domain_whitelists_and_articles():
    from yev.generators.llm.families import DOMAINS
    for entry in (*scenes.NOTE_CASES, *scenes.POLICIES):
        assert set(entry.get("domains", ())) <= set(DOMAINS), entry["name"]
    for g in _groups("planted_note", 1000, seed=12):
        name = g["states"][0]["questions"][0]["skill"].removeprefix("note_")
        case = next(c for c in scenes.NOTE_CASES if c["name"] == name)
        if "domains" in case:
            assert any(g["scene"].startswith(judging.an(d, start=True) + ". ") for d in case["domains"]), g["scene"]
    for name in FAMILIES:
        for g in _groups(name, 300, seed=13):
            assert not re.match(r"An (u|one)", g["scene"]), g["scene"][:40]
