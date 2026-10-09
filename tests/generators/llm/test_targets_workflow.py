"""Workflow-decision planned families: invoice_line, cs_scales_actions, security_context.

The coherence tests read the written facts (scene and brief) the way a blind reader would, with a reader written
here and not shared with the planner, and check that the reader reaches every gold the planner set.
"""

import re
from collections import Counter
from datetime import datetime

from yev.generators.llm.planned import PlannedFamily, plan_batches, writer_prompt
from yev.generators.llm.targets import PLANNED_FAMILIES
from yev.generators.llm.targets import workflow
from yev.generators.llm.targets import workflow_scenes as scenes

N = 500
BIG = 2000   # coherence tests sample at least this many groups per family
FAMILIES = ("invoice_line", "cs_scales_actions", "security_context")
SCOPE_SHARES = {"priced_unit": 55, "component_also_billed": 7, "extra_approved": 7, "extra_unapproved": 15,
                "not_ours": 8, "unsure": 8}
COMPLETION_SHARES = {"confirmed": 50, "less": 15, "not_done": 8, "vendor_only": 8, "none": 12, "unsure": 7}
# Benchmark wordings, hand-written from the gap report's descriptions. They must never be our questions.
BANNED_QUESTIONS = (
    "Does the quoted figure count the delivered quantity in the line's units?",
    "Does the context explain the activity?",
)
# Words a writer would copy into a state as a verdict; briefs give facts only.
LABEL_WORDS = ("priced unit", "also billed", "not ours", "unapproved", "out of scope", "in scope", "unsure",
               "vendor's word", "counts the", "does not count", "accounted for", "explains the", "legitimate",
               "suspicious", "benign", "malicious", "urgency", "frustration", "level ", "should ")


def _groups(name, n=N, seed=0):
    return [g for b in plan_batches(PLANNED_FAMILIES[name], n, seed) for g in b["groups"]]


def _norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def _block(text, heading):
    """The "- " rows under `heading (copy ...):` up to the next blank line."""
    body = re.split(re.escape(heading) + r" \(copy[^)]*\):\n", text, maxsplit=1)[1].split("\n\n", 1)[0]
    return [line[2:] for line in body.splitlines() if line.startswith("- ")]


def _fields(row):
    """("Line 3", {"Site": ..., ...}) from a " | " row; parts without "Key: " go to the positional list."""
    pos, kv = [], {}
    for part in row.split(" | "):
        m = re.match(r"^([A-Z][A-Za-z ]*?): (.*)$", part)
        if m:
            kv[m.group(1)] = m.group(2)
        else:
            pos.append(part)
    return pos, kv


# --- registry -------------------------------------------------------------------------------------

def test_families_registered_with_table_sizes():
    want = {"invoice_line": ((2000, 8000), 4), "cs_scales_actions": ((500, 3000), 13),
            "security_context": ((600, 3000), 6)}
    for name, (tokens, per_batch) in want.items():
        fam = PLANNED_FAMILIES[name]
        assert isinstance(fam, PlannedFamily) and fam.name == name
        assert fam.tokens == tokens and fam.per_batch == per_batch, name
        states = len(_groups(name, 1)[0]["states"])
        assert per_batch == max(1, 24000 // (states * (sum(tokens) // 2)))
    assert workflow.INVOICE_LINE is PLANNED_FAMILIES["invoice_line"]
    assert workflow.CS_SCALES_ACTIONS is PLANNED_FAMILIES["cs_scales_actions"]
    assert workflow.SECURITY_CONTEXT is PLANNED_FAMILIES["security_context"]


def test_group_shapes():
    for g in _groups("invoice_line", 200):
        assert len(g["states"]) == 1
        qs = g["states"][0]["questions"]
        assert len(qs) == 6 and [q["type"] for q in qs].count("noul") == 2
    for g in _groups("cs_scales_actions", 200):
        assert len(g["states"]) == 1
        assert [q["type"] for q in g["states"][0]["questions"]] == ["score", "score", "choice"]
    for g in _groups("security_context", 200):
        assert sorted(s["sid"] for s in g["states"]) == ["covered", "fault"]
        for s in g["states"]:
            assert [q["type"] for q in s["questions"]] == ["noul", "choice"]


# --- invoice reader -------------------------------------------------------------------------------

def _trade(packet):
    po = [_fields(r)[0][1] for r in _block(packet, "Purchase order items") if r.startswith("Item ")]
    hits = [t for t in scenes.TRADES if all(any(p.startswith(i[0]) for p in po) for i in t["items"])]
    assert len(hits) == 1, po
    return hits[0]


def _packet(g):
    s = g["states"][0]
    text = g["scene"] + "\n\n" + s["brief"]
    header = dict(r.split(": ", 1) for r in _block(text, "Invoice header"))
    lines = {}
    for r in _block(text, "Invoice lines"):
        pos, kv = _fields(r)
        lines[int(pos[0].removeprefix("Line "))] = {"text": pos[1], **kv}
    records = {}
    for r in _block(text, "Evidence records"):
        pos, kv = _fields(r)
        records[pos[0]] = kv
    changes = [_fields(r) for r in _block(text, "Change orders and change requests")]
    po = [_fields(r) for r in _block(text, "Purchase order items")]
    return {"text": text, "header": header, "lines": lines, "records": records, "changes": changes, "po": po,
            "trade": _trade(text), "questions": s["questions"], "brief": s["brief"]}


def _item_of(p, text):
    for k, item in enumerate(p["trade"]["items"]):
        if text in (item[0], item[1]):
            return k
    return None


def _ref(p, n):
    """The Item field our records (and the vendor's) use for invoice line n."""
    line = p["lines"][n]
    k = _item_of(p, line["text"])
    if line["Site"] != p["header"]["Our site"]:
        return f"{line['text']} (delivered to {line['Site']})"
    return f"PO item {k + 1}: {p['trade']['items'][k][0]}" if k is not None else line["text"]


def _ours(rec):
    return rec["Kept by"] == "our team"


def _read_scope(p, n):
    line, trade = p["lines"][n], p["trade"]
    if line["Site"] != p["header"]["Our site"]:
        return "not_ours"
    if line["text"] == trade["open"][2]:
        assert any(r.startswith(f"Open point: whether this order covers {trade['open'][1]} ")
                   for r in _block(p["text"], "Purchase order items"))
        return "unsure"
    if line["text"] == trade["part"][2]:
        parent = trade["items"][trade["part"][0]]
        others = [m for m, ln in p["lines"].items() if m != n and ln["text"] in parent[:2]]
        assert others, "component billed without its parent item"
        assert any(trade["part"][1] in pos[1] for pos, _ in p["po"] if pos[0].startswith("Item "))
        return "component_also_billed"
    k = _item_of(p, line["text"])
    if k is not None:
        po_row = next(kv for pos, kv in p["po"] if pos[0] == f"Item {k + 1}")
        assert po_row["Rate per unit"] == line["Rate per unit"] and po_row["Unit"] == line["Unit"]
        return "priced_unit"
    approved = [kv for pos, kv in p["changes"] if kv.get("Adds") == line["text"] and "Approved by" in kv]
    if approved:
        assert approved[0]["Rate per unit"] == line["Rate per unit"]
        return "extra_approved"
    return "extra_unapproved"


def _read_completion(p, n):
    line, ref, period = p["lines"][n], _ref(p, n), p["header"]["Service period"]
    mine = [r for r in p["records"].values() if r["Item"] == ref]
    counting = [r for r in mine if _ours(r) and r["Unit"] == line["Unit"] and r["Period"] == period]
    if len({r["Figure"] for r in counting}) >= 2:
        return "unsure"
    if counting:
        fig, qty = int(counting[0]["Figure"]), int(line["Quantity"])
        return "confirmed" if fig == qty else "not_done" if fig == 0 else "less" if fig < qty else "over"
    in_period = [r for r in mine if r["Period"] == period]
    if any(not _ours(r) and r["Unit"] == line["Unit"] for r in in_period):
        assert all(not _ours(r) for r in in_period)
        return "vendor_only"
    assert not in_period, "a record for the period exists, so the line cannot be read as having none"
    return "none"


def _read_evidence(p, n, rid):
    rec, line = p["records"][rid], p["lines"][n]
    return "yes" if (_ours(rec) and rec["Item"] == _ref(p, n) and rec["Unit"] == line["Unit"]
                     and rec["Period"] == p["header"]["Service period"]) else "no"


def _line_of(q):
    return int(re.search(r"invoice line (\d+)", q["question"]).group(1))


def _record_of(q):
    return re.search(r"evidence record (E\d+)", q["question"]).group(1)


def test_invoice_golds_follow_the_written_packet():
    seen = Counter()
    for g in _groups("invoice_line", BIG, seed=1):
        p = _packet(g)
        asked = sorted({_line_of(q) for q in p["questions"]})
        assert len(asked) == 2 and all(n in p["lines"] for n in asked)
        assert 4 <= len(p["lines"]) <= 6
        for q in p["questions"]:
            n = _line_of(q)
            kind = q["name"].rsplit("_", 1)[1]
            if kind == "scope":
                assert q["gold"] == _read_scope(p, n), (q["gold"], p["lines"][n])
            elif kind == "completion":
                assert q["gold"] == _read_completion(p, n), (q["gold"], p["lines"][n], p["records"])
            else:
                assert kind == "evidence"
                assert q["gold"] == _read_evidence(p, n, _record_of(q)), (q, p["records"])
            seen[(kind, q["gold"])] += 1
            if q["name"].endswith("_scope") and q["gold"] != "unsure":
                # the open point names no item, and no other gold bills the work it leaves open
                assert p["lines"][n]["text"] != p["trade"]["open"][2]
                assert not any("Open point on item" in r for r in _block(p["text"], "Purchase order items"))
            if q["name"].endswith("_scope") and q["gold"] in ("extra_approved", "extra_unapproved", "unsure"):
                ref = _ref(p, n)
                assert all(r["Period"] == p["header"]["Service period"] for r in p["records"].values()
                           if r["Item"] == ref)
        # the two asked lines are independent: neither line's evidence question points at the other's records
        a, b = asked
        for q in p["questions"]:
            if q["name"].endswith("_evidence"):
                other = b if _line_of(q) == a else a
                assert p["records"][_record_of(q)]["Item"] != _ref(p, other)
    assert all(seen[("scope", k)] for k in SCOPE_SHARES) and all(seen[("completion", k)] for k in COMPLETION_SHARES)


def test_invoice_no_record_never_with_counting_evidence():
    for g in _groups("invoice_line", BIG, seed=2):
        p = _packet(g)
        for q in p["questions"]:
            if q["name"].endswith("_completion") and q["gold"] == "none":
                n = _line_of(q)
                period = p["header"]["Service period"]
                assert not [r for r in p["records"].values() if r["Item"] == _ref(p, n) and r["Period"] == period]
                ev = next(e for e in p["questions"] if e["name"].endswith("_evidence") and _line_of(e) == n)
                assert ev["gold"] == "no"
            if q["name"].endswith("_completion") and q["gold"] in ("vendor_only", "none"):
                n = _line_of(q)
                assert not any(_read_evidence(p, n, rid) == "yes" for rid in p["records"])


def test_invoice_unsure_has_a_defined_trigger():
    """unsure (scope) only with an open point on exactly that line's item; unsure (completion) only when two of our
    records for the period disagree; neither trigger shows up for any other gold."""
    for g in _groups("invoice_line", BIG, seed=3):
        p = _packet(g)
        open_text = p["trade"]["open"][2]
        for q in p["questions"]:
            n = _line_of(q)
            if q["name"].endswith("_scope"):
                assert (q["gold"] == "unsure") == (p["lines"][n]["text"] == open_text and
                                                    p["lines"][n]["Site"] == p["header"]["Our site"])
            if q["name"].endswith("_completion"):
                ref, period, unit = _ref(p, n), p["header"]["Service period"], p["lines"][n]["Unit"]
                counting = [r for r in p["records"].values() if _ours(r) and r["Item"] == ref
                            and r["Period"] == period and r["Unit"] == unit]
                assert (q["gold"] == "unsure") == (len(counting) == 2)
                assert len(counting) <= 2


def test_invoice_hard_negatives_present():
    alias = plain = near = other_extra = pending = told = 0
    for g in _groups("invoice_line", BIG, seed=4):
        p = _packet(g)
        for q in p["questions"]:
            if not q["name"].endswith("_scope"):
                continue
            text = p["lines"][_line_of(q)]["text"]
            if q["gold"] == "priced_unit":
                k = _item_of(p, text)
                alias += text == p["trade"]["items"][k][1]
                plain += text == p["trade"]["items"][k][0]
            if q["gold"] in ("extra_approved", "extra_unapproved"):
                near += text == p["trade"]["near"][2]
                other_extra += text != p["trade"]["near"][2]
            if q["gold"] == "extra_unapproved":
                pending += any(kv.get("Adds") == text and kv.get("Status") for _, kv in p["changes"])
                told += text in p["brief"].split("Messages", 1)[1]
    assert alias / (alias + plain) > 0.4 and near / (near + other_extra) > 0.35
    assert pending > 50 and told > 50


def test_invoice_records_print_unit_and_period():
    for g in _groups("invoice_line", 300, seed=5):
        p = _packet(g)
        for rid, r in p["records"].items():
            assert re.fullmatch(r"E\d+", rid) and r["Figure"].isdigit()
            assert r["Unit"] and r["Period"] and r["Kept by"] and r["Source"]
        for line in p["lines"].values():
            amount = float(line["Amount"].replace(",", ""))
            assert abs(amount - int(line["Quantity"]) * float(line["Rate per unit"].replace(",", ""))) < 0.01


def test_invoice_planned_length_reaches_minimum():
    """Copied tables plus the smallest amounts the length plan asks for reach the 2,000-token ingest minimum."""
    from yev.generators.llm.documents import WORDS_PER_TOKEN
    from yev.generators.llm.targets import workflow_invoice as wi
    lo = PLANNED_FAMILIES["invoice_line"].tokens[0]
    groups = _groups("invoice_line", 1000, seed=11)
    ok = 0
    for g in groups:
        brief = g["states"][0]["brief"]
        n_msgs = len(re.findall(r"^\d+\. ", brief.split("Messages (", 1)[1].split("\n\n", 1)[0], re.M))
        assert n_msgs >= 3
        plan = brief.split("Length plan: ", 1)[1]
        assert f"{wi.TERMS_PARAGRAPHS[0]} to {wi.TERMS_PARAGRAPHS[1]} paragraphs" in plan
        assert f"{wi.MESSAGE_WORDS[0]} to {wi.MESSAGE_WORDS[1]} words" in plan
        words = wi.TERMS_PARAGRAPHS[0] * wi.TERMS_WORDS[0] + n_msgs * wi.MESSAGE_WORDS[0]
        est = len(g["scene"] + brief) / 4 + words / WORDS_PER_TOKEN
        ok += est >= lo
    assert ok / len(groups) >= 0.95, ok


# --- cs reader ------------------------------------------------------------------------------------

def _cue_level(brief, cues, prefix):
    sentence = brief.split(prefix, 1)[1].split(". ", 1)[0]
    hits = set()
    for level, templates in cues.items():
        for t in templates:
            pat = re.escape(t)
            pat = re.sub(r"\\\{\w+\\\}", ".+?", pat)
            if re.fullmatch(pat, sentence.rstrip(".")):
                hits.add(level)
    assert len(hits) == 1, (sentence, hits)
    return hits.pop()


def _cs_facts(g):
    brief = g["states"][0]["brief"]
    u = _cue_level(brief, scenes.URGENCY_CUES, "Timing: the customer ")
    f = _cue_level(brief, scenes.FRUSTRATION_CUES, "Mood: the customer ")
    if u in scenes.URGENCY_DAYS:
        days = int(re.search(r"(\d+) days", brief.split("Timing: ", 1)[1].split(". ", 1)[0]).group(1))
        lo, hi = scenes.URGENCY_DAYS[u]
        assert lo <= days <= hi
    facts = {"person": "asks to speak to a member of staff instead" in brief,
             "owes": "will send it later" in brief, "unknown": "that they do not recognise" in brief,
             "u2": u >= 2, "u3": u >= 3, "u4": u == 4, "f3": f >= 3, "f4": f == 4}
    assert ("never asks to speak to a member of staff" in brief) != facts["person"]
    assert ("gives every detail the assistant asks for" in brief) != facts["owes"]
    assert ("mentions no payment they do not recognise" in brief) != facts["unknown"]
    return u, f, facts


def _cs_rules(scene):
    lines = scene.split("Team rules (copy word for word, in this order):\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert lines[0].startswith("Work down this list")
    conds = {v: k for k, v in scenes.CS_CONDITIONS.items()}
    acts = {v: k for k, v in scenes.CS_ACTION_PHRASES.items()}
    rules, fallback = [], None
    for i, line in enumerate(lines[1:], 1):
        assert line.startswith(f"{i}. ")
        body = line.split(". ", 1)[1].rstrip(".")
        if body.startswith("If none of the above holds, "):
            fallback = acts[body.removeprefix("If none of the above holds, ")]
        else:
            hit = [c for c in conds if body.startswith(f"If {c}, ")]
            assert len(hit) == 1, body
            rules.append((conds[hit[0]], acts[body.removeprefix(f"If {hit[0]}, ")]))
    return rules, fallback


def _apply(rules, fallback, facts):
    return next((a for c, a in rules if facts[c]), fallback)


def test_cs_action_follows_policy_table():
    variants = Counter()
    for g in _groups("cs_scales_actions", BIG, seed=6):
        qs = {q["name"]: q for q in g["states"][0]["questions"]}
        u, f, facts = _cs_facts(g)
        assert qs["urgency"]["gold"] == str(u) and qs["frustration"]["gold"] == str(f)
        rules, fallback = _cs_rules(g["scene"])
        assert qs["next_action"]["gold"] == _apply(rules, fallback, facts)
        named = [k for k, (r, fb) in scenes.CS_POLICIES.items() if list(r) == rules and fb == fallback]
        assert len(named) == 1
        variants[named[0]] += 1
        # the same facts give a different step under at least one other variant
        outs = {_apply(list(r), fb, facts) for r, fb in scenes.CS_POLICIES.values()}
        assert len(outs) >= 2, facts
    assert set(variants) == set(scenes.CS_POLICIES)
    assert all(abs(c / BIG - 1 / 3) <= 0.05 for c in variants.values())


def test_cs_levels_have_concrete_anchors():
    g = _groups("cs_scales_actions", 1)[0]
    qs = {q["name"]: q for q in g["states"][0]["questions"]}
    for name in ("urgency", "frustration"):
        opts = qs[name]["options"]
        assert [o["key"] for o in opts] == ["0", "1", "2", "3", "4"]
        assert len({o["description"] for o in opts}) == 5
    u = [o["description"] for o in qs["urgency"]["options"]]
    assert "more than seven days" in u[1] and "three to seven days" in u[2] and "today or by tomorrow" in u[3]
    assert scenes.URGENCY_DAYS == {1: (10, 30), 2: (3, 7)}   # level 2 starts after "today or by tomorrow"
    fr = [o["description"] for o in qs["frustration"]["options"]]
    assert "first" in fr[1] and "second" in fr[2] and "third or later" in fr[3] and "cancel" in fr[4]


def test_cs_pairs_share_a_writer_batch():
    fam = PLANNED_FAMILIES["cs_scales_actions"]
    for seed in (0, 1, 2):
        where = {}
        for b in plan_batches(fam, BIG, seed):
            for g in b["groups"]:
                where.setdefault(g["scene"], set()).add(b["batch_id"])
        assert all(len(v) == 1 for v in where.values())
        assert sum(1 for b in plan_batches(fam, 40, seed) for _ in b["groups"]) == 40


def test_cs_near_neighbour_pairs():
    groups = _groups("cs_scales_actions", BIG, seed=7)
    by_scene = {}
    for g in groups:
        by_scene.setdefault(g["scene"], []).append(g)
    paired = [gs for gs in by_scene.values() if len(gs) > 1]
    assert all(len(gs) == 2 for gs in paired)
    assert abs(2 * len(paired) / len(groups) - 0.30) <= 0.05
    for a, b in paired:
        ua, fa, xa = _cs_facts(a)
        ub, fb, xb = _cs_facts(b)
        assert sorted((abs(ua - ub), abs(fa - fb))) == [0, 1], (ua, ub, fa, fb)
        assert {k: xa[k] for k in ("person", "owes", "unknown")} == {k: xb[k] for k in ("person", "owes", "unknown")}
        ba, bb = a["states"][0]["brief"].split(". "), b["states"][0]["brief"].split(". ")
        assert len(ba) == len(bb) and sum(x != y for x, y in zip(ba, bb)) == 1


# --- security reader ------------------------------------------------------------------------------

def _dt(s):
    return datetime.strptime(s, "%d %B %Y %H:%M")


def _span(label, value):
    """(start, end) datetimes of a record's time field, or None when it gives no span."""
    if label == "Presence":
        m = re.fullmatch(r"badged in (\d\d:\d\d), badged out (\d\d:\d\d) on (.+)", value)
        if not m:
            assert re.fullmatch(r"no badge entry recorded on .+", value), value
            return None
        return _dt(f"{m.group(3)} {m.group(1)}"), _dt(f"{m.group(3)} {m.group(2)}")
    m = re.fullmatch(r"(\d+ \w+ \d{4}) (\d\d:\d\d) to (\d\d:\d\d)", value)
    if m:
        return _dt(f"{m.group(1)} {m.group(2)}"), _dt(f"{m.group(1)} {m.group(3)}")
    m = re.fullmatch(r"(\d+ \w+ \d{4}) to (\d+ \w+ \d{4})", value)
    return _dt(f"{m.group(1)} 00:00"), _dt(f"{m.group(2)} 23:59")


def _sec(g, s):
    text = g["scene"] + "\n\n" + s["brief"]
    alert = dict(r.split(": ", 1) for r in _block(text, "Alert"))
    kind = next(k for k in scenes.SEC_KINDS if k["rule"] == alert["Rule"])
    rows = [r for r in _block(text, "Context records") if "as given in the brief" not in r]
    rows += _block(s["brief"], "Record")
    return alert, kind, [_fields(r) for r in rows]


def _covers(alert, kind, rec):
    pos, kv = rec
    if len(pos) < 2 or pos[1] != kind["label"]:
        return False
    user = alert["User"].split(" (", 1)[0]
    span = _span(kind["time"], kv[kind["time"]])
    status_ok = kind["status"] is None or kv[kind["status"]] == "approved"
    return (kv[kind["user"]] == user and kv[kind["place"]] == alert[kind["alert_place"]] and span is not None
            and span[0] <= _dt(alert["Time of activity"]) <= span[1] and status_ok)


def _playbook(scene):
    body = scene.split("Playbook (copy word for word, in this order):\n", 1)[1].split("\n\n", 1)[0]
    acts = {v: k for k, v in scenes.SEC_ACTION_PHRASES.items()}
    out = {}
    for line in body.splitlines():
        m = re.fullmatch(r"- (Accounted for|Not accounted for), band (\w+): (.+)\.", line)
        if m:
            out[(m.group(1) == "Accounted for", m.group(2))] = acts[m.group(3)]
    assert len(out) == 8
    return out


def test_security_fault_state_differs_by_one_field():
    faults = Counter()
    for g in _groups("security_context", BIG, seed=8):
        st = {s["sid"]: s for s in g["states"]}
        rec = {sid: _block(s["brief"], "Record") for sid, s in st.items()}
        assert len(rec["covered"]) == len(rec["fault"]) == 1
        a, b = rec["covered"][0].split(" | "), rec["fault"][0].split(" | ")
        assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1, (a, b)
        before, after = st["covered"]["brief"].split(rec["covered"][0]), st["fault"]["brief"].split(rec["fault"][0])
        assert before == after   # the briefs differ only in that record
        faults[st["fault"]["questions"][0]["skill"]] += 1
    assert len(faults) >= 6


def test_security_golds_follow_records_and_playbook():
    actions = Counter()
    for g in _groups("security_context", BIG, seed=9):
        book = _playbook(g["scene"])
        assert scenes.SEC_DEFINITION in g["scene"]
        for s in g["states"]:
            alert, kind, recs = _sec(g, s)
            assert 9 <= len(recs) <= 15
            covering = [r for r in recs if _covers(alert, kind, r)]
            assert len(covering) == (1 if s["sid"] == "covered" else 0), (s["sid"], covering)
            user = alert["User"].split(" (", 1)[0]
            place = alert[kind["alert_place"]]
            key_id = _block(s["brief"], "Record")[0].split(" | ", 1)[0]
            for pos, kv in recs:   # every other record is irrelevant: never the alert's user or place
                if pos[0] != key_id and pos[1] not in ("Asset register", "Staff directory"):
                    assert user not in kv.values() and place not in kv.values(), (pos, kv)
                elif pos[0] != key_id:   # consistent entries: the alert's user (and host), but no approval or time
                    assert not {"Status", kind["time"]} & set(kv), (pos, kv)
            qs = {q["name"]: q for q in s["questions"]}
            assert qs["accounted_for"]["gold"] == ("yes" if s["sid"] == "covered" else "no")
            band = alert["Severity band"]
            assert qs["next_step"]["gold"] == book[(s["sid"] == "covered", band)]
            actions[qs["next_step"]["gold"]] += 1
    assert set(actions) == set(scenes.SEC_ACTION_PHRASES)


# --- shared ---------------------------------------------------------------------------------------

def test_family_label_balance():
    rows = [q for g in _groups("invoice_line", BIG) for q in g["states"][0]["questions"]]
    for kind, want in (("scope", SCOPE_SHARES), ("completion", COMPLETION_SHARES)):
        got = Counter(q["gold"] for q in rows if q["name"].endswith(f"_{kind}"))
        total = sum(got.values())
        assert total == 2 * BIG
        for k, share in want.items():
            assert abs(100 * got[k] / total - share) <= 5, (kind, k, got[k] / total)
    ev = [q["gold"] for q in rows if q["name"].endswith("_evidence")]
    assert abs(ev.count("yes") / len(ev) - 0.40) <= 0.05
    sec = [q["gold"] for g in _groups("security_context") for s in g["states"] for q in s["questions"]
           if q["name"] == "accounted_for"]
    assert sec.count("yes") / len(sec) == 0.5
    cs = [q for g in _groups("cs_scales_actions", BIG) for q in g["states"][0]["questions"]]
    for name in ("urgency", "frustration"):
        got = Counter(q["gold"] for q in cs if q["name"] == name)
        assert all(abs(got[str(k)] / BIG - 0.20) <= 0.05 for k in range(5)), (name, got)


def test_no_arithmetic_questions():
    banned = re.compile(r"\b(sum|total|add up|difference|how many days|number of days|subtract|minus|"
                        r"calculate|compute)\b", re.I)
    for name in FAMILIES:
        for g in _groups(name, 300):
            for s in g["states"]:
                for q in s["questions"]:
                    assert not banned.search(q["question"]), q["question"]
                    for o in q["options"]:
                        assert not banned.search(o["description"]), o["description"]


def test_choice_options_shuffled_half_the_time():
    for name, qname in (("invoice_line", "_scope"), ("cs_scales_actions", "next_action"),
                        ("security_context", "next_step")):
        orders = Counter()
        for g in _groups(name, 600):
            q = next(q for q in g["states"][0]["questions"] if q["name"].endswith(qname))
            orders[tuple(o["key"] for o in q["options"])] += 1
        top = orders.most_common(1)[0][1] / 600
        assert 0.4 <= top <= 0.65 and len(orders) > 10, (name, top)


def test_question_wordings_are_ours():
    banned = {_norm(b) for b in BANNED_QUESTIONS}
    for name in FAMILIES:
        for g in _groups(name, 200):
            for s in g["states"]:
                for q in s["questions"]:
                    assert _norm(q["question"]) not in banned


def test_briefs_have_no_label_words():
    for name in FAMILIES:
        for g in _groups(name, 300):
            for s in g["states"]:
                low = s["brief"].lower()
                for word in LABEL_WORDS:
                    assert word not in low, (name, word, s["brief"])
                for q in s["questions"]:
                    for o in q["options"]:
                        assert o["description"].lower()[:40] not in low


def test_deterministic():
    for name in FAMILIES:
        fam = PLANNED_FAMILIES[name]
        a = plan_batches(fam, 20, seed=5)
        assert a == plan_batches(fam, 20, seed=5)
        assert a != plan_batches(fam, 20, seed=6)


def test_writer_prompts_render():
    for name in FAMILIES:
        fam = PLANNED_FAMILIES[name]
        b = plan_batches(fam, fam.per_batch, seed=1)[0]
        p = writer_prompt(b, "/tmp/x.jsonl")
        assert "gold" not in p.lower() and "{" not in p.split("## Output format")[0]
