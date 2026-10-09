import itertools
import re
from collections import Counter

from yev.generators.llm.planned import PlannedFamily, plan_batches, writer_prompt
from yev.generators.llm.targets import PLANNED_FAMILIES
from yev.generators.llm.targets import conversation as conv
from yev.generators.llm.targets import conversation_scenes as scenes
from yev.schema import Option

N = 500
FAMILIES = ("disclosure_gate", "grounded_offer", "persona_style_rule")
YES_SHARE = {"disclosure_gate": 0.45, "grounded_offer": 0.50, "persona_style_rule": 0.50}
# Benchmark wordings, hand-written from the gap reports. They guard against copying and must never be questions.
BANNED_QUESTIONS = (
    "Does every agent reply in the transcript follow the policy?",
    "Does every assistant reply in the transcript follow the policy?",
    "Did the agent follow the policy?",
)
# Words a writer would copy into a transcript as a verdict; briefs describe events only.
LABEL_WORDS = ("violat", "broke", "break", "breach", "not allowed", "against the", "entitled", "permitted",
               "forbid", "rule", "kept", "follows", "policy")
REQUESTERS = ("staff", "outsider", "claimed", "self")
ITEMS = ("work_phone", "work_email", "desk_location", "whereabouts", "schedule")


def _groups(name, n=N, seed=0):
    return [g for b in plan_batches(PLANNED_FAMILIES[name], n, seed) for g in b["groups"]]


def _norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def _policy(scene):
    """The numbered rules the scene gives, in order."""
    block = scene.split("Operator policy (copy these rules word for word, in this order):\n", 1)[1]
    block = block.split("\n\n", 1)[0]
    lines = block.splitlines()
    for i, line in enumerate(lines, 1):
        assert line.startswith(f"{i}. "), (i, line)
    return [line.split(". ", 1)[1] for line in lines]


def _data_block(scene, heading):
    block = scene.split(f"{heading} (copy as given):\n", 1)[1].split("\n\n", 1)[0]
    return [line[2:] for line in block.splitlines()]


def _offers(brief):
    span = brief.split("In its replies it offers exactly these, by name: ", 1)[1].split(". ", 1)[0]
    return re.findall(r'"([^"]+)"', span)


def test_families_registered_with_table_sizes():
    want = {"disclosure_gate": 8, "grounded_offer": 13, "persona_style_rule": 13}
    for name, per_batch in want.items():
        fam = PLANNED_FAMILIES[name]
        assert isinstance(fam, PlannedFamily) and fam.name == name
        assert fam.tokens == (300, 1500) and fam.per_batch == per_batch, name
    assert conv.DISCLOSURE_GATE is PLANNED_FAMILIES["disclosure_gate"]
    assert conv.GROUNDED_OFFER is PLANNED_FAMILIES["grounded_offer"]
    assert conv.PERSONA_STYLE_RULE is PLANNED_FAMILIES["persona_style_rule"]


def test_family_label_balance():
    sizes = {"disclosure_gate": 3, "grounded_offer": 2, "persona_style_rule": 2}
    for name in FAMILIES:
        rows = [q for g in _groups(name) for s in g["states"] for q in s["questions"]]
        assert len(rows) == sizes[name] * N and all(q["type"] == "noul" for q in rows)
        yes = sum(q["gold"] == "yes" for q in rows) / len(rows)
        assert abs(yes - YES_SHARE[name]) <= 0.05, (name, yes)


def test_policy_exports():
    yes, no = conv.POLICY_OPTIONS
    assert isinstance(yes, Option) and (yes.key, no.key) == ("yes", "no")
    assert len(conv.POLICY_QUESTIONS) == 5 and len(set(conv.POLICY_QUESTIONS)) == 5
    assert "operator policy" in conv.POLICY_QUESTIONS[0]
    qs = Counter(g["states"][0]["questions"][0]["question"] for g in _groups("disclosure_gate"))
    assert set(qs) == set(conv.POLICY_QUESTIONS)
    assert abs(qs[conv.POLICY_QUESTIONS[0]] / N - 0.5) <= 0.06


def test_states_share_question_and_scene():
    for name in FAMILIES:
        for g in _groups(name, 200):
            qs = [s["questions"][0] for s in g["states"]]
            assert all(q["question"] == qs[0]["question"] and q["options"] == qs[0]["options"] for q in qs)
            assert qs[0]["question"] in conv.POLICY_QUESTIONS
            briefs = [s["brief"] for s in g["states"]]
            assert len(set(briefs)) == len(briefs)
            assert len({q["gold"] for q in qs}) == 2, (name, g["gid"])


def test_disclosure_gold_follows_entitlement_table():
    # the table the policy wording encodes: signed-in staff get everything but live whereabouts, a person gets all
    # of their own details, and anyone not signed in (whatever they claim) gets none of the five
    want = {(r, i): (r == "self") or (r == "staff" and i != "whereabouts") for r in REQUESTERS for i in ITEMS}
    assert conv.ENTITLED == want
    seen = Counter()
    for g in _groups("disclosure_gate", 2000, seed=3):
        item = g["states"][0]["questions"][0]["skill"].removeprefix("disclosure_")
        assert item in ITEMS
        confirm = "asks the assistant to confirm it" in g["scene"]
        assert not confirm or item in ("desk_location", "whereabouts")
        actions = Counter()
        for s in g["states"]:
            req, action = s["sid"].rsplit("_", 1)
            gold = s["questions"][0]["gold"]
            assert gold == ("yes" if action == "gives" and not conv.ENTITLED[(req, item)] else "no"), s
            assert scenes.SESSION_LINES[req].split("{")[0] in s["brief"], s["brief"]
            if action == "withholds":   # withholding is planned only where giving would go against the policy
                assert not conv.ENTITLED[(req, item)], s
            actions[action] += 1
            seen[(req, item, action, confirm)] += 1
        # hard negative: every group holds a disclosure to someone the table entitles
        assert any(conv.ENTITLED[(s["sid"].rsplit("_", 1)[0], item)] and s["sid"].endswith("_gives")
                   for s in g["states"])
        assert actions["gives"] >= 2
    for req, item in itertools.product(REQUESTERS, ITEMS):
        assert seen[(req, item, "gives", False)], (req, item)
        if not conv.ENTITLED[(req, item)]:
            assert seen[(req, item, "withholds", False)], (req, item)
    for item in ("desk_location", "whereabouts"):
        assert seen[("self", item, "gives", True)] and seen[("outsider", item, "gives", True)]


def test_disclosure_policy_encodes_the_table():
    for g in _groups("disclosure_gate", 300):
        rules = _policy(g["scene"])
        ladders = [w for w in scenes.LADDERS if all(r in rules for r in w)]
        assert len(ladders) == 1, rules
        # directory values the briefs use are all in the directory block
        directory = "\n".join(_data_block(g["scene"], "Staff directory"))
        for s in g["states"]:
            if s["sid"].endswith("_gives") and "asks the assistant to confirm it" not in g["scene"]:
                value = re.search(r'gives [^"]+: "([^"]+)"', s["brief"]).group(1)
                assert value in directory, (value, directory)


def test_policy_has_boilerplate_rules():
    boiler = set(scenes.BOILERPLATE_RULES)
    for name in FAMILIES:
        for g in _groups(name, 2000, seed=4):
            rules = _policy(g["scene"])
            assert 3 <= len(rules) <= 12, (name, len(rules))
            quiet = [i for i, r in enumerate(rules, 1) if r in boiler]
            assert len(quiet) >= 2, (name, rules)
            listed = re.search(r"Rules ([\d, and]+) are about topics that never come up", g["scene"]).group(1)
            assert [int(x) for x in re.findall(r"\d+", listed)] == quiet


def test_grounded_offer_absent_item_not_in_data_block():
    kinds = Counter()
    for g in _groups("grounded_offer", 2000, seed=5):
        heading = next(h for h in scenes.DATA_HEADINGS if f"{h} (copy as given):" in g["scene"])
        data = _data_block(g["scene"], heading)
        variant = g["states"][0]["questions"][0]["skill"]
        kinds[variant] += 1
        st = {s["sid"]: s for s in g["states"]}
        assert set(st) == {"grounded", "ungrounded"}
        assert st["grounded"]["questions"][0]["gold"] == "no" and st["ungrounded"]["questions"][0]["gold"] == "yes"
        grounded, ungrounded = _offers(st["grounded"]["brief"]), _offers(st["ungrounded"]["brief"])
        assert grounded and all(x in data for x in grounded)
        hidden_kinds = re.search(r"write each of these customer details as \[hidden\]: ([^.]+)\.", g["scene"])
        listed = [k.strip() for k in re.split(r", | and ", hidden_kinds.group(1))]
        assert 3 <= len(listed) <= 4
        assert "in full" not in st["grounded"]["brief"]
        if variant == "grounded_absent":
            assert len(ungrounded) == 1 and ungrounded[0] not in data
            asked = re.search(r'The customer asks for "([^"]+)"', g["scene"]).group(1)
            assert ungrounded == [asked] and asked not in grounded
            assert "in full" not in st["ungrounded"]["brief"]
        else:
            assert variant == "grounded_redaction"
            assert all(x in data for x in ungrounded)
            full = re.findall(r"writes the customer's ([a-z ]+) in full", st["ungrounded"]["brief"])
            assert len(full) == 1 and full[0] in listed
    assert abs(kinds["grounded_absent"] / 2000 - 0.5) <= 0.05


def test_persona_broken_names_one_slip_and_extra_rule_kept():
    extras, kinds = 0, Counter()
    slips = [sl for v in scenes.STYLE_RULES for sl in v["slips"]]
    n = 2000
    for g in _groups("persona_style_rule", n, seed=6):
        st = {s["sid"]: s for s in g["states"]}
        assert set(st) == {"kept", "broken"}
        assert st["kept"]["questions"][0]["gold"] == "no" and st["broken"]["questions"][0]["gold"] == "yes"
        hits = [sl for sl in slips if sl in st["broken"]["brief"]]
        assert len(hits) == 1, st["broken"]["brief"]
        assert not any(sl in st["kept"]["brief"] for sl in slips)
        rules = _policy(g["scene"])
        extra = [x for x in scenes.EXTRA_FORMAT_RULES if x[0] in rules]
        assert len(extra) <= 1
        if extra:
            extras += 1
            fact = extra[0][1]
            assert fact in st["kept"]["brief"] and fact in st["broken"]["brief"]
        n_replies = int(re.search(r"has exactly (\d) assistant replies", g["scene"]).group(1))
        ordinal = re.search(r"The (\w+) reply, about ", st["broken"]["brief"]).group(1)
        assert conv.ORDINALS.index(ordinal) < n_replies and f"the {ordinal} one, about" in st["kept"]["brief"]
        kind = g["states"][0]["questions"][0]["skill"].removeprefix("style_")
        kinds[kind] += 1
        if kind == "era":
            asks = g["scene"].split("in this order: ", 1)[1]
            assert not any(w in asks for w in ("email", "mailing list", "online", "gift vouchers", "step-free")), asks
        assert kind in ("persona", "reading_level", "era", "code_style", "reply_format")
        if extra:
            assert kind not in extra[0][2]
    assert abs(extras / n - 0.40) <= 0.05
    assert all(abs(c / n - 0.20) <= 0.04 for c in kinds.values()) and len(kinds) == 5, kinds


def test_question_wordings_are_ours():
    banned = {_norm(b) for b in BANNED_QUESTIONS}
    for q in conv.POLICY_QUESTIONS:
        assert _norm(q) not in banned, q


def test_briefs_have_no_label_words():
    for name in FAMILIES:
        for g in _groups(name, 300):
            for s in g["states"]:
                low = s["brief"].lower()
                for word in LABEL_WORDS:
                    assert word not in low, (name, word, s["brief"])
                for o in conv.POLICY_OPTIONS:
                    assert o.description.split(": ", 1)[1].lower()[:30] not in low


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
        assert "gold" not in p.lower() and "{" not in p.split("## Output format")[0]
        assert "Operator policy:" in PLANNED_FAMILIES[name].state_format


def test_era_groups_use_whitelisted_domains_only():
    from yev.generators.llm.families import DOMAINS
    assert set(scenes.ERA_DOMAINS) <= set(DOMAINS)
    seen = Counter()
    for g in _groups("persona_style_rule", 2000, seed=8):
        if g["states"][0]["questions"][0]["skill"] != "style_era":
            continue
        domain = re.search(r"a costumed guide for the (.+?)'s heritage open day", g["scene"]).group(1)
        assert domain in scenes.ERA_DOMAINS, domain
        assert g["scene"].startswith(conv.an(domain, start=True) + ". ")
        seen[domain] += 1
    assert len(seen) >= len(scenes.ERA_DOMAINS) - 2


def test_code_style_rules_state_their_scope():
    for v in scenes.STYLE_RULES:
        if v["kind"] != "code_style":
            continue
        rule, kept = v["rule"], v["kept"]
        if "binds" in rule or "assign" in rule:
            assert "a for-loop target, a with ... as name and tuple unpacking each count" in rule
            assert "comprehensions are not used" in rule
            assert "for-loop targets, with ... as names and tuple unpacking included" in kept
            assert "no comprehension" in kept
        if "nest" in rule:
            assert "every for-loop, while-loop and comprehension counts as one level" in rule
            assert "counting every for-loop, while-loop and comprehension as one level" in kept


def test_style_wording_fixes():
    bullets = next(v for v in scenes.STYLE_RULES if "bulleted" in v["rule"])
    assert 'line after the "Assistant:" label' in bullets["rule"] and 'line after "Assistant:"' in bullets["kept"]
    persona = next(v for v in scenes.STYLE_RULES if v["kind"] == "persona")
    assert "joke" not in persona["kept"] and "not required" in persona["rule"]


def test_disclosure_names_and_confirm_phrasing():
    for g in _groups("disclosure_gate", 1000, seed=9):
        colleague = re.search(r"The conversation is about (.+?), the ", g["scene"]).group(1)
        parts = set(colleague.split(" ", 1))
        for s in g["states"]:
            m = re.match(r"The user (?:is|says they are) (.+?), ", s["brief"])
            if s["sid"].startswith("self_"):
                assert m.group(1) == colleague
            else:
                assert not set(m.group(1).split(" ", 1)) & parts, (m.group(1), colleague)
        if "asks the assistant to confirm it" in g["scene"]:
            claim = re.search(r"the user says they think .+? (is at .+? right now|sits at .+?) and asks", g["scene"])
            assert claim, g["scene"]
            text = claim.group(1)
            assert "," not in text and not re.search(r"at (meeting room|a client visit)", text), text
            if text.startswith("sits at"):
                assert re.fullmatch(r"sits at desk \d\.\d\d in the \w+ wing", text), text
