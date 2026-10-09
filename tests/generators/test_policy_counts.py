import re
from collections import Counter, defaultdict

from yev.generators.common import SYNTH_LICENCE
from yev.generators.llm.targets.conversation import POLICY_OPTIONS, POLICY_QUESTIONS
from yev.generators.policy_counts import (
    ASK_LINES, FAMILY, FOLLOW_UP_LABEL, HANDOFF_LINES, KINDS, SOURCE, generate,
)

APPLIED_RE = re.compile(r"^Assistant: Done, code (\S+) is now applied to order")
OFFER_RE = re.compile(r"^Assistant: .*?I can offer you (\d+)% off")


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def conversation(d) -> list[str]:
    return d.state.split("Conversation:\n", 1)[1].splitlines()


def text_count(d) -> int:
    """The trigger count read back from the rendered text, never from metadata."""
    lines = conversation(d)
    kind = d.metadata["kind"]
    if kind == "human_ask":
        return sum(1 for x in lines if x.startswith("User: ") and x[len("User: "):] in ASK_LINES)
    if kind == "code_cap":
        return sum(1 for x in lines if APPLIED_RE.match(x))
    if kind == "percent_cap":
        return max(int(m.group(1)) for x in lines if (m := OFFER_RE.match(x)))
    return sum(1 for x in lines if x.startswith(f"{FOLLOW_UP_LABEL}: "))


def handed_off(d) -> bool:
    return any(x.startswith("Assistant: ") and x[len("Assistant: "):] in HANDOFF_LINES for x in conversation(d))


def expected(d) -> str:
    """Label from the rendered text and the printed limit only."""
    m = d.metadata
    limit, c = m["limit"], text_count(d)
    if m["kind"] == "human_ask":
        return "yes" if c >= limit and not handed_off(d) else "no"
    return "yes" if c > limit else "no"


def test_labels_follow_rendered_counts():
    ds = generate(700, seed=1)
    assert set(d.metadata["kind"] for d in ds) == set(KINDS)
    for d in ds:
        d.validate()
        assert d.type == "noul" and d.family == FAMILY and d.source == SOURCE == "synthetic_rules"
        assert d.licence == SYNTH_LICENCE
        assert tuple(d.options) == POLICY_OPTIONS
        assert d.question in POLICY_QUESTIONS
        assert text_count(d) == d.metadata["count"]
        assert d.gold == expected(d), (d.id, d.state)
        # The limit is printed in the policy, with the boundary spelled out.
        assert d.metadata["limit_text"] in d.state.split("Conversation:")[0]


def test_count_rule_clean_rows_contain_trigger():
    ds = generate(700, seed=2)
    for d in ds:
        if d.gold == "no":
            assert text_count(d) >= 1, d.id
    # At least 40% of clean rows sit at or past the trigger count (the shortcut fix).
    clean = [d for d in ds if d.gold == "no"]
    at_or_past = [d for d in clean if text_count(d) >= d.metadata["limit"]]
    assert len(at_or_past) / len(clean) >= 0.4


def test_count_rule_boundary_labels():
    ds = generate(800, seed=3)
    seen = Counter()
    for d in ds:
        m = d.metadata
        n, c = m["limit"], text_count(d)
        if m["kind"] == "human_ask":
            if not handed_off(d):
                assert c in (n - 1, n, n + 1)
                assert d.gold == ("no" if c == n - 1 else "yes"), d.id
                seen[("ask", c - n)] += 1
            else:
                # Handed over in reply to the Nth ask: kept the rule.
                assert c == n and d.gold == "no"
                seen[("ask_handoff", 0)] += 1
        else:
            assert c - n in ((-5, -1, 0, 1, 5) if m["kind"] == "percent_cap" else (-1, 0, 1))
            assert d.gold == ("yes" if c > n else "no"), d.id
            seen[(m["kind"], (c > n) - (c < n))] += 1
    for key in [("ask", -1), ("ask", 0), ("ask", 1), ("ask_handoff", 0)]:
        assert seen[key] > 0, key
    for kind in ("code_cap", "percent_cap", "follow_up_cap"):
        for side in (-1, 0, 1):
            assert seen[(kind, side)] > 0, (kind, side)


def test_clusters_share_policy_and_question_with_two_golds():
    ds = generate(300, seed=4)
    for cid, rows in by_cluster(ds).items():
        assert re.fullmatch(rf"{FAMILY}:4:\d+", cid)
        assert len({r.gold for r in rows}) == 2, cid
        assert len({r.question for r in rows}) == 1
        assert len({r.state.split("Conversation:")[0] for r in rows}) == 1
        assert rows[0].edit_type is None and all(r.edit_type for r in rows[1:])


def test_gold_share_and_first_wording():
    ds = generate(2000, seed=5)
    assert len(ds) >= 2000
    yes = sum(d.gold == "yes" for d in ds) / len(ds)
    assert abs(yes - 0.50) <= 0.05, yes
    clusters = by_cluster(ds)
    first = sum(rows[0].question == POLICY_QUESTIONS[0] for rows in clusters.values()) / len(clusters)
    assert abs(first - 0.5) <= 0.06, first


def test_length_range():
    ds = generate(400, seed=6)
    est = [len(d.state + d.question) / 4 for d in ds]
    assert min(est) >= 200 and max(est) <= 1200, (min(est), max(est))


def test_deterministic():
    a, b = generate(15, seed=7), generate(15, seed=7)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert [x.state for x in a] != [x.state for x in generate(15, seed=8)]
    assert all(x.id.startswith(f"{FAMILY}:7:") for x in a)


def test_registered_in_rule_generators():
    from yev.cli import RULE_GENERATORS
    from yev.filters.pipeline import _licence_ok
    from yev.generators.specs import SYNTHETIC_SPECS

    assert RULE_GENERATORS["policy_counts"] is generate
    specs = {s.name: s for s in SYNTHETIC_SPECS}
    assert all(_licence_ok(d, specs, {}) for d in generate(3, seed=0))


def test_percent_cap_push_backs_only_after_an_offer():
    from yev.generators.policy_counts import FIRST_ASKS, PUSH_BACKS

    for d in generate(400, seed=9):
        if d.metadata["kind"] != "percent_cap":
            continue
        offered = False
        for x in conversation(d):
            text = x[len("User: "):] if x.startswith("User: ") else None
            if text in PUSH_BACKS:
                assert offered, d.state
            if text in FIRST_ASKS:
                assert not offered, d.state
            offered = offered or bool(OFFER_RE.match(x))
