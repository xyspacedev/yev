import re
from collections import Counter, defaultdict

from yev.generators.agent_triples import FAMILY, MEMBERS, SCENARIOS, SOURCE, generate
from yev.generators.common import SYNTH_LICENCE
from yev.generators.llm.targets.agent import VERDICT_OPTIONS, VERDICT_QUESTIONS
from yev.generators.llm.targets.agent_scenes import LOG_LABELS

GOLDS = {"reckless": "yes", "confirmed_then_wide": "yes", "scoped": "no", "confirmed": "no", "asks_first": "no"}


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def calls(d) -> list[str]:
    return [x[len("Agent (tool call): "):] for x in d.state.splitlines() if x.startswith("Agent (tool call): ")]


def test_rows_valid_and_log_layout():
    ds = generate(200, seed=1)
    for d in ds:
        d.validate()
        assert d.type == "noul" and d.family == FAMILY == "agent_overreach" and d.source == SOURCE
        assert d.licence == SYNTH_LICENCE
        assert tuple(d.options) == VERDICT_OPTIONS and d.question in VERDICT_QUESTIONS
        lines = d.state.splitlines()
        assert lines[0].startswith("Operator brief: ") and lines[1].startswith("User: ")
        labelled = [x for x in lines if re.match(r"^[A-Z][a-z]+( [a-z]+)?( \([a-z ]+\))?: ", x)]
        assert all(x.split(": ", 1)[0] in LOG_LABELS for x in labelled), d.state
        assert len(calls(d)) <= 1  # a single tool call at most


def test_agent_triples_cluster_golds():
    ds = generate(300, seed=2)
    for cid, rows in by_cluster(ds).items():
        assert re.fullmatch(rf"{FAMILY}:2:\d+", cid)
        kinds = [r.metadata["member"] for r in rows]
        assert kinds[0] == "reckless" and set(kinds) <= set(MEMBERS)
        assert "asks_first" in kinds and ("scoped" in kinds or "confirmed" in kinds)
        for r in rows:
            assert r.gold == GOLDS[r.metadata["member"]], r.id
        by = {r.metadata["member"]: r for r in rows}
        wide = calls(by["reckless"])
        assert len(wide) == 1
        # Same operator brief and user request in every member; only the agent's conduct differs.
        assert len({tuple(r.state.splitlines()[:2]) for r in rows}) == 1
        assert calls(by["asks_first"]) == []
        assert by["asks_first"].state.splitlines()[-1].startswith("Agent (reply): ")
        assert by["asks_first"].state.rstrip().endswith("?")
        safe = by.get("scoped") or by["confirmed"]
        assert calls(safe) and calls(safe) != wide
        if "confirmed_then_wide" in by:
            # The user confirmed the narrow scope, then the agent ran the wide call anyway.
            assert calls(by["confirmed_then_wide"]) == wide
            assert "confirmed" not in by or by["confirmed"].state.count("User: ") == 2
            assert by["confirmed_then_wide"].state.count("User: ") == 2


def test_gold_share_and_scenarios():
    ds = generate(600, seed=3)
    assert len(ds) >= 2000
    yes = sum(d.gold == "yes" for d in ds) / len(ds)
    assert abs(yes - 0.50) <= 0.05, yes
    used = Counter(d.metadata["scenario"] for d in ds)
    assert set(used) == set(SCENARIOS)
    clusters = by_cluster(ds)
    first = sum(rows[0].question == VERDICT_QUESTIONS[0] for rows in clusters.values()) / len(clusters)
    assert abs(first - 0.5) <= 0.06, first


def test_length_range():
    est = [len(d.state + d.question) / 4 for d in generate(300, seed=4)]
    assert min(est) >= 200 and max(est) <= 800, (min(est), max(est))


def test_deterministic():
    a, b = generate(12, seed=3), generate(12, seed=3)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert [x.state for x in a] != [x.state for x in generate(12, seed=4)]
    assert all(x.id.startswith(f"{FAMILY}:3:") for x in a)


def test_registered_in_rule_generators():
    from yev.cli import RULE_GENERATORS
    from yev.filters.pipeline import _licence_ok
    from yev.generators.specs import SYNTHETIC_SPECS

    assert RULE_GENERATORS["agent_triples"] is generate
    specs = {s.name: s for s in SYNTHETIC_SPECS}
    assert all(_licence_ok(d, specs, {}) for d in generate(3, seed=0))


def test_social_post_wide_call_has_concrete_harm():
    rows = [d for d in generate(400, seed=5) if d.metadata["scenario"] == "social_post"]
    assert rows
    for d in rows:
        assert "aren't agreed" in d.state.splitlines()[1]  # the request says the hours are not final
        if d.gold == "yes":
            assert "publish_post(" in d.state and "followers" in d.state
