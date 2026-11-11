import json
import re
from dataclasses import replace

import pytest

from yev.generators import prompt_template as pt
from yev.schema import Decision, Option


def count(t):
    return max(1, len(t) // 4)


def row(i="r1", state="Order 77 arrived broken.\n\nCustomer wants a refund.", family="returns_v2", cluster="c1",
        source="synthetic_rules"):
    return Decision(id=i, type="choice", state=state, question="What now?",
                    options=[Option("a", "refund"), Option("b", "deny")], gold="a", family=family, source=source,
                    licence="cc-by-4.0", split="train", soft_gold={"a": 0.8, "b": 0.2}, cluster_id=cluster, weight=0.5)


def templates(n=400, family="moderation", budget=2000, states=("x",)):
    return [pt.make_template(f"u{i}", family, 0, budget, count, states) for i in range(n)]


def unwrap(t, wrapped):
    if t.layout == "json":
        start = wrapped.index("{\n")
        return json.JSONDecoder().raw_decode(wrapped[start:])[0][t.noun]
    return None


def test_state_is_inserted_verbatim_or_as_json():
    state = 'Line one.\n\n  indented "quoted" line\twith tab\nLast line.'
    seen = set()
    for t in templates(states=(state,)):
        wrapped = t.wrap(state)
        seen.add(t.layout)
        if t.layout == "json":
            assert unwrap(t, wrapped) == state
        else:
            assert state in wrapped
    assert seen == set(pt.LAYOUTS)


def test_template_is_a_function_of_seed_and_key():
    a = pt.make_template("k", "triage", 0, 800, count, ("x",))
    assert a == pt.make_template("k", "triage", 0, 800, count, ("x",))
    assert a != pt.make_template("k2", "triage", 0, 800, count, ("x",))
    assert a != pt.make_template("k", "triage", 1, 800, count, ("x",))


def test_smaller_budget_only_shortens_the_template():
    for i in range(50):
        big = pt.make_template(f"k{i}", "agent_overreach", 0, 1000, count, ("x",))
        small = pt.make_template(f"k{i}", "agent_overreach", 0, 150, count, ("x",))
        assert (small.layout, small.style, small.noun, small.data_first) == (big.layout, big.style, big.noun, big.data_first)
        assert small.tokens <= 150 and small.tokens <= big.tokens
        assert small.instructions.split("\n\n")[0] == big.instructions.split("\n\n")[0]


def test_no_template_below_the_minimum():
    assert pt.make_template("k", "triage", 0, pt.MIN_TOKENS - 1, count, ("x",)) is None
    for i in range(100):
        t = pt.make_template(f"k{i}", "triage", 0, pt.MIN_TOKENS, count, ("x",))
        assert count(t.wrap("x")) <= pt.MIN_TOKENS + 20


def test_size_follows_the_requested_range():
    sizes = [count(t.wrap("x")) for t in templates(n=600)]
    assert min(sizes) < 200 and max(sizes) > 800 and max(sizes) <= 1000
    for t in templates(n=200):
        assert abs(count(t.wrap("x")) - t.tokens) <= 0.1 * t.tokens + 25


def test_delimiters_never_occur_in_the_state():
    state = 'a ``` b """ c </item> </record> </submission> </input> [END ITEM] ===== END OF RECORD ====='
    for t in templates(states=(state,)):
        assert not any(m in state for m in pt._delimiters(t.layout, t.noun))
        assert state in t.wrap(state) or t.layout == "json"


def test_open_layouts_end_with_the_record():
    state = "THE RECORD"
    seen = set()
    for t in templates():
        if t.layout in pt.OPEN_LAYOUTS:
            assert t.wrap(state).endswith(state)
            seen.add(t.layout)
    assert seen == set(pt.OPEN_LAYOUTS)


def test_per_record_ids_differ_between_records_of_one_template():
    t = next(t for t in templates() if any(f == "request_id" for f, _, _ in t.fields) and t.layout == "kv")
    a, b = t.wrap("first"), t.wrap("second")
    assert a.replace("first", "") != b.replace("second", "")
    assert t.instructions in a and t.instructions in b


def test_wrapper_text_states_no_numbers_and_fills_every_slot():
    bank = [*pt.CONTEXT, *pt.GUIDELINES, *pt.GROUNDED, *pt.CLOSINGS, *pt.FIELD_NOTES.values()]
    bank += [r for roles, _, _ in pt.DOMAINS.values() for r in roles]
    assert len(set(bank)) == len(bank)
    for text in bank:
        assert not re.search(r"\d", text), text
        assert set(re.findall(r"\{(\w+)\}", text)) <= {"org", "team", "noun", "pipeline", "version"}, text
    for t in templates(n=300):
        assert "{" not in t.instructions + t.closing


def test_metadata_sentences_only_when_metadata_is_printed():
    for t in templates(n=300):
        if not t.fields:
            assert "etadata" not in t.instructions and "request ID" not in t.instructions


def test_domains():
    assert pt.domain_of("eikos_contract") == "policy"
    assert pt.domain_of("never_seen_family") == "generic"
    assert set(pt.DOMAIN_OF.values()) <= set(pt.DOMAINS)


SPEC = {"rate": 0.5, "tokens": [120, 600], "max_tokens": 4000, "exclude_families": ["many_options"],
        "exclude_sources": ["decidebench"]}


def key(d):
    return d.cluster_id or d.id


def test_apply_keeps_labels_and_templates_a_share_of_units():
    rows = [row(f"r{i}", state=f"State {i} here.", cluster=f"c{i // 2}") for i in range(400)]
    out, picks, report = pt.apply(rows, {}, SPEC, 0, key, count)
    assert [o.id for o in out] == [r.id for r in rows] and picks == {}
    changed = [(r, o) for r, o in zip(rows, out) if o.state != r.state]
    assert report["templated"] == len(changed) and report["not_selected"] == 400 - len(changed)
    assert 150 < len(changed) < 250
    for r, o in changed:
        assert replace(o, state=r.state, metadata=None) == r
        assert o.metadata["prompt_template"]["layout"] in pt.LAYOUTS
    by_cluster = {}
    for r, o in zip(rows, out):
        by_cluster.setdefault(r.cluster_id, set()).add(o.state != r.state)
    assert all(len(v) == 1 for v in by_cluster.values())
    assert sum(report["by_layout"].values()) == report["templated"]


def test_cluster_siblings_share_one_wrapper():
    a = row("a", state="Returned on day 30.")
    b = row("b", state="Returned on day 31.")
    (oa, ob), _, _ = pt.apply([a, b], {}, {**SPEC, "rate": 1.0}, 0, key, count)
    assert oa.metadata["prompt_template"] == ob.metadata["prompt_template"]
    strip = lambda s: re.sub(r"[0-9a-f]{8,16}", "", s)
    assert strip(oa.state).replace("day 30", "day 31") == strip(ob.state)


def test_apply_skips_excluded_families_and_sources():
    rows = [row("m", family="many_options"), row("d", source="decidebench"), row("ok")]
    out, _, report = pt.apply(rows, {}, {**SPEC, "rate": 1.0}, 0, lambda d: d.id, count)
    assert out[0] == rows[0] and out[1] == rows[1] and out[2].state != rows[2].state
    assert report["excluded"] == 2 and report["templated"] == 1


def test_examples_get_the_rows_template_and_originals_are_untouched():
    target = row("t", state="Target case.")
    ex = [row("e1", state="Example one.", cluster="x1"), row("e2", state="Example two.", cluster="x2")]
    out, picks, _ = pt.apply([target, *ex], {"t": ex}, {**SPEC, "rate": 1.0}, 0, key, count)
    wrapped = picks["t"]
    assert [e.id for e in wrapped] == ["e1", "e2"]
    for o, e in zip(wrapped, ex):
        assert e.state in o.state or json.dumps(e.state)[1:-1] in o.state
        assert o.state.replace(e.state, "") != e.state
    shell = lambda d, s: re.sub(r"[0-9a-f]{8,16}", "", d.state.replace(s, "<S>"))
    assert shell(out[0], "Target case.") == shell(wrapped[0], "Example one.") == shell(wrapped[1], "Example two.")
    assert ex[0].state == "Example one."


def test_prompt_stays_within_max_tokens_and_overlong_rows_stay_bare():
    big = row("big", state="word " * 4000)  # ~5000 tokens: over max_tokens before any wrapper
    mid = row("mid", state="word " * 2900, cluster="c2")  # ~3650 tokens: room for a short wrapper only
    (ob, om), _, report = pt.apply([big, mid], {}, {**SPEC, "rate": 1.0, "tokens": [500, 600]}, 0, key, count)
    assert ob == big and report["no_room"] == 1
    assert om.state != mid.state and om.metadata["prompt_template"]["tokens"] < 350
    assert count(pt.user_turn(om, om.options)) + 100 <= SPEC["max_tokens"]


def test_examples_share_the_token_room():
    ex = [row(f"e{i}", state="word " * 500, cluster=f"x{i}") for i in range(4)]
    target = row("t", state="word " * 500)
    out, picks, _ = pt.apply([target], {"t": ex}, {**SPEC, "rate": 1.0, "tokens": [500, 600]}, 0, key, count)
    total = sum(count(pt.user_turn(d, d.options)) for d in [*picks["t"], out[0]])
    assert out[0].state != target.state and total + 150 <= SPEC["max_tokens"]


@pytest.mark.parametrize("spec", [{"rate": 0}, {"rate": 1.5}, {"tokens": [10, 50]}, {"tokens": [500, 200]}, {"ratio": 1}])
def test_bad_spec(spec):
    with pytest.raises(pt.TemplateError):
        pt.parse_spec(spec)
