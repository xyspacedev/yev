import random

from jeff.sources.fast_decisions import FAST_DECISIONS_CONFIGS, convert_fast_decisions
from jeff.sources.registry import SOURCES

R = random.Random(0)
INTENTS = [f"intent_{i}" for i in range(28)]


def row(*heads, text="[channel] email\n[utterance] I was charged twice, please refund."):
    return {"input": text, "output": {"classifications": list(heads)}}


def head(task, gold, labels, multi=False):
    return {"task": task, "true_label": gold, "labels": labels, "multi_label": multi}


def test_single_label_head_becomes_choice_with_3_to_6_options():
    for _ in range(20):
        [d] = convert_fast_decisions(row(head("intent", ["intent_7"], INTENTS)), 3, R, [])
        assert d.type == "choice" and d.gold == "intent_7" and 3 <= len(d.keys) <= 6
        assert d.id == "3:intent" and d.family == "fast_decisions"
        assert d.state.startswith("[channel] email")


def test_yes_no_head_becomes_noul():
    [d] = convert_fast_decisions(row(head("is_phishing", ["yes"], ["yes", "no"])), 0, R, [])
    assert d.type == "noul" and d.keys == ["yes", "no"] and d.gold == "yes"
    [d] = convert_fast_decisions(row(head("urgent", ["no"], ["no", "yes"])), 0, R, [])
    assert d.keys == ["yes", "no"] and d.gold == "no"


def test_ordered_heads_become_score_in_order():
    [d] = convert_fast_decisions(row(head("sentiment", ["neutral"], ["negative", "neutral", "positive"])), 0, R, [])
    assert d.type == "score" and d.keys == ["negative", "neutral", "positive"] and d.gold == "neutral"
    [d] = convert_fast_decisions(row(head("urgency", ["critical"], ["low", "normal", "high", "critical"])), 0, R, [])
    assert d.type == "score" and d.keys == ["low", "normal", "high", "critical"]


def test_multi_label_head_becomes_one_noul_per_label():
    labels = ["food", "service", "price"]
    out = convert_fast_decisions(row(head("aspects", ["food", "price"], labels, multi=True)), 5, R, [])
    assert [d.type for d in out] == ["noul"] * 3
    assert {d.id: d.gold for d in out} == {"5:aspects:food": "yes", "5:aspects:service": "no", "5:aspects:price": "yes"}
    assert all("food" in d.question or "service" in d.question or "price" in d.question for d in out)


def test_several_heads_per_row_and_bad_heads_skipped():
    out = convert_fast_decisions(
        row(
            head("sentiment", ["negative"], ["negative", "neutral", "positive"]),
            head("intent", ["not_a_label"], INTENTS),
            head("intent2", ["intent_1", "intent_2"], INTENTS),
        ),
        0, R, [],
    )
    assert [d.id for d in out] == ["0:sentiment"]


def test_blank_input_skipped():
    assert convert_fast_decisions(row(head("urgent", ["no"], ["yes", "no"]), text="  "), 0, R, []) == []


def test_registry_has_one_spec_per_config():
    specs = [s for s in SOURCES if s.hf_id == "fastino/fast-decisions"]
    assert sorted(s.config for s in specs) == sorted(FAST_DECISIONS_CONFIGS)
    assert len(FAST_DECISIONS_CONFIGS) == 17
    assert {s.licence for s in specs} == {"apache-2.0"} and {s.split for s in specs} == {"train"}
