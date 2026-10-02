import random

from yev.sources.intent import NONE_OPTION, convert_go_emotions, intent_converter
from yev.sources.replay import convert_boolq, convert_multiple_choice
from yev.sources.triage import convert_cvss, convert_help_desk, convert_it_support

LABELS = [f"intent_{i}" for i in range(30)] + ["oos"]


def test_intent_in_scope_has_gold_and_3_to_6_options():
    conv = intent_converter("text", "intent", oos_label="oos")
    rng = random.Random(1)
    for i in range(30):
        [d] = conv({"text": "freeze my card", "intent": "intent_4"}, i, rng, LABELS)
        assert d.gold == "intent_4" and 3 <= len(d.keys) <= 6
        assert "oos" not in d.keys


def test_intent_out_of_scope_maps_to_none_option():
    conv = intent_converter("text", "intent", oos_label="oos")
    [d] = conv({"text": "tell me a joke", "intent": "oos"}, 0, random.Random(0), LABELS)
    assert d.gold == NONE_OPTION.key and NONE_OPTION in d.options


def test_intent_blank_text_or_missing_label_skipped():
    conv = intent_converter("text", "label")
    assert conv({"text": " ", "label": "intent_1"}, 0, random.Random(0), LABELS) == []
    assert conv({"text": "hi", "label": None}, 0, random.Random(0), LABELS) == []


def test_go_emotions_single_label_only():
    labels = ["anger", "joy", "neutral", "fear", "love"]
    [d] = convert_go_emotions({"id": "g", "text": "I love this!", "labels": ["love"]}, 0, random.Random(0), labels)
    assert d.gold == "love" and d.family == "sentiment"
    assert convert_go_emotions({"id": "g", "text": "x", "labels": ["joy", "love"]}, 0, random.Random(0), labels) == []


def test_help_desk_priority_is_ordered_score():
    [d] = convert_help_desk({"issue_id": 1, "text": "prod down", "issue_priority": "Blocker"}, 0, random.Random(0), [])
    assert d.type == "score" and d.keys == ["low", "medium", "high", "highest", "blocker"] and d.gold == "blocker"
    assert convert_help_desk({"issue_id": 1, "text": "x", "issue_priority": "Urgent"}, 0, random.Random(0), []) == []


def test_cvss_severity_score():
    row = {"id": "CVE-1", "title": "", "description": "SQL injection via sort parameter.", "severity_band": "high"}
    [d] = convert_cvss(row, 0, random.Random(0), [])
    assert d.keys == ["low", "medium", "high", "critical"] and d.gold == "high"
    assert d.state == "Vulnerability report: SQL injection via sort parameter."


def test_it_support_routes_to_team():
    labels = ["Active Directory", "Computer-Services", "EOL", "Fileservice", "O365", "Software", "Support general"]
    [d] = convert_it_support({"text": "File share access", "label": "Fileservice"}, 0, random.Random(0), labels)
    assert d.gold == "fileservice" and d.family == "triage" and 3 <= len(d.keys) <= 6


def test_boolq_noul():
    [d] = convert_boolq({"question": "is the sky blue", "passage": "The sky is blue.", "answer": True}, 0, random.Random(0), [])
    assert d.type == "noul" and d.gold == "yes" and d.question == "Is the sky blue?"


def test_multiple_choice_arc_shape():
    row = {"id": "q1", "question": "Which is a mammal?", "answerKey": "B",
           "choices": {"label": ["A", "B", "C"], "text": ["Shark", "Whale", "Trout"]}}
    [d] = convert_multiple_choice(row, 0, random.Random(0), [])
    assert d.keys == ["a", "b", "c"] and d.gold == "b" and d.options[1].description == "Whale"
    assert convert_multiple_choice({**row, "answerKey": ""}, 0, random.Random(0), []) == []
