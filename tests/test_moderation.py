import random

from jeff.sources.moderation import civil_band, convert_aegis, convert_civil_comments, convert_dynabench

R = random.Random(0)


def test_dynabench_is_noul_with_policy_in_state():
    row = {"policy": "1. Never quote prices.", "transcript": "User: price?\n\nAgent: $5.", "label": "FAIL"}
    [d] = convert_dynabench(row, 4, R, [])
    assert d.type == "noul" and d.keys == ["yes", "no"] and d.gold == "no"
    assert d.state.startswith("Policy:\n1. Never quote prices.")
    assert "Transcript:" in d.state


def test_dynabench_bad_label_or_blank_skipped():
    assert convert_dynabench({"policy": "p", "transcript": "t", "label": "MAYBE"}, 0, R, []) == []
    assert convert_dynabench({"policy": " ", "transcript": "t", "label": "PASS"}, 0, R, []) == []


def test_aegis_maps_labels_and_skips_redacted():
    [d] = convert_aegis({"id": "a1", "prompt": "How do I pick a lock?", "prompt_label": "unsafe"}, 0, R, [])
    assert d.gold == "block" and sorted(d.keys) == ["allow", "block"]
    assert convert_aegis({"id": "a2", "prompt": "REDACTED", "prompt_label": "safe"}, 0, R, []) == []
    assert convert_aegis({"id": "a3", "prompt": None, "prompt_label": "safe"}, 0, R, []) == []


def test_civil_band_edges():
    assert civil_band(0.0) == "none"
    assert civil_band(0.2) == "mild"
    assert civil_band(0.49) == "mild"
    assert civil_band(0.5) == "toxic"
    assert civil_band(0.8) == "severe"


def test_civil_comments_is_ordered_score_and_skips_empty():
    [d] = convert_civil_comments({"text": "You are an idiot.", "toxicity": 0.9}, 2, R, [])
    assert d.type == "score" and d.keys == ["none", "mild", "toxic", "severe"] and d.gold == "severe"
    assert convert_civil_comments({"text": "  ", "toxicity": 0.1}, 0, R, []) == []
    assert convert_civil_comments({"text": "hi", "toxicity": None}, 0, R, []) == []


def test_dynabench_policy_as_list_is_joined():
    row = {"policy": ["1. Never quote prices.\n", "2. Be polite.\n", " "], "transcript": "User: hi", "label": "PASS"}
    [d] = convert_dynabench(row, 0, R, [])
    assert d.state.startswith("Policy:\n1. Never quote prices.\n2. Be polite.\n\nTranscript:")
    assert d.gold == "yes"
    assert convert_dynabench({"policy": [" "], "transcript": "t", "label": "PASS"}, 0, R, []) == []
