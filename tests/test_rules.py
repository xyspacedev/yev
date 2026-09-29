import random

import pytest

from jeff.sources.rules import convert_eikos, convert_ruletaker

R = random.Random(0)


def eikos(**kw):
    row = {
        "id": "e1", "split": "train", "lang": "English", "family": "tool_selection", "question_type": "choice",
        "state": "Ticket #1: CPT billed wrong.", "instructions": "Select the tool.",
        "options": [
            {"label": "claim_correction_api", "description": "Fix CPT codes."},
            {"label": "patient_payment_plan_api", "description": "Set up a plan."},
            {"label": "medical_record_audit", "description": "Audit records."},
        ],
        "expected": "claim_correction_api", "target_probs": [0.9, 0.05, 0.05],
    }
    row.update(kw)
    return row


def test_eikos_choice_with_soft_gold():
    [d] = convert_eikos(eikos(), 0, R, [])
    assert d.type == "choice" and d.gold == "claim_correction_api"
    assert d.soft_gold == pytest.approx({"claim_correction_api": 0.9, "patient_payment_plan_api": 0.05, "medical_record_audit": 0.05})
    assert d.family == "eikos_tool_selection" and d.question == "Select the tool."


def test_eikos_soft_gold_normalised_and_dropped_when_misaligned():
    [d] = convert_eikos(eikos(target_probs=[2, 1, 1]), 0, R, [])
    assert sum(d.soft_gold.values()) == pytest.approx(1.0)
    [d] = convert_eikos(eikos(target_probs=[1.0]), 0, R, [])
    assert d.soft_gold is None


def test_eikos_noul_puts_yes_first():
    row = eikos(question_type="noul", expected="no", target_probs=[0.0, 1.0],
                options=[{"label": "no", "description": "Locked."}, {"label": "yes", "description": "Unlocked."}])
    [d] = convert_eikos(row, 0, R, [])
    assert d.keys == ["yes", "no"] and d.gold == "no"
    assert d.soft_gold == pytest.approx({"yes": 1.0, "no": 0.0})


def test_eikos_skips_portuguese_non_train_colliding_and_missing_expected():
    assert convert_eikos(eikos(lang="Brazilian Portuguese"), 0, R, []) == []
    assert convert_eikos(eikos(split="validation"), 0, R, []) == []
    colliding = [{"label": "Tool A", "description": "x"}, {"label": "tool_a", "description": "y"},
                 {"label": "b", "description": "z"}]
    assert convert_eikos(eikos(options=colliding, expected="b", target_probs=None), 0, R, []) == []
    assert convert_eikos(eikos(expected="not_an_option"), 0, R, []) == []


def test_eikos_blank_description_falls_back_to_label():
    opts = [{"label": "alpha_tool", "description": None}, {"label": "beta", "description": "B"}]
    [d] = convert_eikos(eikos(options=opts, expected="beta", target_probs=None), 0, R, [])
    assert d.options[0].description == "Alpha tool"


def test_ruletaker_noul():
    row = {"context": "Bob is big. If someone is big then they are kind.", "question": "Bob is kind.", "label": "entailment"}
    [d] = convert_ruletaker(row, 9, R, [])
    assert d.type == "noul" and d.gold == "yes" and "Bob is kind." in d.question
    assert convert_ruletaker({**row, "label": "not entailment"}, 9, R, [])[0].gold == "no"
    assert convert_ruletaker({**row, "label": "unknown"}, 9, R, []) == []
