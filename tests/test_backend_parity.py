"""Dev-row parity of serving backends (Plan 7 Task 5)."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("backend_parity", Path("scripts/backend_parity.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def rec(id_, probs, answer="A"):
    return {"id": id_, "type": "choice", "answer": answer, "probs": probs}


def test_compare_identical_is_at_parity():
    a = [rec("x", [0.9, 0.1]), rec("y", [0.2, 0.8], "B")]
    out = bp.compare_records(a, a)
    assert out["argmax_agreement"] == 1.0 and out["mean_abs_dp"] == 0.0 and out["at_parity"]
    assert out["accuracy_a"] == out["accuracy_b"] == 1.0


def test_compare_flags_disagreement_and_dp():
    a = [rec("x", [0.9, 0.1]), rec("y", [0.6, 0.4])]
    b = [rec("x", [0.88, 0.12]), rec("y", [0.4, 0.6])]
    out = bp.compare_records(a, b)
    assert out["argmax_agreement"] == 0.5 and out["disagreements"] == ["y"]
    assert out["max_abs_dp"] == pytest.approx(0.2)
    assert out["mean_abs_dp"] == pytest.approx((0.02 + 0.2) / 2)
    assert not out["at_parity"]


def test_compare_requires_same_ids():
    with pytest.raises(ValueError, match="ids"):
        bp.compare_records([rec("x", [1.0])], [rec("z", [1.0])])


def test_by_state_order_groups_shared_states():
    def row(i, state):
        return {"id": str(i), "messages": [{"role": "user", "content": json.dumps({"state": state})}]}
    rows = [row(0, "s1"), row(1, "s2"), row(2, "s1")]
    assert [r["id"] for r in bp.by_state(rows)] == ["0", "2", "1"]
