import json
import math

import pytest

from yev.bench import score as S
from yev.bench.common import chat_row, item_state, overlap, states_of
from yev.schema import Decision, Option


def P(id_, gold, pred, keys=("yes", "no"), probs=None, **bench):
    letters = {"ABCDEF"[i]: k for i, k in enumerate(keys)}
    if probs is None:
        probs = [1.0 if k == pred else 0.0 for k in keys]
    return {"id": id_, "letters": letters, "probs": probs, "pred": pred, "gold": gold,
            "answer": "ABCDEF"[list(keys).index(gold)], "type": "noul", "family": bench.pop("family", "f"),
            "source": "b", "cluster_id": bench.get("pair_id"), "bench": bench}


def test_prediction_uses_temperature_and_keys():
    row = {"id": "x", "letters": {"A": "a", "B": "b", "C": "c"}, "answer": "C", "type": "choice", "family": "f",
           "source": "s", "cluster_id": None}
    p = S.prediction(row, [1.0, 3.0, 2.0, 99.0, 0, 0], temperature=2.0)
    assert p["pred"] == "b" and p["gold"] == "c" and len(p["probs"]) == 3
    assert math.isclose(sum(p["probs"]), 1.0)
    assert p["probs"][1] / p["probs"][0] == pytest.approx(math.exp(1.0))


def test_accuracy_and_pair_accuracy():
    preds = [P("1", "yes", "yes", pair_id="p1"), P("2", "no", "no", pair_id="p1"),
             P("3", "yes", "yes", pair_id="p2"), P("4", "no", "yes", pair_id="p2"),
             P("5", "yes", "yes", pair_id="p3")]  # p3 has one member: not a full pair
    assert S.accuracy(preds) == pytest.approx(4 / 5)
    acc, n = S.pair_accuracy(preds, lambda p: p["bench"].get("pair_id"))
    assert (acc, n) == (0.5, 2)
    assert S.accuracy([]) is None
    assert S.pair_accuracy([], lambda p: None) == (None, 0)


def test_binary_f1_counts_and_threshold():
    preds = [P("1", "yes", "yes"), P("2", "yes", "no"), P("3", "no", "yes"), P("4", "no", "no"),
             P("5", "yes", "no", probs=[0.5, 0.5])]  # P(yes) == 0.5 counts as positive
    m = S.binary(preds, "yes")
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(0.5)


def test_binary_edge_cases_never_nan():
    none_pos = S.binary([P("1", "no", "no"), P("2", "no", "no")], "yes")
    assert none_pos["f1"] == 0.0 and none_pos["precision"] == 0.0 and none_pos["recall"] == 0.0
    all_missed = S.binary([P("1", "yes", "no")], "yes")
    assert all_missed["f1"] == 0.0 and all_missed["specificity"] == 0.0
    empty = S.binary([], "yes")
    assert empty["n"] == 0 and empty["accuracy"] is None and empty["f1"] == 0.0
    perfect = S.binary([P("1", "yes", "yes"), P("2", "no", "no")], "yes")
    assert perfect["f1"] == 1.0
    # the positive key can be the second option (DynaBench: violation = "no")
    assert S.binary([P("1", "no", "no"), P("2", "yes", "no")], "no")["precision"] == 0.5


def test_common_has_ece_and_brier():
    m = S.common([P("1", "yes", "yes", probs=[0.8, 0.2]), P("2", "no", "yes", probs=[0.6, 0.4])])
    assert m["n"] == 2 and m["accuracy"] == 0.5
    assert m["brier"] == pytest.approx(((0.2 ** 2 + 0.2 ** 2) + (0.6 ** 2 + 0.6 ** 2)) / 2)
    assert m["ece_15"] == pytest.approx((abs(0.8 - 1) + abs(0.6 - 0)) / 2)


def test_pair_bootstrap_ci_brackets_accuracy():
    preds = [P(str(i), "yes", "yes" if i % 3 else "no", pair_id=f"p{i // 2}") for i in range(40)]
    lo, hi = S.pair_bootstrap_ci(preds, lambda p: p["bench"]["pair_id"])
    assert lo <= S.accuracy(preds) <= hi


def _dec(state, keys=("b", "a", "c"), gold="a", type_="choice"):
    return Decision(id="t:1", type=type_, state=state, question="q?",
                    options=[Option(k, f"desc {k}") for k in keys], gold=gold, family="f", source="t",
                    licence="mit", split="dev")


def test_chat_row_keeps_benchmark_order():
    r = chat_row(_dec("s"), {"pair_id": "p"})
    assert r["letters"] == {"A": "b", "B": "a", "C": "c"} and r["answer"] == "B"
    assert r["target"] == {"A": 0.0, "B": 1.0, "C": 0.0}
    user = json.loads(r["messages"][-1]["content"])
    assert [o["key"] for o in user["options"]] == ["b", "a", "c"] and r["split"] == "test"
    assert r["bench"] == {"pair_id": "p"}
    assert set(r) >= {"id", "messages", "answer", "letters", "target", "weight", "type", "family", "source",
                      "cluster_id", "edit_type", "split"}


def test_overlap_counts_items_sharing_an_8gram():
    a = chat_row(_dec("one two three four five six seven eight nine"))
    b = chat_row(_dec("completely different words that never appear in the training text"))
    b["id"] = "t:2"
    train = [chat_row(_dec("prefix ONE two three, four five six seven eight suffix"))]
    rep = overlap([a, b, a], iter(train))
    assert rep["n_items"] == 2 and rep["n_overlapping"] == 1 and rep["share"] == 0.5
    assert rep["overlapping_ids"] == ["t:1"] and rep["n_train_rows"] == 1


def test_overlap_reads_example_states_in_training_rows():
    ex_state = "alpha beta gamma delta epsilon zeta eta theta"
    row = chat_row(_dec("main"))
    row["messages"].insert(1, {"role": "user", "content": json.dumps({"state": ex_state})})
    row["messages"].insert(2, {"role": "assistant", "content": "A"})
    assert states_of(row) == [ex_state, "main"] and item_state(row) == "main"
    rep = overlap([chat_row(_dec(ex_state))], [row])
    assert rep["n_overlapping"] == 1
