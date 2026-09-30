import math
from jeff.train.readout import probs
from jeff.train.metrics import evaluate, group_key

def p(id_, ans, pr, src="s", cl=None, fam="f", typ="choice"):
    return {"id": id_, "type": typ, "family": fam, "edit_type": None, "source": src, "cluster_id": cl,
            "letters": {"A": "x", "B": "y"}, "answer": ans, "probs": pr}

def test_probs_temperature():
    a = probs([2.0, 0.0, 99.0], n=2, temperature=2.0)
    assert math.isclose(a[0], math.exp(1) / (math.exp(1) + 1))

def test_accuracy_brier_selective():
    r = evaluate([p("1", "A", [0.95, 0.05]), p("2", "B", [0.6, 0.4])])
    assert r["accuracy"] == 0.5
    assert math.isclose(r["brier"], ((0.05**2 * 2) + (0.6**2 * 2)) / 2)
    assert r["selective"]["coverage"] == 0.5 and r["selective"]["accuracy"] == 1.0

def test_decidebench_template_groups():
    assert group_key(p("decidebench:returns_policy-t03-refund", "A", [1, 0], src="decidebench")) == "decidebench:returns_policy-t03"
    preds = [p("decidebench:x-t01-a", "A", [0.9, 0.1], src="decidebench"),
             p("decidebench:x-t01-b", "B", [0.9, 0.1], src="decidebench"),
             p("decidebench:x-t02-a", "A", [0.9, 0.1], src="decidebench"),
             p("decidebench:x-t02-b", "B", [0.1, 0.9], src="decidebench")]
    r = evaluate(preds)
    assert r["decidebench"]["group_accuracy"] == 0.5
    assert r["decidebench"]["vs_tev"]["passes"] is False

def test_ece_perfectly_calibrated_is_zero():
    r = evaluate([p(str(i), "A", [1.0, 0.0]) for i in range(5)])
    assert r["ece"] == 0.0
