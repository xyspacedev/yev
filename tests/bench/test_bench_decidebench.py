import json
import random
import re
from pathlib import Path

import pytest

from jeff.bench import decidebench as B

OPTS = [{"key": "approve", "description": "Approve it."}, {"key": "hold", "description": "Hold it."},
        {"key": "block", "description": "Block it."}]
Q = "How should this action be handled?"


def item(id_, gold, state, difficulty="easy", category="action_review", options=OPTS, question=Q):
    return {"id": id_, "pair_id": id_[:-1], "category": category, "difficulty": difficulty, "state": state,
            "question": question, "options": options, "gold": gold, "canary": "x"}


def example(gold, state, category="action_review", options=OPTS, question=Q):
    return {"id": f"{category}-t01-{gold}", "category": category, "state": state, "question": question,
            "options": options, "gold": gold}


EXAMPLES = [example("approve", "ex approve"), example("hold", "ex hold"), example("block", "ex block"),
            # another template (different options) must never be mixed in
            example("yes", "other", options=[{"key": "yes", "description": "Y"}, {"key": "no", "description": "N"}])]
TEST = [item("action_review-001a", "approve", "state a"), item("action_review-001b", "block", "state b", "hard")]


def test_shots_one_per_option_in_harness_seeded_order():
    pool = B.example_pool(EXAMPLES)
    shots = B.shots_for(TEST[0], pool)
    assert sorted(s["gold"] for s in shots) == ["approve", "block", "hold"]
    expected = EXAMPLES[:3]
    random.Random("action_review-001a").shuffle(expected)
    assert shots == expected
    assert B.shots_for(TEST[0], pool) == shots  # stable


def test_convert_examples_variant_matches_harness_prompt():
    rows = B.convert(TEST, EXAMPLES)
    ex, zs = rows["examples"][0], rows["zeroshot"][0]
    assert ex["id"] == "decidebench:action_review-001a" and ex["cluster_id"] == "action_review-001"
    assert ex["letters"] == {"A": "approve", "B": "hold", "C": "block"} and ex["answer"] == "A"
    assert ex["bench"] == {"pair_id": "action_review-001", "difficulty": "easy", "n_examples": 3}
    msgs = ex["messages"]
    assert msgs[0]["content"].startswith("Evaluate the supplied decision task.")
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    shots = B.shots_for(TEST[0], B.example_pool(EXAMPLES))
    for k, s in enumerate(shots):
        u, a = msgs[1 + 2 * k], msgs[2 + 2 * k]
        assert json.loads(u["content"])["state"] == s["state"]
        assert a["content"] == "ABC"[[o["key"] for o in OPTS].index(s["gold"])]
    last = msgs[-1]["content"]
    # compact JSON exactly as decidebench.prompts.build_user_message
    assert last == json.dumps({"state": "state a", "question": Q, "options": [
        {"label": "A", "key": "approve", "description": "Approve it."},
        {"label": "B", "key": "hold", "description": "Hold it."},
        {"label": "C", "key": "block", "description": "Block it."}]}, ensure_ascii=False, separators=(",", ":"))
    assert ", " not in last.split('"state"')[0]
    assert [m["role"] for m in zs["messages"]] == ["system", "user"] and zs["messages"][-1]["content"] == last
    assert rows["examples"][1]["answer"] == "C" and rows["examples"][1]["bench"]["difficulty"] == "hard"


def test_convert_refuses_incomplete_example_pool():
    with pytest.raises(ValueError, match="examples cover"):
        B.convert(TEST, EXAMPLES[:2])


def _p(id_, pair, gold, pred, family="action_review", difficulty="easy"):
    keys = ["approve", "hold", "block"]
    return {"id": id_, "letters": dict(zip("ABC", keys)), "probs": [float(k == pred) for k in keys], "pred": pred,
            "gold": gold, "answer": "ABC"[keys.index(gold)], "type": "choice", "family": family,
            "source": "decidebench", "cluster_id": pair, "bench": {"pair_id": pair, "difficulty": difficulty}}


def test_metrics_pairs_families_hard_and_reference():
    preds = [_p("a1", "p1", "approve", "approve"), _p("b1", "p1", "block", "block"),
             _p("a2", "p2", "approve", "approve", "ticket_triage", "hard"),
             _p("b2", "p2", "hold", "block", "ticket_triage", "hard")]
    m = B.metrics(preds, variant="examples")
    assert m["accuracy"] == 0.75 and m["pair_accuracy"] == 0.5 and m["n_pairs"] == 2
    assert m["by_family"]["ticket_triage"] == {"n": 2, "accuracy": 0.5}
    assert m["hard"] == {"n": 2, "accuracy": 0.5, "pair_accuracy": 0.0, "n_pairs": 1}
    assert m["reference"]["examples"]["imajev-4b"] == {"accuracy": 0.95, "pair_accuracy": 0.905}
    assert m["reference"]["zeroshot"]["JEV (AI Space)"]["accuracy"] == 0.9825
    assert m["variant"] == "examples" and "ece_15" in m and "brier" in m
    lo, hi = m["ci95"]
    assert lo <= 0.75 <= hi
    # harness macro: action_review labels approve/block both F1 1; ticket_triage approve 1, hold 0, block 0
    assert m["macro_f1"] == pytest.approx((1.0 + (1.0 + 0 + 0) / 3) / 2)


SRC = Path(__file__).resolve().parents[2] / "src" / "jeff"


def test_only_bench_build_reads_the_test_set():
    """load_test_raw is defined in jeff/decidebench.py and called only by jeff/bench/decidebench.py."""
    callers = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                     if re.search(r"\bload_test_raw\b", p.read_text(encoding="utf-8")))
    assert callers == ["bench/decidebench.py", "decidebench.py"]
    src = (SRC / "bench" / "decidebench.py").read_text(encoding="utf-8")
    assert len(re.findall(r"\bload_test_raw\(", src)) == 1
    assert re.search(r"def build\(out: Path\) -> dict:\n    test_items = db\.load_test_raw\(\)", src)
