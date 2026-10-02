"""`yev eval --shuffle-options K`: the position-bias check (Plan 5 Task 10)."""
import json
import random

from yev import cli
from yev.format import LETTERS, user_turn
from yev.schema import Decision, Option
from yev.train import infer
from yev.train.position_bias import permute_row, report

KEYS = ["approve", "block", "escalate", "hold", "refund", "replace", "close", "reopen"]


def _row(id_, n=8, gold="escalate", type_="choice", examples=0):
    d = Decision(id=id_, type=type_, family="f", source="s", licence="mit", state="the state, with “quotes” and ünïcode",
                 question="what now?", options=[Option(k, f"desc of {k}") for k in KEYS[:n]], gold=gold)
    opts = list(d.options)
    user = user_turn(d, opts)
    msgs = [{"role": "system", "content": "sys"}]
    for _ in range(examples):
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": "A"}]
    msgs.append({"role": "user", "content": user})
    letters = {LETTERS[i]: o.key for i, o in enumerate(opts)}
    return {"id": id_, "messages": msgs, "letters": letters, "answer": LETTERS[[o.key for o in opts].index(gold)],
            "target": {L: float(k == gold) for L, k in letters.items()}, "weight": 1.0, "type": type_,
            "family": "f", "source": "s", "cluster_id": None, "edit_type": None, "split": "dev"}


def test_shuffle_keeps_labels_with_keys():
    row = _row("r1", examples=1)
    order = list(range(8))
    random.Random(3).shuffle(order)
    p = permute_row(row, order)
    user = json.loads(p["messages"][-1]["content"])
    orig = {o["key"]: o["description"] for o in json.loads(row["messages"][-1]["content"])["options"]}
    assert [o["label"] for o in user["options"]] == list(LETTERS[:8])
    assert [o["key"] for o in user["options"]] == [KEYS[i] for i in order]
    assert all(orig[o["key"]] == o["description"] for o in user["options"])
    assert p["letters"] == {o["label"]: o["key"] for o in user["options"]}
    assert p["letters"][p["answer"]] == "escalate"
    assert {p["letters"][L]: t for L, t in p["target"].items()} == {k: float(k == "escalate") for k in KEYS}
    # byte-identical to the training renderer, examples untouched
    d = Decision(id="x", type="choice", family="f", source="s", licence="mit", state=user["state"], question=user["question"],
                 options=[Option(KEYS[i], f"desc of {KEYS[i]}") for i in order], gold="escalate")
    assert p["messages"][-1]["content"] == user_turn(d, d.options)
    assert p["messages"][:-1] == row["messages"][:-1]
    assert permute_row(row, list(range(8)))["messages"] == row["messages"]
    assert row["letters"] == {LETTERS[i]: k for i, k in enumerate(KEYS)}  # input not mutated


def _write(tmp_path, rows):
    data = tmp_path / "dev.chat.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return data


def _fake_logits(score_of):
    def letter_logits(model, tok, rows, max_len, batch_tokens=16384, n_letters=6):
        return [[score_of(r, L) for L in LETTERS[:n_letters]] for r in rows]
    return letter_logits


def _rows():
    return ([_row(f"r{i}", n=8) for i in range(4)] + [_row(f"s{i}", n=7) for i in range(2)]
            + [_row("short", n=3), _row("scale", n=8, type_="score")])


def test_agreement_one_for_order_invariant_model(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "load", lambda m, b: (None, None))
    monkeypatch.setattr(cli, "_overlong_ids", lambda tok, rows, max_len: [])
    # logit depends on the key behind the letter only: an order-invariant model
    monkeypatch.setattr(infer, "letter_logits", _fake_logits(
        lambda r, L: float(KEYS.index(r["letters"][L])) if L in r["letters"] else -1e4))
    out = tmp_path / "position_bias.json"
    assert cli.main(["eval", "--model", "m", "--data", str(_write(tmp_path, _rows())), "--out", str(out),
                     "--n-max", "8", "--shuffle-options", "3", "--min-options", "7"]) == 0
    rep = json.loads(out.read_text())
    assert rep["n_rows"] == 6 and rep["n_orders"] == 4
    assert rep["argmax_agreement"] == 1.0 and rep["mean_tv_distance"] < 1e-9
    assert set(rep["by_n_options"]) == {"7", "8"} and rep["by_n_options"]["8"]["n_rows"] == 4
    assert rep["n_skipped_score"] == 1 and rep["n_skipped_few_options"] == 1


def test_position_biased_model_disagrees(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "load", lambda m, b: (None, None))
    monkeypatch.setattr(cli, "_overlong_ids", lambda tok, rows, max_len: [])
    # always prefers letter A, whatever sits behind it
    monkeypatch.setattr(infer, "letter_logits", _fake_logits(lambda r, L: 5.0 if L == "A" else 0.0))
    out = tmp_path / "pb.json"
    assert cli.main(["eval", "--model", "m", "--data", str(_write(tmp_path, _rows())), "--out", str(out),
                     "--n-max", "8", "--shuffle-options", "3"]) == 0
    rep = json.loads(out.read_text())
    assert rep["argmax_agreement"] < 0.5 and rep["mean_tv_distance"] > 0.3


def test_report_metrics_by_hand():
    # two rows, two orders each; per-key distributions
    groups = [
        {"n_options": 7, "gold": "a", "orders": [{"a": 0.6, "b": 0.4}, {"a": 0.6, "b": 0.4}]},
        {"n_options": 8, "gold": "a", "orders": [{"a": 0.7, "b": 0.3}, {"a": 0.2, "b": 0.8}]},
    ]
    rep = report(groups)
    assert rep["argmax_agreement"] == 0.5
    assert abs(rep["mean_tv_distance"] - 0.25) < 1e-12  # (0 + 0.5) / 2
    assert rep["by_n_options"]["8"] == {"n_rows": 1, "argmax_agreement": 0.0, "mean_tv_distance": 0.5}


def test_shuffle_eval_runs_on_tiny_model(tmp_path, monkeypatch, tok, tiny_model):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"temperatures": {"choice": 2.0}}))
    out = tmp_path / "pb.json"
    assert cli.main(["eval", "--model", "m", "--data", str(_write(tmp_path, _rows())), "--out", str(out),
                     "--n-max", "8", "--shuffle-options", "2", "--calibration", str(cal)]) == 0
    rep = json.loads(out.read_text())
    assert rep["n_rows"] == 6 and rep["n_orders"] == 3 and 0.0 <= rep["argmax_agreement"] <= 1.0
    assert len(rep["accuracy_by_order"]) == 3 and rep["run"]["shuffle_options"] == 2
