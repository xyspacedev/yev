"""`jeff bench score` and `jeff bench overlap` end to end on a tiny model (no network)."""
import json

from jeff import cli
from jeff.bench import decidebench as B
from jeff.bench import rjudge
from jeff.bench.common import write_json, write_rows
from jeff.train import infer

OPTS = [{"key": "approve", "description": "Approve it."}, {"key": "block", "description": "Block it."}]


def _items():
    test = [{"id": f"action_review-00{i}{h}", "pair_id": f"action_review-00{i}", "category": "action_review",
             "difficulty": "hard" if i == 1 else "easy", "state": f"state {i}{h} alpha {i}{h} beta {i}{h} gamma {i}{h} delta {i}{h}", "question": "q?",
             "options": OPTS, "gold": "approve" if h == "a" else "block"} for i in (1, 2) for h in "ab"]
    ex = [{"id": f"t-{k}", "category": "action_review", "state": f"example {k}", "question": "q?", "options": OPTS,
           "gold": k} for k in ("approve", "block")]
    return test, ex


def test_bench_score_writes_predictions_and_metrics(tmp_path, monkeypatch, tok, tiny_model):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    test, ex = _items()
    d = tmp_path / "decidebench"
    rows = B.convert(test, ex)
    write_rows(d / "rows_examples.chat.jsonl", rows["examples"])
    write_json(d / "meta.json", {"revision": "abc", "overlap": {"share": 0.0, "n_items": 4, "overlapping_ids": []}})
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"temperatures": {"choice": 2.0}}))
    out = tmp_path / "out"
    assert cli.main(["bench", "score", "--name", "decidebench", "--model", "m", "--base", "b",
                     "--rows", str(d / "rows_examples.chat.jsonl"), "--calibration", str(cal),
                     "--out", str(out), "--max-len", "8"]) == 0
    preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 4 and set(preds[0]) == {"id", "letters", "probs", "pred", "gold"}
    assert preds[0]["gold"] == "approve" and abs(sum(preds[0]["probs"]) - 1) < 1e-6
    m = json.loads((out / "metrics.json").read_text())
    assert m["n"] == 4 and m["n_pairs"] == 2 and m["variant"] == "examples"
    assert {"ece_15", "brier", "pair_accuracy", "hard", "reference"} <= set(m)
    assert m["run"]["temperatures"] == {"choice": 2.0} and m["run"]["n_truncated"] == 4
    assert m["run"]["bench_revision"] == "abc" and m["overlap"] == {"share": 0.0, "n_items": 4}


def test_bench_score_binary_benchmark(tmp_path, monkeypatch, tok, tiny_model):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    rec = {"id": 1, "profile": "p", "label": 1, "attack_type": "unintended",
           "contents": [[{"role": "user", "content": "do it"}, {"role": "agent", "thought": "t", "action": "a"}]]}
    rows = rjudge.convert({"data/IoT/home.json": [rec, {**rec, "id": 2, "label": 0}]})
    write_rows(tmp_path / "rows.chat.jsonl", rows)
    out = tmp_path / "o"
    assert cli.main(["bench", "score", "--name", "rjudge", "--model", "m", "--rows",
                     str(tmp_path / "rows.chat.jsonl"), "--out", str(out)]) == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["n"] == 2 and "f1" in m and m["overlap"] is None and m["run"]["variant"] is None


def test_bench_overlap_updates_meta(tmp_path):
    test, ex = _items()
    d = tmp_path / "decidebench"
    rows = B.convert(test, ex)
    write_rows(d / "rows_examples.chat.jsonl", rows["examples"])
    write_rows(d / "rows_zeroshot.chat.jsonl", rows["zeroshot"])
    write_json(d / "meta.json", {"n": 4})
    train = dict(rows["zeroshot"][0], id="train-1")  # shares item 001a's state
    write_rows(tmp_path / "train.chat.jsonl", [train])
    assert cli.main(["bench", "overlap", "--name", "decidebench", "--dir", str(d),
                     "--train", str(tmp_path / "train.chat.jsonl")]) == 0
    meta = json.loads((d / "meta.json").read_text())
    # example states in the bench rows are not benchmark items; the 4 test items are counted once each
    assert meta["n"] == 4 and meta["overlap"]["n_items"] == 4 and meta["overlap"]["n_overlapping"] == 1
    assert meta["overlap"]["overlapping_ids"] == ["decidebench:action_review-001a"]
    assert json.loads((d / "overlap.json").read_text())["share"] == 0.25


def test_bench_overlap_without_rows_fails(tmp_path):
    assert cli.main(["bench", "overlap", "--name", "jevbench", "--dir", str(tmp_path), "--train", "x"]) == 2
