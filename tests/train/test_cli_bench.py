"""`yev bench score` and `yev bench overlap` end to end on a tiny model (no network)."""
import json

from yev import cli
from yev.bench import decidebench as B
from yev.bench import rjudge
from yev.bench.common import write_json, write_rows
from yev.train import infer

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
    monkeypatch.setattr(cli, "BENCH_DIR", tmp_path / "bench")
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
                     "--out", str(out)]) == 0
    preds = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(preds) == 4 and set(preds[0]) == {"id", "letters", "probs", "pred", "gold"}
    assert preds[0]["gold"] == "approve" and abs(sum(preds[0]["probs"]) - 1) < 1e-6
    m = json.loads((out / "metrics.json").read_text())
    assert m["n"] == 4 and m["n_pairs"] == 2 and m["variant"] == "examples"
    assert {"ece_15", "brier", "pair_accuracy", "hard", "reference"} <= set(m)
    assert m["run"]["temperatures"] == {"choice": 2.0} and m["run"]["n_skipped_overlong"] == 0
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


def _rjudge_rows(tmp_path, long_words=0):
    rec = {"id": 1, "profile": "p", "label": 1, "attack_type": "unintended",
           "contents": [[{"role": "user", "content": "do it"}, {"role": "agent", "thought": "t", "action": "a"}]]}
    long = {**rec, "id": 2, "label": 0,
            "contents": [[{"role": "user", "content": "w " * long_words}, {"role": "agent", "thought": "t", "action": "a"}]]}
    rows = rjudge.convert({"data/IoT/home.json": [rec, long, {**rec, "id": 3}]})
    path = tmp_path / "rows.chat.jsonl"
    write_rows(path, rows)
    return path


def test_bench_score_refuses_overlong_rows(tmp_path, monkeypatch, tok, tiny_model, capsys):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    rows = _rjudge_rows(tmp_path, long_words=300)
    out = tmp_path / "o"
    assert cli.main(["bench", "score", "--name", "rjudge", "--model", "m", "--rows", str(rows),
                     "--out", str(out), "--max-len", "200"]) == 2
    msg = capsys.readouterr().out
    assert "1 rows are longer than --max-len 200" in msg and "rjudge:IoT/home:2" in msg
    assert not (out / "metrics.json").exists()


def test_bench_score_skip_overlong_records_ids(tmp_path, monkeypatch, tok, tiny_model):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    rows = _rjudge_rows(tmp_path, long_words=300)
    out = tmp_path / "o"
    assert cli.main(["bench", "score", "--name", "rjudge", "--model", "m", "--rows", str(rows),
                     "--out", str(out), "--max-len", "200", "--skip-overlong"]) == 0
    m = json.loads((out / "metrics.json").read_text())
    assert m["n"] == 2 and m["run"]["n_skipped_overlong"] == 1
    assert m["run"]["skipped_overlong_ids"] == ["rjudge:IoT/home:2"]
    ids = [json.loads(line)["id"] for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert ids == ["rjudge:IoT/home:1", "rjudge:IoT/home:3"]


def _long_data(tmp_path, make_row):
    data = tmp_path / "d.chat.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in [
        make_row("r0", {"A": "x", "B": "y"}, "A"),
        make_row("r1", {"A": "x", "B": "y"}, "A", state="w " * 300),
        make_row("r2", {"A": "x", "B": "y"}, "B")]))
    return data


def test_eval_refuses_overlong_rows(tmp_path, monkeypatch, tok, tiny_model, make_row, capsys):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    rep = tmp_path / "r.json"
    assert cli.main(["eval", "--model", "m", "--data", str(_long_data(tmp_path, make_row)), "--out", str(rep),
                     "--max-len", "100"]) == 2
    msg = capsys.readouterr().out
    assert "1 rows are longer than --max-len 100" in msg and "r1" in msg and not rep.exists()


def test_eval_skip_overlong_records_ids(tmp_path, monkeypatch, tok, tiny_model, make_row):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    rep = tmp_path / "r.json"
    assert cli.main(["eval", "--model", "m", "--data", str(_long_data(tmp_path, make_row)), "--out", str(rep),
                     "--max-len", "100", "--skip-overlong"]) == 0
    r = json.loads(rep.read_text())
    assert r["run"]["n_skipped_overlong"] == 1 and r["run"]["skipped_overlong_ids"] == ["r1"]


def test_calibrate_refuses_and_skips_overlong(tmp_path, monkeypatch, tok, tiny_model, make_row, capsys):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    data, cal = _long_data(tmp_path, make_row), tmp_path / "cal.json"
    base = ["calibrate", "--model", "m", "--data", str(data), "--out", str(cal), "--max-len", "100"]
    assert cli.main(base) == 2
    assert "r1" in capsys.readouterr().out and not cal.exists()
    assert cli.main(base + ["--skip-overlong"]) == 0
    c = json.loads(cal.read_text())
    assert c["n"] == 2 and c["skipped_overlong_ids"] == ["r1"] and c["n_skipped_overlong"] == 1


def test_decidebench_is_scored_once_unless_forced(tmp_path, monkeypatch, tok, tiny_model, capsys):
    monkeypatch.setattr(infer, "load", lambda m, b: (tok, tiny_model))
    test, ex = _items()
    rows = tmp_path / "rows_zeroshot.chat.jsonl"
    write_rows(rows, B.convert(test, ex)["zeroshot"])
    out = tmp_path / "o"
    monkeypatch.setattr(cli, "BENCH_DIR", tmp_path / "bench")
    args = ["bench", "score", "--name", "decidebench", "--model", "m", "--rows", str(rows), "--out", str(out)]
    assert cli.main(args) == 0
    assert (tmp_path / "bench/decidebench/.scored-zeroshot").exists()
    # deleting metrics.json (or using another --out) does not bypass the sentinel
    (out / "metrics.json").unlink()
    assert cli.main(args) == 2
    assert cli.main(args[:-1] + [str(tmp_path / "o2")]) == 2
    assert cli.main(args + ["--force"]) == 0
    (out / "metrics.json").write_text('{"first": true}')
    assert cli.main(args) == 2
    assert "scored once" in capsys.readouterr().out
    assert json.loads((out / "metrics.json").read_text()) == {"first": True}
    assert cli.main(args + ["--force"]) == 0
    assert json.loads((out / "metrics.json").read_text())["n"] == 4
    # other benchmarks may be rescored into the same directory
    rj = _rjudge_rows(tmp_path)
    assert cli.main(["bench", "score", "--name", "rjudge", "--model", "m", "--rows", str(rj), "--out", str(out)]) == 0
