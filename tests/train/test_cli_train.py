import json

from yev import cli
from yev.train import infer


def _setup(tmp_path, monkeypatch, tok, tiny_model, make_row):
    rows = [make_row("r0", {"A": "x", "B": "y"}, "A", cluster="c0"),
            make_row("r1", {"A": "x", "B": "y", "C": "z"}, "C", cluster="c0"),
            make_row("r2", {"A": "1", "B": "2", "C": "3"}, "B", type_="score")]
    data = tmp_path / "d.chat.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    seen = {}
    def fake_load(model_dir, base):
        seen["args"] = (model_dir, base)
        return tok, tiny_model
    monkeypatch.setattr(infer, "load", fake_load)
    return data, seen


def test_calibrate_then_eval(tmp_path, monkeypatch, tok, tiny_model, make_row):
    data, seen = _setup(tmp_path, monkeypatch, tok, tiny_model, make_row)
    cal = tmp_path / "cal.json"
    assert cli.main(["calibrate", "--model", "m", "--base", "b", "--data", str(data), "--out", str(cal)]) == 0
    c = json.loads(cal.read_text())
    assert set(c["temperatures"]) == {"choice", "noul", "score"} and c["n"] == 3 and c["model"] == "m"
    rep = tmp_path / "rep.json"
    assert cli.main(["eval", "--model", "m", "--base", "b", "--data", str(data), "--calibration", str(cal),
                     "--out", str(rep)]) == 0
    assert seen["args"] == ("m", "b")
    assert json.loads(rep.read_text())


def test_train_cli_rejects_missing_config(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        cli.main(["train", "--config", str(tmp_path / "nope.json")])


def test_stage0_configs_load():
    from pathlib import Path
    from yev.train.trainer import TrainConfig
    root = Path(__file__).resolve().parents[2] / "configs" / "stage0"
    names = {p.stem for p in root.glob("*.json")}
    assert names == {"smoke", "lc25", "lc50", "lc100", "ab_pair", "ab_perm", "ab_both"}
    for p in root.glob("*.json"):
        c = TrainConfig.from_json(str(p))
        assert c.name == p.stem and c.out_dir == f"/home/ubuntu/runs/{c.name}" and c.lambda_rps == 0.5


def test_train_skips_finished_run_unless_forced(tmp_path, monkeypatch, capsys):
    from yev.train import trainer
    out = tmp_path / "out"
    (out / "final").mkdir(parents=True)
    summ = out / "train_summary.json"
    summ.write_text(json.dumps({"steps": 5, "total_steps": 5}))
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"name": "x", "model_path": "m", "train_path": "t", "out_dir": str(out)}))
    calls = []
    monkeypatch.setattr(trainer, "train", lambda c: calls.append(c) or {"steps": 9})
    assert cli.main(["train", "--config", str(cfg)]) == 0
    assert calls == [] and "already holds a finished run" in capsys.readouterr().out
    assert json.loads(summ.read_text()) == {"steps": 5, "total_steps": 5}
    assert cli.main(["train", "--config", str(cfg), "--force"]) == 0 and len(calls) == 1
    summ.write_text(json.dumps({"steps": 3, "total_steps": 5}))  # unfinished: trains (resumes)
    assert cli.main(["train", "--config", str(cfg)]) == 0 and len(calls) == 2


def test_git_sha_falls_back_to_file(tmp_path, monkeypatch):
    from yev import provenance
    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    (tmp_path / "GIT_SHA").write_text("abc123-dirty\n")
    def no_git(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(provenance.subprocess, "run", no_git)
    assert cli._git_sha() == "abc123-dirty"
    (tmp_path / "GIT_SHA").unlink()
    assert cli._git_sha() is None
