"""n_max: 6 (Stage 0) by default, up to 10 options (letters A-J) for Stage 1."""
import json

import pytest
import torch

from yev import cli
from yev.format import LETTERS
from yev.train import infer, readout
from yev.train.data import N_MAX, build_batches, encode, encode_all, letter_token_ids
from yev.train.trainer import TrainConfig, _batch_loss, train


def _letters(n):
    return {LETTERS[i]: f"k{i}" for i in range(n)}


def _cfg(tmp_path, **kw):
    base = dict(name="t", model_path="unused", train_path=str(tmp_path / "train.jsonl"), out_dir=str(tmp_path / "out"),
                lora=False, grad_checkpointing=False, bf16=False, token_budget=4000, grad_accum=1, save_every=100,
                lr=1e-3, lambda_pair=0.5, lambda_perm=0.2, lambda_rps=0.5)
    base.update(kw)
    return TrainConfig(**base)


def test_default_n_max_is_six():
    assert N_MAX == 6
    assert TrainConfig(name="x", model_path="m", train_path="t", out_dir="o").n_max == 6


def test_ten_option_row_targets_and_letters_aligned(tok, tiny_model, make_row):
    row = make_row("r10", _letters(10), "H")
    e = encode(row, tok, max_len=4096, n_max=10)
    assert e["n_letters"] == 10 and len(e["target"]) == 10
    assert e["target"].index(1.0) == LETTERS.index("H") == e["answer_letter"]
    ids = letter_token_ids(tok, 10)
    assert len(ids) == 10 and ids[7] == tok.encode("H")[0]
    z = infer.letter_logits(tiny_model, tok, [row], max_len=4096, n_letters=10)[0]
    assert len(z) == 10
    p = readout.probs(z, n=len(row["letters"]))
    assert len(p) == 10 and abs(sum(p) - 1) < 1e-6


def test_encode_refuses_row_above_n_max(tok, make_row):
    with pytest.raises(ValueError, match="n_max"):
        encode(make_row("r7", _letters(7), "A"), tok, max_len=4096)


def test_n_max_six_bit_identical_to_stage0(tmp_path, tok, tiny_model, make_row):
    rows = [make_row("a", _letters(3), "B", cluster="c"), make_row("b", _letters(6), "A", cluster="c"),
            make_row("c", _letters(4), "D", type_="score"), make_row("d", _letters(2), "A")]
    for r in rows:
        assert encode(r, tok, 4096) == encode(r, tok, 4096, n_max=6)
    implicit, s1 = encode_all(rows, tok, 4096, 4096, 0.05, 0.1, 1.0, 0)
    explicit, s2 = encode_all(rows, tok, 4096, 4096, 0.05, 0.1, 1.0, 0, n_max=6)
    assert implicit == explicit and s1 == s2
    assert build_batches(implicit, 4000, 0) == build_batches(explicit, 4000, 0, n_max=6)
    cfg = _cfg(tmp_path)
    assert cfg.n_max == 6
    ids = torch.tensor(letter_token_ids(tok))
    a, _, pa = _batch_loss(cfg, tiny_model, implicit, ids, 0, "cpu")
    b, _, pb = _batch_loss(_cfg(tmp_path, n_max=6), tiny_model, explicit, torch.tensor(letter_token_ids(tok, 6)), 0, "cpu")
    assert torch.equal(a, b) and pa == pb


def test_ten_option_batch_loss_finite(tmp_path, tok, tiny_model, make_row):
    rows = [make_row("a", _letters(10), "J", cluster="c"), make_row("b", _letters(8), "C", cluster="c"),
            make_row("c", _letters(3), "B")]
    enc, stats = encode_all(rows, tok, 4096, 4096, 0.05, 0.1, 1.0, 0, n_max=10)
    assert stats["kept"] == 3 and all(len(e["target"]) == 10 for e in enc)
    loss, _, parts = _batch_loss(_cfg(tmp_path, n_max=10), tiny_model, enc,
                                 torch.tensor(letter_token_ids(tok, 10)), 0, "cpu")
    assert torch.isfinite(loss) and "pair" in parts and "perm" in parts


def test_build_batches_rejects_rows_above_n_max(tok, make_row):
    enc, _ = encode_all([make_row("a", _letters(8), "A")], tok, 4096, 4096, 0.05, 0.1, 0.0, 0, n_max=10)
    with pytest.raises(ValueError, match="n_max"):
        build_batches(enc, 4000, 0)


def test_trainer_rejects_rows_above_n_max(tmp_path, tok, tiny_model, make_row):
    rows = [make_row(f"r{i}", _letters(3), "A") for i in range(4)] + [make_row("big", _letters(8), "B")]
    enc, stats = encode_all(rows, tok, 4096, 4096, 0.05, 0.1, 0.0, 0)
    assert stats == {"kept": 4, "dropped_overlong": 0, "dropped_n_max": 1}
    assert "big" not in {e["id"] for e in enc}
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = train(_cfg(tmp_path, max_steps=1), tokenizer=tok, model=tiny_model)
    assert out["encode_stats"]["dropped_n_max"] == 1
    out10 = train(_cfg(tmp_path, max_steps=1, n_max=10, out_dir=str(tmp_path / "out10")), tokenizer=tok, model=tiny_model)
    assert out10["encode_stats"]["dropped_n_max"] == 0 and out10["encode_stats"]["kept"] == 5
    saved = json.loads((tmp_path / "out10" / "final" / "train_config.json").read_text())
    assert saved["n_max"] == 10


def _cli_setup(tmp_path, monkeypatch, tok, tiny_model, rows):
    data = tmp_path / "d.chat.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(infer, "load", lambda model_dir, base: (tok, tiny_model))
    return data


def test_eval_and_calibrate_use_n_max(tmp_path, monkeypatch, tok, tiny_model, make_row):
    rows = [make_row("a", _letters(10), "J"), make_row("b", _letters(3), "B")]
    data = _cli_setup(tmp_path, monkeypatch, tok, tiny_model, rows)
    cal, rep = tmp_path / "cal.json", tmp_path / "rep.json"
    # default n_max=6 refuses the 10-option row, like an over-long row
    assert cli.main(["calibrate", "--model", "m", "--data", str(data), "--out", str(cal)]) == 2
    assert cli.main(["eval", "--model", "m", "--data", str(data), "--out", str(rep)]) == 2
    assert not cal.exists() and not rep.exists()
    assert cli.main(["calibrate", "--model", "m", "--data", str(data), "--out", str(cal), "--n-max", "10"]) == 0
    assert json.loads(cal.read_text())["n"] == 2
    assert cli.main(["eval", "--model", "m", "--data", str(data), "--out", str(rep), "--n-max", "10",
                     "--calibration", str(cal)]) == 0
    r = json.loads(rep.read_text())
    assert r["n"] == 2 and r["run"]["n_max"] == 10


def test_eval_reads_n_max_from_run_config(tmp_path, monkeypatch, tok, tiny_model, make_row):
    rows = [make_row("a", _letters(10), "J"), make_row("b", _letters(3), "B")]
    data = _cli_setup(tmp_path, monkeypatch, tok, tiny_model, rows)
    model = tmp_path / "run" / "final"
    model.mkdir(parents=True)
    (model / "train_config.json").write_text(json.dumps({"name": "x", "n_max": 10}))
    rep = tmp_path / "rep.json"
    assert cli.main(["eval", "--model", str(model), "--data", str(data), "--out", str(rep)]) == 0
    assert json.loads(rep.read_text())["run"]["n_max"] == 10
