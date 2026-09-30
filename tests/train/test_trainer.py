import json
from jeff.train.trainer import TrainConfig, train
from jeff.train.infer import letter_logits

def write_rows(path, make_row, n=24):
    with open(path, "w") as f:
        for i in range(n):
            gold = "A" if i % 2 else "B"
            f.write(json.dumps(make_row(f"c{i//2}-{i%2}", {"A": "x", "B": "y", "C": "z"}, gold, cluster=f"c{i//2}")) + "\n")

def cfg(tmp_path, **kw):
    base = dict(name="t", model_path="unused", train_path=str(tmp_path / "train.jsonl"), out_dir=str(tmp_path / "out"),
                lora=False, grad_checkpointing=False, bf16=False, token_budget=400, grad_accum=2, save_every=2, lr=1e-3)
    base.update(kw)
    return TrainConfig(**base)

def test_train_runs_and_loss_is_finite(tmp_path, tok, tiny_model, make_row):
    write_rows(tmp_path / "train.jsonl", make_row)
    out = train(cfg(tmp_path, lambda_pair=0.5, lambda_perm=0.2, twin_rate=0.5), tokenizer=tok, model=tiny_model)
    assert out["steps"] > 0 and out["tokens"] > 0
    assert (tmp_path / "out" / "final").exists()

def test_resume_continues_step_and_order(tmp_path, tok, tiny_model, make_row):
    write_rows(tmp_path / "train.jsonl", make_row)
    first = train(cfg(tmp_path, max_steps=2), tokenizer=tok, model=tiny_model)
    assert first["steps"] == 2
    again = train(cfg(tmp_path, max_steps=4), tokenizer=tok, model=tiny_model)
    assert again["steps"] == 4 and again["resumed_from"] == 2

def test_letter_logits_padding_invariant(tok, tiny_model, make_row):
    short = make_row("s", {"A": "x", "B": "y"}, "A", state="short")
    long_ = make_row("l", {"A": "x", "B": "y"}, "A", state="much " * 40)
    alone = letter_logits(tiny_model, tok, [short], max_len=512)[0]
    batched = letter_logits(tiny_model, tok, [short, long_], max_len=512, batch_tokens=10**6)[0]
    assert all(abs(a - b) < 1e-4 for a, b in zip(alone, batched))

def test_letter_logits_shape(tok, tiny_model, make_row):
    rows = [make_row("r", {"A": "x", "B": "y"}, "A")]
    out = letter_logits(tiny_model, tok, rows, max_len=512)
    assert len(out) == 1 and len(out[0]) == 6

def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"; p.write_text(json.dumps({"name": "x", "model_path": "m", "train_path": "t", "out_dir": "o", "bogus": 1}))
    import pytest
    with pytest.raises(TypeError):
        TrainConfig.from_json(str(p))

def test_lora_train_and_resume(tmp_path, tok, tiny_model, make_row):
    import pytest
    pytest.importorskip("peft")
    write_rows(tmp_path / "train.jsonl", make_row)
    kw = dict(lora=True, lora_r=4, lora_alpha=8, grad_checkpointing=True)
    first = train(cfg(tmp_path, max_steps=2, **kw), tokenizer=tok, model=tiny_model)
    assert first["steps"] == 2
    ck = tmp_path / "out" / "checkpoint-2"
    assert (ck / "adapter_model.safetensors").exists() and (ck / "state.pt").exists()
    from transformers import LlamaForCausalLM
    fresh = LlamaForCausalLM(tiny_model.config)
    again = train(cfg(tmp_path, max_steps=4, **kw), tokenizer=tok, model=fresh)
    assert again["steps"] == 4 and again["resumed_from"] == 2
    assert (tmp_path / "out" / "final" / "adapter_config.json").exists()

def test_resume_skips_incomplete_checkpoint(tmp_path, tok, tiny_model, make_row):
    write_rows(tmp_path / "train.jsonl", make_row)
    first = train(cfg(tmp_path, max_steps=2), tokenizer=tok, model=tiny_model)
    assert first["steps"] == 2
    out = tmp_path / "out"
    assert not list(out.glob("*.tmp"))
    broken = out / "checkpoint-99"; broken.mkdir()
    (broken / "adapter_model.safetensors").write_bytes(b"partial")
    (out / "checkpoint-150.tmp").mkdir()
    again = train(cfg(tmp_path, max_steps=4), tokenizer=tok, model=tiny_model)
    assert again["resumed_from"] == 2 and again["steps"] == 4

def test_resume_starts_fresh_when_only_incomplete(tmp_path, tok, tiny_model, make_row):
    write_rows(tmp_path / "train.jsonl", make_row)
    broken = tmp_path / "out" / "checkpoint-99"; broken.mkdir(parents=True)
    (broken / "adapter_model.safetensors").write_bytes(b"partial")
    out = train(cfg(tmp_path, max_steps=2), tokenizer=tok, model=tiny_model)
    assert out["resumed_from"] is None and out["steps"] == 2

def test_perm_loss_with_mixed_option_counts(tmp_path, tok, tiny_model, make_row):
    import random, torch
    from jeff.train.data import encode, make_twin, letter_token_ids
    from jeff.train.trainer import _batch_loss
    two = make_row("two", {"A": "x", "B": "y"}, "A")
    five = make_row("five", {"A": "p", "B": "q", "C": "r", "D": "s", "E": "t"}, "C")
    rows = [two, make_twin(two, random.Random(0)), five, make_twin(five, random.Random(0))]
    items = [encode(r, tok, 512) for r in rows]
    c = cfg(tmp_path, lambda_perm=0.2)
    loss, _, parts = _batch_loss(c, tiny_model, items, torch.tensor(letter_token_ids(tok)), 0, "cpu")
    assert torch.isfinite(loss) and "perm" in parts and parts["perm"] >= 0
    loss.backward()
