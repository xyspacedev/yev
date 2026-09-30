"""Stage 0 training loop: LoRA, answer-position losses, unit-grouped batches, resumable (Plan 3 Task 6)."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from jeff.train import losses as L
from jeff.train.data import N_MAX, build_batches, encode_all, letter_token_ids, load_rows, subsample_units
from jeff.train.infer import logits_at

LETTERS6 = "ABCDEF"


@dataclass
class TrainConfig:
    name: str
    model_path: str
    train_path: str
    out_dir: str
    fraction: float = 1.0
    seed: int = 0
    epochs: int = 1
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    token_budget: int = 12288
    grad_accum: int = 3
    max_len_plain: int = 2048
    max_len_examples: int = 4096
    eps_hard: float = 0.05
    eps_ordinal: float = 0.10
    lambda_vocab: float = 0.1
    lambda_pair: float = 0.0
    pair_margin: float = 1.0
    lambda_perm: float = 0.0
    twin_rate: float = 0.0
    lambda_rps: float = 0.0
    lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    grad_checkpointing: bool = True
    bf16: bool = True
    save_every: int = 500
    log_every: int = 10
    max_steps: int | None = None

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text()))


def _load(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.bfloat16 if cfg.bf16 else torch.float32)
    return tok, model


def _wrap(cfg, model):
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if cfg.lora:
        from peft import LoraConfig, get_peft_model
        if cfg.grad_checkpointing:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    return model


def _collate(items, pad_id, device):
    Lmax = max(len(e["input_ids"]) for e in items)
    inp = torch.full((len(items), Lmax), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), Lmax), dtype=torch.long)
    for k, e in enumerate(items):  # right-pad; answer_pos indexes the unpadded sequence
        n = len(e["input_ids"])
        inp[k, :n] = torch.tensor(e["input_ids"])
        att[k, :n] = 1
    return inp.to(device), att.to(device)


def _pairs(items, key_idx, gold_key):
    """Same-cluster, non-twin rows with different gold keys (both keys offered in both rows)."""
    pairs = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            if a["twin_of"] or b["twin_of"] or not a["cluster_id"] or a["cluster_id"] != b["cluster_id"]:
                continue
            if gold_key[i] == gold_key[j] or gold_key[i] not in key_idx[j] or gold_key[j] not in key_idx[i]:
                continue
            pairs.append((i, j, key_idx[i][gold_key[i]], key_idx[j][gold_key[i]],
                          key_idx[i][gold_key[j]], key_idx[j][gold_key[j]]))
    return pairs


def _batch_loss(cfg, model, items, letter_ids, pad_id, device):
    inp, att = _collate(items, pad_id, device)
    pos = torch.tensor([e["answer_pos"] for e in items], device=device)
    last = logits_at(model, inp, att, pos).float()                                 # [B, V]
    letter = last[:, letter_ids]                                                   # [B, 6]
    mask = torch.tensor([[i < e["n_letters"] for i in range(N_MAX)] for e in items], device=device)
    target = torch.tensor([e["target"] for e in items], device=device)
    weight = torch.tensor([e["weight"] for e in items], device=device)
    answer_ids = letter_ids[torch.tensor([e["answer_letter"] for e in items], device=device)]
    parts = {"ce": L.ce_soft(letter, target, mask, weight), "vocab": L.full_vocab_ce(last, answer_ids, weight)}
    loss = parts["ce"] + cfg.lambda_vocab * parts["vocab"]
    logp = L.masked_log_softmax(letter, mask)
    key_idx = [{k: LETTERS6.index(Lt) for Lt, k in e["letters"].items()} for e in items]
    gold_key = [e["letters"][LETTERS6[e["answer_letter"]]] for e in items]
    if cfg.lambda_pair:
        parts["pair"] = L.pair_margin(logp, _pairs(items, key_idx, gold_key), cfg.pair_margin)
        loss = loss + cfg.lambda_pair * parts["pair"]
    if cfg.lambda_perm:
        by_id = {e["id"]: k for k, e in enumerate(items)}
        pa, pb = [], []
        for k, e in enumerate(items):
            if e["twin_of"] and e["twin_of"] in by_id:
                o = by_id[e["twin_of"]]
                keys = list(items[o]["letters"].values())
                pa.append(torch.stack([logp[o, key_idx[o][x]] for x in keys]).exp())
                pb.append(torch.stack([logp[k, key_idx[k][x]] for x in keys]).exp())
        if pa:
            # per pair: twins of rows with different option counts can share a micro-batch
            parts["perm"] = torch.stack([L.perm_skl(a[None], b[None]) for a, b in zip(pa, pb)]).mean()
            loss = loss + cfg.lambda_perm * parts["perm"]
    if cfg.lambda_rps:
        sel = [k for k, e in enumerate(items) if e["type"] == "score"]
        if sel:
            s = torch.tensor(sel, device=device)
            parts["rps"] = L.rps(logp[s].exp(), target[s], mask[s])
            loss = loss + cfg.lambda_rps * parts["rps"]
    return loss, int(att.sum()), {k: float(v.detach()) for k, v in parts.items()}


def _latest_checkpoint(out: Path):
    """Newest complete checkpoint; skips *.tmp and directories without state.pt (interrupted saves)."""
    cks = [p for p in out.glob("checkpoint-*")
           if p.is_dir() and p.name.split("-", 1)[1].isdigit() and (p / "state.pt").exists()]
    cks.sort(key=lambda p: int(p.name.split("-", 1)[1]))
    return cks[-1] if cks else None


def _save(model, opt, sched, step, out: Path, cfg, name=None):
    final = out / (name or f"checkpoint-{step}")
    d = final.with_name(final.name + ".tmp")  # written whole, then renamed, so a kill mid-save leaves no partial dir
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    model.save_pretrained(d)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                "rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, d / "state.pt")
    # config.json belongs to save_pretrained for a full model; the run config sits beside it.
    (d / "train_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    if final.exists():
        shutil.rmtree(final)
    os.replace(d, final)


def _load_checkpoint(cfg, model, ck: Path):
    if cfg.lora:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        res = set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors")))
        if res.unexpected_keys:
            raise RuntimeError(f"adapter keys in {ck} do not match the model: {res.unexpected_keys[:5]}")
    else:
        from transformers import AutoModelForCausalLM
        model.load_state_dict(AutoModelForCausalLM.from_pretrained(ck).state_dict())


def train(cfg: TrainConfig, tokenizer=None, model=None) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if tokenizer is None or model is None:
        tokenizer, model = _load(cfg)
    rows = load_rows(cfg.train_path)
    if cfg.fraction < 1.0:
        rows = subsample_units(rows, cfg.fraction, cfg.seed)
    enc, stats = encode_all(rows, tokenizer, cfg.max_len_plain, cfg.max_len_examples, cfg.eps_hard,
                            cfg.eps_ordinal, cfg.twin_rate, cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _wrap(cfg, model).to(device)
    letter_ids = torch.tensor(letter_token_ids(tokenizer), device=device)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    # Each epoch reshuffles units with seed + epoch. max_steps, when set, overrides epochs (as in HF
    # Trainer): training runs exactly max_steps optimizer steps, adding epochs if the data runs out.
    per_epoch = len(build_batches(enc, cfg.token_budget, cfg.seed))
    epochs = cfg.epochs
    if cfg.max_steps:
        epochs = max(1, math.ceil(cfg.max_steps * cfg.grad_accum / max(per_epoch, 1)))
    batches = [b for ep in range(epochs) for b in build_batches(enc, cfg.token_budget, cfg.seed + ep)]
    total_steps = math.ceil(len(batches) / cfg.grad_accum)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_ratio * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    step, resumed_from = 0, None
    ck = _latest_checkpoint(out)
    if ck is not None:
        _load_checkpoint(cfg, model, ck)
        st = torch.load(ck / "state.pt", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = resumed_from = st["step"]
        random.setstate(st["rng"])
        torch.set_rng_state(st["torch_rng"])
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda_rng"])
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / "tb"))
    except Exception:
        pass
    model.train()
    t0, tokens, micro = time.time(), 0, step * cfg.grad_accum
    use_amp = cfg.bf16 and device == "cuda"
    acc: dict[str, float] = {}
    n_acc = 0
    while step < total_steps and micro < len(batches):
        items = [enc[i] for i in batches[micro]]
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
            loss, ntok, parts = _batch_loss(cfg, model, items, letter_ids, pad_id, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}, micro-batch {micro}: {parts}")
        (loss / cfg.grad_accum).backward()
        tokens += ntok
        micro += 1
        n_acc += 1
        for k, v in {"loss": float(loss.detach()), **parts}.items():
            acc[k] = acc.get(k, 0.0) + v
        # the last partial accumulation window still steps, so no gradients are dropped
        if micro % cfg.grad_accum == 0 or micro == len(batches):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if writer and step % cfg.log_every == 0:
                for k, v in acc.items():
                    writer.add_scalar(f"train/{k}", v / n_acc, step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            acc, n_acc = {}, 0
            if step % cfg.save_every == 0:
                _save(model, opt, sched, step, out, cfg)
    if writer:
        writer.close()
    _save(model, opt, sched, step, out, cfg, name="final")
    secs = time.time() - t0
    summary = {"steps": step, "tokens": tokens, "seconds": secs, "tokens_per_s": tokens / max(secs, 1e-9),
               "encode_stats": stats, "resumed_from": resumed_from, "epochs": epochs, "n_batches": len(batches),
               "total_steps": total_steps}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
