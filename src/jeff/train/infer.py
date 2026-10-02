"""Letter logits at the answer position, one forward pass per row (spec §6)."""
from __future__ import annotations

import torch

from jeff.train.data import N_MAX, letter_token_ids, prompt_ids


def logits_at(model, input_ids, attention_mask, pos):
    """Vocab logits at pos[b] of each row: [B, V].

    Only the distinct positions are projected through the LM head (logits_to_keep), so a
    batch never materialises [B, T, V] logits over a ~250k vocabulary.
    """
    uniq, inv = torch.unique(pos, return_inverse=True)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=uniq).logits  # [B, K, V]
    return logits[torch.arange(len(pos), device=logits.device), inv.to(logits.device)]


@torch.no_grad()
def letter_logits(model, tokenizer, rows, max_len: int, batch_tokens: int = 16384,
                  n_letters: int = N_MAX) -> list[list[float]]:
    """Logits of the first n_letters letter tokens at each row's answer position, in row order.

    A row longer than max_len keeps only its last max_len tokens; callers that must not truncate
    check lengths first (jeff.train.data.overlong_ids). Rows with fewer letters than n_letters get
    the extra logits too; readout.probs uses only a row's own.
    """
    model.eval()
    ids_letters = torch.tensor(letter_token_ids(tokenizer, n_letters))
    dev = next(model.parameters()).device
    out: list[list[float] | None] = [None] * len(rows)
    enc = []
    for i, r in enumerate(rows):
        ids = prompt_ids(tokenizer, r["messages"])[-max_len:]
        enc.append((i, ids))
    enc.sort(key=lambda x: len(x[1]))
    pad = getattr(tokenizer, "pad_token_id", 0) or 0

    def flush(batch):
        L = max(len(x[1]) for x in batch)
        inp = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for k, (_, ids) in enumerate(batch):  # right-pad (linear attention must not see leading pads)
            inp[k, : len(ids)] = torch.tensor(ids)
            att[k, : len(ids)] = 1
        last = torch.tensor([len(ids) - 1 for _, ids in batch], device=dev)
        rows_ = logits_at(model, inp.to(dev), att.to(dev), last).float().cpu()
        for k, (i, _) in enumerate(batch):
            out[i] = rows_[k, ids_letters].tolist()

    b: list = []
    for item in enc:
        if b and (len(b) + 1) * max(len(item[1]), max(len(x[1]) for x in b)) > batch_tokens:
            flush(b)
            b = []
        b.append(item)
    if b:
        flush(b)
    return out  # type: ignore[return-value]


def load(model_dir: str, base: str | None):
    """Tokenizer and eval-mode model: a LoRA adapter dir on top of `base`, or a full model dir."""
    from pathlib import Path
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    if (Path(model_dir) / "adapter_config.json").exists():
        if not base:
            raise ValueError(f"{model_dir} is a LoRA adapter; pass --base")
        from peft import PeftModel
        tok = AutoTokenizer.from_pretrained(base)
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype=dtype), model_dir)
    else:
        tok = AutoTokenizer.from_pretrained(base or model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
    return tok, model.to("cuda" if cuda else "cpu").eval()
