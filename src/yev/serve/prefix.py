"""State-prefix KV caching: read a request's shared prompt prefix once, then only each question's suffix.

All questions of a ``/v1/systemone`` request share the system prompt, the chat header and the start of
the JSON user turn (the state), and diverge at the question. ``prefix_letter_logits`` runs the shared
prefix once with ``use_cache=True`` and continues every suffix from a per-batch copy of that cache.

The cache is whatever the model returns. For Qwen3.5 (transformers 5.x) it is a ``DynamicCache``
whose layers are ``DynamicLayer`` (keys/values ``[B, H, T, D]``) for full attention and
``LinearAttentionLayer`` (``conv_states`` / ``recurrent_states`` dicts of ``[B, ...]`` tensors) for the
Gated DeltaNet layers. ``LinearAttentionLayer`` has no ``batch_repeat_interleave``, so
``expand_cache`` copies each layer itself:

- keys/values become ``expand`` views of the batch-1 prefix: a full-attention layer only reads them
  and ``torch.cat`` s the suffix onto them into a fresh tensor, so the prefix is never written;
- every other tensor (the linear-attention conv and recurrent states, updated in place with
  ``copy_``) is materialised with ``repeat``, and every dict is a new dict.

So the prefix cache is never mutated and every batch starts from the same state.

Suffixes are right-padded. The full-attention mask is ones over the prefix plus each suffix's own
mask; the linear-attention layers zero padded positions (``apply_mask_to_padding_states``) and see
pads only after the real tokens; each row is read at its own last real position. Position ids
continue from the prefix length (``P + arange(S)``), which Qwen3.5 expands to its 4-row rope ids.

Batching: a suffix batch holds at most ``SUFFIX_BATCH`` rows and at most ``batch_tokens`` suffix
tokens (rows x longest suffix). The prefix itself is shared, not counted: a row's extra memory is its
full-attention KV for prefix+suffix, which for yev-4b (8 full-attention layers, 4 KV heads x 256,
bf16) is about 32 KB per token, so 16 rows over a 9k-token state are about 5 GB on top of the model.
"""

from __future__ import annotations

import copy

import torch

SUFFIX_BATCH = 16


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Longest common leading token run of all sequences, leaving every sequence at least one token."""
    if not seqs:
        return 0
    cap = min(len(s) for s in seqs) - 1
    first = seqs[0]
    n = 0
    while n < cap and all(s[n] == first[n] for s in seqs):
        n += 1
    return max(n, 0)


def _expand_tensor(t, B: int, view: bool):
    if not isinstance(t, torch.Tensor) or t.dim() == 0 or t.shape[0] != 1:
        return t
    return t.expand(B, *t.shape[1:]) if view else t.repeat(B, *([1] * (t.dim() - 1)))


def expand_cache(cache, B: int):
    """A batch-B copy of a batch-1 cache that shares nothing writable with it."""
    new = copy.copy(cache)
    layers = []
    for layer in cache.layers:
        L = copy.copy(layer)
        for name, v in vars(layer).items():
            if isinstance(v, torch.Tensor):
                setattr(L, name, _expand_tensor(v, B, view=name in ("keys", "values")))
            elif isinstance(v, dict):
                setattr(L, name, {k: _expand_tensor(x, B, view=False) for k, x in v.items()})
            elif isinstance(v, (list, set)):
                setattr(L, name, type(v)(v))
        layers.append(L)
    new.layers = layers
    return new


def _batches(order: list[int], lens: list[int], batch_tokens: int) -> list[list[int]]:
    out: list[list[int]] = []
    b: list[int] = []
    for i in order:
        if b and (len(b) >= SUFFIX_BATCH or (len(b) + 1) * max(lens[i], max(lens[j] for j in b)) > batch_tokens):
            out.append(b)
            b = []
        b.append(i)
    if b:
        out.append(b)
    return out


@torch.no_grad()
def prefix_letter_logits(model, prefix_ids: list[int], suffixes: list[list[int]], letter_ids: list[int],
                         batch_tokens: int, pad_id: int = 0) -> list[list[float]]:
    """Logits of ``letter_ids`` at the last token of ``prefix_ids + suffix``, for each suffix, in order.

    Every suffix must be non-empty (``common_prefix_len`` guarantees it).
    """
    if any(not s for s in suffixes):
        raise ValueError("every suffix needs at least one token")
    model.eval()
    dev = next(model.parameters()).device
    P = len(prefix_ids)
    cache = model(input_ids=torch.tensor([prefix_ids], device=dev), use_cache=True, logits_to_keep=1).past_key_values
    letters = torch.tensor(letter_ids)
    out: list[list[float] | None] = [None] * len(suffixes)
    lens = [len(s) for s in suffixes]
    order = sorted(range(len(suffixes)), key=lambda i: lens[i])
    for batch in _batches(order, lens, batch_tokens):
        B, S = len(batch), max(lens[i] for i in batch)
        inp = torch.full((B, S), pad_id, dtype=torch.long)
        att = torch.zeros((B, P + S), dtype=torch.long)
        att[:, :P] = 1
        for k, i in enumerate(batch):
            inp[k, : lens[i]] = torch.tensor(suffixes[i])
            att[k, P : P + lens[i]] = 1
        pos = torch.arange(P, P + S).unsqueeze(0).expand(B, S)
        last = torch.tensor([lens[i] - 1 for i in batch])
        uniq, inv = torch.unique(last, return_inverse=True)
        logits = model(input_ids=inp.to(dev), attention_mask=att.to(dev), position_ids=pos.to(dev),
                       past_key_values=expand_cache(cache, B), use_cache=True,
                       logits_to_keep=uniq.to(dev)).logits  # [B, K, V]
        rows_ = logits[torch.arange(B, device=logits.device), inv.to(logits.device)].float().cpu()
        for k, i in enumerate(batch):
            out[i] = rows_[k, letters].tolist()
    return out  # type: ignore[return-value]
