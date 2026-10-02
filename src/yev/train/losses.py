"""Losses at the answer-letter position (spec §5)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG = -1e9


def masked_log_softmax(letter_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(letter_logits.masked_fill(~mask, NEG), dim=-1)


def ce_soft(letter_logits, target, mask, weight):
    logp = masked_log_softmax(letter_logits, mask)
    per_row = -(target * logp.masked_fill(~mask, 0.0)).sum(-1)
    return (per_row * weight).sum() / weight.sum().clamp_min(1e-8)


def full_vocab_ce(vocab_logits, answer_ids, weight):
    per_row = F.cross_entropy(vocab_logits.float(), answer_ids, reduction="none")
    return (per_row * weight).sum() / weight.sum().clamp_min(1e-8)


def pair_margin(logp, pairs, margin: float = 1.0):
    """pairs: (i, j, a_i, a_j, b_i, b_j) where a = row i's gold key, b = row j's gold key,
    and x_r is that key's letter index in row r."""
    if not pairs:
        return logp.new_zeros(())
    terms = []
    for i, j, a_i, a_j, b_i, b_j in pairs:
        terms.append(F.relu(margin - (logp[i, a_i] - logp[j, a_j])) + F.relu(margin - (logp[j, b_j] - logp[i, b_i])))
    return torch.stack(terms).mean()


def perm_skl(p_a, p_b, eps: float = 1e-8):
    if p_a.shape[0] == 0:
        return p_a.new_zeros(())
    pa, pb = p_a.clamp_min(eps), p_b.clamp_min(eps)
    kl_ab = (pa * (pa.log() - pb.log())).sum(-1)
    kl_ba = (pb * (pb.log() - pa.log())).sum(-1)
    return (0.5 * (kl_ab + kl_ba)).mean()


def rps(probs, target, mask):
    p = probs.masked_fill(~mask, 0.0).cumsum(-1)
    t = target.masked_fill(~mask, 0.0).cumsum(-1)
    k = mask.sum(-1).clamp_min(2).float()
    return (((p - t) ** 2).masked_fill(~mask, 0.0).sum(-1) / (k - 1)).mean()
