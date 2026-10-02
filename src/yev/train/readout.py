"""Softmax over valid letter logits with a per-type temperature (spec §6)."""
from __future__ import annotations
import math


def probs(letter_logits: list[float], n: int, temperature: float = 1.0) -> list[float]:
    z = [x / temperature for x in letter_logits[:n]]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]
