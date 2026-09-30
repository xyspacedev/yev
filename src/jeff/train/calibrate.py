"""Per-type temperature scaling on the calibration split (spec §5 Stage 2b, §6)."""
from __future__ import annotations
import json, math

TYPES = ("choice", "noul", "score")


def _nll(items, t):
    tot = 0.0
    for it in items:
        z = [x / t for x in it["logits"][: it["n"]]]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        tot += lse - z[it["answer_index"]]
    return tot / len(items)


def _fit(items):
    grid = [math.exp(math.log(0.05) + i * (math.log(20) - math.log(0.05)) / 199) for i in range(200)]
    best = min(grid, key=lambda t: _nll(items, t))
    lo, hi = best / 1.05, best * 1.05
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if _nll(items, a) < _nll(items, b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def fit_temperatures(items: list[dict]) -> dict[str, float]:
    out = {}
    for tp in TYPES:
        sub = [x for x in items if x["type"] == tp]
        out[tp] = _fit(sub) if sub else 1.0
    return out


def write_calibration(path, temps: dict, meta: dict) -> None:
    with open(path, "w") as f:
        json.dump({"temperatures": temps, **meta}, f, indent=2)
