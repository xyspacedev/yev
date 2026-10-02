"""Shared benchmark scoring: predictions from letter logits, accuracy, pair accuracy, binary F1, calibration.

Offline only: nothing here touches the network.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable

from yev.format import LETTERS
from yev.train import metrics as dev_metrics
from yev.train import readout


def prediction(row: dict, logits: list[float], temperature: float = 1.0) -> dict:
    n = len(row["letters"])
    p = readout.probs(logits, n=n, temperature=temperature)
    top = max(range(n), key=lambda i: p[i])
    return {
        "id": row["id"],
        "letters": row["letters"],
        "probs": p,
        "pred": row["letters"][LETTERS[top]],
        "gold": row["letters"][row["answer"]],
        # carried for the scorers and the shared calibration metrics
        "answer": row["answer"],
        "type": row["type"],
        "family": row["family"],
        "source": row["source"],
        "cluster_id": row.get("cluster_id"),
        "bench": row.get("bench") or {},
    }


def prob_of(pred: dict, key: str) -> float:
    for i, (L, k) in enumerate(pred["letters"].items()):
        if k == key:
            return pred["probs"][LETTERS.index(L)]
    raise KeyError(key)


def accuracy(preds: list[dict]) -> float | None:
    return sum(p["pred"] == p["gold"] for p in preds) / len(preds) if preds else None


def pair_accuracy(preds: list[dict], group: Callable[[dict], str | None]) -> tuple[float | None, int]:
    """Share of groups with at least two members where every member is right."""
    by: dict[str, list[bool]] = defaultdict(list)
    for p in preds:
        g = group(p)
        if g is not None:
            by[g].append(p["pred"] == p["gold"])
    full = [all(v) for v in by.values() if len(v) >= 2]
    return (sum(full) / len(full) if full else None), len(full)


def pair_bootstrap_ci(preds: list[dict], group: Callable[[dict], str], n: int = 2000, seed: int = 0) -> list[float]:
    """95% CI for accuracy resampling whole pairs (DecideBench score.py)."""
    by: dict[str, list[bool]] = defaultdict(list)
    for p in preds:
        by[group(p)].append(p["pred"] == p["gold"])
    groups = list(by.values())
    if not groups:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    accs = sorted(
        sum(sum(g) for g in s) / sum(len(g) for g in s)
        for s in ([rng.choice(groups) for _ in groups] for _ in range(n))
    )

    def pct(q: float) -> float:
        i = (len(accs) - 1) * q
        lo, hi = int(i), min(int(i) + 1, len(accs) - 1)
        return accs[lo] + (accs[hi] - accs[lo]) * (i - lo)

    return [pct(0.025), pct(0.975)]


def binary(preds: list[dict], positive: str, threshold: float = 0.5) -> dict:
    """Precision/recall/F1 with `positive` as the positive class; predicted positive when P(positive) >= threshold.

    Zero denominators give 0.0 (sklearn's zero_division=0), so a scorer never returns NaN.
    """
    tp = fp = fn = tn = 0
    for p in preds:
        pred_pos = prob_of(p, positive) >= threshold
        gold_pos = p["gold"] == positive
        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos:
            fp += 1
        elif gold_pos:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    n = tp + fp + fn + tn
    return {
        "n": n, "positive": positive, "threshold": threshold,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "precision": prec, "recall": rec,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / n if n else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def by_field(preds: list[dict], key: Callable[[dict], str]) -> dict:
    out: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        out[str(key(p))].append(p)
    return {k: {"n": len(v), "accuracy": accuracy(v)} for k, v in sorted(out.items())}


def calibration(preds: list[dict]) -> dict:
    """ECE (15 bins, top choice) and multi-class Brier from yev.train.metrics."""
    h = dev_metrics.calibration(preds)
    return {"ece_15": h["ece"], "brier": h["brier"]}


def common(preds: list[dict]) -> dict:
    return {"n": len(preds), "accuracy": accuracy(preds), **calibration(preds)}
