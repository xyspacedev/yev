"""Training-time target smoothing (Plan 3 ruling 2)."""
from __future__ import annotations


def _is_hard(target: dict[str, float]) -> bool:
    vals = sorted(target.values())
    return vals[-1] == 1.0 and all(v == 0.0 for v in vals[:-1])


def smooth_target(target: dict[str, float], type_: str, eps_hard: float = 0.05, eps_ordinal: float = 0.10) -> dict[str, float]:
    if not _is_hard(target):
        return dict(target)
    letters = sorted(target)
    gold = max(letters, key=lambda L: target[L])
    if type_ == "score":
        i = letters.index(gold)
        neigh = [j for j in (i - 1, i + 1) if 0 <= j < len(letters)]
        out = {L: 0.0 for L in letters}
        out[gold] = 1.0 - eps_ordinal
        for j in neigh:
            out[letters[j]] += eps_ordinal / len(neigh)
        return out
    k = len(letters)
    return {L: (1.0 - eps_hard) + eps_hard / k if L == gold else eps_hard / k for L in letters}
