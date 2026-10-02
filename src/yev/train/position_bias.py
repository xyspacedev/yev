"""Position-bias check (stage 1 spec §5): re-ask a dev row with its options in other orders.

Each option's key travels with its description, so the gold key stays right; letters are
re-assigned A.. in the new order and the user turn is re-rendered with yev.format.user_turn,
byte-identical in format to training. Few-shot example turns are left as they are.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from types import SimpleNamespace

from yev.format import LETTERS, user_turn
from yev.schema import Option


def permute_row(row: dict, order: list[int]) -> dict:
    """The row with its last user turn's options in `order` (indices into the current order)."""
    msgs = [dict(m) for m in row["messages"]]
    user = json.loads(msgs[-1]["content"])
    if set(user) != {"state", "question", "options"}:
        raise ValueError(f"{row['id']}: unexpected user-turn fields {sorted(user)}")
    opts = user["options"]
    if sorted(order) != list(range(len(opts))):
        raise ValueError(f"{row['id']}: {order} is not a permutation of {len(opts)} options")
    new = [Option(opts[i]["key"], opts[i]["description"]) for i in order]
    msgs[-1]["content"] = user_turn(SimpleNamespace(state=user["state"], question=user["question"]), new)
    old_by_key = {k: L for L, k in row["letters"].items()}
    letters = {LETTERS[i]: o.key for i, o in enumerate(new)}
    gold_key = row["letters"][row["answer"]]
    out = {**row, "messages": msgs, "letters": letters,
           "answer": next(L for L, k in letters.items() if k == gold_key)}
    if row.get("target") is not None:
        out["target"] = {L: row["target"][old_by_key[k]] for L, k in letters.items()}
    return out


def orders(row_id: str, n: int, k: int, seed: int = 0) -> list[list[int]]:
    """The original order, then k seeded shuffles (each differing from the original)."""
    ident = list(range(n))
    out = [ident]
    for j in range(k):
        rng = random.Random(f"{seed}:{row_id}:{j}")
        perm = list(ident)
        for _ in range(10):
            rng.shuffle(perm)
            if perm != ident:
                break
        out.append(perm)
    return out


def _tv(p: dict, q: dict) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _argmax(p: dict) -> str:
    return max(p, key=p.get)


def _summary(groups: list[dict]) -> dict:
    agree, tv, acc = [], [], []
    for g in groups:
        tops = [_argmax(p) for p in g["orders"]]
        agree.append(len(set(tops)) == 1)
        pairs = list(combinations(g["orders"], 2))
        tv.append(sum(_tv(p, q) for p, q in pairs) / len(pairs) if pairs else 0.0)
        acc.append([t == g["gold"] for t in tops])
    n = len(groups)
    return {"n_rows": n, "argmax_agreement": sum(agree) / n if n else None,
            "mean_tv_distance": sum(tv) / n if n else None,
            "_acc": acc}


def report(groups: list[dict]) -> dict:
    """groups: one per row, {"n_options", "gold" (key), "orders": [per-key probs, original order first]}.

    argmax_agreement: share of rows whose top key is the same in every order. mean_tv_distance:
    mean over rows of the mean pairwise total-variation distance between the per-key distributions.
    """
    head = _summary(groups)
    acc = head.pop("_acc")
    n_orders = len(groups[0]["orders"]) if groups else 0
    if acc:
        head["accuracy_original"] = sum(a[0] for a in acc) / len(acc)
        head["accuracy_by_order"] = [sum(a[j] for a in acc) / len(acc) for j in range(n_orders)]
    by_n = defaultdict(list)
    for g in groups:
        by_n[g["n_options"]].append(g)
    by = {}
    for n in sorted(by_n):
        s = _summary(by_n[n])
        s.pop("_acc")
        by[str(n)] = s
    return {**head, "n_orders": n_orders, "by_n_options": by}
