"""Chat rows → tokenised, smoothed, unit-grouped training batches (Plan 3 Task 3)."""
from __future__ import annotations

import json
import math
import random

from yev.format import LETTERS
from yev.train.targets import smooth_target

N_MAX = 6  # Stage 0 default; Stage 1 configs set n_max up to 10 (letters A-J)


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def unit_of(row: dict) -> str:
    return row.get("cluster_id") or row.get("twin_of") or row["id"]


def make_twin(row: dict, rng: random.Random) -> dict | None:
    if row["type"] == "score" or len(row["letters"]) < 2:
        return None
    msgs = [dict(m) for m in row["messages"]]
    user = json.loads(msgs[-1]["content"])
    opts = list(user["options"])
    for _ in range(10):
        rng.shuffle(opts)
        if [o["key"] for o in opts] != [o["key"] for o in user["options"]]:
            break
    for i, o in enumerate(opts):
        o["label"] = LETTERS[i]
    user["options"] = opts
    msgs[-1]["content"] = json.dumps(user, ensure_ascii=False)
    old_by_key = {k: L for L, k in row["letters"].items()}
    letters = {LETTERS[i]: o["key"] for i, o in enumerate(opts)}
    target = {L: row["target"][old_by_key[k]] for L, k in letters.items()}
    gold_key = row["letters"][row["answer"]]
    answer = next(L for L, k in letters.items() if k == gold_key)
    return {**row, "id": row["id"] + "#twin", "messages": msgs, "letters": letters, "target": target,
            "answer": answer, "twin_of": row["id"]}


def letter_token_ids(tokenizer, n: int = N_MAX) -> list[int]:
    """Token ids of the first n letter labels; each must be a single token, all distinct."""
    assert 1 <= n <= len(LETTERS), f"n = {n} letters, but only {len(LETTERS)} labels exist"
    ids = []
    for L in LETTERS[:n]:
        toks = tokenizer.encode(L, add_special_tokens=False)
        assert len(toks) == 1, f"letter {L!r} is {len(toks)} tokens"
        ids.append(toks[0])
    assert len(set(ids)) == n, f"letters {LETTERS[:n]} share token ids: {ids}"
    return ids


def prompt_ids(tokenizer, messages) -> list[int]:
    """The prompt's token ids, ending in the generation prompt: the answer is read at the last one."""
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def overlong_ids(tokenizer, rows: list[dict], max_len: int) -> list[str]:
    """Ids of rows whose prompt is longer than max_len tokens (letter_logits would cut their start)."""
    return [r["id"] for r in rows if len(prompt_ids(tokenizer, r["messages"])) > max_len]


def encode(row: dict, tokenizer, max_len: int, n_max: int = N_MAX) -> dict | None:
    """Encoded row (target padded to n_max), or None when the prompt is longer than max_len."""
    n = len(row["letters"])
    if n > n_max:
        raise ValueError(f"{row['id']}: {n} options exceeds n_max = {n_max}")
    ids = prompt_ids(tokenizer, row["messages"])
    if len(ids) > max_len:
        return None
    target = [row["target"].get(LETTERS[i], 0.0) for i in range(n)] + [0.0] * (n_max - n)
    return {"input_ids": ids, "answer_pos": len(ids) - 1, "n_letters": n, "target": target,
            "answer_letter": LETTERS.index(row["answer"]), "weight": float(row.get("weight", 1.0)),
            "type": row["type"], "id": row["id"], "unit": unit_of(row), "twin_of": row.get("twin_of"),
            "letters": row["letters"], "cluster_id": row.get("cluster_id")}


def subsample_units(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    units = sorted({unit_of(r) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(units)
    keep = set(units[: math.ceil(fraction * len(units))])
    return [r for r in rows if unit_of(r) in keep]


def encode_all(rows, tokenizer, max_len_plain, max_len_examples, eps_hard, eps_ordinal, twin_rate, seed,
               n_max: int = N_MAX):
    """Smoothed, encoded rows (plus twins); rows longer than their max_len or with more than n_max
    options are dropped and counted."""
    rng = random.Random(seed)
    out, stats = [], {"kept": 0, "dropped_overlong": 0, "dropped_n_max": 0}
    for r in rows:
        if len(r["letters"]) > n_max:
            stats["dropped_n_max"] += 1
            continue
        r = {**r, "target": smooth_target(r["target"], r["type"], eps_hard, eps_ordinal)}
        batch = [r]
        if twin_rate > 0 and rng.random() < twin_rate:
            tw = make_twin(r, rng)
            if tw:
                batch.append(tw)
        n_user = sum(m["role"] == "user" for m in r["messages"])
        max_len = max_len_examples if n_user > 1 else max_len_plain
        encs = [encode(x, tokenizer, max_len, n_max) for x in batch]
        if encs[0] is None:
            stats["dropped_overlong"] += 1
            continue
        stats["kept"] += 1
        out.extend(e for e in encs if e is not None)
    return out, stats


def build_batches(encoded: list[dict], token_budget: int, seed: int, window: int = 50,
                  n_max: int = N_MAX, split_oversized: bool = False) -> list[list[int]]:
    """Pack whole units into batches whose padded size (rows x longest row) stays within token_budget.

    Units are shuffled with `seed`, then sorted by length inside windows of `window` units so rows of
    similar length share a batch. A unit that alone exceeds the budget becomes its own batch, or, with
    split_oversized, budget-sized batches of its own rows (for losses that don't need the unit together).
    Deterministic for a given seed. Rows encoded for a different n_max are an error.
    """
    wide = [e.get("id", "?") for e in encoded if "target" in e and len(e["target"]) != n_max]
    if wide:
        raise ValueError(f"{len(wide)} rows do not match n_max = {n_max} (e.g. {', '.join(wide[:5])})")
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(encoded):
        groups.setdefault(e["unit"], []).append(i)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    longest = {u: max(len(encoded[i]["input_ids"]) for i in groups[u]) for u in order}
    order = [u for w in range(0, len(order), window) for u in sorted(order[w:w + window], key=longest.__getitem__)]
    batches, cur, cur_max = [], [], 0
    for u in order:
        idx = groups[u]
        if split_oversized and len(idx) > 1 and len(idx) * longest[u] > token_budget:
            # A unit too big for one batch (e.g. a long document with many questions) is split into
            # budget-sized batches of its own; keeping a unit together only matters for the pair loss.
            if cur:
                batches.append(cur)
                cur, cur_max = [], 0
            per = max(1, token_budget // longest[u])
            batches.extend(idx[k:k + per] for k in range(0, len(idx), per))
            continue
        new_max = max(cur_max, longest[u])
        if cur and (len(cur) + len(idx)) * new_max > token_budget:
            batches.append(cur)
            cur, new_max = [], longest[u]
        cur.extend(idx)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches
