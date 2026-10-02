"""Family 7, long_padding: short decisions embedded verbatim inside long distractor text.

Convention. The original state is inserted whole at a paragraph boundary of text taken from our own
generated documents (family 3 invoice_ledger, later family 6 Opus documents; Plan Ruling 3, never public
text). The inserted span is wrapped in two plain lines, FORWARDED_OPEN and FORWARDED_CLOSE, in the same
style as the "==========" section breaks of the surrounding documents. They mark where the item sits for a
human reader but say nothing about it being the relevant part. Labels, question, options, soft gold,
weight, source and licence are those of the original row.

Distractors never change the answer: a distractor document is skipped when it shares its title line with
the row's state, and the CLI refuses to pad rows of the same family as the distractor documents.
Distractor documents must come from train/pool rows only, never dev/calibration or benchmark data.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Callable

from yev.schema import Decision

FAMILY = "long_padding"
FORWARDED_OPEN = "--- Forwarded item ---"
FORWARDED_CLOSE = "--- End of forwarded item ---"
EVAL_SPLITS = ("dev", "calibration")
MAX_TRAIN_TOKENS = 12288  # target_tokens bounds the padded state; the prompt must fit on top of hi
TOKENIZER_NAME = "Qwen/Qwen3.5-4B-Base"
ATTEMPTS = 30


class PaddingError(ValueError):
    """The request would break a padding rule."""


def heuristic_tokens(text: str) -> int:
    """len/4 underestimates table-heavy text by about 1.57x; 1.5x is the documented fallback."""
    return int(len(text) / 4 * 1.5) + 1


def load_token_counter(name: str = TOKENIZER_NAME) -> Callable[[str], int]:
    """Real tokenizer from the local HF cache when available, else the heuristic."""
    try:
        from pathlib import Path
        from transformers import AutoTokenizer
        snaps = sorted((Path.home() / ".cache/huggingface/hub" / ("models--" + name.replace("/", "--")) / "snapshots").glob("*"))
        tok = AutoTokenizer.from_pretrained(str(snaps[-1]))
        return lambda text: len(tok.encode(text, add_special_tokens=False))
    except Exception:
        return heuristic_tokens


def doc_id(text: str) -> str:
    return "doc:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def _title(text: str) -> str:
    return text.strip().split("\n", 1)[0].strip()


def _prompt_tokens(d: Decision, count) -> int:
    return count(d.question) + sum(count(o.description) + 4 for o in d.options) + 200


def _walk(paras: list[str], budget: int, count, rng: random.Random) -> tuple[list[str], list[int]]:
    """Contiguous paragraphs from a random start whose token sum stays within budget."""
    if not paras or budget <= 0:
        return [], []
    start = rng.randrange(len(paras))
    out: list[str] = []
    used = 0
    for p in paras[start:] + paras[:start]:
        n = count(p) + 2
        if used + n > budget:
            break
        out.append(p)
        used += n
    return out, [used]


def pad(rows: list[Decision], distractors: list[str], target_tokens: tuple[int, int], seed,
        distractor_ids: list[str] | None = None, count: Callable[[str], int] | None = None
        ) -> list[Decision]:
    lo, hi = target_tokens
    if not 0 < lo <= hi <= MAX_TRAIN_TOKENS:
        raise PaddingError(f"target_tokens {target_tokens} must satisfy 0 < lo <= hi <= {MAX_TRAIN_TOKENS}")
    bad = [r.id for r in rows if r.split in EVAL_SPLITS]
    if bad:
        raise PaddingError(f"refusing to pad dev/calibration rows: {bad[:3]}")
    if not distractors:
        raise PaddingError("no distractor documents")
    ids = distractor_ids or [doc_id(t) for t in distractors]
    if len(ids) != len(distractors):
        raise PaddingError("distractor_ids length differs from distractors")
    count = count or load_token_counter()
    cache: dict[str, int] = {}

    def cnt(t: str) -> int:
        if t not in cache:
            cache[t] = count(t)
        return cache[t]

    parsed = [paragraphs(t) for t in distractors]
    out: list[Decision] = []
    for row in rows:
        rng = random.Random(f"{seed}:{row.id}")
        title = _title(row.state)
        usable = [i for i, t in enumerate(distractors) if _title(t) != title and row.state not in t]
        own = cnt(row.state) + cnt(FORWARDED_OPEN) + cnt(FORWARDED_CLOSE) + 4
        if not usable or own >= hi or hi + _prompt_tokens(row, cnt) > MAX_TRAIN_TOKENS:
            continue
        for _ in range(ATTEMPTS):
            target = rng.randint(max(lo, own), hi)
            budget = target - own
            before_budget = int(budget * rng.random())
            used_docs: list[int] = []
            parts: list[list[str]] = []
            for b in (before_budget, budget - before_budget):
                i = rng.choice(usable)
                paras, used = _walk(parsed[i], b, cnt, rng)
                parts.append(paras)
                if paras:
                    used_docs.append(i)
            blocks = [*parts[0], f"{FORWARDED_OPEN}\n{row.state}\n{FORWARDED_CLOSE}", *parts[1]]
            if lo <= count("\n\n".join(blocks)) <= hi:
                break
        else:
            continue
        meta = dict(row.metadata or {})
        meta.update(orig_id=row.id, distractor_ids=[ids[i] for i in dict.fromkeys(used_docs)])
        out.append(replace(
            row, id=f"pad:{row.id}", state="\n\n".join(blocks), family=FAMILY,
            cluster_id=f"pad:{row.cluster_id or row.id}", metadata=meta,
        ))
    return out
