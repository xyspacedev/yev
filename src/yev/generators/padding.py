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
import logging
import random
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable

from yev.schema import Decision

FAMILY = "long_padding"
FORWARDED_OPEN = "--- Forwarded item ---"
FORWARDED_CLOSE = "--- End of forwarded item ---"
EVAL_SPLITS = ("dev", "calibration")
MAX_TRAIN_TOKENS = 12288  # target_tokens bounds the padded state; the prompt must fit on top of hi
TOKENIZER_NAME = "Qwen/Qwen3.5-4B-Base"
ATTEMPTS = 30
# many_options golds come from a keyword rule over the state; invoice text ("invoice number", "amount due",
# "remit-to") could contradict it. Exact family match, so many_options_intent stays allowed.
PAD_EXCLUDED_FAMILIES = {"many_options"}
QUESTION_PREFIX = "About the forwarded item: "
HEURISTIC_FACTOR = 2.0
RULE_SENSITIVE_FAMILIES = ("returns_v2", "date_number_drills", "returns_policy", "action_review")
# Paragraphs that state rules or policy: handbook/policy sections ("A. Payment terms"), holiday lists,
# payment-terms clauses and values, date-format notes. Correspondence, tables and ledgers otherwise stay.
RULE_RE = re.compile(
    r"(?im)^[A-Z]\d?\.\s+\S|\bhandbook\b|\bholidays?\b|\bpayment terms\b|\bNet \d+\b|\bpolicy\b"
    r"|dates are written|early-payment discount|\bdue date\b|\bterms\b.*\bapply\b"
)
log = logging.getLogger(__name__)


class PaddingError(ValueError):
    """The request would break a padding rule."""


class TokenizerUnavailable(PaddingError):
    """The real tokenizer cannot be loaded; pass allow_heuristic to use the len/4 x 2.0 estimate."""


def heuristic_tokens(text: str) -> int:
    """len/4 undercounts table-heavy text (1.57x measured, more in places); factor 2.0 is the fallback."""
    return int(len(text) / 4 * HEURISTIC_FACTOR) + 1


def load_token_counter(name: str = TOKENIZER_NAME, allow_heuristic: bool = False,
                       cache_root: Path | None = None) -> Callable[[str], int]:
    """The real tokenizer from the local HF cache; the heuristic only when explicitly allowed."""
    root = cache_root or Path.home() / ".cache/huggingface/hub"
    snaps = sorted((root / ("models--" + name.replace("/", "--")) / "snapshots").glob("*/tokenizer.json"))
    problem = None
    if not snaps:
        problem = f"tokenizer {name} not found under {root}"
    else:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(snaps[-1].parent))
            return lambda text: len(tok.encode(text, add_special_tokens=False))
        except ImportError as e:
            problem = f"transformers is not installed ({e}); install the train extra"
        except OSError as e:
            problem = f"tokenizer {name} failed to load: {e}"
    if not allow_heuristic:
        raise TokenizerUnavailable(f"{problem}. Pass --allow-heuristic to estimate with len/4 x {HEURISTIC_FACTOR}.")
    log.warning("%s; using len/4 x %s token estimate", problem, HEURISTIC_FACTOR)
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
    """Contiguous paragraphs from a random start, never wrapping past the end, within budget."""
    if not paras or budget <= 0:
        return [], []
    start = rng.randrange(len(paras))
    out: list[str] = []
    used = 0
    for p in paras[start:]:
        n = count(p) + 2
        if used + n > budget:
            break
        out.append(p)
        used += n
    return out, [used]


def pad(rows: list[Decision], distractors: list[str], target_tokens: tuple[int, int], seed,
        distractor_ids: list[str] | None = None, count: Callable[[str], int] | None = None,
        allow_heuristic: bool = False, skipped: list[tuple[str, str]] | None = None) -> list[Decision]:
    """`skipped`, when given, receives (row id, reason) with reason too_long, no_distractor, same_title
    length_unmet or excluded_family."""
    lo, hi = target_tokens
    if not 0 < lo <= hi <= MAX_TRAIN_TOKENS:
        raise PaddingError(f"target_tokens {target_tokens} must satisfy 0 < lo <= hi <= {MAX_TRAIN_TOKENS}")
    done = [r.id for r in rows if r.family == FAMILY]
    if done:
        raise PaddingError(f"rows are already long_padding: {done[:3]}")
    bad = [r.id for r in rows if r.split in EVAL_SPLITS]
    if bad:
        raise PaddingError(f"refusing to pad dev/calibration rows: {bad[:3]}")
    if not distractors:
        raise PaddingError("no distractor documents")
    ids = distractor_ids or [doc_id(t) for t in distractors]
    if len(ids) != len(distractors):
        raise PaddingError("distractor_ids length differs from distractors")
    count = count or load_token_counter(allow_heuristic=allow_heuristic)
    cache: dict[str, int] = {}

    def cnt(t: str) -> int:
        if t not in cache:
            cache[t] = count(t)
        return cache[t]

    parsed = [paragraphs(t) for t in distractors]
    parsed_clean = [[p for p in ps if not RULE_RE.search(p)] for ps in parsed]
    skipped = skipped if skipped is not None else []
    out: list[Decision] = []
    for row in rows:
        rng = random.Random(f"{seed}:{row.id}")
        if row.family in PAD_EXCLUDED_FAMILIES:
            skipped.append((row.id, "excluded_family"))
            continue
        title = _title(row.state)
        usable = [i for i, t in enumerate(distractors) if _title(t) != title and row.state not in t]
        own = cnt(row.state) + cnt(FORWARDED_OPEN) + cnt(FORWARDED_CLOSE) + 4
        if own >= hi or hi + _prompt_tokens(row, cnt) > MAX_TRAIN_TOKENS:
            skipped.append((row.id, "too_long"))
            continue
        if not usable:
            same = any(_title(t) == title for t in distractors)
            skipped.append((row.id, "same_title" if same else "no_distractor"))
            continue
        pool = parsed_clean if row.family in RULE_SENSITIVE_FAMILIES else parsed
        for _ in range(ATTEMPTS):
            target = rng.randint(max(lo, own), hi)
            budget = target - own
            before_budget = int(budget * rng.random())
            used_docs: list[int] = []
            parts: list[list[str]] = []
            for b in (before_budget, budget - before_budget):
                i = rng.choice(usable)
                paras, used = _walk(pool[i], b, cnt, rng)
                parts.append(paras)
                if paras:
                    used_docs.append(i)
            blocks = [*parts[0], f"{FORWARDED_OPEN}\n{row.state}\n{FORWARDED_CLOSE}", *parts[1]]
            if lo <= count("\n\n".join(blocks)) <= hi:
                break
        else:
            skipped.append((row.id, "length_unmet"))
            continue
        meta = dict(row.metadata or {})
        meta.update(orig_id=row.id, question_prefixed=True, distractor_ids=[ids[i] for i in dict.fromkeys(used_docs)])
        out.append(replace(
            row, id=f"pad:{row.id}", question=QUESTION_PREFIX + row.question, state="\n\n".join(blocks), family=FAMILY,
            cluster_id=f"pad:{row.cluster_id or row.id}", metadata=meta,
        ))
    return out
