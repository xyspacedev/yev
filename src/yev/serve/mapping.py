"""System One questions -> our training chat rows -> wire answers.

Each question becomes one chat row built with ``yev.format`` (``SYSTEM_PROMPT`` and ``user_turn``
on a ``yev.schema.Decision``), so the prompt is byte-identical to the training and bench rows.
Options are never shuffled at serve time, as in ``yev bench``: choice options in criteria order,
score levels in scale order with keys "0".."n-1", and Noul as (yes, no) with yes = criteria.true,
the convention of the training mix and every Noul benchmark.

Non-text content (a JSON object or array as state, instructions or a description) is rendered with
``json.dumps(..., ensure_ascii=False)``, the way ``yev.bench.jevbench`` serialises JSON states.
A missing instruction becomes a default question per type; a missing description becomes the
option's key (training rows never have an empty description).
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from yev.format import LETTERS, SYSTEM_PROMPT, user_turn
from yev.schema import Decision, Option
from yev.serve.contract import ChoiceAnswer, NoulAnswer, ScoreAnswer

MAX_LETTERS = 26
assert len(LETTERS) >= MAX_LETTERS

NOUL_YES, NOUL_NO = "yes", "no"
DEFAULT_QUESTION = {
    "noul": "Is this true of the state?",
    "choice": "Which option best fits the state?",
    "score": "Which level best fits the state?",
}


class Unsupported(Exception):
    """A request the model cannot answer; ``reason`` carries a decision-index refusal marker."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def _options(q) -> list[Option]:
    if q.type == "noul":
        c = q.criteria
        yes = c.true if c is not None else None
        no = c.false if c is not None else None
        return [Option(NOUL_YES, NOUL_YES if yes is None else _text(yes)),
                Option(NOUL_NO, NOUL_NO if no is None else _text(no))]
    if q.type == "choice":
        return [Option(k, k if d is None else _text(d)) for k, d in q.criteria.items()]
    if q.type == "score":
        return [Option(str(i), _text(d)) for i, d in enumerate(q.criteria)]
    raise Unsupported(f"unsupported question type {q.type!r}")


def _check(q) -> None:
    n = len(q.criteria) if q.type in ("choice", "score") else 2
    if q.type == "choice" and n < 2:
        raise Unsupported("a choice needs at least two options")
    if n > MAX_LETTERS:
        raise Unsupported(f"too many options per choice ({n} > {MAX_LETTERS})")


def to_rows(state: Any, questions: Mapping[str, Any]) -> list[dict]:
    """One chat row per question, in request order. Validates every question before building any."""
    for q in questions.values():
        _check(q)
    state_text = _text(state)
    rows = []
    for name, q in questions.items():
        options = _options(q)
        question = DEFAULT_QUESTION[q.type] if q.instructions is None else _text(q.instructions)
        d = Decision(id=name, type=q.type, state=state_text, question=question, options=options,
                     gold=options[0].key, family="serve", source="serve", licence="serve")
        rows.append({
            "name": name,
            "type": q.type,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": user_turn(d, options)}],
            "letters": {LETTERS[i]: o.key for i, o in enumerate(options)},
        })
    return rows


def _normalise(probs: Sequence[float], n: int, name: str) -> list[float]:
    if len(probs) != n:
        raise ValueError(f"{name}: {len(probs)} probabilities for {n} letters")
    p = [float(x) for x in probs]
    total = sum(p)
    if not all(math.isfinite(x) and x >= 0 for x in p) or not total > 0:
        raise ValueError(f"{name}: invalid probabilities {p}")
    return [x / total for x in p]


def to_answers(questions: Mapping[str, Any], rows: list[dict], letter_probs: list[Sequence[float]]) -> dict:
    """Wire answers from per-row probabilities over each row's letters (in letter order)."""
    if len(rows) != len(letter_probs):
        raise ValueError(f"{len(rows)} rows but {len(letter_probs)} probability lists")
    answers = {}
    for row, probs in zip(rows, letter_probs):
        name, q = row["name"], questions[row["name"]]
        keys = list(row["letters"].values())
        p = dict(zip(keys, _normalise(probs, len(keys), name)))
        if q.type == "noul":
            answers[name] = NoulAnswer(noul=p[NOUL_YES])
        elif q.type == "choice":
            best = max(keys, key=lambda k: p[k])
            answers[name] = ChoiceAnswer(choice=best, confidence=p[best], probabilities=p)
        else:
            answers[name] = ScoreAnswer(
                score=sum(int(k) * v for k, v in p.items()),
                confidence=max(p.values()),
                legend={str(i): c for i, c in enumerate(q.criteria)},
                probabilities=p,
            )
    return answers
