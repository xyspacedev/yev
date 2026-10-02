"""The model behind ``/v1/systemone``: calibrated letter probabilities for System One questions.

Each question is one chat row (``jeff.serve.mapping.to_rows``). All rows of a request go through
``jeff.train.infer.letter_logits`` together, the evaluated readout, with ``n_letters`` = the most
letters any row has. Each row's probabilities are then a temperature softmax over its own letters
only (``jeff.train.readout.probs``), so an answer does not depend on its batch mates.

A request is refused whole, before any forward pass, if a question is unsupported or a prompt is
longer than ``max_len`` tokens. Prompts are never truncated.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from jeff.format import LETTERS
from jeff.serve.mapping import MAX_LETTERS, Unsupported, to_answers, to_rows
from jeff.train import readout
from jeff.train.data import letter_token_ids, prompt_ids
from jeff.train.infer import letter_logits

NOT_OUR_FORMAT = ("the last message must be a user turn holding a jeff decision: a JSON object whose "
                  "\"options\" list gives each option a distinct \"label\" from A to Z")


def load_temperatures(path: str | None) -> dict[str, float]:
    """Per-type temperatures from a calibration.json; a type that is missing reads as T = 1.0."""
    if not path:
        return {}
    return {k: float(v) for k, v in json.loads(Path(path).read_text())["temperatures"].items()}


def too_long(n: int, max_len: int) -> Unsupported:
    # "longer than the maximum model length" is a decision-index refusal marker (WIRE_FORMAT.md).
    return Unsupported(f"state too long: longer than the maximum model length ({n} > {max_len} tokens)")


def _message(m: Any) -> dict:
    if isinstance(m, Mapping):
        return {"role": m["role"], "content": m["content"]}
    return {"role": m.role, "content": m.content}


def _labels(messages: list[dict]) -> list[str]:
    """The option labels of the final user turn, which must be in our decision format."""
    if not messages or messages[-1]["role"] != "user" or not isinstance(messages[-1]["content"], str):
        raise Unsupported(NOT_OUR_FORMAT)
    try:
        body = json.loads(messages[-1]["content"])
        labels = [o["label"] for o in body["options"]]
    except (ValueError, TypeError, KeyError):
        raise Unsupported(NOT_OUR_FORMAT) from None
    if not labels or len(set(labels)) != len(labels) or not all(isinstance(L, str) and len(L) == 1
                                                               and L in LETTERS[:MAX_LETTERS] for L in labels):
        raise Unsupported(NOT_OUR_FORMAT)
    return labels


class Engine:
    def __init__(self, model_dir: str, base: str | None, calibration: str | None,
                 max_len: int = 16384, batch_tokens: int = 16384, *, model=None, tokenizer=None):
        if (model is None) != (tokenizer is None):
            raise ValueError("pass both model and tokenizer, or neither")
        if model is None:
            from jeff.train.infer import load
            tokenizer, model = load(model_dir, base)
        # Every serveable letter must be one distinct token, or the readout is wrong: fail at load.
        letter_token_ids(tokenizer, MAX_LETTERS)
        self.model, self.tokenizer = model, tokenizer
        self.temperatures = load_temperatures(calibration)
        self.max_len, self.batch_tokens = max_len, batch_tokens
        self._lock = threading.Lock()  # one forward pass at a time on the one model

    def temperature(self, type_: str) -> float:
        return self.temperatures.get(type_, 1.0)

    def n_tokens(self, messages) -> int:
        return len(prompt_ids(self.tokenizer, [_message(m) for m in messages]))

    def _checked_lengths(self, rows: list[dict]) -> list[int]:
        lengths = [self.n_tokens(r["messages"]) for r in rows]
        if max(lengths) > self.max_len:
            raise too_long(max(lengths), self.max_len)
        return lengths

    def _logits(self, rows: list[dict], n_letters: int) -> list[list[float]]:
        with self._lock:
            return letter_logits(self.model, self.tokenizer, rows, max_len=self.max_len,
                                 batch_tokens=self.batch_tokens, n_letters=n_letters)

    def answer(self, state: Any, questions: Mapping[str, Any]) -> tuple[dict, dict]:
        """Wire answers keyed by question name, and the response's ``usage``."""
        rows = to_rows(state, questions)
        lengths = self._checked_lengths(rows)
        logits = self._logits(rows, max(len(r["letters"]) for r in rows))
        probs = [readout.probs(z, len(r["letters"]), self.temperature(r["type"])) for r, z in zip(rows, logits)]
        return to_answers(questions, rows, probs), {"input_tokens": sum(lengths), "output_tokens": 0}

    def chat(self, messages) -> str:
        """The most probable letter among the options of the final user turn (our decision format)."""
        msgs = [_message(m) for m in messages]
        labels = _labels(msgs)
        rows = [{"messages": msgs}]
        self._checked_lengths(rows)
        z = self._logits(rows, max(LETTERS.index(L) for L in labels) + 1)[0]
        return max(labels, key=lambda L: z[LETTERS.index(L)])
