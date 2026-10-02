"""The model behind ``/v1/systemone``: calibrated letter probabilities for System One questions.

Each question is one chat row (``yev.serve.mapping.to_rows``). All rows of a request go through
``yev.train.infer.letter_logits`` together, the evaluated readout, with ``n_letters`` = the most
letters any row has. Each row's probabilities are then a temperature softmax over its own letters
only (``yev.train.readout.probs``), so an answer does not depend on its batch mates.

A request with at least two questions whose prompts share at least ``min_prefix_tokens`` leading
tokens (the state) reads that prefix once and continues each question from a copy of its KV cache
(``yev.serve.prefix``); ``prefix_cache=False`` or a shorter prefix uses the exact path above.

A request is refused whole, before any forward pass, if a question is unsupported or a prompt is
longer than ``max_len`` tokens. Prompts are never truncated.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Mapping

from yev.format import LETTERS
from yev.serve.mapping import MAX_LETTERS, NOUL_NO, NOUL_YES, Unsupported, to_answers, to_rows
from yev.serve.prefix import common_prefix_len, prefix_letter_logits
from yev.train import readout
from yev.train.data import letter_token_ids, prompt_ids
from yev.train.infer import letter_logits

NOT_OUR_FORMAT = ("the last message must be a user turn holding a yev decision: a JSON object whose "
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


def _infer_type(messages: list[dict]) -> str | None:
    """The question type behind a chat turn, when its option keys reveal it (the mapping's conventions)."""
    try:
        keys = [o["key"] for o in json.loads(messages[-1]["content"])["options"]]
    except (ValueError, TypeError, KeyError):
        return None
    if keys == [NOUL_YES, NOUL_NO]:
        return "noul"
    if keys == [str(i) for i in range(len(keys))]:
        return "score"
    return "choice"


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
                 max_len: int = 16384, batch_tokens: int = 16384, *, model=None, tokenizer=None,
                 prefix_cache: bool = True, min_prefix_tokens: int = 256):
        if (model is None) != (tokenizer is None):
            raise ValueError("pass both model and tokenizer, or neither")
        if model is None:
            from yev.train.infer import load
            tokenizer, model = load(model_dir, base)
        # Every serveable letter must be one distinct token, or the readout is wrong: fail at load.
        letter_token_ids(tokenizer, MAX_LETTERS)
        self.model, self.tokenizer = model, tokenizer
        self.temperatures = load_temperatures(calibration)
        self.max_len, self.batch_tokens = max_len, batch_tokens
        self.prefix_cache, self.min_prefix_tokens = prefix_cache, min_prefix_tokens
        self._lock = threading.Lock()  # one forward pass at a time on the one model

    def temperature(self, type_: str) -> float:
        return self.temperatures.get(type_, 1.0)

    def n_tokens(self, messages) -> int:
        return len(prompt_ids(self.tokenizer, [_message(m) for m in messages]))

    def _checked_ids(self, rows: list[dict]) -> list[list[int]]:
        ids = [prompt_ids(self.tokenizer, r["messages"]) for r in rows]
        longest = max(len(x) for x in ids)
        if longest > self.max_len:
            raise too_long(longest, self.max_len)
        return ids

    def _checked_lengths(self, rows: list[dict]) -> list[int]:
        return [len(x) for x in self._checked_ids(rows)]

    def _logits(self, rows: list[dict], n_letters: int) -> list[list[float]]:
        with self._lock:
            return letter_logits(self.model, self.tokenizer, rows, max_len=self.max_len,
                                 batch_tokens=self.batch_tokens, n_letters=n_letters)

    def _request_logits(self, rows: list[dict], ids: list[list[int]], n_letters: int) -> list[list[float]]:
        """The prefix-cached readout when it applies (2+ questions, a long shared prefix), else the exact one."""
        P = common_prefix_len(ids) if self.prefix_cache and len(rows) >= 2 else 0
        if P < max(self.min_prefix_tokens, 1):
            return self._logits(rows, n_letters)
        with self._lock:
            return prefix_letter_logits(self.model, ids[0][:P], [x[P:] for x in ids],
                                        letter_token_ids(self.tokenizer, n_letters), self.batch_tokens,
                                        pad_id=getattr(self.tokenizer, "pad_token_id", 0) or 0)

    def answer(self, state: Any, questions: Mapping[str, Any]) -> tuple[dict, dict]:
        """Wire answers keyed by question name, and the response's ``usage``."""
        rows = to_rows(state, questions)
        ids = self._checked_ids(rows)
        lengths = [len(x) for x in ids]
        logits = self._request_logits(rows, ids, max(len(r["letters"]) for r in rows))
        probs = [readout.probs(z, len(r["letters"]), self.temperature(r["type"])) for r, z in zip(rows, logits)]
        return to_answers(questions, rows, probs), {"input_tokens": sum(lengths), "output_tokens": 0}

    def chat_probs(self, messages) -> dict[str, float]:
        """Probability of each option letter of the final user turn (our decision format).

        The calibrated temperature is applied for the question's type when the option keys reveal it
        (["yes", "no"] is noul, "0".."n-1" is score, anything else is choice); a turn that does not
        parse to options uses T = 1. Argmax is the same either way.
        """
        msgs = [_message(m) for m in messages]
        labels = _labels(msgs)
        rows = [{"messages": msgs}]
        self._checked_lengths(rows)
        z = self._logits(rows, max(LETTERS.index(L) for L in labels) + 1)[0]
        t = self.temperature(_infer_type(msgs) or "")
        p = readout.probs([z[LETTERS.index(L)] for L in labels], len(labels), t)
        return dict(zip(labels, p))

    def chat(self, messages) -> str:
        """The most probable letter among the options of the final user turn (our decision format)."""
        p = self.chat_probs(messages)
        return max(p, key=p.get)
