"""``/v1/systemone`` answered by a vLLM or SGLang sidecar (``yev serve --backend vllm|sglang``).

Same rows, token ids, length check, calibration and answer shape as ``Engine``; only the letter scores come from
the sidecar (``yev.serve.backends``). Async, so concurrent requests reach the engine together.
"""

from __future__ import annotations

from typing import Any, Mapping

from yev.format import LETTERS
from yev.serve.engine import Engine, _infer_type, _labels, _message, load_temperatures
from yev.serve.mapping import MAX_LETTERS, to_answers, to_rows
from yev.train import readout
from yev.train.data import letter_token_ids


class RemoteEngine(Engine):
    def __init__(self, tokenizer, backend, calibration: str | None, max_len: int = 16384):
        # No model: Engine.__init__ would load one. Only the helpers that need the tokenizer are reused.
        self.tokenizer, self.backend = tokenizer, backend
        self.letter_ids = letter_token_ids(tokenizer, MAX_LETTERS)
        self.temperatures = load_temperatures(calibration)
        self.max_len = max_len

    async def answer_async(self, state: Any, questions: Mapping[str, Any]) -> tuple[dict, dict]:
        rows = to_rows(state, questions)
        ids = self._checked_ids(rows)
        scores = await self.backend.letter_logprobs(ids, [self.letter_ids[: len(r["letters"])] for r in rows])
        probs = [readout.probs(z, len(r["letters"]), self.temperature(r["type"])) for r, z in zip(rows, scores)]
        return to_answers(questions, rows, probs), {"input_tokens": sum(len(x) for x in ids), "output_tokens": 0}

    async def chat_probs_async(self, messages) -> dict[str, float]:
        msgs = [_message(m) for m in messages]
        labels = _labels(msgs)
        ids = self._checked_ids([{"messages": msgs}])
        idx = [LETTERS.index(L) for L in labels]
        z = (await self.backend.letter_logprobs(ids, [[self.letter_ids[i] for i in idx]]))[0]
        p = readout.probs(z, len(labels), self.temperature(_infer_type(msgs) or ""))
        return dict(zip(labels, p))
