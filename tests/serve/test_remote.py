"""RemoteEngine and the async app path (Plan 7 Task 2)."""

import asyncio
import json
import time

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
from pydantic import TypeAdapter  # noqa: E402

from yev.serve.app import create_app  # noqa: E402
from yev.serve.backends import EngineError  # noqa: E402
from yev.serve.contract import QuestionSpec  # noqa: E402
from yev.serve.mapping import Unsupported  # noqa: E402
from yev.serve.remote import RemoteEngine  # noqa: E402
from yev.train import readout  # noqa: E402

QS = TypeAdapter(dict[str, QuestionSpec])
CHOICE = {"type": "choice", "instructions": "Which colour?", "criteria": {"red": "warm", "blue": None, "green": "grass"}}
NOUL = {"type": "noul", "instructions": "Is it spam?", "criteria": {"true": "ads", "false": None}}


class StubBackend:
    """Scores letter i as -i (so A wins), after an optional delay; records calls and peak concurrency."""

    def __init__(self, delay=0.0, fail=None):
        self.delay, self.fail, self.calls, self.inflight, self.peak = delay, fail, [], 0, 0

    async def letter_logprobs(self, prompts, letters):
        self.calls.append((prompts, letters))
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(self.delay)
            if self.fail:
                raise EngineError(self.fail)
            return [[-float(i) for i in range(len(ls))] for ls in letters]
        finally:
            self.inflight -= 1


def remote(tok, backend, **kw):
    return RemoteEngine(tok, backend, None, **kw)


def test_remote_answer_uses_backend_scores(tok):
    b = StubBackend()
    e = remote(tok, b)
    answers, usage = asyncio.run(e.answer_async("a state", QS.validate_python({"c": CHOICE, "n": NOUL})))
    assert answers["c"].choice == "red"
    expect = readout.probs([0.0, -1.0, -2.0], 3)
    assert answers["c"].probabilities["red"] == pytest.approx(expect[0])
    assert answers["n"].noul == pytest.approx(readout.probs([0.0, -1.0], 2)[0])
    prompts, letters = b.calls[0]
    assert len(prompts) == 2 and [len(x) for x in letters] == [3, 2]
    assert letters[0] == [tok.encode(L)[0] for L in "ABC"]
    assert usage["output_tokens"] == 0 and usage["input_tokens"] == sum(len(p) for p in prompts)


def test_remote_too_long_is_422_without_engine_call(tok):
    b = StubBackend()
    e = remote(tok, b, max_len=5)
    with pytest.raises(Unsupported, match="longer than the maximum model length"):
        asyncio.run(e.answer_async("a state " * 20, QS.validate_python({"c": CHOICE})))
    assert b.calls == []


def test_remote_chat_probs(tok):
    e = remote(tok, StubBackend())
    user = json.dumps({"state": "s", "question": "q?",
                       "options": [{"label": "A", "key": "x"}, {"label": "B", "key": "y"}]})
    p = asyncio.run(e.chat_probs_async([{"role": "user", "content": user}]))
    assert set(p) == {"A", "B"} and p["A"] > p["B"]


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_engine_error_is_503(tok):
    app = create_app(remote(tok, StubBackend(fail="vllm engine at http://x: ConnectError: refused")))

    async def go():
        async with _client(app) as c:
            return await c.post("/v1/systemone", json={"state": "s", "questions": {"c": CHOICE}})

    r = asyncio.run(go())
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "engine_unavailable" and "refused" in r.json()["error"]["message"]


def test_remote_requests_overlap(tok):
    b = StubBackend(delay=0.3)
    app = create_app(remote(tok, b))

    async def go():
        async with _client(app) as c:
            t = time.perf_counter()
            rs = await asyncio.gather(*(c.post("/v1/systemone", json={"state": f"s{i}", "questions": {"c": CHOICE}})
                                        for i in range(8)))
            return rs, time.perf_counter() - t

    rs, dt = asyncio.run(go())
    assert all(r.status_code == 200 for r in rs)
    assert b.peak == 8 and dt < 1.5
