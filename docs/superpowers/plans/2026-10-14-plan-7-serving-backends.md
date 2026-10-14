# Plan 7: vLLM and SGLang serving backends — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve yev through vLLM and SGLang sidecars behind the unchanged `/v1/systemone` API, and measure parity,
latency and throughput against the transformers engine.

**Architecture:** `yev serve --backend vllm|sglang --engine-url …` runs a `RemoteEngine` that builds the same prompt
token ids and calibrated softmax as today, and gets each question's letter log-probs from the sidecar over async
HTTP. The sidecars run merged weights in their own virtualenvs on the box. Two scripts measure parity on Stage 1c
dev rows and speed on a latency grid plus a concurrent replay.

**Tech Stack:** Python 3.13, FastAPI, httpx (async), transformers + peft (merge), vLLM and SGLang (box only, own
venvs), pytest with `httpx.MockTransport` / `ASGITransport`.

**Spec:** `docs/superpowers/specs/2026-10-14-serving-backends-design.md`

## Global Constraints

- The wire contract (`src/yev/serve/contract.py`) and the transformers `Engine` behaviour do not change; all existing
  `tests/serve` tests keep passing.
- Prompts go to sidecars as token ids, never text.
- Calibration: `readout.probs(scores, n, T)` on the backend's per-letter scores; the existing calibration JSON is
  reused unchanged.
- Engine unreachable, timeout, non-200, or fewer letter scores than letters → HTTP 503 with the engine's message.
  Never a silent fallback to another backend.
- Parity bar (dev rows only): argmax agreement ≥ 99.5% and mean |dp| ≤ 0.01. A backend below the bar keeps its speed
  numbers, labelled "not at parity".
- Latency grid: state 300 / 2k / 8k tokens × 1 / 4 / 16 questions; 2 warm-up + 10 timed per cell; p50, p95.
- Throughput: 500 dev requests (rows regrouped by state), concurrency 1, 4, 16, 64; requests/s, questions/s,
  p50/p95, peak GPU memory.
- Fairness: same merged weights, same L40S, bf16, no quantisation, no speculative decoding, engine defaults plus
  prefix caching; versions and flags recorded.
- Sidecars: port 8100, `~/venv-vllm`, `~/venv-sglang`, newest stable release at setup time, `pip freeze` recorded.
- Merged weights stay local (`~/models/yev1c-merged`); never uploaded. Host, key and instance id never in the repo
  (`YEV_TRAIN_HOST`, `YEV_TRAIN_KEY`).
- No engine is forked or patched. If an engine can't run Qwen3.5, the report says why.
- Outputs: JSON in `data/bench_results/backends/` (git-ignored); write-up in `docs/serving-backends.md`.

## Rulings (deviations from the spec)

1. **The transformers `Engine` is the transformers backend as is.** The spec moves it "behind the interface with no
   behaviour change". Wrapping it would refactor the prefix-cache path for no measurable gain and risk the
   evaluated readout, so the `Backend` protocol is implemented by the two sidecar clients only, and `RemoteEngine`
   reuses `Engine`'s helpers. Cost if wrong: a later refactor of one class.
2. **Parity script is `scripts/backend_parity.py`**, not `serve_parity.py --backend`. `serve_parity.py` compares
   cached and exact paths on request files; dev-row parity is a different job (token-id rows, two phases because the
   sidecar and the reference model can't both hold the GPU comfortably).
3. **Backend per-question letters.** `letters[i]` is the first `len(row letters)` ids of
   `letter_token_ids(tokenizer, MAX_LETTERS)`, the same ids the transformers readout uses.
4. **mean |dp|** is the mean over rows of `max_k |p_a[k] − p_b[k]|`.

## Review Focus

1. **A vLLM response whose top-logprobs keys are token strings, not `token_id:N`** (server started without
   `return_tokens_as_token_ids` support) must be a 503 naming the problem, not a KeyError 500 —
   `test_vllm_rejects_string_keys`.
2. **A letter missing from the engine's log-probs** (raw-logprob mode, top-k cut it) must be a 503, never a
   softmax over fewer letters — `test_vllm_missing_letter_is_engine_error`, `test_sglang_missing_letter_is_engine_error`.
3. **Concurrent requests must overlap**, not serialise on a lock or the threadpool —
   `test_remote_requests_overlap`.
4. **A prompt longer than `--max-len`** must be refused with today's 422 marker before any sidecar call —
   `test_remote_too_long_is_422_without_engine_call`.
5. **`/v1/chat/completions` on a remote backend** must return the same letter and probabilities shape as on
   transformers — `test_remote_chat_probs`.

---

### Task 1: Sidecar clients (`backends.py`)

**Files:**
- Create: `src/yev/serve/backends.py`
- Test: `tests/serve/test_backends.py`

**Interfaces:**
- Produces:
  - `class EngineError(Exception)` — anything that makes a sidecar answer unusable.
  - `class VLLMBackend(url: str, client: httpx.AsyncClient | None = None, model: str = "yev")` and
    `class SGLangBackend(url: str, client: httpx.AsyncClient | None = None)`, each with
    `async def letter_logprobs(self, prompts: list[list[int]], letters: list[list[int]]) -> list[list[float]]`
    — one score per letter, in letter order; all questions sent concurrently.
  - `def make_backend(name: str, url: str) -> VLLMBackend | SGLangBackend` (`"vllm"` / `"sglang"`, else ValueError).

- [ ] **Step 1: Write the failing tests**

```python
"""Sidecar clients: request shape, log-prob parsing and errors (Plan 7 Task 1)."""

import asyncio
import json

import pytest

httpx = pytest.importorskip("httpx")
from yev.serve.backends import EngineError, SGLangBackend, VLLMBackend, make_backend  # noqa: E402


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://engine")


def run(coro):
    return asyncio.run(coro)


def vllm_ok(request):
    body = json.loads(request.content)
    top = {f"token_id:{t}": -0.5 * (i + 1) for i, t in enumerate(body["allowed_token_ids"])}
    return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [top]}}]})


def test_vllm_request_shape_and_order():
    seen = []

    def handler(request):
        seen.append((request.url.path, json.loads(request.content)))
        return vllm_ok(request)

    b = VLLMBackend("http://engine", client(handler))
    out = run(b.letter_logprobs([[1, 2, 3], [1, 2, 4]], [[10, 11], [10, 11, 12]]))
    assert out == [[-0.5, -1.0], [-0.5, -1.0, -1.5]]
    path, body = seen[0]
    assert path == "/v1/completions"
    assert body["prompt"] == [1, 2, 3] and body["max_tokens"] == 1 and body["temperature"] == 0
    assert body["allowed_token_ids"] == [10, 11] and body["logprobs"] == 2
    assert body["return_tokens_as_token_ids"] is True and body["model"] == "yev"


def test_vllm_missing_letter_is_engine_error():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [{"token_id:10": -0.1}]}}]})

    with pytest.raises(EngineError, match="11"):
        run(VLLMBackend("http://engine", client(handler)).letter_logprobs([[1]], [[10, 11]]))


def test_vllm_rejects_string_keys():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"logprobs": {"top_logprobs": [{"A": -0.1, "B": -2.0}]}}]})

    with pytest.raises(EngineError, match="token_id"):
        run(VLLMBackend("http://engine", client(handler)).letter_logprobs([[1]], [[10, 11]]))


def test_sglang_request_shape_and_parse():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        entries = [[-0.25 * (i + 1), t, None] for i, t in enumerate(reversed(body["token_ids_logprob"]))]
        return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [entries]}})

    out = run(SGLangBackend("http://engine", client(handler)).letter_logprobs([[5, 6]], [[10, 11, 12]]))
    assert out == [[-0.75, -0.5, -0.25]]  # parsed by id, returned in letter order
    path, body = seen[0]
    assert path == "/generate" and body["input_ids"] == [5, 6]
    assert body["sampling_params"] == {"max_new_tokens": 1, "temperature": 0}
    assert body["return_logprob"] is True and body["token_ids_logprob"] == [10, 11, 12]


def test_sglang_missing_letter_is_engine_error():
    def handler(request):
        return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [[[-0.1, 10, None]]]}})

    with pytest.raises(EngineError, match="11"):
        run(SGLangBackend("http://engine", client(handler)).letter_logprobs([[1]], [[10, 11]]))


@pytest.mark.parametrize("make", [lambda c: VLLMBackend("http://engine", c), lambda c: SGLangBackend("http://engine", c)])
def test_non_200_and_unreachable_are_engine_errors(make):
    with pytest.raises(EngineError, match="503"):
        run(make(client(lambda r: httpx.Response(503, text="overloaded"))).letter_logprobs([[1]], [[10]]))

    def down(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(EngineError, match="refused"):
        run(make(client(down)).letter_logprobs([[1]], [[10]]))


def test_make_backend():
    assert isinstance(make_backend("vllm", "http://x"), VLLMBackend)
    assert isinstance(make_backend("sglang", "http://x"), SGLangBackend)
    with pytest.raises(ValueError):
        make_backend("tgi", "http://x")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra serve pytest tests/serve/test_backends.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'yev.serve.backends'`

- [ ] **Step 3: Implement**

```python
"""Letter log-probs from vLLM or SGLang sidecars, for ``yev serve --backend vllm|sglang``.

Each question is one generation call of one token, restricted to (vLLM) or scored on (SGLang) its option letters.
Prompts are token ids, so the sidecar sees exactly the prompt the transformers readout scores. The returned scores
are log-probs, which differ from raw logits by a per-question constant, so ``readout.probs`` gives the same
calibrated probabilities. All questions of a request go out together so the engine can share the state's prefix.
"""

from __future__ import annotations

import asyncio

import httpx

TIMEOUT_S = 600.0


class EngineError(Exception):
    """The sidecar engine is unreachable or its answer can't be used; served as a 503."""


class _Backend:
    name = "engine"

    def __init__(self, url: str, client: httpx.AsyncClient | None = None):
        self.url = url.rstrip("/")
        self.client = client or httpx.AsyncClient(
            timeout=TIMEOUT_S, limits=httpx.Limits(max_connections=512, max_keepalive_connections=512))

    async def _post(self, path: str, body: dict) -> dict:
        try:
            r = await self.client.post(f"{self.url}{path}", json=body)
        except httpx.HTTPError as exc:
            raise EngineError(f"{self.name} engine at {self.url}: {type(exc).__name__}: {exc}") from exc
        if r.status_code != 200:
            raise EngineError(f"{self.name} engine at {self.url} returned {r.status_code}: {r.text[:500]}")
        return r.json()

    async def letter_logprobs(self, prompts: list[list[int]], letters: list[list[int]]) -> list[list[float]]:
        if len(prompts) != len(letters):
            raise ValueError(f"{len(prompts)} prompts but {len(letters)} letter lists")
        return list(await asyncio.gather(*(self._one(p, ls) for p, ls in zip(prompts, letters))))

    def _pick(self, by_id: dict[int, float], letters: list[int]) -> list[float]:
        missing = [t for t in letters if t not in by_id]
        if missing:
            raise EngineError(f"{self.name} engine returned no log-prob for letter token ids {missing}")
        return [float(by_id[t]) for t in letters]


class VLLMBackend(_Backend):
    name = "vllm"

    def __init__(self, url: str, client: httpx.AsyncClient | None = None, model: str = "yev"):
        super().__init__(url, client)
        self.model = model

    async def _one(self, prompt: list[int], letters: list[int]) -> list[float]:
        body = {"model": self.model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
                "allowed_token_ids": letters, "logprobs": len(letters), "return_tokens_as_token_ids": True}
        data = await self._post("/v1/completions", body)
        try:
            top = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise EngineError(f"vllm engine returned no top_logprobs: {str(data)[:300]}") from exc
        by_id = {}
        for key, lp in top.items():
            if not key.startswith("token_id:"):
                raise EngineError(f"vllm top_logprobs key {key!r} is not 'token_id:N' "
                                  "(the server must honour return_tokens_as_token_ids)")
            by_id[int(key.split(":", 1)[1])] = lp
        return self._pick(by_id, letters)


class SGLangBackend(_Backend):
    name = "sglang"

    async def _one(self, prompt: list[int], letters: list[int]) -> list[float]:
        body = {"input_ids": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "return_logprob": True, "token_ids_logprob": letters}
        data = await self._post("/generate", body)
        try:
            entries = data["meta_info"]["output_token_ids_logprobs"][0]
            by_id = {int(e[1]): e[0] for e in entries}
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EngineError(f"sglang engine returned no output_token_ids_logprobs: {str(data)[:300]}") from exc
        return self._pick(by_id, letters)


def make_backend(name: str, url: str) -> VLLMBackend | SGLangBackend:
    if name == "vllm":
        return VLLMBackend(url)
    if name == "sglang":
        return SGLangBackend(url)
    raise ValueError(f"unknown backend {name!r} (vllm or sglang)")
```

Add `"httpx>=0.27"` to the `serve` extra in `pyproject.toml`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --extra serve pytest tests/serve/test_backends.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/yev/serve/backends.py tests/serve/test_backends.py pyproject.toml uv.lock
git commit -m "feat(serve): vLLM and SGLang sidecar clients for letter log-probs"
```

---

### Task 2: `RemoteEngine`, async app path, 503, CLI

**Files:**
- Create: `src/yev/serve/remote.py`
- Modify: `src/yev/serve/app.py` (async handlers when the engine is remote; `EngineError` → 503)
- Modify: `src/yev/cli.py` (`serve --backend`, `--engine-url`, `--tokenizer`; `cmd_serve`)
- Modify: `docs/serving.md` (a "Backends" paragraph pointing to `docs/serving-backends.md`)
- Test: `tests/serve/test_remote.py`

**Interfaces:**
- Consumes: `EngineError`, `make_backend` (Task 1); `Engine._checked_ids`, `Engine.temperature`, `Engine.n_tokens`,
  `_labels`, `_infer_type`, `_message` from `yev.serve.engine`.
- Produces: `class RemoteEngine(Engine)` with `__init__(self, tokenizer, backend, calibration: str | None,
  max_len: int = 16384)`, `async answer_async(state, questions) -> tuple[dict, dict]`,
  `async chat_probs_async(messages) -> dict[str, float]`; `create_app` serves it with `async def` handlers.

- [ ] **Step 1: Write the failing tests**

```python
"""RemoteEngine and the async app path (Plan 7 Task 2)."""

import asyncio
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
    import json
    e = remote(tok, StubBackend())
    user = json.dumps({"state": "s", "question": "q?", "options": [{"label": "A", "key": "x"}, {"label": "B", "key": "y"}]})
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --extra serve --extra train pytest tests/serve/test_remote.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'yev.serve.remote'`

- [ ] **Step 3: Implement `remote.py`**

```python
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
```

- [ ] **Step 4: Async handlers and the 503 in `app.py`**

In `create_app`, add after the `_unsupported` handler:

```python
    @app.exception_handler(EngineError)
    def _engine_down(request: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse({"error": {"type": "engine_unavailable", "message": str(exc)}}, status_code=503)
```

Replace the two POST endpoints with a branch on the engine kind (the response building moves into two helpers so
both branches share it):

```python
    def _systemone_response(answers, usage) -> SystemOneResponse:
        return SystemOneResponse.model_validate({"model": model_name, "answers": answers, "usage": usage})

    def _chat_response(req: ChatRequest, probs: dict) -> ChatResponse:
        letter = max(probs, key=probs.get)
        out = ChatResponse.build(model=model_name, content=letter,
                                 prompt_tokens=engine.n_tokens(req.messages), completion_tokens=1)
        if req.logprobs:
            top = [{"token": L, "logprob": math.log(max(p, 1e-300)), "bytes": list(L.encode())}
                   for L, p in sorted(probs.items(), key=lambda kv: -kv[1])]
            out.choices[0].logprobs = {"content": [{**top[0], "top_logprobs": top}]}
        return out

    if hasattr(engine, "answer_async"):  # a sidecar backend: async, so concurrent requests overlap
        @app.post("/v1/systemone", response_model=SystemOneResponse)
        async def systemone(req: SystemOneRequest) -> SystemOneResponse:
            return _systemone_response(*await engine.answer_async(req.state, req.questions))

        @app.post("/v1/chat/completions", response_model=ChatResponse, response_model_exclude_none=True)
        async def chat(req: ChatRequest) -> ChatResponse:
            return _chat_response(req, await engine.chat_probs_async(req.messages))
    else:
        @app.post("/v1/systemone", response_model=SystemOneResponse)
        def systemone(req: SystemOneRequest) -> SystemOneResponse:
            return _systemone_response(*engine.answer(req.state, req.questions))

        @app.post("/v1/chat/completions", response_model=ChatResponse, response_model_exclude_none=True)
        def chat(req: ChatRequest) -> ChatResponse:
            return _chat_response(req, engine.chat_probs(req.messages))
```

Import `from yev.serve.backends import EngineError` (it only needs httpx, which the serve extra now carries). Update
the module docstring: sync endpoints for the in-process engine, async for sidecars.

- [ ] **Step 5: CLI**

In the `serve` parser add:

```python
    p.add_argument("--backend", choices=["transformers", "vllm", "sglang"], default="transformers",
                   help="transformers runs the model in process; vllm/sglang ask a sidecar at --engine-url")
    p.add_argument("--engine-url", default="http://127.0.0.1:8100", help="the vllm/sglang sidecar")
```

and make `--model` say "LoRA adapter dir or full model dir (with a sidecar: the merged dir, for the tokenizer)".
`cmd_serve`:

```python
def cmd_serve(args) -> int:
    import uvicorn

    from yev.serve.app import create_app
    if args.backend == "transformers":
        from yev.serve.engine import Engine
        engine = Engine(args.model, args.base, args.calibration, max_len=args.max_len,
                        batch_tokens=args.batch_tokens, prefix_cache=args.prefix_cache,
                        min_prefix_tokens=args.min_prefix_tokens)
    else:
        from transformers import AutoTokenizer

        from yev.serve.backends import make_backend
        from yev.serve.remote import RemoteEngine
        tok = AutoTokenizer.from_pretrained(args.base or args.model)
        engine = RemoteEngine(tok, make_backend(args.backend, args.engine_url), args.calibration,
                              max_len=args.max_len)
    uvicorn.run(create_app(engine, model_name=args.model_name), host=args.host, port=args.port)
    return 0
```

- [ ] **Step 6: Run tests**

Run: `uv run --extra serve --extra train pytest tests/serve -v`
Expected: PASS (all existing serve tests plus 5 new)

- [ ] **Step 7: Commit**

```bash
git add src/yev/serve/remote.py src/yev/serve/app.py src/yev/cli.py docs/serving.md tests/serve/test_remote.py
git commit -m "feat(serve): --backend vllm|sglang via RemoteEngine, async handlers and 503 on engine errors"
```

---

### Task 3: `yev export-merged`

**Files:**
- Create: `src/yev/serve/merge.py`
- Modify: `src/yev/cli.py` (`export-merged` subcommand)
- Test: `tests/serve/test_merge.py`

**Interfaces:**
- Produces: `export_merged(adapter_dir: str, base: str, out: str, save_tokenizer: bool = True) -> None` — writes
  bf16 (CUDA) or fp32 (CPU) merged weights, `config.json`, and the base's tokenizer files to `out`.

- [ ] **Step 1: Write the failing test**

```python
"""LoRA merge for the sidecars (Plan 7 Task 3)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import AutoModelForCausalLM  # noqa: E402

from yev.serve.merge import export_merged  # noqa: E402


def test_merged_matches_adapter(tmp_path, tiny_model):
    from peft import LoraConfig, get_peft_model
    base_dir, adapter_dir, out = tmp_path / "base", tmp_path / "adapter", tmp_path / "merged"
    tiny_model.save_pretrained(base_dir)
    peft = get_peft_model(tiny_model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                                                 init_lora_weights=False))
    peft.save_pretrained(adapter_dir)
    x = torch.tensor([[3, 5, 7, 9]])
    with torch.no_grad():
        want = peft(input_ids=x).logits
    export_merged(str(adapter_dir), str(base_dir), str(out), save_tokenizer=False)
    merged = AutoModelForCausalLM.from_pretrained(out)
    assert not (out / "adapter_config.json").exists()
    with torch.no_grad():
        got = merged(input_ids=x).logits
    assert torch.allclose(got, want, atol=1e-4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra train pytest tests/serve/test_merge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'yev.serve.merge'`

- [ ] **Step 3: Implement**

```python
"""Merge a LoRA adapter into its base for engines that serve plain weights (vLLM, SGLang). Local only."""

from __future__ import annotations


def export_merged(adapter_dir: str, base: str, out: str, save_tokenizer: bool = True) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base, dtype=dtype), adapter_dir)
    model.merge_and_unload().save_pretrained(out, safe_serialization=True)
    if save_tokenizer:
        AutoTokenizer.from_pretrained(base).save_pretrained(out)
```

CLI:

```python
    p = sub.add_parser("export-merged", help="merge a LoRA adapter into its base (for vLLM/SGLang; local only)")
    p.add_argument("--model", required=True, help="LoRA adapter dir")
    p.add_argument("--base", required=True, help="base model dir")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export_merged)
```

```python
def cmd_export_merged(args) -> int:
    from yev.serve.merge import export_merged
    export_merged(args.model, args.base, args.out)
    print(f"merged {args.model} into {args.base} -> {args.out}")
    return 0
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra train pytest tests/serve/test_merge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/yev/serve/merge.py src/yev/cli.py tests/serve/test_merge.py
git commit -m "feat(cli): yev export-merged merges a LoRA adapter for sidecar engines"
```

---

### Task 4: Box scripts for the sidecars

**Files:**
- Create: `scripts/aws/setup_vllm.sh`, `scripts/aws/setup_sglang.sh`, `scripts/aws/serve_engine.sh`
- Test: `tests/test_scripts.py` (append if it exists; else create) — `bash -n` syntax check

**Interfaces:**
- Produces: `bash scripts/aws/serve_engine.sh vllm|sglang <model_dir>` → engine on `127.0.0.1:8100`, log at
  `~/engine-<name>.log`, flags at `~/engine-<name>.flags`; returns once `/health` answers (or exits 1 after 15 min).

- [ ] **Step 1: Write the failing test**

```python
import subprocess
from pathlib import Path

import pytest

SCRIPTS = ["scripts/aws/setup_vllm.sh", "scripts/aws/setup_sglang.sh", "scripts/aws/serve_engine.sh"]


@pytest.mark.parametrize("path", SCRIPTS)
def test_engine_scripts_parse(path):
    assert Path(path).exists(), path
    subprocess.run(["bash", "-n", path], check=True)


def test_no_host_in_engine_scripts():
    for path in SCRIPTS:
        text = Path(path).read_text()
        assert "amazonaws" not in text and ".pem" not in text and "i-0" not in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_scripts.py -v`
Expected: FAIL (files missing)

- [ ] **Step 3: Write the scripts**

`scripts/aws/setup_vllm.sh` (setup_sglang.sh is the same with `NAME=sglang` and `PKG="sglang[all]"`):

```bash
#!/usr/bin/env bash
# Build ~/venv-vllm with the newest stable vLLM, separate from the training venv; record what was installed.
set -euo pipefail
NAME=vllm
PKG="${PKG:-vllm}"
VENV="$HOME/venv-$NAME"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" -U "$PKG"
uv pip freeze --python "$VENV/bin/python" > "$HOME/venv-$NAME.freeze"
grep -iE "^($NAME|torch|transformers|flashinfer|triton)[=<>]" "$HOME/venv-$NAME.freeze" || true
```

`scripts/aws/serve_engine.sh`:

```bash
#!/usr/bin/env bash
# Start a sidecar engine on 127.0.0.1:8100 for yev serve --backend vllm|sglang. Usage: serve_engine.sh vllm|sglang MODEL_DIR
set -euo pipefail
NAME="$1"; MODEL="$2"; PORT="${PORT:-8100}"; MAXLEN="${MAXLEN:-16384}"
case "$NAME" in
  vllm)
    FLAGS=(serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name yev --dtype bfloat16
           --max-model-len "$MAXLEN" --enable-prefix-caching --logprobs-mode processed_logprobs
           --max-logprobs 26 --gpu-memory-utilization 0.85 --trust-remote-code)
    CMD=("$HOME/venv-vllm/bin/vllm" "${FLAGS[@]}") ;;
  sglang)
    FLAGS=(--model-path "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype bfloat16
           --context-length "$MAXLEN" --mem-fraction-static 0.85 --trust-remote-code)
    CMD=("$HOME/venv-sglang/bin/python" -m sglang.launch_server "${FLAGS[@]}") ;;
  *) echo "usage: $0 vllm|sglang MODEL_DIR" >&2; exit 2 ;;
esac
echo "${CMD[*]}" > "$HOME/engine-$NAME.flags"
setsid nohup "${CMD[@]}" > "$HOME/engine-$NAME.log" 2>&1 < /dev/null &
for _ in $(seq 180); do
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "$NAME up on $PORT"; exit 0; }
  sleep 5
done
echo "!!! $NAME did not come up; tail of ~/engine-$NAME.log:" >&2; tail -40 "$HOME/engine-$NAME.log" >&2; exit 1
```

`chmod +x` all three.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_scripts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/aws/setup_vllm.sh scripts/aws/setup_sglang.sh scripts/aws/serve_engine.sh tests/test_scripts.py
git commit -m "feat(aws): vLLM and SGLang sidecar setup and launch scripts"
```

---

### Task 5: Dev-row parity (`scripts/backend_parity.py`)

**Files:**
- Create: `scripts/backend_parity.py`
- Test: `tests/test_backend_parity.py`

**Interfaces:**
- Consumes: `make_backend` (Task 1); `yev.train.infer.load`, `letter_logits`; `prompt_ids`, `letter_token_ids`;
  `readout.probs`; `load_temperatures`.
- Produces: CLI `score` (writes one JSON line per dev row: `{"id", "type", "answer", "probs"}` with `probs` over the
  row's letters in letter order) and `compare A B --out parity.json` (exit 1 below the bar). Pure function
  `compare_records(a: list[dict], b: list[dict]) -> dict` with keys `n`, `argmax_agreement`, `mean_abs_dp`,
  `max_abs_dp`, `accuracy_a`, `accuracy_b`, `at_parity`, `disagreements` (ids, at most 50).

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("backend_parity", Path("scripts/backend_parity.py"))
bp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bp)


def rec(id_, probs, answer="A"):
    return {"id": id_, "type": "choice", "answer": answer, "probs": probs}


def test_compare_identical_is_at_parity():
    a = [rec("x", [0.9, 0.1]), rec("y", [0.2, 0.8], "B")]
    out = bp.compare_records(a, a)
    assert out["argmax_agreement"] == 1.0 and out["mean_abs_dp"] == 0.0 and out["at_parity"]
    assert out["accuracy_a"] == out["accuracy_b"] == 1.0


def test_compare_flags_disagreement_and_dp():
    a = [rec("x", [0.9, 0.1]), rec("y", [0.6, 0.4])]
    b = [rec("x", [0.88, 0.12]), rec("y", [0.4, 0.6])]
    out = bp.compare_records(a, b)
    assert out["argmax_agreement"] == 0.5 and out["disagreements"] == ["y"]
    assert out["max_abs_dp"] == pytest.approx(0.2)
    assert out["mean_abs_dp"] == pytest.approx((0.02 + 0.2) / 2)
    assert not out["at_parity"]


def test_compare_requires_same_ids():
    with pytest.raises(ValueError, match="ids"):
        bp.compare_records([rec("x", [1.0])], [rec("z", [1.0])])


def test_by_state_order_groups_shared_states():
    import json
    def row(i, state):
        return {"id": str(i), "messages": [{"role": "user", "content": json.dumps({"state": state})}]}
    rows = [row(0, "s1"), row(1, "s2"), row(2, "s1")]
    assert [r["id"] for r in bp.by_state(rows)] == ["0", "2", "1"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_backend_parity.py -v`
Expected: FAIL (file missing)

- [ ] **Step 3: Implement**

```python
"""Letter-probability parity of a serving backend against the evaluated transformers readout, on dev rows.

Two phases, because the reference model and a sidecar engine shouldn't share the GPU:

    python scripts/backend_parity.py score --data data/mix/stage1c/dev.chat.jsonl --calibration c.json \\
        --model ~/runs/stage1c/final --base ~/models/Qwen3.5-4B-Base --out ref.jsonl
    python scripts/backend_parity.py score --data ... --calibration c.json --backend vllm \\
        --engine-url http://127.0.0.1:8100 --tokenizer ~/models/yev1c-merged --out vllm.jsonl
    python scripts/backend_parity.py compare ref.jsonl vllm.jsonl --out parity_vllm.json

Bar (spec): argmax agreement >= 99.5% and mean |dp| <= 0.01, |dp| being a row's largest per-letter difference.
compare exits 1 below the bar.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

AGREEMENT_BAR, MEAN_DP_BAR = 0.995, 0.01
CHUNK = 64  # rows in flight at once against a sidecar


def by_state(rows: list[dict]) -> list[dict]:
    """Rows reordered so rows sharing a state are adjacent (first-seen order), so prefix caches get reused."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(json.loads(r["messages"][-1]["content"]).get("state", ""), []).append(r)
    return [r for g in groups.values() for r in g]


def compare_records(a: list[dict], b: list[dict]) -> dict:
    ra, rb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if set(ra) != set(rb):
        raise ValueError(f"record ids differ: {len(set(ra) ^ set(rb))} ids in only one file")
    agree, dps, dis, ok_a, ok_b = 0, [], [], 0, 0
    for id_ in sorted(ra):
        pa, pb = ra[id_]["probs"], rb[id_]["probs"]
        ka, kb = max(range(len(pa)), key=pa.__getitem__), max(range(len(pb)), key=pb.__getitem__)
        agree += ka == kb
        if ka != kb:
            dis.append(id_)
        dps.append(max(abs(x - y) for x, y in zip(pa, pb)))
        gold = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(ra[id_]["answer"])
        ok_a += ka == gold
        ok_b += kb == gold
    n = len(ra)
    out = {"n": n, "argmax_agreement": agree / n, "mean_abs_dp": sum(dps) / n, "max_abs_dp": max(dps),
           "accuracy_a": ok_a / n, "accuracy_b": ok_b / n, "disagreements": dis[:50]}
    out["at_parity"] = out["argmax_agreement"] >= AGREEMENT_BAR and out["mean_abs_dp"] <= MEAN_DP_BAR
    return out


def _record(row: dict, scores: list[float], temps: dict) -> dict:
    from yev.train import readout
    n = len(row["letters"])
    return {"id": row["id"], "type": row["type"], "answer": row["answer"],
            "probs": readout.probs(scores, n, temps.get(row["type"], 1.0))}


def score_transformers(rows, args, temps) -> list[dict]:
    from yev.train.infer import letter_logits, load
    tok, model = load(args.model, args.base)
    n = max(len(r["letters"]) for r in rows)
    logits = letter_logits(model, tok, rows, max_len=args.max_len, batch_tokens=args.batch_tokens, n_letters=n)
    return [_record(r, z, temps) for r, z in zip(rows, logits)]


def score_backend(rows, args, temps) -> list[dict]:
    from transformers import AutoTokenizer

    from yev.serve.backends import make_backend
    from yev.train.data import letter_token_ids, prompt_ids
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    letters = letter_token_ids(tok, max(len(r["letters"]) for r in rows))
    backend = make_backend(args.backend, args.engine_url)

    async def run() -> list[dict]:
        out = []
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i : i + CHUNK]
            ids = [prompt_ids(tok, r["messages"]) for r in chunk]
            scores = await backend.letter_logprobs(ids, [letters[: len(r["letters"])] for r in chunk])
            out += [_record(r, z, temps) for r, z in zip(chunk, scores)]
            print(f"{len(out)}/{len(rows)}", file=sys.stderr, end="\r")
        return out

    return asyncio.run(run())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--calibration")
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--backend", choices=["transformers", "vllm", "sglang"], default="transformers")
    s.add_argument("--model")
    s.add_argument("--base")
    s.add_argument("--tokenizer", help="sidecar backends: tokenizer dir (the merged model dir)")
    s.add_argument("--engine-url", default="http://127.0.0.1:8100")
    s.add_argument("--max-len", type=int, default=16384)
    s.add_argument("--batch-tokens", type=int, default=16384)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        load = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]  # noqa: E731
        out = compare_records(load(args.a), load(args.b))
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=2))
        return 0 if out["at_parity"] else 1

    from yev.serve.engine import load_temperatures
    rows = by_state([json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()])
    rows = rows[: args.limit] if args.limit else rows
    temps = load_temperatures(args.calibration)
    recs = score_transformers(rows, args, temps) if args.backend == "transformers" else score_backend(rows, args, temps)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"wrote {len(recs)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_backend_parity.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/backend_parity.py tests/test_backend_parity.py
git commit -m "feat(scripts): dev-row letter-probability parity for serving backends"
```

---

### Task 6: Speed benchmark (`scripts/bench_backends.py`)

**Files:**
- Create: `scripts/bench_backends.py`
- Test: `tests/test_bench_backends.py`

**Interfaces:**
- Produces: CLI `grid --url U --out grid.json` and `replay --url U --data dev.chat.jsonl --out replay.json
  [--n 500] [--concurrency 1,4,16,64] [--seed 0]`; both record `--label` and, with `--gpu-mem`, peak
  `nvidia-smi` memory. Pure functions: `grid_state(tokens: int) -> str`, `grid_questions(n: int) -> dict`,
  `replay_requests(rows: list[dict], n: int, seed: int) -> list[dict]`, `summarise(latencies: list[float],
  wall_s: float, n_questions: int) -> dict` (keys `n`, `requests_per_s`, `questions_per_s`, `p50_s`, `p95_s`).

- [ ] **Step 1: Write the failing tests**

```python
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bench_backends", Path("scripts/bench_backends.py"))
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)


def dev_row(i, state, opts=("x", "y")):
    user = {"state": state, "question": f"q{i}?",
            "options": [{"label": "AB"[k], "key": o, "description": f"d {o}"} for k, o in enumerate(opts)]}
    return {"id": str(i), "type": "choice", "messages": [{"role": "user", "content": json.dumps(user)}]}


def test_replay_groups_rows_by_state_into_requests():
    rows = [dev_row(0, "s1"), dev_row(1, "s2"), dev_row(2, "s1")]
    reqs = bb.replay_requests(rows, n=10, seed=0)
    assert sorted(len(r["questions"]) for r in reqs) == [1, 2]
    two = next(r for r in reqs if len(r["questions"]) == 2)
    assert two["state"] == "s1"
    q = next(iter(two["questions"].values()))
    assert q["type"] == "choice" and q["criteria"] == {"x": "d x", "y": "d y"}


def test_replay_sample_is_seeded_and_capped():
    rows = [dev_row(i, f"s{i}") for i in range(50)]
    a, b = bb.replay_requests(rows, 10, 0), bb.replay_requests(rows, 10, 0)
    assert len(a) == 10 and a == b and a != bb.replay_requests(rows, 10, 1)


def test_grid_questions_mix_types():
    qs = bb.grid_questions(4)
    assert len(qs) == 4 and {q["type"] for q in qs.values()} == {"noul", "choice", "score"}


def test_summarise():
    s = bb.summarise([1.0, 2.0, 3.0, 4.0], wall_s=2.0, n_questions=8)
    assert s["n"] == 4 and s["requests_per_s"] == 2.0 and s["questions_per_s"] == 4.0
    assert s["p50_s"] == pytest.approx(2.5) and s["p95_s"] == pytest.approx(3.85)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_bench_backends.py -v`
Expected: FAIL (file missing)

- [ ] **Step 3: Implement**

```python
"""Latency grid and concurrent replay against any /v1/systemone server (transformers, vLLM or SGLang behind yev serve).

    python scripts/bench_backends.py grid   --url http://127.0.0.1:8000 --label vllm --gpu-mem --out grid_vllm.json
    python scripts/bench_backends.py replay --url http://127.0.0.1:8000 --label vllm --gpu-mem \\
        --data data/mix/stage1c/dev.chat.jsonl --out replay_vllm.json

grid: state ~300 / 2k / 8k tokens x 1 / 4 / 16 questions, 2 warm-up + 10 timed requests per cell, one at a time.
replay: 500 dev requests (rows regrouped by state), seeded, at concurrency 1, 4, 16, 64.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

PARA = ("Ticket 4471: the customer reports that the replacement router shipped last week drops its connection every "
        "evening. They have restarted it twice, swapped the cable and moved it away from the microwave. They ask "
        "whether they can return it for a refund or need another replacement, and mention a double shipping charge. ")
STATE_TOKENS, QUESTIONS, WARMUP, TIMED = (300, 2000, 8000), (1, 4, 16), 2, 10


def grid_state(tokens: int) -> str:
    return "\n".join(PARA for _ in range(max(1, tokens // 60)))  # ~60 tokens per paragraph


def grid_questions(n: int) -> dict:
    out = {}
    for i in range(n):
        if i % 3 == 0:
            out[f"q{i}"] = {"type": "noul", "instructions": f"Does the customer ask for a refund? (q{i})"}
        elif i % 3 == 1:
            out[f"q{i}"] = {"type": "choice", "instructions": f"Which team should handle this? (q{i})",
                            "criteria": {"billing": "charges and refunds", "hardware": "device faults",
                                         "shipping": "delivery issues", "other": None}}
        else:
            out[f"q{i}"] = {"type": "score", "instructions": f"How frustrated is the customer? (q{i})",
                            "criteria": ["calm", "mildly annoyed", "frustrated", "angry", "furious"]}
    return out


def replay_requests(rows: list[dict], n: int, seed: int) -> list[dict]:
    by_state: dict[str, dict] = {}
    for r in rows:
        u = json.loads(r["messages"][-1]["content"])
        req = by_state.setdefault(u["state"], {"state": u["state"], "questions": {}})
        req["questions"][f"q{len(req['questions'])}"] = {
            "type": "choice", "instructions": u["question"],
            "criteria": {o["key"]: o.get("description") for o in u["options"]}}
    reqs = list(by_state.values())
    random.Random(seed).shuffle(reqs)
    return reqs[:n]


def summarise(latencies: list[float], wall_s: float, n_questions: int) -> dict:
    q = statistics.quantiles(latencies, n=100, method="inclusive") if len(latencies) > 1 else latencies * 99
    return {"n": len(latencies), "requests_per_s": len(latencies) / wall_s, "questions_per_s": n_questions / wall_s,
            "p50_s": statistics.median(latencies), "p95_s": q[94]}


class GpuPeak:
    """Peak `nvidia-smi` memory.used (MiB) while running; None if nvidia-smi is missing."""

    def __init__(self, on: bool):
        self.on, self.peak, self._stop = on, None, threading.Event()

    def __enter__(self):
        if self.on:
            threading.Thread(target=self._poll, daemon=True).start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            try:
                used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                                    "--format=csv,noheader,nounits"], text=True).split()[0])
                self.peak = max(self.peak or 0, used)
            except (OSError, subprocess.CalledProcessError, ValueError):
                return
            self._stop.wait(0.5)

    def __exit__(self, *exc):
        self._stop.set()


async def _post(client, url, body) -> float:
    t = time.perf_counter()
    r = await client.post(f"{url}/v1/systemone", json=body)
    r.raise_for_status()
    return time.perf_counter() - t


async def run_grid(url: str) -> list[dict]:
    import httpx
    out = []
    async with httpx.AsyncClient(timeout=600) as c:
        for tok in STATE_TOKENS:
            for nq in QUESTIONS:
                body = {"state": grid_state(tok), "questions": grid_questions(nq)}
                for _ in range(WARMUP):
                    await _post(c, url, body)
                ts = [await _post(c, url, body) for _ in range(TIMED)]
                q = statistics.quantiles(ts, n=100, method="inclusive")
                out.append({"state_tokens": tok, "questions": nq, "p50_s": statistics.median(ts), "p95_s": q[94],
                            "ms_per_question": 1000 * statistics.median(ts) / nq})
                print(json.dumps(out[-1]))
    return out


async def run_replay(url: str, reqs: list[dict], concurrency: int) -> dict:
    import httpx
    sem, lat = asyncio.Semaphore(concurrency), []
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=concurrency)) as c:
        for body in reqs[:2]:
            await _post(c, url, body)  # warm-up

        async def one(body):
            async with sem:
                lat.append(await _post(c, url, body))

        t = time.perf_counter()
        await asyncio.gather(*(one(b) for b in reqs))
        wall = time.perf_counter() - t
    return {"concurrency": concurrency, **summarise(lat, wall, sum(len(b["questions"]) for b in reqs))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grid", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8000")
        p.add_argument("--label", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--gpu-mem", action="store_true")
        if name == "replay":
            p.add_argument("--data", required=True)
            p.add_argument("--n", type=int, default=500)
            p.add_argument("--concurrency", default="1,4,16,64")
            p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    with GpuPeak(args.gpu_mem) as gpu:
        if args.cmd == "grid":
            result = {"grid": asyncio.run(run_grid(args.url))}
        else:
            rows = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
            reqs = replay_requests(rows, args.n, args.seed)
            result = {"n_requests": len(reqs), "n_questions": sum(len(r["questions"]) for r in reqs),
                      "replay": [asyncio.run(run_replay(args.url, reqs, int(k))) for k in args.concurrency.split(",")]}
            for r in result["replay"]:
                print(json.dumps(r))
    result.update({"label": args.label, "url": args.url, "peak_gpu_mib": gpu.peak})
    Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_bench_backends.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/bench_backends.py tests/test_bench_backends.py
git commit -m "feat(scripts): latency grid and concurrent replay benchmark for serving backends"
```

---

### Task 7: Box run — gate, parity, speed, write-up

Operational; the controller runs it (no implementer subagent). Sync with `scripts/aws/sync.sh`; host and key from
`YEV_TRAIN_HOST` / `YEV_TRAIN_KEY`. Every step logs to `~/backends/run.log` with `=== step` markers.

- [ ] **Step 1: Merge and set up.** `yev export-merged --model ~/runs/stage1c/final --base ~/models/Qwen3.5-4B-Base
  --out ~/models/yev1c-merged`; `bash scripts/aws/setup_vllm.sh`; `bash scripts/aws/setup_sglang.sh`.
- [ ] **Step 2: Reference.** `python scripts/backend_parity.py score --backend transformers --model
  ~/runs/stage1c/final --base ~/models/Qwen3.5-4B-Base --calibration ~/runs/stage1c/calibration.json --data
  data/mix/stage1c/dev.chat.jsonl --out ~/backends/ref.jsonl`. Then the transformers speed runs: `yev serve` (1c
  adapter, calibration) on 8000, `bench_backends.py grid` and `replay` with `--label transformers --gpu-mem`, stop it.
- [ ] **Step 3: Per engine (vllm, then sglang) — gate.** `serve_engine.sh <name> ~/models/yev1c-merged`; if it does
  not come up, record the log tail as the gate result and skip to the next engine. Otherwise
  `backend_parity.py score --backend <name> --limit 20` and `compare` against the first 20 reference rows (by
  `id`): gate passes when every letter is returned and argmax agrees on ≥ 19/20. Record whether prefix caching is
  active from the engine log (vLLM: "prefix caching" / hit-rate lines; SGLang: radix cache hit rate).
- [ ] **Step 4: Per engine — parity.** Full `backend_parity.py score --backend <name>` and `compare` →
  `~/backends/parity_<name>.json`.
- [ ] **Step 5: Per engine — speed.** `yev serve --backend <name> --model ~/models/yev1c-merged --calibration
  ~/runs/stage1c/calibration.json --port 8000`; `grid` and `replay` with `--label <name> --gpu-mem`; stop both
  servers (kill by PID recorded at launch, not `pkill -f`).
- [ ] **Step 6: Fetch** `~/backends/*.json`, `~/engine-*.flags`, `~/venv-*.freeze` to `data/bench_results/backends/`.
- [ ] **Step 7: Write `docs/serving-backends.md`:** setup, how to run each backend, versions and flags, the gate
  outcome per engine, parity table (agreement, mean/max |dp|, accuracy, at parity or not), latency grid table per
  backend, replay table per backend and concurrency, peak GPU memory, caveats (transformers serves one request at a
  time; merged vs adapter weights; parity on dev rows only). Link it from `docs/serving.md`. Commit.
- [ ] **Step 8: Stop the instance** and report the tables to the owner.
