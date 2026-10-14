"""FastAPI app: TypeSafe's ``POST /v1/systemone`` and a minimal OpenAI-compatible chat endpoint.

Auth headers are never checked (any bearer key works). The requested ``model`` is accepted and
ignored: one model is loaded and responses report ``model_name``. For the in-process Engine the
endpoints are sync, so FastAPI runs them in its threadpool and the Engine serialises forward passes
itself. For a sidecar backend (``RemoteEngine``, which has ``answer_async``) they are async, so
concurrent requests reach the engine together; an unusable engine answer is a 503.
"""

from __future__ import annotations

import logging
import math

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from yev.serve.backends import EngineError
from yev.serve.contract import (
    ChatRequest,
    ChatResponse,
    ErrorInfo,
    ErrorResponse,
    ModelsResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from yev.serve.mapping import Unsupported

log = logging.getLogger(__name__)


def create_app(engine, model_name: str = "yev-4b") -> FastAPI:
    app = FastAPI(title="yev serve", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(Unsupported)
    def _unsupported(request: Request, exc: Unsupported) -> JSONResponse:
        return JSONResponse(ErrorResponse.unsupported(str(exc)).model_dump(), status_code=422)

    @app.exception_handler(EngineError)
    def _engine_down(request: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse({"error": {"type": "engine_unavailable", "message": str(exc)}}, status_code=503)

    @app.exception_handler(Exception)
    def _internal(request: Request, exc: Exception) -> JSONResponse:
        # A bug or a CUDA OOM: a JSON 500 (the SDK retries 5xx and shows `error.message`), never "unsupported".
        log.error("internal error on %s", request.url.path, exc_info=exc)
        msg = f"{type(exc).__name__}: {exc}"
        return JSONResponse({"error": {"type": "internal_error", "message": msg}}, status_code=500)

    @app.exception_handler(RequestValidationError)
    def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        # TypeSafe's own 422 is FastAPI's {"detail": [...]}, which the SDK reads (`detail[].msg`);
        # the same `error` envelope as a refusal is added for our own clients.
        detail = jsonable_encoder(exc.errors(), custom_encoder={Exception: str})
        msg = "; ".join(f"{'.'.join(str(x) for x in d.get('loc', []))}: {d.get('msg')}" for d in detail)
        body = {"detail": detail,
                "error": ErrorInfo(type="invalid_request", message=msg, reason=msg).model_dump()}
        return JSONResponse(body, status_code=422)

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

    @app.get("/v1/models", response_model=ModelsResponse)
    def models() -> ModelsResponse:
        return ModelsResponse.single(model_name, description="yev System One decision model",
                                     release_date="2026-10-01")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name}

    return app
