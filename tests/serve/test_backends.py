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
    # the model's generation_config must not leak penalties into the letter log-probs
    assert body["repetition_penalty"] == 1.0 and body["frequency_penalty"] == 0 and body["presence_penalty"] == 0


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
    assert body["sampling_params"] == {"max_new_tokens": 1, "temperature": 0, "repetition_penalty": 1.0,
                                       "frequency_penalty": 0, "presence_penalty": 0}
    assert body["return_logprob"] is True and body["token_ids_logprob"] == [10, 11, 12]


def test_sglang_missing_letter_is_engine_error():
    def handler(request):
        return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [[[-0.1, 10, None]]]}})

    with pytest.raises(EngineError, match="11"):
        run(SGLangBackend("http://engine", client(handler)).letter_logprobs([[1]], [[10, 11]]))


@pytest.mark.parametrize("make", [lambda c: VLLMBackend("http://engine", c),
                                  lambda c: SGLangBackend("http://engine", c)])
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


@pytest.mark.parametrize("make", [lambda c: VLLMBackend("http://engine", c),
                                  lambda c: SGLangBackend("http://engine", c)])
def test_dropped_keepalive_connection_is_retried_once(make):
    # The engine's server closes idle keep-alive sockets after 5 s, the same as httpx's pool expiry, so a request
    # can go out on a socket the server just closed. Scoring is idempotent, so one retry is safe.
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadError("")
        body = json.loads(request.content)
        if "allowed_token_ids" in body:
            return vllm_ok(request)
        return httpx.Response(200, json={"meta_info": {"output_token_ids_logprobs": [[[-0.1, 10, None]]]}})

    assert len(run(make(client(handler)).letter_logprobs([[1]], [[10]]))[0]) == 1
    assert len(calls) == 2
