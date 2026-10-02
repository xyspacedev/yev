"""FastAPI app behind /v1/systemone (Plan 4 Task 4)."""

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("torch")
pytest.importorskip("transformers")
from fastapi.testclient import TestClient  # noqa: E402

from jeff.serve.app import create_app  # noqa: E402
from jeff.serve.contract import SystemOneResponse  # noqa: E402
from jeff.serve.engine import Engine  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
REQUESTS = sorted(FIX.glob("request_*.json"))


@pytest.fixture
def client(tok, tiny_model):
    e = Engine("unused", None, None, max_len=4096, model=tiny_model, tokenizer=tok)
    return TestClient(create_app(e, model_name="yev-test"))


def kit_validate(req: dict, body: dict) -> None:
    """The decision-index kit's engines/base.py validate, as summarised in WIRE_FORMAT.md."""
    answers = body["answers"]
    assert set(answers) == set(req["questions"])
    for name, q in req["questions"].items():
        a = answers[name]
        assert a["type"] == q["type"]
        if q["type"] == "noul":
            assert 0.0 <= a["noul"] <= 1.0
        elif q["type"] == "choice":
            p = a["probabilities"]
            assert set(p) == set(q["criteria"])
            assert all(math.isfinite(v) for v in p.values())
            assert abs(sum(p.values()) - 1.0) <= 0.01
            assert a["choice"] in q["criteria"]


@pytest.mark.parametrize("path", REQUESTS, ids=lambda p: p.name)
def test_fixture_requests_ok(client, path):
    req = json.loads(path.read_text())
    r = client.post("/v1/systemone", json=req, headers={"Authorization": "Bearer dummy"})
    assert r.status_code == 200, r.text
    resp = SystemOneResponse.model_validate(r.json())
    assert resp.model == "yev-test"
    assert resp.usage.output_tokens == 0 and resp.usage.input_tokens > 0
    kit_validate(req, r.json())


def test_no_auth_needed_and_extra_top_level_ignored(client):
    req = json.loads((FIX / "request_noul.json").read_text())
    r = client.post("/v1/systemone", json={**req, "whatever": 1})
    assert r.status_code == 200


def test_too_many_options_is_422_unsupported(client):
    q = {"type": "choice", "criteria": {f"k{i}": f"d{i}" for i in range(27)}}
    r = client.post("/v1/systemone", json={"state": "s", "model": "m", "questions": {"q": q}})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["type"] == "unsupported"
    assert "options per choice" in err["reason"] and err["message"] == err["reason"]


def test_validation_error_is_422_with_envelope_and_detail(client):
    r = client.post("/v1/systemone", json={"state": "s", "questions": {}})
    assert r.status_code == 422
    body = r.json()
    assert isinstance(body["detail"], list) and body["detail"][0]["msg"]
    assert body["error"]["type"] == "invalid_request" and body["error"]["message"]


def test_unknown_question_field_rejected(client):
    q = {"type": "noul", "bogus": 1}
    r = client.post("/v1/systemone", json={"state": "s", "questions": {"q": q}})
    assert r.status_code == 422


def test_chat_returns_one_letter(client):
    user = json.dumps({"state": "x", "question": "q?", "options": [
        {"label": "A", "key": "a", "description": "d"}, {"label": "B", "key": "b", "description": "e"}]})
    r = client.post("/v1/chat/completions", json={"model": "gpt", "messages": [
        {"role": "system", "content": "sys"}, {"role": "user", "content": user}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "yev-test"
    assert body["choices"][0]["message"]["content"] in ("A", "B")
    assert body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] == 1


def test_chat_logprobs(client):
    user = json.dumps({"state": "x", "question": "q?", "options": [
        {"label": "A", "key": "a", "description": "d"}, {"label": "B", "key": "b", "description": "e"}]})
    r = client.post("/v1/chat/completions", json={"logprobs": True, "messages": [{"role": "user", "content": user}]})
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    assert {t["token"] for t in top} == {"A", "B"}
    assert abs(sum(math.exp(t["logprob"]) for t in top) - 1.0) < 1e-6


def test_chat_not_our_format_is_422(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 422 and r.json()["error"]["type"] == "unsupported"


def test_health_and_models(client):
    assert client.get("/health").status_code == 200
    m = client.get("/v1/models").json()
    assert m["models"][0]["name"] == "yev-test" and m["data"][0]["id"] == "yev-test" and m["object"] == "list"
