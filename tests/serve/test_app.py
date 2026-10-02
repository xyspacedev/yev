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

from yev.serve.app import create_app  # noqa: E402
from yev.serve.contract import SystemOneResponse  # noqa: E402
from yev.serve.engine import Engine  # noqa: E402

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


def test_internal_error_is_json_500(tok, tiny_model, monkeypatch):
    e = Engine("unused", None, None, max_len=4096, model=tiny_model, tokenizer=tok)
    def boom(*a, **k):
        raise ValueError("kaput")
    monkeypatch.setattr(e, "answer", boom)
    c = TestClient(create_app(e), raise_server_exceptions=False)
    r = c.post("/v1/systemone", json=json.loads((FIX / "request_noul.json").read_text()))
    assert r.status_code == 500
    assert r.json() == {"error": {"type": "internal_error", "message": "ValueError: kaput"}}


def test_chat_logprobs_zero_probability_is_serialisable():
    user = json.dumps({"state": "x", "question": "q?", "options": [
        {"label": "A", "key": "a", "description": "d"}, {"label": "B", "key": "b", "description": "e"}]})
    import yev.serve.app as appmod
    eng = Engine.__new__(Engine)
    eng.chat_probs = lambda m: {"A": 1.0, "B": 0.0}
    eng.n_tokens = lambda m: 3
    c = TestClient(appmod.create_app(eng))
    r = c.post("/v1/chat/completions", json={"logprobs": True, "messages": [{"role": "user", "content": user}]})
    assert r.status_code == 200
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    assert top[1]["token"] == "B" and top[1]["logprob"] < -600


def test_overlong_prompt_is_422_with_kit_marker(tok, tiny_model):
    e = Engine("unused", None, None, max_len=20, model=tiny_model, tokenizer=tok)
    c = TestClient(create_app(e))
    r = c.post("/v1/systemone", json={"state": "word " * 100, "questions": {
        "q": {"type": "noul", "instructions": "x"}}})
    assert r.status_code == 422
    assert r.json()["error"]["type"] == "unsupported"
    assert "maximum model length" in r.json()["error"]["message"]


def test_chat_temperature_by_inferred_type(tok, tiny_model, tmp_path):
    cal = tmp_path / "c.json"
    cal.write_text(json.dumps({"temperatures": {"noul": 4.0}}))
    hot = Engine("unused", None, str(cal), max_len=4096, model=tiny_model, tokenizer=tok)
    plain = Engine("unused", None, None, max_len=4096, model=tiny_model, tokenizer=tok)
    def turn(k1, k2):
        return [{"role": "user", "content": json.dumps({"state": "x", "question": "q?", "options": [
            {"label": "A", "key": k1, "description": "d"}, {"label": "B", "key": k2, "description": "e"}]})}]
    noul_t, noul_1 = hot.chat_probs(turn("yes", "no")), plain.chat_probs(turn("yes", "no"))
    assert max(noul_t.values()) < max(noul_1.values())  # T=4 flattens
    assert max(noul_t, key=noul_t.get) == max(noul_1, key=noul_1.get)
    assert hot.chat_probs(turn("red", "blue")) == plain.chat_probs(turn("red", "blue"))  # choice: no temperature set
