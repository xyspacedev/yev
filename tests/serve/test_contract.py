"""The /v1/systemone wire format, pinned to typesafe-sdk 0.7.2 (see fixtures/WIRE_FORMAT.md)."""

import json
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")
from pydantic import TypeAdapter, ValidationError  # noqa: E402

from jeff.serve.contract import (  # noqa: E402
    ChatRequest,
    ChatResponse,
    ChoiceAnswer,
    ChoiceQuestion,
    ErrorResponse,
    ModelsResponse,
    NoulAnswer,
    NoulQuestion,
    QuestionSpec,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
)

FIXTURES = Path(__file__).parent / "fixtures"
KINDS = ["noul", "choice", "score"]


def load(name):
    return json.loads((FIXTURES / name).read_text())


@pytest.mark.parametrize("kind", KINDS)
def test_fixture_request_parses(kind):
    req = SystemOneRequest.model_validate(load(f"request_{kind}.json"))
    (q,) = req.questions.values()
    assert q.type == kind
    assert req.model == "jev-latest"


def test_question_classes_by_type():
    req = SystemOneRequest.model_validate(load("request_workflowevals.json"))
    qs = req.questions
    assert isinstance(qs["desired_outcome"], ChoiceQuestion)
    assert isinstance(qs["urgency"], ScoreQuestion)
    assert isinstance(qs["requests_human"], NoulQuestion)
    assert list(qs["desired_outcome"].criteria)[0] == "money_back"
    assert qs["category"].criteria == {"billing": None, "technical": None, "other": None}
    assert len(qs["urgency"].criteria) == 4 and qs["urgency"].criteria[0] == "No time pressure mentioned."
    assert qs["requests_human"].criteria.true.startswith("The customer explicitly asks")
    assert qs["bare"].instructions is None and qs["bare"].criteria is None
    assert qs["half"].criteria.false is None
    assert req.state == {"conversation": [{"role": "customer", "text": "I was charged twice. Please help."}]}


def test_question_spec_is_discriminated_by_type():
    q = TypeAdapter(QuestionSpec).validate_python({"type": "score", "criteria": ["a", "b"]})
    assert isinstance(q, ScoreQuestion)


@pytest.mark.parametrize("kind", KINDS)
def test_fixture_response_round_trips(kind):
    raw = load(f"response_{kind}.json")
    resp = SystemOneResponse.model_validate(raw)
    assert resp.model_dump(exclude_none=True) == raw
    assert json.loads(resp.model_dump_json(exclude_none=True)) == raw


def test_answer_classes_and_type_tag_serialised():
    a = NoulAnswer(noul=0.25)
    c = ChoiceAnswer(choice="x", confidence=0.6, probabilities={"x": 0.6, "y": 0.4})
    s = ScoreAnswer(score=0.4, confidence=0.6, legend={"0": "lo", "1": "hi"}, probabilities={"0": 0.6, "1": 0.4})
    resp = SystemOneResponse(model="yev-4b", answers={"a": a, "c": c, "s": s}, usage=Usage(input_tokens=3, output_tokens=0))
    out = resp.model_dump()
    assert out["answers"]["a"] == {"type": "noul", "noul": 0.25}
    assert out["answers"]["c"]["type"] == "choice"
    assert out["answers"]["s"]["type"] == "score"
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 0}


# --- requests: reject unknown fields where the SDK's own question types do (closed TypedDicts,
#     extra="forbid"); tolerate unknown top-level fields, which the SDK's extra_body and the
#     decision-index http engine's `extra` option both send.


def test_unknown_question_field_rejected():
    raw = load("request_noul.json")
    raw["questions"]["billing"]["samples"] = 3
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(raw)


def test_unknown_noul_criteria_key_rejected():
    raw = load("request_noul.json")
    raw["questions"]["billing"]["criteria"] = {"true": "yes", "maybe": "?"}
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(raw)


def test_unknown_top_level_field_ignored():
    raw = load("request_noul.json")
    raw["samples"] = 1
    req = SystemOneRequest.model_validate(raw)
    assert "samples" not in req.model_dump()


def test_unknown_question_type_rejected():
    raw = load("request_noul.json")
    raw["questions"]["billing"]["type"] = "rank"
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(raw)


@pytest.mark.parametrize(
    "question",
    [
        {"type": "score", "criteria": []},
        {"type": "score"},
        {"type": "choice"},
    ],
)
def test_required_criteria(question):
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "questions": {"q": question}})


def test_empty_questions_rejected():
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate({"state": "s", "model": "m", "questions": {}})


def test_model_optional_and_state_may_be_empty_string_or_list():
    req = SystemOneRequest.model_validate({"state": "", "questions": {"q": {"type": "noul"}}})
    assert req.model is None and req.state == ""
    req = SystemOneRequest.model_validate({"state": ["a", 1], "questions": {"q": {"type": "noul"}}})
    assert req.state == ["a", 1]


def test_instructions_may_be_json_object():
    req = SystemOneRequest.model_validate(
        {"state": "s", "questions": {"q": {"type": "noul", "instructions": {"task": "Identify unsolicited advertising."}}}}
    )
    assert req.questions["q"].instructions == {"task": "Identify unsolicited advertising."}


# --- responses: the SDK ignores unknown fields (Schema extra="ignore") but requires every answer's
#     `type`, and `usage`.


def test_response_ignores_unknown_fields():
    raw = load("response_noul.json")
    raw["evaluation_trace"] = {"x": 1}
    raw["answers"]["billing"]["debug"] = "y"
    assert SystemOneResponse.model_validate(raw).model_dump(exclude_none=True) == load("response_noul.json")


def test_response_answer_needs_type():
    raw = load("response_noul.json")
    del raw["answers"]["billing"]["type"]
    with pytest.raises(ValidationError):
        SystemOneResponse.model_validate(raw)


def test_response_needs_usage():
    raw = load("response_noul.json")
    del raw["usage"]
    with pytest.raises(ValidationError):
        SystemOneResponse.model_validate(raw)


def test_error_response_shape():
    body = ErrorResponse.unsupported("too many options per choice (27 > 26)").model_dump()
    assert body == {
        "error": {
            "type": "unsupported",
            "message": "too many options per choice (27 > 26)",
            "reason": "too many options per choice (27 > 26)",
        }
    }


def test_models_response_carries_typesafe_and_openai_shapes():
    out = ModelsResponse.single("yev-4b", description="d", release_date="2026-10-01").model_dump()
    assert out["models"] == [{"name": "yev-4b", "description": "d", "release_date": "2026-10-01"}]
    assert out["object"] == "list"
    assert out["data"][0]["id"] == "yev-4b" and out["data"][0]["object"] == "model"


# --- OpenAI-compatible chat, minimal.


def test_chat_request_minimal_and_tolerant():
    req = ChatRequest.model_validate(
        {"model": "yev-4b", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], "logprobs": True}
    )
    assert [m.role for m in req.messages] == ["system", "user"]
    assert req.max_tokens is None


def test_chat_response_shape():
    resp = ChatResponse.build(model="yev-4b", content="B", prompt_tokens=10, completion_tokens=1)
    out = resp.model_dump()
    assert out["object"] == "chat.completion"
    assert out["choices"] == [{"index": 0, "message": {"role": "assistant", "content": "B"}, "finish_reason": "stop"}]
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    assert out["id"].startswith("chatcmpl-") and isinstance(out["created"], int)
