"""Wire format of TypeSafe's ``POST /v1/systemone``, pinned to typesafe-sdk 0.7.2, plus a minimal
OpenAI-compatible ``POST /v1/chat/completions``.

The authority is the SDK's generated OpenAPI models (``typesafe_sdk/_schemas/models.py``) and the
client-side question and answer types layered on them. ``tests/serve/fixtures/WIRE_FORMAT.md`` is
the human-readable spec, with file:line citations.

Leniency follows the SDK:
- question objects reject unknown fields (the SDK's ``_Question`` is ``extra="forbid"`` and its
  question TypedDicts are closed);
- unknown top-level request fields are ignored (the SDK's ``extra_body`` and the decision-index
  ``http`` engine's ``extra`` option both add them);
- responses ignore unknown fields (the SDK's ``Schema`` is ``extra="ignore"``) but every answer
  must carry its ``type`` and the response must carry ``usage``.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# Text, a JSON object or a JSON array: the SDK's JSONContent (typesafe_sdk/_core/json_types.py:17).
JSONContent = Union[str, dict[str, Any], list[Any]]


# ---------------------------------------------------------------------------------------------
# Requests


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulCriteria(BaseModel):
    """What counts as yes (``true``) and no (``false``); either may be missing or null."""

    model_config = ConfigDict(extra="forbid")

    true: JSONContent | None = None
    false: JSONContent | None = None


class NoulQuestion(_Question):
    type: Literal["noul"] = "noul"
    instructions: JSONContent | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Question):
    type: Literal["choice"] = "choice"
    instructions: JSONContent | None = None
    # Option key -> description; a null description means the key is interpreted by its name alone.
    criteria: dict[str, JSONContent | None]


class ScoreQuestion(_Question):
    type: Literal["score"] = "score"
    instructions: JSONContent | None = None
    # Ordered level descriptions; the position is the level, starting at 0.
    criteria: list[JSONContent] = Field(min_length=1)


QuestionSpec = Annotated[Union[NoulQuestion, ChoiceQuestion, ScoreQuestion], Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # The SDK always sends a model (default "jev-latest"); the decision-index http engine sends
    # "default" unless told otherwise. The server answers with its own model whatever is asked.
    model: str | None = None
    state: JSONContent
    questions: dict[str, QuestionSpec] = Field(min_length=1)


# ---------------------------------------------------------------------------------------------
# Responses


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")


class NoulAnswer(_Answer):
    type: Literal["noul"] = "noul"
    noul: float  # P(yes / true)


class ChoiceAnswer(_Answer):
    type: Literal["choice"] = "choice"
    choice: str  # the most probable criteria key
    confidence: float
    probabilities: dict[str, float]  # keyed by every criteria key


class ScoreAnswer(_Answer):
    type: Literal["score"] = "score"
    score: float  # expected level: sum(level * p); may fall between levels
    confidence: float
    legend: dict[str, JSONContent]  # "0", "1", ... -> the level's criteria entry
    probabilities: dict[str, float]  # same keys as legend


Answer = Annotated[Union[NoulAnswer, ChoiceAnswer, ScoreAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    answers: dict[str, Answer] = Field(min_length=1)
    usage: Usage


class ErrorInfo(BaseModel):
    type: str
    # `message` is what the SDK surfaces (errors.py:48); `reason` is kept for our own clients.
    message: str
    reason: str


class ErrorResponse(BaseModel):
    error: ErrorInfo

    @classmethod
    def unsupported(cls, reason: str) -> ErrorResponse:
        return cls(error=ErrorInfo(type="unsupported", message=reason, reason=reason))


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str  # YYYY-MM-DD


class OpenAIModel(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "jeff"


class ModelsResponse(BaseModel):
    """``GET /v1/models``: TypeSafe's ``models`` list and OpenAI's ``data`` list in one body."""

    models: list[ModelMetadata]
    object: Literal["list"] = "list"
    data: list[OpenAIModel]

    @classmethod
    def single(cls, name: str, *, description: str, release_date: str) -> ModelsResponse:
        return cls(
            models=[ModelMetadata(name=name, description=description, release_date=release_date)],
            data=[OpenAIModel(id=name)],
        )


# ---------------------------------------------------------------------------------------------
# OpenAI-compatible chat (minimal)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None
    logprobs: bool | None = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: dict | None = None  # only when requested: {"content": [{token, logprob, top_logprobs}]}


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage

    @classmethod
    def build(cls, *, model: str, content: str, prompt_tokens: int, completion_tokens: int) -> ChatResponse:
        return cls(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=content))],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
