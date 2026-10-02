# `/v1/systemone` wire format (typesafe-sdk 0.7.2)

This file is the spec that later tasks work from, so they don't need to read the SDK. Every claim
was read from the SDK sdist or from the client code cited below. Where marked "probed", it was
also checked by running the real SDK against an `httpx2.MockTransport`.

Paths: `models.py` is `typesafe_sdk/_schemas/models.py`, generated from
`https://api.typesafe.ai/openapi.json`. The other bare paths are under `typesafe_sdk/_core/`.
The pydantic mirror is `src/yev/serve/contract.py`.

## Request: `POST /v1/systemone`, JSON body

| Field | Type | Required | Notes |
|---|---|---|---|
| `state` | string, JSON object or JSON array | yes | models.py:198. **WorkflowEvals sends a JSON object** (a `dict`). The decision-index kit sends a string or an object, and the string may be `""`. |
| `model` | string | yes on the wire (models.py:204) | The SDK always fills it in, defaulting to `"jev-latest"` (endpoints.py:29, `constants.py:18`). Our model makes it optional and ignores it. |
| `questions` | object: name → question | yes, at least 1 entry | models.py:210. The response must use the same names. |
| anything else | any | no | The SDK's `extra_body` shallow-merges extra top-level keys (client.py:153). The kit's `extra` option does the same (`http.py:39`). **We ignore them.** |

A question is discriminated by `type`. Unknown keys inside a question are rejected, because the
SDK's `_Question` is `extra="forbid"` (question_types.py:68) and its TypedDicts are `closed=True`.

| `type` | `instructions` | `criteria` |
|---|---|---|
| `"noul"` | optional, nullable: string, object or array | optional, nullable `{"true": desc?, "false": desc?}`. Each desc is optional and nullable (string, object or array). No other keys are allowed. models.py:83-104 |
| `"choice"` | optional, nullable | **required** object: option key → description. A description may be `null`, meaning "interpret by the key alone" (models.py:43-50). Key order is the option order. The wire sets no minimum count. The kit uses 2–255 options. |
| `"score"` | optional, nullable | **required** non-empty **array** of level descriptions (string, object or array; never null). An entry's position is its level, starting at 0 (models.py:142-148). The SDK refuses empty criteria client-side (questions.py:26-29). |

### What the clients actually send (probed)

- **WorkflowEvals** passes every question through `question_record` (`core/question_types.py:18-24`), so it **always sends all three keys, with explicit nulls**:
  - `{"type":"noul","instructions":null,"criteria":null}`
  - `{"type":"noul","instructions":"Spam?","criteria":{"true":"ads","false":null}}`
  - `{"type":"choice","instructions":null,"criteria":{"a":null,"b":"B"}}`
  - `{"type":"score","instructions":null,"criteria":["lo","hi"]}`
  - The server must accept every null shown above.
- **The SDK with question objects** drops unset top-level `None`s (question_types.py:70-74). For example: `{"type":"noul","instructions":"x"}`.
- **decision-index `http`** posts `{"model", "state", "questions", **extra}` unmodified (`decision_index/engines/http.py:39`).

## Response: 200, JSON body

```json
{"model": "<served name>", "answers": {"<question name>": <answer>, ...}, "usage": {"input_tokens": 120, "output_tokens": 12}}
```

- `model` (string, required): may differ from the requested alias (models.py:220-225).
- `answers` (required, at least 1 entry): keyed by the request's question names. WorkflowEvals raises an error if the key set differs from the request's (`core/session.py:92`). The kit's validator also requires the key sets to match (`engines/base.py:21`).
- `usage` (required). The wire schema needs `input_tokens` and `output_tokens` as ints (models.py:151-160). The SDK tolerates `{}` and parses it as None/None (response_types.py:65-73, probed), **but a missing `usage` fails** (probed: `Invalid response data at 'usage'`). WorkflowEvals also reads the optional `input_tokens_total`, `output_tokens_total` and `n_retries` (`core/session.py:85-90`). We send only the two required ints. `output_tokens` = 0, since nothing is generated.
- Unknown top-level and per-answer keys are ignored by the SDK (`Schema` is `extra="ignore"`, `_core/schemas/base.py:21`). The kit strips `evaluation_trace` (http.py:46).

**Every answer must carry `type`.** The SDK raises `Invalid response data at 'answers.<name>.type'` without it (response_types.py:87-88, probed). The kit's validator also checks `a["type"] == q["type"]` (base.py:25).

| `type` | Fields (all required) | Semantics |
|---|---|---|
| `"noul"` | `noul: float` | P(yes / true), from 0 to 1 (models.py:73-80). **There is no `probabilities` and no `confidence`.** WorkflowEvals derives `{"true": p, "false": 1-p}` itself (`core/session.py:29-32`). |
| `"choice"` | `choice: str`, `confidence: float`, `probabilities: {key: float}` | `choice` is the argmax key. `probabilities` is keyed by **every** criteria key and sums to about 1. The kit requires the key set to equal the criteria keys exactly and the sum to be within 0.01 (base.py:28-32). `confidence` is free-form, from 0 to 1. WorkflowEvals recomputes confidence as max(probabilities) (`session.py:45`), so return max(p). |
| `"score"` | `score: float`, `confidence: float`, `legend: {"0": desc, ...}`, `probabilities: {"0": float, ...}` | `score` is the **expected level** Σ level·p, a float that may fall between levels (models.py:109-114). There is **no separate categorical/rounded field**. `legend` maps the level index, as a string key, to that level's criteria entry, echoed verbatim. `probabilities` has the same keys as `legend`. JSON keys are strings; the SDK coerces them to `int` (response_types.py:53-57, probed). WorkflowEvals classifies by the modal level from `probabilities`, not by `score` (`session.py:35-37`), and drops `legend` from its records (`question_types.py:30-31`). |

The kit's `validate` has no score branch, so a score question raises "Unsupported question type" there (base.py:36-37). The kit's suite only contains choice and noul questions.

## Errors

- The TypeSafe API is FastAPI-generated. Its 422 is `{"detail": [{"loc": [...], "msg": ..., "type": ..., "input"?, "ctx"?}]}` (models.py:163-190). FastAPI's default request-validation 422 already matches this shape.
- The SDK maps HTTP status to exception classes (errors.py:188-200): 400, 401, 403, 404, **422 → `TypeSafeUnprocessableEntityError`**, 429 and 5xx. It takes the message from `error` (a string), `error.message`, `message`, `detail` (a string), `detail.message` or `detail[].msg` (errors.py:40-65). Otherwise it uses the raw body, truncated to 200 characters.
- Retries (WorkflowEvals config: `RetryPolicy(max_retries=5, backoff_initial=2.0, backoff_max=60.0, timeout=None)`, `core/clients.py:86`) apply to **408, 429 and 5xx**, plus connection errors and timeouts (retry.py:64). **422 is not retried**, so an `unsupported` refusal is final.
- Our refusal body is `{"error": {"type": "unsupported", "message": R, "reason": R}}` (`ErrorResponse.unsupported`). `message` is what the SDK shows (probed: `422 too many options per choice (27 > 26)`). Without `message`, the SDK prints the raw JSON instead.
- **The decision-index kit treats a 400/413/422 as `Unsupported` only if the body text contains one of these markers** (http.py:5-15): `"options per choice"`, `"a choice needs at least two options"`, `"a score takes 2 to 10 levels"`, `"the canvas holds"`, `"maximum context length"`, `"maximum model length"`, `"longer than the maximum model length"`, `"context window"`, `"too many tokens"`. Any other 4xx is raised as an error, which the kit retries on resume and which stops the run after 5 errors before the first success (`docs/engines.md:5`). So the reason strings should contain a marker, for example:
  - `"too many options per choice (27 > 26)"`
  - `"state too long: longer than the maximum model length (N > max_len tokens)"`

  The plan's `"too many options (N > 26)"` and `"state too long"` match no marker.

## `GET /v1/models`

The SDK expects `{"models": [{"name", "description", "release_date": "YYYY-MM-DD"}]}` (models.py:53-70, response_types.py:142-165). Our `ModelsResponse` also includes OpenAI's `{"object": "list", "data": [{"id", "object": "model", ...}]}` in the same body. The SDK ignores the extra keys (probed).

## How each client calls the server

| | WorkflowEvals `typesafe` provider | decision-index `http` engine |
|---|---|---|
| Invocation | `uv run python run.py <workflow> --model typesafe:jev-1.13.0 --base-url http://HOST:PORT [--name yev-4b] [--timeout 600]` (`run.py:39-55`) | `python -m decision_index run --engine http --option base_url=http://HOST:PORT --option model=yev-4b` (`docs/engines.md:12`) |
| Client | `typesafe_sdk.TypeSafeClient(base_url=..., timeout=config.timeout_s, retry=...)` (`core/clients.py:88`), using httpx2 | `httpx.Client(base_url, timeout=600, headers)` (http.py:35) |
| URL | `base_url.rstrip("/") + "/v1/systemone"` (config.py:61, endpoints.py:34). `TYPESAFE_BASE_URL` is used if `--base-url` is absent (providers.py:62-66). | `/v1/systemone` relative to `base_url`, or `DECISION_INDEX_BASE_URL` |
| Model name | **Must be `jev-1.13.0`**: `providers.py:33` rejects any other TypeSafe model. It is sent as `model`. The server must accept it and may answer under its own name. | `model` option, default `"default"` (http.py:22) |
| Auth | `Authorization: Bearer <TYPESAFE_API_KEY>`. **The key must be set, non-empty, printable ASCII and without spaces**, or the SDK raises before sending (config.py:26-33). Any dummy value works against our server. | `Authorization: Bearer $DECISION_INDEX_API_KEY`, only if that variable is set (http.py:29-32) |
| Other headers | `Accept: application/json`, `Content-Type: application/json`, `User-Agent: typesafe-sdk/0.7.2`, `X-TypeSafe-SDK`, `X-TypeSafe-Runtime`, and `X-TypeSafe-Retry-Count` on retries (transport.py:116-131, probed). It reads the optional `x-typesafe-request-id` response header. | — |
| Timeout | 600 s per request by default (`providers.py:27`, `run.py:40`). The SDK's own default is 10 s (`constants.py:21`). | 600 s (http.py:22) |
| Batching | `question_mode="all"`: one request per workflow node carrying all of that node's questions (providers.py:68, session.py:42-46). Up to about 42 questions over a 6–9k-token state for invoices. | One request per suite row, carrying all of that row's questions |
| Response use | The SDK parses the body into `nouls`/`choices`/`scores`; WorkflowEvals then uses `noul`, `probabilities` (modal level for score) and `usage` | The body is used as-is and validated by `engines/base.py:20-37` |

## The OpenAI-compatible chat endpoint (ours, minimal)

`POST /v1/chat/completions`:
- Request: `{model?, messages: [{role, content: str}], max_tokens?, temperature?}`. Other keys are ignored.
- Response: `{id: "chatcmpl-…", object: "chat.completion", created, model, choices: [{index: 0, message: {role: "assistant", content}, finish_reason: "stop"}], usage: {prompt_tokens, completion_tokens, total_tokens}}`.

Neither client calls this endpoint.
