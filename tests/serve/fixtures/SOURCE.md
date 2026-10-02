# Fixture sources

All paths are inside the `typesafe-sdk==0.7.2` sdist (`typesafe_sdk-0.7.2.tar.gz` on PyPI), unless
marked WorkflowEvals (github.com/typesafe-ai/WorkflowEvals @ 0ac3b8a). `models.py` means
`src/typesafe_sdk/_schemas/models.py`, which is generated from `https://api.typesafe.ai/openapi.json`
(models.py:1-2). The values are the OpenAPI `examples=`, copied verbatim. Each request's question
name comes from the SDK too, as noted.

| Fixture | Field | Source |
|---|---|---|
| `request_noul.json` | whole body (`state`, `model`, `questions`) | models.py:201 (first `state` example), :207 (`model`), :213 (`questions`) |
| `request_choice.json` | `state` | models.py:201 (second `state` example, a JSON object) |
| | `model` | models.py:207 |
| | question name `tone` | `src/typesafe_sdk/_core/client/sync/client.py:71` (docstring example) |
| | `instructions` | models.py:40 (`ChoiceQuestion.instructions` example) |
| | `criteria` | models.py:47 (`ChoiceQuestion.criteria` example) |
| `request_score.json` | `state`, `model` | models.py:201, :207 |
| | question name `urgency` | models.py:167 (`ValidationError.loc` example `["body","questions","urgency","score","criteria"]`) |
| | `instructions` | models.py:141 (`ScoreQuestion.instructions` example) |
| | `criteria` | models.py:145 (`ScoreQuestion.criteria` example) |
| `response_noul.json` | `model` | models.py:223 |
| | `answers` | models.py:229 (`SystemOneResponse.answers` example) |
| | `usage` | models.py:235 (`SystemOneResponse.usage` example) |
| `response_choice.json` | `model`, `usage` | models.py:223, :235 |
| | answer `tone` | `ChoiceAnswer` examples: `choice` models.py:16, `confidence` :22, `probabilities` :28 |
| `response_score.json` | `model`, `usage` | models.py:223, :235 |
| | answer `urgency` | `ScoreAnswer` examples: `score` models.py:112, `confidence` :118, `legend` :124, `probabilities` :130 |
| `request_workflowevals.json` | overall shape | WorkflowEvals `core/question_types.py:18-24` (`question_record`: every question is sent as `{type, instructions, criteria}` with explicit nulls) and `core/clients.py:98-103`; `model` from `core/providers.py:33` |
| | `state` (a JSON object) | synthetic; WorkflowEvals passes a `dict` state (`evals/customer_service/workflow.py:41`) |
| | `desired_outcome`, `urgency`, `requests_human` | WorkflowEvals `evals/customer_service/questions.py:24-32`, `:44-52`, `:74-79` |
| | `category` (null descriptions) | SDK `README.md` quickstart (sdist `README.md:23-27`) |
| | `bare`, `half` | synthetic: what `question_record` emits for `Noul()` and `Noul(instructions="Spam?", criteria={"true": "ads"})`, captured from the real SDK's request bytes |

The SDK sdist ships no tests and no JSON samples. The OpenAPI examples above are its only
canonical samples. `ScoreAnswer`'s example is self-consistent: 0·0.1 + 1·0.1 + 2·0.8 = 1.7.
