# Serving yev-4b behind `POST /v1/systemone`

`jeff serve` loads one model and answers TypeSafe-style System One requests (Noul, Choice, Score) plus a minimal
OpenAI-style chat endpoint. The wire format follows `typesafe-sdk` 0.7.2; the details are in
`tests/serve/fixtures/WIRE_FORMAT.md`. It needs the `serve` extra (`pip install -e '.[serve]'`) and a GPU for a real model.

## Running it

```bash
jeff serve --model runs/yev-4b/adapter --base /models/Qwen3-4B-Base \
           --calibration runs/yev-4b/calibration.json --model-name yev-4b
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` (required) | | LoRA adapter dir or full model dir |
| `--base` | | Base model path; required for an adapter, and the tokenizer source |
| `--calibration` | none (T = 1) | JSON from `jeff calibrate`: per-type temperatures applied to the letter logits |
| `--host` / `--port` | `127.0.0.1` / `8000` | Bind address |
| `--model-name` | `yev-4b` | Name reported in responses and `/v1/models`. The `model` a client sends is ignored. |
| `--max-len` | `16384` | A request with a longer prompt (in tokens) is refused with 422; prompts are never truncated |
| `--batch-tokens` | `16384` | Token budget per forward batch when a request carries several questions |
| `--no-prefix-cache` | cache on | Re-read the shared state for every question (the exact path; use to rule the cache out) |
| `--min-prefix-tokens` | `256` | Shortest shared prompt prefix, in tokens, that uses the prefix KV cache |

Other endpoints: `GET /health` (`{"status": "ok", "model": ...}`), `GET /v1/models`. No authentication is checked, so any
`Authorization` header is accepted. Bind to loopback or put it behind your own proxy.

## How a question is answered

Each question becomes one chat row in the training format: the state, the instructions and lettered options (A, B, C, ...),
in key order. The model's next-token logits over the option letters, divided by the calibrated temperature and
softmaxed, are the probabilities. Nothing is generated (`output_tokens` is always 0). One forward pass per question; a
request with several questions is batched up to `--batch-tokens`.

**Prefix KV cache.** All questions of a request share the system prompt, the chat header and the state, and
diverge at the question. When a request has at least two questions and their prompts share at least
`--min-prefix-tokens` leading tokens, that prefix is run once and every question continues from a copy of its cache
(the full-attention keys/values and the Gated DeltaNet conv/recurrent states), so a 9k-token invoice with 42 questions
reads the invoice once instead of 42 times. Suffix batches hold at most 16 questions and at most `--batch-tokens`
suffix tokens; the shared prefix is not counted against the budget. For yev-4b the cache costs about 32 KB per
token per question in the batch (8 full-attention layers), so 16 questions over a 9k-token state need about 5 GB on top
of the model. Single-question requests and shorter prefixes use the exact uncached path, unchanged. The two paths agree
to within 1e-4 on the CPU test models; on the GPU the chunked linear-attention kernels can differ slightly, which
`scripts/serve_parity.py` measures on real requests (`--no-prefix-cache` turns the cache off).

- **noul** has two options, yes first and no second, always. `criteria.true` / `criteria.false` are the descriptions
  when given. The answer is `noul` = P(yes).
- **choice** options are the `criteria` keys in the order sent; `probabilities` has every key.
- **score** levels are the `criteria` array in order, served with numeric keys `"0"` .. `"n-1"`. `score` is the expected level
  (sum of level times probability) and can fall between levels. `legend` echoes each level's description.

## Examples

The request bodies are the fixtures in `tests/serve/fixtures/` (TypeSafe's own OpenAPI examples). The responses are real
output of the app on the tiny random test model, so they show the **shape only**; the numbers mean nothing.

### Noul

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I was charged twice. Please help.",
  "model": "jev-latest",
  "questions": {"billing": {"instructions": "Is this message about billing?", "type": "noul"}}}'
```
```json
{"model": "yev-4b", "answers": {"billing": {"type": "noul", "noul": 0.4974607144866185}}, "usage": {"input_tokens": 75, "output_tokens": 0}}
```

### Choice

Request: `state` `{"message": "Please help.", "subject": "Duplicate charge"}`, question `tone` of type `choice` with
`criteria` `{"angry": "An upset or hostile message", "calm": "A neutral or polite message", "excited": "An enthusiastic or eager message"}`.
```json
{"model": "yev-4b", "answers": {"tone": {"type": "choice", "choice": "excited", "confidence": 0.3458524737341633,
  "probabilities": {"angry": 0.3264586564117663, "calm": 0.3276888698540705, "excited": 0.3458524737341633}}},
 "usage": {"input_tokens": 110, "output_tokens": 0}}
```
`choice` is the argmax key and `confidence` its probability.

### Score

Request: question `urgency` of type `score` with `criteria` `["Can wait", "Needs attention this week", "Needs attention today"]`.
```json
{"model": "yev-4b", "answers": {"urgency": {"type": "score", "score": 1.0272961388520339, "confidence": 0.3495379826455441,
  "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
  "probabilities": {"0": 0.32224184379351023, "1": 0.32822017356094574, "2": 0.3495379826455441}}},
 "usage": {"input_tokens": 94, "output_tokens": 0}}
```

Every answer carries `type`, and the answer keys equal the request's question names. Extra top-level request keys are ignored.

## Chat endpoint

`POST /v1/chat/completions` takes `{model?, messages, max_tokens?, temperature?}` and replies with **one letter** as the
assistant content. It expects the training row format: the last user message is the JSON
`{"state", "question", "options": [{"label", "key", "description"}, ...]}`. Any other message shape is a 422 `unsupported`.
Set `"logprobs": true` to get the log-probability of each letter (they sum to 1 after `exp`) in
`choices[0].logprobs.content[0].top_logprobs`, best first. `usage.completion_tokens` is 1. The calibrated temperature of
the inferred type (noul when the keys are yes/no) is applied; `temperature` in the request is ignored.

Shape example (tiny model):
```json
{"id": "chatcmpl-...", "object": "chat.completion", "model": "yev-4b",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "A"}, "finish_reason": "stop",
   "logprobs": {"content": [{"token": "A", "logprob": -0.677, "bytes": [65], "top_logprobs": [
     {"token": "A", "logprob": -0.677, "bytes": [65]}, {"token": "B", "logprob": -0.709, "bytes": [66]}]}]}}],
 "usage": {"prompt_tokens": 39, "completion_tokens": 1, "total_tokens": 40}}
```

## Errors

| Status | Body | When |
|---|---|---|
| 422 | `{"error": {"type": "unsupported", "message": R, "reason": R}}` | The model cannot answer. A whole request is refused before any forward pass. |
| 422 | `{"detail": [{"loc", "msg", "type"}], "error": {"type": "invalid_request", ...}}` | Malformed request (FastAPI validation: no questions, unknown question field, missing `criteria`) |
| 500 | `{"error": {"type": "internal_error", "message": "ValueError: ..."}}` | A bug or a CUDA out-of-memory. Clients retry 5xx. |

Refusal reasons (R) are worded to contain the phrases the Decision Index `http` engine recognises as "unsupported":

- `too many options per choice (27 > 26)`
- `a choice needs at least two options`
- `state too long: longer than the maximum model length (N > max_len tokens)`

The TypeSafe SDK and WorkflowEvals do not retry 422, so a refusal is final for that request. Any other 4xx makes the kit
count an error instead of an unsupported row.

## Limits and out-of-distribution behaviour

- **Options:** served up to 26 (letters A-Z) for choice and score. The model was **trained on at most 6 options**, so
  accuracy with 7 to 26 options is untested and probably degraded.
- **Length:** trained mostly on prompts of 4k tokens or fewer. `--max-len` (default 16384) only bounds memory, not quality.
  Long states such as invoice packets (about 6 to 9k tokens) are out of distribution.
- **Score:** levels are served as the numeric keys `"0"`..`"n-1"`, while most training Score rows used named levels.
  Expect Score probabilities to be less well calibrated than Noul and Choice, and check them on your workflow.
- **Noul** is always served yes-first (A = yes), so a client's `true`/`false` descriptions never reorder it.
- **Speed:** a request's shared state is encoded once (prefix KV cache, above); each question then costs one forward pass
  over its own suffix. The cache lives for one request only: two requests with the same state each encode it.
- **Calibration:** without `--calibration` temperatures are 1. Pass the file produced by `jeff calibrate` for the model.

## Callers

### WorkflowEvals (typesafe-ai/WorkflowEvals, Apache-2.0)

It uses the TypeSafe SDK. Only `typesafe:jev-1.13.0` is accepted as a TypeSafe model name, so pass that and let the
server ignore it. `TYPESAFE_API_KEY` must be set to any non-empty printable ASCII without spaces.

```bash
export TYPESAFE_API_KEY=local
uv run python run.py customer_service --model typesafe:jev-1.13.0 \
    --base-url http://127.0.0.1:8000 --name yev-4b
```

Workflows: `invoice_processing`, `customer_service`, `agent_trace_observability`, `security_incidents`. It sends one request
per workflow node with all that node's questions, a 600 s timeout and up to 5 retries on 5xx. Results go to
`runs/<workflow>/yev-4b/{results.json,scores.json}` in the checkout, and data comes from the HF datasets `typesafe/evalsafe-*`.

On the GPU box, with the server up, `scripts/aws/workflowevals.sh` runs all four sequentially at a pinned WorkflowEvals
commit (`uv sync --locked`), logs each to `~/workflowevals/yev-4b/<workflow>.log`, copies the scores and results there and
writes `run_meta.json` (WorkflowEvals commit, our `GIT_SHA`, wall-clock seconds and exit code per workflow). It exits 3 if
`/health` does not answer. Run it detached with `scripts/aws/run.sh --bg --log workflowevals bash scripts/aws/workflowevals.sh`.
Scores are agreement with frontier-LLM consensus, not human ground truth.

### Decision Index kit `http` engine

```bash
python -m decision_index run --engine http \
    --option base_url=http://127.0.0.1:8000 --option model=yev-4b
```

It posts `{"model", "state", "questions", ...extra}` to `/v1/systemone` with a 600 s timeout, sends
`Authorization: Bearer $DECISION_INDEX_API_KEY` only if that variable is set, and validates that answer keys, types and
choice probabilities (sum within 0.01) match the request. The suite has Choice and Noul only; its `validate` has no Score branch.
