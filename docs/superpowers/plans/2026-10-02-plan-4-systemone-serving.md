# Plan 4: System One serving (`/v1/systemone` + chat) and WorkflowEvals run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve yev-4b behind TypeSafe's `POST /v1/systemone` contract and an OpenAI-compatible `POST /v1/chat/completions`, using exactly the letter-logit readout we evaluated with. Then score it on TypeSafe WorkflowEvals through the official runner.

**Architecture:** A new `jeff.serve` package with five pieces:
- `contract`: pydantic models of the TypeSafe wire format, pinned to `typesafe-sdk==0.7.2`.
- `mapping`: System One question → our chat row → answer.
- `engine`: model, adapter and calibration loading; batched letter logits for up to 26 letters.
- `app`: FastAPI with `/v1/systemone`, `/v1/chat/completions`, `/v1/models`, `/health`.
- `jeff serve` CLI.

All of it is CPU-testable with the existing tiny-model and FakeTok fixtures. Task 6 is a controller runbook on the AWS box: restart, serve, run WorkflowEvals, stop.

**Tech Stack:** Python ≥ 3.11, FastAPI, uvicorn, pydantic v2, torch, transformers ≥ 5, peft. The `serve` optional extra.

**Spec:** `docs/superpowers/specs/2026-09-29-jeff-4b-system-one-design.md` §6 (readout and serving), §7 (evaluation).

## Global Constraints

- The readout is exactly the evaluated one:
  - TEV system prompt and JSON user turn from `jeff.format`;
  - `apply_chat_template(..., add_generation_prompt=True, tokenize=True, enable_thinking=False)`;
  - right-padding;
  - letter logits at each row's own last position;
  - softmax over the valid letters only, divided by the question type's temperature from `calibration.json`.
- One forward pass per question; no generated text for `/v1/systemone`.
- Letters A–Z. More than 26 options is refused (HTTP 422, `{"error": {"type": "unsupported", "reason": "too many options (N > 26)"}}`), never truncated or filtered.
- Rows longer than `max_len` are refused (422 `unsupported`, reason `state too long`), never truncated.
- No upload anywhere. The box's host, key and instance id never enter the repo (scripts read `JEFF_TRAIN_HOST` and `JEFF_TRAIN_KEY`).
- Probabilities in every answer are finite and sum to 1 within 1e-6. The chosen key is always one of the options.

## Rulings (deviations from the spec)

1. **Transformers backend now, vLLM later.** Spec §6 names vLLM. Qwen3.5's hybrid Gated-DeltaNet architecture with LoRA and letter-restricted logits isn't verified on vLLM, and transformers reproduces our evaluated readout bit for bit. The state prefix is not KV-cached in this plan; the questions of one request are batched together instead. Prefix caching and vLLM become a later plan if latency matters.
2. **Up to 26 letters.** yev-4b was trained on at most 6 options. 7–26 are served (WorkflowEvals has up to 9) but are out of distribution; `/v1/systemone` responses carry no warning, and the README documents it.
3. **Score questions** use the ordered levels as letters in scale order. `score` = the expected level (probability-weighted mean of the numeric levels, rounded to the nearest level for the categorical answer; both reported per the contract).

## Review Focus

1. **Noul polarity:** `criteria` for a Noul question must map so that P(true) is read from the "true" option, never swapped. Test: `test_noul_true_probability_is_true_option`.
2. **Probability normalisation and key echo for Choice:** Test: `test_choice_probs_sum_to_one_and_keys_echo_criteria`.
3. **Request with 27 options:** returns 422 `unsupported`, and the other questions in the same request are not partially answered. Test: `test_too_many_options_refuses_whole_request`.
4. **Batch invariance:** a question's probabilities don't change with the other questions in the request. Test: `test_answers_independent_of_batch_mates`.
5. **Temperatures:** a type missing from `calibration.json` uses T = 1.0. Test: `test_missing_temperature_defaults_to_one`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/jeff/serve/__init__.py` | package |
| `src/jeff/serve/contract.py` | pydantic request/response models for `/v1/systemone` and `/v1/chat/completions` |
| `src/jeff/serve/mapping.py` | `to_rows(state, questions)` and `to_answers(questions, letter_probs)` |
| `src/jeff/serve/engine.py` | `Engine(model_dir, base, calibration, max_len, batch_tokens)` with `.answer(state, questions)` and `.chat(messages)` |
| `src/jeff/serve/app.py` | FastAPI app factory `create_app(engine, model_name)` |
| `src/jeff/train/data.py` | generalise `letter_token_ids(tokenizer, n=6)` |
| `src/jeff/train/infer.py` | `letter_logits(..., n_letters=6)` |
| `src/jeff/cli.py` | `jeff serve` |
| `scripts/aws/workflowevals.sh` | box runbook helper |
| `tests/serve/test_*.py` | tests |
| `pyproject.toml` | `serve` extra: `fastapi>=0.115`, `uvicorn>=0.30`, `pydantic>=2.7` |

---

### Task 1: Contract models pinned to the TypeSafe SDK

**Files:**
- Create: `src/jeff/serve/__init__.py`, `src/jeff/serve/contract.py`, `tests/serve/__init__.py`, `tests/serve/test_contract.py`, `tests/serve/fixtures/` (JSON samples)

**Interfaces:**
- Produces: pydantic v2 models.
  - `QuestionSpec`: `type: Literal["noul","choice","score"]`, `instructions: str`, `criteria`.
  - `SystemOneRequest`: `model: str | None`, `state: str`, `questions: dict[str, QuestionSpec]`.
  - Answer models `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`.
  - `SystemOneResponse`: `model`, `answers: dict[str, …]`, `usage`.
  - `ChatRequest` / `ChatResponse`: OpenAI-compatible, minimal.

- [ ] **Step 1: Pin the wire format from the source.**
  1. `pip download typesafe-sdk==0.7.2 --no-deps` into a scratch dir and unpack it.
  2. Read its request/response types for `/v1/systemone`: the exact `criteria` shapes per type (Choice: key → description? Noul: true/false descriptions? Score: levels and legend?), and the answer shapes. The research notes say:
     - noul: `{noul: p}`
     - choice: `{choice, confidence, probabilities}`
     - score: `{score, confidence, legend, probabilities{level: p}}`
     - plus a `usage` block.

     Confirm or correct them.
  3. Also read `decision_index/engines/http.py` from https://github.com/apolinario/decision-index (its `http` engine posts `{model, state, questions}`).
  4. Save 3 request and 3 response samples (one per type) **copied from the SDK's own docs, tests or types** as `tests/serve/fixtures/*.json`, with a `SOURCE.md` naming the file and line each came from.
- [ ] **Step 2: Write failing tests.** Each fixture request parses into `SystemOneRequest`. Each fixture response round-trips through `SystemOneResponse` unchanged (`model_dump(exclude_none=True)` equals the fixture). Unknown fields are rejected on requests and ignored on responses only if the SDK does so.
- [ ] **Step 3: Implement `contract.py`** to make them pass.
- [ ] **Step 4:** Run `uv run pytest tests/serve -q`, then the full suite.
- [ ] **Step 5: Commit** `feat(serve): TypeSafe /v1/systemone contract models pinned to typesafe-sdk 0.7.2`.

---

### Task 2: Mapping (question ↔ our chat row ↔ answer)

**Files:**
- Create: `src/jeff/serve/mapping.py`, `tests/serve/test_mapping.py`

**Interfaces:**
- Consumes: `jeff.format.SYSTEM_PROMPT`, `user_turn` (or the same JSON layout), `contract` models.
- Produces:
  - `MAX_LETTERS = 26`.
  - `class Unsupported(Exception)` with `reason`.
  - `to_rows(state: str, questions: dict[str, QuestionSpec]) -> list[dict]`. One chat row per question, in request order. Each row is `{name, type, messages, letters: {letter: key}}`, with keys in the criteria's own order; score levels go in ascending scale order. It raises `Unsupported` before building any row if any question has more than 26 options.
  - `to_answers(questions, rows, letter_probs: list[list[float]]) -> dict[str, answer]`. Builds per-type answers from per-row probabilities, already temperature-scaled and normalised over each row's letters. Noul gives P(true). Choice gives the argmax key, its probability as confidence, and the full map. Score gives the expected level, its confidence (probability of the argmax level), the legend and probabilities.

- [ ] **Step 1: Write failing tests:**
  - `test_noul_true_probability_is_true_option`: a Noul with true/false criteria and probs `[0.8, 0.2]` in the row's letter order gives `noul == 0.8` when "true" is letter A. Swap the criteria order and the answer stays tied to the "true" key.
  - `test_choice_probs_sum_to_one_and_keys_echo_criteria`.
  - `test_score_expected_level`: levels 1..5, probs `[0, 0, 0.5, 0.5, 0]` give score 3.5 (or per the contract's rounding) and confidence 0.5.
  - `test_too_many_options_refuses_whole_request`: one question with 27 options in a request of three raises `Unsupported` and builds no rows.
  - `test_user_turn_matches_training_format`: the final user turn of a built row equals `jeff.format.user_turn` output for the equivalent Decision.
- [ ] **Step 2: Run** them and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and they pass; run the full suite.
- [ ] **Step 5: Commit** `feat(serve): map System One questions to chat rows and letter probabilities to answers`.

---

### Task 3: Engine (model, calibration, N-letter logits)

**Files:**
- Modify: `src/jeff/train/data.py` (`letter_token_ids(tokenizer, n=N_MAX)`), `src/jeff/train/infer.py` (`letter_logits(..., n_letters=N_MAX)`)
- Create: `src/jeff/serve/engine.py`, `tests/serve/test_engine.py`

**Interfaces:**
- Consumes: `jeff.train.infer.load`, `letter_logits`, `jeff.train.readout.probs`, `mapping`.
- Produces: `Engine(model_dir: str, base: str | None, calibration: str | None, max_len: int = 16384, batch_tokens: int = 16384, *, model=None, tokenizer=None)`. Tests inject `model` and `tokenizer`.
  - `.answer(state, questions) -> (answers, usage)`, where `usage = {"questions": n, "input_tokens": total}`.
  - `.chat(messages) -> letter`: the argmax over the letters present in the final user turn's options, which must be our JSON format; anything else raises `Unsupported`.
  - Temperatures from `calibration.json["temperatures"]`, defaulting to 1.0 per type.

- [ ] **Step 1: Write failing tests** (tiny model, FakeTok with letters A–Z single-token; extend the fixture if needed):
  - `test_answers_independent_of_batch_mates`
  - `test_missing_temperature_defaults_to_one`
  - `test_overlong_state_refused`
  - `test_letter_logits_default_unchanged`: `n_letters` defaults to 6, so the training and eval paths are bit-identical to before.
- [ ] **Step 2: Run** them and watch them fail.
- [ ] **Step 3: Implement.** `letter_token_ids(tokenizer, n)` asserts single, distinct tokens for the first n letters. Inference passes `n_letters = max letters in the batch`, and the softmax uses only each row's own letters.
- [ ] **Step 4: Run** the tests and they pass; run the full suite.
- [ ] **Step 5: Commit** `feat(serve): engine with calibrated N-letter readout`.

---

### Task 4: FastAPI app and `jeff serve`

**Files:**
- Create: `src/jeff/serve/app.py`, `tests/serve/test_app.py`
- Modify: `src/jeff/cli.py`, `pyproject.toml` (the `serve` extra)

**Interfaces:**
- `create_app(engine, model_name: str = "yev-4b") -> FastAPI`, with routes:
  - `POST /v1/systemone` → `SystemOneResponse`. 422 with an `unsupported` error body on `Unsupported`, and 422 on validation errors.
  - `POST /v1/chat/completions` → OpenAI-compatible response with a one-letter `content`. If `logprobs: true`, it includes the letter probabilities as `top_logprobs`.
  - `GET /v1/models`, `GET /health`.
  - The request's `model` field is accepted and echoed but not used to choose a model, since one model is loaded.
- CLI: `jeff serve --model <adapter or full dir> --base <base> --calibration <json> --host 127.0.0.1 --port 8000 [--max-len 16384]`.

- [ ] **Step 1: Write failing tests** with `fastapi.testclient.TestClient` and an engine built on the tiny model:
  - every fixture request from Task 1 gets a 200 whose body validates as `SystemOneResponse`;
  - a 27-option request gets 422 `unsupported`;
  - chat returns exactly one letter in `A..` for a JSON user turn;
  - `/health` returns 200.
- [ ] **Step 2: Run** them and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and they pass; run the full suite.
- [ ] **Step 5: Commit** `feat(serve): FastAPI /v1/systemone and /v1/chat/completions with jeff serve`.

---

### Task 5: WorkflowEvals helper

**Files:**
- Create: `scripts/aws/workflowevals.sh`, `docs/serving.md`
- Test: extend `tests/test_aws_scripts.py`, so the new script also requires the env vars and contains no IP or `.pem`.

**Interfaces:**
- `workflowevals.sh` runs on the box, with the server already running at `127.0.0.1:8000`. It:
  1. clones the WorkflowEvals repo at a pinned commit, which is recorded;
  2. runs `uv sync --locked`;
  3. runs `uv run python run.py <workflow> --model typesafe:jev-1.13.0 --base-url http://127.0.0.1:8000 --name yev-4b` for each of the four workflows, with `TYPESAFE_API_KEY=local` set if the SDK needs a key;
  4. collects the scores files into `~/workflowevals/yev-4b/`.

  The runner rejects every TypeSafe model name except `jev-1.13.0`. We pass that name and our server ignores it. The README states this explicitly.
- `docs/serving.md` covers:
  - how to run `jeff serve`, with request and response examples for all three types and the chat endpoint;
  - the 26-letter limit and the out-of-distribution note for more than 6 options;
  - how WorkflowEvals and the Decision Index kit's `http` engine call it.

- [ ] Steps: write the test, implement, run the suite, commit `feat(serve): WorkflowEvals runner helper and serving docs`.

---

### Task 6: Run on the box (controller runbook)

- [ ] **Step 1:** Start the instance (`aws ec2 start-instances`). Read the new public IP. If ssh fails, check the laptop IP against the security group. Export `JEFF_TRAIN_HOST` and `JEFF_TRAIN_KEY`.
- [ ] **Step 2:** `scripts/aws/sync.sh`, then `scripts/aws/run.sh pip install -e '.[train,serve]'`. Copy `~/runs/lc100/final` and `calibration.json` back to the laptop under `release/yev-4b/weights/` (git-ignored), and record the train file's sha256 and the box's code commit for the release checklist.
- [ ] **Step 3: Smoke-test the server.** Start `jeff serve --model ~/runs/lc100/final --base ~/models/Qwen3.5-4B-Base --calibration ~/runs/lc100/calibration.json` under nohup and setsid, then send the Task 1 fixtures with curl. Every response must validate. Then run 20 DecideBench *dev* rows through `/v1/chat/completions`: their letters must equal `jeff eval`'s argmax for the same rows. This is a parity check, so the dev rows are fine to use; never the test set.
- [ ] **Step 4:** Run `scripts/aws/workflowevals.sh`. Then fetch the scores to `data/bench_results/workflowevals/`.
- [ ] **Step 5:** Stop the instance (`aws ec2 stop-instances`). Report to the user:
  - the WorkflowEvals scores per workflow against Jev (invoice 0.6178, customer service 0.7598, security 0.6167, agent trace 0.7162, overall 67.8%) and the frontier entries;
  - latency per question;
  - the parity result.

---

### Task 7: Prefix (state) KV caching in the Engine

Added 2026-10-02 at the user's request. WorkflowEvals invoice requests are a 6–9k-token state with about 42 questions, and the uncached path re-reads the state once per question (about 45 s per case on the L40S).

**Files:**
- Create: `src/jeff/serve/prefix.py`
- Modify: `src/jeff/serve/engine.py`, `src/jeff/cli.py` (`--no-prefix-cache`, `--min-prefix-tokens`), `docs/serving.md`
- Test: `tests/serve/test_prefix.py`

**Interfaces:**
- `common_prefix_len(seqs: list[list[int]]) -> int` returns the longest common leading token run of all sequences. It is capped at `min(len(s)) - 1`, so every row keeps at least one suffix token: the answer position must be computed inside the suffix pass.
- `prefix_letter_logits(model, prefix_ids, suffixes, letter_ids, batch_tokens) -> list[list[float]]`:
  1. One forward pass over `prefix_ids` with `use_cache=True`.
  2. For each batch of suffixes: deep-copy or expand the cache to the batch size. For the hybrid Qwen3.5 cache this covers both the KV of the full-attention layers and the conv/recurrent states of the Gated DeltaNet layers.
  3. Right-pad the suffixes, set `attention_mask` = ones over the prefix plus the suffix mask, and set `position_ids` (or `cache_position`) to continue from the prefix length.
  4. Gather each row's own last real position, and return the letter logits there.
- `Engine` uses the prefix path when a request has at least 2 questions and the common prefix is at least `min_prefix_tokens` (default 256). Otherwise it uses the existing exact path, unchanged.

**Steps:**
- [ ] **Step 1: Write failing tests.**
  - `common_prefix_len` edge cases: identical sequences, no common prefix, one sequence a prefix of another.
  - **Parity:** `prefix_letter_logits` on the tiny Llama fixture equals the full-sequence `letter_logits` to within 1e-4, for rows of different suffix lengths and letter counts.
  - **The same parity on a tiny randomly initialised `Qwen3_5ForCausalLM`** (hybrid layers, CPU torch fallback), if transformers can build one on CPU. If it can't, say why in the report and mark the test skip-if.
  - **Engine:** answers with the prefix cache on equal answers with it off.
  - **The cache is never mutated across batches:** the second batch's results don't depend on batch order.
- [ ] **Step 2: Run** them and watch them fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the tests and they pass; run the full suite.
- [ ] **Step 5: Commit** `feat(serve): state-prefix KV caching for multi-question requests`.
