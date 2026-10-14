# Serving backends: vLLM and SGLang next to transformers

Status: draft for review, 2026-10-14. Builds on `2026-10-02-plan-4-systemone-serving.md` (transformers engine, prefix
KV cache) and `docs/serving.md`.

## Goal

A head-to-head comparison of three backends behind the same `/v1/systemone` and `/v1/chat/completions` API:
transformers (today), vLLM and SGLang. All three are kept. The deliverable is a published measurement of parity,
single-request latency and throughput under concurrent load, on the same weights and GPU.

## Decisions already made by the owner

- **Purpose:** comparison, not a production switch. No backend becomes the default in this work.
- **Parity bar:** Stage 1c dev rows only. No benchmark reruns through the new backends.
- **Workload:** latency grid plus concurrent replay of real dev requests.
- **Architecture:** sidecar engines (approach A), each in its own virtualenv, behind our API.

## Architecture

```
client ─► yev serve (FastAPI, contract unchanged)
            │  mapping.py: request → chat rows → token ids (our tokenizer + chat template)
            │  Backend.letter_logprobs(prompts, letters)
            │  calibration temperature → softmax → answers
            ▼
   transformers (in process)  |  vllm sidecar  |  sglang sidecar   (own venv, merged weights, port 8100)
```

### Backend interface

```python
class Backend(Protocol):
    def letter_logprobs(self, prompts: list[list[int]], letters: list[list[int]]) -> list[list[float]]: ...
```

`prompts[i]` is the token ids of question i. `letters[i]` is the token ids of its option letters, in option order.
The result is one score per letter. Scores may be raw logits or log-probs over any token set containing the letters;
each differs from the raw logits by a per-question constant, so `softmax(score / T)` is the same and the existing
calibration file applies unchanged.

- **transformers** moves today's `Engine._logits` / prefix-cache path behind the interface with no behaviour
  change.
- The engine keeps prompt building, length checks, temperatures and answer shaping. Only the logits source changes.
- Prompts go to the sidecars **as token ids**, never text, so chat-template and tokenizer drift cannot cause parity
  error.

### vLLM backend

One `POST /v1/completions` per question: `prompt` = token ids, `max_tokens=1`, `allowed_token_ids` = letters,
`logprobs` = number of letters, `temperature=0`. The server is started with processed (post-mask) log-probs so every
allowed letter is returned. All questions of a request are sent concurrently so automatic prefix caching shares the
state.

### SGLang backend

One `POST /generate` per question: `input_ids`, `sampling_params={"max_new_tokens": 1, "temperature": 0}`,
`return_logprob=true`, `token_ids_logprob` = letters. This reads each letter's log-prob directly. The radix cache
shares the prefix.

### Async path

Sidecar backends are async (`httpx.AsyncClient`), and `/v1/systemone` gets an async handler for them, so concurrent
requests reach the engine together and its batching across requests is what gets measured. The transformers backend
keeps its current synchronous path.

### Weights

`yev export-merged --model <adapter> --base <base> --out <dir>` merges the LoRA adapter into bf16 weights and copies
the tokenizer. The default is the Stage 1c adapter, written to `~/models/yev1c-merged` on the box. It is never
uploaded.

### Sidecars

- `scripts/aws/setup_vllm.sh` and `scripts/aws/setup_sglang.sh` build `~/venv-vllm` and `~/venv-sglang` with pinned
  versions (the newest stable releases at setup time, recorded in the report).
- `scripts/aws/serve_engine.sh vllm|sglang <model_dir>` starts the engine on port 8100 with prefix caching on and
  the flags written to a log.
- `yev serve --backend vllm|sglang --engine-url http://127.0.0.1:8100` starts our API in front of it.

### Errors

If the engine is unreachable, times out, or returns fewer letter scores than options, the request returns a 503
carrying the engine's message. There is never a silent fallback to another backend.

## Evaluation

### 1. Feasibility gate (first GPU step, about 1 h)

Per engine: loads the merged model; answers 20 dev questions; prefix caching is active with the Gated DeltaNet
layers; the log-prob mode returns every letter. If an engine cannot run Qwen3.5 at its newest stable release, the
gate stops for that engine and the report records why. Engines are not forked or patched. If prefix caching does not
work on the hybrid layers, the engine is benchmarked without it and flagged.

### 2. Parity

`scripts/serve_parity.py --backend …` runs all 3,963 Stage 1c dev rows through each backend and compares against
transformers. Bar: argmax agreement ≥ 99.5% and mean |dp| ≤ 0.01. Also reported: max |dp| and accuracy per backend.
A backend below the bar keeps its speed numbers, labelled "not at parity".

### 3. Speed (`scripts/bench_backends.py`)

- **Latency grid:** state sizes 300 / 2k / 8k tokens × 1 / 4 / 16 questions; 2 warm-up and 10 timed requests per cell;
  p50 and p95.
- **Throughput:** 500 real dev requests (dev rows regrouped by state into multi-question requests) replayed at
  concurrency 1, 4, 16 and 64; requests/s, questions/s, p50/p95 latency, peak GPU memory.
- **Fairness:** same merged weights, same L40S, bf16, no quantisation, no speculative decoding, each engine with its
  documented defaults plus prefix caching. Flags and versions are recorded. The transformers backend serves one
  request at a time, and the report says so rather than adding batching to it for this comparison.

### Outputs

- JSON results in `data/bench_results/backends/` (git-ignored, like other results).
- `docs/serving-backends.md`: tables, versions, flags, caveats.

## Testing (laptop, no GPU)

- The transformers backend gives identical outputs through the new interface (existing serve tests keep passing).
- Sidecar backends against stub HTTP servers returning fixed log-probs: request shape (token ids, letters,
  `max_tokens`), softmax equivalence with raw logits, missing letters → 503, engine down → 503, and concurrent
  requests overlap rather than serialise.

## Cost and scope

- GPU: about 4–6 h on the L40S (~$8–12), run after the Stage 1c benchmarks.
- Out of scope: quantisation, multi-GPU, any change to training, making a sidecar the default, publishing weights.
