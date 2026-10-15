# Serving backends: transformers, vLLM and SGLang

`yev serve` answers `/v1/systemone` in one of three ways. All three read the same thing: the logits of the option
letters for the next token, with the same prompt token ids and the same per-type calibration. Only the process that
runs the model changes.

| Backend | How it runs | Weights |
|---|---|---|
| `transformers` (default) | In process, one request at a time; the questions of a request share the state's KV cache | LoRA adapter + base |
| `vllm` | Sidecar engine on `127.0.0.1:8100`; one 1-token completion per question, restricted to its letters | Merged (`yev export-merged`) |
| `sglang` | Sidecar engine on `127.0.0.1:8100`; one 1-token generation per question, scored on its letters | Merged (`yev export-merged`) |

The numbers below are for the Stage 1c model (`yev1c`, Qwen3.5-4B-Base + LoRA r64) on one L40S 48 GB (AWS
g6e.xlarge), 2026-10-15, with vLLM 0.31.0 and SGLang 0.5.21 (torch 2.13.0). Raw results are in
`data/bench_results/backends/final/` (git-ignored).

## Summary

| | transformers | vLLM bf16 | SGLang bf16 | vLLM FP8 | SGLang FP8 |
|---|---|---|---|---|---|
| Parity with transformers (3,963 dev rows) | reference | ✅ 99.72% / 0.0025 | ✅ 99.80% / 0.0025 | ✅ 99.57% / 0.0060 | ❌ 99.32% / 0.0066 |
| Dev accuracy | 95.18% | 95.28% | 95.28% | 95.31% | 95.03% |
| Latency, 1 question, ~2k-token state (p50) | 0.26 s | 0.14 s | 0.12 s | 0.13 s | 0.09 s |
| Latency, 1 question, ~8k-token state (p50) | 1.19 s | 0.47 s | 0.44 s | 0.41 s | 0.36 s |
| Throughput, dev replay, 64 in flight | 6.2 req/s | 21.3 req/s | 19.8 req/s | 26.0 req/s | 26.1 req/s |
| p95 latency at 16 in flight (dev replay) | 4.06 s | 1.34 s | 1.58 s | 1.04 s | 1.44 s |
| Startup to first answer | 55 s | 60 s | 41 s | 60 s | 41 s |
| Extra setup | none | own venv, merged weights | own venv, merged weights | as vLLM | as SGLang |

Parity is argmax agreement / mean |dp|, where |dp| is a row's largest per-letter probability difference; the bar is
≥ 99.5% and ≤ 0.01.

**Recommendation.** Serve with **vLLM bf16**. It matches transformers within the bar, takes 3.4× the throughput and
cuts single-request latency by 1.9–2.5×, and it handles requests with several questions better than SGLang (below).
SGLang bf16 is equally exact and starts faster, with a slightly lower ceiling. FP8 adds about 22% more throughput on
vLLM and stays (just) within the bar; SGLang FP8 does not. transformers stays the reference and the zero-setup option.

## Correctness

Both engines were scored on the 3,963 Stage 1c dev rows and compared with the transformers readout of the same rows.

| | Argmax agreement | Mean \|dp\| | Max \|dp\| | Accuracy |
|---|---|---|---|---|
| vLLM bf16 | 99.72% | 0.0025 | 0.26 | 95.28% |
| SGLang bf16 | 99.80% | 0.0025 | 0.38 | 95.28% |
| vLLM FP8 | 99.57% | 0.0060 | 0.41 | 95.31% |
| SGLang FP8 | 99.32% | 0.0066 | 0.58 | 95.03% |
| transformers | — | — | — | 95.18% |

At bf16 the remaining differences are kernel numerics (different attention and Gated DeltaNet kernels, different
reduction orders); the rows that flip are close calls, and accuracy moves by 0.1 pt either way. FP8 (`--quantization
fp8`, dynamic per-tensor) roughly doubles the probability drift.

## Latency: one request at a time

`scripts/bench_backends.py grid`: synthetic support-ticket states of about 300, 2,000 and 8,000 tokens with 1, 4 or 16
questions, 10 timed requests per cell after 2 warm-ups. Every request has a state of its own, so no cell reuses
another's prefix cache. p50 seconds (p95 is within 5% everywhere):

| State | Questions | transformers | vLLM bf16 | SGLang bf16 | vLLM FP8 | SGLang FP8 |
|---|---|---|---|---|---|---|
| ~300 | 1 | 0.110 | 0.041 | 0.047 | 0.041 | 0.040 |
| ~300 | 4 | 0.213 | 0.118 | 0.120 | 0.104 | 0.097 |
| ~300 | 16 | 0.355 | 0.412 | 0.309 | 0.354 | 0.262 |
| ~2k | 1 | 0.262 | 0.139 | 0.119 | 0.125 | 0.088 |
| ~2k | 4 | 0.356 | 0.173 | 0.353 | 0.154 | 0.282 |
| ~2k | 16 | 0.542 | 0.317 | 0.594 | 0.283 | 0.491 |
| ~8k | 1 | 1.185 | 0.473 | 0.442 | 0.413 | 0.363 |
| ~8k | 4 | 1.307 | 0.553 | 0.559 | 0.471 | 0.454 |
| ~8k | 16 | 1.659 | 0.853 | 0.931 | 0.748 | 0.795 |

- Both engines are 2–3× faster than transformers on long states.
- The questions of a request reach the engine together. vLLM lets them share the state's prefill sooner than SGLang:
  at ~2k tokens and 4 questions, SGLang takes 3× its 1-question time and vLLM 1.2×.
- With 16 short questions transformers is competitive, because it reads the state once and batches the questions.

## Throughput: dev replay

`scripts/bench_backends.py replay`: Stage 1c dev rows grouped by state into requests (about 1.1 questions each), 500
requests per level, a separate set of requests at each concurrency level.

| In flight | transformers | vLLM bf16 | SGLang bf16 | vLLM FP8 | SGLang FP8 |
|---|---|---|---|---|---|
| 1 | 6.2 req/s (p95 0.53 s) | 12.8 (0.25 s) | 13.4 (0.25 s) | 13.8 (0.20 s) | 16.0 (0.20 s) |
| 4 | 6.2 (1.71 s) | 19.2 (0.59 s) | 18.1 (0.71 s) | 22.3 (0.47 s) | 22.4 (0.56 s) |
| 16 | 6.1 (4.06 s) | 21.7 (1.34 s) | 18.6 (1.58 s) | 26.7 (1.04 s) | 23.2 (1.44 s) |
| 64 | 6.2 (13.41 s) | 21.3 (3.89 s) | 19.8 (4.42 s) | 26.0 (3.25 s) | 26.1 (3.12 s) |

transformers serves one request at a time, so its throughput is flat and its latency grows with the queue. Both
engines level off at about 20 requests/s at bf16, and that ceiling is the GPU, not `yev serve`:

- Tokenizing a dev row costs 1.8 ms, so the server's own work is small against 50 ms per request.
- Sent pre-tokenized prompts directly, with no `yev serve` in front, each engine peaks at 25–27 rows/s, which is
  about 21k prefill tokens/s for these ~1,000-token rows.

## What was tuned

A sweep on 650 dev rows with distinct states, sent directly to each engine (`data/bench_results/backends/tune/`).
Rows/s by number in flight:

| Engine | Setting | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|
| vLLM | default | 11.6 | 18.4 | 26.8 | 24.3 |
| vLLM | `--max-num-batched-tokens 16384` | 11.7 | 18.4 | 25.1 | 23.2 |
| vLLM | `--quantization fp8` | 12.2 | 22.6 | 33.1 | 30.0 |
| SGLang | default | 12.6 | 18.1 | 27.1 | 25.4 |
| SGLang | `--chunked-prefill-size 16384` | 12.5 | 17.9 | 25.0 | 23.3 |
| SGLang | `--quantization fp8` | 17.3 | 24.5 | 34.7 | 32.2 |

Larger prefill batches don't help: the GPU is already compute-bound on prefill. FP8 is the only setting that moves
the ceiling, and it costs probability drift (see Correctness). The defaults in `scripts/aws/serve_engine.sh` are the
bf16 defaults; set `ENGINE_ARGS="--quantization fp8"` to try FP8.

## Running a sidecar

```bash
bash scripts/aws/setup_vllm.sh                      # or setup_sglang.sh: ~/venv-vllm, with python3.12-dev
yev export-merged --model runs/stage1c/final --base /models/Qwen3.5-4B-Base --out /models/yev1c-merged
bash scripts/aws/serve_engine.sh vllm /models/yev1c-merged   # engine on 127.0.0.1:8100
yev serve --backend vllm --model /models/yev1c-merged --calibration runs/stage1c/calibration.json
```

`scripts/backend_parity.py score|compare` checks a backend against the transformers readout, and
`scripts/bench_backends.py grid|replay` measures it.

## Caveats

- **GPU memory is not comparable.** The engines pre-allocate 85% of the card for KV and state caches (about 39–42 GB
  in use whatever the load); transformers used 17.9 GB at most.
- **Merged weights must use text-only names.** Qwen3.5's checkpoint declares `Qwen3_5ForCausalLM` but saves its
  weights under the multimodal `model.language_model.` prefix. vLLM and transformers remap that; SGLang 0.5.21
  silently loads none of those weights and serves NaN logits (it answers `!!!!`). `yev export-merged` renames them.
- **The engines need Python headers and their venv on PATH.** Triton builds a C helper against `Python.h` at startup
  (the DLAMI ships without `python3.12-dev`) and SGLang JIT-builds kernels with `ninja` from its venv.
- **One GPU, one model, one day.** Other cards (FP8 is native on L40S, H100 and newer), other engine versions or much
  longer states would give other numbers.
