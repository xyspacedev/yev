# yev

`yev` builds, trains, evaluates and serves the **yev** models: open "System One" decision models, and an experiment in
one question: **for a small decision model, does better data or a better base model matter more?**

A System One model answers a typed decision question in one forward pass and returns calibrated probabilities. The question
is one of three types:

- **Choice:** 2–10 options.
- **Noul:** yes or no.
- **Score:** an ordered scale.

The model reads the question as a JSON prompt and answers with a single option letter. Probabilities come from the letter
logits.

## Models

| Model | Base | Training data | Hugging Face |
|---|---|---|---|
| yev0-4b | Qwen3.5-4B-Base | Stage 0: 44,228 rows, 28.2M tokens | [choyiny/yev0-4b](https://huggingface.co/choyiny/yev0-4b) |
| yev1-4b | Qwen3.5-4B-Base | Stage 1c: 80,730 rows, 101.5M tokens | [choyiny/yev1-4b](https://huggingface.co/choyiny/yev1-4b) |
| yev2-35b-a3b | Qwen3.6-35B-A3B | Stage 1c, the same rows as yev1 | [choyiny/yev2-35b-a3b](https://huggingface.co/choyiny/yev2-35b-a3b), [FP8](https://huggingface.co/choyiny/yev2-35b-a3b-fp8) |

All weights are Apache-2.0 (see `LICENSE`). Third-party datasets keep their own licences; each model repo has a `NOTICE`.

## Better data or a better model?

Each step changed one thing:

1. **yev0 → yev1: better data, same model.** Stage 1c added long documents, questions with up to 10 options, agent
   traces and policy rules, and new decision families. The base model and the LoRA recipe stayed the same.
2. **yev1 → yev2: better model, same data.** The same 80,730 training rows (same file, same sha256) and the same
   hyper-parameters, on a larger base: Qwen3.6-35B-A3B, a mixture-of-experts model with about 3B parameters active per
   token.

| Benchmark | yev0-4b | yev1-4b | yev2-35b-a3b | Jev (TypeSafe, API) | Open model with a result |
|---|---:|---:|---:|---:|---|
| DecideBench v2 (450), accuracy | 78.2 | 79.3 | **88.0** | 86.2 | 85.3, Clef 27B (self-hosted) |
| DecideBench v2, pair accuracy | 60.0 | 62.7 | **78.2** | 72.9 | 72.4, Clef 27B (self-hosted) |
| DecideBench v1 (400), accuracy | 94.5 | 95.25 | 97.75 | **98.0** | 95.0, imajev-4b |
| JevBench public (231), accuracy | 78.4 | 82.7 | **90.5** | 86.6 (Jev 1.13.0) | 86.2, Imajev-4B |
| DynaBench test (543), F1 | 55.7 | 67.5 | **75.9** | – | 73.1, DynaGuard-8B (published) |

**What each step did:**

- **Better data** moved the benchmarks closest to what it added, policy rules and long documents: DynaBench +12 F1,
  JevBench +4. It barely moved DecideBench (v2 +1.1, v1 +0.75).
- **A better model** moved every benchmark, including DecideBench, which the data step left nearly flat: DecideBench v2
  +8.7, v1 +2.5, JevBench +7.8, DynaBench +8.

So far, the base model explains more of the gain on held-out decision sets. Better data fixed specific failure modes
that a better model alone may or may not fix.

**The missing cell.** The steps happened in sequence, so the effects are not yet separated. A fourth run, yev0's
Stage 0 data and config on the Qwen3.6-35B-A3B base, completes the 2×2 and shows how much of yev2's gain the larger
base gets on its own. *In progress.*

| | Stage 0 data | Stage 1c data |
|---|---|---|
| Qwen3.5-4B-Base | yev0-4b | yev1-4b |
| Qwen3.6-35B-A3B | *in progress* | yev2-35b-a3b |

**Caveats.**

- One training run per cell. DecideBench v2's 95 % interval is about ±3 points, so differences of 1–2 points are noise.
- Stage 0 reads at most 6 options and trains on prompts up to 4,096 tokens. DecideBench v2 has questions with 8–10
  options and long prompts, so on v2 the data comparison also measures whether the model can read those at all.
- A run on gemma-4-E4B (Stage 1c data) stayed at chance accuracy after 1,000 steps and was stopped. The likely cause is
  the chat template, which the base model does not ship and was borrowed from the instruction-tuned model. It was not
  investigated further, so it says nothing about the question.
- The Jev and open-model columns come from the DecideBench leaderboard and published results, not from runs in this
  repository.

## Cost and speed

DecideBench v2 on one NVIDIA RTX PRO 6000 Blackwell (96 GB), vLLM 0.31.0 behind `yev serve`, 4 requests in flight.
Cost is wall-clock GPU time at $3.36 per hour, divided by decisions.

| Model | GPU memory | p50 | Cost per 1M decisions |
|---|---|---:|---:|
| yev1-4b | one 24 GB GPU | 0.28 s | $71 |
| yev2-35b-a3b (bf16) | one 80 GB GPU | 0.38 s | $95 |
| yev2-35b-a3b-fp8 | one 48 GB GPU | 0.30 s | $78 |
| Jev (TypeSafe API) | – | 0.33 s | $75 |
| Clef 27B, self-hosted | one 96 GB GPU (as measured) | 0.93 s | $216 |

The FP8 checkpoint scores 87.6 on DecideBench v2 (bf16: 88.0) and 89.2 on JevBench (bf16: 90.5).

## Install

```bash
uv sync                                   # data pipeline and tests
uv sync --extra train --extra serve       # training, evaluation and serving (torch, transformers, peft, fastapi)
```

## Serve a model

The model cards have the full commands. vLLM is recommended:

```bash
hf download choyiny/yev1-4b --local-dir yev1-4b
vllm serve ./yev1-4b --host 127.0.0.1 --port 8100 --served-model-name yev --dtype bfloat16 \
  --max-model-len 16385 --enable-prefix-caching --generation-config vllm \
  --logprobs-mode processed_logprobs --max-logprobs 26 --gpu-memory-utilization 0.85 --trust-remote-code
uv run --extra train --extra serve yev serve --backend vllm --model yev1-4b \
  --calibration yev1-4b/calibration.json --model-name yev1-4b
```

yev2 needs `--max-num-seqs 64` on the vLLM command as well. Without `--backend vllm`, `yev serve` runs the model with
transformers.

This serves TypeSafe's `POST /v1/systemone` and an OpenAI-compatible `POST /v1/chat/completions`. A request with several
questions about one long state reuses the state's KV cache across the questions. `docs/serving.md` has request and
response examples, limits and error behaviour.

## Pipeline

| Step | Command | Code |
|---|---|---|
| Public data | `yev build-public` | `src/yev/sources/` |
| Rule-based contrastive clusters | `yev gen-rules --family <name>` | `src/yev/generators/` |
| Opus-written clusters and documents, with blind checks | `yev synth ...` | `src/yev/generators/llm/` |
| Licence, contamination, dedupe and shortcut filters | `yev filter`, `yev shortcut` | `src/yev/filters/` |
| Train / dev / calibration mix | `yev mix --recipe recipes/stage1c.json` | `src/yev/mix.py` |
| LoRA training | `yev train --config configs/stage1/stage1c.json` | `src/yev/train/` |
| Calibration and evaluation | `yev calibrate`, `yev eval` | `src/yev/train/` |
| External benchmarks | `yev bench build / score / overlap` | `src/yev/bench/` |
| Merged weights | `yev export-merged` | `src/yev/train/` |
| Serving | `yev serve` | `src/yev/serve/` |

Training configs: `configs/stage0/lc100.json` (yev0), `configs/stage1/stage1c.json` (yev1),
`configs/yev2/yev2-qwen36-35b-a3b.json` (yev2) and `configs/yev2/yev0data-qwen36-35b-a3b.json` (the fourth cell).
Design documents and implementation plans are in `docs/superpowers/`. `scripts/aws/` runs training on a single cloud
GPU. It reads the host and SSH key from the `YEV_TRAIN_HOST` and `YEV_TRAIN_KEY` environment variables.

## Contamination

DecideBench test items are never trained on. Every training row goes through:

- a canary check;
- an 8-gram overlap filter against DecideBench;
- an embedding-similarity filter against DecideBench.

DecideBench v2 was published after every training set here was frozen.

## Tests

```bash
uv run --extra train --extra serve pytest -q
```
