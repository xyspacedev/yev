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
| yev0-4b | Qwen3.5-4B-Base | `yev-data-short`: 44,228 rows, 28.2M tokens | [choyiny/yev0-4b](https://huggingface.co/choyiny/yev0-4b) |
| yev1-4b | Qwen3.5-4B-Base | `yev-data-long`: 80,730 rows, 101.5M tokens | [choyiny/yev1-4b](https://huggingface.co/choyiny/yev1-4b) |
| yev0-35b-a3b | Qwen3.6-35B-A3B | `yev-data-short`, the same rows as yev0 | [choyiny/yev0-35b-a3b](https://huggingface.co/choyiny/yev0-35b-a3b) (LoRA adapter) |
| yev2-35b-a3b | Qwen3.6-35B-A3B | `yev-data-long`, the same rows as yev1 | [choyiny/yev2-35b-a3b](https://huggingface.co/choyiny/yev2-35b-a3b), [FP8](https://huggingface.co/choyiny/yev2-35b-a3b-fp8) |

`yev-data-short` is the `stage0` recipe: short decisions with up to 6 options. `yev-data-long` is the `stage1c` recipe:
it adds long documents, up to 10 options, agent traces and policy rules.

`recipes/stage1d.json` is the next data set, not yet trained on: the `stage1c` rows with 80 % of their states
rewritten as interpolated production prompts (see [Prompt templates](#prompt-templates)).

All weights are Apache-2.0 (see `LICENSE`). Third-party datasets keep their own licences; each model repo has a `NOTICE`.

## Better data or a better model?

Two training sets and two base models, every combination trained with the same LoRA recipe:

| | `yev-data-short` (44k rows) | `yev-data-long` (81k rows) |
|---|---|---|
| **Qwen3.5-4B-Base** | yev0-4b | yev1-4b |
| **Qwen3.6-35B-A3B** | yev0-35b-a3b | yev2-35b-a3b |

Qwen3.6-35B-A3B is a mixture-of-experts model with about 3B parameters active per token. Each model trains on its data
set's own recipe (`stage0` or `stage1c`), with only the base model swapped; yev1 and yev2 use the same file (same
sha256).

| Benchmark | yev0-4b | yev1-4b | yev0-35b-a3b | yev2-35b-a3b | Jev (TypeSafe, API) | Open model with a result |
|---|---:|---:|---:|---:|---:|---|
| DecideBench v2 (450), accuracy | 78.2 | 79.3 | 86.0 | **88.0** | 86.2 | 85.3, Clef 27B (self-hosted) |
| DecideBench v2, pair accuracy | 60.0 | 62.7 | 74.7 | **78.2** | 72.9 | 72.4, Clef 27B (self-hosted) |
| DecideBench v1 (400), accuracy | 94.5 | 95.25 | **98.0** | 97.75 | **98.0** | 95.0, imajev-4b |
| JevBench public (231), accuracy | 78.4 | 82.7 | 87.4 | **90.5** | 86.6 (Jev 1.13.0) | 86.2, Imajev-4B |
| DynaBench test (543), F1 | 55.7 | 67.5 | **83.7** | 75.9 | – | 73.1, DynaGuard-8B (published) |

**The answer: the base model, by a wide margin.**

| Benchmark | Better model, on `yev-data-short` | Better model, on `yev-data-long` | Better data, on 4B | Better data, on 35B |
|---|---:|---:|---:|---:|
| DecideBench v2 | +7.8 | +8.7 | +1.1 | +2.0 |
| DecideBench v1 | +3.5 | +2.5 | +0.75 | −0.25 |
| JevBench public | +9.0 | +7.8 | +4.3 | +3.1 |
| DynaBench F1 | +28.0 | +8.4 | +11.8 | −7.8 |

- **The larger base helps on every benchmark, with either data set.** On DecideBench v2 it adds about 8 points either
  way; on DynaBench it adds 28 F1 with the short data.
- **Better data helps a little on top, but not everywhere.** It adds 1–4 points on DecideBench v2 and JevBench with
  either base. On the 4B base it was worth 12 F1 on DynaBench; on the 35B base the long data scores 8 F1 lower there.
- **The best model combines both** on DecideBench v2 and JevBench, but most of yev2's lead over yev1 comes from the base:
  yev0-35b-a3b already beats yev1 on every benchmark here, beats Jev on JevBench and ties it on
  DecideBench v2.

**Caveats.**

- One training run per cell. DecideBench v2's 95 % interval is about ±3 points, so differences of 1–2 points are noise.
- `yev-data-short` has at most 6 options and prompts up to 4,096 tokens. It did not hold the short-data models back on
  DecideBench v2's 100 questions with 8–10 options: every model scored higher on those than on the rest.
- A run on gemma-4-E4B (`yev-data-long`) stayed at chance accuracy after 1,000 steps and was stopped. The likely cause is
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

### Prompt templates

Callers rarely send a bare record. They keep a prompt template (a role, background on the pipeline, a list of
instructions) and interpolate each production record into it. A recipe with a `prompt_template` block makes `yev mix`
rewrite states that way (`src/yev/generators/prompt_template.py`):

```json
"prompt_template": {"rate": 0.8, "tokens": [120, 1000], "max_tokens": 11800,
                    "exclude_families": ["many_options"], "exclude_sources": ["decidebench"]}
```

- **What changes:** only `state`. Rows, splits, worked examples, option order, questions and labels are those of the
  same recipe without the block, so `stage1d` differs from `stage1c` in the wrapper alone.
- **The wrapper:** instruction text of 120–1,000 tokens in one of five styles, then the record with opaque metadata
  (request ID, tenant, prompt version) in one of eight layouts (XML tags, fenced block, JSON, `key: value` and others).
  It tells the reader how to read the input and states no policy, number or default answer.
- **Consistency:** a template is drawn per unit, so contrastive siblings share a wrapper, and a row's worked examples
  are wrapped with the row's own template.
- **Length:** the whole prompt stays within `max_tokens`; the wrapper is shortened to fit, and a row with no room
  stays bare. `mix_report.json` counts templated, excluded and no-room rows and the tokens added.
- `many_options` is excluded because its labels come from a keyword rule over the state. DecideBench dev rows are
  never templated.

`docs/samples/stage1d-prompt-templates.md` has before-and-after examples.

Training configs: `configs/stage0/lc100.json` (yev0), `configs/stage1/stage1c.json` (yev1),
`configs/yev2/yev2-qwen36-35b-a3b.json` (yev2) and `configs/yev2/yev0-35b-a3b.json` (yev0-35b-a3b).
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
