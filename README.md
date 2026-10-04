# yev

`yev` builds, trains, evaluates and serves **yev0-4b**, an open "System One" decision model fine-tuned from
[Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base).

A System One model answers a typed decision question in one forward pass and returns calibrated probabilities. The question
is one of three types:

- **Choice:** 2–10 options.
- **Noul:** yes or no.
- **Score:** an ordered scale.

The model reads the question as a JSON prompt and answers with a single option letter. Probabilities come from the letter
logits.

- **Model:** [choyiny/yev0-4b](https://huggingface.co/choyiny/yev0-4b). The model card has results, training data and
  limitations.
- **Licence:** Apache-2.0 (see `LICENSE`). Third-party datasets keep their own licences; see
  `release/yev0-4b/NOTICE`.

## Results (DecideBench test, 400 items)

| Model | Accuracy | Pair accuracy | Zero-shot accuracy |
|---|---:|---:|---:|
| **yev0-4b** | **94.5** | **89.0** | **95.0** |
| imajev-4b | 95.0 | 90.5 | — |
| TEV | 92.75 | 86.0 | 90.0 |

yev0-4b beats TEV in both modes and ties imajev-4b within the 95% confidence interval. It is weaker on long, multi-step
workflows: 51.5% on TypeSafe WorkflowEvals, against Jev's 67.8%. A Stage 1 model trained on long-context and many-option
data is in progress. The model card has the full comparison, including the benchmarks where yev0-4b does not lead.

## Install

```bash
uv sync                                   # data pipeline and tests
uv sync --extra train --extra serve       # training, evaluation and serving (torch, transformers, peft, fastapi)
```

## Serve the model

```bash
uv run python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('choyiny/yev0-4b', 'calibration.json', local_dir='.'))"
uv run --extra train --extra serve yev serve \
  --model choyiny/yev0-4b --calibration calibration.json --port 8000
```

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
| Train / dev / calibration mix | `yev mix --recipe recipes/stage0.json` | `src/yev/mix.py` |
| LoRA training | `yev train --config configs/stage0/lc100.json` | `src/yev/train/` |
| Calibration and evaluation | `yev calibrate`, `yev eval` | `src/yev/train/` |
| External benchmarks | `yev bench build / score / overlap` | `src/yev/bench/` |
| Serving | `yev serve` | `src/yev/serve/` |

Design documents and implementation plans are in `docs/superpowers/`. `scripts/aws/` runs training on a single cloud GPU.
It reads the host and SSH key from the `YEV_TRAIN_HOST` and `YEV_TRAIN_KEY` environment variables.

## Contamination

DecideBench test items are never trained on. Every training row goes through:

- a canary check;
- an 8-gram overlap filter against DecideBench;
- an embedding-similarity filter against DecideBench.

The Stage 1 pipeline extends the filter to every benchmark we report.

## Tests

```bash
uv run --extra train --extra serve pytest -q
```
