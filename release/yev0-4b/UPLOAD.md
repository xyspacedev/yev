# yev0-4b: upload layout and checklist

Target repo: `choyiny/yev0-4b` (model). Nothing has been uploaded; this folder is prepared locally only.

The model is the Stage 0 run `lc100` (previously drafted as "yev-4b" in `release/yev-4b/`).

## Repo layout

```
choyiny/yev0-4b/
├── README.md                          # model card (this folder)
├── LICENSE                            # Apache-2.0 full text (this folder)
├── NOTICE                             # base model + dataset licences (this folder)
├── ATTRIBUTION.json                   # skill-atlas per-skill attribution (this folder)
├── calibration.json                   # per-type temperatures (this folder)
├── inference_example.py               # runnable readout (this folder)
├── config.json                        # merged model, from the controller's merge
├── generation_config.json             # merged model, if the merge writes one
├── model.safetensors.index.json       # merged model (or a single model.safetensors)
├── model-0000X-of-0000N.safetensors   # merged model shards
├── tokenizer.json                     # tokenizer, from Qwen/Qwen3.5-4B-Base (unchanged)
├── tokenizer_config.json              # includes the chat template with enable_thinking
├── vocab.json
├── merges.txt
└── adapter/
    ├── adapter_config.json            # base_model_name_or_path = "Qwen/Qwen3.5-4B-Base"
    └── adapter_model.safetensors
```

Local sources (all under `release/*/weights/`, which `.gitignore` excludes; checked with `git check-ignore`):

| Repo path | Local source |
|---|---|
| root model + tokenizer files | `release/yev0-4b/weights/merged/` (controller's `merge_and_unload` output + `tokenizer.save_pretrained`) |
| `adapter/adapter_model.safetensors` | `release/yev-4b/weights/adapter/adapter_model.safetensors` (sha256 `46c97c89457197e5ef2472c2c22f16cb460f29405763240ea847a3f733f3c948`) |
| `adapter/adapter_config.json` | `release/yev-4b/weights/adapter/adapter_config.json`, with `base_model_name_or_path` rewritten |

Do **not** upload from `release/yev-4b/weights/adapter/`: `README.md` (PEFT's auto-generated card) or
`train_config.json` (it holds box paths under `/home/ubuntu/`).

## Before upload

- [ ] Copy the adapter into `release/yev0-4b/weights/adapter/` and set `base_model_name_or_path` to
      `"Qwen/Qwen3.5-4B-Base"`. The local copy currently says `/home/ubuntu/models/Qwen3.5-4B-Base`.
- [ ] Merged weights present in `release/yev0-4b/weights/merged/`, made from the same adapter (sha256 above).
- [ ] Merged `config.json` loads with `AutoModelForCausalLM.from_pretrained` (the base repo's `config.json` declares
      `Qwen3_5ForConditionalGeneration`; the trainer loaded the base with `AutoModelForCausalLM`).
- [ ] No `/home/ubuntu` or other box paths in any uploaded JSON (`grep -r /home/ubuntu`).
- [ ] Tokenizer files saved next to the merged weights; `tokenizer_config.json` keeps the `chat_template` with
      `enable_thinking` (it does in the cached base snapshot `1001bb4d`).
- [ ] Merged-weights parity: assemble the repo layout locally, then run `python inference_example.py --model <that dir>`
      and `python inference_example.py --adapter --model <that dir>` and confirm the probabilities agree (only the adapter path has been run; see below).
- [ ] User gives the explicit go.

## What was verified (2026-10-04)

- Every DecideBench number in the card and `model-index` matches `data/bench_results/decidebench_{examples,zeroshot}/metrics.json`:
  94.5 / 89.0 (CI 92.25–96.5), 95.0 / 90.0 (CI 92.75–96.76), ECE 0.060 / 0.062, Brier 0.088 / 0.103, per-family,
  hard subset, macro F1, 0 / 400 8-gram overlap. Leaderboard references come from the files' `reference` blocks.
- WorkflowEvals numbers were recomputed from `data/bench_results/workflowevals/*/scores.json` (`series`, consensus
  labels): invoice 46.67 exact / 75.78 primary, customer service 55.56, agent trace 51.80, security 52.08, equal mean
  51.53. OpenAI and Anthropic label sets also recomputed. Runner commit from `run_meta.json`. Jev per-workflow figures
  (61.78 / 75.98 / 71.62 / 61.67, mean 67.76) are from `docs/superpowers/plans/2026-10-02-plan-4-systemone-serving.md`.
- The other sections (training data, filters, ablation, learning curve, calibration, JevBench, DynaBench, R-Judge) are
  carried over from `release/yev-4b/README.md`, whose numbers had been checked against the result files; only names
  changed.
- Public dataset list and licences in `NOTICE` match `src/yev/sources/registry.py` and the train `by_source` counts in
  `data/mix/stage0/mix_report.json` (Stage 1 `public_long` sources are not in the Stage 0 mix and are not listed).
- `calibration.json` is byte-identical to `release/yev-4b/calibration.json`; it has no "yev-4b" or "jeff" label to
  rename (`model` is `lc100/final`).
- `ATTRIBUTION.json` is byte-identical to `release/yev-4b/ATTRIBUTION.json` ("Jeff" appears only in third-party GitHub
  user names).
- `LICENSE` is the standard Apache-2.0 text (identical across the peft, safetensors, accelerate and huggingface_hub
  copies in the venv).
- `inference_example.py` passes `python -m py_compile`. Its `--adapter` mode was run on CPU against a local
  repo-shaped dir (the `lc100` adapter + `calibration.json`) with the cached base: it printed `approve_capped` at
  0.9159 on the built-in expense example. The default merged-weights mode was **not** run (no merged weights locally yet).
- `yev serve` commands follow `docs/serving.md` and `src/yev/train/infer.py:load` (a dir without `adapter_config.json`
  is loaded as a full model; the tokenizer comes from `--base` or the model dir).

## TBD

1. Frontier-LLM WorkflowEvals leaderboard figures (the card says "higher still (figures TBD)"); no result file in the
   repo records them.
2. 8-gram overlap of the Stage 0 train file against the WorkflowEvals cases.
3. WildGuardTest results.
4. DecideBench self-hosted leaderboard submission.
5. Training-run git SHA.
6. URL of Together's "How to train your own Jev" guide.
7. Code repository URL ("the yev code repository (link TBD at release)").
8. Merged weights: file names, dtype and a parity check against the adapter (controller's merge).
9. Base-model licence: the card and NOTICE state Apache-2.0 as instructed. The cached base snapshot lists a
   `LICENSE` file of 11,343 bytes, but its text and the Qwen model card were not read here; confirm before upload.
