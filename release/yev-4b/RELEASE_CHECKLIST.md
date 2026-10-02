# yev-4b release checklist

**Standing rule: nothing goes to Hugging Face until everything below is finished.** That means no model repo, no dataset
repo, no uploads and no pushes until the whole list is ✅ and the user says go. Everything here is prepared locally only.

Legend: ✅ have · ❌ missing · ⚠️ needs a decision

Spec §8 requires:

- A model repo with weights, tokenizer, `calibration.json`, minimal serving code, and a model card covering:
  - recipe and ablation table;
  - public sources with licences;
  - filter report;
  - calibration metrics;
  - DecideBench results;
  - the contamination statement.
- A dataset repo.
- The training code in a public GitHub repo.

Released checkpoint: **`lc100`**, the ablation winner on DecideBench-dev template-group accuracy.

## 1. Model repo contents

| # | Item | Status | Where / notes |
|---|---|---|---|
| 1.1 | LoRA adapter weights (`adapter_model.safetensors` + `adapter_config.json`) | ❌ | Only on the **stopped** AWS box at `~/runs/lc100/final`. Copying it down needs the box started, which is the user's call. Nothing is held locally. |
| 1.2 | `adapter_config.json` → `base_model_name_or_path` | ❌ | The trainer loaded the base from `/home/ubuntu/models/Qwen3.5-4B-Base`, so the saved config almost certainly points there. Rewrite it to `Qwen/Qwen3.5-4B-Base` before publishing, and check the file for any other box paths. |
| 1.3 | Also ship merged full weights? | ⚠️ | No merged checkpoint exists, so one would have to be made with `merge_and_unload` on the box. Pros: plain `transformers`/vLLM loading and no peft dependency. Cons: about 8 GB more, and the card's `library_name` would change from `peft`. Decide: adapter only, merged only, or both (for example a `-merged` repo). |
| 1.4 | Tokenizer files | ❌ | Not in `release/yev-4b/`. The base tokenizer is unchanged. Save it from `Qwen/Qwen3.5-4B-Base`, including the chat template that supports `enable_thinking`, so the repo is self-contained. Letters A–F are single tokens 32–37 (checked locally against the cached base tokenizer). |
| 1.5 | `calibration.json` (lc100 temperatures) | ✅ | `release/yev-4b/calibration.json`, copied from `data/runs/lc100/calibration.json`: choice 0.9360, noul 0.9812, score 1.0108, n = 2000. ⚠️ It still holds the box paths in `model`/`data` (`/home/ubuntu/...`). These are harmless, but decide whether to strip them. |
| 1.6 | Model card | ✅ draft | `release/yev-4b/README.md`. Its TBDs are listed in §6 below. Re-check every number if anything is re-run. |
| 1.7 | Example inference script | ✅ / ⚠️ | `release/yev-4b/inference_example.py`. Checks done: `py_compile` passes; the rendered messages equal `yev.format.render`; letter ids match `yev.train.data.letter_token_ids`. On the **base model** (no adapter), its letter logits are bit-identical to `yev.train.infer.letter_logits` for a right-padded 2-row batch. **Not yet run with the adapter.** Run it once on the box and compare with `yev eval` on a few dev rows. |
| 1.8 | Minimal serving code (spec §6): `POST /v1/systemone` + `POST /v1/chat/completions`, letter-restricted, state-prefix caching | ✅ | Done: `yev serve`, documented in `docs/serving.md`. Transformers backend (not vLLM); state-prefix KV caching for multi-question requests; letters A-Z are served but the model was trained on at most 6 options. Returns a Score item's distribution plus its expected value. |
| 1.9 | `LICENSE` (Apache-2.0 full text) | ❌ | Not in the repo or in `release/`. Add it to the model repo, and to the code repo too. |
| 1.10 | Base-model licence confirmed compatible | ⚠️ | The card says the base is "distributed under its own licence (TBD)". Confirm the `Qwen/Qwen3.5-4B-Base` licence on its model card, and state it. |
| 1.11 | Third-party attribution: skill-atlas | ✅ / ⚠️ | `release/yev-4b/ATTRIBUTION.json` is a copy of `data/public/skill_atlas_filtered/ATTRIBUTION.json`: 6,106 skills from 2,368 repos (4,174 MIT, 1,932 Apache-2.0). It covers the whole filtered pool, which is a superset of the rows used. Decide whether to trim it to the skills actually in train/dev. MIT/Apache notice-retention clearly applies to the dataset repo, which redistributes skill text. For the weights it is a courtesy; decide whether to keep it there. |
| 1.12 | Third-party attribution: datasets (NOTICE) | ❌ / ⚠️ | The card's source table lists each dataset and licence. A `NOTICE` or `THIRD_PARTY.md` with full credits (authors, URLs, licence links) for the CC-BY / CC-BY-SA sources has not been written. |
| 1.13 | CC-BY-SA sources (VitaminC, BoolQ, ARC; MultiNLI fiction) vs Apache-2.0 weights | ⚠️ | Common practice treats weights as not a share-alike adaptation of the data, but this is a licensing call for the user. It matters more for the dataset repo (§2). |
| 1.14 | Safety and misuse note | ✅ draft | It is in the card under "Safety and misuse" and "Intended use / out of scope". Review the wording. |
| 1.15 | HF repo id and org | ⚠️ | Not decided. The card uses `TBD/yev-4b` in the snippet and `TBD` in the citation URL. |
| 1.16 | Training curves / TensorBoard logs | ❌ (optional) | `~/runs/*/tb` on the box only. |
| 1.17 | Benchmark predictions (`predictions.jsonl`) | ❌ (optional) | Only `metrics.json` is local, under `data/bench_results/*/`. Per-row predictions are on the box. Do **not** publish R-Judge rows, which have no licence file. |

## 2. Dataset repo (spec §8)

| # | Item | Status | Where / notes |
|---|---|---|---|
| 2.1 | Dataset repo: synthetic + mixed training data with per-row `source`, `licence`, `cluster_id`, `edit_type` | ❌ | Not built. The data is at `data/mix/stage0/{train,dev,calibration}.jsonl` (+ `.chat.jsonl`), `data/synthetic/s0-filtered/`, and `data/stage0-pool/`. All of `data/` is git-ignored and local. |
| 2.2 | What goes in it | ⚠️ | The options are synthetic only (Opus + rules, stamped CC-BY-4.0), or the full mix including public rows. Redistributing public rows means honouring each upstream licence per row, including CC-BY-SA share-alike and MultiNLI's OANC-mixed terms. `dev.jsonl` contains the 297 DecideBench examples (CC-BY-4.0, dev only). Decide whether to drop them or mark them clearly. |
| 2.3 | Dataset card | ❌ | Needs to cover: provenance (Opus 5.5 writer → blind 3-sample check → soft labels → edit attribution; rule specs); the "no independent label check" statement; filter reports; licence per source; the skill-atlas attribution; and the contamination statement. |
| 2.4 | skill-atlas `ATTRIBUTION.json` shipped with the dataset | ❌ | Required: the rows carry MIT/Apache skill text. |
| 2.5 | Filter / shortcut / mix reports in the dataset repo | ❌ | Sources are `data/public/*filtered*/filter_report.json`, `data/synthetic/s0-filtered/filter_report.json`, `data/stage0-pool/shortcut_report.json` and `data/mix/stage0/mix_report.json`. Their numbers are already summarised in the model card. |
| 2.6 | Dataset licence | ⚠️ | A per-row licence field exists. Choose the repo-level licence statement, for example "mixed, see per-row `licence`" or CC-BY-4.0 for a synthetic-only release. |

## 3. Code repo

| # | Item | Status | Where / notes |
|---|---|---|---|
| 3.1 | Rename `jeff` → `yev` in code | ✅ | Done 2026-10-02: package `yev` (`src/yev/`), CLI `yev`, `pyproject.toml` `name = "yev"`, spec/plan titles and paths, configs, docs, scripts (box checkout `~/yev`, env vars `YEV_TRAIN_HOST`/`YEV_TRAIN_KEY`). Only third-party GitHub usernames in `ATTRIBUTION.json` still contain "Jeff". The GitHub repo and local directory are still named `jeff`. |
| 3.2 | Public GitHub repo with the training code (spec §8) | ❌ | It is local only, on branch `feat/plan-3`; nothing is pushed or public. Before publishing, re-check that no host, key path or instance id is committed (`tests/test_aws_scripts.py` guards `scripts/aws/`). |
| 3.3 | Code repo README | ❌ | None exists. |
| 3.4 | Tests green at the release SHA | ⚠️ | The last recorded run was 384 passed, 3 deselected, at `6212219` (bench report). Re-run at the release commit. |

## 4. Reproducibility

| # | Item | Status | Where / notes |
|---|---|---|---|
| 4.1 | Training run git SHA | ❌ | Not recorded: `train_summary.json` has no `git_sha`, and the bench runs log `git_sha: null`. The training code was last changed at `c3bd5a2` ("per-pair perm loss…") before the runs, but which commit was synced to the box is not logged. Record it as "TBD", or recover it from `~/yev` on the box. Add a `git_sha` field to the trainer and to `bench score` for future runs. |
| 4.2 | Run configs | ✅ | `configs/stage0/{lc25,lc50,lc100,ab_pair,ab_perm,ab_both}.json`; the other hyperparameters are `TrainConfig` defaults in `src/yev/train/trainer.py`. |
| 4.3 | Seeds | ✅ | Training seed 0 (`TrainConfig.seed` default) and mix seed 0 (`recipes/stage0.json`). Rule generator seeds: 201–203 (Plan 2b). Opus plan seeds: 10, 11, … (Plan 2b). |
| 4.4 | Data recipe and filter provenance | ✅ | `recipes/stage0.json`. The filter reports record the DecideBench revision `36df882…`, the embedding model and the filter git SHAs (`da234f2`, `7df679c`, `a3c9c95`). |
| 4.5 | Hash of the exact train file | ❌ | Compute a sha256 of `data/mix/stage0/train.chat.jsonl` and record it in the card and the dataset repo. |
| 4.6 | Environment | ✅ | torch 2.14.1+cu130, transformers 5.18, peft 0.21, flash-linear-attention, causal-conv1d (Plan 3 and `scripts/aws/setup.sh`). Freeze a `pip freeze` from the box venv if it is restarted. |

## 5. Evaluation

| # | Item | Status | Where / notes |
|---|---|---|---|
| 5.1 | DecideBench test, with examples + zero-shot | ✅ | `data/bench_results/decidebench_{examples,zeroshot}/metrics.json`: 94.5 / 89.0 and 95.0 / 90.0. Run once each. |
| 5.2 | DecideBench self-hosted leaderboard entry (spec §7) | ❌ | Not submitted. The spec asks for an entry in the DecideBench repo with both columns, 4 requests in flight, and pricing at the L4 GPU-hour rate. It uses the serving endpoint (1.8, done), because DecideBench runs TEV through `/v1/chat/completions`. ⚠️ The spec named the DGX Spark as the host, but training and scoring used the AWS L40S, so decide which host and rate to report. Our scores came from transformers letter-logit readout, not through the harness's HTTP path; re-confirm through the endpoint without rescoring the test more than the submission requires. |
| 5.3 | WildGuardTest | ❌ | Not run. `allenai/wildguardmix` is gated, and the AI2 Responsible Use terms have not been accepted on the user's account (the agent did not accept them). After the user accepts: `yev bench build --name wildguardtest`, then `overlap`, then `score`. The card currently says TBD. |
| 5.4 | JevBench public, DynaBench test, R-Judge | ✅ | `data/bench_results/{jevbench,dynabench_test,rjudge}/metrics.json`. The card reports them with the "doesn't top these" framing. |
| 5.5 | Stretch claim wording | ⚠️ | The spec §1 must-bar (> 92.8 % and > 86.0 %) is met. The stretch (≥ 95.0 %) is met zero-shot (95.0) but not with examples (94.5). imajev-4b leads with examples (95.0 / 90.5). The card says "tie, not a win"; keep that framing. |
| 5.6 | Release Stage 0, or wait for Stages 1 / 2 / 2b | ⚠️ | The spec's full pipeline is not done. The card is written for a Stage 0 release, and the date-arithmetic fixes are in the returns diagnosis. Decide whether to publish this now or after Stage 1. |

## 6. TBDs in the model card

These must be resolved or explicitly accepted before publishing.

1. Final HF repo id: the snippet `ADAPTER = "TBD/yev-4b"` and the citation URL.
2. Citation author list.
3. URL of Together's "How to train your own Jev" recipe in the acknowledgements.
4. Training-run git SHA (4.1).
5. WildGuardTest results (5.3).
6. Merged-weights decision (1.3).
7. Base-model licence statement (1.10).
8. DecideBench leaderboard entry (5.2).

## 7. Final gate

- [ ] Every ❌ above is ✅, or explicitly dropped by the user.
- [ ] Every ⚠️ has a recorded decision.
- [ ] Numbers in `README.md` re-checked against `data/runs/lc100/*` and `data/bench_results/*/metrics.json`.
- [ ] User gives the explicit go. Only then create the HF repos and upload, never before.

Ops note (rename): the box's old checkout at `~/jeff` should be moved (`mv ~/jeff ~/yev`) or re-synced, then reinstalled with `pip install -e '.[train,serve]'`.
