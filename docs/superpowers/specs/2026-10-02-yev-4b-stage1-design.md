# yev-4b Stage 1: targeted data and retrain — design

Date: 2026-10-02. Amends `docs/superpowers/specs/2026-09-29-yev-4b-system-one-design.md` (the main spec). Where this document is silent, the main spec holds; where it differs, this document wins for Stage 1.

## 1. Why

Stage 0 (`lc100`) results:

- **DecideBench test:** ties imajev-4b. 94.5% / 89.0% pair with examples; 95.0% / 90.0% zero-shot.
- **Weak on long and structured inputs:**
  - WorkflowEvals 51.5% overall, against Jev's 67.8%;
  - JevBench public 78.4% (hard tier 55.9%, temporal/numeric 1/15);
  - DynaBench test F1 55.7;
  - R-Judge F1 50.8, which is chance.
- **The causes these point to:**
  - training rows mostly ≤ 4k tokens and ≤ 6 options;
  - few numbered score scales;
  - no ISO-date arithmetic;
  - little multi-turn or agent-trace data.

The user decided on 2026-10-02 to run Stage 1 before publishing.

## 2. Goal and publish gate

**Target:** WorkflowEvals overall ≥ 60%, with DecideBench test ≥ 94%.

**Minimum to publish:** clearly better on the long-input benchmarks, with no regressions.

| Check | Minimum to publish | Target |
|---|---|---|
| WorkflowEvals overall (consensus labels, equal mean of the 4 workflows) | ≥ 54.5% | ≥ 60% |
| DecideBench test, with examples (scored once on the Stage 1 model) | ≥ 94.0% | ≥ 95% |
| JevBench public accuracy | ≥ 77% | up, especially the hard tier |
| DynaBench test F1 | ≥ 54 | up |
| R-Judge F1 | ≥ 49 | up |
| Stage 0 dev accuracy | ≥ Stage 0 − 1 point (≥ 93.0%) | up |
| ECE-15 (all dev) | ≤ 0.05 | ≤ 0.04 |

- **If the gate passes:** run the Decision Index once, then go to the release plan.
- **If it fails:** report the numbers and a diagnosis to the user before doing anything else.

## 3. Hard rules (in addition to main spec §2)

1. **Never train on any benchmark we report.** That covers:
   - DecideBench test;
   - WorkflowEvals (all four `typesafe/evalsafe-*` datasets);
   - JevBench public;
   - DynaBench test;
   - R-Judge;
   - the test sets of the 38 Decision Index 0.2.1 benchmarks.
2. **The contamination filter is extended to all of them.** It applies the main spec's 8-gram and embedding filters with each benchmark's states as the reference set, and drops any matching row.
3. **Opus writers never see WorkflowEvals or any other benchmark material.** Scenarios come from our own topic lists.
4. **DecideBench test is scored exactly once on the Stage 1 model,** after the model is final, and never used to choose between checkpoints or runs. The user approved this second scoring on 2026-10-02; Stage 0's scoring stays on the record.
5. **Nothing goes to Hugging Face until the user says go.**

## 4. Data

Stage 1 data = the Stage 0 mix (44,459 training rows) plus the new families below. Counts are starting targets.

| # | Family | Source | Rows | Fixes |
|---|---|---|---|---|
| 1 | `returns_v2` | Programmatic. ISO dates across month and year boundaries and leap days; policies with and without a store-credit tier; the policy text in the state (DecideBench-style). | ~3,000 | `returns_policy`, dates |
| 2 | `date_number_drills` | Programmatic. Elapsed days, business days, deadlines, sums against thresholds, percentages. | ~3,000 | temporal/numeric |
| 3 | `invoice_ledger` | Programmatic. 2–8k-token documents covering line totals, tax, purchase-order and receipt matching, due dates and duplicate invoices; 6–10 questions per document. | ~3,000 (≈400 documents) | long states |
| 4 | `numbered_scales` | Programmatic. Half of the severity and ordinal generator output re-rendered with level keys "0".."n-1". | ~2,000 | Score format |
| 5 | `many_options` | Programmatic routing and category questions with 7–10 options, plus existing intent sources reframed with 7–10 candidate options. | ~3,000 | 7–9 option questions |
| 6 | `long_workflow` | Opus-written support threads, incident reports, agent traces and multi-turn transcripts (4–10k tokens), with 6–12 typed questions each. | ~4,000 (≈600 documents) | WorkflowEvals, R-Judge, DynaBench |
| 7 | `long_padding` | Existing short decisions embedded in long, realistic distractor documents (2–10k tokens), with labels unchanged. | ~3,000 | finding the relevant span |
| 8 | `public_long` | Public datasets that appear in none of the §3.1 benchmarks, with allowed licences. A research step chooses them. | ≤ 3,000 | variety |

**Rules for the families:**

- **Programmatic families compute their labels from code.** Values are sampled at every boundary: exactly at the threshold and one step either side.
- **Family 6 follows main spec §4.3:**
  - a writer, then an independent blind check of every question;
  - soft labels from 3 blind samples;
  - injection on about 5% of documents.
  - **Pilot first:** 50 documents, to measure survival rates before scaling.
- **Every new row passes the full filter pipeline:** licence, canary, the extended contamination filter, dedupe, and the shortcut filter.
- **Each new family holds out a dev share** of 5%, at least 100 rows, plus a calibration share.
- **Options and scales:** up to 10 options (letters A–J). Score scales keep their level order.

## 5. Training

- **One main run.** No learning-curve runs (the user decided this on 2026-10-02).
- **A fresh LoRA from `Qwen/Qwen3.5-4B-Base`,** with the `lc100` recipe:
  - r = 64, α = 128, on all linear layers;
  - learning rate 1e-4, cosine schedule, 3% warmup;
  - the same losses and label smoothing;
  - no pair or permutation loss.
- **Changes from Stage 0:**
  - `N_MAX` goes from 6 to **10**.
  - Maximum length goes from 4,096 to **12,288** tokens.
  - Batches are length-bucketed under the padded token budget, so a long row may form a batch of one.
- **One epoch.** The main spec's 2 epochs are replaced, because Stage 1 data is mostly synthetic. If dev is still improving at the end, report it rather than extending the run.
- **Per-type temperatures** are refitted on the Stage 1 calibration split.
- **Checkpoints and resume** work as in Stage 0. The finished-run guard prevents overwriting.
- **Position-bias check after training:** dev questions with 7–10 options are re-asked with shuffled option order. Measure argmax agreement and probability drift. If they show position dependence, a follow-up run adds L_perm.

## 6. Evaluation

Run in this order on the final model:

1. Dev: Stage 0 dev, per-family Stage 1 dev, and calibration metrics.
2. The position-bias check.
3. WorkflowEvals, through `yev serve` (with prefix cache), exactly as in Stage 0.
4. JevBench public, DynaBench test and R-Judge.
5. DecideBench test, once, with examples and zero-shot.
6. **If the gate passes:** the Decision Index 0.2.1. Declare the 26-option limit and the BANKING77/CLINC150 train-split use.

## 7. Cost and timeline

- **Training:** about 100M tokens at Stage 0's ~1,660 tokens/s, so 16–22 GPU-h.
- **Evaluation:** about 1.5 GPU-h, plus 3–5 GPU-h for the Decision Index.
- **Total:** 20–28 GPU-h, about $40–55, over 3–4 days elapsed.
- **Opus generation** runs on the subscription.
- **The instance is stopped** whenever no job is running.

## 8. Out of scope

- vLLM serving.
- Error mining and weight averaging (main spec Stage 2).
- Multi-letter keys for more than 26 options.
- Release work (model card update, merged weights, dataset repo). That belongs to the following release plan.
