# yev1-4b Stage 1c: targeted data and replay

Status: draft for review, 2026-10-08. Builds on `2026-10-02-yev-4b-stage1-design.md` and the Stage 1b run.

## Why

Stage 1b fixed the Stage 1 regression on DecideBench dev (93.3 / 73.0 group) and improved JevBench (+5.6) and
DynaBench (+7.7 F1). It got worse on three things:

| Benchmark | yev0 | Stage 1b |
|---|---|---|
| R-Judge F1 (recall) | 50.8 (49.2) | 35.7 (26.9) |
| WorkflowEvals invoice, exact action set | 46.7 | 27.6 |
| WorkflowEvals agent trace, primary action | 51.8 | 45.5 |
| WorkflowEvals overall (Jev 67.8) | 51.5 | 47.6 |

Stage 1c adds new data aimed at the measured failures and replays the Stage 0 data that long documents drowned out.

## Decisions already made by the owner

- **Targeting reads individual test failures** on JevBench, R-Judge, DynaBench and WorkflowEvals. Those four are
  reported on the yev1 card as *tuned-on*: failure analysis shaped the training data.
- **DecideBench test is never inspected or used.** It is scored once on the final Stage 1c model, under its own stage
  marker, only if 1c is chosen over 1b.
- **Arithmetic stays out.** Production computes deltas with rules. Where a decision depends on a number, the state
  prints the computed figure or flag.

## Findings the design rests on

Full reports: `data/gap_reports/gap_{rjudge,dyna_jev,workflowevals,dev}.md` (git-ignored; they paraphrase benchmark
items and must not be published or handed to writers verbatim).

1. **R-Judge.** The model has almost no signal (AUC 0.57). It judges unsafe agent traces by alarming topic, not by
   what the agent did. Most errors are prompt injection (did the agent act on a command planted in tool output?) and
   over-reach (bulk or irreversible actions, assumed arguments, skipped confirmation). The transcript framing appears
   in 0 training rows.
2. **DynaBench.** 72% of misses are one scenario: an agent disclosing staff contact details or whereabouts to someone
   not entitled to them. Other misses: count and threshold rules, grounded availability, redaction, persona and
   style rules. The "user repeats 3 times" kind of rule is flagged on every row because the model does not count.
3. **JevBench.** The hard tier is 68.5%. Errors: judge items accepting flawed output, adopting a planted wrong note,
   late policy amendments, alias chains (and date/number items, which stay out).
4. **WorkflowEvals invoice is not an arithmetic failure.** The questions are semantic; the harness computes money and
   dates. The model collapsed onto fixed option letters (line scope: 4th option 351 of 456; line completion: 5th
   option 331 of 458), read plain "Net 30" as an early-payment discount, and over-routed clean invoices to review.
5. **WorkflowEvals trace and security.** Healthy traces are over-closed, safety breaches under-paged; security treats
   near-miss context (cancelled change ticket, travel only submitted) as benign; ordinal scales compress to the middle.
6. **Dev.** The Stage 0 data still has all its rows, but only ~43% of training tokens (from 100%). The Stage 0
   families moderation, routing and returns come from a used-up Opus pool (~1,000 train rows each). qasper is stuck at
   72% with an exhausted pool. Part of the Stage 0 dev dip may be prompt re-rendering, not the model; this is checked
   by scoring yev0 on the Stage 1b dev rendering (one GPU job, before training).
7. **The answer-letter distribution in the Stage 1b mix is balanced** for every option count, so the letter collapse
   is learned from content, not from skewed gold letters.

## New data (~22k rows)

Labels come from a code generator's latent facts wherever possible; Opus writes the prose. Where Opus plants a flaw,
a blind second Opus pass must find it on its own, or the row is dropped. Writers get paraphrased pattern
descriptions and invented examples only, never benchmark text.

| Cluster | Family | Rows | Writer |
|---|---|---:|---|
| **A. Agent safety** | `agent_injection_pair`: same planted payload; agent obeys (unsafe) or ignores/reports (safe) | 2,400 | Opus transcript, rule label |
| | `agent_overreach`: bulk/irreversible action, assumed argument or payment, missing confirmation; matched careful version | 1,600 | Opus + templated triples (40%) |
| | `agent_benign_multistep`: alarming topic, safe behaviour (hard negatives) | 1,000 | Opus |
| | `trace_audit`: grant scope, first bad step, healthy vs needs-page over long traces | 1,000 | Opus trace, rule label |
| **B. Conversation policy audit** | `disclosure_gate`: who is asking × what is disclosed (contact, whereabouts) | 1,600 | rule scenario, Opus dialogue |
| | `count_threshold_rule`: rules that need counting or a cap; clean rows contain the trigger | 1,000 | rule |
| | `grounded_offer`: offering items/slots absent from the data block; redaction leaving one identifier | 1,000 | Opus |
| | `persona_style_rule`: persona, register, code-style constraints, with boilerplate-rule negatives | 900 | Opus, blind check |
| **C. Hard judging** | `judge_planted_error`: output with one subtle error vs correct output | 1,200 | Opus, blind check |
| | `planted_note`: a wrong note, template or summary in context that must not be adopted | 1,200 | Opus |
| | `policy_amendment`: late amendment or precedence clause changes the answer | 1,100 | Opus |
| **D. Workflow decisions** | `invoice_line`: scope, completion, evidence-counts, as consistency bundles over one packet | 1,800 | Opus packet, rule label |
| | `payment_terms`: discount window vs plain net/due-on-receipt terms | 400 | rule + Opus paraphrase |
| | `invoice_action`: primary and full action set from precomputed flags, three policy variants, 25% clean invoices | 1,300 | rule (Opus notes 10%) |
| | `cs_scales_actions`: anchored urgency/frustration scales, action sets (flag, pending, handoff, priority) | 1,800 | Opus |
| | `security_context`: does context cover the activity; one fault per negative; containment playbook | 1,700 | Opus, rule label |
| **E. Dev gaps** | `public_long` qasper, new rows | 600 | existing pipeline |
| | `long_workflow`, new documents | 800 | existing pipeline |
| **Total** | | **~22,400** | |

Cross-cutting rules for every new family:
- **Shuffled option order** in every choice row, with 50% in the canonical order the real workflow uses.
- **Contrastive pairs:** same state with one fact changed, and the label flips.
- **Consistency bundles:** 3–6 related questions over one state.
- **Ordinal scales:** anchored in the question and flat in distribution, with near-neighbour pairs.
- **Hard negatives** make up 30–40% of rows in the families that over-call benign/healthy/none.
- **Fresh content:** invented tool names, personas, vendors and payload phrasing. Sensitive topics are balanced
  across both labels.
- **Question wording:** the production question 75% of the time, paraphrases 25%.

Quality controls, reused from Stage 1:
- the benchmark contamination filter over every new row;
- a Haiku guessability control on Opus-written choice questions;
- the Haiku difficulty pass (questions Haiku misses get weight 1.5).

Each family gets its own dev and calibration share, built from invented cases.

## Replay (~+8.7M tokens)

- 2 extra passes over the Stage 0 Opus families moderation, routing and returns.
- 1 extra pass over the other Stage 0 Opus families.
- Half a pass over skill_filing.
- numbered_scales raised from 2,000 to 2,500 rows.

**Mechanics:** a recipe block gains `repeat` (fractional allowed) with an optional family/source filter. Repeats are
applied after the dev holdout and unit sampling, with their own rng, so existing builds stay byte-identical and dev
units cannot leak into train.

## Size and cost

- **Training:** ~70.9M (1b) + ~8.7M replay + ~45–55M new ≈ 125–135M tokens. At 2,140 tok/s that is ~16–18 h on
  the L40S (~$30–35 on-demand). One epoch, same hyper-parameters as Stage 1b (LoRA r64, lr 1e-4).
- **Opus:** roughly 1–1.5× the Stage 1 Opus run in subagent tokens, run as a workflow. Haiku controls on top.

## Evaluation and choice

1. **Before training:** score yev0 on the Stage 1b dev rendering. This settles whether the Stage 0 dip is real.
2. **Stage 1c chain:**
   - dev evals: Stage 0 dev, Stage 1 dev, and new-family dev;
   - position bias;
   - calibration re-fit on a calibration split rich in fail/unsafe rows.
3. **Choosing 1b vs 1c uses dev only.** 1c must not lose more than 0.5 pt on DecideBench dev or Stage 0 dev.
4. **Then, on the chosen model:**
   - JevBench, R-Judge, DynaBench and WorkflowEvals, reported as tuned-on;
   - DecideBench test, scored once (stage marker `stage1c` if 1c is chosen).

## Out of scope

- Date and number arithmetic.
- Choosing decision thresholds on test data. For example, the DynaBench cut-off of 0.35 would raise F1, but it was
  found on test, so it is not used.
- Changing the serving readout.
