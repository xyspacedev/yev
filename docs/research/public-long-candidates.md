# Public long slice (family 8): candidate datasets

Plan 5, Task 8. Family 8 (`public_long`) holds at most 3,000 rows from public datasets with long inputs, cast to Choice/Noul/Score. Read on 2026-10-02.

## Rules each pick must meet

- **Licence.** `yev.licences.is_allowed` accepts the SPDX id: `apache-2.0`, `mit`, `cc0-1.0`, `cc-by-3.0`, `cc-by-4.0`, `cc-by-sa-3.0`, `cc-by-sa-4.0` or `oanc-mixed`. Non-commercial and no-derivatives licences fail.
- **Not a reported benchmark.** Not in `DECISION_INDEX_SOURCES` or `DECISION_INDEX_SHOWN_SOURCES` (see `decision-index-0.2.1-sources.md`). Not DecideBench, WorkflowEvals (`typesafe/evalsafe-*`), JevBench, DynaBench, R-Judge or WildGuardTest/WildGuardMix. Not a mirror of one; the repo name after the slash is checked too.
- **Not derived from one.** The dataset is not built on a benchmark's underlying corpus: ContractNLI's NDAs, QuALITY, MuSR, HoVer (and HotpotQA, which HoVer is built on), ANLI, NLI4CT, RAGTruth, and so on.
- **Long.** At least 2,000 Qwen3.5 tokens per state, counted with the real tokenizer. The full rendered prompt must fit in 12,288 tokens. Longer rows are dropped and counted.
- **Answerable from the state.** The gold depends only on the document, with no outside knowledge. Only the train split is read.

## Chosen

| Source name | Hub id @ revision | Files | Licence | Raw size | Cast |
|---|---|---|---|---|---|
| `cuad` | `theatticusproject/cuad-qa` @ `d9c4ee0250ae2eb97bdb5b50773ab14ea62d0631` (the `refs/convert/parquet` commit) | config `default`, split `train` | cc-by-4.0 | 22,450 rows = 408 contracts × 41 categories (with repeats) | Noul: does the contract contain a clause of category X? |
| `maud` | `theatticusproject/maud` @ `37d5c3b95d18dcd8404cc5ce3fd5069be062392f` | `MAUD_v1/MAUD_train.csv` | cc-by-4.0 | 25,827 rows; 13,256 `main` rows over 152 merger agreements | Choice: which answer describes deal point X? (Noul when the answers are Yes/No) |
| `qasper` | `allenai/qasper` @ `06806e4608976fc2fac0a090ac425d5b2b29caf4` (the `refs/convert/parquet` commit) | `qasper/train/0000.parquet` | cc-by-4.0 | 888 papers, 2,593 questions | Choice: yes / no / the paper does not say |

### CUAD (Contract Understanding Atticus Dataset)

- **Source.** The Atticus Project. 510 commercial contracts from SEC EDGAR, labelled by law students and checked by lawyers for 41 clause categories. The CUAD-QA train split holds 408 of them, each with its full text as `context`.
- **Licence.** CC BY 4.0 (Hub card and the CUAD README).
- **Why it fits.** Contract review is a workflow decision: "does this agreement contain a non-compete, an audit right, a change-of-control clause?" The gold is whether experts marked any span for the category, so it comes only from the contract text. 219 of the 408 train contracts are 2,000–11,000 tokens long (median 7,366 tokens; 10th percentile 1,532; 90th 26,522).
- **Cast.** One Noul row per (contract, category). 32 categories are used: the 31 whose definition is a yes/no question, plus "Competitive Restriction Exception". The extractive ones are dropped: Document Name, Parties, the dates, Renewal Term, Notice Period, Governing Law and Warranty Duration. The question is CUAD's own category definition. Each contract gives up to 8 rows, half yes and half no where possible.
- **Exclusion checks.** CUAD is not in the Decision Index list or any other reported benchmark, and it is not built from one. ContractNLI (DI #11) also drew from EDGAR, but its documents are NDAs. CUAD has no NDA category. As a precaution, the one train contract whose title names a confidentiality/non-disclosure agreement (`WOMENSGOLFUNLIMITEDINC_..._Confidentiality and Non-Use Obligations Agreement`) is excluded, along with any title matching `disclos|confidential`.

### MAUD (Merger Agreement Understanding Dataset)

- **Source.** The Atticus Project. 152 public merger agreements (2021 ABA deal-points study) in the train CSV. Experts answer multiple-choice deal-point questions (144 question/sub-question pairs in train), each tied to an excerpt of the agreement.
- **Licence.** CC BY 4.0 (Hub card).
- **Why it fits.** Deal-point review is a choice over a fixed answer set for each question (for example "Type of Consideration": All Cash / All Stock / Mixed Cash/Stock / Mixed Cash/Stock: Election). The state joins all of one agreement's `main` excerpts into a single document: median 8,403 tokens, range 3,194–12,565. The decision then means finding the relevant provision inside a long agreement and reading it. The excerpt that decides the answer is always inside the state.
- **Cast.** One Choice row per (agreement, question). Only questions with no sub-question and 2–10 distinct answers in the train split are used; the options are those answers. A question whose answers are exactly Yes/No becomes Noul. The question names the agreement section (MAUD `text_type`, for example "MAE Definition") and the cleaned deal-point name. Each agreement gives up to 8 rows.
- **Exclusion checks.** MAUD is not in any reported benchmark and is not built from one. Merger agreements are not NDAs, so MAUD does not overlap ContractNLI.

### QASPER

- **Source.** AllenAI. NLP papers from S2ORC. Questions were written by readers who saw only the title and abstract. Answers came from other annotators who read the full paper.
- **Licence.** CC BY 4.0 (Hub card).
- **Why it fits.** The question asks whether a long technical document says something: yes, no, or not at all. This is the "long document with entailment-style questions" pattern. The papers are 2,611 tokens at the 10th percentile (median 5,168); 808 of 888 are 2,000–11,000 tokens.
- **Cast.** One Choice row per question with options `yes`, `no` and `not_stated`. A question is kept only if every annotator gave the same kind of answer. It is dropped if any evidence is a figure or table (`FLOAT SELECTED:`), because table contents are not in the text. Unanswerable questions are kept only when the question itself is yes/no-shaped (starts with an auxiliary verb), as `not_stated`. Free-form and extractive answers are not used.
- **Exclusion checks.** QASPER is not in any reported benchmark and is not built from one.

### Note on overlap with benchmarks we do not report

CUAD and MAUD are the sources of LegalBench's `cuad_*` and `maud_*` tasks. QASPER (with ContractNLI and QuALITY) is part of SCROLLS and LongBench. None of these is a benchmark we report: LegalBench is on the licence module's evaluation-only list, not on the reported list. If any of them is ever reported, this slice must be revisited.

## Considered and rejected

| Candidate | Reason |
|---|---|
| ContractNLI (`stanfordnlp/contract-nli`, `kiddothe2b/contract-nli`) | DI #11, and also forbidden in `licences.py`. |
| QuALITY | Named in the exclusion list. Also part of SCROLLS/LongBench. |
| MuSR, HoVer, HotpotQA | DI #32 and #61. HotpotQA is HoVer's source corpus. |
| DocNLI | Built partly from ANLI (DI #12). |
| NLI4CT, RAGTruth | DI #42 and #59. |
| SCROLLS / ZeroSCROLLS / LongBench / L-Eval / InfiniteBench | Aggregates that include ContractNLI and QuALITY. |
| LegalBench (`nguha/legalbench`) | Evaluation-only (forbidden in `licences.py`). We use the CUAD and MAUD originals instead. |
| QAngaroo WikiHop (`QAngaroo/wiki_hop`, cc-by-sa-3.0) | Multi-hop Wikipedia QA, close to the HoVer/HotpotQA corpus. Rejected to be safe. Its support sets are also mostly under 2k tokens, and it is not workflow-like. |
| NarrativeQA (apache-2.0) | Free-form answers; whole books exceed 12k tokens. |
| MeetingBank | CC BY-NC-ND 4.0. |
| QMSum, AMI, ICSI | Summarization queries only, with no decision label to cast. |
| ECtHR / LexGLUE `ecthr_*`, Multi-EURLEX | The labels (articles violated, EuroVoc concepts) need outside legal knowledge. |
| OPP-115, PolicyQA | CC BY-NC. |
| PrivacyQA (`alzoubi36/privacy_qa`) | The Hub copy has no licence tag. The labels are per sentence, and the full policy cannot be rebuilt reliably. |
| CLAUDETTE / LexGLUE `unfair_tos`, `joelniklaus/online_terms_of_service` | Sentence-level; inputs far below 2k tokens. |
| ConditionalQA (cc-by-sa-4.0, GitHub only) | A good fit (UK benefit policies with yes/no eligibility questions), but it is not on the Hub. The source loaders read the Hub. Deferred. |
| MultiDoc2Dial / doc2dial (apache-2.0) | The labels are dialogue acts, which the state does not determine. |
| ABCD, Taskmaster, customer-support-on-twitter | Dialogues are well under 2k tokens. |
| `Tobi-Bueck/customer-support-tickets`, Yelp, Amazon reviews multi, SST-5 | Forbidden in `licences.py` (non-commercial or unclear). |
| IMDB, other long-review sets | Licence `other`/unclear, and almost no reviews reach 2k tokens. No permissive long-review dataset with ratings was found, so the slice has no Score rows. |
| BillSum, GovReport | Summarization only; no decision label. |
| PeerRead and similar accept/reject sets | The decision depends on reviewers' outside judgment. Mixed licences. |
| Loghub | Research-use terms, not an allowed SPDX licence. |

## Build results (2026-10-02)

`yev build-public --out data/public/public_long_raw --only cuad maud qasper`, then `yev filter --in data/public/public_long_raw --out data/public/public_long --benchmarks data/bench/fingerprints.json`. Token counts use the real Qwen3.5 tokenizer.

| Source | Documents read | Rows converted | Too short (< 2,000) | Too long (prompt > 12,288) | Kept by pool | Benchmark drops | Duplicates | Final rows (documents) |
|---|---|---|---|---|---|---|---|---|
| cuad | 407 | 1,864 | 416 | 976 | 1,500 | 1,010 (`workflowevals_invoice_processing`) | 3 | 487 (76) |
| maud | 152 | 1,208 | 0 | 8 | 1,100 | 0 | 0 | 1,100 (151) |
| qasper | 888 | 375 | 20 | 17 | 375 | 0 | 0 | 375 (285) |

The "too short" and "too long" columns count rows, not documents. The filter's other checks (licence, canary, n-gram, embedding, every other benchmark) dropped nothing.

**The CUAD benchmark drops are legal boilerplate.** The WorkflowEvals invoice-processing cases embed contract text (MSA terms). Every shared 8-gram is a stock clause, for example "shall be governed by and construed in accordance with the laws of the state", "the entire agreement between the parties with respect to the subject matter hereof" or "including but not limited to acts of god". The benchmark stoplist was built from benchmark items alone, so it does not treat these phrases as boilerplate. 157 of 233 sampled contracts are flagged. The rule was kept as it is.

Final state lengths in tokens (min / p25 / median / p75 / p90 / max):

| Source | State tokens | Largest prompt |
|---|---|---|
| cuad | 2,012 / 3,442 / 4,525 / 7,366 / 9,465 / 11,790 | 12,265 |
| maud | 3,150 / 6,340 / 7,706 / 9,131 / 10,223 / 11,391 | 11,638 |
| qasper | 2,052 / 3,618 / 5,067 / 6,383 / 8,080 / 11,956 | 12,153 |

Labels: cuad yes 223 / no 264 (Noul). maud: 613 Noul (Yes/No deal points) and 487 Choice with 3–8 options. qasper: yes 186 / no 142 / not_stated 47.
