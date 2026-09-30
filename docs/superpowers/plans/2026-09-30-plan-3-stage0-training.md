# Plan 3: Stage 0 training on AWS (L40S) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Stage 0 of jeff-4b from `Qwen/Qwen3.5-4B-Base` with LoRA on the AWS L40S box. Measure a learning curve and the spec's loss ablation, calibrate per-type temperatures, and report dev metrics against TEV.

**Architecture:** A new `jeff.train` package. It is made of pure, CPU-testable parts:
- `targets`: label smoothing and ordinal targets.
- `losses`
- `data`: tokenisation, permuted twins and unit-grouped batches.
- `readout` and `metrics`
- `calibrate`

A plain PyTorch training loop (`trainer`) ties these together. An `infer` module runs a model over chat rows for eval and calibration. Everything is testable locally with a tiny randomly-initialised model and a fake tokenizer. The real runs happen on the GPU box via `scripts/aws/*.sh`, which read the host and key from environment variables.

**Tech Stack:** Python ≥ 3.11, torch ≥ 2.9, transformers ≥ 5.0, peft ≥ 0.17, tensorboard. uv locally; an isolated venv on the box (torch cu130).

**Spec:** `docs/superpowers/specs/2026-09-29-jeff-4b-system-one-design.md` (§5 Training, §6 Readout, §7 Evaluation).

## Global Constraints

- DecideBench is never trained on. `data/dev/decidebench_examples.jsonl` rows appear only in `dev.chat.jsonl`, and the DecideBench test set is never read in this plan (spec §2 rules 1 and 5).
- Nothing is uploaded to Hugging Face or anywhere outside the user's own AWS account. Checkpoints stay on the box.
- The box's host, SSH key path and instance id never appear in the repo. Scripts read `JEFF_TRAIN_HOST` (e.g. `ubuntu@1.2.3.4`) and `JEFF_TRAIN_KEY` (path to the key) from the environment and fail with a clear message if either is unset.
- Prompt format is exactly `jeff.format` (TEV system prompt, JSON user turn, letters A–F). The model is supervised only at the answer-letter position.
- Starting hyperparameters are given in Task 6 and are not changed without a ledger ruling.
- Pass bar reported against (spec §1): accuracy > 92.8 % and pair accuracy > 86.0 % (TEV); stretch ≥ 95.0 %.

## Rulings carried in from the conversation (deviations from the spec)

1. **Hardware and method.** Training runs on an AWS g6e.xlarge (1× L40S 48 GB, 4 vCPU, 30 GB RAM), not the DGX Spark. It uses **LoRA r=64** (spec §9's named fallback), not a full fine-tune. LoRA learning rate is 1e-4, since the spec's 1e-5 is a full-fine-tune value. A full fine-tune on an H100 stays possible later.
2. **Target smoothing, added.**
   - Hard Choice and Noul targets get uniform label smoothing ε = 0.05.
   - Hard Score targets get ordinal smoothing: 0.10 moved from gold to its immediate neighbours.
   - Rows that already carry `soft_gold` are left unchanged.
3. **Learning curve, added before the ablation.** L_ce only, on 25 %, 50 % and 100 % of train, subsampled by whole clusters.
4. **Training loop.** A plain PyTorch loop instead of HF `Trainer`, because unit-grouped batching and permuted twins are awkward to fit into Trainer. The spec's "custom compute_loss" is honoured as `losses.total_loss`.
5. **DecideBench group accuracy.** The dev copy of DecideBench examples has no `pair_id`. "Pair accuracy" on it is computed over template groups, meaning the id with its last `-<gold>` segment removed, and a group counts as correct only when every member is correct. Our own dev clusters use `cluster_id`.

## Review Focus

1. **Letter tokenisation.** In the Qwen tokenizer, " A" and "A" may be different tokens. The answer id must be the token the model would generate right after the generation prompt. Test: `data.letter_token_ids` asserts every letter A–F encodes to exactly one token and returns those ids (Task 3).
2. **Rows longer than the length cap.** They must be dropped and counted, not silently truncated so that the answer position is lost. Test: `test_overlong_rows_dropped_and_counted` (Task 3).
3. **Score rows under permutation.** Score options are an ordered scale and must never be shuffled into a twin. Test: `test_score_rows_get_no_twin` (Task 3).
4. **A temperature fit on a type absent from the calibration split.** It must fall back to T = 1.0, not crash or produce NaN. Test: `test_missing_type_defaults_to_one` (Task 5).
5. **Spot interruption mid-run.** Resuming from the last checkpoint must continue at the same step with the same data order. Test: `test_resume_continues_step_and_order` (Task 6).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/jeff/train/__init__.py` | package marker |
| `src/jeff/train/targets.py` | smoothing: `smooth_target(row, eps_hard, eps_ordinal) -> dict[str, float]` |
| `src/jeff/train/losses.py` | `ce_soft`, `full_vocab_ce`, `pair_margin`, `perm_skl`, `rps`, `total_loss` |
| `src/jeff/train/data.py` | parse chat rows, permuted twins, tokenise, unit-grouped token-budget batches, unit subsampling |
| `src/jeff/train/readout.py` | letter-logit probabilities with temperature |
| `src/jeff/train/metrics.py` | accuracy, group accuracy, ECE-15, Brier, selective accuracy, breakdowns |
| `src/jeff/train/calibrate.py` | per-type temperature fit → `calibration.json` |
| `src/jeff/train/infer.py` | run a model over rows → per-row letter logits |
| `src/jeff/train/trainer.py` | config dataclass, model/LoRA load, training loop, checkpoint/resume, TensorBoard |
| `src/jeff/cli.py` | new `train`, `eval`, `calibrate` subcommands |
| `configs/stage0/*.json` | run configs (learning curve and ablation) |
| `scripts/aws/sync.sh`, `scripts/aws/run.sh`, `scripts/aws/setup.sh` | push code and data, run a command on the box, set up the venv |
| `tests/train/test_*.py` | one test file per module, plus `conftest.py` with the fake tokenizer and tiny model |
| `pyproject.toml` | `train` optional extra |

---

### Task 1: Targets (label smoothing and ordinal smoothing)

**Files:**
- Create: `src/jeff/train/__init__.py` (empty), `src/jeff/train/targets.py`
- Test: `tests/train/__init__.py` (empty), `tests/train/test_targets.py`

**Interfaces:**
- Produces: `smooth_target(target: dict[str, float], type_: str, eps_hard: float = 0.05, eps_ordinal: float = 0.10) -> dict[str, float]`. Letters map to probabilities summing to 1.
  - A target is "hard" when exactly one value is 1.0 and the rest are 0.0. Soft targets are returned unchanged.
  - For `type_ == "score"`, letters are in scale order A, B, C…

- [ ] **Step 1: Write the failing tests**

```python
import math
from jeff.train.targets import smooth_target

def close(a, b): return all(math.isclose(a[k], b[k], abs_tol=1e-9) for k in a) and a.keys() == b.keys()

def test_hard_choice_uniform_smoothing():
    t = smooth_target({"A": 0.0, "B": 1.0, "C": 0.0}, "choice", eps_hard=0.06)
    assert close(t, {"A": 0.02, "B": 0.96, "C": 0.02})

def test_noul_is_smoothed_like_choice():
    t = smooth_target({"A": 1.0, "B": 0.0}, "noul", eps_hard=0.05)
    assert close(t, {"A": 0.975, "B": 0.025})

def test_soft_target_unchanged():
    soft = {"A": 0.67, "B": 0.33}
    assert smooth_target(soft, "choice") == soft

def test_score_middle_gold_splits_to_both_neighbours():
    t = smooth_target({"A": 0, "B": 0, "C": 1.0, "D": 0, "E": 0}, "score", eps_ordinal=0.1)
    assert close(t, {"A": 0, "B": 0.05, "C": 0.9, "D": 0.05, "E": 0})

def test_score_edge_gold_gives_all_to_single_neighbour():
    t = smooth_target({"A": 1.0, "B": 0, "C": 0}, "score", eps_ordinal=0.1)
    assert close(t, {"A": 0.9, "B": 0.1, "C": 0})

def test_sums_to_one():
    for tp in ("choice", "noul", "score"):
        t = smooth_target({"A": 0, "B": 1.0, "C": 0, "D": 0}, tp)
        assert math.isclose(sum(t.values()), 1.0)
```

- [ ] **Step 2: Run** `uv run pytest tests/train/test_targets.py -v`. Expected: FAIL with an import error.

- [ ] **Step 3: Implement**

```python
"""Training-time target smoothing (Plan 3 ruling 2)."""
from __future__ import annotations


def _is_hard(target: dict[str, float]) -> bool:
    vals = sorted(target.values())
    return vals[-1] == 1.0 and all(v == 0.0 for v in vals[:-1])


def smooth_target(target: dict[str, float], type_: str, eps_hard: float = 0.05, eps_ordinal: float = 0.10) -> dict[str, float]:
    if not _is_hard(target):
        return dict(target)
    letters = sorted(target)
    gold = max(letters, key=lambda L: target[L])
    if type_ == "score":
        i = letters.index(gold)
        neigh = [j for j in (i - 1, i + 1) if 0 <= j < len(letters)]
        out = {L: 0.0 for L in letters}
        out[gold] = 1.0 - eps_ordinal
        for j in neigh:
            out[letters[j]] += eps_ordinal / len(neigh)
        return out
    k = len(letters)
    return {L: (1.0 - eps_hard) + eps_hard / k if L == gold else eps_hard / k for L in letters}
```

- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git add src/jeff/train tests/train && git commit -m "feat(train): label and ordinal target smoothing"`.

---

### Task 2: Losses

**Files:**
- Create: `src/jeff/train/losses.py`
- Test: `tests/train/test_losses.py`

**Interfaces:**
- Consumes: nothing from earlier tasks. Plain tensors only.
- Produces, where each tensor is float32 and `mask` is True for valid letters:
  - `ce_soft(letter_logits: Tensor[B,K], target: Tensor[B,K], mask: Tensor[B,K], weight: Tensor[B]) -> Tensor[]`: the weighted mean over rows of −Σ target·log_softmax(masked logits).
  - `full_vocab_ce(vocab_logits: Tensor[B,V], answer_ids: Tensor[B], weight: Tensor[B]) -> Tensor[]`: weighted mean CE on the gold letter's token over the whole vocabulary.
  - `pair_margin(logp: Tensor[B,K], pairs: list[tuple[int,int,int,int]], margin: float = 1.0) -> Tensor[]`: each tuple is `(i, j, gi, gj)`, meaning rows i and j with gold letter indices gi and gj in their own orderings. The term is `relu(m − (logp[i,gi] − logp[j,gi_in_j])) + relu(m − (logp[j,gj] − logp[i,gj_in_i]))`. Callers pass the **key-aligned** indices, so tuples are `(i, j, a_i, a_j, b_i, b_j)`; see the code. It returns the mean over pairs, or 0.
  - `perm_skl(p_a: Tensor[N,K], p_b: Tensor[N,K]) -> Tensor[]`: symmetric KL between key-aligned distributions, mean over N, or 0 when N = 0.
  - `rps(probs: Tensor[B,K], target: Tensor[B,K], mask: Tensor[B,K]) -> Tensor[]`: ranked probability score over valid positions, averaged over rows.

- [ ] **Step 1: Write the failing tests**

```python
import math, torch
from jeff.train.losses import ce_soft, full_vocab_ce, pair_margin, perm_skl, rps

def test_ce_soft_ignores_masked_letters_and_weights_rows():
    logits = torch.tensor([[2.0, 0.0, 99.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    w = torch.tensor([1.0, 0.0])
    got = ce_soft(logits, target, mask, w)
    exp = -math.log(math.exp(2) / (math.exp(2) + 1))
    assert math.isclose(got.item(), exp, rel_tol=1e-5)

def test_full_vocab_ce_matches_cross_entropy():
    v = torch.randn(3, 11); ids = torch.tensor([1, 5, 7]); w = torch.ones(3)
    assert torch.allclose(full_vocab_ce(v, ids, w), torch.nn.functional.cross_entropy(v, ids))

def test_pair_margin_zero_when_separated():
    logp = torch.log(torch.tensor([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]]))
    # row0 gold key X at index0; row1 gold key Y at index1; key X in row1 is index0, key Y in row0 is index1
    assert pair_margin(logp, [(0, 1, 0, 0, 1, 1)], margin=1.0).item() == 0.0

def test_pair_margin_positive_when_identical():
    logp = torch.log(torch.full((2, 2), 0.5))
    assert pair_margin(logp, [(0, 1, 0, 0, 1, 1)], margin=1.0).item() == 2.0

def test_perm_skl_zero_for_equal_and_empty():
    p = torch.tensor([[0.2, 0.8]])
    assert perm_skl(p, p).item() == 0.0
    assert perm_skl(torch.zeros(0, 2), torch.zeros(0, 2)).item() == 0.0

def test_rps_perfect_is_zero_and_far_is_larger():
    mask = torch.ones(1, 3, dtype=torch.bool)
    t = torch.tensor([[0.0, 0.0, 1.0]])
    assert rps(t.clone(), t, mask).item() == 0.0
    near = rps(torch.tensor([[0.0, 1.0, 0.0]]), t, mask).item()
    far = rps(torch.tensor([[1.0, 0.0, 0.0]]), t, mask).item()
    assert far > near > 0
```

- [ ] **Step 2: Run** `uv run pytest tests/train/test_losses.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement**

```python
"""Losses at the answer-letter position (spec §5)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

NEG = -1e9


def masked_log_softmax(letter_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(letter_logits.masked_fill(~mask, NEG), dim=-1)


def ce_soft(letter_logits, target, mask, weight):
    logp = masked_log_softmax(letter_logits, mask)
    per_row = -(target * logp.masked_fill(~mask, 0.0)).sum(-1)
    return (per_row * weight).sum() / weight.sum().clamp_min(1e-8)


def full_vocab_ce(vocab_logits, answer_ids, weight):
    per_row = F.cross_entropy(vocab_logits.float(), answer_ids, reduction="none")
    return (per_row * weight).sum() / weight.sum().clamp_min(1e-8)


def pair_margin(logp, pairs, margin: float = 1.0):
    """pairs: (i, j, a_i, a_j, b_i, b_j) where a = row i's gold key, b = row j's gold key,
    and x_r is that key's letter index in row r."""
    if not pairs:
        return logp.new_zeros(())
    terms = []
    for i, j, a_i, a_j, b_i, b_j in pairs:
        terms.append(F.relu(margin - (logp[i, a_i] - logp[j, a_j])) + F.relu(margin - (logp[j, b_j] - logp[i, b_i])))
    return torch.stack(terms).mean()


def perm_skl(p_a, p_b, eps: float = 1e-8):
    if p_a.shape[0] == 0:
        return p_a.new_zeros(())
    pa, pb = p_a.clamp_min(eps), p_b.clamp_min(eps)
    kl_ab = (pa * (pa.log() - pb.log())).sum(-1)
    kl_ba = (pb * (pb.log() - pa.log())).sum(-1)
    return (0.5 * (kl_ab + kl_ba)).mean()


def rps(probs, target, mask):
    p = probs.masked_fill(~mask, 0.0).cumsum(-1)
    t = target.masked_fill(~mask, 0.0).cumsum(-1)
    k = mask.sum(-1).clamp_min(2).float()
    return (((p - t) ** 2).masked_fill(~mask, 0.0).sum(-1) / (k - 1)).mean()
```

- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** with `git commit -am "feat(train): answer-position losses"` (after `git add src/jeff/train/losses.py tests/train/test_losses.py`).

---

### Task 3: Data (parsing, twins, tokenisation, batching, subsampling)

**Files:**
- Create: `src/jeff/train/data.py`, `tests/train/conftest.py`
- Test: `tests/train/test_data.py`

**Interfaces:**
- Consumes: `jeff.format.LETTERS`; `smooth_target` (Task 1).
- Produces:
  - `load_rows(path: str) -> list[dict]`: reads `*.chat.jsonl`.
  - `make_twin(row: dict, rng: random.Random) -> dict | None`: re-shuffles the final user turn's options into a new letter order. It remaps `letters`, `target` and `answer`, sets `twin_of = row["id"]`, and returns None for `type == "score"` or rows with fewer than 2 options.
  - `letter_token_ids(tokenizer) -> list[int]`: the ids for "A"… "F" as generated after the assistant prompt. It asserts each letter is exactly one token and all ids are distinct.
  - `encode(row, tokenizer, max_len: int) -> dict | None`, returning:
    - `input_ids: list[int]`: `apply_chat_template(messages, add_generation_prompt=True, tokenize=True)`;
    - `answer_pos: int`, which is `len(input_ids) - 1`;
    - `n_letters: int`;
    - `target: list[float]` (length 6, zero-padded);
    - `answer_letter: int`;
    - `weight`, `type`, `id`, `unit`, `twin_of`, `letters`.

    It returns None if `len(input_ids) > max_len`.
  - `subsample_units(rows, fraction: float, seed: int) -> list[dict]`: keeps whole units, where a unit is `cluster_id` or else `id`. It keeps `ceil(fraction * n_units)` units, chosen with the seed.
  - `build_batches(encoded: list[dict], token_budget: int, seed: int) -> list[list[int]]`: indices grouped so that every member of a unit and every twin share one batch. Units are shuffled with the seed. Batches are packed until the next unit would exceed `token_budget`, counted as the sum of lengths. A single unit larger than the budget becomes its own batch.
  - `EncodeStats` counter: `{"kept", "dropped_overlong"}`, returned by `encode_all(rows, tokenizer, max_len_plain, max_len_examples, eps_hard, eps_ordinal, twin_rate, seed) -> tuple[list[dict], dict]`. It applies smoothing to each row's target (twins inherit smoothing), adds twins for a `twin_rate` fraction of rows (spec: 0.5, and 0 when L_perm is off), and uses `max_len_examples` for rows whose messages contain more than one user turn.

- [ ] **Step 1: Write `tests/train/conftest.py`.** It holds a fake tokenizer: a word-level vocab built on the fly, single-token letters, and a ChatML-like template. It also holds a tiny model factory.

```python
import json, pytest, torch

class FakeTok:
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1}
        for L in "ABCDEF":
            self.vocab[L] = len(self.vocab)
        self.pad_token_id = 0
        self.eos_token_id = 1
    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) % 2000 + 8 if len(self.vocab) >= 2000 else len(self.vocab)
        return self.vocab[w]
    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        words = []
        for m in messages:
            words += ["<|" + m["role"] + "|>"] + m["content"].replace('"', " ").split()
        if add_generation_prompt:
            words.append("<|assistant|>")
        return [self._id(w) for w in words]

@pytest.fixture
def tok():
    return FakeTok()

@pytest.fixture
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=2048, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return LlamaForCausalLM(cfg)

def chat_row(id_, letters, gold_letter, type_="choice", cluster=None, examples=0, state="the state text"):
    opts = [{"label": L, "key": k, "description": f"desc {k}"} for L, k in letters.items()]
    user = json.dumps({"state": state, "question": "q?", "options": opts})
    msgs = [{"role": "system", "content": "sys"}]
    for _ in range(examples):
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": "A"}]
    msgs.append({"role": "user", "content": user})
    return {"id": id_, "messages": msgs, "answer": gold_letter, "letters": letters,
            "target": {L: float(L == gold_letter) for L in letters}, "weight": 1.0, "type": type_,
            "family": "f", "source": "s", "cluster_id": cluster, "edit_type": None, "split": "train"}

@pytest.fixture
def make_row():
    return chat_row
```

- [ ] **Step 2: Write the failing tests** in `tests/train/test_data.py`.

```python
import random
from jeff.train.data import make_twin, letter_token_ids, encode, subsample_units, build_batches, encode_all

def test_letter_token_ids_single_and_distinct(tok):
    ids = letter_token_ids(tok)
    assert len(ids) == 6 and len(set(ids)) == 6

def test_twin_remaps_letters_and_target(make_row):
    r = make_row("r1", {"A": "x", "B": "y", "C": "z"}, "B")
    tw = make_twin(r, random.Random(3))
    assert tw["twin_of"] == "r1"
    gold_key = r["letters"][r["answer"]]
    assert tw["letters"][tw["answer"]] == gold_key
    assert tw["target"][tw["answer"]] == 1.0
    assert sorted(tw["letters"].values()) == ["x", "y", "z"]

def test_score_rows_get_no_twin(make_row):
    r = make_row("s1", {"A": "1", "B": "2", "C": "3"}, "C", type_="score")
    assert make_twin(r, random.Random(0)) is None

def test_encode_answer_pos_is_last_prompt_token(tok, make_row):
    e = encode(make_row("r", {"A": "x", "B": "y"}, "A"), tok, max_len=512)
    assert e["answer_pos"] == len(e["input_ids"]) - 1
    assert e["n_letters"] == 2 and len(e["target"]) == 6

def test_overlong_rows_dropped_and_counted(tok, make_row):
    long_row = make_row("L", {"A": "x", "B": "y"}, "A", state="w " * 600)
    enc, stats = encode_all([long_row, make_row("s", {"A": "x", "B": "y"}, "A")], tok, 200, 400, 0.05, 0.1, 0.0, 0)
    assert stats == {"kept": 1, "dropped_overlong": 1}

def test_subsample_keeps_whole_units(make_row):
    rows = [make_row(f"c{i}-{j}", {"A": "x", "B": "y"}, "A", cluster=f"c{i}") for i in range(10) for j in range(3)]
    sub = subsample_units(rows, 0.5, seed=1)
    clusters = {r["cluster_id"] for r in sub}
    assert len(clusters) == 5 and len(sub) == 15

def test_batches_keep_units_and_twins_together(tok, make_row):
    rows = [make_row(f"c{i}-{j}", {"A": "x", "B": "y", "C": "z"}, "A", cluster=f"c{i}") for i in range(6) for j in range(2)]
    enc, _ = encode_all(rows, tok, 512, 1024, 0.05, 0.1, 1.0, 0)
    batches = build_batches(enc, token_budget=120, seed=0)
    assert sorted(i for b in batches for i in b) == list(range(len(enc)))
    where = {i: bi for bi, b in enumerate(batches) for i in b}
    for i, e in enumerate(enc):
        mates = [k for k, f in enumerate(enc) if f["unit"] == e["unit"]]
        assert {where[k] for k in mates} == {where[i]}
```

- [ ] **Step 3: Run** `uv run pytest tests/train/test_data.py -v`. Expected: FAIL.

- [ ] **Step 4: Implement `src/jeff/train/data.py`**

```python
"""Chat rows → tokenised, smoothed, unit-grouped training batches (Plan 3 Task 3)."""
from __future__ import annotations

import json
import math
import random

from jeff.format import LETTERS
from jeff.train.targets import smooth_target

N_MAX = 6


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def unit_of(row: dict) -> str:
    return row.get("cluster_id") or row.get("twin_of") or row["id"]


def make_twin(row: dict, rng: random.Random) -> dict | None:
    if row["type"] == "score" or len(row["letters"]) < 2:
        return None
    msgs = [dict(m) for m in row["messages"]]
    user = json.loads(msgs[-1]["content"])
    opts = list(user["options"])
    for _ in range(10):
        rng.shuffle(opts)
        if [o["key"] for o in opts] != [o["key"] for o in user["options"]]:
            break
    for i, o in enumerate(opts):
        o["label"] = LETTERS[i]
    user["options"] = opts
    msgs[-1]["content"] = json.dumps(user, ensure_ascii=False)
    old_by_key = {k: L for L, k in row["letters"].items()}
    letters = {LETTERS[i]: o["key"] for i, o in enumerate(opts)}
    target = {L: row["target"][old_by_key[k]] for L, k in letters.items()}
    gold_key = row["letters"][row["answer"]]
    answer = next(L for L, k in letters.items() if k == gold_key)
    return {**row, "id": row["id"] + "#twin", "messages": msgs, "letters": letters, "target": target,
            "answer": answer, "twin_of": row["id"]}


def letter_token_ids(tokenizer) -> list[int]:
    ids = []
    for L in LETTERS[:N_MAX]:
        toks = tokenizer.encode(L, add_special_tokens=False)
        assert len(toks) == 1, f"letter {L!r} is {len(toks)} tokens"
        ids.append(toks[0])
    assert len(set(ids)) == N_MAX
    return ids


def encode(row: dict, tokenizer, max_len: int) -> dict | None:
    ids = tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    ids = list(ids)
    if len(ids) > max_len:
        return None
    n = len(row["letters"])
    target = [row["target"].get(LETTERS[i], 0.0) for i in range(n)] + [0.0] * (N_MAX - n)
    return {"input_ids": ids, "answer_pos": len(ids) - 1, "n_letters": n, "target": target,
            "answer_letter": LETTERS.index(row["answer"]), "weight": float(row.get("weight", 1.0)),
            "type": row["type"], "id": row["id"], "unit": unit_of(row), "twin_of": row.get("twin_of"),
            "letters": row["letters"], "cluster_id": row.get("cluster_id")}


def subsample_units(rows: list[dict], fraction: float, seed: int) -> list[dict]:
    units = sorted({unit_of(r) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(units)
    keep = set(units[: math.ceil(fraction * len(units))])
    return [r for r in rows if unit_of(r) in keep]


def encode_all(rows, tokenizer, max_len_plain, max_len_examples, eps_hard, eps_ordinal, twin_rate, seed):
    rng = random.Random(seed)
    out, stats = [], {"kept": 0, "dropped_overlong": 0}
    for r in rows:
        r = {**r, "target": smooth_target(r["target"], r["type"], eps_hard, eps_ordinal)}
        batch = [r]
        if twin_rate > 0 and rng.random() < twin_rate:
            tw = make_twin(r, rng)
            if tw:
                batch.append(tw)
        n_user = sum(m["role"] == "user" for m in r["messages"])
        max_len = max_len_examples if n_user > 1 else max_len_plain
        encs = [encode(x, tokenizer, max_len) for x in batch]
        if encs[0] is None:
            stats["dropped_overlong"] += 1
            continue
        stats["kept"] += 1
        out.extend(e for e in encs if e is not None)
    return out, stats


def build_batches(encoded: list[dict], token_budget: int, seed: int) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(encoded):
        groups.setdefault(e["unit"], []).append(i)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    batches, cur, cur_tok = [], [], 0
    for u in order:
        idx = groups[u]
        size = sum(len(encoded[i]["input_ids"]) for i in idx)
        if cur and cur_tok + size > token_budget:
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.extend(idx)
        cur_tok += size
    if cur:
        batches.append(cur)
    return batches
```

- [ ] **Step 5: Run** the tests. Expected: PASS. Then run `uv run pytest -q`; the full suite must still pass.
- [ ] **Step 6: Commit** with `feat(train): tokenisation, permuted twins and unit-grouped batching`.

---

### Task 4: Readout and metrics

**Files:**
- Create: `src/jeff/train/readout.py`, `src/jeff/train/metrics.py`
- Test: `tests/train/test_metrics.py`

**Interfaces:**
- Produces:
  - `readout.probs(letter_logits: list[float], n: int, temperature: float = 1.0) -> list[float]`: a softmax over the first n logits divided by T.
  - `metrics.evaluate(preds: list[dict]) -> dict`. Each pred is `{id, type, family, edit_type, source, cluster_id, letters, answer, probs}`, with probs in letter order. The report has:
    - `n`, `accuracy`, `brier` (multi-class, mean Σ(p − onehot)²), `ece` (15 equal-width bins on max-prob);
    - `selective`: `{threshold: 0.9, coverage, accuracy}`;
    - `group_accuracy`: DecideBench template groups for `source == "decidebench"`, cluster groups otherwise; only groups of size ≥ 2 count;
    - `by_family`, `by_edit_type`, `by_source`, each `{key: {n, accuracy}}`;
    - `decidebench`: the same headline metrics restricted to `source == "decidebench"`, plus `vs_tev: {accuracy_bar: 0.928, group_bar: 0.860, passes: bool}`.
  - `metrics.group_key(pred) -> str | None`: for DecideBench, the id with its last `-segment` removed; otherwise `cluster_id`.

- [ ] **Step 1: Write the failing tests**

```python
import math
from jeff.train.readout import probs
from jeff.train.metrics import evaluate, group_key

def p(id_, ans, pr, src="s", cl=None, fam="f", typ="choice"):
    return {"id": id_, "type": typ, "family": fam, "edit_type": None, "source": src, "cluster_id": cl,
            "letters": {"A": "x", "B": "y"}, "answer": ans, "probs": pr}

def test_probs_temperature():
    a = probs([2.0, 0.0, 99.0], n=2, temperature=2.0)
    assert math.isclose(a[0], math.exp(1) / (math.exp(1) + 1))

def test_accuracy_brier_selective():
    r = evaluate([p("1", "A", [0.95, 0.05]), p("2", "B", [0.6, 0.4])])
    assert r["accuracy"] == 0.5
    assert math.isclose(r["brier"], ((0.05**2 * 2) + (0.6**2 * 2)) / 2)
    assert r["selective"]["coverage"] == 0.5 and r["selective"]["accuracy"] == 1.0

def test_decidebench_template_groups():
    assert group_key(p("decidebench:returns_policy-t03-refund", "A", [1, 0], src="decidebench")) == "decidebench:returns_policy-t03"
    preds = [p("decidebench:x-t01-a", "A", [0.9, 0.1], src="decidebench"),
             p("decidebench:x-t01-b", "B", [0.9, 0.1], src="decidebench"),
             p("decidebench:x-t02-a", "A", [0.9, 0.1], src="decidebench"),
             p("decidebench:x-t02-b", "B", [0.1, 0.9], src="decidebench")]
    r = evaluate(preds)
    assert r["decidebench"]["group_accuracy"] == 0.5
    assert r["decidebench"]["vs_tev"]["passes"] is False

def test_ece_perfectly_calibrated_is_zero():
    r = evaluate([p(str(i), "A", [1.0, 0.0]) for i in range(5)])
    assert r["ece"] == 0.0
```

- [ ] **Step 2: Run** the tests. Expected: FAIL.

- [ ] **Step 3: Implement**

`readout.py`:
```python
"""Softmax over valid letter logits with a per-type temperature (spec §6)."""
from __future__ import annotations
import math


def probs(letter_logits: list[float], n: int, temperature: float = 1.0) -> list[float]:
    z = [x / temperature for x in letter_logits[:n]]
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]
```

`metrics.py`:
```python
"""Dev metrics (spec §7) and the TEV comparison."""
from __future__ import annotations
from collections import defaultdict
from jeff.format import LETTERS

TEV_ACC, TEV_GROUP = 0.928, 0.860


def group_key(pred: dict) -> str | None:
    if pred["source"] == "decidebench":
        return pred["id"].rsplit("-", 1)[0]
    return pred.get("cluster_id")


def _headline(preds: list[dict]) -> dict:
    n = len(preds)
    if n == 0:
        return {"n": 0}
    correct, brier, bins = [], 0.0, defaultdict(list)
    for x in preds:
        gi = LETTERS.index(x["answer"])
        pi = max(range(len(x["probs"])), key=lambda i: x["probs"][i])
        ok = pi == gi
        correct.append(ok)
        brier += sum((q - (1.0 if i == gi else 0.0)) ** 2 for i, q in enumerate(x["probs"]))
        conf = x["probs"][pi]
        bins[min(int(conf * 15), 14)].append((conf, ok))
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins.values())
    sel = [ok for x, ok in zip(preds, correct) if max(x["probs"]) >= 0.9]
    groups = defaultdict(list)
    for x, ok in zip(preds, correct):
        k = group_key(x)
        if k:
            groups[k].append(ok)
    g = [all(v) for v in groups.values() if len(v) >= 2]
    return {"n": n, "accuracy": sum(correct) / n, "brier": brier / n, "ece": ece,
            "selective": {"threshold": 0.9, "coverage": len(sel) / n, "accuracy": (sum(sel) / len(sel)) if sel else None},
            "group_accuracy": (sum(g) / len(g)) if g else None, "n_groups": len(g)}


def _by(preds, field):
    out = defaultdict(list)
    for x in preds:
        out[str(x.get(field))].append(x)
    return {k: {"n": len(v), "accuracy": _headline(v)["accuracy"]} for k, v in sorted(out.items())}


def evaluate(preds: list[dict]) -> dict:
    rep = _headline(preds)
    rep.update(by_family=_by(preds, "family"), by_edit_type=_by(preds, "edit_type"), by_source=_by(preds, "source"))
    db = [x for x in preds if x["source"] == "decidebench"]
    dbh = _headline(db)
    if db:
        dbh["vs_tev"] = {"accuracy_bar": TEV_ACC, "group_bar": TEV_GROUP,
                         "passes": dbh["accuracy"] > TEV_ACC and (dbh["group_accuracy"] or 0) > TEV_GROUP}
        dbh["by_family"] = _by(db, "family")
    rep["decidebench"] = dbh
    return rep
```

- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** with `feat(train): readout and dev metrics with TEV comparison`.

---

### Task 5: Per-type temperature calibration

**Files:**
- Create: `src/jeff/train/calibrate.py`
- Test: `tests/train/test_calibrate.py`

**Interfaces:**
- Consumes: `readout.probs` (Task 4).
- Produces:
  - `fit_temperatures(items: list[dict]) -> dict[str, float]`. Items are `{type, logits: list[float], n, answer_index}`. For each of `choice`, `noul` and `score` it returns the T in a log-spaced grid of 200 values over [0.05, 20] that minimises NLL, followed by a golden-section refinement. A type with no items gets 1.0.
  - `write_calibration(path, temps: dict, meta: dict)` writes `{"temperatures": temps, **meta}`.

- [ ] **Step 1: Write the failing tests**

```python
import math, random
from jeff.train.calibrate import fit_temperatures

def test_recovers_overconfident_temperature():
    rng = random.Random(0)
    items = []
    for _ in range(2000):
        true = [rng.gauss(0, 1) for _ in range(3)]
        m = max(true); e = [math.exp(x - m) for x in true]; s = sum(e); pr = [x / s for x in e]
        ans = rng.choices(range(3), weights=pr)[0]
        items.append({"type": "choice", "logits": [x * 3.0 for x in true], "n": 3, "answer_index": ans})
    t = fit_temperatures(items)["choice"]
    assert 2.5 < t < 3.5

def test_missing_type_defaults_to_one():
    t = fit_temperatures([{"type": "choice", "logits": [1.0, 0.0], "n": 2, "answer_index": 0}] * 10)
    assert t["noul"] == 1.0 and t["score"] == 1.0
```

- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement**

```python
"""Per-type temperature scaling on the calibration split (spec §5 Stage 2b, §6)."""
from __future__ import annotations
import json, math

TYPES = ("choice", "noul", "score")


def _nll(items, t):
    tot = 0.0
    for it in items:
        z = [x / t for x in it["logits"][: it["n"]]]
        m = max(z)
        lse = m + math.log(sum(math.exp(x - m) for x in z))
        tot += lse - z[it["answer_index"]]
    return tot / len(items)


def _fit(items):
    grid = [math.exp(math.log(0.05) + i * (math.log(20) - math.log(0.05)) / 199) for i in range(200)]
    best = min(grid, key=lambda t: _nll(items, t))
    lo, hi = best / 1.05, best * 1.05
    g = (math.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if _nll(items, a) < _nll(items, b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def fit_temperatures(items: list[dict]) -> dict[str, float]:
    out = {}
    for tp in TYPES:
        sub = [x for x in items if x["type"] == tp]
        out[tp] = _fit(sub) if sub else 1.0
    return out


def write_calibration(path, temps: dict, meta: dict) -> None:
    with open(path, "w") as f:
        json.dump({"temperatures": temps, **meta}, f, indent=2)
```

- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** with `feat(train): per-type temperature calibration`.

---

### Task 6: Trainer, inference and CLI

**Files:**
- Create: `src/jeff/train/trainer.py`, `src/jeff/train/infer.py`, `configs/stage0/lc25.json`, `lc50.json`, `lc100.json`, `ab_pair.json`, `ab_perm.json`, `ab_both.json`, `smoke.json`
- Modify: `src/jeff/cli.py` (add `train`, `eval` and `calibrate` subcommands), `pyproject.toml` (add a `train` extra: `["transformers>=5.0", "peft>=0.17", "accelerate>=1.0", "tensorboard>=2.17"]`)
- Test: `tests/train/test_trainer.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `trainer.TrainConfig`, a dataclass with these fields and defaults:
    - `name`, `model_path`, `train_path`, `out_dir`
    - `fraction=1.0`, `seed=0`, `epochs=1`, `lr=1e-4`, `warmup_ratio=0.03`, `weight_decay=0.0`
    - `token_budget=12288`, `grad_accum=3`
    - `max_len_plain=2048`, `max_len_examples=4096`
    - `eps_hard=0.05`, `eps_ordinal=0.10`
    - `lambda_vocab=0.1`, `lambda_pair=0.0`, `pair_margin=1.0`, `lambda_perm=0.0`, `twin_rate=0.0`, `lambda_rps=0.0`
    - `lora_r=64`, `lora_alpha=128`, `lora_dropout=0.05`, `lora=True`
    - `grad_checkpointing=True`, `bf16=True`, `save_every=500`, `log_every=10`, `max_steps=None`

    Loaded with `TrainConfig.from_json(path)`; unknown keys raise.
  - `trainer.train(cfg: TrainConfig, tokenizer=None, model=None) -> dict`. It returns `{steps, tokens, seconds, tokens_per_s, encode_stats}` and writes to `out_dir`:
    - `checkpoint-<step>/`, each with an adapter or model, `state.pt` (optimizer, scheduler, step, rng) and `config.json`;
    - `final/`;
    - `train_summary.json`;
    - `tb/`.

    Tests pass `tokenizer` and `model`; the CLI loads them from `model_path`.
  - `infer.letter_logits(model, tokenizer, rows: list[dict], max_len: int, batch_tokens: int = 16384) -> list[list[float]]`: one forward pass per row, returning the 6 letter logits at the last position.
  - CLI:
    - `jeff train --config configs/stage0/lc100.json`
    - `jeff eval --model <dir> --base <model_path> --data <dev.chat.jsonl> [--calibration <json>] --out <report.json>`
    - `jeff calibrate --model <dir> --base <model_path> --data <calibration.chat.jsonl> --out <calibration.json>`

    `--model` is a LoRA adapter dir or merged model dir. `--base` is needed for adapters.

**Loss per micro-batch.** `total = ce_soft + lambda_vocab·full_vocab_ce + lambda_pair·pair_margin + lambda_perm·perm_skl + lambda_rps·rps(score rows only)`.
- `pair_margin` pairs: rows in the batch sharing `cluster_id` with different gold keys, where neither row is a twin.
- `perm_skl` pairs: each twin with its original, key-aligned by `letters`.
- Loss is scaled by `1/grad_accum`, and the optimizer steps every `grad_accum` micro-batches.
- The scheduler is linear warmup (`warmup_ratio`) then cosine to 0, over `ceil(n_batches·epochs / grad_accum)` optimizer steps.

**Resume.** If `out_dir` contains `checkpoint-*`, load the latest one: model or adapter weights, optimizer, scheduler, step and RNG state. Batches are rebuilt with the same seed, and training skips `step·grad_accum` micro-batches.

- [ ] **Step 1: Write the failing tests** (CPU, tiny model, fake tokenizer):

```python
import json
from jeff.train.trainer import TrainConfig, train
from jeff.train.infer import letter_logits

def write_rows(path, make_row, n=24):
    with open(path, "w") as f:
        for i in range(n):
            gold = "A" if i % 2 else "B"
            f.write(json.dumps(make_row(f"c{i//2}-{i%2}", {"A": "x", "B": "y", "C": "z"}, gold, cluster=f"c{i//2}")) + "\n")

def cfg(tmp_path, **kw):
    base = dict(name="t", model_path="unused", train_path=str(tmp_path / "train.jsonl"), out_dir=str(tmp_path / "out"),
                lora=False, grad_checkpointing=False, bf16=False, token_budget=400, grad_accum=2, save_every=2, lr=1e-3)
    base.update(kw)
    return TrainConfig(**base)

def test_train_runs_and_loss_is_finite(tmp_path, tok, tiny_model, make_row):
    write_rows(tmp_path / "train.jsonl", make_row)
    out = train(cfg(tmp_path, lambda_pair=0.5, lambda_perm=0.2, twin_rate=0.5), tokenizer=tok, model=tiny_model)
    assert out["steps"] > 0 and out["tokens"] > 0
    assert (tmp_path / "out" / "final").exists()

def test_resume_continues_step_and_order(tmp_path, tok, tiny_model, make_row):
    write_rows(tmp_path / "train.jsonl", make_row)
    first = train(cfg(tmp_path, max_steps=2), tokenizer=tok, model=tiny_model)
    assert first["steps"] == 2
    again = train(cfg(tmp_path, max_steps=4), tokenizer=tok, model=tiny_model)
    assert again["steps"] == 4 and again["resumed_from"] == 2

def test_letter_logits_shape(tok, tiny_model, make_row):
    rows = [make_row("r", {"A": "x", "B": "y"}, "A")]
    out = letter_logits(tiny_model, tok, rows, max_len=512)
    assert len(out) == 1 and len(out[0]) == 6

def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"; p.write_text(json.dumps({"name": "x", "model_path": "m", "train_path": "t", "out_dir": "o", "bogus": 1}))
    import pytest
    with pytest.raises(TypeError):
        TrainConfig.from_json(str(p))
```

- [ ] **Step 2: Run** the tests. Expected: FAIL.

- [ ] **Step 3: Implement `infer.py`**

```python
"""Letter logits at the answer position, one forward pass per row (spec §6)."""
from __future__ import annotations
import torch
from jeff.train.data import letter_token_ids, N_MAX


@torch.no_grad()
def letter_logits(model, tokenizer, rows, max_len: int, batch_tokens: int = 16384) -> list[list[float]]:
    model.eval()
    ids_letters = torch.tensor(letter_token_ids(tokenizer))
    dev = next(model.parameters()).device
    out: list[list[float] | None] = [None] * len(rows)
    enc = []
    for i, r in enumerate(rows):
        ids = tokenizer.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True)
        ids = list(ids["input_ids"] if hasattr(ids, "input_ids") else ids)[-max_len:]
        enc.append((i, ids))
    enc.sort(key=lambda x: len(x[1]))
    pad = getattr(tokenizer, "pad_token_id", 0) or 0
    b = []
    def flush(batch):
        L = max(len(x[1]) for x in batch)
        inp = torch.full((len(batch), L), pad, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for k, (_, ids) in enumerate(batch):  # left-pad so the answer position is the last column
            inp[k, L - len(ids):] = torch.tensor(ids); att[k, L - len(ids):] = 1
        logits = model(input_ids=inp.to(dev), attention_mask=att.to(dev)).logits[:, -1, :].float().cpu()
        for k, (i, _) in enumerate(batch):
            out[i] = logits[k, ids_letters].tolist()
    for item in enc:
        if b and (len(b) + 1) * max(len(item[1]), max(len(x[1]) for x in b)) > batch_tokens:
            flush(b); b = []
        b.append(item)
    if b:
        flush(b)
    return out  # type: ignore[return-value]
```

- [ ] **Step 4: Implement `trainer.py`.** Follow the Interfaces block exactly. Core loop:

```python
"""Stage 0 training loop: LoRA, answer-position losses, unit-grouped batches, resumable (Plan 3 Task 6)."""
from __future__ import annotations

import dataclasses, json, math, os, random, time
from dataclasses import dataclass, fields
from pathlib import Path

import torch

from jeff.train import losses as L
from jeff.train.data import N_MAX, build_batches, encode_all, letter_token_ids, load_rows, subsample_units


@dataclass
class TrainConfig:
    name: str
    model_path: str
    train_path: str
    out_dir: str
    fraction: float = 1.0
    seed: int = 0
    epochs: int = 1
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    token_budget: int = 12288
    grad_accum: int = 3
    max_len_plain: int = 2048
    max_len_examples: int = 4096
    eps_hard: float = 0.05
    eps_ordinal: float = 0.10
    lambda_vocab: float = 0.1
    lambda_pair: float = 0.0
    pair_margin: float = 1.0
    lambda_perm: float = 0.0
    twin_rate: float = 0.0
    lambda_rps: float = 0.0
    lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    grad_checkpointing: bool = True
    bf16: bool = True
    save_every: int = 500
    log_every: int = 10
    max_steps: int | None = None

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        return cls(**json.loads(Path(path).read_text()))


def _load(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, dtype=torch.bfloat16 if cfg.bf16 else torch.float32)
    return tok, model


def _wrap(cfg, model):
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if cfg.lora:
        from peft import LoraConfig, get_peft_model
        if cfg.grad_checkpointing:
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    return model


def _collate(items, pad_id, device):
    Lmax = max(len(e["input_ids"]) for e in items)
    inp = torch.full((len(items), Lmax), pad_id, dtype=torch.long)
    att = torch.zeros((len(items), Lmax), dtype=torch.long)
    for k, e in enumerate(items):  # right-pad; answer_pos indexes the unpadded sequence
        n = len(e["input_ids"]); inp[k, :n] = torch.tensor(e["input_ids"]); att[k, :n] = 1
    return inp.to(device), att.to(device)


def _batch_loss(cfg, model, items, letter_ids, pad_id, device):
    inp, att = _collate(items, pad_id, device)
    logits = model(input_ids=inp, attention_mask=att).logits
    pos = torch.tensor([e["answer_pos"] for e in items], device=device)
    last = logits[torch.arange(len(items), device=device), pos].float()           # [B, V]
    letter = last[:, letter_ids]                                                   # [B, 6]
    mask = torch.tensor([[i < e["n_letters"] for i in range(N_MAX)] for e in items], device=device)
    target = torch.tensor([e["target"] for e in items], device=device)
    weight = torch.tensor([e["weight"] for e in items], device=device)
    answer_ids = letter_ids[torch.tensor([e["answer_letter"] for e in items], device=device)]
    loss = L.ce_soft(letter, target, mask, weight) + cfg.lambda_vocab * L.full_vocab_ce(last, answer_ids, weight)
    logp = L.masked_log_softmax(letter, mask)
    key_idx = [{k: "ABCDEF".index(Lt) for Lt, k in e["letters"].items()} for e in items]
    gold_key = [e["letters"]["ABCDEF"[e["answer_letter"]]] for e in items]
    if cfg.lambda_pair:
        pairs = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["twin_of"] or b["twin_of"] or not a["cluster_id"] or a["cluster_id"] != b["cluster_id"]:
                    continue
                if gold_key[i] == gold_key[j] or gold_key[i] not in key_idx[j] or gold_key[j] not in key_idx[i]:
                    continue
                pairs.append((i, j, key_idx[i][gold_key[i]], key_idx[j][gold_key[i]], key_idx[i][gold_key[j]], key_idx[j][gold_key[j]]))
        loss = loss + cfg.lambda_pair * L.pair_margin(logp, pairs, cfg.pair_margin)
    if cfg.lambda_perm:
        by_id = {e["id"]: k for k, e in enumerate(items)}
        pa, pb = [], []
        for k, e in enumerate(items):
            if e["twin_of"] and e["twin_of"] in by_id:
                o = by_id[e["twin_of"]]
                keys = list(items[o]["letters"].values())
                pa.append(torch.stack([logp[o, key_idx[o][x]] for x in keys]).exp())
                pb.append(torch.stack([logp[k, key_idx[k][x]] for x in keys]).exp())
        if pa:
            loss = loss + cfg.lambda_perm * L.perm_skl(torch.stack(pa), torch.stack(pb))
    if cfg.lambda_rps:
        sel = [k for k, e in enumerate(items) if e["type"] == "score"]
        if sel:
            s = torch.tensor(sel, device=device)
            loss = loss + cfg.lambda_rps * L.rps(logp[s].exp(), target[s], mask[s])
    return loss, int(att.sum())


def _latest_checkpoint(out: Path):
    cks = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return cks[-1] if cks else None


def _save(model, opt, sched, step, out: Path, cfg, name=None):
    d = out / (name or f"checkpoint-{step}")
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                "rng": random.getstate(), "torch_rng": torch.get_rng_state()}, d / "state.pt")
    (d / "config.json.train").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))


def train(cfg: TrainConfig, tokenizer=None, model=None) -> dict:
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if tokenizer is None or model is None:
        tokenizer, model = _load(cfg)
    rows = load_rows(cfg.train_path)
    if cfg.fraction < 1.0:
        rows = subsample_units(rows, cfg.fraction, cfg.seed)
    enc, stats = encode_all(rows, tokenizer, cfg.max_len_plain, cfg.max_len_examples, cfg.eps_hard, cfg.eps_ordinal, cfg.twin_rate, cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _wrap(cfg, model).to(device)
    letter_ids = torch.tensor(letter_token_ids(tokenizer), device=device)
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0
    batches = [b for ep in range(cfg.epochs) for b in build_batches(enc, cfg.token_budget, cfg.seed + ep)]
    total_steps = math.ceil(len(batches) / cfg.grad_accum)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_ratio * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    step, resumed_from = 0, None
    ck = _latest_checkpoint(out)
    if ck is not None:
        if cfg.lora:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(str(ck / "adapter_model.safetensors")))
        else:
            from transformers import AutoModelForCausalLM
            model.load_state_dict(AutoModelForCausalLM.from_pretrained(ck).state_dict())
        st = torch.load(ck / "state.pt", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = resumed_from = st["step"]; random.setstate(st["rng"]); torch.set_rng_state(st["torch_rng"])
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(out / "tb"))
    except Exception:
        pass
    model.train()
    t0, tokens, micro = time.time(), 0, step * cfg.grad_accum
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (cfg.bf16 and device == "cuda") else torch.autocast(device_type="cpu", enabled=False)
    while step < total_steps and micro < len(batches):
        items = [enc[i] for i in batches[micro]]
        with autocast:
            loss, ntok = _batch_loss(cfg, model, items, letter_ids, pad_id, device)
        (loss / cfg.grad_accum).backward()
        tokens += ntok; micro += 1
        if micro % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if writer and step % cfg.log_every == 0:
                writer.add_scalar("loss", loss.item(), step); writer.add_scalar("lr", sched.get_last_lr()[0], step)
            if step % cfg.save_every == 0:
                _save(model, opt, sched, step, out, cfg)
    _save(model, opt, sched, step, out, cfg, name="final")
    secs = time.time() - t0
    summary = {"steps": step, "tokens": tokens, "seconds": secs, "tokens_per_s": tokens / max(secs, 1e-9),
               "encode_stats": stats, "resumed_from": resumed_from, "n_batches": len(batches), "total_steps": total_steps}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
```

Adjust only what the tests prove necessary. For example, `save_pretrained` on a bare `LlamaForCausalLM` in the resume test must be matched by the full-model load branch.

- [ ] **Step 5: Add the CLI subcommands** in `src/jeff/cli.py`, following the existing argparse pattern:
  - `train` calls `train(TrainConfig.from_json(args.config))` and prints the summary JSON.
  - `eval`:
    - loads the base model and tokenizer from `--base` (bf16, CUDA if available), then applies `peft.PeftModel.from_pretrained(model, args.model)` when `adapter_config.json` exists in `--model`, otherwise loads `--model` directly;
    - reads rows, calls `letter_logits`, applies `readout.probs` with the type's temperature from `--calibration` (default 1.0);
    - builds `preds` and writes `metrics.evaluate(preds)` to `--out`.
  - `calibrate`: same loading, then builds items and writes `fit_temperatures` output with `write_calibration`, with meta `{model, data, n}`.
- [ ] **Step 6: Write the configs.** Every config sets `model_path: "/home/ubuntu/models/Qwen3.5-4B-Base"`, `train_path: "/home/ubuntu/jeff/data/mix/stage0/train.chat.jsonl"` and `out_dir: "/home/ubuntu/runs/<name>"`.

| file | name | fraction | lambda_pair | lambda_perm | twin_rate | lambda_rps | max_steps |
|---|---|---|---|---|---|---|---|
| smoke.json | smoke | 1.0 | 0.5 | 0.2 | 0.5 | 0.5 | 200 |
| lc25.json | lc25 | 0.25 | 0 | 0 | 0 | 0.5 | — |
| lc50.json | lc50 | 0.5 | 0 | 0 | 0 | 0.5 | — |
| lc100.json | lc100 | 1.0 | 0 | 0 | 0 | 0.5 | — |
| ab_pair.json | ab_pair | 1.0 | 0.5 | 0 | 0 | 0.5 | — |
| ab_perm.json | ab_perm | 1.0 | 0 | 0.2 | 0.5 | 0.5 | — |
| ab_both.json | ab_both | 1.0 | 0.5 | 0.2 | 0.5 | 0.5 | — |

  `lc100` doubles as ablation arm 1 (L_ce, plus L_rps on Score rows per spec §5).
- [ ] **Step 7: Run** `uv run pytest -q`. Expected: all pass, including the 4 new trainer tests on CPU.
- [ ] **Step 8: Commit** with `feat(train): training loop, inference, eval/calibrate CLI and Stage 0 configs`.

---

### Task 7: AWS scripts

**Files:**
- Create: `scripts/aws/sync.sh`, `scripts/aws/run.sh`, `scripts/aws/setup.sh`
- Test: `tests/test_aws_scripts.py`

**Interfaces:**
- `sync.sh`: requires `JEFF_TRAIN_HOST` and `JEFF_TRAIN_KEY`, and exits 2 with a message naming the missing variable. It then:
  - runs `git archive HEAD | ssh … 'mkdir -p ~/jeff && tar -x -C ~/jeff'`;
  - rsyncs `data/mix/stage0/` and `data/dev/decidebench_examples.jsonl` to `~/jeff/data/…`;
  - never sends anything else under `data/`.
- `run.sh <cmd…>`: same env checks. It runs `cd ~/jeff && ~/venv/bin/pip install -q -e '.[train]' --no-deps && <cmd>` over ssh inside `nohup … > ~/runs/<first arg basename>.log 2>&1 &` when `--bg` is the first argument, and in the foreground otherwise.
- `setup.sh`: the remote venv recipe already run on the box. It builds an **isolated** venv (no system site packages: Ubuntu's `cryptography` and `pyOpenSSL` break transformers' imports) from `/opt/pytorch/bin/python`, then runs `pip install torch --index-url https://download.pytorch.org/whl/cu130` and the `train` extra's packages, then `hf download Qwen/Qwen3.5-4B-Base --local-dir ~/models/Qwen3.5-4B-Base`. It is kept so the box can be rebuilt.

- [ ] **Step 1: Write the failing test.** Both scripts exit 2 when the environment variables are unset, and no script contains an IP address or a `.pem` path.

```python
import os, re, subprocess
from pathlib import Path

def test_scripts_require_env_and_contain_no_host():
    for s in ("sync.sh", "run.sh"):
        p = Path("scripts/aws") / s
        r = subprocess.run(["bash", str(p), "echo"], env={"PATH": os.environ["PATH"]}, capture_output=True, text=True)
        assert r.returncode == 2 and "JEFF_TRAIN_" in r.stderr
    for p in Path("scripts/aws").glob("*.sh"):
        t = p.read_text()
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", t) and ".pem" not in t
```

- [ ] **Step 2: Run** the test. Expected: FAIL.
- [ ] **Step 3: Implement the three scripts** (`set -euo pipefail`; `: "${JEFF_TRAIN_HOST:?JEFF_TRAIN_HOST is not set}"` style checks, but exit 2: `[ -n "${JEFF_TRAIN_HOST:-}" ] || { echo "JEFF_TRAIN_HOST is not set" >&2; exit 2; }`).
- [ ] **Step 4: Run** the test. Expected: PASS.
- [ ] **Step 5: Commit** with `feat(aws): sync and run scripts for the training box`.

---

### Task 8: Runs on the box (controller runbook, not a subagent code task)

Before starting, set `JEFF_TRAIN_HOST` and `JEFF_TRAIN_KEY` from the controller's memory notes. They are never written into the repo.

- [ ] **Step 1: Sync and smoke test.**
  1. `scripts/aws/sync.sh`
  2. `scripts/aws/run.sh jeff train --config configs/stage0/smoke.json`

  Expected: 200 steps, finite loss, and a `train_summary.json` with `tokens_per_s`. Record `encode_stats.dropped_overlong`; if it is above 0.5 % of rows, report it. Estimate hours per full epoch as `train tokens / tokens_per_s`. If that is over 6 h, STOP and report before the learning curve, with the dollar estimate at the spot price.
- [ ] **Step 2: Real-model tokenizer check.** On the box, run `~/venv/bin/python -c` to call `letter_token_ids` on the Qwen tokenizer. It must pass. If a letter isn't a single token, STOP and report.
- [ ] **Step 3: Learning curve.** Run `lc25`, `lc50` and `lc100` one after another with `run.sh --bg`, watching the logs. After each run:
  1. `jeff calibrate --model ~/runs/<name>/final --base ~/models/Qwen3.5-4B-Base --data ~/jeff/data/mix/stage0/calibration.chat.jsonl --out ~/runs/<name>/calibration.json`
  2. `jeff eval … --data ~/jeff/data/mix/stage0/dev.chat.jsonl --calibration ~/runs/<name>/calibration.json --out ~/runs/<name>/dev_report.json`
  3. Copy both JSON files back to `data/runs/<name>/` locally (git-ignored).
- [ ] **Step 4: Learning-curve checkpoint.** Report to the user:
  - a table over 25/50/100 % of overall accuracy, Opus-holdout group accuracy, DecideBench accuracy and template-group accuracy, ECE and Brier;
  - the slope from 50 % to 100 %;
  - hours and dollars spent.

  Continue to the ablation without waiting, unless the 100 % run is worse than 50 % on DecideBench accuracy by more than 2 points; that points to a data problem, so STOP.
- [ ] **Step 5: Ablation.** Run `ab_pair`, `ab_perm` and `ab_both`, each followed by the same calibrate and eval. The winner is the highest DecideBench template-group accuracy, with ties broken by DecideBench accuracy and then Opus-holdout group accuracy.
- [ ] **Step 6: Report** to the user:
  - an ablation table with all four arms (`lc100` is arm 1);
  - the winner's full dev report: per family, per edit type, calibration metrics and selective accuracy;
  - the `vs_tev` pass/fail against 92.8 % / 86.0 %;
  - total GPU hours and dollars.

  Checkpoints stay on the box, and nothing is uploaded. Offer to stop the instance.

---

## Self-review

- **Spec coverage:**
  - §5 losses: L_ce with the 0.1 vocab term, L_pair, L_perm and L_rps are in Tasks 2 and 6.
  - §5 batching (units and twins share a micro-batch) is Task 3.
  - §5 checkpointing, TensorBoard and resume are Task 6.
  - §5 hyperparameters: effective batch ≈ 64–80 sequences (12,288-token micro-batches ≈ 25–29 sequences at the ~420-token median, × `grad_accum` 3); max length 2,048/4,096 covered.
  - §5 Stage 0 ablation is Task 8 Step 5.
  - §5 smoke test is Task 8 Step 1.
  - §6 readout with per-type temperature is Tasks 4 and 5.
  - §7 dev report metrics are Task 4.
  - Deferred to later plans: §6 vLLM serving and `/v1/systemone` (Plan 4), Stage 1/2/2b, the final DecideBench test run.
- **Placeholders:** none. Configs are fully tabulated.
- **Type consistency:**
  - `encode` output keys are consumed by `_batch_loss`: `input_ids`, `answer_pos`, `n_letters`, `target`, `answer_letter`, `weight`, `type`, `id`, `unit`, `twin_of`, `letters` and `cluster_id`.
  - `letter_token_ids` is used by `infer` and `trainer`.
  - `probs` and `fit_temperatures` are used by the CLI.
- **Review Focus:** each of the five items has a named test in its owning task (3, 3, 3, 5, 6).

