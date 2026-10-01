"""DynaBench test split (montehoover/DynaBench, config DynaBench, split test; MIT).

Converted with the training converter (`jeff.sources.moderation.convert_dynabench`) so the prompt matches
training: yes = PASS (every agent reply follows the policy), no = FAIL. The DynaGuard paper's F1 takes FAIL
(a violation) as positive, so the positive key here is `no`, predicted when P(no) >= 0.5.
No state-length cap (training drops states over 6000 chars; the benchmark keeps every item).
"""
from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

from jeff.bench import score as S
from jeff.bench.common import META_FILE, ROWS_FILE, chat_row, write_json, write_rows
from jeff.sources.moderation import convert_dynabench

NAME = "dynabench_test"
HF_ID = "montehoover/DynaBench"
CONFIG = "DynaBench"
REVISION = "b5bf2061252ca634708c6bbd178c1609505ba6cd"
POSITIVE = "no"  # FAIL / violation
REFERENCE = {
    "metric": "F1 (%), FAIL (violation) positive; DynaGuard arXiv 2509.02563 Table 3",
    "f1": {"DynaGuard-4B": 72.0, "DynaGuard-8B": 73.1, "DynaGuard-1.7B": 65.2, "GPT-4o-mini": 70.1,
           "Qwen3-8B": 60.7, "GuardReasoner-8B": 22.0, "ShieldGemma": 38.2, "NemoGuard": 23.7,
           "WildGuard": 20.9, "LlamaGuard3": 13.1},
    "flag": "in-distribution: yev-4b and DynaGuard are both fine-tuned on DynaBench train (rules disjoint from test)",
}


def convert(raw: list[dict]) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    rng = random.Random(0)  # convert_dynabench does not use it
    for i, r in enumerate(raw):
        got = convert_dynabench(r, i, rng, [])
        if not got:
            skipped += 1
            continue
        d = replace(got[0], id=f"{NAME}:{i}", source=NAME, licence="mit", split="dev")
        rows.append(chat_row(d, {"label": r["label"], "base_id": r.get("base_id")}))
    return rows, skipped


def build(out: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset(HF_ID, CONFIG, split="test", revision=REVISION)
    rows, skipped = convert(list(ds))
    n = write_rows(out / ROWS_FILE, rows)
    meta = {
        "name": NAME, "source": f"https://huggingface.co/datasets/{HF_ID} ({CONFIG}, test)", "revision": REVISION,
        "licence": "MIT", "n": n, "skipped": skipped,
        "notes": ["Training converter convert_dynabench reused (yes = PASS, no = FAIL); positive = no.",
                  "No 6000-char state cap; long items need --max-len >= 8192 at score time."],
    }
    write_json(out / META_FILE, meta)
    return meta


def metrics(preds: list[dict], variant: str | None = None) -> dict:
    return {**S.common(preds), **{k: v for k, v in S.binary(preds, POSITIVE).items() if k != "n"},
            "reference": REFERENCE}
