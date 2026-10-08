"""Build the Stage 1b pools: the Stage 1 pools without date and arithmetic material.

Production computes date and amount deltas with rules, so the model no longer needs to learn the arithmetic.
yev1-4b (Stage 1) learned it poorly (date drills 49%, returns_v2 64%) and regressed on DecideBench dev.

Removed:
- the returns_v2 and date_number_drills families;
- long_workflow questions labelled `arithmetic` or `temporal`;
- invoice_ledger questions that need arithmetic (totals, tax, due dates, discounts, quantities, variances,
  line checks); approval-band and bank-detail questions stay;
- long_padding rows padded from returns_v2 or date_number_drills, and any padded row whose distractor
  document no longer has a question in the pool (the mix's leak check needs the document present).

Usage: uv run python scripts/stage1b_prune.py   (reads data/stage1-pool, writes data/stage1b-pool)
"""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
from pathlib import Path

SRC, DST = Path("data/stage1-pool"), Path("data/stage1b-pool")
DROP_FAMILIES = {"returns_v2", "date_number_drills"}
DROP_SKILLS = {"arithmetic", "temporal"}
KEEP_INVOICE_KINDS = {"approval", "bank_match"}


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in open(SRC / f"{name}.jsonl")]


def write(name: str, data: list[dict]) -> None:
    with open(DST / f"{name}.jsonl", "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")


def doc_id(state: str) -> str:  # same hash as yev.generators.padding.doc_id
    return "doc:" + hashlib.sha1(state.encode()).hexdigest()[:10]


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    report = collections.OrderedDict()
    for f in sorted(SRC.glob("*.jsonl")):
        name = f.stem
        if name in DROP_FAMILIES or name in ("long_workflow", "invoice_ledger", "long_padding", "distractor_docs"):
            continue
        shutil.copy(f, DST / f.name)
        report[name] = {"kept": sum(1 for _ in open(f))}
    for name in DROP_FAMILIES:
        report[name] = {"kept": 0, "dropped": len(rows(name))}

    lw = rows("long_workflow")
    lw_keep = [r for r in lw if (r.get("metadata") or {}).get("skill") not in DROP_SKILLS]
    inv = rows("invoice_ledger")
    inv_keep = [r for r in inv if r["metadata"].get("kind") in KEEP_INVOICE_KINDS]
    write("long_workflow", lw_keep)
    write("invoice_ledger", inv_keep)
    report["long_workflow"] = {"kept": len(lw_keep), "dropped": len(lw) - len(lw_keep)}
    report["invoice_ledger"] = {"kept": len(inv_keep), "dropped": len(inv) - len(inv_keep)}

    present = {doc_id(r["state"]) for r in lw_keep + inv_keep}
    pad = rows("long_padding")
    def orig(r):  # orig_family is set for public sources only; generated rows carry the family in orig_id
        m = r["metadata"]
        return m.get("orig_family") or m.get("orig_id", "").split(":")[0]
    pad_keep = [r for r in pad if orig(r) not in DROP_FAMILIES
                and set(r["metadata"].get("distractor_ids", [])) <= present]
    write("long_padding", pad_keep)
    report["long_padding"] = {"kept": len(pad_keep), "dropped": len(pad) - len(pad_keep)}

    (DST / "prune_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
