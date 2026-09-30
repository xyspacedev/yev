"""Assemble stage datasets from a JSON recipe and render them in TEV's chat format (spec §4.6-4.7)."""

from __future__ import annotations

import glob
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from jeff.filters.contamination import Fingerprints, normalize_tokens
from jeff.filters.pipeline import assert_clean
from jeff.format import ExamplePool, render
from jeff.schema import Decision, read_jsonl, write_jsonl


def state_key(d: Decision) -> str:
    return hashlib.sha1(" ".join(normalize_tokens(d.state)).encode()).hexdigest()


def unit_key(d: Decision) -> str:
    return f"cluster:{d.cluster_id}" if d.cluster_id else f"state:{state_key(d)}"


def holdout_key(d: Decision) -> str:
    if d.source == "synthetic_opus" and d.cluster_id:
        return "batch:" + d.cluster_id.rsplit(":", 1)[0]
    return unit_key(d)


def load(patterns: list[str]) -> list[Decision]:
    paths = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not paths:
        raise FileNotFoundError(f"no files match {patterns}")
    return [d for p in paths for d in read_jsonl(p)]


def _units(decisions: list[Decision]) -> dict[str, list[Decision]]:
    units: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        units[unit_key(d)].append(d)
    return units


def take_units(decisions: list[Decision], rows: int, rng: random.Random) -> list[Decision]:
    units = _units(decisions)
    keys = sorted(units)
    rng.shuffle(keys)
    out: list[Decision] = []
    for k in keys:
        if len(out) >= rows:
            break
        out.extend(units[k])
    return out


def _summary(rows: list[Decision]) -> dict:
    return {
        "rows": len(rows),
        "by_family": dict(Counter(d.family for d in rows)),
        "by_type": dict(Counter(d.type for d in rows)),
        "by_source": dict(Counter(d.source for d in rows)),
        "soft_labels": sum(d.soft_gold is not None for d in rows),
        "downweighted": sum(d.weight < 1 for d in rows),
        "clusters": len({d.cluster_id for d in rows if d.cluster_id}),
    }


@dataclass
class MixResult:
    train: list[Decision]
    dev: list[Decision]
    calibration: list[Decision]
    report: dict = field(default_factory=dict)


def build_mix(recipe: dict, fingerprints: Fingerprints) -> MixResult:
    seed = recipe.get("seed", 0)
    report: dict = {"name": recipe.get("name"), "blocks": {}}
    train: list[Decision] = []
    dev: list[Decision] = []
    for block in recipe["blocks"]:
        rng = random.Random(f"{seed}:{block['name']}")
        pool = load(block["files"])
        available = len(pool)
        held: list[Decision] = []
        share = block.get("dev_share", 0.0)
        if share > 0:
            hkeys = sorted({holdout_key(d) for d in pool})
            rng.shuffle(hkeys)
            chosen = set(hkeys[: math.ceil(share * len(hkeys))])
            held = [replace(d, split="dev") for d in pool if holdout_key(d) in chosen]
            pool = [d for d in pool if holdout_key(d) not in chosen]
        taken = [replace(d, split="train") for d in take_units(pool, block["rows"], rng)]
        train.extend(taken)
        dev.extend(held)
        report["blocks"][block["name"]] = {
            "available": available, "train": len(taken), "dev": len(held),
            "shortfall": max(0, block["rows"] - len(taken)),
        }
    for pattern in recipe.get("dev_files", []):
        dev.extend(replace(d, split="dev") for d in load([pattern]))

    overlap = {unit_key(d) for d in train} & {unit_key(d) for d in dev}
    if overlap:
        raise ValueError(f"{len(overlap)} units appear in both train and dev")

    calibration: list[Decision] = []
    if "calibration" in recipe:
        cal = recipe["calibration"]
        used = {state_key(d) for d in train + dev}
        units = _units(load(cal["files"]))
        clean = [d for members in units.values() if not any(state_key(x) in used for x in members) for d in members]
        rng = random.Random(f"{seed}:calibration")
        calibration = [replace(d, split="calibration") for d in take_units(clean, cal["rows"], rng)]

    assert_clean(train + calibration + [d for d in dev if d.source != "decidebench"], fingerprints)
    for name, rows in (("train", train), ("dev", dev), ("calibration", calibration)):
        report[name] = _summary(rows)
    return MixResult(train, dev, calibration, report)


def write_mix(result: MixResult, recipe: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = ExamplePool(result.train)
    rate = recipe.get("examples_rate", 0.0)
    rng = random.Random(f"{recipe.get('seed', 0)}:format")
    counts: Counter = Counter()
    for split, rows in (("train", result.train), ("dev", result.dev), ("calibration", result.calibration)):
        write_jsonl(out / f"{split}.jsonl", rows)
        with (out / f"{split}.chat.jsonl").open("w", encoding="utf-8") as f:
            for d in rows:
                examples = None
                if split != "calibration" and rng.random() < rate:
                    examples = pool.pick(d, rng)
                counts[f"{split}_examples"] += int(examples is not None)
                f.write(json.dumps(render(d, rng, examples), ensure_ascii=False) + "\n")
    report = dict(result.report)
    report["format"] = {"train_examples": counts["train_examples"], "dev_examples": counts["dev_examples"]}
    (out / "mix_report.json").write_text(json.dumps(report, indent=2))
    return report
