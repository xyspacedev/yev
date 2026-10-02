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

from yev.filters.contamination import Fingerprints, normalize_tokens
from yev.filters.pipeline import assert_clean
from yev.format import ExamplePool, render
from yev.licences import is_allowed
from yev.schema import Decision, read_jsonl, write_jsonl


def state_key(d: Decision) -> str:
    return hashlib.sha1(" ".join(normalize_tokens(d.state)).encode()).hexdigest()


def unit_key(d: Decision) -> str:
    return f"cluster:{d.cluster_id}" if d.cluster_id else f"state:{state_key(d)}"


def holdout_key(d: Decision) -> str:
    if d.source == "synthetic_opus" and d.cluster_id:
        return "batch:" + d.cluster_id.rsplit(":", 1)[0]
    return unit_key(d)


def load(patterns: list[str]) -> list[Decision]:
    paths: set[str] = set()
    for pat in patterns:
        found = glob.glob(pat)
        if not found:
            raise FileNotFoundError(f"no files match {pat}")
        paths.update(found)
    return [d for p in sorted(paths) for d in read_jsonl(p)]


def _stratified_holdout(pool: list[Decision], share: float, rng: random.Random, min_rows: int = 0,
                        eligible=lambda d: True) -> set[str]:
    """Holdout keys, `share` of them per family (a key's family is its rows' most common one), adding keys
    until the family's held-out rows reach `min_rows`. Keys with any ineligible row are never held out."""
    fam_counts: dict[str, Counter] = defaultdict(Counter)
    blocked: set[str] = set()
    for d in pool:
        fam_counts[holdout_key(d)][d.family] += 1
        if not eligible(d):
            blocked.add(holdout_key(d))
    by_family: dict[str, list[str]] = defaultdict(list)
    for key in sorted(fam_counts):
        by_family[fam_counts[key].most_common(1)[0][0]].append(key)
    chosen: set[str] = set()
    for family in sorted(by_family):
        keys = by_family[family]
        rng.shuffle(keys)
        keys = [k for k in keys if k not in blocked]
        n = math.ceil(share * len(by_family[family]))
        rows = sum(fam_counts[k].total() for k in keys[:n])
        while n < len(keys) and rows < min_rows:
            rows += fam_counts[keys[n]].total()
            n += 1
        chosen.update(keys[:n])
    return chosen


def _links(d: Decision) -> set[str]:
    """What ties a row to rows of other blocks: its id, its original's id (reframed or padded rows), its state."""
    out = {f"id:{d.id}", f"state:{state_key(d)}"}
    orig = (d.metadata or {}).get("orig_id")
    if orig:
        out.add(f"id:{orig}")
    return out


def _block_refs(block: dict, key: str, done: dict[str, tuple[list[Decision], list[Decision]]]) -> list[str]:
    names = block.get(key, [])
    missing = [n for n in names if n not in done]
    if missing:
        raise ValueError(f"block {block['name']}: {key} names {missing}, which is not an earlier block")
    return names


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
    done: dict[str, tuple[list[Decision], list[Decision]]] = {}
    for block in recipe["blocks"]:
        rng = random.Random(f"{seed}:{block['name']}")
        pool = load(block["files"])
        available = len(pool)
        # exclude_from: drop every row tied to a row an earlier block took (no utterance twice).
        # split_follows: drop rows tied to an earlier block's dev rows; rows tied to its train rows stay out of dev.
        banned: set[str] = set()
        for name in _block_refs(block, "exclude_from", done):
            banned.update(k for d in done[name][0] + done[name][1] for k in _links(d))
        train_links: set[str] = set()
        for name in _block_refs(block, "split_follows", done):
            banned.update(k for d in done[name][1] for k in _links(d))
            train_links.update(k for d in done[name][0] for k in _links(d))
        excluded = 0
        if banned:
            # whole units go: a unit with any banned row is dropped
            bad_units = {unit_key(d) for d in pool if _links(d) & banned}
            kept = [d for d in pool if unit_key(d) not in bad_units]
            excluded, pool = len(pool) - len(kept), kept
        held: list[Decision] = []
        share = block.get("dev_share", 0.0)
        if share > 0:
            chosen = _stratified_holdout(pool, share, rng, block.get("dev_min_rows", 0),
                                         eligible=lambda d: not (_links(d) & train_links))
            held = [replace(d, split="dev") for d in pool if holdout_key(d) in chosen]
            pool = [d for d in pool if holdout_key(d) not in chosen]
        taken = [replace(d, split="train") for d in take_units(pool, block["rows"], rng)]
        train.extend(taken)
        dev.extend(held)
        done[block["name"]] = (taken, held)
        info = {
            "available": available, "train": len(taken), "dev": len(held),
            "shortfall": max(0, block["rows"] - len(taken)),
        }
        if "exclude_from" in block or "split_follows" in block:
            info["excluded"] = excluded
        if block.get("report_gold"):
            info["train_gold"] = dict(Counter(d.gold for d in taken))
            info["dev_gold"] = dict(Counter(d.gold for d in held))
        report["blocks"][block["name"]] = info
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

    bad = sorted({d.licence for d in train + calibration if not is_allowed(d.licence)})
    if bad:
        raise ValueError(f"train/calibration rows carry disallowed licences: {bad}")

    assert_clean(train + calibration + [d for d in dev if d.source != "decidebench"], fingerprints)
    for name, rows in (("train", train), ("dev", dev), ("calibration", calibration)):
        report[name] = _summary(rows)
    return MixResult(train, dev, calibration, report)


def write_mix(result: MixResult, recipe: dict, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Long families (rows near the length limit) neither get worked examples nor serve as examples.
    no_examples = set(recipe.get("examples_exclude_families", []))
    pool = ExamplePool([d for d in result.train if d.family not in no_examples])
    rate = recipe.get("examples_rate", 0.0)
    rng = random.Random(f"{recipe.get('seed', 0)}:format")
    counts: Counter = Counter()
    picks: dict[str, dict[str, list[Decision]]] = {}
    for split, rows in (("train", result.train), ("dev", result.dev)):
        cand = {d.id: None if d.family in no_examples else pool.pick(d, rng) for d in rows}
        eligible = [d.id for d in rows if cand[d.id] is not None]
        n = min(len(eligible), round(rate * len(rows)))
        chosen = set(rng.sample(eligible, n))
        picks[split] = {i: cand[i] for i in chosen}
        counts[f"{split}_eligible"] = len(eligible)
        counts[f"{split}_examples"] = n
    for split, rows in (("train", result.train), ("dev", result.dev), ("calibration", result.calibration)):
        write_jsonl(out / f"{split}.jsonl", rows)
        with (out / f"{split}.chat.jsonl").open("w", encoding="utf-8") as f:
            for d in rows:
                f.write(json.dumps(render(d, rng, picks.get(split, {}).get(d.id)), ensure_ascii=False) + "\n")
    report = dict(result.report)
    report["format"] = {k: counts[k] for k in ("train_examples", "train_eligible", "dev_examples", "dev_eligible")}
    (out / "mix_report.json").write_text(json.dumps(report, indent=2))
    return report
