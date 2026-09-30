"""Edit attribution: a fresh checker names the edit between base and variant; mismatches are dropped."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import random

from jeff.generators.common import (
    EDIT_TIE_BREAKS, EDIT_TYPE_DEFINITIONS, NUMERIC_GROUP, keep_valid_clusters, spread_into_parts,
)
from jeff.schema import Decision



def prepare_pairs(decisions: list[Decision], part_size: int = 100, seed: int = 0) -> tuple[list[tuple[str, list[dict]]], dict]:
    clusters: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        clusters[d.cluster_id].append(d)
    key: dict = {"_no_base": []}
    items: list[dict] = []
    owner: list[str] = []
    for cid in sorted(clusters):
        members = clusters[cid]
        base = next((d for d in members if d.edit_type is None), None)
        if base is None:
            key["_no_base"].extend(d.id for d in members)
            continue
        for d in members:
            if d is base:
                continue
            pair_id = hashlib.sha1(f"{seed}:pair:{d.id}".encode()).hexdigest()[:12]
            key[pair_id] = {"decision": d.id, "edit_type": d.edit_type}
            items.append({
                "pair_id": pair_id,
                "a": {"state": base.state, "options": [o.description for o in base.options]},
                "b": {"state": d.state, "options": [o.description for o in d.options]},
            })
            owner.append(cid)
    rng = random.Random(f"{seed}:pairs")
    parts = [(f"pairs-part-{i}", part) for i, part in enumerate(spread_into_parts(items, owner, part_size, rng))]
    return parts, key


def render_attrib_prompt(pairs_path: str, answers_path: str) -> str:
    definitions = "\n".join(f"- `{name}`: {text}" for name, text in EDIT_TYPE_DEFINITIONS.items())
    return f"""# Name the edit between two versions

Read `{pairs_path}`. Each line is a pair: version `a` and version `b` of the same decision item, each with an input
`state` and its option rule texts. Compare them and name the ONE kind of edit that turns `a` into `b`:

{definitions}

{EDIT_TIE_BREAKS}

Do not open any other file or directory.

Write your answers to `{answers_path}` with the Write tool: one JSON object per line, exactly
{{"pair_id": "<the pair's id>", "edit_type": "<one name from the list>"}}, one line per pair, no prose.
Then reply with only the number of answers written.
"""


def score_attribution(decisions: list[Decision], key: dict, answers: list[dict]) -> tuple[list[Decision], Counter]:
    stats: Counter = Counter()
    answered: dict[str, str] = {}
    for a in answers:
        if not isinstance(a, dict):
            stats["non_object_answer"] += 1
            continue
        pair_id = str(a.get("pair_id", ""))
        if pair_id == "_no_base" or pair_id not in key:
            stats["unknown_pair"] += 1
        elif pair_id in answered:
            stats["duplicate_answer"] += 1
        else:
            answered[pair_id] = str(a.get("edit_type", "")).strip().lower()
    verdict: dict[str, bool] = {}
    for pair_id, entry in key.items():
        if pair_id == "_no_base":
            continue
        got = answered.get(pair_id)
        want = entry["edit_type"]
        if got is None:
            stats["attrib_missing"] += 1
            verdict[entry["decision"]] = False
        elif got == want:
            verdict[entry["decision"]] = True
        elif got in NUMERIC_GROUP and want in NUMERIC_GROUP:
            stats["attrib_numeric_group_match"] += 1
            verdict[entry["decision"]] = True
        else:
            stats["attrib_mismatch"] += 1
            verdict[entry["decision"]] = False
    no_base = set(key.get("_no_base", []))
    stats["no_base"] = len(no_base)
    survivors = [d for d in decisions if d.id not in no_base and (d.edit_type is None or verdict.get(d.id, False))]
    kept = keep_valid_clusters(survivors)
    stats["cluster_collapsed"] += len(survivors) - len(kept)
    stats["kept"] = len(kept)
    return kept, stats
