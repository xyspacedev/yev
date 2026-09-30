"""Edit attribution: a fresh checker names the edit between base and variant; mismatches are dropped."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from jeff.generators.common import keep_valid_clusters
from jeff.schema import Decision

EDIT_TYPE_DEFINITIONS = {
    "negation": "something is made true/false: 'is' vs 'is not', 'lost' vs 'kept', 'has' vs 'lacks'",
    "threshold": "a number moves across a limit: $500 vs $501, 99 vs 100 users",
    "date": "a date or duration moves across a window: day 30 vs day 31, before vs after a deadline",
    "entity_swap": "one thing is replaced by another of the same kind: production vs staging, internal vs external",
    "quantifier": "how many is changed: all vs some, every vs one, none vs any",
    "exception": "an exception or special condition is added or removed: 'final sale', 'has an approved ticket'",
    "unit": "a unit or scale changes: hours vs days, MB vs GB, per month vs per year",
    "policy_edit": "the input text is identical but one option's rule text changed",
    "injection": "the input gains a sentence that tries to instruct the reader, and nothing else changes",
}


def prepare_pairs(decisions: list[Decision], part_size: int = 100, seed: int = 0) -> tuple[list[tuple[str, list[dict]]], dict]:
    clusters: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        clusters[d.cluster_id].append(d)
    key: dict = {"_no_base": []}
    items: list[dict] = []
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
    parts = [(f"pairs-part-{p // part_size}", items[p : p + part_size]) for p in range(0, len(items), part_size)]
    return parts, key


def render_attrib_prompt(pairs_path: str, answers_path: str) -> str:
    definitions = "\n".join(f"- `{name}`: {text}" for name, text in EDIT_TYPE_DEFINITIONS.items())
    return f"""# Name the edit between two versions

Read `{pairs_path}`. Each line is a pair: version `a` and version `b` of the same decision item, each with an input
`state` and its option rule texts. Compare them and name the ONE kind of edit that turns `a` into `b`:

{definitions}

Do not open any other file or directory.

Write your answers to `{answers_path}` with the Write tool: one JSON object per line, exactly
{{"pair_id": "<the pair's id>", "edit_type": "<one name from the list>"}}, one line per pair, no prose.
Then reply with only the number of answers written.
"""


def score_attribution(decisions: list[Decision], key: dict, answers: list[dict]) -> tuple[list[Decision], Counter]:
    stats: Counter = Counter()
    answered = {str(a.get("pair_id", "")): str(a.get("edit_type", "")).strip().lower() for a in answers}
    verdict: dict[str, bool] = {}
    for pair_id, entry in key.items():
        if pair_id == "_no_base":
            continue
        got = answered.get(pair_id)
        if got is None:
            stats["attrib_missing"] += 1
            verdict[entry["decision"]] = False
        elif got != entry["edit_type"]:
            stats["attrib_mismatch"] += 1
            verdict[entry["decision"]] = False
        else:
            verdict[entry["decision"]] = True
    no_base = set(key.get("_no_base", []))
    stats["no_base"] = len(no_base)
    survivors = [d for d in decisions if d.id not in no_base and (d.edit_type is None or verdict.get(d.id, False))]
    kept = keep_valid_clusters(survivors)
    stats["cluster_collapsed"] += len(survivors) - len(kept)
    stats["kept"] = len(kept)
    return kept, stats
