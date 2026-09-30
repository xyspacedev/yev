"""Parse and validate Opus writer output (one cluster per JSON line)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from jeff.generators.common import EDIT_TYPES, MAX_EDIT_TOKENS, make_cluster, token_edit_size
from jeff.schema import Decision, Option, SchemaError

SOURCE = "synthetic_opus"


def _options(raw: list[dict]) -> list[Option]:
    return [Option(str(o["key"]).strip(), str(o["description"]).strip()) for o in raw]


def _policy_edit_ok(base: list[Option], new: list[Option]) -> bool:
    if [o.key for o in base] != [o.key for o in new]:
        return False
    changed = [(a, b) for a, b in zip(base, new) if a.description != b.description]
    return len(changed) == 1 and token_edit_size(changed[0][0].description, changed[0][1].description) <= MAX_EDIT_TOKENS


def parse_cluster(obj: dict, *, cluster_id: str, family: str, qtype: str) -> tuple[list[Decision], Counter]:
    reasons: Counter = Counter()
    question = str(obj["question"]).strip()
    options = _options(obj["options"])
    base_state, base_gold = str(obj["base"]["state"]).strip(), str(obj["base"]["gold"]).strip()
    members: list[tuple[str, list[Option], str, str | None]] = [(base_state, options, base_gold, None)]
    for v in obj.get("variants") or []:
        state, gold, edit = str(v["state"]).strip(), str(v["gold"]).strip(), v.get("edit_type")
        if edit not in EDIT_TYPES or edit == "policy_edit":
            reasons["bad_edit_type"] += 1
        elif token_edit_size(base_state, state) > MAX_EDIT_TOKENS:
            reasons["edit_too_large"] += 1
        elif edit == "injection" and gold != base_gold:
            reasons["injection_changed_gold"] += 1
        elif edit != "injection" and gold == base_gold:
            reasons["same_gold"] += 1
        else:
            members.append((state, options, gold, edit))
    for v in obj.get("policy_variants") or []:
        new_options = _options(v["options"])
        gold = str(v["gold"]).strip()
        if v.get("edit_type") != "policy_edit" or not _policy_edit_ok(options, new_options) or gold == base_gold:
            reasons["bad_policy_edit"] += 1
        else:
            members.append((base_state, new_options, gold, "policy_edit"))

    decisions = make_cluster(cluster_id=cluster_id, family=family, source=SOURCE, qtype=qtype,
                             question=question, members=members)
    valid: list[Decision] = []
    for k, d in enumerate(decisions):
        try:
            valid.append(d.validate())
        except SchemaError:
            if k == 0:
                reasons["cluster_rejected"] += 1
                return [], reasons
            reasons["invalid"] += 1
    if len({d.gold for d in valid}) < 2:
        reasons["cluster_rejected"] += 1
        return [], reasons
    return valid, reasons


def ingest_file(path: Path | str, batch: dict) -> tuple[list[Decision], Counter]:
    stats: Counter = Counter()
    out: list[Decision] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        stats["lines"] += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            continue
        try:
            decisions, reasons = parse_cluster(obj, cluster_id=f"{batch['batch_id']}:{n}",
                                               family=batch["family"], qtype=batch["qtype"])
        except (KeyError, TypeError, AttributeError):
            stats["bad_shape"] += 1
            continue
        stats.update(reasons)
        if decisions:
            stats["clusters_kept"] += 1
            out.extend(decisions)
    return out, stats
