"""Parse and validate Opus writer output (one cluster per JSON line)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from yev.generators.common import EDIT_TYPES, MAX_EDIT_TOKENS, make_cluster, token_edit_size
from yev.generators.llm.check import LETTERS
from yev.schema import Decision, Option, SchemaError

SOURCE = "synthetic_opus"
MIN_CHOICE_OPTIONS = 3


def _options(raw: list[dict]) -> list[Option]:
    return [Option(str(o["key"]).strip().lower(), str(o["description"]).strip()) for o in raw]


def _policy_edit_check(base: list[Option], new: list[Option]) -> str | None:
    """None when the edit is valid, else the rejection reason."""
    if [o.key for o in base] != [o.key for o in new]:
        return "bad_policy_edit"
    changed = [(a, b) for a, b in zip(base, new) if a.description != b.description]
    if not changed:
        return "no_edit"
    if len(changed) != 1:
        return "bad_policy_edit"
    old, cur = changed[0][0].description, changed[0][1].description
    size = token_edit_size(old, cur)
    if size < 1 or old.lower() == cur.lower():
        return "no_edit"
    return None if size <= MAX_EDIT_TOKENS else "bad_policy_edit"


def parse_cluster(obj: dict, *, cluster_id: str, family: str, qtype: str) -> tuple[list[Decision], Counter]:
    reasons: Counter = Counter()
    if len(obj["options"]) > len(LETTERS):
        reasons["too_many_options"] += 1
        reasons["cluster_rejected"] += 1
        return [], reasons
    if qtype == "choice" and len(obj["options"]) < MIN_CHOICE_OPTIONS:
        reasons["too_few_options"] += 1
        reasons["cluster_rejected"] += 1
        return [], reasons
    question = str(obj["question"]).strip()
    options = _options(obj["options"])
    base_state, base_gold = str(obj["base"]["state"]).strip(), str(obj["base"]["gold"]).strip().lower()
    members: list[tuple[str, list[Option], str, str | None]] = [(base_state, options, base_gold, None)]
    for v in obj.get("variants") or []:
        state, gold = str(v["state"]).strip(), str(v["gold"]).strip().lower()
        edit = str(v.get("edit_type") or "").strip().lower()
        if edit not in EDIT_TYPES or edit == "policy_edit":
            reasons["bad_edit_type"] += 1
        elif token_edit_size(base_state, state) > MAX_EDIT_TOKENS:
            reasons["edit_too_large"] += 1
        elif token_edit_size(base_state, state) < 1:
            reasons["no_edit"] += 1
        elif edit == "injection" and gold != base_gold:
            reasons["injection_changed_gold"] += 1
        elif edit != "injection" and gold == base_gold:
            reasons["same_gold"] += 1
        else:
            members.append((state, options, gold, edit))
    for v in obj.get("policy_variants") or []:
        if len(v["options"]) > len(LETTERS):
            reasons["too_many_options"] += 1
            continue
        new_options = _options(v["options"])
        gold = str(v["gold"]).strip().lower()
        problem = _policy_edit_check(options, new_options)
        if str(v.get("edit_type") or "").strip().lower() != "policy_edit" or gold == base_gold:
            reasons["bad_policy_edit"] += 1
        elif problem:
            reasons[problem] += 1
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


def _handle(obj, n: int, batch: dict, stats: Counter, out: list[Decision], run_name: str | None) -> None:
    prefix = f"{run_name}:" if run_name else ""
    try:
        decisions, reasons = parse_cluster(obj, cluster_id=f"{prefix}{batch['batch_id']}:{n}",
                                           family=batch["family"], qtype=batch["qtype"])
    except (KeyError, TypeError, AttributeError):
        stats["bad_shape"] += 1
        return
    stats.update(reasons)
    if decisions:
        stats["clusters_kept"] += 1
        out.extend(decisions)


def _scan_objects(text: str):
    dec, pos = json.JSONDecoder(), 0
    while (pos := text.find("{", pos)) != -1:
        try:
            obj, end = dec.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        yield obj
        pos = end


def ingest_file(path: Path | str, batch: dict, run_name: str | None = None) -> tuple[list[Decision], Counter]:
    text = Path(path).read_text(encoding="utf-8")
    stats: Counter = Counter()
    out: list[Decision] = []
    dicts = 0
    for n, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        stats["lines"] += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            continue
        if not isinstance(obj, dict):
            stats["bad_shape"] += 1
            continue
        dicts += 1
        _handle(obj, n, batch, stats, out, run_name)
    if dicts == 0 and "{" in text:
        stats, out = Counter(), []
        for n, obj in enumerate(_scan_objects(text)):
            if isinstance(obj, dict):
                stats["recovered_multiline"] += 1
                _handle(obj, n, batch, stats, out, run_name)
    return out, stats
