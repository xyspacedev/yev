"""Shared pieces for external benchmarks: chat rows in benchmark order, files, network fetch, 8-gram overlap."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Iterable

from yev.filters.contamination import ngrams, normalize_tokens
from yev.format import LETTERS, SYSTEM_PROMPT, user_turn
from yev.schema import Decision

ROWS_FILE = "rows.chat.jsonl"
META_FILE = "meta.json"
BENCH_SPLIT = "test"


def chat_row(d: Decision, bench: dict | None = None) -> dict:
    """One chat row with the options in the order the benchmark gives them (never shuffled).

    Same keys as the training chat rows, plus `bench` (scorer-only fields such as pair id or subset).
    """
    d.validate()
    options = list(d.options)
    letters = {LETTERS[i]: o.key for i, o in enumerate(options)}
    target = {L: (d.soft_gold[k] if d.soft_gold else float(k == d.gold)) for L, k in letters.items()}
    return {
        "id": d.id,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_turn(d, options)}],
        "answer": LETTERS[d.keys.index(d.gold)],
        "letters": letters,
        "target": target,
        "weight": d.weight,
        "type": d.type,
        "family": d.family,
        "source": d.source,
        "cluster_id": d.cluster_id,
        "edit_type": d.edit_type,
        "split": BENCH_SPLIT,
        "bench": bench or {},
    }


def write_rows(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_rows(path: Path | str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "yev-bench"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def github_raw(repo: str, sha: str, path: str) -> bytes:
    return http_get(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}")


def github_tree(repo: str, sha: str) -> list[str]:
    data = json.loads(http_get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"))
    if data.get("truncated"):
        raise RuntimeError(f"{repo}@{sha}: tree listing truncated")
    return [t["path"] for t in data["tree"] if t["type"] == "blob"]


# --- 8-gram overlap with the training mix ---------------------------------------------------------

def states_of(row: dict) -> list[str]:
    """Every `state` in a chat row's user turns (worked examples included)."""
    out = []
    for m in row["messages"]:
        if m["role"] != "user":
            continue
        try:
            obj = json.loads(m["content"])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("state"), str):
            out.append(obj["state"])
    return out


def item_state(row: dict) -> str:
    """The state of the item itself (the last user turn), not of any worked example."""
    return json.loads(row["messages"][-1]["content"])["state"]


def overlap(bench_rows: list[dict], train_rows: Iterable[dict]) -> dict:
    """Share of benchmark items whose state shares a normalised 8-gram with any training row's state."""
    by_gram: dict[tuple, set[str]] = {}
    ids: dict[str, None] = {}
    for r in bench_rows:
        ids[r["id"]] = None
        for g in ngrams(normalize_tokens(item_state(r))):
            by_gram.setdefault(g, set()).add(r["id"])
    hit: set[str] = set()
    n_train = 0
    for t in train_rows:
        n_train += 1
        for s in states_of(t):
            for g in ngrams(normalize_tokens(s)):
                if g in by_gram:
                    hit |= by_gram[g]
    n = len(ids)
    return {"n_items": n, "n_overlapping": len(hit), "share": (len(hit) / n) if n else 0.0,
            "n_train_rows": n_train, "ngram": 8, "overlapping_ids": sorted(hit)}


def iter_rows(path: Path | str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
