"""The one record type every stage reads and writes."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

QUESTION_TYPES = ("choice", "noul", "score")
SPLITS = ("pool", "train", "dev", "calibration")
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
NOUL_KEYS = ("yes", "no")


class SchemaError(ValueError):
    """A Decision breaks a schema rule."""


@dataclass(frozen=True)
class Option:
    key: str
    description: str


@dataclass
class Decision:
    id: str
    type: str
    state: str
    question: str
    options: list[Option]
    gold: str
    family: str
    source: str
    licence: str
    split: str = "pool"
    soft_gold: dict[str, float] | None = None
    cluster_id: str | None = None
    edit_type: str | None = None
    weight: float = 1.0

    @property
    def keys(self) -> list[str]:
        return [o.key for o in self.options]

    def validate(self) -> Decision:
        def fail(msg: str) -> None:
            raise SchemaError(f"{self.id or '<no id>'}: {msg}")

        if not self.id:
            fail("empty id")
        if self.type not in QUESTION_TYPES:
            fail(f"type {self.type!r} not in {QUESTION_TYPES}")
        if not self.state.strip():
            fail("empty state")
        if not self.question.strip():
            fail("empty question")
        keys = self.keys
        if len(set(keys)) != len(keys):
            fail(f"duplicate option keys {keys}")
        for o in self.options:
            if not KEY_RE.match(o.key):
                fail(f"bad option key {o.key!r}")
            if not o.description.strip():
                fail(f"empty description for {o.key!r}")
        if self.type == "choice" and not 2 <= len(keys) <= 6:
            fail(f"choice needs 2-6 options, got {len(keys)}")
        if self.type == "noul" and tuple(keys) != NOUL_KEYS:
            fail(f"noul options must be {NOUL_KEYS} in order, got {keys}")
        if self.type == "score" and not 3 <= len(keys) <= 11:
            fail(f"score needs 3-11 options, got {len(keys)}")
        if self.gold not in keys:
            fail(f"gold {self.gold!r} not in {keys}")
        if self.soft_gold is not None:
            if set(self.soft_gold) != set(keys):
                fail("soft_gold keys differ from option keys")
            if any(p < 0 for p in self.soft_gold.values()):
                fail("negative soft_gold probability")
            if not math.isclose(sum(self.soft_gold.values()), 1.0, abs_tol=1e-3):
                fail("soft_gold does not sum to 1")
        if self.split not in SPLITS:
            fail(f"split {self.split!r} not in {SPLITS}")
        if not self.weight > 0:
            fail("weight must be positive")
        if not (self.family and self.source and self.licence):
            fail("family, source and licence are required")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Decision:
        d = dict(d)
        d["options"] = [Option(**o) for o in d["options"]]
        return cls(**d)


def write_jsonl(path: Path | str, decisions: Iterable[Decision]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    n = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for d in decisions:
                f.write(json.dumps(d.validate().to_dict(), ensure_ascii=False) + "\n")
                n += 1
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return n


def read_jsonl(path: Path | str) -> Iterator[Decision]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Decision.from_dict(json.loads(line)).validate()
