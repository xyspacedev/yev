"""Near-duplicate key: normalised state + question + option keys."""

from __future__ import annotations

import hashlib

from jeff.filters.contamination import normalize_tokens
from jeff.schema import Decision


def dedupe_key(d: Decision) -> str:
    text = " ".join(normalize_tokens(f"{d.state}\n{d.question}"))
    keys = "|".join(sorted(o.key for o in d.options))
    return hashlib.sha1(f"{text}\n{keys}".encode()).hexdigest()
