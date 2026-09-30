"""Near-duplicate key: normalised state + question + each option's key and description."""

from __future__ import annotations

import hashlib

from jeff.filters.contamination import normalize_tokens
from jeff.schema import Decision


def dedupe_key(d: Decision) -> str:
    text = " ".join(normalize_tokens(f"{d.state}\n{d.question}"))
    options = "|".join(
        f"{o.key}={' '.join(normalize_tokens(o.description))}" for o in sorted(d.options, key=lambda o: o.key)
    )
    return hashlib.sha1(f"{text}\n{options}".encode()).hexdigest()
