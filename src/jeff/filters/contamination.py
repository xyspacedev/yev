"""DecideBench contamination filters: canary, 8-gram overlap, embedding similarity."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable

import numpy as np

from jeff import decidebench
from jeff.schema import Decision

TOKEN_RE = re.compile(r"[a-z0-9]+")
NGRAM = 8


def normalize_tokens(text: str) -> list[str]:
    """Strip accents and format characters, then split on anything that isn't a-z/0-9."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed
                       if unicodedata.category(ch) != "Cf" and not unicodedata.combining(ch))
    return TOKEN_RE.findall(stripped.lower())


def ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def decision_text(d: Decision) -> str:
    return "\n".join([d.state, d.question, *(o.description for o in d.options)])


def raw_text(item: dict) -> str:
    return "\n".join([item["state"], item["question"], *(o["description"] for o in item["options"])])


@dataclass
class Fingerprints:
    ngrams: set[tuple[str, ...]]
    states: list[str]


def build_fingerprints(raw_items: list[dict]) -> Fingerprints:
    grams: set[tuple[str, ...]] = set()
    for item in raw_items:
        grams |= ngrams(normalize_tokens(raw_text(item)))
    return Fingerprints(ngrams=grams, states=[item["state"] for item in raw_items])


def load_fingerprints() -> Fingerprints:
    return build_fingerprints(decidebench.load_all_raw())


def has_canary(d: Decision) -> bool:
    return decidebench.CANARY_GUID.lower() in json.dumps(d.to_dict(), ensure_ascii=False).lower()


def overlaps(d: Decision, fp: Fingerprints) -> bool:
    return not ngrams(normalize_tokens(decision_text(d))).isdisjoint(fp.ngrams)


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class EmbeddingFilter:
    def __init__(self, reference_texts: list[str], encode: Callable[[list[str]], np.ndarray], threshold: float = 0.85):
        self.encode = encode
        self.threshold = threshold
        self.reference = _unit(encode(reference_texts))

    def too_similar(self, texts: list[str]) -> list[bool]:
        sims = _unit(self.encode(texts)) @ self.reference.T
        return (sims.max(axis=1) > self.threshold).tolist()


def sentence_transformer_encoder(model: str = "BAAI/bge-small-en-v1.5") -> Callable[[list[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model)
    return lambda texts: m.encode(texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True)
