"""Family 8, public_long: public datasets with long inputs (spec §3, Plan 5 Task 8).

Each state is one whole document of 2,000 Qwen3.5 tokens or more, and the rendered prompt fits in 12,288
tokens (`SourceSpec.token_window`; `build` drops and counts the rest). Every gold comes from the document
alone. The research and exclusion evidence is in docs/research/public-long-candidates.md; none of these
datasets is, mirrors, or is derived from a benchmark we report.

- cuad: commercial contracts (CUAD), Noul "does the contract contain a clause of this category?".
- maud: merger agreements (MAUD), Choice over the expert answer set of one deal-point question (Noul when
  that set is Yes/No).
- qasper: NLP papers (QASPER), Choice yes / no / the paper does not say.

Documents carry `metadata.doc_id`; rows of one document share the state, so the mix holds them out together.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from yev.schema import Decision, Option
from yev.sources.base import SourceSpec, slug

FAMILY = "public_long"
TOKEN_WINDOW = (2000, 12288)
MAX_STATE_CHARS = 2_000_000  # the token window is the real bound (and counts the drops); this only stops runaway text
PER_DOC = 8

NOUL_OPTIONS = [Option("yes", "Yes, the contract contains such a clause."),
                Option("no", "No, the contract contains no such clause.")]


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _balanced(items: list[dict], key, k: int, rng: random.Random) -> list[dict]:
    """Up to k items, as even as possible across the values of key(item)."""
    groups: dict[object, list[dict]] = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    buckets = [groups[g] for g in sorted(groups, key=str)]
    for b in buckets:
        rng.shuffle(b)
    out: list[dict] = []
    while len(out) < k and any(buckets):
        for b in buckets:
            if b and len(out) < k:
                out.append(b.pop())
    return out


# --- CUAD -------------------------------------------------------------------------------------------

CUAD_QUESTION_RE = re.compile(r'related to "(?P<cat>[^"]+)".*?Details:\s*(?P<details>.*)$', re.S)
# Extractive categories (names, dates, terms, governing law) have no yes/no reading.
CUAD_SKIP = frozenset({
    "Document Name", "Parties", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term",
    "Notice Period To Terminate Renewal", "Governing Law", "Warranty Duration",
})
CUAD_RESTRICTION_EXCEPTION = (
    "Does the contract contain an exception or carve-out to a non-compete, exclusivity or "
    "no-solicit-of-customers restriction?"
)
# ContractNLI (Decision Index #11) is built from NDAs; keep any confidentiality agreement out.
CUAD_NDA_RE = re.compile(r"disclos|confidential", re.I)


def prepare_cuad(rows: list[dict]) -> list[dict]:
    """One record per contract: its text and, per category, whether experts marked any clause."""
    docs: dict[str, dict] = {}
    for r in rows:
        title = r["title"]
        if CUAD_NDA_RE.search(title):
            continue
        m = CUAD_QUESTION_RE.search(r["question"])
        if not m or m["cat"] in CUAD_SKIP:
            continue
        doc = docs.setdefault(title, {"doc_id": title, "text": _clean(r["context"]), "items": {}})
        cat = m["cat"]
        item = doc["items"].setdefault(cat, {"category": cat, "details": _clean(m["details"]), "present": False})
        item["present"] = item["present"] or any(t.strip() for t in r["answers"]["text"])
    return [{**d, "items": list(d["items"].values())} for _, d in sorted(docs.items())]


def convert_cuad(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    out = []
    for it in _balanced(row["items"], lambda it: it["present"], PER_DOC, rng):
        cat = it["category"]
        ask = CUAD_RESTRICTION_EXCEPTION if cat == "Competitive Restriction Exception" else it["details"]
        out.append(Decision(
            id=f"{slug(row['doc_id'])}:{slug(cat)}", type="noul", state=row["text"],
            question=f"Contract review, {cat}: {ask}", options=list(NOUL_OPTIONS),
            gold="yes" if it["present"] else "no", family=FAMILY, source="", licence="",
            metadata={"doc_id": row["doc_id"], "item": cat},
        ))
    return out


# --- MAUD -------------------------------------------------------------------------------------------

MAUD_NO_SUB = "<NONE>"


MAUD_NOUL_OPTIONS = [Option("yes", "Yes, for this agreement."), Option("no", "No, not for this agreement.")]


def _maud_topic(question: str) -> str:
    """'Actions taken by Buyer-Answer (Y/N)' -> 'Actions taken by Buyer'."""
    topic = re.sub(r"\s*-?\s*\banswer\b", "", question, flags=re.I)
    topic = re.sub(r"\s*\(Y/N?\)?\s*$", "", topic)
    return re.sub(r"\s+", " ", topic).strip()


def prepare_maud(rows: list[dict]) -> list[dict]:
    """One record per agreement: all its `main` excerpts joined, and the single-answer questions on it."""
    main = [r for r in rows if r["data_type"] == "main" and (r["answer"] or "").strip() and (r["text"] or "").strip()]
    universe: dict[str, set[str]] = defaultdict(set)
    for r in main:
        if r["subquestion"] == MAUD_NO_SUB:
            universe[r["question"]].add(r["answer"].strip())
    usable = {q for q, answers in universe.items() if 2 <= len(answers) <= 10}
    docs: dict[str, dict] = {}
    for r in main:
        doc = docs.setdefault(r["contract_name"], {"doc_id": r["contract_name"], "excerpts": {}, "items": {}})
        if r["subquestion"] != MAUD_NO_SUB:
            continue
        doc["excerpts"].setdefault(_clean(r["text"]), None)
        if r["question"] in usable:
            doc["items"][r["question"]] = {"question": r["question"], "answer": r["answer"].strip(),
                                           "options": sorted(universe[r["question"]]),
                                           "section": _clean(r["text_type"] or "")}
    return [{"doc_id": d["doc_id"], "text": "\n\n".join(d["excerpts"]), "items": list(d["items"].values())}
            for _, d in sorted(docs.items())]


def _option_keys(descriptions: list[str]) -> list[str]:
    keys, seen = [], set()
    for desc in descriptions:
        base = slug(desc)[:48].strip("_") or "option"
        key, n = base, 1
        while key in seen:
            n += 1
            key = f"{base}_{n}"
        seen.add(key)
        keys.append(key)
    return keys


def convert_maud(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    items = list(row["items"])
    rng.shuffle(items)
    out = []
    for it in items[:PER_DOC]:
        topic = _maud_topic(it["question"])
        where = f"Merger agreement review, {it['section']}" if it["section"] else "Merger agreement review"
        if [o.lower() for o in it["options"]] == ["no", "yes"]:
            kind, options, gold = "noul", list(MAUD_NOUL_OPTIONS), it["answer"].lower()
            question = f"{where}. {topic}: yes or no?"
        else:
            keys = _option_keys(it["options"])
            kind, options = "choice", [Option(k, d) for k, d in zip(keys, it["options"])]
            gold = keys[it["options"].index(it["answer"])]
            question = f"{where}. Which answer describes this agreement's deal point \"{topic}\"?"
        out.append(Decision(
            id=f"{slug(row['doc_id'])}:{slug(topic)}", type=kind, state=row["text"], question=question,
            options=options, gold=gold, family=FAMILY, source="", licence="",
            metadata={"doc_id": row["doc_id"], "item": topic},
        ))
    return out


# --- QASPER -----------------------------------------------------------------------------------------

QASPER_OPTIONS = [Option("yes", "Yes, the paper says so."),
                  Option("no", "No, the paper says otherwise."),
                  Option("not_stated", "The paper does not say.")]
YES_NO_SHAPED = re.compile(
    r"^(do|does|did|is|are|was|were|can|could|has|have|had|will|would|should|shall|may|might)\b", re.I)


def _qasper_text(row: dict) -> str:
    parts = [row["title"].strip(), "Abstract\n" + row["abstract"].strip()]
    for name, paras in zip(row["full_text"]["section_name"], row["full_text"]["paragraphs"]):
        body = "\n".join(p.strip() for p in paras if p and p.strip())
        if body:
            parts.append(f"{(name or '').strip()}\n{body}".strip())
    return _clean("\n\n".join(parts))


def _qasper_gold(question: str, answers: list[dict]) -> str | None:
    kinds = set()
    for a in answers:
        if any(e.startswith("FLOAT SELECTED") for e in a["evidence"]):
            return None  # the evidence is a figure or table, whose contents are not in the text
        if a["unanswerable"]:
            kinds.add("not_stated")
        elif a["yes_no"] is not None:
            kinds.add("yes" if a["yes_no"] else "no")
        else:
            kinds.add("other")
    if len(kinds) != 1 or "other" in kinds:
        return None
    gold = kinds.pop()
    if gold == "not_stated" and not YES_NO_SHAPED.match(question.strip()):
        return None
    return gold


def convert_qasper(row: dict, i: int, rng: random.Random, labels: list[str]) -> list[Decision]:
    text = None
    out = []
    for q, qid, ann in zip(row["qas"]["question"], row["qas"]["question_id"], row["qas"]["answers"]):
        gold = _qasper_gold(q, ann["answer"])
        if gold is None:
            continue
        text = text or _qasper_text(row)
        out.append(Decision(
            id=f"{row['id']}:{qid}", type="choice", state=text,
            question=q.strip(), options=list(QASPER_OPTIONS), gold=gold,
            family=FAMILY, source="", licence="", metadata={"doc_id": row["id"]},
        ))
    return out


def _spec(name: str, hf_id: str, convert, pool_size: int, **kw) -> SourceSpec:
    return SourceSpec(name, hf_id, kw.pop("config", None), "train", "cc-by-4.0", convert, pool_size,
                      max_state_chars=MAX_STATE_CHARS, token_window=TOKEN_WINDOW, **kw)


# Pool sizes sum to the family cap of 3,000.
SOURCES: dict[str, SourceSpec] = {
    "cuad": _spec("cuad", "theatticusproject/cuad-qa", convert_cuad, 1500, config="default",
                  revision="d9c4ee0250ae2eb97bdb5b50773ab14ea62d0631", prepare=prepare_cuad),
    "maud": _spec("maud", "theatticusproject/maud", convert_maud, 1100,
                  revision="37d5c3b95d18dcd8404cc5ce3fd5069be062392f", data_files="MAUD_v1/MAUD_train.csv",
                  prepare=prepare_maud),
    "qasper": _spec("qasper", "allenai/qasper", convert_qasper, 400,
                    revision="06806e4608976fc2fac0a090ac425d5b2b29caf4", data_files="qasper/train/0000.parquet"),
}
