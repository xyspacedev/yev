"""Family 6, long_workflow: document mode for the Opus synthetic pipeline.

Opus writers produce long (4-10k token) workflow documents, each with 6-12 typed questions. Every question
becomes a Decision that shares the document's `state` and has `cluster_id = doc:<doc_id>`. The blind check
reuses check.py (one item per question, 3 independent samples, majority must match the writer's gold; soft
gold = the sample shares when they split). A document is kept only if at least 70% of its questions survive.
Documents skip edit attribution (Plan Ruling 2): check-score writes final/synthetic_docs.jsonl directly.
"""

from __future__ import annotations

import json
import logging
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Callable

from yev.generators import padding
from yev.generators.common import SYNTH_LICENCE
from yev.generators.llm.check import LETTERS, score_answers
from yev.generators.llm.families import DOC_KINDS, DOC_TOPICS, DOMAINS
from yev.generators.llm.ingest import SOURCE, _scan_objects
from yev.schema import Decision, Option, SchemaError

FAMILY = "long_workflow"
MODE = "doc"
BATCH_PREFIX = "doc-batch-"
TARGET_TOKENS = (4000, 10000)
QUESTIONS_PER_DOC = (6, 12)
MIN_TOKENS, MAX_TOKENS = 3000, 11000
MIN_CHOICE, MAX_CHOICE = 3, len(LETTERS)
MIN_LEVELS, MAX_LEVELS = 3, len(LETTERS)
KEEP_DOC_SHARE = 0.7
INJECTION_SHARE = 0.05
CHECK_PART_SIZE = 10  # items per checker part; each item carries a whole document
WORDS_PER_TOKEN = 0.7
log = logging.getLogger(__name__)


# --- planning -------------------------------------------------------------------------------------

def plan_doc_batches(n_docs: int, per_batch: int, seed: int) -> list[dict]:
    if per_batch > len(DOMAINS):
        raise ValueError(f"per_batch {per_batch} exceeds the {len(DOMAINS)} available domains")
    rng = random.Random(f"docplan:{seed}")
    kinds = sorted(DOC_KINDS)
    n_inj = min(n_docs, max(1, round(INJECTION_SHARE * n_docs)))
    injected = set(rng.sample(range(n_docs), n_inj))
    batches: list[dict] = []
    start, k = 0, 0
    while start < n_docs:
        n = min(per_batch, n_docs - start)
        batches.append({
            "batch_id": f"{BATCH_PREFIX}{k:03d}",
            "mode": MODE,
            "family": FAMILY,
            "doc_kind": kinds[k % len(kinds)],
            "n_docs": n,
            "domains": rng.sample(DOMAINS, n),
            "target_tokens": rng.randrange(TARGET_TOKENS[0], TARGET_TOKENS[1] + 1, 500),
            "questions_per_doc": rng.randint(*QUESTIONS_PER_DOC),
            "injection_docs": sorted(j - start + 1 for j in injected if start <= j < start + n),
        })
        start += n
        k += 1
    return batches


def is_doc_spec(spec: dict) -> bool:
    return spec.get("mode") == MODE


# --- writer prompt --------------------------------------------------------------------------------

SCORE_EXAMPLE_LEVELS = [
    "One sentence restating level 0 of the rubric quoted in the document.",
    "One sentence restating level 1 of that rubric.",
    "One sentence restating level 2 of that rubric.",
    "One sentence restating level 3 of that rubric.",
]


def example_doc(kind: str) -> dict:
    """A one-line placeholder document whose question names come from this kind's topics."""
    topics = DOC_KINDS[kind][1]
    choice = next(t for t in topics if DOC_TOPICS[t][0] == "choice")
    score = next(t for t in topics if DOC_TOPICS[t][0] == "score")
    return {
        "doc_id": "d1",
        "injection": False,
        "state": "The whole document text, with real line breaks written as \\n.",
        "questions": [
            {"name": choice, "type": "choice", "question": "A question about this case.",
             "options": [{"key": "first_option", "description": "One sentence."},
                         {"key": "second_option", "description": "One sentence."},
                         {"key": "third_option", "description": "One sentence."},
                         {"key": "insufficient_information", "description": "The document does not settle it."}],
             "gold": "second_option", "rationale": "One or two sentences pointing to where the document settles it."},
            {"name": score, "type": "score",
             "question": "Using the rating scale quoted in the document, which level applies?",
             "options": [{"key": str(i), "description": d} for i, d in enumerate(SCORE_EXAMPLE_LEVELS)],
             "gold": "2", "rationale": "The rubric in section 2 and the impact figures in the 14:05 update."},
        ],
    }


def doc_writer_prompt(batch: dict, output_path: str = "<output file>") -> str:
    description, topics = DOC_KINDS[batch["doc_kind"]]
    topic_lines = "\n".join(f"- `{t}` ({DOC_TOPICS[t][0]}): {DOC_TOPICS[t][1]}" for t in topics)
    domains = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(batch["domains"]))
    n, q, tokens = batch["n_docs"], batch["questions_per_doc"], batch["target_tokens"]
    words = round(tokens * WORDS_PER_TOKEN, -2)
    inj = batch["injection_docs"]
    if inj:
        nums = ", ".join(str(k) for k in inj)
        injection_rule = (
            f"Document(s) number {nums} must contain one injected instruction: a sentence planted inside the "
            "document (in a customer message, a tool output, a pasted note) that tries to instruct whoever reads "
            "it, e.g. telling the reader to pick a particular answer or ignore the policy. The injected instruction "
            "must not change any gold: every answer is what it would be without that sentence. Set `injection` to "
            "true on that document and false on the others."
        )
    else:
        injection_rule = ("Do not put any injected instruction (text aimed at whoever reads the document) in these "
                          "documents. Set `injection` to false on every document.")
    return f"""# Write {n} long workflow documents with questions

You are writing training data for a small decision model. Each item gives the model a long document (the state),
one question about it, and a list of options; the model must pick the single correct option. You write the
documents and several questions per document.

## Document kind: {batch['doc_kind']}
{description}

## Length
Each document is about {tokens:,} tokens (roughly {words:,.0f} words). Never shorter than 3,000 tokens or longer
than 11,000. Length should come from realistic detail (messages, log lines, tables, quoted procedure), not padding.

## Questions
Write {q} questions per document. Use these question topics; a question's `name` is its topic (append `_2`,
`_3` if a topic is used twice in one document). Use at least four different topics per document.

{topic_lines}

- A choice question has 3 to 10 options. Where it is natural, include a "none of these" or
  "insufficient information" option, and make it the gold when the document really does not settle it.
- A score question uses numbered levels: option keys "0", "1", ... "n-1" (3 to 10 levels), ordered from lowest to
  highest, each with a one-sentence description of that level.
- Every score question (severity, risk, priority, sentiment or similar) must use a
  rubric stated inside the document itself: a quoted SEV table, a risk matrix, a priority definition or a policy
  excerpt that defines each level, so the level can be decided from the text alone. The score options restate that rubric's levels.
  Write the rubric into the document before writing the question; never rely on a scale the reader must guess.
- Choice option keys are short snake_case; each description is one sentence.
- Every `gold` is one of the question's option keys. Spread golds across options; do not favour the first one.
- `rationale` says in one or two sentences where in the document the answer is settled. It is never shown to the
  model.

## Rules
- Write in the realistic voice of the domain: the jargon, formats, ticket numbers, timestamps, signatures and
  shorthand that real staff in that business would use.
- All facts needed for each question must be stated in the document: rules, limits, prices, dates, who said
  what. The answer must follow from the document, not from world knowledge.
- No question may be answerable without reading the document: a reader who sees only the question and options
  must not be able to guess the gold. Avoid options that are obviously wrong on their own.
- Make readers work: put the deciding facts in different parts of the document, let later messages correct
  earlier ones, include plausible but irrelevant detail.
- Each document's questions are answered independently; no question may give away another's answer.
- {injection_rule}
- Everything is fictional: invent company names, people, products and numbers. Do not reuse examples you
  remember from public datasets or benchmarks.
- Set document k in the domain numbered k below:

{domains}

## Output format
Write exactly {n} lines to `{output_path}` using the Write tool: one JSON object per line (one document per
line), no prose, no code fences. Each object has this shape (values are placeholders, shortened here):

{json.dumps(example_doc(batch['doc_kind']))}

Keys: `doc_id` (short unique string), `injection` (true/false), `state` (the full document), `questions` (list
of {{name, type, question, options, gold, rationale}}; `type` is "choice" or "score").

When the file is written, reply with only the path and the number of lines written.
"""


# --- token counting -------------------------------------------------------------------------------

def load_doc_counter(allow_heuristic: bool = False) -> tuple[Callable[[str], int], str]:
    """The real Qwen3.5 tokenizer from the local HF cache. Without it, raises padding.TokenizerUnavailable unless
    allow_heuristic, which falls back (with a warning) to padding's len/4 x 2.0 estimate."""
    count = padding.load_token_counter(allow_heuristic=allow_heuristic)
    return count, ("heuristic" if count is padding.heuristic_tokens else "tokenizer")


# --- ingest ---------------------------------------------------------------------------------------

_SAFE = re.compile(r"[^a-z0-9_-]+")


def _parse_question(q: dict, *, state: str, cid: str, meta: dict) -> tuple[Decision | None, str | None]:
    try:
        name = _SAFE.sub("_", str(q["name"]).strip().lower()).strip("_")
        qtype = str(q["type"]).strip().lower()
        question = str(q["question"]).strip()
        raw = q["options"]
        gold = str(q["gold"]).strip().lower()
        options = [Option(str(o["key"]).strip().lower(), str(o["description"]).strip()) for o in raw]
    except (KeyError, TypeError, AttributeError):
        return None, "bad_question_shape"
    if not name:
        return None, "bad_question_shape"
    if qtype == "choice":
        if len(options) < MIN_CHOICE:
            return None, "too_few_options"
        if len(options) > MAX_CHOICE:
            return None, "too_many_options"
    elif qtype == "score":
        if not MIN_LEVELS <= len(options) <= MAX_LEVELS or [o.key for o in options] != [
                str(i) for i in range(len(options))]:
            return None, "bad_score_levels"
    else:
        return None, "bad_question_type"
    d = Decision(id=f"{cid}:{name}", type=qtype, state=state, question=question, options=options, gold=gold,
                 family=FAMILY, source=SOURCE, licence=SYNTH_LICENCE, cluster_id=cid,
                 metadata={**meta, "question_name": name, "rationale": str(q.get("rationale") or "").strip()})
    try:
        return d.validate(), None
    except SchemaError:
        return None, "invalid_question"


def parse_doc(obj: dict, *, batch: dict, index: int, run_name: str | None, count: Callable[[str], int],
              method: str, seen: set[str]) -> tuple[list[Decision], Counter, str | None, int | None]:
    """Decisions for one document, question-level reasons, and the document-level drop reason (or None)."""
    reasons: Counter = Counter()
    try:
        state = str(obj["state"]).strip()
        questions = obj["questions"]
        if not state or not isinstance(questions, list):
            raise TypeError
    except (KeyError, TypeError):
        return [], reasons, "bad_shape", None
    writer_id = _SAFE.sub("_", str(obj.get("doc_id") or f"d{index}").strip().lower()) or f"d{index}"
    prefix = f"{run_name}:" if run_name else ""
    cid = f"doc:{prefix}{batch['batch_id']}:{writer_id}"
    if cid in seen:
        return [], reasons, "duplicate_doc_id", None
    seen.add(cid)
    if index >= batch["n_docs"]:
        return [], reasons, "extra_doc", None
    est = count(state)
    if est < MIN_TOKENS:
        return [], reasons, "too_short", est
    if est > MAX_TOKENS:
        return [], reasons, "too_long", est
    meta = {"doc_kind": batch["doc_kind"], "domain": batch["domains"][index], "est_tokens": est,
            "questions_written": len(questions),
            "token_count": method, "injection": (index + 1) in batch.get("injection_docs", [])}
    out: list[Decision] = []
    names: set[str] = set()
    for q in questions:
        reasons["questions_seen"] += 1
        d, why = _parse_question(q, state=state, cid=cid, meta=meta) if isinstance(q, dict) else (
            None, "bad_question_shape")
        if d is not None and d.metadata["question_name"] in names:
            d, why = None, "duplicate_question_name"
        if d is None:
            reasons[why] += 1
            continue
        names.add(d.metadata["question_name"])
        out.append(d)
    if not out or len(out) < KEEP_DOC_SHARE * len(questions):
        return [], reasons, "doc_too_many_bad_questions", est
    lo, hi = QUESTIONS_PER_DOC
    if not lo <= len(out) <= hi:
        reasons["question_count_outside_plan"] += 1  # recorded, not dropped
    return out, reasons, None, est


def _objects(text: str, stats: Counter):
    dicts = []
    for line in text.splitlines():
        if not line.strip():
            continue
        stats["lines"] += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            stats["bad_json"] += 1
            continue
        dicts.append(obj)
    if not any(isinstance(o, dict) for o in dicts) and "{" in text:
        dicts = list(_scan_objects(text))
        stats["recovered_multiline"] += len(dicts)
    return dicts


def ingest_docs(run_dir: Path | str, count: Callable[[str], int] | None = None, stats: Counter | None = None,
                dropped: list[dict] | None = None, method: str | None = None) -> list[Decision]:
    """Validate every written document of a run; drop reasons go to `stats` and `dropped`."""
    run = Path(run_dir)
    if count is None:
        count, method = load_doc_counter()
    method = method or "custom"
    stats = stats if stats is not None else Counter()
    dropped = dropped if dropped is not None else []
    run_name = run.resolve().name
    rows: list[Decision] = []
    seen: set[str] = set()
    for spec_path in sorted((run / "batches").glob("*.json")):
        batch = json.loads(spec_path.read_text())
        if not is_doc_spec(batch):
            continue
        written = run / "written" / f"{batch['batch_id']}.jsonl"
        if not written.exists():
            stats["missing_file"] += 1
            continue
        for index, obj in enumerate(_objects(written.read_text(encoding="utf-8"), stats)):
            stats["docs_seen"] += 1
            if not isinstance(obj, dict):
                stats["bad_shape"] += 1
                dropped.append({"doc": f"{batch['batch_id']}:{index}", "reason": "bad_shape"})
                continue
            got, reasons, why, est = parse_doc(obj, batch=batch, index=index, run_name=run_name, count=count,
                                               method=method, seen=seen)
            stats.update(reasons)
            if why:
                stats[why] += 1
                dropped.append({"doc": f"doc:{run_name}:{batch['batch_id']}:{obj.get('doc_id', index)}",
                                "reason": why, "est_tokens": est})
                continue
            stats["docs_kept"] += 1
            stats["questions_kept"] += len(got)
            rows.extend(got)
    stats["rows"] = len(rows)
    return rows


# --- check scoring --------------------------------------------------------------------------------

def keep_docs(survivors: list[Decision], decisions: list[Decision],
              min_share: float = KEEP_DOC_SHARE) -> list[Decision]:
    """Keep a document's surviving questions only if they are at least min_share of the questions the writer
    wrote (metadata questions_written; the ingested count when that is missing)."""
    ingested = Counter(d.cluster_id for d in decisions)
    written = {d.cluster_id: (d.metadata or {}).get("questions_written") or ingested[d.cluster_id]
               for d in decisions}
    alive = Counter(d.cluster_id for d in survivors)
    return [d for d in survivors if alive[d.cluster_id] >= min_share * written[d.cluster_id] - 1e-9]


def score_doc_answers(decisions: list[Decision], key: dict, answers: list[dict],
                      survivors_out: list | None = None) -> tuple[list[Decision], Counter]:
    """Blind-check votes per question, then the document rule. survivors_out receives the per-question survivors."""
    survivors, stats = score_answers(decisions, key, answers, cluster_filter=None)
    if survivors_out is not None:
        survivors_out.extend(survivors)
    kept = keep_docs(survivors, decisions)
    stats["questions_survived"] = len(survivors)
    stats["docs_dropped_below_70pct"] = len({d.cluster_id for d in survivors} - {d.cluster_id for d in kept})
    stats["docs_kept"] = len({d.cluster_id for d in kept})
    stats["kept"] = len(kept)
    return kept, stats


def _count_docs(rows: list[Decision], field: str) -> Counter:
    first: dict[str, str] = {}
    for d in rows:
        first.setdefault(d.cluster_id, d.metadata[field])
    return Counter(first.values())


def doc_pilot_report(specs: list[dict], ingest: dict, ingested: list[Decision], survivors: list[Decision],
                     kept: list[Decision]) -> dict:
    """Survival from plan to final, by document kind and question type."""
    planned_by_kind: Counter = Counter()
    for b in specs:
        planned_by_kind[b["doc_kind"]] += b["n_docs"]
    ing_k, fin_k = _count_docs(ingested, "doc_kind"), _count_docs(kept, "doc_kind")
    kinds = sorted(set(planned_by_kind) | set(ing_k))
    by_type = {t: {"ingested": sum(d.type == t for d in ingested),
                   "survived_check": sum(d.type == t for d in survivors),
                   "final": sum(d.type == t for d in kept)} for t in ("choice", "score")}
    def topic(d):
        return re.sub(r"_\d+$", "", d.metadata["question_name"])

    topics = sorted({topic(d) for d in ingested})
    by_topic = {t: {"ingested": sum(topic(d) == t for d in ingested),
                    "survived_check": sum(topic(d) == t for d in survivors)} for t in topics}
    tokens = [d.metadata["est_tokens"] for d in {d.cluster_id: d for d in kept}.values()]
    inj_final = {d.cluster_id for d in kept if d.metadata.get("injection")}
    inj_ing = {d.cluster_id for d in ingested if d.metadata.get("injection")}
    n_ing_q = len(ingested)
    return {
        "planned_docs": sum(b["n_docs"] for b in specs),
        "written_docs": ingest.get("docs_seen", 0),
        "ingested_docs": len({d.cluster_id for d in ingested}),
        "ingest_drop_reasons": {k: ingest[k] for k in ("too_short", "too_long", "bad_shape", "duplicate_doc_id",
                                                       "extra_doc", "doc_too_many_bad_questions", "missing_file")
                                if ingest.get(k)},
        "question_drop_reasons": {k: ingest[k] for k in ("bad_question_shape", "bad_question_type",
                                                         "too_few_options", "too_many_options", "bad_score_levels",
                                                         "invalid_question", "duplicate_question_name")
                                  if ingest.get(k)},
        "questions_ingested": n_ing_q,
        "questions_survived_check": len(survivors),
        "question_survival_rate": round(len(survivors) / max(n_ing_q, 1), 4),
        "final_docs": len({d.cluster_id for d in kept}),
        "final_questions": len(kept),
        "doc_survival_rate": round(len({d.cluster_id for d in kept}) / max(len({d.cluster_id for d in ingested}), 1), 4),
        "survival_by_doc_kind": {k: {"planned": planned_by_kind[k], "ingested_docs": ing_k[k],
                                     "final_docs": fin_k[k]} for k in kinds},
        "question_survival_by_type": by_type,
        "question_survival_by_topic": by_topic,
        "soft_label_share": round(sum(d.soft_gold is not None for d in kept) / max(len(kept), 1), 4),
        "injection_docs": {"ingested": len(inj_ing), "final": len(inj_final)},
        "final_doc_tokens": {"min": min(tokens), "median": statistics.median(tokens), "max": max(tokens)}
        if tokens else {},
        "token_count_method": sorted({d.metadata.get("token_count", "") for d in ingested}),
    }
