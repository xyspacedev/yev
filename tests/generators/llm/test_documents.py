import inspect
import json
import math
import re
from collections import Counter
from dataclasses import replace as _replace

import pytest

from yev import cli
from yev.generators.common import SYNTH_LICENCE, make_cluster
from yev.generators.llm import documents
from yev.generators.llm.check import prepare_sheets, render_checker_prompt, score_answers
from yev.generators.llm.documents import (
    FAMILY, HARD_WEIGHT, NODOC_SAMPLES, NODOC_STATE, SKILLS, doc_writer_prompt, ingest_docs, keep_docs,
    plan_doc_batches, prepare_diff, prepare_nodoc, render_nodoc_prompt, score_difficulty, score_doc_answers,
    score_nodoc,
)
from yev.generators.llm.families import DOC_KINDS, DOC_TOPICS, DOMAINS
from yev.schema import Option, read_jsonl


def words(n: int) -> str:
    """A document of n words; tests count tokens as words."""
    return " ".join(f"w{i % 97}" for i in range(n))


def count_words(text: str) -> int:
    return len(text.split())


def choice_q(name, gold="refund", n=4):
    keys = ["refund", "escalate", "wait", "insufficient_information", "close", "replace", "call",
            "credit", "deny", "reship"][:n]
    return {"name": name, "type": "choice", "question": f"Question {name}?",
            "options": [{"key": k, "description": f"Do {k}."} for k in keys],
            "gold": gold, "rationale": "Stated in message 4."}


def score_q(name, gold="2", n=5):
    return {"name": name, "type": "score", "question": f"Rate {name}.",
            "options": [{"key": str(i), "description": f"Level {i}."} for i in range(n)],
            "gold": gold, "rationale": "The last message."}


def doc(doc_id, n_words=5000, questions=None):
    return {"doc_id": doc_id, "state": words(n_words),
            "questions": questions if questions is not None else
            [choice_q("next_action"), score_q("customer_sentiment"), choice_q("escalation", "escalate"),
             choice_q("resolution_status", "wait"), score_q("severity", "0", 4), choice_q("claim_support", "close", 5)]}


def write_run(tmp_path, docs_by_batch, n_docs=None):
    run = tmp_path / "run"
    # one batch per list of documents (the plan spreads documents over kinds, so build the specs directly)
    template = plan_doc_batches(1, per_batch=1, seed=0)[0]
    batches = [{**template, "batch_id": f"doc-batch-{i:03d}", "n_docs": len(docs),
                "domains": list(DOMAINS[i:i + len(docs)]), "injection_docs": []}
               for i, docs in enumerate(docs_by_batch)]
    (run / "batches").mkdir(parents=True)
    (run / "written").mkdir()
    for b, docs in zip(batches, docs_by_batch):
        (run / "batches" / f"{b['batch_id']}.json").write_text(json.dumps(b))
        (run / "written" / f"{b['batch_id']}.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    return run, batches


def test_plan_doc_batches_ranges():
    batches = plan_doc_batches(23, per_batch=3, seed=0)
    assert sum(b["n_docs"] for b in batches) == 23 and len(batches) == 8
    assert {b["doc_kind"] for b in batches} == set(DOC_KINDS)
    for b in batches:
        assert b["mode"] == "doc" and b["batch_id"].startswith("doc-batch-")
        assert len(b["domains"]) == b["n_docs"] and set(b["domains"]) <= set(DOMAINS)
        assert 4000 <= b["target_tokens"] <= 10000
        assert 6 <= b["questions_per_doc"] <= 12
        assert all(1 <= k <= b["n_docs"] for k in b["injection_docs"])
    n_inj = sum(len(b["injection_docs"]) for b in batches)
    assert n_inj == 1  # round(5% of 23)
    assert plan_doc_batches(23, 3, 0) == batches and plan_doc_batches(23, 3, 1) != batches
    assert sum(len(b["injection_docs"]) for b in plan_doc_batches(100, 4, 0)) == 5
    with pytest.raises(ValueError):
        plan_doc_batches(5, per_batch=len(DOMAINS) + 1, seed=0)


def test_doc_kinds_and_topics_described():
    assert set(DOC_KINDS) == {"support_thread", "incident_report", "agent_trace", "multi_turn_transcript",
                              "ops_handover", "invoice_email_thread", "contract_amendment", "security_alert_log"}
    for kind, (description, topics) in DOC_KINDS.items():
        assert description and topics and set(topics) <= set(DOC_TOPICS)
    for name, (qtype, description) in DOC_TOPICS.items():
        assert qtype in ("choice", "score") and description


def test_writer_prompt_contains_no_benchmark_names():
    batches = plan_doc_batches(24, per_batch=3, seed=0)
    assert {b["doc_kind"] for b in batches} == set(DOC_KINDS)
    for b in batches:
        prompt = doc_writer_prompt(b, "/tmp/out.jsonl").lower()
        for name in ("WorkflowEvals", "evalsafe", "DecideBench", "JevBench", "R-Judge", "DynaBench"):
            assert name.lower() not in prompt


def test_writer_prompt_instructions():
    b = next(b for b in plan_doc_batches(40, per_batch=2, seed=0) if b["injection_docs"])
    p = doc_writer_prompt(b, "/tmp/out.jsonl")
    assert "/tmp/out.jsonl" in p and DOC_KINDS[b["doc_kind"]][0] in p
    assert all(d in p for d in b["domains"])
    assert f"{b['target_tokens']:,}" in p and str(b["questions_per_doc"]) in p
    for topic in DOC_KINDS[b["doc_kind"]][1]:
        assert topic in p
    for phrase in ('"0"', "3 to 10 options", "insufficient information", "must not change any gold",
                   "voice", "stated in the document", "without reading the document"):
        assert phrase in p, phrase
    assert "injected instruction" in p.lower()
    no_inj = next(b for b in plan_doc_batches(40, per_batch=2, seed=0) if not b["injection_docs"])
    assert "Do not put any injected instruction" in doc_writer_prompt(no_inj, "/tmp/o.jsonl")


def test_ingest_converts_and_shares_state(tmp_path):
    run, batches = write_run(tmp_path, [[doc("alpha"), doc("beta")]])
    rows = ingest_docs(run, count=count_words)
    assert len(rows) == 12
    alpha = [d for d in rows if d.cluster_id.endswith(":alpha")]
    assert len(alpha) == 6 and len({d.state for d in alpha}) == 1 and len({d.cluster_id for d in rows}) == 2
    b = batches[0]
    for d in alpha:
        assert d.cluster_id == f"doc:run:{b['batch_id']}:alpha" and d.id.startswith(d.cluster_id + ":")
        assert d.family == FAMILY == "long_workflow" and d.source.startswith("synthetic_")
        assert d.licence == SYNTH_LICENCE and d.edit_type is None
        assert d.metadata["doc_kind"] == b["doc_kind"] and d.metadata["domain"] == b["domains"][0]
        assert d.metadata["est_tokens"] == 5000 and d.metadata["question_name"] in d.id
        d.validate()
    sent = next(d for d in alpha if d.metadata["question_name"] == "customer_sentiment")
    assert sent.type == "score" and sent.keys == ["0", "1", "2", "3", "4"] and sent.gold == "2"
    assert next(d for d in rows if d.cluster_id.endswith(":beta")).metadata["domain"] == b["domains"][1]


def test_ingest_validates_questions(tmp_path):
    bad = [choice_q("a"), choice_q("b", n=2), choice_q("c", gold="nope"), {**choice_q("d"), "options": [{"key": f"k{i}", "description": "x"} for i in range(11)], "gold": "k0"},
           {"name": "e", "type": "score", "question": "q", "gold": "low", "rationale": "",
            "options": [{"key": k, "description": k} for k in ("low", "mid", "high")]},
           choice_q("a"), choice_q("f", n=10), {**choice_q("g"), "type": "ranking"}, {"name": "h"}]
    ok = [choice_q(f"ok{i}") for i in range(20)]
    stats, dropped = {}, []
    run, _ = write_run(tmp_path, [[doc("mixed", questions=ok + bad)]])
    from collections import Counter
    stats = Counter()
    rows = ingest_docs(run, count=count_words, stats=stats, dropped=dropped)
    assert {d.metadata["question_name"] for d in rows} == {f"ok{i}" for i in range(20)} | {"a", "f"}
    assert stats["too_few_options"] == 1 and stats["invalid_question"] == 1
    assert stats["too_many_options"] == 1 and stats["bad_score_levels"] == 1
    assert stats["duplicate_question_name"] == 1 and stats["bad_question_type"] == 1
    assert stats["bad_question_shape"] == 1 and stats["questions_kept"] == 22 and stats["docs_kept"] == 1


def test_ingest_drops_out_of_range_length(tmp_path):
    from collections import Counter
    run, _ = write_run(tmp_path, [[doc("short", 2999), doc("ok_low", 3000), doc("ok_high", 11000),
                                   doc("long", 11001)]])
    stats, dropped = Counter(), []
    rows = ingest_docs(run, count=count_words, stats=stats, dropped=dropped)
    assert {d.cluster_id.rsplit(":", 1)[1] for d in rows} == {"ok_low", "ok_high"}
    assert stats["too_short"] == 1 and stats["too_long"] == 1 and stats["docs_kept"] == 2
    assert {(x["doc"].rsplit(":", 1)[1], x["reason"]) for x in dropped} == {("short", "too_short"),
                                                                             ("long", "too_long")}


def test_ingest_drops_doc_with_mostly_bad_questions(tmp_path):
    from collections import Counter
    qs = [choice_q("a"), choice_q("b"), choice_q("c", n=2), choice_q("d", n=2), choice_q("e", n=2), choice_q("f")]
    run, _ = write_run(tmp_path, [[doc("x", questions=qs)]])
    stats = Counter()
    assert ingest_docs(run, count=count_words, stats=stats) == []
    assert stats["doc_too_many_bad_questions"] == 1


def test_counter_refuses_without_tokenizer_unless_allowed(monkeypatch):
    def missing(*a, allow_heuristic=False, **k):
        if not allow_heuristic:
            raise documents.padding.TokenizerUnavailable("no tokenizer")
        return documents.padding.heuristic_tokens
    monkeypatch.setattr(documents.padding, "load_token_counter", missing)
    with pytest.raises(documents.padding.TokenizerUnavailable):
        documents.load_doc_counter()
    count, method = documents.load_doc_counter(allow_heuristic=True)
    assert method == "heuristic" and count("x" * 4000) == documents.padding.heuristic_tokens("x" * 4000) == 2001


def test_cli_doc_ingest_refuses_without_tokenizer(tmp_path, monkeypatch, capsys):
    def missing(*a, allow_heuristic=False, **k):
        if not allow_heuristic:
            raise documents.padding.TokenizerUnavailable("tokenizer Qwen/Qwen3.5-4B-Base not found")
        return documents.padding.heuristic_tokens
    monkeypatch.setattr(documents.padding, "load_token_counter", missing)
    run = tmp_path / "run"
    assert cli.main(["synth", "doc-plan", "--dir", str(run), "--docs", "1", "--seed", "0"]) == 0
    (run / "written" / "doc-batch-000.jsonl").write_text(json.dumps({**doc("d0"), "state": "x" * 8000}) + "\n")
    assert cli.main(["synth", "doc-ingest", "--dir", str(run)]) == 2
    out = capsys.readouterr().out
    assert "tokenizer" in out and "--allow-heuristic" in out and not (run / "ingested.jsonl").exists()
    assert cli.main(["synth", "doc-ingest", "--dir", str(run), "--allow-heuristic"]) == 0
    ingest = json.loads((run / "reports" / "ingest.json").read_text())
    assert ingest["token_count"] == "heuristic" and ingest["docs_kept"] == 1  # 8000 chars -> 4001 tokens


def _doc_decisions(n_docs=2, n_q=10):
    out = []
    opts = [Option(k, f"Do {k}.") for k in ("a", "b", "c")]
    for k in range(n_docs):
        rows = make_cluster(cluster_id=f"doc:{k}", family=FAMILY, source="synthetic_opus", qtype="choice",
                            question="q", members=[(f"document {k}", opts, "a", None) for _ in range(n_q)])
        for j, d in enumerate(rows):
            d.id = f"doc:{k}:q{j}"
        out += rows
    return out


def _answers(parts, key, pick):
    out = []
    for _, items in parts:
        for it in items:
            k = key[it["item_id"]]
            letter = next(l for l, ok in k["letters"].items() if ok == pick(k["decision"], k["sheet"]))
            out.append({"item_id": it["item_id"], "letter": letter})
    return out


def test_doc_kept_only_if_70pct_questions_survive():
    ds = _doc_decisions()
    parts, key = prepare_sheets(ds, n_sheets=3, part_size=4, seed=0)
    # doc 0 loses 3 of 10 questions (70% survive: kept), doc 1 loses 4 (60%: dropped)
    lose = {"doc:0:q0", "doc:0:q1", "doc:0:q2", "doc:1:q0", "doc:1:q1", "doc:1:q2", "doc:1:q3"}
    kept, stats = score_doc_answers(ds, key, _answers(parts, key, lambda d, s: "b" if d in lose else "a"))
    assert {d.cluster_id for d in kept} == {"doc:0"} and len(kept) == 7
    assert stats["questions_survived"] == 13 and stats["docs_dropped_below_70pct"] == 1
    assert stats["docs_kept"] == 1 and stats["kept"] == 7
    assert keep_docs(ds[:7], ds) == ds[:7] and keep_docs(ds[:6], ds) == []


def test_soft_gold_from_samples():
    ds = _doc_decisions(n_docs=1)
    parts, key = prepare_sheets(ds, n_sheets=3, part_size=4, seed=1)

    def pick(did, sheet):
        if did == "doc:0:q1" and sheet == 2:
            return "c"
        if did == "doc:0:q2" and sheet in (0, 1):
            return "b"
        return "a"

    kept, stats = score_doc_answers(ds, key, _answers(parts, key, pick))
    by_id = {d.id: d for d in kept}
    assert by_id["doc:0:q1"].soft_gold == pytest.approx({"a": 2 / 3, "b": 0.0, "c": 1 / 3})
    assert by_id["doc:0:q0"].soft_gold is None
    assert "doc:0:q2" not in by_id and stats["checker_disagrees"] == 1


def test_doc_check_parts_never_hold_two_questions_of_one_doc():
    ds = _doc_decisions(n_docs=5, n_q=9)
    parts, key = prepare_sheets(ds, n_sheets=3, part_size=4, seed=0)
    for _, items in parts:
        docs = [key[it["item_id"]]["decision"].rsplit(":", 1)[0] for it in items]
        assert len(docs) == len(set(docs))
        assert all(set(it) == {"item_id", "state", "question", "options"} for it in items)


# --- check.py: 10 letters for documents, existing <=6-option behaviour pinned -------------------

def test_check_supports_ten_options():
    opts = [Option(f"k{i}", f"Option {i}.") for i in range(10)]
    ds = make_cluster(cluster_id="doc:z", family=FAMILY, source="synthetic_opus", qtype="choice", question="q",
                      members=[("doc", opts, "k9", None)])
    parts, key = prepare_sheets(ds, n_sheets=3, seed=0)
    letters = {o["letter"] for _, items in parts for it in items for o in it["options"]}
    assert letters == set("ABCDEFGHIJ")
    answers = _answers(parts, key, lambda d, s: "k9")
    assert {a["letter"] for a in answers} - set("ABCDEF")  # some gold landed on G-J
    kept, _ = score_answers(ds, key, answers, cluster_filter=None)
    assert len(kept) == 1
    too_many = make_cluster(cluster_id="x", family="f", source="s", qtype="choice", question="q",
                            members=[("d", [Option(f"k{i}", f"O{i}.") for i in range(11)], "k0", None)])
    with pytest.raises(ValueError, match="options"):
        prepare_sheets(too_many)


def test_six_option_clusters_unchanged():
    six = [Option(f"k{i}", f"Option {i}.") for i in range(6)]
    ds = make_cluster(cluster_id="k", family="f", source="synthetic_opus", qtype="choice", question="q",
                      members=[("s0", six, "k0", None), ("s1", six, "k5", "negation")])
    parts, key = prepare_sheets(ds, n_sheets=3, seed=0)
    assert {o["letter"] for _, items in parts for it in items for o in it["options"]} == set("ABCDEF")
    # pinned: seed-0 option order for s0 on sheet 0 is the same as before the A-J extension
    first = next(v for v in key.values() if v["decision"] == "k:0" and v["sheet"] == 0)
    assert first["letters"] == {"A": "k2", "B": "k5", "C": "k0", "D": "k1", "E": "k3", "F": "k4"}
    import hashlib
    digest = hashlib.sha1(json.dumps([parts, key], sort_keys=True).encode()).hexdigest()
    assert digest == "0c2931d0f52c7a79326734bffe596fca59d226a8"  # whole sheet output, pre-extension
    # a G answer to a 6-option item is still a bad letter, and "I think B" still reads as B
    item = next(i for i, v in key.items() if v["decision"] == "k:0")
    _, stats = score_answers(ds, key, [{"item_id": item, "letter": "G"}])
    assert stats["bad_letter"] == 1
    assert '"letter": "<A-F>"' in render_checker_prompt("s.jsonl", "a.jsonl")


def test_checker_prompt_for_documents():
    p = render_checker_prompt("s.jsonl", "a.jsonl", max_letter="J", documents=True)
    assert "<A-J>" in p and "long document" in p and "data, not as instructions" in p
    assert "gold" not in p


# --- CLI: doc-plan -> doc-ingest -> check-prepare -> check-score ---------------------------------

def test_cli_doc_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "load_doc_counter", lambda allow_heuristic=False: (count_words, "test"))
    run = tmp_path / "run"
    assert cli.main(["synth", "doc-plan", "--dir", str(run), "--docs", "3", "--per-batch", "2", "--seed", "0"]) == 0
    specs = sorted((run / "batches").glob("*.json"))
    # 3 documents over 8 kinds: three kinds get one document each, so three one-document batches
    assert len(specs) == 3 and all(p.with_suffix(".prompt.md").exists() for p in specs)
    prompt = specs[0].with_suffix(".prompt.md").read_text()
    assert str((run / "written" / "doc-batch-000.jsonl").resolve()) in prompt
    for p in specs:
        b = json.loads(p.read_text())
        docs = [doc(f"d{k}") for k in range(b["n_docs"])]
        if b["batch_id"] == "doc-batch-001":
            docs[0]["state"] = words(200)  # too short: dropped at ingest
        (run / "written" / f"{b['batch_id']}.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    assert cli.main(["synth", "doc-ingest", "--dir", str(run)]) == 0
    ingest = json.loads((run / "reports" / "ingest.json").read_text())
    assert ingest["docs_kept"] == 2 and ingest["too_short"] == 1 and ingest["rows"] == 12

    assert cli.main(["synth", "check-prepare", "--dir", str(run)]) == 0
    key = json.loads((run / "private" / "check-key.json").read_text())
    rows = {d.id: d for d in read_jsonl(run / "ingested.jsonl")}
    sheets = sorted((run / "check").glob("sheet-*.jsonl"))
    assert sheets and "long document" in sheets[0].with_suffix(".prompt.md").read_text()
    for sheet in sheets:
        items = [json.loads(l) for l in sheet.read_text().splitlines()]
        assert len(items) <= documents.CHECK_PART_SIZE
        docs = [rows[key[it["item_id"]]["decision"]].cluster_id for it in items]
        assert len(docs) == len(set(docs))
        out = []
        for it in items:
            k = key[it["item_id"]]
            d = rows[k["decision"]]
            want = d.gold if not ("doc-batch-002" in d.cluster_id and d.metadata["question_name"] in
                                  ("next_action", "escalation", "severity")) else \
                next(x for x in d.keys if x != d.gold)
            out.append(json.dumps({"item_id": it["item_id"],
                                   "letter": next(l for l, x in k["letters"].items() if x == want)}))
        (run / "check" / f"answers-{sheet.stem}.jsonl").write_text("\n".join(out) + "\n")
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 0
    final = list(read_jsonl(run / "final" / "synthetic_docs.jsonl"))
    assert len(final) == 6 and {d.family for d in final} == {FAMILY}
    pilot = json.loads((run / "reports" / "pilot.json").read_text())
    assert pilot["planned_docs"] == 3 and pilot["ingested_docs"] == 2 and pilot["final_docs"] == 1
    assert pilot["questions_ingested"] == 12 and pilot["questions_survived_check"] == 9
    assert pilot["final_questions"] == 6 and pilot["ingest_drop_reasons"]["too_short"] == 1
    assert set(pilot["survival_by_doc_kind"]) <= set(DOC_KINDS)
    assert pilot["question_survival_by_type"]["score"]["ingested"] == 4
    assert cli.main(["synth", "attrib-prepare", "--dir", str(run)]) == 2
    assert cli.main(["synth", "ingest", "--dir", str(run)]) == 2


# --- fix round 1 ---------------------------------------------------------------------------------

def test_doc_dropped_when_survivors_below_70pct_of_written(tmp_path):
    from collections import Counter
    qs = [choice_q(f"q{i}") for i in range(7)] + [choice_q(f"bad{i}", n=2) for i in range(3)]  # 10 written
    run, _ = write_run(tmp_path, [[doc("x", questions=qs)]])
    rows = ingest_docs(run, count=count_words, stats=Counter())
    assert len(rows) == 7 and all(d.metadata["questions_written"] == 10 for d in rows)
    parts, key = prepare_sheets(rows, n_sheets=3, part_size=4, seed=0)
    lose = {rows[0].id, rows[1].id}  # 5 of 7 ingested survive: 71% of ingested, but 50% of written
    kept, stats = score_doc_answers(rows, key, _answers(parts, key,
                                                        lambda d, s: "escalate" if d in lose else "refund"))
    assert kept == [] and stats["docs_dropped_below_70pct"] == 1
    # without the metadata (older rows), the ingested count is the denominator
    for d in rows:
        d.metadata = None
    assert len(keep_docs(rows[2:], rows)) == 5


def test_writer_prompt_requires_rubric_in_document():
    for b in plan_doc_batches(10, per_batch=2, seed=0):
        p = doc_writer_prompt(b, "/tmp/o.jsonl")
        assert "rubric stated inside the document" in p
        assert "SEV" in p and "risk matrix" in p
        line = next(l for l in p.splitlines() if l.startswith('{"doc_id"'))
        example = json.loads(line)
        assert {q["name"] for q in example["questions"]} <= set(DOC_KINDS[b["doc_kind"]][1])
        assert {q["type"] for q in example["questions"]} == {"choice", "score"}


def test_letter_parsing_wide_items():
    from yev.generators.llm.check import parse_letter
    for text, want in [("I think B", "B"), ("B", "B"), (" b ", "B"), ("(J)", "J"), ("Answer: H", "H"),
                       ("answer is g", "G"), ("I'd pick C", "C"), ("The answer: I", "I"), ("I", "I"),
                       ("A or B", None), ("I think B, maybe C", None), ("", None), ("K", None),
                       ("I choose", None)]:
        assert parse_letter(text, 10) == want, text
    assert parse_letter("H", 8) == "H" and parse_letter("J", 8) is None


def test_letter_parsing_six_options_unchanged():
    import re
    from yev.generators.llm.check import parse_letter
    old = re.compile(r"\b([A-F])\b")
    for text in ["I think B", "B", " b ", "A or B", "G", "Answer: C", "(D)", "I", "a, then f", "K", ""]:
        for n in (2, 3, 4, 5, 6):
            m = old.search(text.upper())
            assert parse_letter(text, n) == (m.group(1) if m else None), (text, n)


def test_score_answers_reads_i_think_b_on_ten_options():
    opts = [Option(f"k{i}", f"Option {i}.") for i in range(10)]
    ds = make_cluster(cluster_id="doc:z", family=FAMILY, source="synthetic_opus", qtype="choice", question="q",
                      members=[("doc", opts, "k3", None)])
    parts, key = prepare_sheets(ds, n_sheets=3, seed=0)
    answers = []
    for i, v in key.items():
        letter = next(l for l, k in v["letters"].items() if k == "k3")
        answers.append({"item_id": i, "letter": f"I think {letter}"})
    kept, stats = score_answers(ds, key, answers, cluster_filter=None)
    assert len(kept) == 1 and stats["bad_letter"] == 0


def test_answer_prefix_only_when_letter_is_last():
    from yev.generators.llm.check import parse_letter
    # "Answer:" counts only when one in-range letter is all that follows; otherwise the general rules apply
    assert parse_letter("Answer: I think B", 10) == "B"   # pronoun I ignored, B is the only candidate
    assert parse_letter("Answer: A or B", 10) is None     # two candidates
    assert parse_letter("Answer: a bit of B", 10) == "B"  # lowercase "a" is not a candidate
    assert parse_letter("Answer: J", 9) is None           # J is out of range for 9 options
    assert parse_letter("Answer: C.", 9) == "C"
    assert parse_letter("answer is (g)", 10) == "G"


# --- Task 7b: harder documents, skills, new kinds, no-document control, difficulty signal ------------

def test_writer_prompt_harder_requirements():
    for b in plan_doc_batches(24, per_batch=3, seed=0):
        p = doc_writer_prompt(b, "/tmp/o.jsonl")
        need = math.ceil(0.6 * b["questions_per_doc"])
        assert f"At least {need} of the {b['questions_per_doc']} questions" in p and "60%" in p
        assert "7 to 10 options" in p and "at least half of each document's choice questions" in p.lower()
        assert "no filler" in p.lower() and "plausible" in p
        assert "distractor" in p.lower() and "superseded" in p
        for skill in SKILLS:
            assert f"`{skill}`" in p
        for step in ("two or more separated parts", "not stated", "the document states which update governs and why",
                     "exception stated elsewhere"):
            assert step in p, step
        assert "supersedes an earlier statement" not in p  # no blanket "later wins"
        rules = p[p.index("### Deciding rules are in the document"):p.index("### Distractors")]
        for skill in ("conflict", "exception", "temporal", "arithmetic"):
            assert f"`{skill}`" in rules, skill
        for need in ("precedence", "authority", "scope", "business or calendar days", "inclusive", "time zone",
                     "rounding"):
            assert need in rules, need
        example = json.loads(next(l for l in p.splitlines() if l.startswith('{"doc_id"')))
        assert all(q["skill"] in SKILLS for q in example["questions"])
        assert any(len(q["options"]) >= 7 for q in example["questions"] if q["type"] == "choice")
        # pilot-proven rules stay
        for phrase in ("rubric stated inside the document", "stated in the document", "benchmarks"):
            assert phrase in p, phrase


def test_new_kinds_planned_evenly_and_default_per_batch():
    assert set(SKILLS) == {"lookup", "multi_hop", "arithmetic", "temporal", "conflict", "exception"}
    for kind in ("invoice_email_thread", "contract_amendment", "security_alert_log"):
        description, topics = DOC_KINDS[kind]
        assert description and set(topics) <= set(DOC_TOPICS)
        assert {DOC_TOPICS[t][0] for t in topics} == {"choice", "score"}
    assert inspect.signature(plan_doc_batches).parameters["per_batch"].default == 2
    assert cli.make_parser().parse_args(["synth", "doc-plan", "--dir", "x", "--docs", "1"]).per_batch == 2
    parse = cli.make_parser().parse_args
    assert parse(["synth", "diff-prepare", "--dir", "x"]).part_size == 10 == documents.DIFF_PART_SIZE
    assert parse(["synth", "nodoc-prepare", "--dir", "x"]).part_size == 40
    for n in (8, 50, 420, 421):
        batches = plan_doc_batches(n, seed=0)
        per_kind = Counter()
        for b in batches:
            per_kind[b["doc_kind"]] += b["n_docs"]
            assert 1 <= b["n_docs"] <= 2
        assert set(per_kind) == set(DOC_KINDS) and sum(per_kind.values()) == n
        assert max(per_kind.values()) - min(per_kind.values()) <= 1, (n, per_kind)
        assert len({b["batch_id"] for b in batches}) == len(batches)


def test_ingest_keeps_skill(tmp_path):
    qs = [{**choice_q("next_action"), "skill": "multi_hop"}, {**choice_q("escalation"), "skill": "Temporal"},
          {**choice_q("claim_support"), "skill": "guessing"}, choice_q("resolution_status")]
    run, _ = write_run(tmp_path, [[doc("x", questions=qs)]])
    stats = Counter()
    rows = {d.metadata["question_name"]: d for d in ingest_docs(run, count=count_words, stats=stats)}
    assert rows["next_action"].metadata["skill"] == "multi_hop"
    assert rows["escalation"].metadata["skill"] == "temporal"
    assert rows["claim_support"].metadata["skill"] is None and rows["resolution_status"].metadata["skill"] is None
    assert stats["bad_skill"] == 2 and stats["questions_kept"] == 4


def _rows(n_docs=2, n_q=5, n_opts=8, written=None):
    out = []
    for k in range(n_docs):
        for j in range(n_q):
            opts = [Option(f"k{i}", f"Option {i}.") for i in range(n_opts if j % 2 == 0 else 4)]
            d = make_cluster(cluster_id=f"doc:{k}", family=FAMILY, source="synthetic_opus", qtype="choice",
                             question=f"q{j}?", members=[(f"long document {k}", opts, "k1", None)])[0]
            d.id = f"doc:{k}:q{j}"
            d.metadata = {"doc_kind": ["support_thread", "contract_amendment"][k % 2], "question_name": f"t{j}",
                          "skill": SKILLS[j % len(SKILLS)], "questions_written": written or n_q}
            out.append(d)
    return out


def _pick_answers(parts, key, pick):
    """pick(decision_id, sample) -> option key, or None for no answer."""
    out = []
    for _, items in parts:
        for it in items:
            k = key[it["item_id"]]
            want = pick(k["decision"], k["sheet"])
            if want is not None:
                out.append({"item_id": it["item_id"],
                            "letter": next(l for l, x in k["letters"].items() if x == want)})
    return out


def test_nodoc_state_exact_two_samples_independent_shuffles():
    assert NODOC_STATE == "(The document is not available.)"
    rows = _rows(n_docs=3, n_q=6)
    parts, key = prepare_nodoc(rows, part_size=40, seed=0)
    items = [it for _, its in parts for it in its]
    assert NODOC_SAMPLES == 3
    assert len(items) == 3 * len(rows) and {it["state"] for it in items} == {NODOC_STATE}
    assert all(set(it) == {"item_id", "state", "question", "options"} for it in items)
    assert all(name.startswith(("sheet-0-", "sheet-1-", "sheet-2-")) for name, _ in parts)
    by_q = {}
    for v in key.values():
        by_q.setdefault(v["decision"], []).append(v["letters"])
    assert all(len(v) == 3 for v in by_q.values())
    assert any(len({json.dumps(x, sort_keys=True) for x in v}) == 3 for v in by_q.values())  # independent shuffles
    for _, its in parts:  # never two questions of one document in a part
        docs = [key[it["item_id"]]["decision"].rsplit(":", 1)[0] for it in its]
        assert len(docs) == len(set(docs))
    p = render_nodoc_prompt("/r/nodoc/s.jsonl", "/r/nodoc/a.jsonl", "J")
    assert "not available" in p and "most likely" in p and "Do not refuse" in p and "gold" not in p
    assert "ordinary option" in p and "not because the document is withheld" in p
    tail = render_checker_prompt("/r/nodoc/s.jsonl", "/r/nodoc/a.jsonl", max_letter="J", documents=True)
    assert p.endswith(tail[tail.index("Write your answers"):])  # output format unchanged


def test_nodoc_drops_question_only_when_all_three_samples_correct():
    rows = _rows(n_docs=1, n_q=10)
    parts, key = prepare_nodoc(rows, part_size=40, seed=0)
    three, two, one = "doc:0:q0", "doc:0:q1", "doc:0:q2"

    def pick(d, s):
        right = d == three or (d == two and s in (0, 1)) or (d == one and s == 0)
        return "k1" if right else "k0"
    kept, report = score_nodoc(rows, rows, key, _pick_answers(parts, key, pick))
    by_id = {d.id: d for d in kept}
    assert three not in by_id and len(kept) == 9  # 3 of 3 correct: dropped; 90% of written survive
    assert by_id[two].metadata["nodoc_correct"] == 2  # 2 of 3 correct: kept
    assert by_id[one].metadata["nodoc_correct"] == 1 and by_id["doc:0:q3"].metadata["nodoc_correct"] == 0
    assert all(d.metadata["nodoc_answered"] == 3 for d in kept)
    assert report["dropped_all_correct"] == 1 and report["questions_in"] == 10
    assert report["nodoc_correct"] == {"0": 7, "1": 1, "2": 1, "3": 1}
    assert report["by_skill"][SKILLS[0]]["dropped"] == 1
    assert report["by_doc_kind"]["support_thread"]["dropped"] == 1
    assert report["by_question_type"]["choice"]["questions"] == 10
    assert all(d.state == "long document 0" for d in kept)  # final rows keep their document


def test_nodoc_never_drops_score_questions():
    rows = _rows(n_docs=1, n_q=4)
    levels = [Option(str(i), f"Level {i}.") for i in range(4)]
    rows[0] = _replace(rows[0], type="score", options=levels, gold="2")
    rows[1] = _replace(rows[1], type="score", options=levels, gold="1")
    parts, key = prepare_nodoc(rows, part_size=40, seed=0)
    right = {"doc:0:q0", "doc:0:q2"}  # a score and a choice question, each 3 of 3 correct
    by_row = {d.id: d for d in rows}
    kept, report = score_nodoc(rows, rows, key, _pick_answers(
        parts, key, lambda d, s: by_row[d].gold if d in right else next(k for k in by_row[d].keys
                                                                       if k != by_row[d].gold)))
    by_id = {d.id: d for d in kept}
    assert "doc:0:q0" in by_id and by_id["doc:0:q0"].metadata["nodoc_correct"] == 3  # score: kept
    assert "doc:0:q2" not in by_id  # choice: dropped
    assert report["dropped_all_correct"] == 1
    assert report["score_all_correct"] == 1 and report["score_questions"] == 2
    assert report["score_all_correct_rate"] == 0.5
    assert report["by_question_type"]["choice"]["dropped"] == 1 and "score" not in report["by_question_type"]


def test_nodoc_reapplies_document_rule():
    rows = _rows(n_docs=2, n_q=10)
    parts, key = prepare_nodoc(rows, part_size=40, seed=0)
    # doc 0 loses 3 of 10 (70%: kept); doc 1 loses 4 (60%: whole document dropped)
    lose = {f"doc:0:q{j}" for j in range(3)} | {f"doc:1:q{j}" for j in range(4)}
    kept, report = score_nodoc(rows, rows, key, _pick_answers(parts, key, lambda d, s: "k1" if d in lose else "k2"))
    assert {d.cluster_id for d in kept} == {"doc:0"} and len(kept) == 7
    assert report["docs_dropped_below_70pct"] == 1 and report["final_docs"] == 1 and report["final_questions"] == 7
    # the 70% rule counts against the questions the writer wrote, not those that reached this step
    written = _rows(n_docs=1, n_q=10, written=14)
    parts, key = prepare_nodoc(written, part_size=40, seed=0)
    kept, _ = score_nodoc(written, written, key,
                          _pick_answers(parts, key, lambda d, s: "k1" if d == "doc:0:q0" else "k2"))
    assert kept == []  # 9 of 14 written = 64%


def test_nodoc_missing_answers_never_drop():
    rows = _rows(n_docs=1, n_q=4)
    parts, key = prepare_nodoc(rows, part_size=40, seed=0)
    # q0: right in the 2 answered samples, third missing -> kept; q1: no answers at all
    answers = _pick_answers(parts, key, lambda d, s: None if (d == "doc:0:q0" and s == 2) or d == "doc:0:q1"
                            else ("k1" if d == "doc:0:q0" else "k0"))
    kept, report = score_nodoc(rows, rows, key, answers)
    by_id = {d.id: d for d in kept}
    assert len(kept) == 4 and report["missing_samples"] == 2
    assert by_id["doc:0:q0"].metadata["nodoc_correct"] == 2 and by_id["doc:0:q0"].metadata["nodoc_answered"] == 2
    assert by_id["doc:0:q1"].metadata["nodoc_answered"] == 0


def test_difficulty_weights_and_missing_answers():
    rows = _rows(n_docs=3, n_q=4)
    parts, key = prepare_diff(rows, part_size=30, seed=0)
    items = [it for _, its in parts for it in its]
    assert len(items) == len(rows) and {it["state"] for it in items} == {d.state for d in rows}
    assert all(n.startswith("sheet-0-") for n, _ in parts)
    wrong, missing = {"doc:0:q0", "doc:1:q2"}, {"doc:2:q3"}
    answers = _pick_answers(parts, key, lambda d, s: None if d in missing else ("k0" if d in wrong else "k1"))
    out, report = score_difficulty(rows, key, answers)
    by_id = {d.id: d for d in out}
    assert len(out) == len(rows)
    assert {d.weight for d in out} <= {1.0, HARD_WEIGHT} and HARD_WEIGHT == 1.5
    for d in out:
        if d.id in wrong:
            assert d.weight == 1.5 and d.metadata["small_model_correct"] is False
        elif d.id in missing:
            assert d.weight == 1.0 and "small_model_correct" not in d.metadata
        else:
            assert d.weight == 1.0 and d.metadata["small_model_correct"] is True
    assert report["missing"] == 1 and report["answered"] == 11 and report["missed"] == 2
    assert report["miss_rate"] == round(2 / 11, 4)
    assert report["by_option_count"]["8"]["answered"] == 6 and report["by_option_count"]["8"]["missed"] == 2
    assert set(report["by_skill"]) and set(report["by_doc_kind"])
    # idempotent: scoring the scored rows again gives the same rows
    again, _ = score_difficulty(out, key, answers)
    assert [d.to_dict() for d in again] == [d.to_dict() for d in out]
    assert by_id["doc:0:q0"].soft_gold is None


# --- CLI: check-score -> nodoc-prepare -> nodoc-score -> diff-prepare -> diff-score ------------------

def _skilled_doc(doc_id):
    qs = [{**choice_q("next_action", n=8), "skill": "multi_hop"}, {**score_q("customer_sentiment"), "skill": "lookup"},
          {**choice_q("escalation", "escalate", n=9), "skill": "temporal"},
          {**choice_q("resolution_status", "wait"), "skill": "conflict"},
          {**score_q("severity", "0", 4), "skill": "exception"},
          {**choice_q("claim_support", "close", 7), "skill": "arithmetic"}]
    return doc(doc_id, questions=qs)


def _answer_all(run, sub, key_name, pick):
    """pick(row, sample) -> option key or None. Writes answers-<part>.jsonl next to every sheet in run/sub."""
    key = json.loads((run / "private" / key_name).read_text())
    rows = {d.id: d for d in read_jsonl(run / "ingested.jsonl")}
    for sheet in sorted((run / sub).glob("sheet-*.jsonl")):
        out = []
        for line in sheet.read_text().splitlines():
            it = json.loads(line)
            k = key[it["item_id"]]
            want = pick(rows[k["decision"]], k["sheet"])
            if want is not None:
                out.append(json.dumps({"item_id": it["item_id"],
                                       "letter": next(l for l, x in k["letters"].items() if x == want)}))
        (run / sub / f"answers-{sheet.stem}.jsonl").write_text("\n".join(out) + "\n")


def _checked_run(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "load_doc_counter", lambda allow_heuristic=False: (count_words, "test"))
    run = tmp_path / "run"
    assert cli.main(["synth", "doc-plan", "--dir", str(run), "--docs", "2", "--seed", "0"]) == 0
    for i, p in enumerate(sorted((run / "batches").glob("*.json"))):
        b = json.loads(p.read_text())
        (run / "written" / f"{b['batch_id']}.jsonl").write_text(
            "".join(json.dumps(_skilled_doc(f"{'ab'[i]}{k}")) + "\n" for k in range(b["n_docs"])))
    assert cli.main(["synth", "doc-ingest", "--dir", str(run)]) == 0
    assert cli.main(["synth", "check-prepare", "--dir", str(run)]) == 0
    _answer_all(run, "check", "check-key.json", lambda d, s: d.gold)
    return run


def test_cli_doc_steps_in_order(tmp_path, monkeypatch, capsys):
    run = _checked_run(tmp_path, monkeypatch)
    for step in ("nodoc-prepare", "nodoc-score", "diff-prepare", "diff-score"):
        assert cli.main(["synth", step, "--dir", str(run)]) == 2, step  # check-score has not run
    assert "check-score" in capsys.readouterr().out
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 0
    assert cli.main(["synth", "nodoc-score", "--dir", str(run)]) == 2
    assert cli.main(["synth", "diff-prepare", "--dir", str(run)]) == 2
    assert "nodoc-score" in capsys.readouterr().out

    assert cli.main(["synth", "nodoc-prepare", "--dir", str(run)]) == 0
    nodoc_sheets = sorted((run / "nodoc").glob("sheet-*.jsonl"))
    assert nodoc_sheets and all(s.with_suffix(".prompt.md").exists() for s in nodoc_sheets)
    items = [json.loads(l) for s in nodoc_sheets for l in s.read_text().splitlines()]
    assert len(items) == 36 and {it["state"] for it in items} == {NODOC_STATE}
    assert "not available" in nodoc_sheets[0].with_suffix(".prompt.md").read_text()
    assert cli.main(["synth", "nodoc-prepare", "--dir", str(run)]) == 0  # idempotent before answers exist
    assert sorted((run / "nodoc").glob("sheet-*.jsonl")) == nodoc_sheets
    assert cli.main(["synth", "nodoc-score", "--dir", str(run)]) == 2  # no answers yet
    # the no-doc model gets a0's next_action right in all 3 samples (dropped) and nothing else
    _answer_all(run, "nodoc", "nodoc-key.json",
                lambda d, s: d.gold if (d.cluster_id.endswith(":a0") and d.metadata["question_name"] == "next_action")
                else next(k for k in d.keys if k != d.gold))
    assert cli.main(["synth", "nodoc-prepare", "--dir", str(run)]) == 2  # would discard answers
    capsys.readouterr()
    assert cli.main(["synth", "nodoc-score", "--dir", str(run)]) == 0
    assert cli.main(["synth", "nodoc-score", "--dir", str(run)]) == 0  # idempotent
    final = list(read_jsonl(run / "final" / "synthetic_docs.jsonl"))
    assert len(final) == 11 and all(d.metadata["nodoc_correct"] in (0, 1, 2, 3) for d in final)
    pilot = json.loads((run / "reports" / "pilot.json").read_text())
    assert pilot["nodoc"]["dropped_all_correct"] == 1 and pilot["questions_survived_check"] == 12
    # the top-level final counts follow final/ after every scoring step
    assert pilot["nodoc"]["final_questions"] == 11 == pilot["final_questions"] and pilot["final_docs"] == 2
    assert cli.main(["synth", "diff-score", "--dir", str(run)]) == 2

    assert cli.main(["synth", "diff-prepare", "--dir", str(run)]) == 0
    diff_sheets = sorted((run / "diff").glob("sheet-*.jsonl"))
    items = [json.loads(l) for s in diff_sheets for l in s.read_text().splitlines()]
    assert len(items) == 11 and NODOC_STATE not in {it["state"] for it in items}
    def norm(text):  # same checker wording; only the paths (and the letter range of the part) differ
        return re.sub(r"`[^`]*`", "`<path>`", re.sub(r"<A-[A-J]>", "<A-?>", text))
    diff_prompt = diff_sheets[0].with_suffix(".prompt.md").read_text()
    assert str((run / "diff").resolve()) in diff_prompt
    assert norm(diff_prompt) == norm((run / "check" / "sheet-0-part-0.prompt.md").read_text())
    assert cli.main(["synth", "nodoc-score", "--dir", str(run)]) == 2  # a later step has started
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 2
    # small model misses claim_support everywhere; one answer is missing
    missing = {"n": 0}

    def small(d, s):
        if d.metadata["question_name"] == "resolution_status" and d.cluster_id.endswith(":a0") and not missing["n"]:
            missing["n"] = 1
            return None
        return next(k for k in d.keys if k != d.gold) if d.metadata["question_name"] == "claim_support" else d.gold
    _answer_all(run, "diff", "diff-key.json", small)
    assert cli.main(["synth", "diff-score", "--dir", str(run)]) == 0
    first = (run / "final" / "synthetic_docs.jsonl").read_text()
    assert cli.main(["synth", "diff-score", "--dir", str(run)]) == 0
    assert (run / "final" / "synthetic_docs.jsonl").read_text() == first  # idempotent
    final = list(read_jsonl(run / "final" / "synthetic_docs.jsonl"))
    assert len(final) == 11
    for d in final:
        if d.metadata["question_name"] == "claim_support":
            assert d.weight == 1.5 and d.metadata["small_model_correct"] is False
        elif d.metadata["question_name"] == "resolution_status" and d.cluster_id.endswith(":a0"):
            assert d.weight == 1.0 and "small_model_correct" not in d.metadata
        else:
            assert d.weight == 1.0 and d.metadata["small_model_correct"] is True
        assert "nodoc_correct" in d.metadata and d.metadata["skill"] in SKILLS
    pilot = json.loads((run / "reports" / "pilot.json").read_text())
    assert pilot["difficulty"]["missing"] == 1 and pilot["difficulty"]["missed"] == 2
    assert pilot["final_questions"] == 11 and pilot["final_docs"] == 2
    assert pilot["nodoc"]["dropped_all_correct"] == 1
    assert set(pilot["difficulty"]["by_option_count"]) == {"4", "5", "7", "8", "9"}
    # --force reruns an earlier step and discards the later ones
    assert cli.main(["synth", "nodoc-score", "--dir", str(run), "--force"]) == 0
    assert not (run / "diff").exists() and not (run / "private" / "diff-key.json").exists()
    pilot = json.loads((run / "reports" / "pilot.json").read_text())
    assert "difficulty" not in pilot and pilot["final_questions"] == 11
    assert all(d.weight == 1.0 for d in read_jsonl(run / "final" / "synthetic_docs.jsonl"))


def test_cluster_runs_refuse_doc_steps(tmp_path):
    run = tmp_path / "run"
    assert cli.main(["synth", "plan", "--dir", str(run), "--clusters", "2", "--per-batch", "1", "--seed", "0"]) == 0
    for step in ("nodoc-prepare", "nodoc-score", "diff-prepare", "diff-score"):
        assert cli.main(["synth", step, "--dir", str(run)]) == 2
