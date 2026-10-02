import json

import pytest

from yev import cli
from yev.generators.common import SYNTH_LICENCE, make_cluster
from yev.generators.llm import documents
from yev.generators.llm.check import prepare_sheets, render_checker_prompt, score_answers
from yev.generators.llm.documents import (
    FAMILY, doc_writer_prompt, ingest_docs, keep_docs, plan_doc_batches, score_doc_answers,
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
    n = n_docs or sum(len(d) for d in docs_by_batch)
    batches = plan_doc_batches(n, per_batch=max(len(d) for d in docs_by_batch), seed=0)
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
                              "ops_handover"}
    for kind, (description, topics) in DOC_KINDS.items():
        assert description and topics and set(topics) <= set(DOC_TOPICS)
    for name, (qtype, description) in DOC_TOPICS.items():
        assert qtype in ("choice", "score") and description


def test_writer_prompt_contains_no_benchmark_names():
    for b in plan_doc_batches(10, per_batch=2, seed=0):
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
    assert len(specs) == 2 and all(p.with_suffix(".prompt.md").exists() for p in specs)
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
            want = d.gold if not (d.cluster_id.endswith("d1") and d.metadata["question_name"] in
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
