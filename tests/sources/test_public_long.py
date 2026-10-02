import random
from dataclasses import replace

import pytest

from yev import licences
from yev.filters.benchmarks import decision_index_ids
from yev.schema import Decision, NOUL_KEYS
from yev.sources import public_long as pl
from yev.sources.base import SourceSpec, build
from yev.sources.registry import SOURCES as REGISTRY

CUAD_Q = 'Highlight the parts (if any) of this contract related to "{c}" that should be reviewed by a lawyer. Details: {d}'


def cuad_row(title, cat, details, answers, context="CONTRACT TEXT  with   spaces"):
    return {"id": f"{title}__{cat}", "title": title, "context": context,
            "question": CUAD_Q.format(c=cat, d=details), "answers": {"text": answers, "answer_start": [0] * len(answers)}}


def valid(d):
    replace(d, source="x", licence="cc-by-4.0").validate()


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def count_words(text):
    return len(text.split())


# --- CUAD -------------------------------------------------------------------------------------------

def test_cuad_prepare_groups_by_contract_and_ors_repeated_rows():
    rows = [
        cuad_row("A-LICENSE AGREEMENT", "Audit Rights", "Does a party have the right to audit?", []),
        cuad_row("A-LICENSE AGREEMENT", "Audit Rights", "Does a party have the right to audit?", ["may audit"]),
        cuad_row("A-LICENSE AGREEMENT", "Insurance", "Is there a requirement for insurance?", []),
        cuad_row("A-LICENSE AGREEMENT", "Parties", "The two or more parties who signed the contract", ["X"]),
    ]
    docs = pl.prepare_cuad(rows)
    assert len(docs) == 1
    items = {it["category"]: it["present"] for it in docs[0]["items"]}
    assert items == {"Audit Rights": True, "Insurance": False}  # extractive Parties is skipped
    assert docs[0]["text"] == "CONTRACT TEXT with spaces"


def test_cuad_prepare_drops_confidentiality_agreements():
    rows = [cuad_row("B-ENDORSEMENT AGREEMENT - Confidentiality and Non-Use Obligations Agreement",
                     "Insurance", "Is there a requirement for insurance?", ["x"]),
            cuad_row("C-NON-DISCLOSURE AGREEMENT", "Insurance", "Is there a requirement for insurance?", ["x"])]
    assert pl.prepare_cuad(rows) == []


def test_convert_cuad_noul_with_cuad_definition_and_balanced_cap():
    cats = [(f"Cat {k}", f"Is there clause {k}?", k % 3 == 0) for k in range(20)]
    rows = [cuad_row("D-SUPPLY AGREEMENT", c, d, ["span"] if p else [], context="Supply text.") for c, d, p in cats]
    [doc] = pl.prepare_cuad(rows)
    out = pl.convert_cuad(doc, 0, random.Random(0), [])
    assert len(out) == pl.PER_DOC
    golds = [d.gold for d in out]
    assert golds.count("yes") == golds.count("no") == pl.PER_DOC // 2
    for d in out:
        assert d.type == "noul" and tuple(d.keys) == NOUL_KEYS and d.family == pl.FAMILY
        assert d.state == "Supply text."
        cat = d.metadata["item"]
        k = int(cat.split()[1])
        assert d.gold == ("yes" if k % 3 == 0 else "no")
        assert f"Is there clause {k}?" in d.question and cat in d.question
        assert d.metadata["doc_id"] == "D-SUPPLY AGREEMENT"
        valid(d)


# --- MAUD -------------------------------------------------------------------------------------------

def maud_row(contract, q, a, text, sub="<NONE>", data_type="main", section="MAE Definition"):
    return {"data_type": data_type, "contract_name": contract, "text": text, "answer": a, "label": "0",
            "question": q, "subquestion": sub, "text_type": section, "id": "1", "category": "c"}


def test_maud_prepare_joins_excerpts_and_takes_options_from_all_contracts():
    rows = [
        maud_row("contract_1", "Type of Consideration-Answer", "All Cash", "Excerpt one."),
        maud_row("contract_1", "Bringdown Timing Answer", "At Closing Only", "Excerpt two."),
        maud_row("contract_1", "Bringdown Timing Answer", "At Closing Only", "Excerpt two.", data_type="abridged"),
        maud_row("contract_1", "Types of R&Ws", "Authority", "Excerpt three.", sub="Authority"),
        maud_row("contract_2", "Type of Consideration-Answer", "All Stock", "Other excerpt."),
        maud_row("contract_2", "Bringdown Timing Answer", "At Signing & At Closing", "Other two."),
        maud_row("contract_2", "Bringdown Timing Answer", None, "Unlabelled."),
        maud_row("contract_2", "Type of Consideration-Answer", "All Cash", None),
    ]
    docs = {d["doc_id"]: d for d in pl.prepare_maud(rows)}
    c1 = docs["contract_1"]
    assert c1["text"] == "Excerpt one.\n\nExcerpt two."  # sub-question rows and non-main rows are left out
    items = {it["question"]: it for it in c1["items"]}
    assert set(items) == {"Type of Consideration-Answer", "Bringdown Timing Answer"}
    assert items["Type of Consideration-Answer"]["options"] == ["All Cash", "All Stock"]
    assert items["Bringdown Timing Answer"]["answer"] == "At Closing Only"
    assert docs["contract_2"]["text"] == "Other excerpt.\n\nOther two."  # rows without answer or text are skipped


def test_maud_prepare_skips_questions_with_one_or_too_many_answers():
    rows = [maud_row(f"c{i}", "Wide question", f"answer {i}", f"text {i}") for i in range(11)]
    rows += [maud_row(f"c{i}", "Single question", "same", f"text {i}") for i in range(3)]
    assert all(not d["items"] for d in pl.prepare_maud(rows))


def test_convert_maud_choice():
    rows = [maud_row("contract_1", "Type of Consideration-Answer", "Mixed Cash/Stock: Election", "Excerpt one."),
            maud_row("contract_2", "Type of Consideration-Answer", "All Cash", "Other.")]
    docs = {d["doc_id"]: d for d in pl.prepare_maud(rows)}
    [d] = pl.convert_maud(docs["contract_1"], 0, random.Random(0), [])
    assert d.type == "choice" and d.family == pl.FAMILY
    assert {o.description for o in d.options} == {"Mixed Cash/Stock: Election", "All Cash"}
    gold = {o.key: o.description for o in d.options}[d.gold]
    assert gold == "Mixed Cash/Stock: Election"
    assert "Type of Consideration" in d.question and "Answer" not in d.question
    assert d.state == "Excerpt one."
    valid(d)


def test_convert_maud_yes_no_answer_set_is_noul_with_clean_topic():
    rows = [maud_row("c1", "Actions taken by Buyer-Answer (Y/N)", "Yes", "MAE text."),
            maud_row("c2", "Actions taken by Buyer-Answer (Y/N)", "No", "Other.")]
    docs = {d["doc_id"]: d for d in pl.prepare_maud(rows)}
    [d] = pl.convert_maud(docs["c2"], 0, random.Random(0), [])
    assert d.type == "noul" and tuple(d.keys) == NOUL_KEYS and d.gold == "no"
    assert "Actions taken by Buyer" in d.question and "Answer" not in d.question and "Y/N" not in d.question
    assert d.question.startswith("Merger agreement review, MAE Definition. ")  # section = text_type
    valid(d)


def test_maud_topic_cleaning():
    assert pl._maud_topic("Knowledge Definition limited to one or more identified persons-Answer (Y/") == \
        "Knowledge Definition limited to one or more identified persons"
    assert pl._maud_topic("Fiduciary exception:  Board determination standard-Answer (no-shop)") == \
        "Fiduciary exception: Board determination standard (no-shop)"
    assert pl._maud_topic("COR standard (board determination only)-answer") == "COR standard (board determination only)"


# --- QASPER -----------------------------------------------------------------------------------------

def ans(yes_no=None, unanswerable=False, evidence=("Some paragraph.",), extractive=()):
    return {"unanswerable": unanswerable, "extractive_spans": list(extractive), "yes_no": yes_no,
            "free_form_answer": "", "evidence": list(evidence), "highlighted_evidence": list(evidence)}


def qasper_row(qas):
    return {"id": "1234.5678", "title": "A Paper", "abstract": "We study things.",
            "full_text": {"section_name": ["Introduction", "Method"], "paragraphs": [["Intro para."], ["Method para."]]},
            "qas": {"question": [q for q, _ in qas], "question_id": [f"q{i}" for i in range(len(qas))],
                    "answers": [{"answer": a, "annotation_id": ["x"] * len(a), "worker_id": ["w"] * len(a)}
                                for _, a in qas]}}


def test_convert_qasper_keeps_agreed_yes_no_and_yes_no_shaped_unanswerable():
    row = qasper_row([
        ("Do they use BERT?", [ans(True), ans(True)]),
        ("Is the dataset public?", [ans(False)]),
        ("Are there privacy concerns?", [ans(unanswerable=True, evidence=())]),
        ("What is the seed lexicon?", [ans(unanswerable=True, evidence=())]),  # not yes/no-shaped
        ("Do they beat the baseline?", [ans(True), ans(False)]),  # annotators disagree
        ("Does it work on French?", [ans(True, evidence=("FLOAT SELECTED: Table 2",))]),  # evidence in a table
        ("Which models?", [ans(extractive=("BERT",))]),
    ])
    out = pl.convert_qasper(row, 0, random.Random(0), [])
    got = {d.question: d.gold for d in out}
    assert got == {"Do they use BERT?": "yes", "Is the dataset public?": "no", "Are there privacy concerns?": "not_stated"}
    d = out[0]
    assert d.type == "choice" and sorted(d.keys) == ["no", "not_stated", "yes"] and d.family == pl.FAMILY
    assert d.state.startswith("A Paper\n\nAbstract\nWe study things.") and "Method\nMethod para." in d.state
    for d in out:
        valid(d)


# --- length window and specs ------------------------------------------------------------------------

def _doc_convert(row, i, rng, labels):
    from yev.schema import Option
    return [Decision(id=str(i), type="noul", state=row["t"], question="q?",
                     options=[Option("yes", "y"), Option("no", "n")], gold="yes",
                     family="f", source="", licence="")]


def test_build_applies_token_window_and_counts_drops():
    spec = SourceSpec(name="toy", hf_id="toy/toy", config=None, split="train", licence="mit",
                      convert=_doc_convert, pool_size=10, max_state_chars=10**6, token_window=(50, 400))
    rows = [{"t": words(10)}, {"t": words(100)}, {"t": words(1000)}]
    kept, stats = build(spec, rows, count=count_words)
    assert [len(d.state.split()) for d in kept] == [100]
    assert stats.dropped == {"too_short": 1, "too_long": 1}
    assert stats.as_dict()["dropped"] == {"too_short": 1, "too_long": 1}


def test_sources_are_licence_clean_pinned_and_capped():
    assert set(pl.SOURCES) == {"cuad", "maud", "qasper"}
    assert sum(s.pool_size for s in pl.SOURCES.values()) <= 3000
    for name, s in pl.SOURCES.items():
        assert s.name == name and licences.is_allowed(s.licence) and s.revision and s.split == "train"
        licences.check(s.hf_id, s.licence, config=s.config, split=s.split, data_files=s.data_files)
        assert s.token_window == (2000, 12288)
    assert all(s in REGISTRY for s in pl.SOURCES.values())


def test_public_long_ids_not_in_decision_index():
    ids = decision_index_ids()
    tails = {i.rsplit("/", 1)[-1] for i in ids}
    for s in pl.SOURCES.values():
        assert s.hf_id.lower() not in ids and s.hf_id.lower().rsplit("/", 1)[-1] not in tails
        assert s.hf_id.lower() not in {x.lower() for x in licences.FORBIDDEN_DATASETS}
