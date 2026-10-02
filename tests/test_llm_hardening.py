import json

import pytest

from yev.generators.common import EDIT_TYPE_DEFINITIONS, make_cluster
from yev.generators.llm.attrib import prepare_pairs, score_attribution
from yev.generators.llm.check import prepare_sheets, score_answers
from yev.generators.llm.families import DOMAINS
from yev.generators.llm.ingest import ingest_file, parse_cluster
from yev.generators.llm.plan import plan_batches, render_writer_prompt
from yev.schema import Option

OPTS = [{"key": "approve", "description": "Approve if the claim is under $500 and has a receipt."},
        {"key": "partial", "description": "Pay half if the receipt is missing but the claim is under $500."},
        {"key": "reject", "description": "Reject claims of $500 or more."}]
BASE = "Dana at Kestrel Tours claims $480 for a client dinner and attaches the itemised receipt from the restaurant."
AB = [Option("a", "Option a."), Option("b", "Option b.")]


def cluster(**kw):
    obj = {
        "question": "Under the expense policy, what happens to this claim?",
        "options": OPTS,
        "base": {"state": BASE, "gold": "approve"},
        "variants": [
            {"state": BASE.replace("$480", "$520"), "gold": "reject", "edit_type": "threshold", "edit": "480->520"},
        ],
        "policy_variants": [],
    }
    obj.update(kw)
    return obj


def many_clusters(n=30):
    out = []
    for i in range(n):
        members = [(f"s{i}-0", AB, "a", None), (f"s{i}-1", AB, "b", "negation")] + [
            (f"s{i}-{j}", AB, "a", "threshold") for j in range(2, 3 + i % 2)]
        out += make_cluster(cluster_id=f"c{i}", family="f", source="synthetic_opus", qtype="choice",
                            question="q?", members=members)
    return out


def attrib_rows():
    return make_cluster(cluster_id="k1", family="f", source="synthetic_opus", qtype="choice", question="q?",
                        members=[("s0", AB, "a", None), ("s1", AB, "b", "negation"), ("s2", AB, "b", "threshold")])


def test_writer_example_single_line_and_definitions():
    b = plan_batches(10, per_batch=10, seed=0)[0]
    text = render_writer_prompt(b, "/tmp/o.jsonl")
    assert "shown on one line" in text and '{"question"' in text
    assert "Tie-breaks" in text and all(EDIT_TYPE_DEFINITIONS[t] in text for t in ("negation", "threshold", "unit"))


def test_multiline_fallback(tmp_path):
    path = tmp_path / "b.jsonl"
    path.write_text("```json\n" + json.dumps(cluster(), indent=2) + "\n```\n")
    ds, stats = ingest_file(path, {"batch_id": "b", "family": "expenses", "qtype": "choice"})
    assert len({d.cluster_id for d in ds}) == 1 and stats["recovered_multiline"] == 1


def test_parts_never_hold_two_members_of_a_cluster():
    ds = many_clusters()
    parts, key = prepare_sheets(ds, part_size=10, seed=5)
    for _, items in parts:
        cl = [key[i["item_id"]]["decision"].rsplit(":", 1)[0] for i in items]
        assert len(cl) == len(set(cl))
    for s in range(3):
        seen = [key[i["item_id"]]["decision"] for _, items in parts for i in items if key[i["item_id"]]["sheet"] == s]
        assert sorted(seen) == sorted(d.id for d in ds)
    pparts, pkey = prepare_pairs(ds, part_size=10)
    for _, items in pparts:
        cl = [pkey[i["pair_id"]]["decision"].rsplit(":", 1)[0] for i in items]
        assert len(cl) == len(set(cl))
    assert sum(len(i) for _, i in pparts) == len([d for d in ds if d.edit_type])


def test_attrib_numeric_group_first_answer_wins_and_junk():
    ds = attrib_rows()
    _, key = prepare_pairs(ds)
    pid = {v["decision"]: p for p, v in key.items() if p != "_no_base"}
    kept, stats = score_attribution(ds, key, [
        {"pair_id": pid["k1:1"], "edit_type": "negation"},
        {"pair_id": pid["k1:1"], "edit_type": "unit"},
        {"pair_id": pid["k1:2"], "edit_type": "date"},
        {"pair_id": "zzz", "edit_type": "date"}, "junk"])
    assert [d.id for d in kept] == ["k1:0", "k1:1", "k1:2"]
    assert stats["attrib_numeric_group_match"] == 1 and stats["duplicate_answer"] == 1
    assert stats["unknown_pair"] == 1 and stats["non_object_answer"] == 1
    kept, _ = score_attribution(ds, key, [{"pair_id": pid["k1:1"], "edit_type": "negation"},
                                          {"pair_id": pid["k1:2"], "edit_type": "negation"}])
    assert [d.id for d in kept] == ["k1:0", "k1:1"]


def test_too_many_options():
    seven = [{"key": f"k{i}", "description": f"Option {i}."} for i in range(7)]
    ds, reasons = parse_cluster(cluster(options=seven), cluster_id="x", family="f", qtype="choice")
    assert ds == [] and reasons["too_many_options"] == 1
    big = [Option(f"k{i}", f"Option {i}.") for i in range(7)]
    d = make_cluster(cluster_id="k", family="f", source="s", qtype="choice", question="q",
                     members=[("a", big, "k0", None), ("b", big, "k1", "negation")])
    with pytest.raises(ValueError, match="options"):
        prepare_sheets(d)


def test_normalisation():
    obj = cluster(options=[{"key": " Approve ", "description": OPTS[0]["description"]}, *OPTS[1:]],
                  base={"state": BASE, "gold": "Approve"},
                  variants=[{"state": BASE.replace("$480", "$520"), "gold": " REJECT", "edit_type": "Threshold", "edit": "x"}])
    ds, reasons = parse_cluster(obj, cluster_id="n", family="f", qtype="choice")
    assert len(ds) == 2 and ds[1].edit_type == "threshold" and not reasons


def test_no_edit_variants_dropped():
    variants = [{"state": BASE, "gold": "reject", "edit_type": "threshold", "edit": "x"},
                {"state": BASE, "gold": "approve", "edit_type": "injection", "edit": "x"},
                {"state": BASE.replace("$480", "$520"), "gold": "reject", "edit_type": "threshold", "edit": "x"}]
    ds, reasons = parse_cluster(cluster(variants=variants), cluster_id="e", family="f", qtype="choice")
    assert len(ds) == 2 and reasons == {"no_edit": 2}


def test_letter_parsing_retry_and_non_dict():
    ds = many_clusters(2)
    parts, key = prepare_sheets(ds, n_sheets=1, seed=0)
    item = parts[0][1][0]
    k = key[item["item_id"]]
    d = next(x for x in ds if x.id == k["decision"])
    letter = next(l for l, ok in k["letters"].items() if ok == d.gold)
    other = next(l for l in k["letters"] if l != letter)
    # one sheet only: a single valid vote is < 2 so the row is missing_votes, but stats show parsing
    _, stats = score_answers(ds, key, [{"item_id": item["item_id"], "letter": "Z"},
                                       {"item_id": item["item_id"], "letter": f"({letter})"}, 5])
    assert stats["bad_letter"] == 1 and stats["duplicate_answer"] == 0 and stats["non_object_answer"] == 1
    _, s2 = score_answers(ds, key, [{"item_id": item["item_id"], "letter": f"Option {other}"},
                                    {"item_id": item["item_id"], "letter": letter}])
    assert s2["bad_letter"] == 0 and s2["duplicate_answer"] == 1  # first valid wins


def test_per_batch_too_large():
    with pytest.raises(ValueError):
        plan_batches(10, per_batch=len(DOMAINS) + 1, seed=0)
