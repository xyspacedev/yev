from jeff.generators.common import EDIT_TYPES, make_cluster
from jeff.generators.llm.attrib import EDIT_TYPE_DEFINITIONS, prepare_pairs, render_attrib_prompt, score_attribution
from jeff.schema import Option

OPTS = [Option("a", "Option a."), Option("b", "Option b.")]


def rows():
    k1 = make_cluster(cluster_id="k1", family="f", source="synthetic_opus", qtype="choice", question="q?",
                      members=[("s0", OPTS, "a", None), ("s1", OPTS, "b", "negation"), ("s2", OPTS, "b", "threshold")])
    headless = make_cluster(cluster_id="k2", family="f", source="synthetic_opus", qtype="choice", question="q?",
                            members=[("t0", OPTS, "a", None), ("t1", OPTS, "b", "negation")])[1:]
    return k1 + headless


def test_definitions_cover_every_edit_type():
    assert set(EDIT_TYPE_DEFINITIONS) == set(EDIT_TYPES)


def test_pairs_skip_headless_clusters():
    parts, key = prepare_pairs(rows())
    items = [it for _, part in parts for it in part]
    assert len(items) == 2
    assert all(it["a"]["state"] == "s0" for it in items)
    assert "edit_type" not in str(items)
    assert key["_no_base"] == ["k2:1"]


def test_scoring_keeps_matches_and_counts_problems():
    ds = rows()
    parts, key = prepare_pairs(ds)
    by_decision = {v["decision"]: pid for pid, v in key.items() if pid != "_no_base"}
    answers = [{"pair_id": by_decision["k1:1"], "edit_type": " Negation "},
               {"pair_id": by_decision["k1:2"], "edit_type": "unit"}]
    kept, stats = score_attribution(ds, key, answers)
    assert [d.id for d in kept] == ["k1:0", "k1:1"]
    assert stats["attrib_mismatch"] == 1 and stats["no_base"] == 1


def test_prompt_lists_definitions_and_paths():
    text = render_attrib_prompt("/x/pairs.jsonl", "/x/attrib-answers.jsonl")
    assert "/x/pairs.jsonl" in text and "/x/attrib-answers.jsonl" in text
    assert all(name in text for name in EDIT_TYPES)
