import random

from yev.sources.nli import (
    convert_multi_nli,
    convert_negation,
    convert_snli_cf,
    convert_vitaminc,
    convert_wanli,
)

R = random.Random(0)


def test_vitaminc_maps_labels_and_clusters_by_case():
    row = {"unique_id": "u1", "case_id": "c9", "claim": "Sales rose.", "evidence": "Sales fell 3%.", "label": "REFUTES",
           "revision_type": "real", "FEVER_id": None}
    [d] = convert_vitaminc(row, 0, R, [])
    assert d.gold == "refuted"
    assert d.cluster_id == "c9"
    assert sorted(d.keys) == ["not_enough_info", "refuted", "supported"]
    assert "Sales rose." in d.state and "Sales fell 3%." in d.state


def test_vitaminc_unknown_label_or_blank_text_skipped():
    assert convert_vitaminc({"unique_id": "u", "case_id": "c", "claim": "x", "evidence": "y", "label": "MAYBE", "revision_type": "real", "FEVER_id": None}, 0, R, []) == []
    assert convert_vitaminc({"unique_id": "u", "case_id": "c", "claim": " ", "evidence": "y", "label": "SUPPORTS", "revision_type": "real", "FEVER_id": None}, 0, R, []) == []


def test_wanli_and_snli_cf_map_nli_labels():
    [w] = convert_wanli({"id": 5, "premise": "p", "hypothesis": "h", "gold": "contradiction"}, 0, R, [])
    assert w.gold == "refuted" and w.id == "5"
    [s] = convert_snli_cf({"idx": 7, "premise": "p", "hypothesis": "h", "label": "neutral"}, 0, R, [])
    assert s.gold == "not_enough_info"


def test_multi_nli_none_label_skipped():
    assert convert_multi_nli({"pairID": "1", "premise": "p", "hypothesis": "h", "label": None}, 0, R, []) == []
    [d] = convert_multi_nli({"pairID": "1", "premise": "p", "hypothesis": "h", "label": "entailment"}, 0, R, [])
    assert d.gold == "supported"


def test_negation_yields_a_contrastive_pair():
    row = {"anchor": "Two girls play outside.", "entailment": "Girls are playing.", "negative": "Girls are not playing."}
    pair = convert_negation(row, 3, R, [])
    assert [d.gold for d in pair] == ["supported", "refuted"]
    assert pair[0].cluster_id == pair[1].cluster_id == "3"
    assert pair[0].id != pair[1].id


def test_vitaminc_drops_fever_derived_rows():
    base = {"unique_id": "u", "case_id": "c", "claim": "x", "evidence": "y", "label": "SUPPORTS"}
    assert convert_vitaminc({**base, "revision_type": "synthetic", "FEVER_id": 123}, 0, R, []) == []
    assert convert_vitaminc({**base, "revision_type": "real", "FEVER_id": 123}, 0, R, []) == []
    assert convert_vitaminc({**base, "revision_type": "synthetic", "FEVER_id": None}, 0, R, []) == []
    assert len(convert_vitaminc({**base, "revision_type": "real", "FEVER_id": ""}, 0, R, [])) == 1
    assert len(convert_vitaminc({**base, "revision_type": "real"}, 0, R, [])) == 1


def test_snli_cf_orig_and_counterfactuals_share_a_cluster():
    def conv(idx):
        return convert_snli_cf({"idx": idx, "premise": "p", "hypothesis": "h", "label": "neutral"}, 0, R, [])[0]

    a, b = conv("3021531305.jpg#0r1n-orig"), conv("3021531305.jpg#0r1n-cf-2")
    assert a.cluster_id == b.cluster_id == "3021531305.jpg#0r1n"
    assert conv("3021531305.jpg#0r1n-cf-0").cluster_id == a.cluster_id
    assert conv("other#0r1n-orig").cluster_id != a.cluster_id


def test_negation_pair_shares_framing_and_option_order():
    row = {"anchor": "Two girls play outside.", "entailment": "Girls are playing.", "negative": "Girls are not playing."}
    for i in range(30):
        a, b = convert_negation(row, i, random.Random(i), [])
        assert a.question == b.question and a.keys == b.keys
        assert a.state.split(":")[0] == b.state.split(":")[0]
