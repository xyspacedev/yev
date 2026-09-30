import random
from jeff.train.data import make_twin, letter_token_ids, encode, subsample_units, build_batches, encode_all

def test_letter_token_ids_single_and_distinct(tok):
    ids = letter_token_ids(tok)
    assert len(ids) == 6 and len(set(ids)) == 6

def test_twin_remaps_letters_and_target(make_row):
    r = make_row("r1", {"A": "x", "B": "y", "C": "z"}, "B")
    tw = make_twin(r, random.Random(3))
    assert tw["twin_of"] == "r1"
    gold_key = r["letters"][r["answer"]]
    assert tw["letters"][tw["answer"]] == gold_key
    assert tw["target"][tw["answer"]] == 1.0
    assert sorted(tw["letters"].values()) == ["x", "y", "z"]

def test_score_rows_get_no_twin(make_row):
    r = make_row("s1", {"A": "1", "B": "2", "C": "3"}, "C", type_="score")
    assert make_twin(r, random.Random(0)) is None

def test_encode_answer_pos_is_last_prompt_token(tok, make_row):
    e = encode(make_row("r", {"A": "x", "B": "y"}, "A"), tok, max_len=512)
    assert e["answer_pos"] == len(e["input_ids"]) - 1
    assert e["n_letters"] == 2 and len(e["target"]) == 6

def test_overlong_rows_dropped_and_counted(tok, make_row):
    long_row = make_row("L", {"A": "x", "B": "y"}, "A", state="w " * 600)
    enc, stats = encode_all([long_row, make_row("s", {"A": "x", "B": "y"}, "A")], tok, 200, 400, 0.05, 0.1, 0.0, 0)
    assert stats == {"kept": 1, "dropped_overlong": 1}

def test_subsample_keeps_whole_units(make_row):
    rows = [make_row(f"c{i}-{j}", {"A": "x", "B": "y"}, "A", cluster=f"c{i}") for i in range(10) for j in range(3)]
    sub = subsample_units(rows, 0.5, seed=1)
    clusters = {r["cluster_id"] for r in sub}
    assert len(clusters) == 5 and len(sub) == 15

def test_batches_keep_units_and_twins_together(tok, make_row):
    rows = [make_row(f"c{i}-{j}", {"A": "x", "B": "y", "C": "z"}, "A", cluster=f"c{i}") for i in range(6) for j in range(2)]
    enc, _ = encode_all(rows, tok, 512, 1024, 0.05, 0.1, 1.0, 0)
    batches = build_batches(enc, token_budget=120, seed=0)
    assert sorted(i for b in batches for i in b) == list(range(len(enc)))
    where = {i: bi for bi, b in enumerate(batches) for i in b}
    for i, e in enumerate(enc):
        mates = [k for k, f in enumerate(enc) if f["unit"] == e["unit"]]
        assert {where[k] for k in mates} == {where[i]}
