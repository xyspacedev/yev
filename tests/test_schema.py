import json

import pytest

from jeff.schema import Decision, Option, SchemaError, read_jsonl, write_jsonl


def make(**overrides) -> Decision:
    base = dict(
        id="t:1",
        type="choice",
        state="Agent wants to drop the prod table.",
        question="How should this action be handled?",
        options=[Option("approve", "Proceed."), Option("review", "Hold."), Option("block", "Refuse.")],
        gold="block",
        family="policy",
        source="test",
        licence="mit",
    )
    base.update(overrides)
    return Decision(**base)


def test_valid_choice_passes():
    assert make().validate().keys == ["approve", "review", "block"]


def test_gold_must_be_an_option():
    with pytest.raises(SchemaError, match="gold"):
        make(gold="escalate").validate()


def test_duplicate_keys_rejected():
    with pytest.raises(SchemaError, match="duplicate"):
        make(options=[Option("a", "x"), Option("a", "y")], gold="a").validate()


def test_bad_key_rejected():
    with pytest.raises(SchemaError, match="bad option key"):
        make(options=[Option("Approve", "x"), Option("b", "y")], gold="b").validate()


def test_choice_option_count_bounds():
    seven = [Option(f"o{i}", "d") for i in range(7)]
    with pytest.raises(SchemaError, match="2-6"):
        make(options=seven, gold="o0").validate()


def test_noul_requires_yes_no_in_order():
    ok = make(type="noul", options=[Option("yes", "y"), Option("no", "n")], gold="no")
    assert ok.validate().gold == "no"
    with pytest.raises(SchemaError, match="noul"):
        make(type="noul", options=[Option("no", "n"), Option("yes", "y")], gold="no").validate()


def test_score_needs_three_to_eleven():
    with pytest.raises(SchemaError, match="3-11"):
        make(type="score", options=[Option("low", "l"), Option("high", "h")], gold="low").validate()


def test_soft_gold_must_match_keys_and_sum_to_one():
    make(soft_gold={"approve": 0.1, "review": 0.2, "block": 0.7}).validate()
    with pytest.raises(SchemaError, match="sum"):
        make(soft_gold={"approve": 0.1, "review": 0.2, "block": 0.2}).validate()
    with pytest.raises(SchemaError, match="keys"):
        make(soft_gold={"approve": 1.0}).validate()


def test_empty_state_and_blank_description_rejected():
    with pytest.raises(SchemaError, match="state"):
        make(state="   ").validate()
    with pytest.raises(SchemaError, match="description"):
        make(options=[Option("a", " "), Option("b", "y")], gold="b").validate()


def test_unknown_split_and_missing_provenance_rejected():
    with pytest.raises(SchemaError, match="split"):
        make(split="test").validate()
    with pytest.raises(SchemaError, match="licence"):
        make(licence="").validate()


def test_jsonl_round_trip(tmp_path):
    d = make(soft_gold={"approve": 0.0, "review": 0.0, "block": 1.0}, cluster_id="c1")
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, [d]) == 1
    [back] = list(read_jsonl(path))
    assert back == d
    assert json.loads(path.read_text())["options"][0] == {"key": "approve", "description": "Proceed."}


def test_write_jsonl_refuses_invalid_rows(tmp_path):
    with pytest.raises(SchemaError):
        write_jsonl(tmp_path / "x.jsonl", [make(gold="nope")])


def test_write_jsonl_atomic_on_error(tmp_path):
    path = tmp_path / "x.jsonl"

    # Write a valid row first
    write_jsonl(path, [make()])
    assert path.exists()

    # Try to write with a valid row followed by an invalid row
    invalid = [make(), make(gold="nope")]
    with pytest.raises(SchemaError):
        write_jsonl(path, invalid)

    # The file and temp file should not exist
    assert not path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
