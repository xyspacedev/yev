import pytest

from jeff.generators.common import make_cluster
from jeff.generators.llm.check import prepare_sheets, render_checker_prompt, score_answers
from jeff.schema import Option

CHOICE = [Option("a", "Option a."), Option("b", "Option b."), Option("c", "Option c.")]
SCALE = [Option("low", "Low."), Option("mid", "Mid."), Option("high", "High.")]


def decisions():
    choice = make_cluster(cluster_id="k1", family="f", source="synthetic_opus", qtype="choice", question="Which?",
                          members=[("s0", CHOICE, "a", None), ("s1", CHOICE, "b", "negation"), ("s2", CHOICE, "c", "threshold")])
    score = make_cluster(cluster_id="k2", family="f", source="synthetic_opus", qtype="score", question="How bad?",
                         members=[("t0", SCALE, "low", None), ("t1", SCALE, "high", "threshold")])
    return choice + score


def answer_all(parts, key, pick):
    out = []
    for _, items in parts:
        for it in items:
            k = key[it["item_id"]]
            wanted = pick(k["decision"], k["sheet"])
            letter = next(l for l, ok in k["letters"].items() if ok == wanted)
            out.append({"item_id": it["item_id"], "letter": letter})
    return out


def test_sheets_blind_shuffled_and_score_order_kept():
    ds = decisions()
    parts, key = prepare_sheets(ds, n_sheets=3, part_size=2, seed=0)
    assert all(len(items) <= 2 for _, items in parts)
    per_sheet = {}
    for name, items in parts:
        for it in items:
            assert set(it) == {"item_id", "state", "question", "options"}
            assert "gold" not in str(it) and all(set(o) == {"letter", "description"} for o in it["options"])
            per_sheet.setdefault(key[it["item_id"]]["sheet"], []).append(key[it["item_id"]]["decision"])
            if key[it["item_id"]]["decision"].startswith("k2"):
                assert [o["description"] for o in it["options"]] == ["Low.", "Mid.", "High."]
    assert {s: sorted(v) for s, v in per_sheet.items()} == {s: sorted(d.id for d in ds) for s in range(3)}
    def orders(did):
        return {tuple(key[i]["letters"].values()) for i in key if key[i]["decision"] == did}

    assert any(len(orders(did)) > 1 for did in ("k1:0", "k1:1", "k1:2"))


def test_unanimous_answers_keep_rows_without_soft_labels():
    ds = decisions()
    parts, key = prepare_sheets(ds, seed=1)
    gold = {d.id: d.gold for d in ds}
    kept, stats = score_answers(ds, key, answer_all(parts, key, lambda d, s: gold[d]))
    assert len(kept) == 5 and all(d.soft_gold is None for d in kept)


def test_majority_gives_soft_label_and_minority_drops_row():
    ds = decisions()
    parts, key = prepare_sheets(ds, seed=2)
    gold = {d.id: d.gold for d in ds}

    def pick(did, sheet):
        if did == "k1:1" and sheet == 0:
            return "c"
        if did == "k1:2" and sheet in (0, 1):
            return "a"
        return gold[did]

    kept, stats = score_answers(ds, key, answer_all(parts, key, pick))
    by_id = {d.id: d for d in kept}
    assert by_id["k1:1"].soft_gold == pytest.approx({"a": 0.0, "b": 2 / 3, "c": 1 / 3})
    assert "k1:2" not in by_id and stats["checker_disagrees"] == 1


def test_messy_answers_are_normalised_or_counted():
    ds = decisions()
    parts, key = prepare_sheets(ds, seed=3)
    gold = {d.id: d.gold for d in ds}
    answers = answer_all(parts, key, lambda d, s: gold[d])
    answers[0]["letter"] = f"  {answers[0]['letter'].lower()} "
    answers.append(dict(answers[1]))
    answers.append({"item_id": "nope", "letter": "A"})
    answers.append({"item_id": answers[2]["item_id"], "letter": "Z"})
    kept, stats = score_answers(ds, key, answers)
    assert len(kept) == 5
    assert stats["duplicate_answer"] == 2 and stats["unknown_item"] == 1


def test_cluster_collapses_when_only_one_gold_survives():
    ds = decisions()
    parts, key = prepare_sheets(ds, seed=4)
    gold = {d.id: d.gold for d in ds}
    kept, stats = score_answers(ds, key, answer_all(parts, key, lambda d, s: "low" if d == "k2:1" else gold[d]))
    assert {d.cluster_id for d in kept} == {"k1"} and stats["cluster_collapsed"] == 1


def test_checker_prompt_mentions_paths_and_blindness():
    text = render_checker_prompt("/x/sheet-0-part-0.jsonl", "/x/answers-sheet-0-part-0.jsonl")
    assert "/x/sheet-0-part-0.jsonl" in text and "/x/answers-sheet-0-part-0.jsonl" in text
    assert "Do not open any other file" in text
