"""System One question <-> our training chat row <-> wire answer (Plan 4 Task 2)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")
from pydantic import TypeAdapter  # noqa: E402

from jeff.format import SYSTEM_PROMPT, user_turn  # noqa: E402
from jeff.schema import Decision, Option  # noqa: E402
from jeff.serve.contract import (  # noqa: E402
    ChoiceAnswer,
    NoulAnswer,
    QuestionSpec,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)
from jeff.serve.mapping import (  # noqa: E402
    DEFAULT_QUESTION,
    MAX_LETTERS,
    Unsupported,
    to_answers,
    to_rows,
)

FIXTURES = Path(__file__).parent / "fixtures"
QS = TypeAdapter(dict[str, QuestionSpec])


def qs(raw: dict):
    return QS.validate_python(raw)


def user(row) -> dict:
    return json.loads(row["messages"][-1]["content"])


def decision(state, question, options, type_):
    return Decision(id="x", type=type_, state=state, question=question,
                    options=[Option(k, d) for k, d in options], gold=options[0][0],
                    family="f", source="s", licence="l")


# ---------------------------------------------------------------------------------------------
# Noul


@pytest.mark.parametrize("criteria", [
    {"true": "It is spam.", "false": "It is not spam."},
    {"false": "It is not spam.", "true": "It is spam."},
])
def test_noul_true_probability_is_true_option(criteria):
    questions = qs({"spam": {"type": "noul", "instructions": "Spam?", "criteria": criteria}})
    (row,) = to_rows("Buy now!", questions)
    assert row["letters"] == {"A": "yes", "B": "no"}  # training's Noul keys, unshuffled: yes first
    opts = user(row)["options"]
    assert opts[0] == {"label": "A", "key": "yes", "description": "It is spam."}
    assert opts[1] == {"label": "B", "key": "no", "description": "It is not spam."}
    answers = to_answers(questions, [row], [[0.8, 0.2]])
    assert answers["spam"] == NoulAnswer(noul=0.8)
    assert answers["spam"].model_dump() == {"type": "noul", "noul": pytest.approx(0.8)}


def test_noul_null_criteria_and_null_instructions():
    questions = qs({
        "bare": {"type": "noul", "instructions": None, "criteria": None},
        "half": {"type": "noul", "instructions": "Spam?", "criteria": {"true": "ads", "false": None}},
        "absent": {"type": "noul"},
    })
    bare, half, absent = to_rows("s", questions)
    assert user(bare)["question"] == DEFAULT_QUESTION["noul"]
    assert user(bare)["options"] == [{"label": "A", "key": "yes", "description": "yes"},
                                     {"label": "B", "key": "no", "description": "no"}]
    assert user(half)["question"] == "Spam?"
    assert [o["description"] for o in user(half)["options"]] == ["ads", "no"]
    assert user(absent) == user(bare)


# ---------------------------------------------------------------------------------------------
# Choice


def test_choice_probs_sum_to_one_and_keys_echo_criteria():
    questions = qs({"c": {"type": "choice", "instructions": "Which?",
                          "criteria": {"zeta": "Z.", "alpha": "A.", "mid": "M."}}})
    (row,) = to_rows("s", questions)
    assert row["letters"] == {"A": "zeta", "B": "alpha", "C": "mid"}  # criteria order, no shuffle
    ans = to_answers(questions, [row], [[0.2, 0.5, 0.3]])["c"]
    assert isinstance(ans, ChoiceAnswer)
    assert list(ans.probabilities) == ["zeta", "alpha", "mid"]
    assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert ans.choice == "alpha" and ans.choice in ans.probabilities
    assert ans.confidence == pytest.approx(0.5)


def test_choice_null_description_uses_key_and_null_instructions_default():
    questions = qs({"c": {"type": "choice", "instructions": None, "criteria": {"billing": None, "other": "Else."}}})
    (row,) = to_rows("s", questions)
    u = user(row)
    assert u["question"] == DEFAULT_QUESTION["choice"]
    assert u["options"] == [{"label": "A", "key": "billing", "description": "billing"},
                            {"label": "B", "key": "other", "description": "Else."}]


def test_structured_instructions_and_descriptions_render_as_json_text():
    questions = qs({"c": {"type": "choice", "instructions": {"ask": "which", "n": 2},
                          "criteria": {"a": ["x", "y"], "b": {"k": "ü"}}}})
    u = user(to_rows("s", questions)[0])
    assert u["question"] == '{"ask": "which", "n": 2}'
    assert [o["description"] for o in u["options"]] == ['["x", "y"]', '{"k": "ü"}']


def test_probabilities_are_renormalised():
    questions = qs({"c": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
    rows = to_rows("s", questions)
    ans = to_answers(questions, rows, [[2.0, 6.0]])["c"]
    assert ans.probabilities == {"a": pytest.approx(0.25), "b": pytest.approx(0.75)}


def test_wrong_number_of_probabilities_is_an_error():
    questions = qs({"c": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
    rows = to_rows("s", questions)
    with pytest.raises(ValueError):
        to_answers(questions, rows, [[1.0]])


# ---------------------------------------------------------------------------------------------
# Score


def test_score_expected_level():
    questions = qs({"u": {"type": "score", "instructions": "How?",
                          "criteria": ["one", "two", "three", "four", "five"]}})
    (row,) = to_rows("s", questions)
    assert row["letters"] == {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4"}  # scale order
    assert [o["description"] for o in user(row)["options"]] == ["one", "two", "three", "four", "five"]
    ans = to_answers(questions, [row], [[0, 0, 0.5, 0.5, 0]])["u"]
    assert isinstance(ans, ScoreAnswer)
    assert ans.score == pytest.approx(2.5)  # levels are positions 0..n-1
    assert ans.confidence == pytest.approx(0.5)
    assert ans.legend == {"0": "one", "1": "two", "2": "three", "3": "four", "4": "five"}
    assert list(ans.probabilities) == list(ans.legend)
    assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_score_legend_echoes_structured_entries_verbatim_and_null_instructions():
    questions = qs({"u": {"type": "score", "instructions": None, "criteria": [{"lvl": "lo"}, "hi"]}})
    (row,) = to_rows("s", questions)
    assert user(row)["question"] == DEFAULT_QUESTION["score"]
    assert user(row)["options"][0]["description"] == '{"lvl": "lo"}'
    ans = to_answers(questions, [row], [[0.25, 0.75]])["u"]
    assert ans.legend == {"0": {"lvl": "lo"}, "1": "hi"}
    assert ans.score == pytest.approx(0.75)


# ---------------------------------------------------------------------------------------------
# Refusals


def test_too_many_options_refuses_whole_request():
    many = {f"k{i}": f"Option {i}." for i in range(MAX_LETTERS + 1)}
    questions = qs({
        "ok1": {"type": "noul", "instructions": "Yes?"},
        "big": {"type": "choice", "instructions": "Which?", "criteria": many},
        "ok2": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}},
    })
    with pytest.raises(Unsupported) as e:
        to_rows("s", questions)
    assert e.value.reason == "too many options per choice (27 > 26)"
    assert str(e.value) == e.value.reason


def test_twenty_six_options_is_fine():
    questions = qs({"c": {"type": "choice", "criteria": {f"k{i}": "d" for i in range(26)}}})
    (row,) = to_rows("s", questions)
    assert list(row["letters"]) == [chr(ord("A") + i) for i in range(26)]


def test_too_many_score_levels_refused():
    questions = qs({"u": {"type": "score", "criteria": [str(i) for i in range(27)]}})
    with pytest.raises(Unsupported, match="options per choice"):
        to_rows("s", questions)


@pytest.mark.parametrize("criteria", [{}, {"only": "One."}])
def test_choice_needs_two_options(criteria):
    questions = qs({"ok": {"type": "noul"}, "c": {"type": "choice", "criteria": criteria}})
    with pytest.raises(Unsupported) as e:
        to_rows("s", questions)
    assert e.value.reason == "a choice needs at least two options"


# ---------------------------------------------------------------------------------------------
# Byte-identity with training


def test_user_turn_matches_training_format():
    questions = qs({
        "n": {"type": "noul", "instructions": "Spam?", "criteria": {"true": "ads", "false": "not ads"}},
        "c": {"type": "choice", "instructions": "Which?", "criteria": {"a": "A.", "b": "B."}},
        "s": {"type": "score", "instructions": "How bad?", "criteria": ["lo", "mid", "hi"]},
    })
    rows = to_rows("Some state.", questions)
    expected = {
        "n": decision("Some state.", "Spam?", [("yes", "ads"), ("no", "not ads")], "noul"),
        "c": decision("Some state.", "Which?", [("a", "A."), ("b", "B.")], "choice"),
        "s": decision("Some state.", "How bad?", [("0", "lo"), ("1", "mid"), ("2", "hi")], "score"),
    }
    assert [r["name"] for r in rows] == ["n", "c", "s"]
    for row in rows:
        d = expected[row["name"]].validate()
        assert row["type"] == d.type
        assert row["messages"] == [{"role": "system", "content": SYSTEM_PROMPT},
                                   {"role": "user", "content": user_turn(d, d.options)}]


def test_json_state_is_serialised_like_jevbench():
    state = {"conversation": [{"role": "customer", "text": "Café, charged twice."}]}
    (row,) = to_rows(state, qs({"n": {"type": "noul"}}))
    assert user(row)["state"] == json.dumps(state, ensure_ascii=False)
    (row,) = to_rows([1, "two"], qs({"n": {"type": "noul"}}))
    assert user(row)["state"] == '[1, "two"]'
    (row,) = to_rows("", qs({"n": {"type": "noul"}}))
    assert user(row)["state"] == ""


def test_workflowevals_fixture_round_trip():
    req = SystemOneRequest.model_validate(json.loads((FIXTURES / "request_workflowevals.json").read_text()))
    rows = to_rows(req.state, req.questions)
    assert [r["name"] for r in rows] == list(req.questions)
    probs = [[1 / len(r["letters"])] * len(r["letters"]) for r in rows]
    answers = to_answers(req.questions, rows, probs)
    resp = SystemOneResponse.model_validate(
        {"model": "yev-4b", "answers": {k: a.model_dump() for k, a in answers.items()},
         "usage": {"input_tokens": 1, "output_tokens": 0}})
    assert set(resp.answers) == set(req.questions)
    for name, q in req.questions.items():
        assert resp.answers[name].type == q.type
