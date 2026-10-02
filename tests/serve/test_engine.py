"""Engine: calibrated N-letter readout behind /v1/systemone (Plan 4 Task 3)."""

import json

import pytest

pytest.importorskip("pydantic")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from pydantic import TypeAdapter  # noqa: E402

from jeff.format import LETTERS  # noqa: E402
from jeff.serve.contract import ChoiceAnswer, NoulAnswer, QuestionSpec, ScoreAnswer, Usage  # noqa: E402
from jeff.serve.engine import Engine  # noqa: E402
from jeff.serve.mapping import Unsupported, to_rows  # noqa: E402
from jeff.train import readout  # noqa: E402
from jeff.train.data import N_MAX, letter_token_ids  # noqa: E402
from jeff.train.infer import letter_logits  # noqa: E402

QS = TypeAdapter(dict[str, QuestionSpec])
MARKER = "longer than the maximum model length"


def questions(**qs):
    return QS.validate_python(qs)


CHOICE = {"type": "choice", "instructions": "Which colour?", "criteria": {"red": "warm", "blue": None, "green": "grass"}}
NOUL = {"type": "noul", "instructions": "Is it spam?", "criteria": {"true": "ads", "false": None}}
SCORE = {"type": "score", "instructions": None, "criteria": ["low", "mid", "high", "very high"]}
WIDE = {"type": "choice", "instructions": None, "criteria": {f"k{i}": f"option {i}" for i in range(26)}}


def engine(tok, model, **kw):
    return Engine("unused", None, kw.pop("calibration", None), model=model, tokenizer=tok, **kw)


def calibration(tmp_path, temps):
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"temperatures": temps, "model": "m", "data": "d", "n": 1}))
    return str(p)


def test_answers_independent_of_batch_mates(tok, tiny_model):
    e = engine(tok, tiny_model, max_len=512, batch_tokens=10**6)
    alone, _ = e.answer("a short state", questions(c=CHOICE))
    crowd, _ = e.answer("a short state", questions(c=CHOICE, n=NOUL, s=SCORE, w=WIDE))
    assert set(crowd) == {"c", "n", "s", "w"}
    a, b = alone["c"], crowd["c"]
    assert isinstance(b, ChoiceAnswer) and a.choice == b.choice
    assert set(b.probabilities) == {"red", "blue", "green"}
    for k in a.probabilities:
        assert a.probabilities[k] == pytest.approx(b.probabilities[k], abs=1e-4)
    # The 26-letter batch mate must not leak its letters into a 3-letter row's softmax.
    assert sum(b.probabilities.values()) == pytest.approx(1.0)
    assert set(crowd["w"].probabilities) == set(WIDE["criteria"])


def test_answer_types_and_usage(tok, tiny_model):
    e = engine(tok, tiny_model, max_len=512)
    qs = questions(c=CHOICE, n=NOUL, s=SCORE)
    answers, usage = e.answer({"subject": "hello", "body": "buy now"}, qs)
    assert isinstance(answers["c"], ChoiceAnswer)
    assert isinstance(answers["n"], NoulAnswer) and 0 < answers["n"].noul < 1
    assert isinstance(answers["s"], ScoreAnswer) and 0 <= answers["s"].score <= 3
    rows = to_rows({"subject": "hello", "body": "buy now"}, qs)
    assert usage == {"input_tokens": sum(e.n_tokens(r["messages"]) for r in rows), "output_tokens": 0}
    Usage(**usage)


def test_probabilities_are_calibrated_softmax_over_own_letters(tok, tiny_model, tmp_path):
    temps = {"choice": 0.5, "noul": 2.0, "score": 1.5}
    e = engine(tok, tiny_model, calibration=calibration(tmp_path, temps), max_len=512)
    qs = questions(c=CHOICE, n=NOUL, s=SCORE)
    answers, _ = e.answer("state", qs)
    rows = to_rows("state", qs)
    z = letter_logits(tiny_model, tok, rows, max_len=512)
    want = {r["name"]: readout.probs(zz, len(r["letters"]), temps[r["type"]]) for r, zz in zip(rows, z)}
    assert answers["n"].noul == pytest.approx(want["n"][0], abs=1e-5)
    assert list(answers["c"].probabilities.values()) == pytest.approx(want["c"], abs=1e-5)
    assert list(answers["s"].probabilities.values()) == pytest.approx(want["s"], abs=1e-5)


def test_missing_temperature_defaults_to_one(tok, tiny_model, tmp_path):
    plain = engine(tok, tiny_model, max_len=512)
    only_choice = engine(tok, tiny_model, calibration=calibration(tmp_path, {"choice": 3.0}), max_len=512)
    assert plain.temperatures == {} and only_choice.temperatures == {"choice": 3.0}
    qs = questions(c=CHOICE, n=NOUL, s=SCORE)
    a, _ = plain.answer("state", qs)
    b, _ = only_choice.answer("state", qs)
    assert b["n"].noul == pytest.approx(a["n"].noul, abs=1e-6)
    assert b["s"].probabilities == pytest.approx(a["s"].probabilities, abs=1e-6)
    assert b["c"].probabilities != pytest.approx(a["c"].probabilities, abs=1e-6)


def test_overlong_state_refused(tok, tiny_model, monkeypatch):
    e = engine(tok, tiny_model, max_len=120)
    called = []
    monkeypatch.setattr("jeff.serve.engine.letter_logits", lambda *a, **k: called.append(1))
    short = to_rows("short", questions(c=CHOICE))
    assert e.n_tokens(short[0]["messages"]) <= 120
    with pytest.raises(Unsupported) as err:
        e.answer("word " * 100, questions(c=CHOICE, n=NOUL))
    assert MARKER in err.value.reason and "> 120 tokens)" in err.value.reason
    assert err.value.reason.startswith("state too long: ")
    assert called == []  # refused before any forward pass; never truncated


def test_unsupported_question_refused_before_inference(tok, tiny_model, monkeypatch):
    e = engine(tok, tiny_model, max_len=512)
    monkeypatch.setattr("jeff.serve.engine.letter_logits", lambda *a, **k: pytest.fail("ran the model"))
    too_wide = {"type": "choice", "criteria": {f"k{i}": None for i in range(27)}}
    with pytest.raises(Unsupported, match="options per choice"):
        e.answer("s", questions(c=CHOICE, w=too_wide))


def test_letter_logits_default_unchanged(tok, tiny_model, make_row):
    rows = [make_row("a", {"A": "x", "B": "y"}, "A"), make_row("b", {"A": "x", "B": "y", "C": "z"}, "B", state="much " * 30)]
    default = letter_logits(tiny_model, tok, rows, max_len=512)
    six = letter_logits(tiny_model, tok, rows, max_len=512, n_letters=N_MAX)
    wide = letter_logits(tiny_model, tok, rows, max_len=512, n_letters=26)
    assert N_MAX == 6 and all(len(z) == 6 for z in default)
    assert default == six  # bit-identical
    assert [z[:6] for z in wide] == default and all(len(z) == 26 for z in wide)
    assert letter_token_ids(tok) == [2, 3, 4, 5, 6, 7] == letter_token_ids(tok, N_MAX)


def test_letter_token_ids_asserts_single_distinct_tokens(tok):
    ids = letter_token_ids(tok, 26)
    assert len(ids) == len(set(ids)) == 26 and ids[:6] == [2, 3, 4, 5, 6, 7]

    class SplitsZ(type(tok)):
        def encode(self, text, add_special_tokens=False):
            return [9, 10] if text == "Z" else super().encode(text, add_special_tokens)

    assert len(letter_token_ids(SplitsZ())) == 6  # the training path only needs A-F
    with pytest.raises(AssertionError, match="'Z'"):
        letter_token_ids(SplitsZ(), 26)


def test_engine_checks_all_26_letters_at_load(tok, tiny_model):
    class SplitsQ(type(tok)):
        def encode(self, text, add_special_tokens=False):
            return [9, 10] if text == "Q" else super().encode(text, add_special_tokens)

    with pytest.raises(AssertionError, match="'Q'"):
        engine(SplitsQ(), tiny_model)


def test_chat_returns_argmax_over_present_letters(tok, tiny_model):
    e = engine(tok, tiny_model, max_len=512)
    qs = questions(w=WIDE, c=CHOICE)
    answers, _ = e.answer("state", qs)
    for row in to_rows("state", qs):
        letter = e.chat(row["messages"])
        assert letter in row["letters"]
        assert row["letters"][letter] == answers[row["name"]].choice


def test_chat_only_takes_our_decision_format(tok, tiny_model):
    e = engine(tok, tiny_model, max_len=512)
    for bad in (
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": json.dumps({"state": "s", "question": "q"})}],
        [{"role": "user", "content": json.dumps({"options": [{"label": "AA", "key": "x"}]})}],
        [{"role": "user", "content": json.dumps({"options": [{"label": "A"}, {"label": "A"}]})}],
        [{"role": "system", "content": "sys"}],
        [],
    ):
        with pytest.raises(Unsupported):
            e.chat(bad)
