"""State-prefix KV caching (Plan 4 Task 7): the cached readout must equal the full-sequence one."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pydantic")
from pydantic import TypeAdapter  # noqa: E402

import yev.serve.engine as engine_mod  # noqa: E402
from yev.serve.contract import QuestionSpec  # noqa: E402
from yev.serve.engine import Engine  # noqa: E402
from yev.serve.mapping import to_rows  # noqa: E402
from yev.serve.prefix import common_prefix_len, prefix_letter_logits  # noqa: E402
from yev.train.data import letter_token_ids, prompt_ids  # noqa: E402
from yev.train.infer import letter_logits  # noqa: E402

QS = TypeAdapter(dict[str, QuestionSpec])
CHOICE = {"type": "choice", "instructions": "Which colour fits the long state best?",
          "criteria": {"red": "warm", "blue": None, "green": "grass"}}
NOUL = {"type": "noul", "instructions": "Is it spam?", "criteria": {"true": "ads", "false": None}}
SCORE = {"type": "score", "instructions": None, "criteria": ["low", "mid", "high", "very high"]}
WIDE = {"type": "choice", "instructions": None, "criteria": {f"k{i}": f"option {i}" for i in range(26)}}
STATE = {"invoice": " ".join(f"line{i} amount {i * 7}" for i in range(40)), "vendor": "Acme"}


def questions(**qs):
    return QS.validate_python(qs)


def rows_and_ids(tok, qs=None):
    rows = to_rows(STATE, qs or questions(c=CHOICE, n=NOUL, s=SCORE, w=WIDE))
    return rows, [prompt_ids(tok, r["messages"]) for r in rows]


@pytest.fixture
def tiny_qwen():
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5ForCausalLM"):
        pytest.skip("this transformers has no Qwen3_5ForCausalLM")
    torch.manual_seed(0)
    cfg = tr.Qwen3_5TextConfig(
        vocab_size=2048, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"])
    return tr.Qwen3_5ForCausalLM(cfg).eval()


# --- common_prefix_len ---------------------------------------------------------------------------

def test_common_prefix_identical_keeps_one_suffix_token():
    assert common_prefix_len([[1, 2, 3], [1, 2, 3]]) == 2


def test_common_prefix_none():
    assert common_prefix_len([[1, 2, 3], [4, 2, 3]]) == 0


def test_common_prefix_one_is_prefix_of_other():
    assert common_prefix_len([[1, 2], [1, 2, 3, 4]]) == 1
    assert common_prefix_len([[1, 2, 3, 4], [1, 2]]) == 1


def test_common_prefix_partial_and_single():
    assert common_prefix_len([[1, 2, 3, 9], [1, 2, 3, 8], [1, 2, 7]]) == 2
    assert common_prefix_len([[5, 6, 7]]) == 2


# --- parity with the full-sequence readout ------------------------------------------------------

def _parity(model, tok, batch_tokens):
    rows, ids = rows_and_ids(tok)
    P = common_prefix_len(ids)
    assert P > 50 and len({len(s) - P for s in ids}) > 1  # a real shared state, suffixes of different lengths
    want = letter_logits(model, tok, rows, max_len=4096, n_letters=26)
    got = prefix_letter_logits(model, ids[0][:P], [s[P:] for s in ids], letter_token_ids(tok, 26), batch_tokens)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g == pytest.approx(w, abs=1e-4)
    return got


def test_prefix_parity_tiny_llama(tok, tiny_model):
    _parity(tiny_model, tok, batch_tokens=10**6)  # all suffixes in one batch
    _parity(tiny_model, tok, batch_tokens=1)      # one suffix per batch


def test_prefix_parity_tiny_qwen3_5_hybrid(tok, tiny_qwen):
    _parity(tiny_qwen, tok, batch_tokens=10**6)
    _parity(tiny_qwen, tok, batch_tokens=1)


@pytest.mark.parametrize("which", ["llama", "qwen"])
def test_cache_not_mutated_across_batches(tok, tiny_model, request, which):
    model = tiny_model if which == "llama" else request.getfixturevalue("tiny_qwen")
    _, ids = rows_and_ids(tok)
    P = common_prefix_len(ids)
    sufs = [s[P:] for s in ids]
    L = letter_token_ids(tok, 26)
    fwd = prefix_letter_logits(model, ids[0][:P], sufs, L, batch_tokens=1)
    rev = prefix_letter_logits(model, ids[0][:P], sufs[::-1], L, batch_tokens=1)[::-1]
    big = prefix_letter_logits(model, ids[0][:P], sufs, L, batch_tokens=10**6)
    # Exactly equal (not just close) when each suffix runs alone: the prefix cache is the same every time.
    assert fwd == rev
    for a, b in zip(fwd, big):
        assert a == pytest.approx(b, abs=1e-4)


# --- Engine -------------------------------------------------------------------------------------

def _engine(tok, model, **kw):
    return Engine("unused", None, None, model=model, tokenizer=tok, max_len=4096, **kw)


def test_engine_prefix_cache_answers_equal(tok, tiny_model, monkeypatch):
    calls = []
    real = engine_mod.prefix_letter_logits
    monkeypatch.setattr(engine_mod, "prefix_letter_logits", lambda *a, **k: calls.append(1) or real(*a, **k))
    qs = questions(c=CHOICE, n=NOUL, s=SCORE, w=WIDE)
    on, u_on = _engine(tok, tiny_model, prefix_cache=True, min_prefix_tokens=16).answer(STATE, qs)
    assert calls == [1]
    off, u_off = _engine(tok, tiny_model, prefix_cache=False).answer(STATE, qs)
    assert calls == [1] and u_on == u_off
    for k in qs:
        a, b = on[k].model_dump(), off[k].model_dump()
        assert a.keys() == b.keys()
        for f in a:
            if f == "probabilities":
                assert a[f] == pytest.approx(b[f], abs=1e-4)
            elif isinstance(a[f], float):
                assert a[f] == pytest.approx(b[f], abs=1e-4)
            else:
                assert a[f] == b[f]


def test_engine_uses_exact_path_for_one_question_or_short_prefix(tok, tiny_model, monkeypatch):
    monkeypatch.setattr(engine_mod, "prefix_letter_logits", lambda *a, **k: pytest.fail("prefix path used"))
    e = _engine(tok, tiny_model)  # prefix cache on by default, min_prefix_tokens 256
    assert e.prefix_cache and e.min_prefix_tokens == 256
    e.answer(STATE, questions(c=CHOICE))                    # one question
    e.answer("short", questions(c=CHOICE, n=NOUL))           # common prefix < 256 tokens
    _engine(tok, tiny_model, prefix_cache=False, min_prefix_tokens=1).answer(STATE, questions(c=CHOICE, n=NOUL))


def test_engine_exact_path_bit_identical(tok, tiny_model):
    qs = questions(c=CHOICE, n=NOUL)
    rows = to_rows("short", qs)
    e = _engine(tok, tiny_model)
    assert e._logits(rows, 3) == letter_logits(tiny_model, tok, rows, max_len=4096, batch_tokens=16384, n_letters=3)


def test_serve_cli_prefix_flags():
    from yev.cli import make_parser
    a = make_parser().parse_args(["serve", "--model", "m"])
    assert a.prefix_cache is True and a.min_prefix_tokens == 256
    a = make_parser().parse_args(["serve", "--model", "m", "--no-prefix-cache", "--min-prefix-tokens", "64"])
    assert a.prefix_cache is False and a.min_prefix_tokens == 64


def test_serve_parity_script_compare(tok, tiny_model):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "scripts" / "serve_parity.py"
    spec = importlib.util.spec_from_file_location("serve_parity", path)
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    body = {"state": STATE, "questions": {"c": CHOICE, "n": NOUL, "s": SCORE, "w": WIDE}}
    rows, _, _ = sp.compare(_engine(tok, tiny_model, min_prefix_tokens=1),
                            _engine(tok, tiny_model, prefix_cache=False), body)
    assert [r["name"] for r in rows] == ["c", "n", "s", "w"]
    assert all(r["agree"] and r["max_diff"] < 1e-4 for r in rows)
