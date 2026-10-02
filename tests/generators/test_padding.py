import pytest

from yev.generators.padding import FORWARDED_OPEN, PaddingError, pad
from yev.schema import Decision, Option


def count(t):
    return len(t.split())


def row(i="r1", split="pool", cluster="c1"):
    return Decision(id=i, type="choice", state="Order 77 arrived broken.\n\nCustomer wants a refund.",
                    question="What now?", options=[Option("a", "refund"), Option("b", "deny")], gold="a",
                    family="returns_v2", source="synthetic", licence="synthetic-cc0", split=split,
                    soft_gold={"a": 0.8, "b": 0.2}, cluster_id=cluster, weight=0.5)


DOCS = ["\n\n".join(f"Doc{d} paragraph {p} " + "word " * 30 for p in range(40)) for d in range(3)]


def run(rows, **kw):
    return pad(rows, DOCS, (300, 600), seed=1, count=count, **kw)


def test_original_state_preserved_verbatim():
    r = row()
    (p,) = run([r])
    assert r.state in p.state
    assert p.state.index(FORWARDED_OPEN) < p.state.index(r.state)
    i = p.state.index(r.state)
    assert p.state[:i].endswith("\n") and p.state[i + len(r.state):].startswith("\n")


def test_labels_unchanged():
    r = row()
    (p,) = run([r])
    for f in ("type", "question", "options", "gold", "soft_gold", "weight", "source", "licence", "split"):
        assert getattr(p, f) == getattr(r, f)
    assert p.family == "long_padding" and p.cluster_id == "pad:c1"
    assert p.metadata["orig_id"] == "r1" and p.metadata["distractor_ids"]
    p.validate()


def test_cluster_falls_back_to_id():
    (p,) = run([row(cluster=None)])
    assert p.cluster_id == "pad:r1"


def test_length_within_target():
    out = run([row(f"r{i}") for i in range(20)])
    assert len(out) == 20
    for p in out:
        assert 300 <= count(p.state) <= 600


def test_deterministic():
    assert run([row()]) == run([row()])


@pytest.mark.parametrize("split", ["dev", "calibration"])
def test_never_pads_eval_rows(split):
    with pytest.raises(PaddingError):
        run([row(split=split)])


def test_same_title_distractor_skipped():
    r = row()
    docs = [r.state + "\n\n" + "x " * 400, "Other title\n\n" + "\n\n".join("y " * 30 for _ in range(40))]
    (p,) = pad([r], docs, (200, 400), seed=0, count=count)
    assert "x x" not in p.state


def test_cli_pad_roundtrip(tmp_path):
    from yev import cli
    from yev.schema import read_jsonl, write_jsonl
    write_jsonl(tmp_path / "rows.jsonl", [row()])
    doc = Decision(id="d", type="choice", state=DOCS[0], question="q?", options=[Option("a", "x"), Option("b", "y")],
                   gold="a", family="invoice_ledger", source="synthetic", licence="synthetic-cc0")
    write_jsonl(tmp_path / "docs.jsonl", [doc])
    assert cli.main(["pad", "--in", str(tmp_path / "rows.jsonl"), "--distractors", str(tmp_path / "docs.jsonl"),
                     "--out", str(tmp_path / "o.jsonl"), "--target-tokens", "2000:3000", "--seed", "1"]) == 0
    (p,) = read_jsonl(tmp_path / "o.jsonl")
    assert p.family == "long_padding"
    assert cli.main(["pad", "--in", str(tmp_path / "docs.jsonl"), "--distractors", str(tmp_path / "docs.jsonl"),
                     "--out", str(tmp_path / "o2.jsonl")]) == 2
