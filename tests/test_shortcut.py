import json
import random

from jeff import cli
from jeff.filters.shortcut import apply_shortcut, lexical_scores, masked_input, run_shortcut, shortcut_scores
from jeff.schema import Decision, Option, read_jsonl, write_jsonl

OPTS = [Option("approve", "Approve the request."), Option("deny", "Deny the request.")]


def row(i, gold, state, source="toy_public", cluster=None, family="f"):
    return Decision(id=f"{source}:{i}", type="choice", state=state, question="What now?", options=list(OPTS),
                    gold=gold, family=family, source=source, licence="mit", cluster_id=cluster)


def keyword_rows(n=200, source="toy_public"):
    rng = random.Random(0)
    out = []
    for i in range(n):
        gold = rng.choice(["approve", "deny"])
        word = "sunshine" if gold == "approve" else "thunder"
        out.append(row(i, gold, f"Ticket {i}: the weather today is {word} and the queue is long.", source))
    return out


def test_masked_input_hides_numbers():
    text = masked_input(row(0, "approve", "Refund of $480 on day 31."))
    assert "480" not in text and "31" not in text and "NUM" in text


def test_lexical_detector_finds_topic_word_matching():
    opts = [Option("billing", "Questions about billing and invoices."), Option("shipping", "Questions about shipping and parcels.")]
    ds = [Decision(id=f"l:{i}", type="choice", state=("My invoice billing is wrong." if i % 2 else "My parcel shipping is late."),
                   question="Which team?", options=list(opts), gold=("billing" if i % 2 else "shipping"), family="f",
                   source="lex", licence="mit") for i in range(60)]
    assert all(ok for _, ok in lexical_scores(ds))


def test_keyword_shortcut_is_found_and_public_rows_downweighted():
    ds = keyword_rows()
    scores = shortcut_scores(ds)
    assert sum(ok for _, ok in scores) / len(ds) > 0.9
    kept, report = apply_shortcut(ds, scores)
    assert len(kept) == len(ds)
    assert report["toy_public"]["downweighted"] > 0.8 * len(ds)
    assert all(d.weight in (1.0, 0.3) for d in kept)


def test_random_labels_are_not_downweighted_much():
    rng = random.Random(1)
    ds = [row(i, rng.choice(["approve", "deny"]), f"Case {i} " + " ".join(rng.choice("abcdefgh") * 3 for _ in range(8)))
          for i in range(200)]
    kept, report = run_shortcut(ds)
    assert report["toy_public"]["downweighted"] < 0.2 * len(ds)


def test_synthetic_clusters_fully_solved_are_dropped():
    ds = []
    for c in range(100):
        ds.append(row(2 * c, "approve", f"Case {c}: sunshine outside.", "synthetic_rules", f"k{c}"))
        ds.append(row(2 * c + 1, "deny", f"Case {c}: thunder outside.", "synthetic_rules", f"k{c}"))
    kept, report = run_shortcut(ds)
    assert report["synthetic_rules"]["dropped"] > 150 and len(kept) < 50


def test_tiny_or_single_class_groups_pass_through():
    tiny = keyword_rows(20, source="tiny_src")
    one_class = [row(i, "approve", f"Only approvals {i}.", "mono_src") for i in range(80)]
    kept, report = run_shortcut(tiny + one_class)
    assert len(kept) == 100
    assert report["tiny_src"]["downweighted"] == 0 and report["mono_src"]["downweighted"] == 0


def test_cli_shortcut_writes_files_and_report(tmp_path):
    src, out = tmp_path / "in", tmp_path / "out"
    write_jsonl(src / "toy_public.jsonl", keyword_rows(120))
    assert cli.main(["shortcut", "--in", str(src), "--out", str(out)]) == 0
    assert len(list(read_jsonl(out / "toy_public.jsonl"))) == 120
    assert json.loads((out / "shortcut_report.json").read_text())["toy_public"]["in"] == 120
    assert cli.main(["shortcut", "--in", str(src), "--out", str(src)]) == 2
    assert cli.main(["shortcut", "--in", str(tmp_path / "empty"), "--out", str(out)]) == 2
