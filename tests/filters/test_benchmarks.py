import importlib.util
import json

import pytest

from yev import cli
from yev.filters.benchmarks import (
    DECISION_INDEX_SHOWN_SOURCES,
    DECISION_INDEX_SOURCES,
    BenchmarkFingerprints,
    build_benchmark_fingerprints,
    chat_row_sources,
    decision_index_ids,
    flatten_strings,
    overlaps_any,
)
from yev.filters.contamination import build_fingerprints
from yev.filters.pipeline import ContaminationError, assert_clean, run_filters
from yev.schema import Decision, Option, read_jsonl, write_jsonl
from yev.sources.base import SourceSpec

B1 = "the invoice total does not match the purchase order amount by forty dollars"
B2 = "agent proposes to drop the production orders table during business hours without approval"
B3 = "customer asks for a refund on a jacket bought ninety days ago with receipt"

BFP = build_benchmark_fingerprints({"b1": [B1], "b2": [B2], "b3": [B3]})
EMPTY_FP = build_fingerprints([])


def spec_for(name="s", licence="mit") -> SourceSpec:
    return SourceSpec(name=name, hf_id=f"toy/{name}", config=None, split="train", licence=licence,
                      convert=lambda *a: [], pool_size=10)


SPECS = {"s": spec_for()}


def dec(i, state, cluster=None) -> Decision:
    return Decision(id=f"s:{i}", type="choice", state=state, question="Which applies?",
                    options=[Option("a", "Option text."), Option("b", "Other.")], gold="a",
                    family="f", source="s", licence="mit", split="pool", cluster_id=cluster)


def test_drops_rows_overlapping_any_benchmark():
    # 8 consecutive tokens of B2 ("to drop the production orders table during business") inside new text
    bad = dec(1, "Ticket 7: the agent proposes to drop the production orders table during business; deny?")
    ok = dec(2, "A clean, unrelated ticket about a printer.")
    assert overlaps_any(bad, BFP) == "b2"
    assert overlaps_any(ok, BFP) is None
    kept, report = run_filters([bad, ok], EMPTY_FP, embed=None, specs=SPECS, bench_fp=BFP)
    assert [d.id for d in kept] == ["s:2"]
    r = report["s"]
    assert r["benchmark:b2"] == 1 and r["benchmark:b1"] == 0 and r["benchmark:b3"] == 0
    assert r["in"] == 2 and r["out"] == 1


def test_short_shared_phrases_do_not_trigger():
    seven = " ".join(B2.split()[:7])  # "agent proposes to drop the production orders"
    row = dec(1, f"Note: {seven}. Unrelated tail here.")
    assert overlaps_any(row, BFP) is None
    kept, _ = run_filters([row], EMPTY_FP, embed=None, specs=SPECS, bench_fp=BFP)
    assert [d.id for d in kept] == ["s:1"]


def test_first_overlapping_benchmark_wins_in_source_order():
    both = build_benchmark_fingerprints({"x": [B1], "y": [B3]})
    assert overlaps_any(dec(1, f"{B3}. {B1}"), both) == "x"


def test_benchmark_drop_takes_whole_cluster_and_counts_balance():
    bad = dec(1, B3, cluster="c1")
    sib = dec(2, "A sibling with clean text.", cluster="c1")
    other = dec(3, "Independent clean row.")
    kept, report = run_filters([bad, sib, other], EMPTY_FP, embed=None, specs=SPECS, bench_fp=BFP)
    assert [d.id for d in kept] == ["s:3"]
    r = report["s"]
    assert r["benchmark:b3"] == 2
    assert r["in"] == sum(v for k, v in r.items() if k != "in")


def test_without_bench_fp_report_keys_unchanged():
    _, report = run_filters([dec(1, B2)], EMPTY_FP, embed=None, specs=SPECS)
    assert not any(k.startswith("benchmark:") for k in report["s"])


def test_assert_clean_checks_benchmarks():
    assert_clean([dec(1, "clean")], EMPTY_FP, BFP)
    with pytest.raises(ContaminationError, match="b1"):
        assert_clean([dec(2, B1)], EMPTY_FP, BFP)


def test_fingerprints_roundtrip(tmp_path):
    path = tmp_path / "fp.json"
    BFP.save(path)
    data = json.loads(path.read_text())
    assert data["ngram"] == 8
    assert data["meta"]["item_min_shared"] == 3
    for name in ("b1", "b2", "b3"):
        hashes = data["benchmarks"][name]["hashes"]
        assert hashes == sorted(hashes) and all(isinstance(h, int) for h in hashes)
        assert len(data["benchmarks"][name]["owners"]) == len(hashes)
        assert data["benchmarks"][name]["items"] == 1
    loaded = BenchmarkFingerprints.load(path)
    assert loaded.names() == ["b1", "b2", "b3"]
    assert loaded.grams == BFP.grams and loaded.items == BFP.items and loaded.thresholds == BFP.thresholds
    assert overlaps_any(dec(1, B2), loaded) == "b2"


def test_load_rejects_other_ngram_size(tmp_path):
    path = tmp_path / "fp.json"
    path.write_text(json.dumps({"ngram": 7, "hash": "blake2b-8", "benchmarks": {}}))
    with pytest.raises(ValueError):
        BenchmarkFingerprints.load(path)


def test_chat_row_sources_reads_item_states_once_per_id(tmp_path):
    def row(i, state, example_state=None):
        msgs = [{"role": "system", "content": "sys"}]
        if example_state:
            msgs += [{"role": "user", "content": json.dumps({"state": example_state, "question": "q"})},
                     {"role": "assistant", "content": "A"}]
        msgs.append({"role": "user", "content": json.dumps({"state": state, "question": "q"})})
        return json.dumps({"id": i, "messages": msgs}) + "\n"

    (tmp_path / "bench_a").mkdir()
    (tmp_path / "bench_a" / "rows_zeroshot.chat.jsonl").write_text(row("a:1", B1) + row("a:2", B2))
    (tmp_path / "bench_a" / "rows_examples.chat.jsonl").write_text(row("a:1", B1, example_state="worked example"))
    (tmp_path / "bench_b").mkdir()
    (tmp_path / "bench_b" / "rows.chat.jsonl").write_text(row("b:1", B3))
    (tmp_path / "empty").mkdir()
    src = chat_row_sources(tmp_path)
    assert sorted(src) == ["bench_a", "bench_b"]
    assert sorted(src["bench_a"]) == sorted([B1, B2]) and src["bench_b"] == [B3]


def test_flatten_strings_takes_string_values_not_keys():
    obj = {"case_id": "ap_1", "invoice": {"vendor_name": "Haymaker Media", "total": 12.5,
                                          "lines": [{"desc": "Freight"}, "note"]}, "ok": True}
    assert flatten_strings(obj) == ["ap_1", "Haymaker Media", "Freight", "note"]


def test_decision_index_lists_38_benchmarks():
    assert len(DECISION_INDEX_SOURCES) == 38
    assert all(ids for ids in DECISION_INDEX_SOURCES.values())
    assert {"BANKING77", "CLINC150+OOS", "GSM8K", "FinEntity"} <= set(DECISION_INDEX_SOURCES)
    assert {"ARC-Easy", "ARC-Challenge", "MMLU"} <= set(DECISION_INDEX_SHOWN_SOURCES)
    ids = decision_index_ids()
    assert "legacy-datasets/banking77" in ids and "clinc/clinc_oos" in ids and "openai/gsm8k" in ids


def test_public_long_sources_not_in_decision_index():
    if importlib.util.find_spec("yev.sources.public_long") is None:
        pytest.skip("yev.sources.public_long arrives in Task 8")
    from yev.sources.public_long import SOURCES

    ids = decision_index_ids()
    tails = {i.rsplit("/", 1)[-1] for i in ids}
    for s in SOURCES:
        sid = (getattr(s, "hf_id", None) or str(s)).lower()
        assert sid not in ids, sid
        assert sid.rsplit("/", 1)[-1] not in tails, sid


def test_filter_command_with_benchmarks(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    good = Decision(id="toy:1", type="noul", state="clean state", question="q?",
                    options=[Option("yes", "y"), Option("no", "n")], gold="yes",
                    family="f", source="toy", licence="mit")
    bad = Decision(id="toy:2", type="noul", state=f"Prefix. {B1}.", question="q?",
                   options=[Option("yes", "y"), Option("no", "n")], gold="yes",
                   family="f", source="toy", licence="mit")
    write_jsonl(raw / "toy.jsonl", [good, bad])
    fp_path = tmp_path / "fingerprints.json"
    BFP.save(fp_path)
    monkeypatch.setattr(cli, "load_fingerprints", lambda: EMPTY_FP)
    monkeypatch.setattr(cli, "SOURCES", [SourceSpec(name="toy", hf_id="toy/toy", config=None, split="train",
                                                   licence="mit", convert=lambda *a: [], pool_size=10)])
    assert cli.main(["filter", "--in", str(raw), "--out", str(out), "--no-embed",
                     "--benchmarks", str(fp_path)]) == 0
    assert [d.id for d in read_jsonl(out / "toy.jsonl")] == ["toy:1"]
    report = json.loads((out / "filter_report.json").read_text())
    assert report["sources"]["toy"]["benchmark:b1"] == 1
    assert report["meta"]["benchmarks"]["path"] == str(fp_path)
    assert report["meta"]["benchmarks"]["names"] == ["b1", "b2", "b3"]


def test_bench_fingerprint_command_offline(tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    (bench / "toy").mkdir(parents=True)
    msgs = [{"role": "user", "content": json.dumps({"state": B1})}]
    (bench / "toy" / "rows.chat.jsonl").write_text(json.dumps({"id": "t:1", "messages": msgs}) + "\n")
    monkeypatch.setattr("yev.filters.benchmarks.hub_sources", lambda: {"hubby": [B3]})
    out = tmp_path / "fp.json"
    assert cli.main(["bench", "fingerprint", "--out", str(out), "--bench-dir", str(bench)]) == 0
    fp = BenchmarkFingerprints.load(out)
    assert fp.names() == ["toy", "hubby"] and fp.items == {"toy": 1, "hubby": 1}
    assert overlaps_any(dec(1, B3), fp) == "hubby"


# --- Fix round 1: boilerplate stoplist and short items ------------------------------------------

STOCK = "is there anything else i can assist you with today"


def test_phrase_shared_by_two_benchmarks_does_not_trigger():
    fp = build_benchmark_fingerprints({"x": [f"Ticket about a lost parcel in Leeds. {STOCK}"],
                                       "y": [f"Trace of a deploy rollback in prod. {STOCK}"]})
    assert overlaps_any(dec(1, f"Our own chat. {STOCK}"), fp) is None
    assert fp.meta["stoplist"]["dropped"]["x"] > 0 and fp.meta["stoplist"]["dropped"]["y"] > 0


def test_phrase_repeated_in_20_items_of_one_benchmark_does_not_trigger():
    items = [f"Case {i} about item number {i * 7}. {STOCK}" for i in range(20)]
    fp = build_benchmark_fingerprints({"x": items})
    assert overlaps_any(dec(1, f"Our own chat. {STOCK}"), fp) is None
    # 19 repeats are below the threshold and still count
    fp19 = build_benchmark_fingerprints({"x": items[:19]})
    assert overlaps_any(dec(1, f"Our own chat. {STOCK}"), fp19) == "x"


def test_item_specific_ngram_still_triggers_next_to_stoplisted_ones():
    fp = build_benchmark_fingerprints({"x": [f"{B1}. {STOCK}"], "y": [f"{B2}. {STOCK}"]})
    assert overlaps_any(dec(1, f"Copied: {B1}."), fp) == "x"
    assert overlaps_any(dec(2, f"Copied: {B2}."), fp) == "y"


def test_short_item_exact_state_match_triggers():
    fp = build_benchmark_fingerprints({"intents": ["How do I locate my card?", "Balance?"]})
    assert overlaps_any(dec(1, "how do I locate my CARD"), fp) == "intents"
    assert overlaps_any(dec(2, "balance"), fp) == "intents"  # exact match works below 4 tokens
    assert overlaps_any(dec(3, "How do I locate my parcel?"), fp) is None


def test_short_item_contained_as_token_run_only_from_6_tokens():
    fp = build_benchmark_fingerprints({"intents": ["How do I locate my card?", "check my balance"]})
    assert overlaps_any(dec(1, "Customer wrote: how do I locate my card? Please help."), fp) == "intents"
    # 3-token item is not matched inside longer text
    assert overlaps_any(dec(2, "Customer wants to check my balance today."), fp) is None
    # whole tokens only: "cards" is not "card"
    assert overlaps_any(dec(3, "Customer wrote: how do I locate my cards?"), fp) is None


def test_short_items_roundtrip(tmp_path):
    fp = build_benchmark_fingerprints({"intents": ["How do I locate my card?"], "x": [B1]})
    fp.save(tmp_path / "fp.json")
    loaded = BenchmarkFingerprints.load(tmp_path / "fp.json")
    assert loaded.short == fp.short
    assert overlaps_any(dec(1, "so: how do i locate my card"), loaded) == "intents"


# --- Fix round 2: item-level matching -------------------------------------------------------------

STOCK9 = "please let me know if you need anything else"  # 9 tokens: two 8-grams


def test_one_stock_phrase_shared_with_an_item_does_not_drop():
    fp = build_benchmark_fingerprints({"x": [f"{B1}. {STOCK9}"]})
    assert overlaps_any(dec(1, f"Our own long document about pumps. {STOCK9}. More text."), fp) is None


def test_three_distinct_ngrams_from_one_item_drop():
    fp = build_benchmark_fingerprints({"x": [B1]})
    toks = B1.split()
    assert overlaps_any(dec(1, "Intro. " + " ".join(toks[:10]) + " outro."), fp) == "x"  # 3 grams
    assert overlaps_any(dec(2, "Intro. " + " ".join(toks[:9]) + " outro."), fp) is None  # 2 grams


def test_three_ngrams_spread_over_three_items_do_not_drop():
    fp = build_benchmark_fingerprints({"x": [B1, B2, B3]})
    row = " zz ".join(" ".join(t.split()[:8]) for t in (B1, B2, B3))
    assert overlaps_any(dec(1, row), fp) is None


def test_single_decidebench_ngram_still_drops():
    fp = build_benchmark_fingerprints({"decidebench": [B1], "x": [B2]})
    assert fp.thresholds == {"decidebench": 1, "x": 3}
    assert overlaps_any(dec(1, "Intro. " + " ".join(B1.split()[:8]) + " outro."), fp) == "decidebench"


def test_five_token_short_item_matches_exactly_but_not_contained():
    fp = build_benchmark_fingerprints({"intents": ["Where is my new card?"]})
    assert overlaps_any(dec(1, "Customer: where is my new card please"), fp) is None
    assert overlaps_any(dec(2, "where is my new CARD"), fp) == "intents"


# --- Fix round 3: per-item thresholds, exact short items up to 9 tokens, strict benchmarks -------

def test_nine_token_item_copied_verbatim_inside_row_drops():
    fp = build_benchmark_fingerprints({"intents": ["Please send me a new card to my home."]})  # 9 tokens, 2 grams
    assert overlaps_any(dec(1, "Hi, please send me a new card to my home address. Thanks"), fp) == "intents"
    assert overlaps_any(dec(2, "please send me a new card to my home"), fp) == "intents"
    assert overlaps_any(dec(3, "Hi, please send me a new card to my office."), fp) is None


def test_eight_and_nine_token_items_exact_match_hashes_stored():
    fp = build_benchmark_fingerprints({"intents": ["one two three four five six seven eight",
                                                   "one two three four five six seven eight nine"]})
    assert set(fp.short["intents"]) == {8, 9}


def test_mostly_stoplisted_item_caught_by_remaining_grams():
    # STOCK's 3 grams are shared with y and stoplisted; x keeps only the gram ending in "purple"
    fp = build_benchmark_fingerprints({"x": [f"{STOCK} purple"], "y": [STOCK]})
    assert len(fp.grams["x"]) == 1
    assert overlaps_any(dec(1, f"Note: anything else i can assist you with today purple, ok"), fp) == "x"
    assert overlaps_any(dec(2, f"Note: {STOCK}, ok"), fp) is None


def test_strict_benchmark_exempt_from_stoplist():
    eight = " ".join(B1.split()[:8])
    fp = build_benchmark_fingerprints({"decidebench": [B1], "x": [B1]})
    assert fp.meta["stoplist"]["dropped"]["decidebench"] == 0
    assert overlaps_any(dec(1, f"Intro. {eight} outro."), fp) == "decidebench"
