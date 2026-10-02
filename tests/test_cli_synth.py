from yev import cli
from yev.schema import read_jsonl


def test_gen_rules_writes_valid_clusters(tmp_path):
    out = tmp_path / "rules" / "returns.jsonl"
    assert cli.main(["gen-rules", "--family", "returns", "--clusters", "5", "--seed", "7", "--out", str(out)]) == 0
    rows = list(read_jsonl(out))
    assert len({r.cluster_id for r in rows}) == 5
    assert {r.source for r in rows} == {"synthetic_rules"}


def test_gen_rules_rejects_unknown_family(tmp_path, capsys):
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["gen-rules", "--family", "nope", "--clusters", "1", "--out", str(tmp_path / "x.jsonl")])


import json


def test_synth_end_to_end_with_simulated_subagents(tmp_path):
    run = tmp_path / "run"
    assert cli.main(["synth", "plan", "--dir", str(run), "--clusters", "2", "--per-batch", "1", "--seed", "0"]) == 0
    specs = sorted((run / "batches").glob("*.json"))
    assert len(specs) == 2 and all(p.with_suffix(".prompt.md").exists() for p in specs)

    # simulated writer: one valid cluster per batch
    opts = [{"key": "yes_ok", "description": "Allowed when under the limit."},
            {"key": "no_way", "description": "Refused when over the limit."},
            {"key": "ask_boss", "description": "Sent to a manager when the limit is unclear."}]
    (run / "written").mkdir(exist_ok=True)
    for p in specs:
        b = json.loads(p.read_text())
        line = {"question": "What happens?", "options": opts,
                "base": {"state": "The request is for 40 units, under the 50 unit limit.", "gold": "yes_ok"},
                "variants": [{"state": "The request is for 60 units, under the 50 unit limit.", "gold": "no_way",
                              "edit_type": "threshold", "edit": "40->60"}],
                "policy_variants": []}
        (run / "written" / f"{b['batch_id']}.jsonl").write_text(json.dumps(line) + "\n")
    assert cli.main(["synth", "ingest", "--dir", str(run)]) == 0
    assert json.loads((run / "reports" / "ingest.json").read_text())["clusters_kept"] == 2

    assert cli.main(["synth", "check-prepare", "--dir", str(run)]) == 0
    key = json.loads((run / "private" / "check-key.json").read_text())
    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in (run / "ingested.jsonl").read_text().splitlines()}
    for sheet in sorted((run / "check").glob("sheet-*.jsonl")):
        answers = []
        for line in sheet.read_text().splitlines():
            it = json.loads(line)
            k = key[it["item_id"]]
            letter = next(l for l, ok in k["letters"].items() if ok == gold[k["decision"]])
            answers.append(json.dumps({"item_id": it["item_id"], "letter": letter}))
        (run / "check" / f"answers-{sheet.stem}.jsonl").write_text("\n".join(answers) + "\n")
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 0

    assert cli.main(["synth", "attrib-prepare", "--dir", str(run)]) == 0
    akey = json.loads((run / "private" / "attrib-key.json").read_text())
    for part in sorted((run / "attrib").glob("pairs-part-*.jsonl")):
        lines = [json.dumps({"pair_id": json.loads(l)["pair_id"], "edit_type": akey[json.loads(l)["pair_id"]]["edit_type"]})
                 for l in part.read_text().splitlines()]
        (run / "attrib" / f"answers-{part.stem}.jsonl").write_text("\n".join(lines) + "\n")
    assert cli.main(["synth", "attrib-score", "--dir", str(run)]) == 0

    final = list(read_jsonl(run / "final" / "synthetic_opus.jsonl"))
    assert len(final) == 4 and {d.source for d in final} == {"synthetic_opus"}
    pilot = json.loads((run / "reports" / "pilot.json").read_text())
    assert pilot["planned_clusters"] == 2 and pilot["final_clusters"] == 2
    assert pilot["final_rows_by_edit_type"] == {"base": 2, "threshold": 2}


def _pipeline_to_check(run, answer_all=True):
    cli.main(["synth", "plan", "--dir", str(run), "--clusters", "2", "--per-batch", "1"])
    opts = [{"key": "yes_ok", "description": "Allowed when under the limit."},
            {"key": "no_way", "description": "Refused when over the limit."},
            {"key": "ask_boss", "description": "Sent to a manager when the limit is unclear."}]
    for p in sorted((run / "batches").glob("*.json")):
        b = json.loads(p.read_text())
        line = {"question": "What happens?", "options": opts,
                "base": {"state": "The request is for 40 units, under the 50 unit limit.", "gold": "yes_ok"},
                "variants": [{"state": "The request is for 60 units, under the 50 unit limit.", "gold": "no_way",
                              "edit_type": "threshold", "edit": "40->60"}],
                "policy_variants": []}
        (run / "written" / f"{b['batch_id']}.jsonl").write_text(json.dumps(line) + "\n")
    cli.main(["synth", "ingest", "--dir", str(run)])
    cli.main(["synth", "check-prepare", "--dir", str(run)])


def _answer_sheets(run, limit=None):
    key = json.loads((run / "private" / "check-key.json").read_text())
    gold = {json.loads(l)["id"]: json.loads(l)["gold"] for l in (run / "ingested.jsonl").read_text().splitlines()}
    n = 0
    for sheet in sorted((run / "check").glob("sheet-*.jsonl")):
        out = []
        for line in sheet.read_text().splitlines():
            if limit is not None and n >= limit:
                break
            it = json.loads(line)
            k = key[it["item_id"]]
            letter = next(l for l, ok in k["letters"].items() if ok == gold[k["decision"]])
            out.append(json.dumps({"item_id": it["item_id"], "letter": letter}))
            n += 1
        (run / "check" / f"answers-{sheet.stem}.jsonl").write_text("\n".join(out) + "\n")


def test_check_prepare_rerun_removes_old_answers(tmp_path):
    run = tmp_path / "run"
    _pipeline_to_check(run)
    _answer_sheets(run)
    assert list((run / "check").glob("answers-*.jsonl"))
    assert cli.main(["synth", "check-prepare", "--dir", str(run)]) == 0
    assert not list((run / "check").glob("answers-*.jsonl"))


def test_check_score_without_answers_fails(tmp_path):
    run = tmp_path / "run"
    _pipeline_to_check(run)
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 2
    assert not (run / "checked.jsonl").exists()


def test_check_score_partial_answers_recorded(tmp_path):
    run = tmp_path / "run"
    _pipeline_to_check(run)
    _answer_sheets(run, limit=3)
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 0
    rep = json.loads((run / "reports" / "check.json").read_text())
    assert rep["answered"] == 3 and rep["answered"] < rep["expected"]


def test_ingest_rerun_removes_downstream(tmp_path):
    run = tmp_path / "run"
    _pipeline_to_check(run)
    _answer_sheets(run)
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 0
    (run / "final").mkdir()
    (run / "final" / "x.jsonl").write_text("")
    assert (run / "checked.jsonl").exists()
    assert cli.main(["synth", "ingest", "--dir", str(run)]) == 0
    assert not (run / "checked.jsonl").exists() and not (run / "final").exists()
    assert not (run / "check").exists() and not (run / "private").exists()


def _through_attrib_prepare(run):
    _pipeline_to_check(run)
    _answer_sheets(run)
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 0
    assert cli.main(["synth", "attrib-prepare", "--dir", str(run)]) == 0


def _answer_pairs(run, junk=False):
    akey = json.loads((run / "private" / "attrib-key.json").read_text())
    for part in sorted((run / "attrib").glob("pairs-part-*.jsonl")):
        lines = [json.dumps({"pair_id": json.loads(l)["pair_id"],
                             "edit_type": "nonsense" if junk else akey[json.loads(l)["pair_id"]]["edit_type"]})
                 for l in part.read_text().splitlines()]
        (run / "attrib" / f"answers-{part.stem}.jsonl").write_text("\n".join(lines) + "\n")


def test_check_score_clears_attribution_stage(tmp_path):
    run = tmp_path / "run"
    _through_attrib_prepare(run)
    _answer_pairs(run)
    assert cli.main(["synth", "attrib-score", "--dir", str(run)]) == 0
    assert (run / "final").exists() and (run / "reports" / "pilot.json").exists()
    assert cli.main(["synth", "check-score", "--dir", str(run)]) == 0
    assert not (run / "attrib").exists() and not (run / "private" / "attrib-key.json").exists()
    assert not (run / "final").exists() and not (run / "reports" / "attrib.json").exists()
    assert not (run / "reports" / "pilot.json").exists()
    assert (run / "private" / "check-key.json").exists()
    assert cli.main(["synth", "attrib-score", "--dir", str(run)]) == 2
    assert cli.main(["synth", "attrib-prepare", "--dir", str(run)]) == 0


def test_attrib_score_fails_when_key_misses_checked_rows(tmp_path):
    run = tmp_path / "run"
    _through_attrib_prepare(run)
    _answer_pairs(run)
    akey_path = run / "private" / "attrib-key.json"
    akey = json.loads(akey_path.read_text())
    victim = next(k for k in akey if k != "_no_base")
    del akey[victim]
    akey_path.write_text(json.dumps(akey))
    assert cli.main(["synth", "attrib-score", "--dir", str(run)]) == 2
    assert not (run / "final").exists()


def test_plan_clears_downstream_but_keeps_written(tmp_path):
    run = tmp_path / "run"
    _through_attrib_prepare(run)
    assert list((run / "written").glob("*.jsonl"))
    assert cli.main(["synth", "plan", "--dir", str(run), "--clusters", "2", "--per-batch", "1"]) == 0
    for rel in ("ingested.jsonl", "checked.jsonl", "check", "attrib", "private", "final", "reports"):
        assert not (run / rel).exists(), rel
    assert list((run / "written").glob("*.jsonl"))


def test_coverage_counts_only_valid_answers(tmp_path):
    key = {"i1": {"decision": "d1", "letters": {"A": "x", "B": "y"}},
           "i2": {"decision": "d2", "letters": {"A": "x", "B": "y"}}}
    answers = [{"item_id": "i1", "letter": "B"}, {"item_id": "i2", "letter": "F"}]
    assert cli._coverage(key, answers, "item_id") == (1, 2)
    akey = {"p1": {"decision": "d", "edit_type": "threshold"}, "p2": {"decision": "e", "edit_type": "date"}, "_no_base": []}
    ans = [{"pair_id": "p1", "edit_type": " Threshold "}, {"pair_id": "p2", "edit_type": "banana"}]
    assert cli._coverage(akey, ans, "pair_id") == (1, 2)


def test_pilot_report_has_survival_breakdowns(tmp_path):
    run = tmp_path / "run"
    _through_attrib_prepare(run)
    _answer_pairs(run)
    assert cli.main(["synth", "attrib-score", "--dir", str(run)]) == 0
    pilot = json.loads((run / "reports" / "pilot.json").read_text())
    assert pilot["checked_clusters"] == 2
    fam = pilot["survival_by_family"]
    assert sum(v["planned"] for v in fam.values()) == 2
    assert sum(v["ingested_clusters"] for v in fam.values()) == 2
    assert sum(v["checked_clusters"] for v in fam.values()) == 2
    assert sum(v["final_clusters"] for v in fam.values()) == 2
    assert pilot["survival_by_edit_type"] == {"base": {"ingested": 2, "checked": 2, "final": 2},
                                              "threshold": {"ingested": 2, "checked": 2, "final": 2}}
