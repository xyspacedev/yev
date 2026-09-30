from jeff import cli
from jeff.schema import read_jsonl


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
            {"key": "no_way", "description": "Refused when over the limit."}]
    (run / "written").mkdir()
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
