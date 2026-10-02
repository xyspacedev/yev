"""`yev gate`: the Stage 1 publish gate (Plan 5 Task 10, spec §2)."""
import json

import pytest

from yev import cli, gate

WORKFLOWS = {"invoice_processing": "exact_actions", "customer_service": "exact_actions",
             "agent_trace_observability": "primary_action", "security_incidents": "exact_actions"}
# Stage 0 (lc100) numbers, in the shapes the real files have
STAGE0 = {"invoice_processing": 0.4666666666666667, "customer_service": 0.5555555555555556,
          "agent_trace_observability": 0.5180180180180181, "security_incidents": 0.5208333333333334}


def _w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _scores(eval_id, consensus: dict, openai: dict | None = None):
    """A WorkflowEvals scores.json: one series per label set; per-decision `scores` trimmed."""
    other = openai or {k: v - 0.05 for k, v in consensus.items()}
    return {"schema_version": "1.3", "eval_id": eval_id, "cohorts": {}, "scores": [],
            "series": [{"run_id": "yev-4b", "label_set_id": "labels-openai", "decisions": 10, "metrics": other},
                       {"run_id": "yev-4b", "label_set_id": "labels-anthropic", "decisions": 10, "metrics": other},
                       {"run_id": "yev-4b", "label_set_id": "labels-consensus", "decisions": 10,
                        "metrics": consensus}]}


def _fixture(tmp_path, wf=None, decidebench=0.945, jev=0.7835497835497836, dyna_f1=0.5571142284569139,
             rjudge_f1=0.5077186963979418, stage0_acc=0.940463645943098, ece=0.037323712508089545,
             variant="examples"):
    wf = STAGE0 if wf is None else wf
    root = tmp_path / "in"
    for name, value in wf.items():
        metrics = {WORKFLOWS[name]: value}
        if name == "invoice_processing":  # invoice also reports primary_action; the headline is exact_actions
            metrics["primary_action"] = 0.99
        _w(root / "workflowevals" / name / "scores.json", _scores(name, metrics))
    _w(root / "bench" / "decidebench_examples" / "metrics.json",
       {"n": 400, "accuracy": decidebench, "ece_15": 0.06, "pair_accuracy": 0.89, "variant": variant})
    _w(root / "bench" / "jevbench" / "metrics.json", {"n": 231, "accuracy": jev, "pair_accuracy": 0.97})
    _w(root / "bench" / "dynabench_test" / "metrics.json", {"n": 543, "accuracy": 0.59, "f1": dyna_f1})
    _w(root / "bench" / "rjudge" / "metrics.json", {"n": 571, "accuracy": 0.497, "f1": rjudge_f1})
    _w(root / "runs" / "stage0_dev.json", {"n": 1898, "accuracy": stage0_acc, "ece": 0.03, "brier": 0.08})
    _w(root / "runs" / "stage1_dev.json", {"n": 3000, "accuracy": 0.9, "ece": ece, "brier": 0.1})
    return {"stage0_dev": root / "runs" / "stage0_dev.json", "stage1_dev": root / "runs" / "stage1_dev.json",
            "workflowevals": root / "workflowevals", "bench": root / "bench",
            "decidebench": root / "bench" / "decidebench_examples"}


def _run(paths, out, drop=()):
    argv = ["gate", "--out", str(out)]
    for k, flag in (("stage0_dev", "--stage0-dev"), ("stage1_dev", "--stage1-dev"),
                    ("workflowevals", "--workflowevals"), ("bench", "--bench"), ("decidebench", "--decidebench")):
        if k not in drop:
            argv += [flag, str(paths[k])]
    code = cli.main(argv)
    return code, json.loads(out.read_text())


def _status(rep):
    return {r["check"]: r["status"] for r in rep["rows"]}


def test_gate_on_stage0_results(tmp_path, capsys):
    code, rep = _run(_fixture(tmp_path), tmp_path / "gate.json")
    st = _status(rep)
    assert st["workflowevals_overall"] == "FAIL" and st["decidebench_examples"] == "PASS"
    assert st["jevbench"] == "PASS" and st["dynabench_f1"] == "PASS" and st["rjudge_f1"] == "PASS"
    assert st["stage0_dev_accuracy"] == "PASS" and st["ece"] == "TARGET"
    assert rep["publishable"] is False and code == 1
    row = {r["check"]: r for r in rep["rows"]}
    assert round(row["workflowevals_overall"]["value"], 1) == 51.5
    assert row["decidebench_examples"]["value"] == pytest.approx(94.5)
    out = capsys.readouterr().out
    assert "PUBLISHABLE: no" in out and "workflowevals_overall" in out and "FAIL" in out


def test_gate_workflowevals_overall_equal_mean(tmp_path):
    wf = {"invoice_processing": 0.40, "customer_service": 0.60, "agent_trace_observability": 0.70,
          "security_incidents": 0.50}
    _, rep = _run(_fixture(tmp_path, wf=wf), tmp_path / "gate.json")
    row = next(r for r in rep["rows"] if r["check"] == "workflowevals_overall")
    assert row["value"] == pytest.approx(55.0)  # (40 + 60 + 70 + 50) / 4, not decision-weighted
    assert row["detail"]["invoice_processing"] == {"metric": "exact_actions", "value": pytest.approx(0.40)}
    assert row["detail"]["agent_trace_observability"]["metric"] == "primary_action"
    assert row["status"] == "PASS"


def _all_at(tmp_path, delta, ece_delta=0.0):
    wf = {k: 0.545 + delta for k in WORKFLOWS}
    return _fixture(tmp_path, wf=wf, decidebench=0.94 + delta, jev=0.77 + delta, dyna_f1=0.54 + delta,
                    rjudge_f1=0.49 + delta, stage0_acc=0.93 + delta, ece=0.05 + ece_delta)


def test_gate_thresholds_exact(tmp_path):
    code, rep = _run(_all_at(tmp_path / "a", 0.0), tmp_path / "a.json")
    assert set(_status(rep).values()) == {"PASS"} and rep["publishable"] is True and code == 0
    _, rep = _run(_all_at(tmp_path / "b", -0.0001, ece_delta=0.0001), tmp_path / "b.json")
    assert set(_status(rep).values()) == {"FAIL"} and rep["publishable"] is False
    # targets: WorkflowEvals 60, DecideBench 95, ECE 0.04 (inclusive); the others have no target
    _, rep = _run(_fixture(tmp_path / "c", wf={k: 0.60 for k in WORKFLOWS}, decidebench=0.95, jev=0.99,
                           ece=0.04), tmp_path / "c.json")
    st = _status(rep)
    assert st["workflowevals_overall"] == st["decidebench_examples"] == st["ece"] == "TARGET"
    assert st["jevbench"] == "PASS"
    _, rep = _run(_fixture(tmp_path / "d", wf={k: 0.5999 for k in WORKFLOWS}, decidebench=0.9499, ece=0.0401),
                  tmp_path / "d.json")
    st = _status(rep)
    assert st["workflowevals_overall"] == st["decidebench_examples"] == st["ece"] == "PASS"


def test_gate_thresholds_table():
    t = {c.name: (c.minimum, c.target) for c in gate.CHECKS}
    assert t == {"workflowevals_overall": (54.5, 60.0), "decidebench_examples": (94.0, 95.0),
                 "jevbench": (77.0, None), "dynabench_f1": (54.0, None), "rjudge_f1": (49.0, None),
                 "stage0_dev_accuracy": (93.0, None), "ece": (0.05, 0.04)}


@pytest.mark.parametrize("drop", ["stage0_dev", "stage1_dev", "workflowevals", "bench", "decidebench"])
def test_gate_missing_input_is_fail_not_pass(tmp_path, drop):
    paths = _fixture(tmp_path, wf={k: 0.65 for k in WORKFLOWS}, decidebench=0.96)
    code, rep = _run(paths, tmp_path / "all.json")
    assert rep["publishable"] is True and code == 0
    code, rep = _run(paths, tmp_path / "gate.json", drop=(drop,))
    assert rep["publishable"] is False and code == 1
    assert "FAIL" in _status(rep).values()
    assert all(r["status"] != "FAIL" or r["value"] is None for r in rep["rows"])


def test_gate_missing_pieces_fail(tmp_path):
    good = {k: 0.65 for k in WORKFLOWS}
    # one workflow missing: the mean of three is not reported as the overall
    paths = _fixture(tmp_path / "a", wf={k: v for k, v in good.items() if k != "security_incidents"})
    _, rep = _run(paths, tmp_path / "a.json")
    row = next(r for r in rep["rows"] if r["check"] == "workflowevals_overall")
    assert row["status"] == "FAIL" and row["value"] is None and "security_incidents" in row["note"]
    # one benchmark file missing
    paths = _fixture(tmp_path / "b", wf=good, decidebench=0.96)
    (paths["bench"] / "rjudge" / "metrics.json").unlink()
    _, rep = _run(paths, tmp_path / "b.json")
    assert _status(rep)["rjudge_f1"] == "FAIL" and rep["publishable"] is False
    # a key missing from a dev report, and a zero-shot DecideBench result passed by mistake
    paths = _fixture(tmp_path / "c", wf=good, decidebench=0.96, variant="zeroshot")
    paths["stage1_dev"].write_text(json.dumps({"n": 10, "accuracy": 0.9}))
    _, rep = _run(paths, tmp_path / "c.json")
    assert _status(rep)["ece"] == "FAIL" and _status(rep)["decidebench_examples"] == "FAIL"
    # no consensus series in a scores.json
    paths = _fixture(tmp_path / "d", wf=good, decidebench=0.96)
    f = paths["workflowevals"] / "customer_service" / "scores.json"
    s = json.loads(f.read_text())
    s["series"] = [x for x in s["series"] if x["label_set_id"] != "labels-consensus"]
    f.write_text(json.dumps(s))
    _, rep = _run(paths, tmp_path / "d.json")
    assert _status(rep)["workflowevals_overall"] == "FAIL"
