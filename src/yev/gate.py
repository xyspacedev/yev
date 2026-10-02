"""Stage 1 publish gate (stage 1 spec §2): PASS, TARGET or FAIL per check, and PUBLISHABLE yes/no.

A missing or unreadable input is FAIL, never PASS. Percent-style checks are compared in percent
(fractions in the result files are multiplied by 100); ECE is compared as a fraction, lower is better.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

# WorkflowEvals headline metric per workflow, consensus labels; overall = equal mean of the four
WORKFLOW_HEADLINES = {
    "invoice_processing": "exact_actions",
    "customer_service": "exact_actions",
    "agent_trace_observability": "primary_action",
    "security_incidents": "exact_actions",
}
CONSENSUS = "labels-consensus"


@dataclass(frozen=True)
class Check:
    name: str
    label: str
    minimum: float
    target: float | None
    higher_is_better: bool = True


CHECKS = (
    Check("workflowevals_overall", "WorkflowEvals overall (consensus, equal mean)", 54.5, 60.0),
    Check("decidebench_examples", "DecideBench test, with examples", 94.0, 95.0),
    Check("jevbench", "JevBench accuracy", 77.0, None),
    Check("dynabench_f1", "DynaBench test F1", 54.0, None),
    Check("rjudge_f1", "R-Judge F1", 49.0, None),
    Check("stage0_dev_accuracy", "Stage 0 dev accuracy", 93.0, None),
    Check("ece", "ECE-15 (all dev)", 0.05, 0.04, higher_is_better=False),
)


class Missing(Exception):
    pass


def _read(path: Path | None, what: str) -> dict:
    if path is None:
        raise Missing(f"{what}: not given")
    path = Path(path)
    if not path.is_file():
        raise Missing(f"{what}: {path} not found")
    try:
        return json.loads(path.read_text())
    except ValueError as e:
        raise Missing(f"{what}: {path} is not JSON ({e})") from None


def _num(obj: dict, key: str, where: str) -> float:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise Missing(f"{where}: no numeric {key!r}")
    return float(v)


def status(check: Check, value: float | None) -> str:
    if value is None:
        return "FAIL"
    v = round(value, 9)  # 0.545 * 100 must compare equal to 54.5
    better = (lambda a, b: a >= b) if check.higher_is_better else (lambda a, b: a <= b)
    if not better(v, check.minimum):
        return "FAIL"
    return "TARGET" if check.target is not None and better(v, check.target) else "PASS"


def workflowevals_overall(d: Path | None) -> tuple[float, dict]:
    """Equal mean (percent) of the four workflows' consensus headline metrics, and the per-workflow values."""
    if d is None:
        raise Missing("WorkflowEvals: --workflowevals not given")
    detail, missing = {}, []
    for wf, metric in WORKFLOW_HEADLINES.items():
        path = Path(d) / wf / "scores.json"
        try:
            scores = _read(path, f"WorkflowEvals {wf}")
            series = [s for s in scores.get("series", []) if s.get("label_set_id") == CONSENSUS]
            if len(series) != 1:
                raise Missing(f"WorkflowEvals {wf}: {len(series)} {CONSENSUS} series in {path}")
            detail[wf] = {"metric": metric, "value": _num(series[0].get("metrics", {}), metric, f"{path} {CONSENSUS}")}
        except Missing as e:
            missing.append(str(e))
    if missing:
        raise Missing("; ".join(missing))
    return 100 * sum(x["value"] for x in detail.values()) / len(detail), detail


def _decidebench(d: Path | None) -> tuple[float, dict]:
    if d is None:
        raise Missing("DecideBench: --decidebench not given")
    path = Path(d) / "metrics.json"
    m = _read(path, "DecideBench")
    variant = m.get("variant", (m.get("run") or {}).get("variant"))
    if variant != "examples":
        raise Missing(f"DecideBench: {path} is variant {variant!r}, the gate needs 'examples'")
    return 100 * _num(m, "accuracy", str(path)), {"source": str(path), "variant": variant,
                                                  "stage": (m.get("run") or {}).get("stage")}


def _bench(d: Path | None, name: str, key: str) -> tuple[float, dict]:
    if d is None:
        raise Missing(f"{name}: --bench not given")
    path = Path(d) / name / "metrics.json"
    return 100 * _num(_read(path, name), key, str(path)), {"source": str(path), "key": key}


def _dev(path: Path | None, key: str, what: str, scale: float) -> tuple[float, dict]:
    return scale * _num(_read(path, what), key, f"{what} {path}"), {"source": str(path), "key": key}


def evaluate(stage0_dev=None, stage1_dev=None, workflowevals=None, bench=None, decidebench=None) -> dict:
    getters = {
        "workflowevals_overall": lambda: workflowevals_overall(workflowevals),
        "decidebench_examples": lambda: _decidebench(decidebench),
        "jevbench": lambda: _bench(bench, "jevbench", "accuracy"),
        "dynabench_f1": lambda: _bench(bench, "dynabench_test", "f1"),
        "rjudge_f1": lambda: _bench(bench, "rjudge", "f1"),
        "stage0_dev_accuracy": lambda: _dev(stage0_dev, "accuracy", "Stage 0 dev report", 100.0),
        "ece": lambda: _dev(stage1_dev, "ece", "Stage 1 dev report", 1.0),
    }
    rows = []
    for c in CHECKS:
        try:
            value, detail = getters[c.name]()
            note = None
        except Missing as e:
            value, detail, note = None, None, str(e)
        rows.append({"check": c.name, "label": c.label, "value": value, "minimum": c.minimum, "target": c.target,
                     "direction": ">=" if c.higher_is_better else "<=", "status": status(c, value),
                     "detail": detail, "note": note})
    return {"rows": rows, "publishable": all(r["status"] != "FAIL" for r in rows),
            "inputs": {"stage0_dev": _s(stage0_dev), "stage1_dev": _s(stage1_dev),
                       "workflowevals": _s(workflowevals), "bench": _s(bench), "decidebench": _s(decidebench)}}


def _s(p) -> str | None:
    return None if p is None else str(p)


def table(rep: dict) -> str:
    def fmt(v, ece):
        return "-" if v is None else (f"{v:.4f}" if ece else f"{v:.2f}")
    lines = [f"{'check':<24} {'value':>8} {'min':>8} {'target':>8}  status", "-" * 60]
    for r in rep["rows"]:
        ece = r["check"] == "ece"
        lines.append(f"{r['check']:<24} {fmt(r['value'], ece):>8} {r['direction'] + fmt(r['minimum'], ece):>8} "
                     f"{fmt(r['target'], ece):>8}  {r['status']}" + (f"  ({r['note']})" if r["note"] else ""))
    lines.append("-" * 60)
    lines.append(f"PUBLISHABLE: {'yes' if rep['publishable'] else 'no'}")
    return "\n".join(lines)
