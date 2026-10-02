"""Dev metrics (spec §7) and the TEV comparison."""
from __future__ import annotations
from collections import defaultdict
from yev.format import LETTERS

TEV_ACC, TEV_GROUP = 0.928, 0.860


def group_key(pred: dict) -> str | None:
    if pred["source"] == "decidebench":
        return pred["id"].rsplit("-", 1)[0]
    return pred.get("cluster_id")


def _headline(preds: list[dict]) -> dict:
    n = len(preds)
    if n == 0:
        return {"n": 0}
    correct, brier, bins = [], 0.0, defaultdict(list)
    for x in preds:
        gi = LETTERS.index(x["answer"])
        pi = max(range(len(x["probs"])), key=lambda i: x["probs"][i])
        ok = pi == gi
        correct.append(ok)
        brier += sum((q - (1.0 if i == gi else 0.0)) ** 2 for i, q in enumerate(x["probs"]))
        conf = x["probs"][pi]
        bins[min(int(conf * 15), 14)].append((conf, ok))
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins.values())
    sel = [ok for x, ok in zip(preds, correct) if max(x["probs"]) >= 0.9]
    groups = defaultdict(list)
    for x, ok in zip(preds, correct):
        k = group_key(x)
        if k:
            groups[k].append(ok)
    g = [all(v) for v in groups.values() if len(v) >= 2]
    return {"n": n, "accuracy": sum(correct) / n, "brier": brier / n, "ece": ece,
            "selective": {"threshold": 0.9, "coverage": len(sel) / n, "accuracy": (sum(sel) / len(sel)) if sel else None},
            "group_accuracy": (sum(g) / len(g)) if g else None, "n_groups": len(g)}


def calibration(preds: list[dict]) -> dict:
    """ECE (15 bins on the top-choice probability) and multi-class Brier; preds need `answer` and `probs`."""
    h = _headline([{**x, "source": x.get("source"), "cluster_id": None} for x in preds])
    return {"ece": h.get("ece"), "brier": h.get("brier")}


def _by(preds, field):
    out = defaultdict(list)
    for x in preds:
        out[str(x.get(field))].append(x)
    return {k: {"n": len(v), "accuracy": _headline(v)["accuracy"]} for k, v in sorted(out.items())}


def evaluate(preds: list[dict]) -> dict:
    rep = _headline(preds)
    rep.update(by_family=_by(preds, "family"), by_edit_type=_by(preds, "edit_type"), by_source=_by(preds, "source"))
    db = [x for x in preds if x["source"] == "decidebench"]
    dbh = _headline(db)
    if db:
        dbh["vs_tev"] = {"accuracy_bar": TEV_ACC, "group_bar": TEV_GROUP,
                         "passes": dbh["accuracy"] > TEV_ACC and (dbh["group_accuracy"] or 0) > TEV_GROUP}
        dbh["by_family"] = _by(db, "family")
    rep["decidebench"] = dbh
    return rep
