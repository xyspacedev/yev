"""Shortcut filter (spec §4.4 step 6): rows a shallow model can answer without reading the options carefully."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

from yev.schema import Decision

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
SYNTHETIC_PREFIX = "synthetic_"
Score = tuple[float, bool]


def _mask(text: str) -> str:
    return NUM_RE.sub(" NUM ", text)


def masked_input(d: Decision) -> str:
    return f"{_mask(d.state)} || {_mask(d.question)}"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _solved(p: np.ndarray, gi: int) -> Score:
    top = p.max()
    return float(p[gi]), bool(p[gi] == top and int((p == top).sum()) == 1)


def _uniform(decisions: list[Decision]) -> list[Score]:
    return [(1.0 / len(d.options), False) for d in decisions]


def key_prior_scores(decisions: list[Decision], n_folds: int = 5) -> list[Score]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    out = _uniform(decisions)
    if not decisions:
        return out
    common, count = Counter(tuple(sorted(d.keys)) for d in decisions).most_common(1)[0]
    idx = [i for i, d in enumerate(decisions) if tuple(sorted(d.keys)) == common]
    ys = [decisions[i].gold for i in idx]
    if count * 2 < len(decisions) or len(set(ys)) < 2:
        return out
    groups = [decisions[i].cluster_id or decisions[i].id for i in idx]
    k = min(n_folds, len(set(groups)))
    if k < 2:
        return out
    texts = [masked_input(decisions[i]) for i in idx]
    for tr, te in GroupKFold(n_splits=k).split(np.zeros(len(idx)), groups=groups):
        if len({ys[j] for j in tr}) < 2:
            continue
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
        try:
            x = vec.fit_transform([texts[j] for j in tr])
        except ValueError:  # empty vocabulary
            continue
        clf = LogisticRegression(max_iter=1000, C=10.0).fit(x, [ys[j] for j in tr])
        classes = list(clf.classes_)
        for row, j in zip(clf.predict_proba(vec.transform([texts[j] for j in te])), te):
            d = decisions[idx[j]]
            p = np.array([row[classes.index(key)] if key in classes else 0.0 for key in d.keys])
            if p.sum() > 0:
                out[idx[j]] = _solved(p / p.sum(), d.keys.index(d.gold))
    return out


def lexical_scores(decisions: list[Decision]) -> list[Score]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [_mask(d.state) for d in decisions] + [_mask(o.description) for d in decisions for o in d.options]
    try:
        vec = TfidfVectorizer(sublinear_tf=True).fit(docs)
    except ValueError:
        return _uniform(decisions)
    out: list[Score] = []
    for d in decisions:
        state = vec.transform([_mask(d.state)])
        opts = vec.transform([_mask(o.description) for o in d.options])
        sims = (opts @ state.T).toarray().ravel()  # TF-IDF rows are L2-normalised, so this is cosine
        out.append(_solved(_softmax(sims * 10.0), d.keys.index(d.gold)))
    return out


def _best(a: Score, b: Score) -> Score:
    solved = [s for s in (a, b) if s[1]]
    return max(solved) if solved else max(a, b)


def detector_scores(decisions: list[Decision], n_folds: int = 5) -> list[tuple[Score, Score]]:
    """Per row: (key_prior, lexical). Kept separate because no single text-only model combines them."""
    return list(zip(key_prior_scores(decisions, n_folds), lexical_scores(decisions)))


def shortcut_scores(decisions: list[Decision], n_folds: int = 5) -> list[Score]:
    return [_best(a, b) for a, b in detector_scores(decisions, n_folds)]


def _signature(d: Decision) -> tuple:
    return (_mask(d.state), _mask(d.question), tuple(sorted(_mask(o.description) for o in d.options)))


def apply_shortcut(
    decisions: list[Decision], detectors: list[tuple[Score, Score]], *, threshold: float = 0.9,
    public_weight: float = 0.3,
) -> tuple[list[Decision], dict]:
    scores = [_best(kp, lex) for kp, lex in detectors]
    members: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decisions):
        if d.source.startswith(SYNTHETIC_PREFIX) and d.cluster_id:
            members[d.cluster_id].append(i)
    doomed: set[str] = set()
    for c, idx in members.items():
        if len(idx) < 2:
            continue
        golds_by_sig: dict[tuple, set[str]] = defaultdict(set)
        for i in idx:
            golds_by_sig[_signature(decisions[i])].add(decisions[i].gold)
        if any(len(g) > 1 for g in golds_by_sig.values()):
            continue  # a masked twin with another gold: no text-only model can solve this cluster
        if all(detectors[i][0][1] for i in idx) or all(detectors[i][1][1] for i in idx):
            doomed.add(c)
    report: dict[str, Counter] = defaultdict(Counter)
    kept: list[Decision] = []
    for d, (p, ok), (kp, lex) in zip(decisions, scores, detectors):
        r = report[d.source]
        r["in"] += 1
        r["shortcut_correct"] += int(ok)
        r["solved_key_prior"] += int(kp[1])
        r["solved_lexical"] += int(lex[1])
        if d.source.startswith(SYNTHETIC_PREFIX) and d.cluster_id in doomed:
            r["dropped"] += 1
            continue
        if not d.source.startswith(SYNTHETIC_PREFIX) and ok and p > threshold:
            d = replace(d, weight=public_weight)
            r["downweighted"] += 1
        r["out"] += 1
        kept.append(d)
    keys = ("in", "shortcut_correct", "solved_key_prior", "solved_lexical", "downweighted", "dropped", "out")
    return kept, {s: {k: c.get(k, 0) for k in keys} for s, c in report.items()}


def run_shortcut(
    decisions: list[Decision], *, min_rows: int = 50, threshold: float = 0.9, public_weight: float = 0.3
) -> tuple[list[Decision], dict]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decisions):
        key = f"{d.source}/{d.family}" if d.source.startswith(SYNTHETIC_PREFIX) else d.source
        groups[key].append(i)
    detectors = [(u, u) for u in _uniform(decisions)]
    for idx in groups.values():
        if len(idx) < min_rows:
            continue
        for i, pair in zip(idx, detector_scores([decisions[i] for i in idx])):
            detectors[i] = pair
    return apply_shortcut(decisions, detectors, threshold=threshold, public_weight=public_weight)
