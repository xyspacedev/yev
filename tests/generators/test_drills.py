from collections import Counter, defaultdict
from datetime import date, timedelta
from fractions import Fraction

from yev.generators.common import EDIT_TYPES, MAX_EDIT_TOKENS, token_edit_size
from yev.generators.dates import fmt
from yev.generators.drills import KINDS, generate


def by_cluster(ds):
    out = defaultdict(list)
    for d in ds:
        out[d.cluster_id].append(d)
    return out


def _bdays(a: date, b: date, hol) -> int:
    return sum(1 for k in range(1, (b - a).days + 1)
               if (a + timedelta(days=k)).weekday() < 5 and (a + timedelta(days=k)) not in hol)


def _deadline(m: dict) -> date:
    start, n = date.fromisoformat(m["start"]), m["n"]
    hol = {date.fromisoformat(h) for h in m["holidays"]}
    if m["unit"] == "calendar":
        return start + timedelta(days=n)
    d = start
    while _bdays(start, d, hol) < n or d.weekday() >= 5 or d in hol:
        d += timedelta(days=1)
    return d


def expected(d) -> str:
    m = d.metadata
    kind = m["kind"]
    if kind == "deadline_met":
        return "yes" if date.fromisoformat(m["done"]) <= _deadline(m) else "no"
    if kind == "deadline_choice":
        target = fmt(_deadline(m), m["option_style"])
        [key] = [o.key for o in d.options if o.description == target]
        return key
    if kind == "business_days":
        hol = {date.fromisoformat(h) for h in m["holidays"]}
        n = _bdays(date.fromisoformat(m["start"]), date.fromisoformat(m["end"]), hol)
        [key] = [o.key for o in d.options if o.description.startswith(f"{n} business day")]
        return key
    if kind == "sum_threshold":
        total = sum(c for c, counted in zip(m["amounts"], m["counted"]) if counted)
        hit = total > m["threshold"] if m["strict"] else total >= m["threshold"]
        return "yes" if hit else "no"
    if kind == "pct_change":
        change = Fraction(m["new"] - m["old"], m["old"]) * 100
        return str(sum(1 for c in m["cuts"] if change >= c))
    raise AssertionError(kind)


def test_drills_labels_computed():
    ds = generate(500, seed=7)
    kinds = Counter(d.metadata["kind"] for d in ds)
    assert set(kinds) == set(KINDS) and min(kinds.values()) > 100, kinds
    for d in ds:
        d.validate()
        assert d.family == "date_number_drills" and d.source == "rules:date_number_drills"
        assert d.gold == expected(d), (d.id, d.metadata, d.state)


def test_drills_question_shapes():
    for d in generate(300, seed=8):
        kind = d.metadata["kind"]
        if kind in ("deadline_met", "sum_threshold"):
            assert d.type == "noul"
        elif kind == "deadline_choice":
            assert d.type == "choice" and 4 <= len(d.options) <= 6
        elif kind == "business_days":
            assert d.type == "choice" and 3 <= len(d.options) <= 6
        else:
            assert d.type == "score"
            assert d.keys == [str(i) for i in range(len(d.keys))]
            assert 4 <= len(d.keys) <= 6


def test_drills_clusters_are_contrastive():
    for members in by_cluster(generate(300, seed=9)).values():
        assert len({d.gold for d in members}) >= 2
        base = members[0]
        assert base.edit_type is None
        for d in members:
            assert d.question == base.question and d.options == base.options
        for d in members[1:]:
            assert d.edit_type in EDIT_TYPES
            assert d.state != base.state
            assert token_edit_size(base.state, d.state) <= MAX_EDIT_TOKENS


def test_drills_gold_spread():
    ds = generate(500, seed=10)
    for kind in ("deadline_met", "sum_threshold"):
        golds = Counter(d.gold for d in ds if d.metadata["kind"] == kind)
        assert min(golds["yes"], golds["no"]) > 0.25 * sum(golds.values()), (kind, golds)
    pos = Counter(d.keys.index(d.gold) for d in ds if d.metadata["kind"] == "deadline_choice")
    assert len(pos) >= 4, pos


def test_generators_deterministic():
    a, b = generate(40, seed=1), generate(40, seed=1)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    assert [x.state for x in a] != [x.state for x in generate(40, seed=2)]
    assert all(x.id.startswith("date_number_drills:1:") for x in a)
