import json
from yev.mix import build_mix, repeat_rows, unit_key
from yev.schema import Decision, Option, write_jsonl


def rows(n, family="moderation", source="synthetic_opus", cluster=True):
    out = []
    for i in range(n):
        out.append(Decision(id=f"{family}:{i}", type="choice", state=f"state {family} {i}", question="q?",
                            options=[Option("a", "A"), Option("b", "B"), Option("c", "C")], gold="a",
                            family=family, source=source, licence="cc-by-4.0",
                            cluster_id=f"c:{family}:{i // 2}" if cluster else None))
    return out


def test_repeat_whole_and_fraction():
    taken = rows(40)
    extra = repeat_rows(taken, {"name": "b", "repeat": 1.5}, seed=0)
    units = {unit_key(d) for d in taken}
    assert len(extra) == 40 + 2 * round(0.5 * len(units))


def test_repeat_filter_limits_families():
    taken = rows(10, "moderation") + rows(10, "intent")
    extra = repeat_rows(taken, {"name": "b", "repeat": 2, "repeat_filter": {"family": ["moderation"]}}, seed=0)
    assert {d.family for d in extra} == {"moderation"} and len(extra) == 20


def test_repeat_filter_rejects_unknown_key():
    import pytest
    with pytest.raises(ValueError):
        repeat_rows(rows(4), {"name": "b", "repeat": 1, "repeat_filter": {"gold": ["a"]}}, seed=0)


def test_repeat_copies_are_separate_train_units():
    from yev.train.data import unit_of
    taken = rows(6, cluster=False)
    extra = repeat_rows(taken, {"name": "b", "repeat": 2}, seed=0)
    assert len({d.id for d in taken + extra}) == 18
    assert len({unit_key(d) for d in taken + extra}) == 18
    assert len({unit_of({"id": d.id, "cluster_id": d.cluster_id}) for d in taken + extra}) == 18


def test_repeat_extra_passes():
    taken = rows(10, "moderation") + rows(10, "intent")
    block = {"name": "b", "repeat": 2, "repeat_filter": {"family": ["moderation"]},
             "repeat_extra": [{"repeat": 1, "repeat_filter": {"family": ["intent"]}}]}
    extra = repeat_rows(taken, block, seed=0)
    assert sum(d.family == "moderation" for d in extra) == 20 and sum(d.family == "intent" for d in extra) == 10
    assert len({d.id for d in extra}) == 30


def test_no_repeat_is_byte_identical(tmp_path, empty_fingerprints):
    pool = tmp_path / "p.jsonl"
    write_jsonl(pool, rows(50, source="synthetic_planned"))
    base = {"seed": 0, "blocks": [{"name": "b", "files": [str(pool)], "rows": 30, "dev_share": 0.1}]}
    a = build_mix(base, empty_fingerprints)
    b = build_mix(json.loads(json.dumps(base)), empty_fingerprints)
    assert [d.id for d in a.train] == [d.id for d in b.train]
    assert "repeated" not in a.report["blocks"]["b"]


def test_repeat_in_build_mix_train_only(tmp_path, empty_fingerprints):
    pool = tmp_path / "p.jsonl"
    write_jsonl(pool, rows(50, source="synthetic_planned"))
    recipe = {"seed": 0, "blocks": [{"name": "b", "files": [str(pool)], "rows": 30, "dev_share": 0.1, "repeat": 1}]}
    res = build_mix(recipe, empty_fingerprints)
    plain = build_mix({**recipe, "blocks": [{**recipe["blocks"][0], "repeat": 0}]}, empty_fingerprints)
    assert len(res.train) == 2 * len(plain.train)
    assert [d.id for d in res.dev] == [d.id for d in plain.dev]
    assert res.report["blocks"]["b"]["repeated"] == len(plain.train)


def test_planned_group_shares_split(tmp_path, empty_fingerprints):
    group_rows = []
    for g in range(40):
        for s in ("obey", "ignore"):
            group_rows.append(Decision(id=f"plan:r:agent_injection_pair:g{g}:{s}:verdict", type="noul",
                                       state=f"trace {g} {s}", question="q?",
                                       options=[Option("yes", "Y"), Option("no", "N")],
                                       gold="yes" if s == "obey" else "no", family="agent_injection_pair",
                                       source="synthetic_planned", licence="cc-by-4.0",
                                       cluster_id=f"plan:r:agent_injection_pair:g{g}"))
    pool = tmp_path / "p.jsonl"
    write_jsonl(pool, group_rows)
    res = build_mix({"seed": 0, "blocks": [{"name": "b", "files": [str(pool)], "rows": 80, "dev_share": 0.2}]},
                    empty_fingerprints)
    split = {}
    for d in res.train + res.dev:
        split.setdefault(d.cluster_id, set()).add(d.split)
    assert all(len(v) == 1 for v in split.values())
    assert res.dev  # holdout is per group, not one unit for the whole family


def test_repeat_filter_matching_nothing_raises():
    import pytest
    taken = rows(10, "moderation")
    with pytest.raises(ValueError, match=r"block b.*intent"):
        repeat_rows(taken, {"name": "b", "repeat": 1, "repeat_filter": {"family": ["intent"]}}, seed=0)
    with pytest.raises(ValueError, match=r"block b.*routing"):
        repeat_rows(taken, {"name": "b", "repeat": 1,
                            "repeat_extra": [{"repeat": 1, "repeat_filter": {"family": ["routing"]}}]}, seed=0)
    # repeat 0 with a filter is a no-op, not an error
    assert repeat_rows(taken, {"name": "b", "repeat": 0, "repeat_filter": {"family": ["intent"]}}, seed=0) == []


def test_build_mix_rejects_duplicate_ids(tmp_path, empty_fingerprints):
    import pytest
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, rows(20, source="synthetic_planned"))
    dup = [Decision(id=d.id, type=d.type, state=d.state + " other run", question=d.question, options=d.options,
                    gold=d.gold, family=d.family, source=d.source, licence=d.licence,
                    cluster_id="other:" + d.cluster_id) for d in rows(20, source="synthetic_planned")]
    write_jsonl(b, dup)
    recipe = {"seed": 0, "blocks": [{"name": "a", "files": [str(a)], "rows": 20},
                                    {"name": "b", "files": [str(b)], "rows": 20, "dev_share": 0.1}]}
    with pytest.raises(ValueError, match=r"duplicate ids.*moderation:"):
        build_mix(recipe, empty_fingerprints)
    # repeats get unique #r{k} ids and do not trip the check
    build_mix({"seed": 0, "blocks": [{"name": "a", "files": [str(a)], "rows": 20, "repeat": 2.5}]}, empty_fingerprints)
