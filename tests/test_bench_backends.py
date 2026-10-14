"""Latency grid and replay helpers for the serving-backend benchmark (Plan 7 Task 6)."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bench_backends", Path("scripts/bench_backends.py"))
bb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bb)


def dev_row(i, state, opts=("x", "y")):
    user = {"state": state, "question": f"q{i}?",
            "options": [{"label": "AB"[k], "key": o, "description": f"d {o}"} for k, o in enumerate(opts)]}
    return {"id": str(i), "type": "choice", "messages": [{"role": "user", "content": json.dumps(user)}]}


def test_replay_groups_rows_by_state_into_requests():
    rows = [dev_row(0, "s1"), dev_row(1, "s2"), dev_row(2, "s1")]
    reqs = bb.replay_requests(rows, n=10, seed=0)
    assert sorted(len(r["questions"]) for r in reqs) == [1, 2]
    two = next(r for r in reqs if len(r["questions"]) == 2)
    assert two["state"] == "s1"
    q = next(iter(two["questions"].values()))
    assert q["type"] == "choice" and q["criteria"] == {"x": "d x", "y": "d y"}


def test_replay_sample_is_seeded_and_capped():
    rows = [dev_row(i, f"s{i}") for i in range(50)]
    a, b = bb.replay_requests(rows, 10, 0), bb.replay_requests(rows, 10, 0)
    assert len(a) == 10 and a == b and a != bb.replay_requests(rows, 10, 1)


def test_grid_questions_mix_types():
    qs = bb.grid_questions(4)
    assert len(qs) == 4 and {q["type"] for q in qs.values()} == {"noul", "choice", "score"}


def test_summarise():
    s = bb.summarise([1.0, 2.0, 3.0, 4.0], wall_s=2.0, n_questions=8)
    assert s["n"] == 4 and s["requests_per_s"] == 2.0 and s["questions_per_s"] == 4.0
    assert s["p50_s"] == pytest.approx(2.5) and s["p95_s"] == pytest.approx(3.85)


def test_grid_bodies_have_unique_states_of_similar_length():
    bodies = [bb.grid_body(2000, 4, i) for i in range(12)]
    states = [b["state"] for b in bodies]
    assert len(set(states)) == 12  # no cross-request prefix-cache hits between repeats
    lens = [len(s.split()) for s in states]
    assert max(lens) - min(lens) <= 2 and len(bodies[0]["questions"]) == 4


def test_replay_sets_are_disjoint_per_level_and_warmup_is_separate():
    rows = [dev_row(i, f"s{i}") for i in range(60)]
    warm, sets = bb.replay_sets(rows, n=10, seed=0, levels=4, warmup=2)
    states = [r["state"] for r in warm] + [r["state"] for s in sets for r in s]
    assert len(warm) == 2 and [len(s) for s in sets] == [10, 10, 10, 10]
    assert len(set(states)) == len(states)  # every request has a state no other pass has seen


def test_replay_sets_shrink_evenly_when_states_run_out():
    rows = [dev_row(i, f"s{i}") for i in range(22)]
    warm, sets = bb.replay_sets(rows, n=10, seed=0, levels=4, warmup=2)
    assert [len(s) for s in sets] == [5, 5, 5, 5]
