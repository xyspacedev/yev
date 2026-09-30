import math
from jeff.train.targets import smooth_target

def close(a, b): return all(math.isclose(a[k], b[k], abs_tol=1e-9) for k in a) and a.keys() == b.keys()

def test_hard_choice_uniform_smoothing():
    t = smooth_target({"A": 0.0, "B": 1.0, "C": 0.0}, "choice", eps_hard=0.06)
    assert close(t, {"A": 0.02, "B": 0.96, "C": 0.02})

def test_noul_is_smoothed_like_choice():
    t = smooth_target({"A": 1.0, "B": 0.0}, "noul", eps_hard=0.05)
    assert close(t, {"A": 0.975, "B": 0.025})

def test_soft_target_unchanged():
    soft = {"A": 0.67, "B": 0.33}
    assert smooth_target(soft, "choice") == soft

def test_score_middle_gold_splits_to_both_neighbours():
    t = smooth_target({"A": 0, "B": 0, "C": 1.0, "D": 0, "E": 0}, "score", eps_ordinal=0.1)
    assert close(t, {"A": 0, "B": 0.05, "C": 0.9, "D": 0.05, "E": 0})

def test_score_edge_gold_gives_all_to_single_neighbour():
    t = smooth_target({"A": 1.0, "B": 0, "C": 0}, "score", eps_ordinal=0.1)
    assert close(t, {"A": 0.9, "B": 0.1, "C": 0})

def test_sums_to_one():
    for tp in ("choice", "noul", "score"):
        t = smooth_target({"A": 0, "B": 1.0, "C": 0, "D": 0}, tp)
        assert math.isclose(sum(t.values()), 1.0)
