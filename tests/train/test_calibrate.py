import math, random
from yev.train.calibrate import fit_temperatures

def test_recovers_overconfident_temperature():
    rng = random.Random(0)
    items = []
    for _ in range(2000):
        true = [rng.gauss(0, 1) for _ in range(3)]
        m = max(true); e = [math.exp(x - m) for x in true]; s = sum(e); pr = [x / s for x in e]
        ans = rng.choices(range(3), weights=pr)[0]
        items.append({"type": "choice", "logits": [x * 3.0 for x in true], "n": 3, "answer_index": ans})
    t = fit_temperatures(items)["choice"]
    assert 2.5 < t < 3.5

def test_missing_type_defaults_to_one():
    t = fit_temperatures([{"type": "choice", "logits": [1.0, 0.0], "n": 2, "answer_index": 0}] * 10)
    assert t["noul"] == 1.0 and t["score"] == 1.0
