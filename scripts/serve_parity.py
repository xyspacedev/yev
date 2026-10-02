"""Cached vs uncached /v1/systemone answers on real requests (Plan 4 Task 7 box parity check).

    uv run --extra train --extra serve python scripts/serve_parity.py \\
        --model runs/yev-4b/adapter --base /models/Qwen3.5-4B-Base [--calibration c.json] req1.json [req2.json ...]

Each request JSON is a /v1/systemone body ({"state", "questions", ...}). The model is loaded once and
answered by two Engines that share it: one with the prefix KV cache (min prefix 1 token, so it is used
whenever a request has 2+ questions) and one on the exact uncached path. Prints, per question, whether
the argmax key agrees and the max absolute probability difference, then a summary and the timings.
Exit status 1 if any argmax disagrees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def probs_of(answer) -> dict[str, float]:
    a = answer.model_dump()
    if a["type"] == "noul":
        return {"yes": a["noul"], "no": 1.0 - a["noul"]}
    return dict(a["probabilities"])


def compare(cached, exact, body: dict) -> tuple[list[dict], float, float]:
    """Per-question rows {name, agree, argmax_cached, argmax_exact, max_diff} and the two wall times."""
    from pydantic import TypeAdapter

    from yev.serve.contract import QuestionSpec
    qs = TypeAdapter(dict[str, QuestionSpec]).validate_python(body["questions"])
    t0 = time.perf_counter()
    a, _ = cached.answer(body["state"], qs)
    t1 = time.perf_counter()
    b, _ = exact.answer(body["state"], qs)
    t2 = time.perf_counter()
    out = []
    for name in qs:
        pa, pb = probs_of(a[name]), probs_of(b[name])
        ka, kb = max(pa, key=pa.get), max(pb, key=pb.get)
        out.append({"name": name, "agree": ka == kb, "argmax_cached": ka, "argmax_exact": kb,
                    "max_diff": max(abs(pa[k] - pb[k]) for k in pa)})
    return out, t1 - t0, t2 - t1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("requests", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base")
    ap.add_argument("--calibration")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batch-tokens", type=int, default=16384)
    args = ap.parse_args(argv)

    from yev.serve.engine import Engine
    from yev.train.infer import load
    tok, model = load(args.model, args.base)
    kw = dict(model=model, tokenizer=tok, max_len=args.max_len, batch_tokens=args.batch_tokens)
    cached = Engine(args.model, args.base, args.calibration, prefix_cache=True, min_prefix_tokens=1, **kw)
    exact = Engine(args.model, args.base, args.calibration, prefix_cache=False, **kw)

    n = agree = 0
    worst = 0.0
    tc = te = 0.0
    for path in args.requests:
        rows, dc, de = compare(cached, exact, json.loads(Path(path).read_text()))
        tc, te = tc + dc, te + de
        print(f"== {path}  cached {dc:.2f}s  exact {de:.2f}s")
        for r in rows:
            flag = "ok  " if r["agree"] else "DIFF"
            print(f"  {flag} {r['name']}: cached={r['argmax_cached']} exact={r['argmax_exact']} "
                  f"max|dp|={r['max_diff']:.2e}")
            n, agree, worst = n + 1, agree + r["agree"], max(worst, r["max_diff"])
    print(f"argmax agreement {agree}/{n}, max |dp| {worst:.2e}, time cached {tc:.2f}s vs exact {te:.2f}s")
    return 0 if agree == n else 1


if __name__ == "__main__":
    sys.exit(main())
