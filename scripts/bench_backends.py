"""Latency grid and concurrent replay against any /v1/systemone server (transformers, vLLM or SGLang behind yev serve).

    python scripts/bench_backends.py grid   --url http://127.0.0.1:8000 --label vllm --gpu-mem --out grid_vllm.json
    python scripts/bench_backends.py replay --url http://127.0.0.1:8000 --label vllm --gpu-mem \\
        --data data/mix/stage1c/dev.chat.jsonl --out replay_vllm.json

grid: state ~300 / 2k / 8k tokens x 1 / 4 / 16 questions, 2 warm-up + 10 timed requests per cell, one at a time;
every request's state starts with its own line, so repeats can't be served from a cross-request prefix cache.
replay: 500 dev requests (rows regrouped by state), seeded, at concurrency 1, 4, 16, 64; each level gets its own
disjoint set of states and the warm-up uses two more, so no pass is timed on another pass's cached prefixes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path

PARA = ("Ticket 4471: the customer reports that the replacement router shipped last week drops its connection every "
        "evening. They have restarted it twice, swapped the cable and moved it away from the microwave. They ask "
        "whether they can return it for a refund or need another replacement, and mention a double shipping charge. ")
STATE_TOKENS, QUESTIONS, WARMUP, TIMED = (300, 2000, 8000), (1, 4, 16), 2, 10


def grid_state(tokens: int) -> str:
    return "\n".join(PARA for _ in range(max(1, tokens // 60)))  # ~60 tokens per paragraph


def grid_questions(n: int) -> dict:
    out = {}
    for i in range(n):
        if i % 3 == 0:
            out[f"q{i}"] = {"type": "noul", "instructions": f"Does the customer ask for a refund? (q{i})"}
        elif i % 3 == 1:
            out[f"q{i}"] = {"type": "choice", "instructions": f"Which team should handle this? (q{i})",
                            "criteria": {"billing": "charges and refunds", "hardware": "device faults",
                                         "shipping": "delivery issues", "other": None}}
        else:
            out[f"q{i}"] = {"type": "score", "instructions": f"How frustrated is the customer? (q{i})",
                            "criteria": ["calm", "mildly annoyed", "frustrated", "angry", "furious"]}
    return out


def grid_body(tokens: int, n_questions: int, i: int) -> dict:
    """Request i of a grid cell: a state unique to i (so no repeat hits a sidecar's cross-request prefix cache)."""
    return {"state": f"Request {i}.\n" + grid_state(tokens), "questions": grid_questions(n_questions)}


def replay_requests(rows: list[dict], n: int, seed: int) -> list[dict]:
    by_state: dict[str, dict] = {}
    for r in rows:
        u = json.loads(r["messages"][-1]["content"])
        req = by_state.setdefault(u["state"], {"state": u["state"], "questions": {}})
        req["questions"][f"q{len(req['questions'])}"] = {
            "type": "choice", "instructions": u["question"],
            "criteria": {o["key"]: o.get("description") for o in u["options"]}}
    reqs = list(by_state.values())
    random.Random(seed).shuffle(reqs)
    return reqs[:n]


def replay_sets(rows: list[dict], n: int, seed: int, levels: int, warmup: int = 2) -> tuple[list[dict], list[list[dict]]]:
    """Warm-up requests and one disjoint set of requests per concurrency level, so no pass reuses a state another
    pass (or the warm-up) already put in a sidecar's prefix cache. With too few states, every set shrinks evenly."""
    reqs = replay_requests(rows, len(rows), seed)
    warm, rest = reqs[:warmup], reqs[warmup:]
    per = min(n, len(rest) // levels)
    return warm, [rest[k * per : (k + 1) * per] for k in range(levels)]


def _p95(xs: list[float]) -> float:
    return statistics.quantiles(xs, n=100, method="inclusive")[94] if len(xs) > 1 else xs[0]


def summarise(latencies: list[float], wall_s: float, n_questions: int) -> dict:
    return {"n": len(latencies), "requests_per_s": len(latencies) / wall_s, "questions_per_s": n_questions / wall_s,
            "p50_s": statistics.median(latencies), "p95_s": _p95(latencies)}


class GpuPeak:
    """Peak `nvidia-smi` memory.used (MiB) while running; None if nvidia-smi is missing."""

    def __init__(self, on: bool):
        self.on, self.peak, self._stop = on, None, threading.Event()

    def __enter__(self):
        if self.on:
            threading.Thread(target=self._poll, daemon=True).start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            try:
                used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                                    "--format=csv,noheader,nounits"], text=True).split()[0])
                self.peak = max(self.peak or 0, used)
            except (OSError, subprocess.CalledProcessError, ValueError):
                return
            self._stop.wait(0.5)

    def __exit__(self, *exc):
        self._stop.set()


async def _post(client, url, body) -> float:
    t = time.perf_counter()
    r = await client.post(f"{url}/v1/systemone", json=body)
    r.raise_for_status()
    return time.perf_counter() - t


async def run_grid(url: str) -> list[dict]:
    import httpx
    out = []
    async with httpx.AsyncClient(timeout=600) as c:
        for tok in STATE_TOKENS:
            for nq in QUESTIONS:
                for i in range(WARMUP):
                    await _post(c, url, grid_body(tok, nq, -1 - i))
                ts = [await _post(c, url, grid_body(tok, nq, i)) for i in range(TIMED)]
                out.append({"state_tokens": tok, "questions": nq, "p50_s": statistics.median(ts), "p95_s": _p95(ts),
                            "ms_per_question": 1000 * statistics.median(ts) / nq})
                print(json.dumps(out[-1]))
    return out


async def run_replay(url: str, warm: list[dict], reqs: list[dict], concurrency: int) -> dict:
    import httpx
    sem, lat = asyncio.Semaphore(concurrency), []
    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=concurrency)) as c:
        for body in warm:
            await _post(c, url, body)  # warm-up: requests outside the timed set

        async def one(body):
            async with sem:
                lat.append(await _post(c, url, body))

        t = time.perf_counter()
        await asyncio.gather(*(one(b) for b in reqs))
        wall = time.perf_counter() - t
    return {"concurrency": concurrency, **summarise(lat, wall, sum(len(b["questions"]) for b in reqs))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("grid", "replay"):
        p = sub.add_parser(name)
        p.add_argument("--url", default="http://127.0.0.1:8000")
        p.add_argument("--label", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--gpu-mem", action="store_true")
        if name == "replay":
            p.add_argument("--data", required=True)
            p.add_argument("--n", type=int, default=500)
            p.add_argument("--concurrency", default="1,4,16,64")
            p.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    with GpuPeak(args.gpu_mem) as gpu:
        if args.cmd == "grid":
            result = {"grid": asyncio.run(run_grid(args.url))}
        else:
            rows = [json.loads(x) for x in Path(args.data).read_text().splitlines() if x.strip()]
            levels = [int(k) for k in args.concurrency.split(",")]
            warm, sets = replay_sets(rows, args.n, args.seed, len(levels))
            if len(sets[0]) < args.n:
                print(f"only {len(sets[0])} distinct requests per level (asked {args.n})")
            result = {"n_requests_per_level": len(sets[0]),
                      "replay": [{**asyncio.run(run_replay(args.url, warm, reqs, k)),
                                  "n_questions": sum(len(r["questions"]) for r in reqs)}
                                 for k, reqs in zip(levels, sets)]}
            for r in result["replay"]:
                print(json.dumps(r))
    result.update({"label": args.label, "url": args.url, "peak_gpu_mib": gpu.peak})
    Path(args.out).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
