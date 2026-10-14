"""Letter-probability parity of a serving backend against the evaluated transformers readout, on dev rows.

Two phases, because the reference model and a sidecar engine shouldn't share the GPU:

    python scripts/backend_parity.py score --data data/mix/stage1c/dev.chat.jsonl --calibration c.json \\
        --model ~/runs/stage1c/final --base ~/models/Qwen3.5-4B-Base --out ref.jsonl
    python scripts/backend_parity.py score --data ... --calibration c.json --backend vllm \\
        --engine-url http://127.0.0.1:8100 --tokenizer ~/models/yev1c-merged --out vllm.jsonl
    python scripts/backend_parity.py compare ref.jsonl vllm.jsonl --out parity_vllm.json

Bar (spec): argmax agreement >= 99.5% and mean |dp| <= 0.01, |dp| being a row's largest per-letter difference.
compare exits 1 below the bar.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

AGREEMENT_BAR, MEAN_DP_BAR = 0.995, 0.01
CHUNK = 64  # rows in flight at once against a sidecar
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def by_state(rows: list[dict]) -> list[dict]:
    """Rows reordered so rows sharing a state are adjacent (first-seen order), so prefix caches get reused."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(json.loads(r["messages"][-1]["content"]).get("state", ""), []).append(r)
    return [r for g in groups.values() for r in g]


def compare_records(a: list[dict], b: list[dict]) -> dict:
    ra, rb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if set(ra) != set(rb):
        raise ValueError(f"record ids differ: {len(set(ra) ^ set(rb))} ids in only one file")
    agree, dps, dis, ok_a, ok_b = 0, [], [], 0, 0
    for id_ in sorted(ra):
        pa, pb = ra[id_]["probs"], rb[id_]["probs"]
        ka, kb = max(range(len(pa)), key=pa.__getitem__), max(range(len(pb)), key=pb.__getitem__)
        agree += ka == kb
        if ka != kb:
            dis.append(id_)
        dps.append(max(abs(x - y) for x, y in zip(pa, pb)))
        gold = LETTERS.index(ra[id_]["answer"])
        ok_a += ka == gold
        ok_b += kb == gold
    n = len(ra)
    out = {"n": n, "argmax_agreement": agree / n, "mean_abs_dp": sum(dps) / n, "max_abs_dp": max(dps),
           "accuracy_a": ok_a / n, "accuracy_b": ok_b / n, "disagreements": dis[:50]}
    out["at_parity"] = out["argmax_agreement"] >= AGREEMENT_BAR and out["mean_abs_dp"] <= MEAN_DP_BAR
    return out


def _record(row: dict, scores: list[float], temps: dict) -> dict:
    from yev.train import readout
    n = len(row["letters"])
    return {"id": row["id"], "type": row["type"], "answer": row["answer"],
            "probs": readout.probs(scores, n, temps.get(row["type"], 1.0))}


def score_transformers(rows, args, temps) -> list[dict]:
    from yev.train.infer import letter_logits, load
    tok, model = load(args.model, args.base)
    n = max(len(r["letters"]) for r in rows)
    logits = letter_logits(model, tok, rows, max_len=args.max_len, batch_tokens=args.batch_tokens, n_letters=n)
    return [_record(r, z, temps) for r, z in zip(rows, logits)]


def score_backend(rows, args, temps) -> list[dict]:
    from transformers import AutoTokenizer

    from yev.serve.backends import make_backend
    from yev.train.data import letter_token_ids, prompt_ids
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    letters = letter_token_ids(tok, max(len(r["letters"]) for r in rows))
    backend = make_backend(args.backend, args.engine_url)

    async def run() -> list[dict]:
        out = []
        for i in range(0, len(rows), CHUNK):
            chunk = rows[i : i + CHUNK]
            ids = [prompt_ids(tok, r["messages"]) for r in chunk]
            scores = await backend.letter_logprobs(ids, [letters[: len(r["letters"])] for r in chunk])
            out += [_record(r, z, temps) for r, z in zip(chunk, scores)]
            print(f"{len(out)}/{len(rows)}", file=sys.stderr, end="\r")
        return out

    return asyncio.run(run())


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--calibration")
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int)
    s.add_argument("--backend", choices=["transformers", "vllm", "sglang"], default="transformers")
    s.add_argument("--model")
    s.add_argument("--base")
    s.add_argument("--tokenizer", help="sidecar backends: tokenizer dir (the merged model dir)")
    s.add_argument("--engine-url", default="http://127.0.0.1:8100")
    s.add_argument("--max-len", type=int, default=16384)
    s.add_argument("--batch-tokens", type=int, default=16384)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "compare":
        out = compare_records(_read_jsonl(args.a), _read_jsonl(args.b))
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=2))
        return 0 if out["at_parity"] else 1

    from yev.serve.engine import load_temperatures
    rows = by_state(_read_jsonl(args.data))
    rows = rows[: args.limit] if args.limit else rows
    temps = load_temperatures(args.calibration)
    recs = score_transformers(rows, args, temps) if args.backend == "transformers" else score_backend(rows, args, temps)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"wrote {len(recs)} records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
