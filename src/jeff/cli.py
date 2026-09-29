"""Command-line entry point: python -m jeff <command>."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from jeff import decidebench
from jeff.schema import write_jsonl
from jeff.sources.base import build, fetch
from jeff.sources.registry import SOURCES


def _write_stats_atomically(stats_path: Path, all_stats: dict) -> None:
    """Write stats to file atomically."""
    tmp_path = stats_path.with_suffix(stats_path.suffix + ".tmp")
    try:
        with tmp_path.open("w") as f:
            json.dump(all_stats, f, indent=2)
        os.replace(tmp_path, stats_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def cmd_build_public(args: argparse.Namespace) -> int:
    # Validate --only names
    if args.only:
        valid_names = {spec.name for spec in SOURCES}
        unknown = [name for name in args.only if name not in valid_names]
        if unknown:
            print(f"Unknown sources: {', '.join(unknown)}")
            return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "build_stats.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    failed = []

    for spec in SOURCES:
        if args.only and spec.name not in args.only:
            continue
        try:
            decisions, stats = build(spec, fetch(spec), seed=args.seed)
            write_jsonl(out / f"{spec.name}.jsonl", decisions)
            all_stats[spec.name] = asdict(stats)
            print(f"{spec.name}: {asdict(stats)}")
            _write_stats_atomically(stats_path, all_stats)
        except Exception as exc:
            print(f"{spec.name}: FAILED: {exc!r}")
            failed.append(spec.name)

    return 1 if failed else 0


def cmd_dev(args: argparse.Namespace) -> int:
    n = write_jsonl(args.out, decidebench.load_examples())
    print(f"wrote {n} dev items to {args.out}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeff")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-public", help="download and convert public datasets")
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_build_public)

    p = sub.add_parser("dev", help="write the DecideBench examples pool as the dev set")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dev)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)
