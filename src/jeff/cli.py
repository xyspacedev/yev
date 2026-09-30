"""Command-line entry point: python -m jeff <command>."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from jeff import decidebench
from jeff.filters.contamination import (
    EMBED_MODEL,
    EMBED_THRESHOLD,
    EmbeddingFilter,
    load_fingerprints,
    sentence_transformer_encoder,
)
from jeff.filters.pipeline import assert_clean, run_filters
from jeff.generators import actions, returns, severity
from jeff.generators.llm.attrib import prepare_pairs, render_attrib_prompt, score_attribution
from jeff.generators.llm.check import prepare_sheets, render_checker_prompt, score_answers
from jeff.generators.llm.ingest import ingest_file
from jeff.generators.llm.plan import plan_batches, render_writer_prompt
from jeff.generators.specs import SYNTHETIC_SPECS
from jeff.schema import read_jsonl, write_jsonl
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


def _git_sha() -> str | None:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = res.stdout.strip()
    return sha if res.returncode == 0 and sha else None


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


RULE_GENERATORS = {"returns": returns.generate, "actions": actions.generate, "severity": severity.generate}


def cmd_gen_rules(args: argparse.Namespace) -> int:
    rows = RULE_GENERATORS[args.family](args.clusters, args.seed)
    n = write_jsonl(args.out, rows)
    print(f"{args.family}: {n} rows in {len({r.cluster_id for r in rows})} clusters -> {args.out}")
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    src, out = Path(args.inp), Path(args.out)
    if src.resolve() == out.resolve():
        print("--in and --out must be different directories")
        return 2
    files = sorted(src.glob("*.jsonl"))
    if not files:
        print(f"no *.jsonl files found in {src}")
        return 2
    decisions = [d for path in files for d in read_jsonl(path)]
    fp = load_fingerprints()
    embed = None if args.no_embed else EmbeddingFilter(fp.states, sentence_transformer_encoder())
    kept, report = run_filters(decisions, fp, embed, {s.name: s for s in [*SOURCES, *SYNTHETIC_SPECS]})
    assert_clean(kept, fp)
    by_source: dict[str, list] = defaultdict(list)
    for d in kept:
        by_source[d.source].append(d)
    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("*.jsonl"), *out.glob("*.jsonl.tmp")]:
        stale.unlink()
    for source, rows in by_source.items():
        write_jsonl(out / f"{source}.jsonl", rows)
    report = {
        "meta": {
            "embedding": None if args.no_embed else EMBED_MODEL,
            "threshold": EMBED_THRESHOLD,
            "decidebench_revision": decidebench.REVISION,
            "git_sha": _git_sha(),
        },
        "sources": report,
    }
    _write_stats_atomically(out / "filter_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_stats_atomically(path, obj)


def _read_answer_lines(paths) -> tuple[list[dict], int]:
    answers, bad = [], 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                answers.append(obj)
            else:
                bad += 1
    return answers, bad


def _dump_items(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")


def cmd_synth_plan(args) -> int:
    run = Path(args.dir)
    for b in plan_batches(args.clusters, args.per_batch, args.seed):
        spec_path = run / "batches" / f"{b['batch_id']}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(b, indent=2))
        out = (run / "written" / f"{b['batch_id']}.jsonl").resolve()
        spec_path.with_suffix(".prompt.md").write_text(render_writer_prompt(b, str(out)))
    print(f"planned {args.clusters} clusters in {len(list((run / 'batches').glob('*.json')))} batches")
    return 0


def cmd_synth_ingest(args) -> int:
    run = Path(args.dir)
    rows, stats = [], Counter()
    for spec_path in sorted((run / "batches").glob("*.json")):
        b = json.loads(spec_path.read_text())
        written = run / "written" / f"{b['batch_id']}.jsonl"
        if not written.exists():
            stats["missing_file"] += 1
            continue
        got, s = ingest_file(written, b)
        rows.extend(got)
        stats.update(s)
    write_jsonl(run / "ingested.jsonl", rows)
    stats["rows"] = len(rows)
    _write_json(run / "reports" / "ingest.json", dict(stats))
    print(json.dumps(dict(stats)))
    return 0


def cmd_synth_check_prepare(args) -> int:
    run = Path(args.dir)
    parts, key = prepare_sheets(list(read_jsonl(run / "ingested.jsonl")), n_sheets=args.sheets,
                                part_size=args.part_size, seed=args.seed)
    for name, items in parts:
        sheet = (run / "check" / f"{name}.jsonl").resolve()
        _dump_items(sheet, items)
        answers = sheet.with_name(f"answers-{name}.jsonl")
        sheet.with_suffix(".prompt.md").write_text(render_checker_prompt(str(sheet), str(answers)))
    _write_json(run / "private" / "check-key.json", key)
    print(f"{len(parts)} checker parts")
    return 0


def cmd_synth_check_score(args) -> int:
    run = Path(args.dir)
    key = json.loads((run / "private" / "check-key.json").read_text())
    answers, bad = _read_answer_lines(sorted((run / "check").glob("answers-*.jsonl")))
    kept, stats = score_answers(list(read_jsonl(run / "ingested.jsonl")), key, answers)
    stats["bad_answer_line"] = bad
    write_jsonl(run / "checked.jsonl", kept)
    _write_json(run / "reports" / "check.json", dict(stats))
    print(json.dumps(dict(stats)))
    return 0


def cmd_synth_attrib_prepare(args) -> int:
    run = Path(args.dir)
    parts, key = prepare_pairs(list(read_jsonl(run / "checked.jsonl")), part_size=args.part_size, seed=args.seed)
    for name, items in parts:
        path = (run / "attrib" / f"{name}.jsonl").resolve()
        _dump_items(path, items)
        path.with_suffix(".prompt.md").write_text(
            render_attrib_prompt(str(path), str(path.with_name(f"answers-{name}.jsonl"))))
    _write_json(run / "private" / "attrib-key.json", key)
    print(f"{len(parts)} attribution parts")
    return 0


def cmd_synth_attrib_score(args) -> int:
    run = Path(args.dir)
    key = json.loads((run / "private" / "attrib-key.json").read_text())
    answers, bad = _read_answer_lines(sorted((run / "attrib").glob("answers-*.jsonl")))
    checked = list(read_jsonl(run / "checked.jsonl"))
    kept, stats = score_attribution(checked, key, answers)
    stats["bad_answer_line"] = bad
    write_jsonl(run / "final" / "synthetic_opus.jsonl", kept)
    _write_json(run / "reports" / "attrib.json", dict(stats))
    ingest = json.loads((run / "reports" / "ingest.json").read_text())
    planned = sum(json.loads(p.read_text())["n_clusters"] for p in (run / "batches").glob("*.json"))
    clusters_by_family: Counter = Counter()
    seen: set[str] = set()
    for d in kept:
        if d.cluster_id not in seen:
            seen.add(d.cluster_id)
            clusters_by_family[d.family] += 1
    pilot = {
        "planned_clusters": planned,
        "ingested_clusters": ingest.get("clusters_kept", 0),
        "ingested_rows": ingest.get("rows", 0),
        "checked_rows": len(checked),
        "final_rows": len(kept),
        "final_clusters": len(seen),
        "final_clusters_by_family": dict(clusters_by_family),
        "final_rows_by_edit_type": dict(Counter(d.edit_type or "base" for d in kept)),
        "soft_label_share": round(sum(d.soft_gold is not None for d in kept) / max(len(kept), 1), 4),
    }
    _write_json(run / "reports" / "pilot.json", pilot)
    print(json.dumps(pilot, indent=2))
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

    p = sub.add_parser("filter", help="apply licence, contamination and dedupe filters")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-embed", action="store_true", help="skip the embedding filter (tests only)")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("gen-rules", help="generate rule-based contrastive clusters")
    p.add_argument("--family", required=True, choices=sorted(RULE_GENERATORS))
    p.add_argument("--clusters", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gen_rules)
    synth = sub.add_parser("synth", help="Opus-subagent synthetic cluster pipeline")
    ssub = synth.add_subparsers(dest="synth_command", required=True)
    for name, func in [("plan", cmd_synth_plan), ("ingest", cmd_synth_ingest),
                       ("check-prepare", cmd_synth_check_prepare), ("check-score", cmd_synth_check_score),
                       ("attrib-prepare", cmd_synth_attrib_prepare), ("attrib-score", cmd_synth_attrib_score)]:
        p = ssub.add_parser(name)
        p.add_argument("--dir", required=True)
        p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=func)
        if name == "plan":
            p.add_argument("--clusters", type=int, required=True)
            p.add_argument("--per-batch", type=int, default=10)
        if name == "check-prepare":
            p.add_argument("--sheets", type=int, default=3)
        if name in ("check-prepare", "attrib-prepare"):
            p.add_argument("--part-size", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)
