"""Command-line entry point: python -m jeff <command>."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
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
from jeff.filters.shortcut import run_shortcut
from jeff.mix import build_mix, write_mix
from jeff.generators import actions, returns, severity
from jeff.generators.common import EDIT_TYPES
from jeff.generators.llm.attrib import prepare_pairs, render_attrib_prompt, score_attribution
from jeff.generators.llm.check import prepare_sheets, render_checker_prompt, score_answers
from jeff.generators.llm.ingest import ingest_file
from jeff.generators.llm.plan import plan_batches, render_writer_prompt
from jeff.generators.specs import SYNTHETIC_SPECS
from jeff.schema import read_jsonl, write_jsonl
from jeff.sources.base import build, fetch
from jeff.sources import skill_atlas
from jeff.sources.registry import LOCAL_SOURCES, SOURCES


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
        valid_names = {spec.name for spec in SOURCES} | set(LOCAL_SOURCES)
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
            decisions, stats = build(spec, fetch(spec), seed=args.seed, natural=args.natural, pool_scale=args.pool_scale)
            write_jsonl(out / f"{spec.name}.jsonl", decisions)
            all_stats[spec.name] = asdict(stats)
            print(f"{spec.name}: {asdict(stats)}")
            _write_stats_atomically(stats_path, all_stats)
        except Exception as exc:
            print(f"{spec.name}: FAILED: {exc!r}")
            failed.append(spec.name)

    if args.only and skill_atlas.NAME in args.only:
        try:
            stats = skill_atlas.run(out, root=args.skill_atlas_root, seed=args.seed)
            all_stats[skill_atlas.NAME] = stats
            print(f"{skill_atlas.NAME}: {json.dumps(stats)}")
            _write_stats_atomically(stats_path, all_stats)
        except Exception as exc:
            print(f"{skill_atlas.NAME}: FAILED: {exc!r}")
            failed.append(skill_atlas.NAME)

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
    for stale in [*out.glob("*.jsonl"), *out.glob("*.jsonl.tmp"), out / skill_atlas.ATTRIBUTION_FILE]:
        if stale.exists():
            stale.unlink()
    for source, rows in by_source.items():
        write_jsonl(out / f"{source}.jsonl", rows)
    attribution = skill_atlas.attribution_rows({"kept": by_source.get(skill_atlas.NAME, [])})
    if attribution:
        skill_atlas.write_attribution(out, attribution)
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


def cmd_mix(args: argparse.Namespace) -> int:
    recipe_path = Path(args.recipe)
    if not recipe_path.is_file():
        print(f"recipe not found: {recipe_path}")
        return 2
    recipe = json.loads(recipe_path.read_text())
    result = build_mix(recipe, load_fingerprints())
    report = write_mix(result, recipe, args.out)
    print(f"train={report['train']['rows']} dev={report['dev']['rows']} calibration={report['calibration']['rows']}")
    for name, b in report["blocks"].items():
        print(f"  {name}: shortfall={b['shortfall']}")
    return 0


def cmd_shortcut(args: argparse.Namespace) -> int:
    src, out = Path(args.inp), Path(args.out)
    if src.resolve() == out.resolve():
        print("--in and --out must be different directories")
        return 2
    files = sorted(src.glob("*.jsonl"))
    if not files:
        print(f"no *.jsonl files found in {src}")
        return 2
    decisions = [d for path in files for d in read_jsonl(path)]
    kept, report = run_shortcut(decisions, min_rows=args.min_rows, threshold=args.threshold)
    by_source: dict[str, list] = defaultdict(list)
    for d in kept:
        by_source[d.source].append(d)
    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("*.jsonl"), *out.glob("*.jsonl.tmp"), out / skill_atlas.ATTRIBUTION_FILE]:
        if stale.exists():
            stale.unlink()
    for source, rows in by_source.items():
        write_jsonl(out / f"{source}.jsonl", rows)
    attribution = skill_atlas.attribution_rows({"kept": by_source.get(skill_atlas.NAME, [])})
    if attribution:
        skill_atlas.write_attribution(out, attribution)
    _write_stats_atomically(out / "shortcut_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_stats_atomically(path, obj)


def _clear(run: Path, *rels: str) -> None:
    """Remove files/dirs under run; refuses anything resolving outside it."""
    root = run.resolve()
    for rel in rels:
        target = (root / rel).resolve()
        if target == root or root not in target.parents:
            raise ValueError(f"refusing to remove {target}: outside {root}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()


def _coverage(key: dict, answers: list[dict], field: str) -> tuple[int, int]:
    """Count items with a usable answer: a letter that maps to an option, or a known edit type."""
    expected = {k for k in key if k != "_no_base"}
    got: set[str] = set()
    for a in answers:
        if not isinstance(a, dict):
            continue
        item = str(a.get(field, ""))
        if item not in expected:
            continue
        if field == "item_id":
            m = re.search(r"\b([A-F])\b", str(a.get("letter", "")).upper())
            valid = bool(m) and m.group(1) in key[item].get("letters", {})
        else:
            valid = str(a.get("edit_type", "")).strip().lower() in EDIT_TYPES
        if valid:
            got.add(item)
    return len(got), len(expected)


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
    _clear(run, "batches", "ingested.jsonl", "checked.jsonl", "check", "attrib", "private", "final", "reports")
    planned = plan_batches(args.clusters, args.per_batch, args.seed)
    (run / "written").mkdir(parents=True, exist_ok=True)
    live = {b["batch_id"] for b in planned}
    for stale in (run / "written").glob("*.jsonl"):
        if stale.stem not in live:
            stale.unlink()
    for b in planned:
        spec_path = run / "batches" / f"{b['batch_id']}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(b, indent=2))
        out = (run / "written" / f"{b['batch_id']}.jsonl").resolve()
        spec_path.with_suffix(".prompt.md").write_text(render_writer_prompt(b, str(out)))
    print(f"planned {args.clusters} clusters in {len(list((run / 'batches').glob('*.json')))} batches")
    return 0


def cmd_synth_ingest(args) -> int:
    run = Path(args.dir)
    _clear(run, "ingested.jsonl", "checked.jsonl", "final", "check", "attrib", "private",
           "reports/check.json", "reports/attrib.json", "reports/pilot.json")
    rows, stats = [], Counter()
    for spec_path in sorted((run / "batches").glob("*.json")):
        b = json.loads(spec_path.read_text())
        written = run / "written" / f"{b['batch_id']}.jsonl"
        if not written.exists():
            stats["missing_file"] += 1
            continue
        got, s = ingest_file(written, b, run_name=run.resolve().name)
        rows.extend(got)
        stats.update(s)
    write_jsonl(run / "ingested.jsonl", rows)
    stats["rows"] = len(rows)
    _write_json(run / "reports" / "ingest.json", dict(stats))
    print(json.dumps(dict(stats)))
    return 0


def cmd_synth_check_prepare(args) -> int:
    run = Path(args.dir)
    if not (run / "ingested.jsonl").exists():
        print(f"missing {run / 'ingested.jsonl'}; run synth ingest first")
        return 2
    _clear(run, "check", "checked.jsonl", "attrib", "private", "final",
           "reports/check.json", "reports/attrib.json", "reports/pilot.json")
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
    for need in (run / "private" / "check-key.json", run / "ingested.jsonl"):
        if not need.exists():
            print(f"error: missing {need}")
            return 2
    key = json.loads((run / "private" / "check-key.json").read_text())
    answers, bad = _read_answer_lines(sorted((run / "check").glob("answers-*.jsonl")))
    n, m = _coverage(key, answers, "item_id")
    print(f"answered {n} of {m}")
    if n == 0:
        print("error: no answers found")
        return 2
    kept, stats = score_answers(list(read_jsonl(run / "ingested.jsonl")), key, answers)
    stats["bad_answer_line"] = bad
    stats["answered"], stats["expected"] = n, m
    if n < m:
        print(f"WARNING: only {n} of {m} check items answered")
    _clear(run, "attrib", "private/attrib-key.json", "final", "reports/attrib.json", "reports/pilot.json")
    write_jsonl(run / "checked.jsonl", kept)
    _write_json(run / "reports" / "check.json", dict(stats))
    print(json.dumps(dict(stats)))
    return 0


def cmd_synth_attrib_prepare(args) -> int:
    run = Path(args.dir)
    if not (run / "checked.jsonl").exists():
        print(f"missing {run / 'checked.jsonl'}; run synth check-score first")
        return 2
    _clear(run, "attrib", "private/attrib-key.json", "final",
           "reports/attrib.json", "reports/pilot.json")
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
    for need in (run / "private" / "attrib-key.json", run / "checked.jsonl", run / "reports" / "ingest.json"):
        if not need.exists():
            print(f"error: missing {need}")
            return 2
    key = json.loads((run / "private" / "attrib-key.json").read_text())
    answers, bad = _read_answer_lines(sorted((run / "attrib").glob("answers-*.jsonl")))
    n, m = _coverage(key, answers, "pair_id")
    print(f"answered {n} of {m}")
    if n == 0 and m > 0:
        print("error: no answers found")
        return 2
    checked = list(read_jsonl(run / "checked.jsonl"))
    keyed = {v["decision"] for k, v in key.items() if k != "_no_base"}
    has_base = {d.cluster_id for d in checked if d.edit_type is None}
    stale = [d.id for d in checked if d.edit_type is not None and d.cluster_id in has_base and d.id not in keyed]
    if stale:
        print(f"error: attribution key is stale ({len(stale)} checked rows missing, e.g. {stale[0]}); "
              "run synth attrib-prepare again")
        return 2
    kept, stats = score_attribution(checked, key, answers)
    stats["bad_answer_line"] = bad
    stats["answered"], stats["expected"] = n, m
    if n < m:
        print(f"WARNING: only {n} of {m} attribution pairs answered")
    write_jsonl(run / "final" / "synthetic_opus.jsonl", kept)
    _write_json(run / "reports" / "attrib.json", dict(stats))
    ingest = json.loads((run / "reports" / "ingest.json").read_text())
    specs = [json.loads(p.read_text()) for p in (run / "batches").glob("*.json")]
    planned = sum(b["n_clusters"] for b in specs)
    ingested = list(read_jsonl(run / "ingested.jsonl")) if (run / "ingested.jsonl").exists() else []

    def cluster_counts(rows) -> Counter:
        seen_c: dict[str, str] = {}
        for d in rows:
            seen_c.setdefault(d.cluster_id, d.family)
        return Counter(seen_c.values())

    planned_by_family: Counter = Counter()
    for b in specs:
        planned_by_family[b["family"]] += b["n_clusters"]
    ing_c, chk_c, fin_c = cluster_counts(ingested), cluster_counts(checked), cluster_counts(kept)
    families = sorted(set(planned_by_family) | set(ing_c) | set(chk_c) | set(fin_c))
    survival_by_family = {f: {"planned": planned_by_family[f], "ingested_clusters": ing_c[f],
                              "checked_clusters": chk_c[f], "final_clusters": fin_c[f]} for f in families}
    by_edit = {name: Counter(d.edit_type or "base" for d in rows)
               for name, rows in (("ingested", ingested), ("checked", checked), ("final", kept))}
    edit_types = sorted(set().union(*by_edit.values()))
    survival_by_edit_type = {e: {name: by_edit[name][e] for name in by_edit} for e in edit_types}
    clusters_by_family = fin_c
    seen = {d.cluster_id for d in kept}
    pilot = {
        "planned_clusters": planned,
        "ingested_clusters": ingest.get("clusters_kept", 0),
        "ingested_rows": ingest.get("rows", 0),
        "checked_clusters": len({d.cluster_id for d in checked}),
        "checked_rows": len(checked),
        "final_rows": len(kept),
        "final_clusters": len(seen),
        "final_clusters_by_family": dict(clusters_by_family),
        "final_rows_by_edit_type": dict(Counter(d.edit_type or "base" for d in kept)),
        "survival_by_family": survival_by_family,
        "survival_by_edit_type": survival_by_edit_type,
        "soft_label_share": round(sum(d.soft_gold is not None for d in kept) / max(len(kept), 1), 4),
    }
    _write_json(run / "reports" / "pilot.json", pilot)
    print(json.dumps(pilot, indent=2))
    return 0


def _letter_logits_for(args) -> tuple[list[dict], list[list[float]]]:
    from jeff.train import infer
    from jeff.train.data import load_rows
    rows = load_rows(args.data)
    tok, model = infer.load(args.model, args.base)
    return rows, infer.letter_logits(model, tok, rows, max_len=args.max_len, batch_tokens=args.batch_tokens)


def cmd_train(args) -> int:
    from jeff.train.trainer import TrainConfig, train
    print(json.dumps(train(TrainConfig.from_json(args.config)), indent=2))
    return 0


def cmd_eval(args) -> int:
    from jeff.train import metrics, readout
    temps = json.loads(Path(args.calibration).read_text())["temperatures"] if args.calibration else {}
    rows, logits = _letter_logits_for(args)
    preds = [{**{k: r.get(k) for k in ("id", "type", "family", "edit_type", "source", "cluster_id", "letters", "answer")},
              "probs": readout.probs(z, n=len(r["letters"]), temperature=temps.get(r["type"], 1.0))}
             for r, z in zip(rows, logits)]
    report = metrics.evaluate(preds)
    _write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def cmd_calibrate(args) -> int:
    from jeff.format import LETTERS
    from jeff.train.calibrate import fit_temperatures, write_calibration
    rows, logits = _letter_logits_for(args)
    items = [{"type": r["type"], "logits": z, "n": len(r["letters"]), "answer_index": LETTERS.index(r["answer"])}
             for r, z in zip(rows, logits)]
    temps = fit_temperatures(items)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_calibration(args.out, temps, {"model": args.model, "data": args.data, "n": len(items)})
    print(json.dumps(temps))
    return 0


BENCH_DIR = Path("data/bench")


def cmd_bench_build(args) -> int:
    from jeff.bench import BENCHES
    out = Path(args.out or BENCH_DIR / args.name)
    meta = BENCHES[args.name].build(out)
    print(json.dumps(meta, indent=2))
    return 0


def _bench_variant(rows_path: Path) -> str | None:
    m = re.match(r"rows_(\w+)\.chat\.jsonl$", rows_path.name)
    return m.group(1) if m else None


def _overlong_ids(tok, rows: list[dict], max_len: int) -> list[str]:
    """Ids of rows whose prompt is longer than max_len tokens (letter_logits would cut their start)."""
    out = []
    for r in rows:
        ids = tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=True, enable_thinking=False)
        ids = ids["input_ids"] if hasattr(ids, "input_ids") else ids
        if len(ids) > max_len:
            out.append(r["id"])
    return out


def cmd_bench_score(args) -> int:
    from jeff.bench import BENCHES
    from jeff.bench import score as bench_score
    from jeff.bench.common import META_FILE, read_rows
    from jeff.train import infer
    out = Path(args.out)
    if args.name == "decidebench" and (out / "metrics.json").exists() and not args.force:
        print(f"error: {out / 'metrics.json'} exists; the DecideBench test is scored once (pass --force to rerun)")
        return 2
    rows_path = Path(args.rows)
    rows = read_rows(rows_path)
    temps = json.loads(Path(args.calibration).read_text())["temperatures"] if args.calibration else {}
    tok, model = infer.load(args.model, args.base)
    overlong = _overlong_ids(tok, rows, args.max_len)
    if overlong and not args.skip_overlong:
        print(f"error: {len(overlong)} rows are longer than --max-len {args.max_len} tokens "
              f"(e.g. {', '.join(overlong[:5])}); raise --max-len or pass --skip-overlong")
        return 2
    skip = set(overlong)
    rows = [r for r in rows if r["id"] not in skip]
    logits = infer.letter_logits(model, tok, rows, max_len=args.max_len, batch_tokens=args.batch_tokens)
    preds = [bench_score.prediction(r, z, temps.get(r["type"], 1.0)) for r, z in zip(rows, logits)]
    variant = _bench_variant(rows_path)
    report = BENCHES[args.name].metrics(preds, variant=variant)
    meta_path = rows_path.parent / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ov = meta.get("overlap")
    report["run"] = {"model": args.model, "base": args.base, "rows": str(rows_path), "variant": variant,
                     "calibration": args.calibration, "temperatures": temps or None, "max_len": args.max_len,
                     "n_skipped_overlong": len(overlong), "skipped_overlong_ids": overlong,
                     "git_sha": _git_sha(), "bench_revision": meta.get("revision")}
    report["overlap"] = {k: v for k, v in ov.items() if k != "overlapping_ids"} if ov else None
    out.mkdir(parents=True, exist_ok=True)
    keep = ("id", "letters", "probs", "pred", "gold")
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps({k: p[k] for k in keep}, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")
    _write_json(out / "metrics.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("reference", "by_family", "by_category")}, indent=2))
    return 0


def cmd_bench_overlap(args) -> int:
    from jeff.bench.common import META_FILE, iter_rows, overlap, read_rows, write_json
    d = Path(args.dir or BENCH_DIR / args.name)
    files = sorted(d.glob("rows*.chat.jsonl"))
    if not files:
        print(f"no rows*.chat.jsonl in {d}; run jeff bench build --name {args.name} first")
        return 2
    seen: dict[str, dict] = {}
    for f in files:
        for r in read_rows(f):
            seen.setdefault(r["id"], r)
    rep = overlap(list(seen.values()), iter_rows(args.train))
    rep["train"] = args.train
    meta_path = d / META_FILE
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["overlap"] = rep
    write_json(meta_path, meta)
    write_json(d / "overlap.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "overlapping_ids"}, indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jeff")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-public", help="download and convert public datasets")
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--natural", action="store_true", help="sample with natural label priors (calibration)")
    p.add_argument("--pool-scale", type=float, default=1.0)
    p.add_argument("--skill-atlas-root", default=str(skill_atlas.DEFAULT_ROOT),
                   help="local skill-atlas checkout, read only by --only skill_atlas")
    p.set_defaults(func=cmd_build_public)

    p = sub.add_parser("dev", help="write the DecideBench examples pool as the dev set")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_dev)

    p = sub.add_parser("filter", help="apply licence, contamination and dedupe filters")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-embed", action="store_true", help="skip the embedding filter (tests only)")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("shortcut", help="down-weight shortcut-solvable public rows, drop solved synthetic clusters")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--min-rows", type=int, default=50)
    p.set_defaults(func=cmd_shortcut)

    p = sub.add_parser("mix", help="assemble train/dev/calibration splits from a recipe")
    p.add_argument("--recipe", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_mix)

    p = sub.add_parser("gen-rules", help="generate rule-based contrastive clusters")
    p.add_argument("--family", required=True, choices=sorted(RULE_GENERATORS))
    p.add_argument("--clusters", type=int, required=True)
    p.add_argument("--seed", type=int, default=0,
                   help="ids are '<family>:<seed>:<k>'; use a different seed for every run that gets merged")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_gen_rules)
    synth = sub.add_parser("synth", help="Opus-subagent synthetic cluster pipeline")
    ssub = synth.add_subparsers(dest="synth_command", required=True)
    for name, func in [("plan", cmd_synth_plan), ("ingest", cmd_synth_ingest),
                       ("check-prepare", cmd_synth_check_prepare), ("check-score", cmd_synth_check_score),
                       ("attrib-prepare", cmd_synth_attrib_prepare), ("attrib-score", cmd_synth_attrib_score)]:
        p = ssub.add_parser(name)
        p.add_argument("--dir", required=True)
        if name in ("plan", "check-prepare", "attrib-prepare"):
            p.add_argument("--seed", type=int, default=0)
        p.set_defaults(func=func)
        if name == "plan":
            p.add_argument("--clusters", type=int, required=True)
            p.add_argument("--per-batch", type=int, default=25)
        if name == "check-prepare":
            p.add_argument("--sheets", type=int, default=3)
        if name in ("check-prepare", "attrib-prepare"):
            p.add_argument("--part-size", type=int, default=250)

    p = sub.add_parser("train", help="Stage 0 LoRA training from a JSON config")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_train)
    for name, func, hlp in [("eval", cmd_eval, "dev metrics for a trained model"),
                            ("calibrate", cmd_calibrate, "fit per-type temperatures")]:
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--model", required=True, help="LoRA adapter dir or full model dir")
        p.add_argument("--base", help="base model path (required for adapters; tokenizer source)")
        p.add_argument("--data", required=True, help="*.chat.jsonl rows")
        p.add_argument("--out", required=True)
        p.add_argument("--max-len", type=int, default=4096)
        p.add_argument("--batch-tokens", type=int, default=16384)
        if name == "eval":
            p.add_argument("--calibration", help="calibration JSON from jeff calibrate (default T=1)")
        p.set_defaults(func=func)
    from jeff.bench import BENCHES
    bench = sub.add_parser("bench", help="external benchmarks: build rows, score a model, check overlap")
    bsub = bench.add_subparsers(dest="bench_command", required=True)
    p = bsub.add_parser("build", help="download a benchmark and write chat rows (network)")
    p.add_argument("--name", required=True, choices=sorted(BENCHES))
    p.add_argument("--out", help="default data/bench/<name>")
    p.set_defaults(func=cmd_bench_build)
    p = bsub.add_parser("score", help="score a model on benchmark rows (offline)")
    p.add_argument("--name", required=True, choices=sorted(BENCHES))
    p.add_argument("--model", required=True, help="LoRA adapter dir or full model dir")
    p.add_argument("--base", help="base model path (required for adapters; tokenizer source)")
    p.add_argument("--rows", required=True, help="rows*.chat.jsonl from jeff bench build")
    p.add_argument("--calibration", help="calibration JSON from jeff calibrate (default T=1)")
    p.add_argument("--out", required=True, help="directory for predictions.jsonl and metrics.json")
    p.add_argument("--max-len", type=int, default=16384,
                   help="rows longer than this are an error unless --skip-overlong")
    p.add_argument("--skip-overlong", action="store_true",
                   help="leave rows longer than --max-len out of scoring (ids recorded in metrics.json)")
    p.add_argument("--force", action="store_true", help="decidebench: overwrite an existing <out>/metrics.json")
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.set_defaults(func=cmd_bench_score)
    p = bsub.add_parser("overlap", help="share of benchmark states sharing a normalised 8-gram with training")
    p.add_argument("--name", required=True, choices=sorted(BENCHES))
    p.add_argument("--train", required=True, help="training *.chat.jsonl")
    p.add_argument("--dir", help="default data/bench/<name>")
    p.set_defaults(func=cmd_bench_overlap)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)
