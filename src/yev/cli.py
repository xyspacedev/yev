"""Command-line entry point: python -m yev <command>."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from collections import Counter
from pathlib import Path

from yev import decidebench
from yev.filters.contamination import (
    EMBED_MODEL,
    EMBED_THRESHOLD,
    EmbeddingFilter,
    load_fingerprints,
    sentence_transformer_encoder,
)
from yev.filters.pipeline import assert_clean, run_filters
from yev.filters.shortcut import run_shortcut
from yev.mix import build_mix, write_mix
from yev.generators import padding
from yev.generators import actions, drills, invoices, many_options, returns, returns_v2, scales, severity
from yev.generators import agent_triples, invoice_actions, policy_counts
from yev.generators.common import EDIT_TYPES
from yev.generators.llm.attrib import prepare_pairs, render_attrib_prompt, score_attribution
from yev.generators.llm import documents, planned
from yev.generators.llm.targets import PLANNED_FAMILIES
from yev.generators.llm.check import LETTERS, parse_letter, prepare_sheets, render_checker_prompt, score_answers
from yev.generators.llm.ingest import ingest_file
from yev.generators.llm.plan import plan_batches, render_writer_prompt
from yev.generators.specs import SYNTHETIC_SPECS
from yev.schema import read_jsonl, write_jsonl
from yev.sources.base import build, fetch
from yev.sources import skill_atlas
from yev.sources.registry import LOCAL_SOURCES, SOURCES


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
    from yev import provenance
    return provenance.git_sha()


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
            all_stats[spec.name] = stats.as_dict()
            print(f"{spec.name}: {stats.as_dict()}")
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


RULE_GENERATORS = {"returns": returns.generate, "actions": actions.generate, "severity": severity.generate,
                   "returns_v2": returns_v2.generate, "date_number_drills": drills.generate,
                   "numbered_scales": scales.generate, "many_options": many_options.generate,
                   "invoice_ledger": invoices.generate, "policy_counts": policy_counts.generate,
                   "payment_terms": invoice_actions.generate_payment_terms,
                   "invoice_action": invoice_actions.generate_invoice_action, "agent_triples": agent_triples.generate}


def cmd_gen_rules(args: argparse.Namespace) -> int:
    rows = RULE_GENERATORS[args.family](args.clusters, args.seed)
    n = write_jsonl(args.out, rows)
    print(f"{args.family}: {n} rows in {len({r.cluster_id for r in rows})} clusters -> {args.out}")
    return 0


def cmd_pad(args: argparse.Namespace) -> int:
    lo, _, hi = args.target_tokens.partition(":")
    target = (int(lo), int(hi))
    rows = list(read_jsonl(args.inp))
    docs = list(read_jsonl(args.distractors))
    eval_docs = [d.id for d in docs if d.split in padding.EVAL_SPLITS]
    if eval_docs:
        print(f"distractor docs must not come from dev/calibration rows: {eval_docs[:3]}")
        return 2
    families = {d.family for d in docs}
    clash = [r.id for r in rows if r.family in families]
    if clash:
        print(f"refusing to pad rows of the same family as the distractor docs: {clash[:3]}")
        return 2
    states = list(dict.fromkeys(d.state for d in docs))
    try:
        skipped: list[tuple[str, str]] = []
        out = padding.pad(rows, states, target, args.seed, allow_heuristic=args.allow_heuristic, skipped=skipped)
    except padding.PaddingError as e:
        print(str(e))
        return 2
    n = write_jsonl(args.out, out)
    if skipped:
        print(f"pad: skipped {dict(Counter(r for _, r in skipped))}")
    print(f"pad: {n}/{len(rows)} rows padded to {target[0]}-{target[1]} tokens -> {args.out}")
    return 0


def cmd_filter(args: argparse.Namespace) -> int:
    src = Path(args.inp)
    out_file = Path(args.out_file) if args.out_file else None
    out = out_file.parent if out_file else Path(args.out)
    if out_file is None and src.resolve() == out.resolve():
        print("--in and --out must be different directories")
        return 2
    if out_file is not None and out_file.resolve().parent == src.resolve():
        print("--out-file must not sit in the --in directory")
        return 2
    if out_file is not None and out_file.exists() and not args.force:
        print(f"{out_file} exists; pass --force to overwrite it")
        return 2
    files = sorted(src.glob("*.jsonl"))
    if not files:
        print(f"no *.jsonl files found in {src}")
        return 2
    decisions = [d for path in files for d in read_jsonl(path)]
    fp = load_fingerprints()
    bench_fp = None
    if args.benchmarks:
        from yev.filters.benchmarks import BenchmarkFingerprints
        if not Path(args.benchmarks).is_file():
            print(f"benchmark fingerprints not found: {args.benchmarks}; run yev bench fingerprint first")
            return 2
        bench_fp = BenchmarkFingerprints.load(args.benchmarks)
    embed = None if args.no_embed else EmbeddingFilter(fp.states, sentence_transformer_encoder())
    kept, report = run_filters(decisions, fp, embed, {s.name: s for s in [*SOURCES, *SYNTHETIC_SPECS]},
                               bench_fp=bench_fp)
    assert_clean(kept, fp, bench_fp)
    by_source: dict[str, list] = defaultdict(list)
    for d in kept:
        by_source[d.source].append(d)
    meta = {
        "embedding": None if args.no_embed else EMBED_MODEL,
        "threshold": EMBED_THRESHOLD,
        "decidebench_revision": decidebench.REVISION,
        "benchmarks": None if bench_fp is None else {"path": args.benchmarks, "names": bench_fp.names(),
                                                     "built": bench_fp.meta.get("built")},
        "git_sha": _git_sha(),
    }
    if out_file is not None:
        # One pool file: all kept rows go to exactly this path; nothing else in its directory is touched
        # except its own report beside it (<name>.filter_report.json).
        if by_source.get(skill_atlas.NAME):
            print(f"--out-file cannot carry {skill_atlas.NAME} rows (they need their attribution file); use --out")
            return 2
        write_jsonl(out_file, kept)
        report = {"meta": {**meta, "out_file": str(out_file)}, "sources": report}
        _write_stats_atomically(out_file.with_name(out_file.stem + ".filter_report.json"), report)
        print(json.dumps(report, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for stale in [*out.glob("*.jsonl"), *out.glob("*.jsonl.tmp"), out / skill_atlas.ATTRIBUTION_FILE]:
        if stale.exists():
            stale.unlink()
    for source, rows in by_source.items():
        write_jsonl(out / f"{source}.jsonl", rows)
    attribution = skill_atlas.attribution_rows({"kept": by_source.get(skill_atlas.NAME, [])})
    if attribution:
        skill_atlas.write_attribution(out, attribution)
    report = {"meta": meta, "sources": report}
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
            letters = key[item].get("letters", {})
            valid = parse_letter(a.get("letter", ""), len(letters)) in letters
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


def _is_doc_run(run: Path) -> bool:
    """A document run or a planned run: both use the document steps (check, nodoc, diff) unchanged."""
    return any(documents.is_doc_spec(s) or planned.is_planned_spec(s) for s in _batch_specs(run))


PLANNED_SHEETS = 1  # plan 6: a planned question survives only if one blind Opus sheet returns the gold


def _is_planned_run(run: Path) -> bool:
    return any(planned.is_planned_spec(s) for s in _batch_specs(run))


def _batch_specs(run: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run / "batches").glob("*.json"))]


def _refuse_doc_run(run: Path, what: str) -> bool:
    if _is_doc_run(run):
        print(f"error: {run} is a document run; {what}")
        return True
    return False


# check keys under private/; ingest and check-prepare clear these, leaving private/plan/ (planned runs) intact
PRIVATE_KEYS = ("private/check-key.json", "private/attrib-key.json", "private/nodoc-key.json",
                "private/diff-key.json")


def _clear_keys(run: Path) -> None:
    """Remove the check keys; private/ itself goes too when nothing else (a planned run's plan) is left in it."""
    _clear(run, *PRIVATE_KEYS)
    private = run / "private"
    if private.is_dir() and not any(private.iterdir()):
        private.rmdir()


def _start_doc_run(run: Path, batches: list[dict], prompt, public=None) -> None:
    """Run setup shared by doc-plan and planned-plan: clear the earlier plan and everything derived from it, then
    write batches/<id>.json and batches/<id>.prompt.md per batch (writer output goes to written/<id>.jsonl).

    With `public` (planned runs) batches/ gets only public(batch), the full plan goes to private/plan/, and all of
    written/ is cleared: planned outputs are bound to the plan, so a stale file would be relabelled with new golds.
    Without it (document runs) only written/ files of batches no longer planned are removed."""
    _clear(run, "batches", "ingested.jsonl", "checked.jsonl", "check", "attrib", "private", "final", "reports",
           *DOC_LATER_FILES)
    if public is not None:
        _clear(run, "written")
    (run / "written").mkdir(parents=True, exist_ok=True)
    live = {b["batch_id"] for b in batches}
    for stale in (run / "written").glob("*.jsonl"):
        if stale.stem not in live:
            stale.unlink()
    for b in batches:
        spec_path = run / "batches" / f"{b['batch_id']}.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(public(b) if public else b, indent=2))
        if public is not None:
            path = planned.plan_path(run, b["batch_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(b, indent=2))
        out = (run / "written" / f"{b['batch_id']}.jsonl").resolve()
        spec_path.with_suffix(".prompt.md").write_text(prompt(b, str(out)))


def _doc_counter(args, step: str, what: str):
    """(count, method) for doc-ingest / planned-ingest, or None after printing why the tokenizer is unavailable."""
    try:
        return documents.load_doc_counter(allow_heuristic=args.allow_heuristic)
    except padding.TokenizerUnavailable as e:
        print(f"error: {e}")
        print(f"{step} measures {what} with the real Qwen3.5 tokenizer (run with --extra train and the "
              "tokenizer in ~/.cache/huggingface); pass --allow-heuristic to estimate with len/4 x "
              f"{padding.HEURISTIC_FACTOR} instead.")
        return None


def _clear_for_ingest(run: Path) -> None:
    _clear(run, "ingested.jsonl", "checked.jsonl", "final", "check", "attrib",
           "reports/check.json", "reports/pilot.json", *DOC_LATER_FILES)
    _clear_keys(run)


def cmd_synth_doc_plan(args) -> int:
    run = Path(args.dir)
    batches = documents.plan_doc_batches(args.docs, args.per_batch, args.seed)
    _start_doc_run(run, batches, documents.doc_writer_prompt)
    n_inj = sum(len(b["injection_docs"]) for b in batches)
    print(f"planned {args.docs} documents in {len(batches)} batches ({n_inj} with an injected instruction)")
    return 0


def cmd_synth_doc_ingest(args) -> int:
    run = Path(args.dir)
    if not any(documents.is_doc_spec(s) for s in _batch_specs(run)):
        hint = "it is a planned run; use synth planned-ingest" if _is_planned_run(run) else "run synth doc-plan first"
        print(f"error: {run} has no document batches; {hint}")
        return 2
    got = _doc_counter(args, "doc-ingest", "documents")
    if got is None:
        return 2
    count, method = got
    _clear_for_ingest(run)
    stats, dropped = Counter(), []
    rows = documents.ingest_docs(run, count=count, stats=stats, dropped=dropped, method=method)
    write_jsonl(run / "ingested.jsonl", rows)
    _write_json(run / "reports" / "ingest.json", {**dict(stats), "token_count": method})
    _write_json(run / "reports" / "ingest-dropped.json", dropped)
    print(json.dumps({**dict(stats), "token_count": method}))
    return 0


def cmd_synth_planned_plan(args) -> int:
    family = PLANNED_FAMILIES.get(args.family)
    if family is None:
        print(f"error: unknown planned family {args.family!r}; known: {', '.join(sorted(PLANNED_FAMILIES)) or 'none'}")
        return 2
    run = Path(args.dir)
    batches = planned.plan_batches(family, args.groups, args.seed)
    _start_doc_run(run, batches, planned.writer_prompt, public=planned.public_spec)
    n_states = sum(len(g["states"]) for b in batches for g in b["groups"])
    print(f"planned {args.groups} {family.name} groups ({n_states} states) in {len(batches)} batches")
    return 0


def cmd_synth_planned_ingest(args) -> int:
    run = Path(args.dir)
    if not _is_planned_run(run):
        print(f"error: {run} has no planned batches; run synth planned-plan first")
        return 2
    got = _doc_counter(args, "planned-ingest", "states")
    if got is None:
        return 2
    count, method = got
    _clear_for_ingest(run)
    rows, report = planned.ingest_planned(run, count=count, method=method)
    dropped = report.pop("dropped", [])
    write_jsonl(run / "ingested.jsonl", rows)
    _write_json(run / "reports" / "ingest.json", report)
    _write_json(run / "reports" / "ingest-dropped.json", dropped)
    print(json.dumps(report))
    return 0


def cmd_synth_ingest(args) -> int:
    run = Path(args.dir)
    if _refuse_doc_run(run, "use synth doc-ingest"):
        return 2
    _clear(run, "ingested.jsonl", "checked.jsonl", "final", "check", "attrib",
           "reports/check.json", "reports/attrib.json", "reports/pilot.json")
    _clear_keys(run)
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
    _clear_keys(run)
    _clear(run, "check", "checked.jsonl", "attrib", "final",
           "reports/check.json", "reports/attrib.json", "reports/pilot.json", *DOC_LATER_FILES)
    doc_run = _is_doc_run(run)
    part_size = args.doc_part_size if doc_run else args.part_size
    # planned runs: one blind sheet must return the planner's gold (score_answers then needs 1 vote)
    sheets = args.sheets if args.sheets is not None else (PLANNED_SHEETS if _is_planned_run(run) else 3)
    parts, key = prepare_sheets(list(read_jsonl(run / "ingested.jsonl")), n_sheets=sheets,
                                part_size=part_size, seed=args.seed)
    for name, items in parts:
        sheet = (run / "check" / f"{name}.jsonl").resolve()
        _dump_items(sheet, items)
        answers = sheet.with_name(f"answers-{name}.jsonl")
        widest = max(len(it["options"]) for it in items)
        prompt = render_checker_prompt(str(sheet), str(answers), max_letter=LETTERS[max(widest, 6) - 1],
                                       documents=doc_run)
        sheet.with_suffix(".prompt.md").write_text(prompt)
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
    if _is_doc_run(run):
        if _doc_later_steps_block(run, "check-score", args.force):
            return 2
        return _doc_check_score(run, key, answers, bad, n, m)
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


def _doc_check_score(run: Path, key: dict, answers: list[dict], bad: int, n: int, m: int) -> int:
    """Documents skip attribution (Plan Ruling 2): check-score writes the final rows and the pilot report."""
    ingested = list(read_jsonl(run / "ingested.jsonl"))
    survivors: list = []
    is_planned = _is_planned_run(run)
    score = planned.score_planned_answers if is_planned else documents.score_doc_answers
    kept, stats = score(ingested, key, answers, survivors_out=survivors)
    stats["bad_answer_line"] = bad
    stats["answered"], stats["expected"] = n, m
    if n < m:
        print(f"WARNING: only {n} of {m} check items answered")
    write_jsonl(run / "checked.jsonl", kept)
    write_jsonl(run / "final" / "synthetic_docs.jsonl", kept)
    _write_json(run / "reports" / "check.json", dict(stats))
    specs = [json.loads(p.read_text()) for p in sorted((run / "batches").glob("*.json"))]
    ingest = json.loads((run / "reports" / "ingest.json").read_text()) if (run / "reports" / "ingest.json").exists() else {}
    if is_planned:
        pilot = planned.planned_pilot_report(planned.load_plans(run), ingest, ingested, survivors, kept)
    else:
        pilot = documents.doc_pilot_report([b for b in specs if documents.is_doc_spec(b)], ingest, ingested,
                                           survivors, kept)
    _write_json(run / "reports" / "pilot.json", pilot)
    print(json.dumps(dict(stats)))
    print(json.dumps(pilot, indent=2))
    return 0


# --- Task 7b: no-document control and difficulty signal (document runs only) ------------------------

DOC_STEPS = ("check-score", "nodoc-prepare", "nodoc-score", "diff-prepare", "diff-score")
DOC_STEP_ORDER = " -> ".join(DOC_STEPS)
# step -> (the file whose presence means the step has run, the files the step owns)
DOC_STEP_FILES = {
    "check-score": ("checked.jsonl", ()),
    "nodoc-prepare": ("private/nodoc-key.json", ("nodoc", "private/nodoc-key.json")),
    "nodoc-score": ("nodoc-checked.jsonl", ("nodoc-checked.jsonl", "reports/nodoc.json")),
    "diff-prepare": ("private/diff-key.json", ("diff", "private/diff-key.json")),
    "diff-score": ("reports/difficulty.json", ("reports/difficulty.json",)),
}
DOC_LATER_FILES = tuple(f for step in DOC_STEPS[1:] for f in DOC_STEP_FILES[step][1])
PILOT_SECTION = {"nodoc-score": "nodoc", "diff-score": "difficulty"}


def _doc_step_done(run: Path, step: str) -> bool:
    return (run / DOC_STEP_FILES[step][0]).exists()


def _doc_later_steps_block(run: Path, step: str, force: bool) -> bool:
    """True (after printing why) when a later step has already run; with force, discard the later steps."""
    later = [s for s in DOC_STEPS[DOC_STEPS.index(step) + 1:] if _doc_step_done(run, s)]
    if not later:
        return False
    if not force:
        print(f"error: {', '.join(later)} already ran in {run}; re-running {step} would leave them stale "
              f"(order: {DOC_STEP_ORDER}). Pass --force to discard {', '.join(later)} and run {step} again.")
        return True
    _clear(run, *(f for s in DOC_STEPS[DOC_STEPS.index(step) + 1:] for f in DOC_STEP_FILES[s][1]))
    pilot_path = run / "reports" / "pilot.json"
    if pilot_path.exists():
        pilot = json.loads(pilot_path.read_text())
        for s in DOC_STEPS[DOC_STEPS.index(step) + 1:]:
            pilot.pop(PILOT_SECTION.get(s, ""), None)
        _write_json(pilot_path, pilot)
    print(f"--force: discarded {', '.join(later)}")
    return False


def _doc_step_guard(run: Path, step: str, force: bool) -> bool:
    """True (after printing why) when step must not run now: not a document run, the previous step has not run,
    or a later step already ran (unless force)."""
    if not (run / "batches").is_dir() or not _is_doc_run(run):
        print(f"error: {run} is not a document run; synth {step} only works on runs planned with synth doc-plan "
              "or synth planned-plan")
        return True
    prev = DOC_STEPS[DOC_STEPS.index(step) - 1]
    if not _doc_step_done(run, prev):
        print(f"error: run synth {prev} before synth {step} (order: {DOC_STEP_ORDER}); "
              f"missing {run / DOC_STEP_FILES[prev][0]}")
        return True
    return _doc_later_steps_block(run, step, force)


def _write_doc_final(run: Path, rows, drop: tuple[str, ...] = (), section: str | None = None,
                     report: dict | None = None) -> None:
    """Rewrite final/synthetic_docs.jsonl in place and keep reports/pilot.json's sections in step with it."""
    write_jsonl(run / "final" / "synthetic_docs.jsonl", rows)
    pilot_path = run / "reports" / "pilot.json"
    pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
    for name in drop:
        pilot.pop(name, None)
    pilot["final_questions"] = len(rows)  # the top-level counts always describe the current final/
    pilot["final_docs"] = len({d.cluster_id for d in rows})
    if section:
        pilot[section] = report
    _write_json(pilot_path, pilot)


def _doc_prepare(args, step: str, sub: str, source: str, prepare, render) -> int:
    run = Path(args.dir)
    if _doc_step_guard(run, step, args.force):
        return 2
    answered = sorted((run / sub).glob("answers-*.jsonl"))
    if answered and not args.force:
        print(f"error: {run / sub} already holds {len(answered)} answer file(s); re-running {step} would discard "
              f"them. Pass --force to start {sub}/ again.")
        return 2
    rows = list(read_jsonl(run / source))
    _clear(run, *DOC_STEP_FILES[step][1])
    # the final file goes back to this step's input, so final/ always matches the last scoring step
    _write_doc_final(run, rows, drop=("nodoc", "difficulty") if step == "nodoc-prepare" else ("difficulty",))
    parts, key = prepare(rows, part_size=args.part_size, seed=args.seed)
    for name, items in parts:
        sheet = (run / sub / f"{name}.jsonl").resolve()
        _dump_items(sheet, items)
        widest = max(len(it["options"]) for it in items)
        sheet.with_suffix(".prompt.md").write_text(
            render(str(sheet), str(sheet.with_name(f"answers-{name}.jsonl")), LETTERS[max(widest, 6) - 1]))
    _write_json(run / DOC_STEP_FILES[step][0], key)
    print(f"{len(parts)} {sub} parts, {sum(len(i) for _, i in parts)} items, {len(rows)} questions")
    return 0


def _doc_answers(run: Path, sub: str, key_rel: str) -> tuple[dict, list[dict]] | None:
    key = json.loads((run / key_rel).read_text())
    answers, bad = _read_answer_lines(sorted((run / sub).glob("answers-*.jsonl")))
    n, m = _coverage(key, answers, "item_id")
    print(f"answered {n} of {m}")
    if bad:
        print(f"WARNING: {bad} unreadable answer line(s)")
    if n == 0:
        print(f"error: no answers found in {run / sub}")
        return None
    if n < m:
        print(f"WARNING: only {n} of {m} {sub} items answered")
    return key, answers


def cmd_synth_nodoc_prepare(args) -> int:
    return _doc_prepare(args, "nodoc-prepare", "nodoc", "checked.jsonl", documents.prepare_nodoc,
                        documents.render_nodoc_prompt)


def cmd_synth_nodoc_score(args) -> int:
    run = Path(args.dir)
    if _doc_step_guard(run, "nodoc-score", args.force):
        return 2
    got = _doc_answers(run, "nodoc", "private/nodoc-key.json")
    if got is None:
        return 2
    kept, report = documents.score_nodoc(list(read_jsonl(run / "checked.jsonl")),
                                         list(read_jsonl(run / "ingested.jsonl")), *got)
    write_jsonl(run / "nodoc-checked.jsonl", kept)
    _write_json(run / "reports" / "nodoc.json", report)
    _write_doc_final(run, kept, drop=("difficulty",), section="nodoc", report=report)
    print(json.dumps(report, indent=2))
    return 0


def cmd_synth_diff_prepare(args) -> int:
    def render(sheet, answers, max_letter):
        return render_checker_prompt(sheet, answers, max_letter=max_letter, documents=True)
    return _doc_prepare(args, "diff-prepare", "diff", "nodoc-checked.jsonl", documents.prepare_diff, render)


def cmd_synth_diff_score(args) -> int:
    run = Path(args.dir)
    if _doc_step_guard(run, "diff-score", args.force):
        return 2
    got = _doc_answers(run, "diff", "private/diff-key.json")
    if got is None:
        return 2
    rows, report = documents.score_difficulty(list(read_jsonl(run / "nodoc-checked.jsonl")), *got)
    _write_json(run / "reports" / "difficulty.json", report)
    _write_doc_final(run, rows, section="difficulty", report=report)
    print(json.dumps(report, indent=2))
    return 0


def cmd_synth_attrib_prepare(args) -> int:
    run = Path(args.dir)
    if _refuse_doc_run(run, "documents skip edit attribution; check-score already wrote final/synthetic_docs.jsonl"):
        return 2
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
    if _refuse_doc_run(run, "documents skip edit attribution; check-score already wrote final/synthetic_docs.jsonl"):
        return 2
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


def _n_max_for(args) -> int:
    """--n-max, else the run config's n_max saved beside the model (train_config.json), else 6."""
    from yev.train.data import N_MAX
    if args.n_max is not None:
        return args.n_max
    saved = Path(args.model) / "train_config.json"
    if saved.exists():
        return int(json.loads(saved.read_text()).get("n_max", N_MAX))
    return N_MAX


def _letter_logits_for(args, prepare=None) -> tuple[list[dict], list[list[float]], list[str]] | None:
    """Rows kept, their letter logits, and the skipped over-long ids; None when over-long rows, or rows
    with more options than n_max, are refused. Sets args.n_max to the resolved value. `prepare` maps
    the loaded rows to the rows actually scored (the position-bias check)."""
    from yev.train import infer
    from yev.train.data import load_rows
    rows = load_rows(args.data)
    if prepare is not None:
        rows = prepare(rows)
    args.n_max = _n_max_for(args)
    wide = [r["id"] for r in rows if len(r["letters"]) > args.n_max]
    if wide:
        print(f"error: {len(wide)} rows have more than n_max = {args.n_max} options "
              f"(e.g. {', '.join(wide[:5])}); pass --n-max")
        return None
    tok, model = infer.load(args.model, args.base)
    overlong = _overlong_ids(tok, rows, args.max_len)
    if overlong and not args.skip_overlong:
        print(f"error: {len(overlong)} rows are longer than --max-len {args.max_len} tokens "
              f"(e.g. {', '.join(overlong[:5])}); raise --max-len or pass --skip-overlong")
        return None
    skip = set(overlong)
    rows = [r for r in rows if r["id"] not in skip]
    return rows, infer.letter_logits(model, tok, rows, max_len=args.max_len, batch_tokens=args.batch_tokens,
                                     n_letters=args.n_max), overlong


def cmd_train(args) -> int:
    from yev.train.trainer import TrainConfig, train
    cfg = TrainConfig.from_json(args.config)
    out = Path(cfg.out_dir)
    if not args.force and (out / "final").exists() and (out / "train_summary.json").exists():
        try:
            done = json.loads((out / "train_summary.json").read_text())
        except ValueError:
            done = {}
        if done.get("steps") is not None and done.get("steps") == done.get("total_steps"):
            print(f"{out} already holds a finished run ({done['steps']}/{done['total_steps']} steps); "
                  "nothing touched (pass --force to train again)")
            return 0
    print(json.dumps(train(cfg), indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from yev.serve.app import create_app
    from yev.serve.engine import Engine

    engine = Engine(args.model, args.base, args.calibration, max_len=args.max_len,
                    batch_tokens=args.batch_tokens, prefix_cache=args.prefix_cache,
                    min_prefix_tokens=args.min_prefix_tokens)
    uvicorn.run(create_app(engine, model_name=args.model_name), host=args.host, port=args.port)
    return 0


def _eval_position_bias(args, temps: dict) -> int:
    """yev eval --shuffle-options K: re-ask each row with >= --min-options options in its original
    order and K seeded permutations; compare the per-key distributions (spec §5)."""
    from yev.train import position_bias as pb
    from yev.train import readout
    skipped = {"score": 0, "few_options": 0}

    def prepare(rows):
        out = []
        for r in rows:
            if r["type"] == "score":  # a scale's order is its meaning; training never shuffles it
                skipped["score"] += len(r["letters"]) >= args.min_options
                continue
            if len(r["letters"]) < args.min_options:
                skipped["few_options"] += 1
                continue
            for j, order in enumerate(pb.orders(r["id"], len(r["letters"]), args.shuffle_options, args.seed)):
                out.append({**pb.permute_row(r, order), "id": f"{r['id']}#order{j}", "_base": r["id"], "_order": j})
        return out

    got = _letter_logits_for(args, prepare)
    if got is None:
        return 2
    rows, logits, overlong = got
    n_orders = args.shuffle_options + 1
    by_base: dict[str, dict] = {}
    for r, z in zip(rows, logits):
        p = readout.probs(z, n=len(r["letters"]), temperature=temps.get(r["type"], 1.0))
        g = by_base.setdefault(r["_base"], {"n_options": len(r["letters"]), "gold": r["letters"][r["answer"]],
                                            "orders": [None] * n_orders})
        g["orders"][r["_order"]] = {r["letters"][L]: q for L, q in zip(r["letters"], p)}
    groups = [g for g in by_base.values() if all(o is not None for o in g["orders"])]
    overlong_base = sorted({i.rsplit("#order", 1)[0] for i in overlong})
    rep = pb.report(groups)
    rep.update(n_skipped_score=skipped["score"], n_skipped_few_options=skipped["few_options"],
               n_skipped_overlong=len(overlong_base))
    rep["run"] = {"model": args.model, "data": args.data, "max_len": args.max_len, "n_max": args.n_max,
                  "calibration": args.calibration, "shuffle_options": args.shuffle_options,
                  "min_options": args.min_options, "seed": args.seed, "skipped_overlong_ids": overlong_base}
    _write_json(Path(args.out), rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "run"}, indent=2))
    return 0


def cmd_eval(args) -> int:
    from yev.train import metrics, readout
    temps = json.loads(Path(args.calibration).read_text())["temperatures"] if args.calibration else {}
    if args.shuffle_options:
        return _eval_position_bias(args, temps)
    got = _letter_logits_for(args)
    if got is None:
        return 2
    rows, logits, overlong = got
    preds = [{**{k: r.get(k) for k in ("id", "type", "family", "edit_type", "source", "cluster_id", "letters", "answer")},
              "probs": readout.probs(z, n=len(r["letters"]), temperature=temps.get(r["type"], 1.0))}
             for r, z in zip(rows, logits)]
    report = metrics.evaluate(preds)
    report["run"] = {"model": args.model, "data": args.data, "max_len": args.max_len, "n_max": args.n_max,
                     "n_skipped_overlong": len(overlong), "skipped_overlong_ids": overlong}
    _write_json(Path(args.out), report)
    print(json.dumps(report, indent=2))
    return 0


def cmd_calibrate(args) -> int:
    from yev.format import LETTERS
    from yev.train.calibrate import fit_temperatures, write_calibration
    got = _letter_logits_for(args)
    if got is None:
        return 2
    rows, logits, overlong = got
    items = [{"type": r["type"], "logits": z, "n": len(r["letters"]), "answer_index": LETTERS.index(r["answer"])}
             for r, z in zip(rows, logits)]
    temps = fit_temperatures(items)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_calibration(args.out, temps, {"model": args.model, "data": args.data, "n": len(items), "n_max": args.n_max,
                                              "n_skipped_overlong": len(overlong), "skipped_overlong_ids": overlong})
    print(json.dumps(temps))
    return 0


BENCH_DIR = Path("data/bench")


def cmd_bench_build(args) -> int:
    from yev.bench import BENCHES
    out = Path(args.out or BENCH_DIR / args.name)
    meta = BENCHES[args.name].build(out)
    print(json.dumps(meta, indent=2))
    return 0


def _bench_variant(rows_path: Path) -> str | None:
    m = re.match(r"rows_(\w+)\.chat\.jsonl$", rows_path.name)
    return m.group(1) if m else None


def _overlong_ids(tok, rows: list[dict], max_len: int) -> list[str]:
    """Ids of rows whose prompt is longer than max_len tokens (letter_logits would cut their start)."""
    from yev.train.data import overlong_ids
    return overlong_ids(tok, rows, max_len)


def cmd_bench_score(args) -> int:
    from yev.bench import BENCHES
    from yev.bench import score as bench_score
    from yev.bench.common import META_FILE, read_rows
    from yev.train import infer
    out = Path(args.out)
    rows_path = Path(args.rows)
    sentinel = None
    if args.stage is not None and not re.fullmatch(r"[A-Za-z0-9_]+", args.stage):
        print(f"error: --stage {args.stage!r} must be letters, digits or _ (it names a sentinel file)")
        return 2
    if args.name == "decidebench":
        # Stage 0 kept the bare .scored-<variant>; a later stage gets its own once-guard
        stage = f"{args.stage}-" if args.stage else ""
        sentinel = BENCH_DIR / "decidebench" / f".scored-{stage}{_bench_variant(rows_path) or 'unknown'}"
        for guard in (out / "metrics.json", sentinel):
            if guard.exists() and not args.force:
                print(f"error: {guard} exists; the DecideBench test is scored once (pass --force to rerun)")
                return 2
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
                     "git_sha": _git_sha(), "bench_revision": meta.get("revision"), "stage": args.stage}
    report["overlap"] = {k: v for k, v in ov.items() if k != "overlapping_ids"} if ov else None
    out.mkdir(parents=True, exist_ok=True)
    keep = ("id", "letters", "probs", "pred", "gold")
    (out / "predictions.jsonl").write_text(
        "".join(json.dumps({k: p[k] for k in keep}, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8")
    _write_json(out / "metrics.json", report)
    if sentinel is not None:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(json.dumps({"out": str(out), "git_sha": _git_sha(), "stage": args.stage}) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("reference", "by_family", "by_category")}, indent=2))
    return 0


def cmd_bench_overlap(args) -> int:
    from yev.bench.common import META_FILE, iter_rows, overlap, read_rows, write_json
    d = Path(args.dir or BENCH_DIR / args.name)
    files = sorted(d.glob("rows*.chat.jsonl"))
    if not files:
        print(f"no rows*.chat.jsonl in {d}; run yev bench build --name {args.name} first")
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


def cmd_bench_fingerprint(args) -> int:
    from datetime import datetime, timezone

    from yev.filters import benchmarks as bm
    bench_dir = Path(args.bench_dir)
    sources = bm.chat_row_sources(bench_dir) if bench_dir.is_dir() else {}
    if not sources:
        print(f"no {bench_dir}/*/rows*.chat.jsonl; run yev bench build first")
        return 2
    hub = {} if args.no_hub else bm.hub_sources()
    clash = set(sources) & set(hub)
    if clash:
        print(f"benchmark names used twice: {sorted(clash)}")
        return 2
    meta = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_sha": _git_sha(),
            "bench_dir": str(bench_dir), "hub": {} if args.no_hub else bm.hub_meta()}
    fp = bm.build_benchmark_fingerprints({**sources, **hub}, meta=meta)
    fp.save(args.out)
    summary = {n: {"items": fp.items[n], "ngrams": len(fp.grams[n]), "threshold": fp.thresholds[n],
                   "stoplisted": fp.meta["stoplist"]["dropped"][n],
                   "short_items": fp.meta["short_items"]["counts"][n]} for n in fp.names()}
    print(json.dumps(summary, indent=2))
    return 0


def cmd_gate(args) -> int:
    """Stage 1 publish gate; exit 0 when PUBLISHABLE, 1 when not (a missing input is FAIL)."""
    from yev import gate
    rep = gate.evaluate(stage0_dev=args.stage0_dev, stage1_dev=args.stage1_dev, workflowevals=args.workflowevals,
                        bench=args.bench, decidebench=args.decidebench, decidebench_stage=args.decidebench_stage)
    rep["git_sha"] = _git_sha()
    _write_json(Path(args.out), rep)
    print(gate.table(rep))
    return 0 if rep["publishable"] else 1


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yev")
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
    dest = p.add_mutually_exclusive_group(required=True)
    dest.add_argument("--out", help="directory: cleared of *.jsonl, then one <source>.jsonl per source")
    dest.add_argument("--out-file", help="write every kept row to exactly this file (e.g. "
                                         "data/stage1c-pool/<family>.jsonl); nothing else is deleted, and its "
                                         "report goes to <name>.filter_report.json beside it")
    p.add_argument("--force", action="store_true", help="with --out-file: overwrite the file if it exists")
    p.add_argument("--no-embed", action="store_true", help="skip the embedding filter (tests only)")
    p.add_argument("--benchmarks", help="fingerprints JSON from yev bench fingerprint; drops a row (reason "
                                        "benchmark:<name>) that shares 3 distinct 8-grams with one benchmark item "
                                        "(all of them if the item has fewer; any one for DecideBench), or whose state, "
                                        "a line of it (label prefix stripped) or a quoted span (3+ tokens) equals an "
                                        "item of up to 9 tokens, or whose state contains one of 6-7 tokens")
    p.set_defaults(func=cmd_filter)

    p = sub.add_parser("pad", help="embed short decisions in long distractor documents (family long_padding)")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--distractors", required=True, help="Decision JSONL; distinct states are the distractor docs")
    p.add_argument("--out", required=True)
    p.add_argument("--target-tokens", default="2000:10000", help="LO:HI tokens of the padded state")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-heuristic", action="store_true",
                   help="if the Qwen tokenizer is not cached, estimate tokens as len/4 x 2.0 (warns)")
    p.set_defaults(func=cmd_pad)

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
                       ("attrib-prepare", cmd_synth_attrib_prepare), ("attrib-score", cmd_synth_attrib_score),
                       ("doc-plan", cmd_synth_doc_plan), ("doc-ingest", cmd_synth_doc_ingest),
                       ("nodoc-prepare", cmd_synth_nodoc_prepare), ("nodoc-score", cmd_synth_nodoc_score),
                       ("diff-prepare", cmd_synth_diff_prepare), ("diff-score", cmd_synth_diff_score),
                       ("planned-plan", cmd_synth_planned_plan), ("planned-ingest", cmd_synth_planned_ingest)]:
        p = ssub.add_parser(name)
        p.add_argument("--dir", required=True)
        if name == "planned-plan":
            p.add_argument("--family", required=True, help="a name in targets.PLANNED_FAMILIES")
            p.add_argument("--groups", type=int, required=True)
        if name in ("doc-ingest", "planned-ingest"):
            p.add_argument("--allow-heuristic", action="store_true",
                           help=f"if the real tokenizer is unavailable, estimate tokens as len/4 x "
                                f"{padding.HEURISTIC_FACTOR} instead of refusing")
        if name == "doc-plan":
            p.add_argument("--docs", type=int, required=True)
            p.add_argument("--per-batch", type=int, default=documents.PER_BATCH)
        if name in ("plan", "doc-plan", "planned-plan", "check-prepare", "attrib-prepare", "nodoc-prepare",
                    "diff-prepare"):
            p.add_argument("--seed", type=int, default=0)
        if name in ("check-score", "nodoc-prepare", "nodoc-score", "diff-prepare", "diff-score"):
            p.add_argument("--force", action="store_true",
                           help="document runs: discard the later steps (and this step's answers) and run again")
        if name == "nodoc-prepare":
            p.add_argument("--part-size", type=int, default=documents.NODOC_PART_SIZE)
        if name == "diff-prepare":
            p.add_argument("--part-size", type=int, default=documents.DIFF_PART_SIZE)
        p.set_defaults(func=func)
        if name == "plan":
            p.add_argument("--clusters", type=int, required=True)
            p.add_argument("--per-batch", type=int, default=25)
        if name == "check-prepare":
            p.add_argument("--sheets", type=int, default=None,
                           help=f"checker sheets per item (default {PLANNED_SHEETS} for planned runs, else 3)")
        if name in ("check-prepare", "attrib-prepare"):
            p.add_argument("--part-size", type=int, default=250)
        if name == "check-prepare":
            p.add_argument("--doc-part-size", type=int, default=documents.CHECK_PART_SIZE,
                           help="items per checker part in a document run (each item carries a whole document)")

    p = sub.add_parser("train", help="Stage 0 LoRA training from a JSON config")
    p.add_argument("--config", required=True)
    p.add_argument("--force", action="store_true", help="train even if out_dir already holds a finished run")
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
        p.add_argument("--n-max", type=int, default=None,
                       help="letters read per row (default: n_max in <model>/train_config.json, else 6); "
                            "rows with more options are an error")
        p.add_argument("--skip-overlong", action="store_true",
                       help="leave rows longer than --max-len out (ids recorded in the output JSON); default is an error")
        if name == "eval":
            p.add_argument("--calibration", help="calibration JSON from yev calibrate (default T=1)")
            p.add_argument("--shuffle-options", type=int, default=0, metavar="K",
                           help="position-bias check: re-ask each row with >= --min-options options in K seeded "
                                "option orders (plus the original); --out gets position_bias JSON, not dev metrics")
            p.add_argument("--min-options", type=int, default=7, help="rows with fewer options are left out")
            p.add_argument("--seed", type=int, default=0, help="seed of the option permutations")
        p.set_defaults(func=func)
    p = sub.add_parser("serve", help="serve the model behind TypeSafe's POST /v1/systemone (needs the serve extra)")
    p.add_argument("--model", required=True, help="LoRA adapter dir or full model dir")
    p.add_argument("--base", help="base model path (required for adapters; tokenizer source)")
    p.add_argument("--calibration", help="calibration JSON from yev calibrate (default T=1)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-len", type=int, default=16384, help="requests with a longer prompt are refused (422)")
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--no-prefix-cache", dest="prefix_cache", action="store_false",
                   help="re-read the shared state for every question (the exact, uncached path)")
    p.add_argument("--min-prefix-tokens", type=int, default=256,
                   help="shortest shared prompt prefix (tokens) that uses the prefix KV cache")
    p.add_argument("--model-name", default="yev-4b", help="model name reported in responses and /v1/models")
    p.set_defaults(func=cmd_serve)
    from yev.bench import BENCHES
    p = sub.add_parser("gate", help="Stage 1 publish-gate report (exit 1 unless PUBLISHABLE)")
    p.add_argument("--stage0-dev", help="yev eval report of the model on the Stage 0 dev set (accuracy)")
    p.add_argument("--stage1-dev", help="yev eval report of the model on the Stage 1 (all) dev set (ECE)")
    p.add_argument("--workflowevals", help="directory of <workflow>/scores.json")
    p.add_argument("--bench", help="directory of jevbench/, dynabench_test/, rjudge/ metrics.json")
    p.add_argument("--decidebench", help="directory of the DecideBench test (with examples) metrics.json")
    p.add_argument("--decidebench-stage", default="stage1",
                   help="the DecideBench metrics must carry this run.stage (from yev bench score --stage)")
    p.add_argument("--out", required=True, help="gate.json")
    p.set_defaults(func=cmd_gate)

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
    p.add_argument("--rows", required=True, help="rows*.chat.jsonl from yev bench build")
    p.add_argument("--calibration", help="calibration JSON from yev calibrate (default T=1)")
    p.add_argument("--out", required=True, help="directory for predictions.jsonl and metrics.json")
    p.add_argument("--max-len", type=int, default=16384,
                   help="rows longer than this are an error unless --skip-overlong")
    p.add_argument("--skip-overlong", action="store_true",
                   help="leave rows longer than --max-len out of scoring (ids recorded in metrics.json)")
    p.add_argument("--force", action="store_true", help="decidebench: overwrite an existing <out>/metrics.json")
    p.add_argument("--stage", help="decidebench: once-guard per training stage (sentinel .scored-<stage>-<variant>; "
                                   "without it, Stage 0's .scored-<variant>)")
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.set_defaults(func=cmd_bench_score)
    p = bsub.add_parser("overlap", help="share of benchmark states sharing a normalised 8-gram with training")
    p.add_argument("--name", required=True, choices=sorted(BENCHES))
    p.add_argument("--train", required=True, help="training *.chat.jsonl")
    p.add_argument("--dir", help="default data/bench/<name>")
    p.set_defaults(func=cmd_bench_overlap)
    p = bsub.add_parser("fingerprint", help="8-gram fingerprints of every reported benchmark (Hub reads: network)")
    p.add_argument("--out", required=True, help="e.g. data/bench/fingerprints.json")
    p.add_argument("--bench-dir", default=str(BENCH_DIR), help="directory of <name>/rows*.chat.jsonl")
    p.add_argument("--no-hub", action="store_true", help="only the local chat rows (skip WorkflowEvals and Decision Index)")
    p.set_defaults(func=cmd_bench_fingerprint)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    return args.func(args)
