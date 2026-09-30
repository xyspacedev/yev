import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from jeff import cli
from jeff.schema import read_jsonl
from jeff.sources import skill_atlas as sa

LEAVES = {  # dept -> team -> leaves
    "eng": {"eng-a": ["leaf-a1", "leaf-a2", "leaf-a3"], "eng-b": ["leaf-b1", "leaf-b2", "leaf-b3"]},
    "tiny": {"tiny-t": ["tiny-1", "tiny-2"]},
}

# name -> (licence, herd flags)
SKILLS = {
    "mit1": ("MIT", []), "mit2": ("MIT", []), "apache1": ("Apache-2.0", []), "apache2": ("Apache-2.0", []),
    "bsd1": ("BSD-3-Clause", []), "gpl1": ("GPL-3.0", []), "none1": (None, []), "noassert1": ("NOASSERTION", []),
    "flaggedherd": ("MIT", ["variant"]), "flaggedfile": ("MIT", []),
    "mit3": ("MIT", []), "mit4": ("MIT", []), "mit5": ("MIT", []), "mit6": ("MIT", []), "tiny1": ("MIT", []),
}


def sid(name):
    return sa.skill_id(f"org/{name}", f"skills/{name}/SKILL.md")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_atlas(root: Path) -> Path:
    out = root / "pipeline" / "out"
    out.mkdir(parents=True)
    onto = {"version": "test.1", "departments": [
        {"id": d, "name": d.title(), "description": f"{d} dept", "teams": [
            {"id": t, "name": t, "categories": [{"id": l, "name": l.title(), "description": f"About {l}."} for l in ls]}
            for t, ls in teams.items()]} for d, teams in LEAVES.items() if d != "tiny"],
        "bins": [{"id": "tiny", "name": "Tiny", "description": "bin", "teams": [
            {"id": "tiny-t", "name": "Tiny", "categories": [{"id": l, "name": l, "description": l} for l in LEAVES["tiny"]["tiny-t"]]}]}]}
    (root / "pipeline" / "ontology.json").write_text(json.dumps(onto))
    (root / "pipeline" / "ontology_migrations.json").write_text(json.dumps({"leaf_map": {"old-a1": "leaf-a1"}}))
    write_jsonl(out / "herd.jsonl", [{"skill_id": sid(n), "license": lic, "flags": fl, "origin_repo": f"org/{n}"}
                                     for n, (lic, fl) in SKILLS.items()])
    write_jsonl(out / "flags.jsonl", [{"skill_id": sid("flaggedfile"), "kind": "prompt-injection"}])
    names = list(SKILLS)
    table = pa.table({
        "repo": [f"org/{n}" for n in names], "path": [f"skills/{n}/SKILL.md" for n in names],
        "name": names, "description": [f"Does {n} things." for n in names],
        "body": [f"Step one for {n}.\n\n  Step two.  " + ("word " * (2000 if n == "mit1" else 5)) for n in names],
    })
    pq.write_table(table, out / "corpus.parquet")
    # scope: several scopes; the gpl/flagged ones must drop
    write_jsonl(out / "scope" / "scope-000.jsonl", [
        {"n": 1, "skill_id": sid("mit1"), "scope": "portable"},
        {"n": 2, "skill_id": sid("apache1"), "scope": "vendor"},
        {"n": 3, "skill_id": sid("gpl1"), "scope": "internal"},
        {"n": 4, "skill_id": sid("flaggedherd"), "scope": "internal"},
        {"n": 5, "skill_id": sid("flaggedfile"), "scope": "portable"},
        {"n": 6, "skill_id": sid("none1"), "scope": "portable"},
        {"n": 7, "skill_id": sid("bsd1"), "scope": "portable"},
    ])
    write_jsonl(out / "scope" / "scope-fill-000.jsonl", [{"n": 1, "skill_id": sid("mit2"), "scope": "private"}])
    # judge: one group with many large gaps, plus a group judged only by judge4 (Sonnet)
    g1 = ["mit1", "mit2", "mit3", "mit4", "mit5", "apache1"]
    g2 = ["mit6", "apache2"]
    write_jsonl(out / "judge_groups.jsonl", [
        {"group_id": "leaf-a1#final", "leaf": "leaf-a1", "kind": "final", "members": [sid(n) for n in g1]},
        {"group_id": "leaf-a2#final", "leaf": "leaf-a2", "kind": "final", "members": [sid(n) for n in g2]},
    ])
    scores = {"mit1": 90, "mit2": 85, "mit3": 60, "mit4": 50, "mit5": 30, "apache1": 80}
    write_jsonl(out / "judge" / "judge6-000.jsonl", [
        {"group_id": "leaf-a1#final", "ranking": [{"skill_id": sid(n), "overall": s} for n, s in scores.items()]}])
    write_jsonl(out / "judge" / "judge4-000.jsonl", [
        {"group_id": "leaf-a2#final", "ranking": [{"skill_id": sid("mit6"), "overall": 90},
                                                  {"skill_id": sid("apache2"), "overall": 20}]}])
    # labels: Opus fallback/thin vs Sonnet seed and fb3
    def lab(n, leaf, conf, method, secondary=None):
        return {"skill_id": sid(n), "leaf": leaf, "confidence": conf, "secondary": secondary, "method": method,
                "ontology_version": "test.0"}
    write_jsonl(out / "labels_fb.jsonl", [lab("mit1", "old-a1", "high", "fallback", "leaf-b1"),
                                          lab("mit2", "leaf-a2", "medium", "fallback"),
                                          lab("mit3", "leaf-a3", "medium", "fallback"),
                                          lab("gpl1", "leaf-a1", "high", "fallback"),
                                          lab("flaggedherd", "leaf-a1", "high", "fallback")])
    write_jsonl(out / "labels_thin.jsonl", [lab("apache1", "leaf-b2", "high", "thin"),
                                            lab("tiny1", "tiny-1", "high", "thin")])
    write_jsonl(out / "labels_seed.jsonl", [lab("mit4", "leaf-a1", "high", "seed")])
    write_jsonl(out / "labels_fb3.jsonl", [lab("mit5", "leaf-a1", "high", "fallback")])
    write_jsonl(out / "labels_audit.jsonl", [lab("mit2", "leaf-a2", "high", "audit"),
                                             lab("mit3", "leaf-b1", "high", "audit")])
    return root


def build(tmp_path, seed=0, name="out"):
    root = make_atlas(tmp_path / "atlas") if not (tmp_path / "atlas").exists() else tmp_path / "atlas"
    out = tmp_path / name
    stats = sa.run(out, root=root, seed=seed)
    fams = {f: list(read_jsonl(out / f"{f}.jsonl")) for f in ("skill_scope", "skill_judge", "skill_filing")}
    return fams, stats, out


def ids_in(d):
    m = d.metadata
    return set(m["skill_id"]) if isinstance(m["skill_id"], list) else {m["skill_id"]}


def test_licence_and_flag_filters(tmp_path):
    fams, stats, _ = build(tmp_path)
    used = set().union(*(ids_in(d) for ds in fams.values() for d in ds))
    for bad in ("gpl1", "none1", "noassert1", "bsd1", "flaggedherd", "flaggedfile"):
        assert sid(bad) not in used, bad
    scope = {d.metadata["skill_id"]: d for d in fams["skill_scope"]}
    assert set(scope) == {sid("mit1"), sid("apache1"), sid("mit2")}
    assert scope[sid("mit2")].gold == "internal"  # alias "private" normalised
    assert scope[sid("apache1")].licence == "apache-2.0" and scope[sid("apache1")].metadata["licence"] == "Apache-2.0"
    assert stats["skill_scope"]["dropped_flags"] == 2 and stats["skill_scope"]["dropped_licence"] == 3
    for ds in fams.values():
        for d in ds:
            assert d.source == "skill_atlas" and d.type == "choice"


def test_state_is_normalised_and_capped(tmp_path):
    fams, _, _ = build(tmp_path)
    d = next(d for d in fams["skill_scope"] if d.metadata["skill_id"] == sid("mit1"))
    assert len(d.state) <= sa.SCOPE_CAP and d.state.endswith(" …")
    assert d.state.startswith("Skill: mit1\nDescription: Does mit1 things.\nBody: Step one for mit1. Step two. word")


def test_opus_stage_filter(tmp_path):
    fams, stats, _ = build(tmp_path)
    filing = {d.metadata["skill_id"]: d for d in fams["skill_filing"]}
    assert sid("mit4") not in filing and sid("mit5") not in filing  # seed and fb3 are not Opus
    assert all(d.metadata["method"] in {"fallback", "thin"} for d in filing.values())
    groups = {d.metadata["group_id"] for d in fams["skill_judge"]}
    assert groups == {"leaf-a1#final"}  # judge4-only group excluded
    assert stats["skill_judge"]["dropped_provenance"] == 1


def test_judge_gap_and_pair_cap(tmp_path):
    fams, _, _ = build(tmp_path)
    judge = fams["skill_judge"]
    assert len(judge) == 3
    for d in judge:
        a, b = d.metadata["skills"]
        assert abs(a["overall"] - b["overall"]) >= 15
        hi = "a" if a["overall"] > b["overall"] else "b"
        assert d.gold == hi and d.keys == ["a", "b"]
        assert "=== Skill A ===" in d.state and "=== Skill B ===" in d.state
    gaps = sorted((d.metadata["gap"] for d in judge), reverse=True)
    assert gaps == [60, 55, 50]  # largest gaps first: 90-30, 85-30, 80-30


def test_filing_options(tmp_path):
    fams, _, _ = build(tmp_path)
    filing = {d.metadata["skill_id"]: d for d in fams["skill_filing"]}
    # mit1 high (old-a1 -> leaf-a1 via leaf_map); mit2 audit-confirmed; mit3 medium and audit disagrees
    assert set(filing) == {sid("mit1"), sid("mit2"), sid("apache1")}
    assert sid("tiny1") not in filing  # only 2 leaves in its bin -> fewer than 4 options
    for d in filing.values():
        assert 4 <= len(d.options) <= 6 and d.gold in d.keys
        assert d.metadata["ontology_version"] == "test.0"
    d = filing[sid("mit1")]
    assert d.gold == "leaf_a1" and "leaf_b1" in d.keys and len(d.keys) == 5


def test_attribution_file(tmp_path):
    fams, stats, out = build(tmp_path)
    rows = [json.loads(line) for line in (out / "ATTRIBUTION.jsonl").read_text().splitlines()]
    used = set().union(*(ids_in(d) for ds in fams.values() for d in ds))
    assert {r["skill_id"] for r in rows} == used and len(rows) == len(used)
    for r in rows:
        assert set(r) == {"repo", "path", "licence", "skill_id"}
        assert r["licence"] in sa.SPDX_KEEP and sa.skill_id(r["repo"], r["path"]) == r["skill_id"]


def test_deterministic(tmp_path):
    build(tmp_path, name="one")
    build(tmp_path, name="two")
    for f in ("skill_scope.jsonl", "skill_judge.jsonl", "skill_filing.jsonl", "ATTRIBUTION.jsonl"):
        assert (tmp_path / "one" / f).read_text() == (tmp_path / "two" / f).read_text()


def test_missing_root_skips(tmp_path, capsys):
    stats = sa.run(tmp_path / "out", root=tmp_path / "nope")
    assert stats["skipped"] is True
    assert "skipped" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_cli_only_skill_atlas(tmp_path, monkeypatch):
    root = make_atlas(tmp_path / "atlas")
    monkeypatch.setattr(cli, "SOURCES", [])
    rc = cli.main(["build-public", "--out", str(tmp_path / "o"), "--only", "skill_atlas",
                   "--skill-atlas-root", str(root)])
    assert rc == 0
    assert (tmp_path / "o" / "skill_judge.jsonl").exists() and (tmp_path / "o" / "ATTRIBUTION.jsonl").exists()
    stats = json.loads((tmp_path / "o" / "build_stats.json").read_text())
    assert stats["skill_atlas"]["skill_judge"]["written"] == 3
