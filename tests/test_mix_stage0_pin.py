"""Pins the mix output for a Stage 0-style recipe (dev_share blocks, an Opus batch block, natural calibration,
DecideBench dev files, worked examples). The hashes were produced by the code before Plan 5 Task 9 (e669235),
so any change to how Stage 0 recipes mix shows up here."""

import hashlib
import sys

from yev.filters.contamination import build_fingerprints
from yev.mix import build_mix, write_mix
from yev.schema import Decision, Option, write_jsonl

OPTS = [Option("a", "Option a."), Option("b", "Option b."), Option("c", "Option c.")]

PINNED = {  # computed with PYTHONPATH=<worktree at e669235>/src python tests/test_mix_stage0_pin.py <dir>
    "train.jsonl": "c3e30fdee02f91cbf9acb9d1bedd62e9686b470e",
    "dev.jsonl": "e215be5881833ec65f97d6770a43e7607141ba11",
    "calibration.jsonl": "e6dd4f9242bd99843d3c6c04028b90f19572585b",
    "train.chat.jsonl": "53ad79deede7f30f815478e301a6b9f772bb5543",
    "dev.chat.jsonl": "98e8a5d032e8e0be7af0923e03f0f1a51a104911",
    "calibration.chat.jsonl": "781a2f6f8c850e8e6fb24c2aeaba5f744c71bdc7",
}


def _dec(i, gold, source, family="f", state=None, cluster=None, split="pool"):
    return Decision(id=f"{source}:{i}", type="choice", state=state or f"{source} {family} state number {i}.",
                    question="Which?", options=list(OPTS), gold=gold, family=family, source=source, licence="mit",
                    cluster_id=cluster, split=split)


def build_fixture_recipe(root):
    pub = [_dec(i, "abc"[i % 3], "pub", family=f"pf{i % 4}") for i in range(400)]
    rules = [_dec(i, "abc"[(i // 2) % 3], "synthetic_rules", family=f"rf{i % 3}", cluster=f"rules:{i // 2}")
             for i in range(300)]
    opus = [_dec(b * 100 + c * 10 + m, "ab"[m], "synthetic_opus", family=f"of{b % 2}", cluster=f"run:batch-{b:03d}:{c}")
            for b in range(12) for c in range(5) for m in range(2)]
    nat = [_dec(i, "abc"[i % 3], "nat", family="natural") for i in range(150)]
    dev = [_dec(i, "a", "decidebench", split="dev") for i in range(8)]
    for name, rows in (("pool/pub", pub), ("pool/rules", rules), ("pool/opus", opus), ("nat/n", nat), ("dev/db", dev)):
        write_jsonl(root / f"{name}.jsonl", rows)
    return {
        "name": "pin", "seed": 0, "examples_rate": 0.35,
        "blocks": [
            {"name": "public", "files": [str(root / "pool" / "pub.jsonl")], "rows": 250},
            {"name": "synthetic_rules", "files": [str(root / "pool" / "rules.jsonl")], "rows": 200, "dev_share": 0.05},
            {"name": "synthetic_opus", "files": [str(root / "pool" / "opus.jsonl")], "rows": 90, "dev_share": 0.1},
        ],
        "dev_files": [str(root / "dev" / "*.jsonl")],
        "calibration": {"files": [str(root / "nat" / "*.jsonl")], "rows": 60},
    }


def output_hashes(root) -> dict[str, str]:
    recipe = build_fixture_recipe(root)
    write_mix(build_mix(recipe, build_fingerprints([])), recipe, root / "mix")
    return {f: hashlib.sha1((root / "mix" / f).read_bytes()).hexdigest()
            for f in ("train.jsonl", "dev.jsonl", "calibration.jsonl",
                      "train.chat.jsonl", "dev.chat.jsonl", "calibration.chat.jsonl")}


def test_stage0_style_output_pinned(tmp_path):
    assert output_hashes(tmp_path) == PINNED


if __name__ == "__main__":  # prints the hashes for the yev on sys.path (used to pin against e669235)
    from pathlib import Path

    import yev.mix
    print(yev.mix.__file__)
    print(output_hashes(Path(sys.argv[1])))
