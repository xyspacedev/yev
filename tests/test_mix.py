import json

import pytest

from jeff import cli
from jeff.filters.contamination import build_fingerprints
from jeff.filters.pipeline import ContaminationError
from jeff.mix import build_mix, holdout_key, unit_key, write_mix
from jeff.schema import Decision, Option, read_jsonl, write_jsonl

OPTS = [Option("a", "Option a."), Option("b", "Option b."), Option("c", "Option c.")]
NO_BENCH = build_fingerprints([])


def dec(i, gold, source, state=None, cluster=None, split="pool"):
    return Decision(id=f"{source}:{i}", type="choice", state=state or f"{source} state number {i}.", question="Which?",
                    options=list(OPTS), gold=gold, family="f", source=source, licence="mit", cluster_id=cluster,
                    split=split)


def setup_pool(tmp_path):
    pub = [dec(i, "abc"[i % 3], "pub") for i in range(300)]
    shared = dec(9999, "a", "pub", state="The shared state appears twice.")
    opus = []
    for b in range(10):
        for c in range(5):
            for m in range(2):
                opus.append(dec(b * 100 + c * 10 + m, "ab"[m], "synthetic_opus", cluster=f"run:batch-{b:03d}:{c}"))
    nat = [dec(5000 + i, "a", "pub") for i in range(100)] + [dec(9998, "b", "pub", state="The shared state appears twice.")]
    dev = [dec(7000 + i, "a", "decidebench", split="dev") for i in range(5)]
    write_jsonl(tmp_path / "pool" / "pub.jsonl", pub + [shared])
    write_jsonl(tmp_path / "pool" / "synthetic_opus.jsonl", opus)
    write_jsonl(tmp_path / "nat" / "pub.jsonl", nat)
    write_jsonl(tmp_path / "dev" / "db.jsonl", dev)
    return {
        "name": "t", "seed": 0, "examples_rate": 1.0,
        "blocks": [
            {"name": "public", "files": [str(tmp_path / "pool" / "pub.jsonl")], "rows": 200},
            {"name": "opus", "files": [str(tmp_path / "pool" / "synthetic_opus.jsonl")], "rows": 1000, "dev_share": 0.2},
        ],
        "dev_files": [str(tmp_path / "dev" / "*.jsonl")],
        "calibration": {"files": [str(tmp_path / "nat" / "*.jsonl")], "rows": 50},
    }


def test_blocks_sizes_shortfall_and_whole_units(tmp_path):
    res = build_mix(setup_pool(tmp_path), NO_BENCH)
    blocks = res.report["blocks"]
    assert 200 <= blocks["public"]["train"] <= 201
    assert blocks["opus"]["dev"] == 20 and blocks["opus"]["train"] == 80 and blocks["opus"]["shortfall"] == 920
    train_clusters = {}
    for d in res.train:
        if d.cluster_id:
            train_clusters.setdefault(d.cluster_id, 0)
            train_clusters[d.cluster_id] += 1
    assert set(train_clusters.values()) == {2}


def test_splits_disjoint_and_dev_by_batch(tmp_path):
    res = build_mix(setup_pool(tmp_path), NO_BENCH)
    assert not ({unit_key(d) for d in res.train} & {unit_key(d) for d in res.dev})
    dev_batches = {holdout_key(d) for d in res.dev if d.source == "synthetic_opus"}
    train_batches = {holdout_key(d) for d in res.train if d.source == "synthetic_opus"}
    assert len(dev_batches) == 2 and not (dev_batches & train_batches)
    assert sum(d.source == "decidebench" for d in res.dev) == 5
    assert {d.split for d in res.train} == {"train"} and {d.split for d in res.dev} == {"dev"}


def test_calibration_excludes_states_used_elsewhere(tmp_path):
    recipe = setup_pool(tmp_path)
    recipe["calibration"]["rows"] = 1000
    res = build_mix(recipe, NO_BENCH)
    used = any(d.state == "The shared state appears twice." for d in res.train)
    in_cal = any(d.state == "The shared state appears twice." for d in res.calibration)
    assert not (used and in_cal)
    assert {d.split for d in res.calibration} == {"calibration"}


def test_contaminated_training_row_raises(tmp_path):
    recipe = setup_pool(tmp_path)
    recipe["blocks"][0]["rows"] = 10_000  # take every public row, so pub:1 is certainly in train
    bench = build_fingerprints([{"state": "pub state number 1.", "question": "Which?",
                                 "options": [{"description": o.description} for o in OPTS]}])
    with pytest.raises(ContaminationError):
        build_mix(recipe, bench)


def test_write_mix_outputs_and_cli(tmp_path, monkeypatch):
    recipe = setup_pool(tmp_path)
    out = tmp_path / "mix"
    report = write_mix(build_mix(recipe, NO_BENCH), recipe, out)
    for split in ("train", "dev", "calibration"):
        rows = list(read_jsonl(out / f"{split}.jsonl"))
        chats = [json.loads(l) for l in (out / f"{split}.chat.jsonl").read_text().splitlines()]
        assert len(rows) == len(chats) and all(c["messages"][0]["role"] == "system" for c in chats)
    assert report["format"]["train_examples"] > 0
    assert json.loads((out / "mix_report.json").read_text())["train"]["rows"] == report["train"]["rows"]

    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(cli, "load_fingerprints", lambda: NO_BENCH)
    assert cli.main(["mix", "--recipe", str(tmp_path / "recipe.json"), "--out", str(tmp_path / "mix2")]) == 0
    assert cli.main(["mix", "--recipe", str(tmp_path / "nope.json"), "--out", str(tmp_path / "mix3")]) == 2
