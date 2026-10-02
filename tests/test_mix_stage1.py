"""Stage 1 recipe, mix extensions (document holdout, dev minimums, disjoint pools, split links, long-family
example exclusion) and training config."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from yev.filters.contamination import build_fingerprints
from yev.mix import build_mix, state_key, write_mix
from yev.schema import Decision, Option, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
NO_BENCH = build_fingerprints([])
OPTS = [Option("a", "Option a."), Option("b", "Option b."), Option("c", "Option c.")]
NEW_FAMILIES = ["returns_v2", "date_number_drills", "invoice_ledger", "numbered_scales", "many_options",
                "many_options_intent", "long_workflow", "long_padding", "public_long"]
LONG_FAMILIES = {"invoice_ledger", "long_workflow", "long_padding", "public_long"}


def dec(id_, gold="a", source="synthetic_rules", family="f", state=None, cluster=None, metadata=None, question="Which?"):
    return Decision(id=id_, type="choice", state=state or f"State of row {id_}.", question=question,
                    options=list(OPTS), gold=gold, family=family, source=source, licence="mit", cluster_id=cluster,
                    split="pool", metadata=metadata)


def _recipe(blocks, **extra):
    return {"name": "t", "seed": 0, "blocks": blocks, **extra}


def _load_json(rel):
    return json.loads((ROOT / rel).read_text())


def _docs(tmp_path):
    """Invoice-style docs (cluster_id), Opus docs (doc:<run>:<batch>:<k>), public docs (shared state, no cluster)."""
    inv = [dec(f"invoices:0:{k}:{j}", "abc"[j % 3], family="invoice_ledger", state=f"Invoice document {k}.",
               cluster=f"invoices:0:{k}") for k in range(60) for j in range(8)]
    opus = [dec(f"doc:r:b{k % 12}:{k}:q{j}", "abc"[j % 3], source="synthetic_opus", family="long_workflow",
                state=f"Opus document {k}.", cluster=f"doc:r:b{k % 12}:{k}") for k in range(60) for j in range(7)]
    pub = [dec(f"cuad:{k}:{j}", "abc"[j % 3], source="cuad", family="public_long", state=f"Contract number {k}.",
               question=f"Clause {j}?") for k in range(60) for j in range(6)]
    for name, rows in (("inv", inv), ("opus", opus), ("pub", pub)):
        write_jsonl(tmp_path / f"{name}.jsonl", rows)
    return _recipe([
        {"name": n, "files": [str(tmp_path / f"{n}.jsonl")], "rows": 300, "dev_share": 0.05, "dev_min_rows": 100}
        for n in ("inv", "opus", "pub")
    ])


def test_document_questions_share_split(tmp_path):
    res = build_mix(_docs(tmp_path), NO_BENCH)
    splits_by_doc = defaultdict(set)
    for d in res.train + res.dev:
        splits_by_doc[d.cluster_id or f"state:{d.state}"].add(d.split)
    assert splits_by_doc and all(len(s) == 1 for s in splits_by_doc.values())
    # documents are taken or held out whole: every document present has all its questions
    sizes = defaultdict(int)
    for d in res.train + res.dev:
        sizes[d.cluster_id or d.state] += 1
    assert set(sizes.values()) <= {6, 7, 8}
    assert {d.family for d in res.dev} == {"invoice_ledger", "long_workflow", "public_long"}


def test_new_family_dev_share_min_100(tmp_path):
    res = build_mix(_docs(tmp_path), NO_BENCH)
    for name in ("inv", "opus", "pub"):
        assert res.report["blocks"][name]["dev"] >= 100  # 5% of 60 documents alone would give 3 documents
    # the minimum only adds units: a large pool still holds out ~5%
    big = [dec(f"x:{i}", "abc"[i % 3]) for i in range(4000)]
    write_jsonl(tmp_path / "big.jsonl", big)
    res = build_mix(_recipe([{"name": "big", "files": [str(tmp_path / "big.jsonl")], "rows": 5000,
                              "dev_share": 0.05, "dev_min_rows": 100}]), NO_BENCH)
    assert res.report["blocks"]["big"]["dev"] == 200
    # recipe: every new family block holds out 5% with at least 100 rows
    blocks = {b["name"]: b for b in _load_json("recipes/stage1.json")["blocks"]}
    for fam in NEW_FAMILIES:
        assert blocks[fam]["dev_share"] == 0.05 and blocks[fam]["dev_min_rows"] >= 100


def test_stage0_blocks_unchanged():
    s0, s1 = _load_json("recipes/stage0.json"), _load_json("recipes/stage1.json")
    assert s1["blocks"][: len(s0["blocks"])] == s0["blocks"]
    assert [b["name"] for b in s1["blocks"][len(s0["blocks"]):]] == NEW_FAMILIES
    for key in ("seed", "examples_rate", "dev_files", "calibration"):
        assert s1[key] == s0[key]
    targets = {b["name"]: b["rows"] for b in s1["blocks"]}
    assert targets["returns_v2"] == targets["date_number_drills"] == targets["invoice_ledger"] == 3000
    assert targets["numbered_scales"] == 2000
    assert targets["many_options"] + targets["many_options_intent"] == 3000
    assert targets["long_workflow"] == 4000 and targets["long_padding"] == 3000 and targets["public_long"] == 3000
    for b in s1["blocks"][len(s0["blocks"]):]:
        assert b["files"] == [f"data/stage1-pool/{b['name']}.jsonl"]


def test_stage1_config_matches_lc100_except_limits():
    from yev.train.trainer import TrainConfig

    lc, s1 = _load_json("configs/stage0/lc100.json"), _load_json("configs/stage1/main.json")
    changed = {k for k in lc.keys() | s1.keys() if lc.get(k) != s1.get(k)}
    assert changed == {"name", "train_path", "out_dir", "n_max", "max_len_plain", "max_len_examples", "token_budget",
                       "sort_window", "epochs"}
    assert s1["n_max"] == 10 and s1["max_len_plain"] == s1["max_len_examples"] == s1["token_budget"] == 12288
    assert s1["epochs"] == 1 and s1["sort_window"] > TrainConfig.sort_window
    assert s1["out_dir"] == "/home/ubuntu/runs/stage1"
    assert s1["train_path"] == "/home/ubuntu/yev/data/mix/stage1/train.chat.jsonl"
    TrainConfig.from_json(str(ROOT / "configs/stage1/main.json"))  # every key is a real config field


def test_trainer_passes_sort_window(tmp_path, monkeypatch, tok, tiny_model, make_row):
    from yev.train import trainer

    seen = []
    real = trainer.build_batches

    def spy(*a, **kw):
        seen.append(kw.get("window"))
        return real(*a, **kw)

    monkeypatch.setattr(trainer, "build_batches", spy)
    rows = [make_row(f"r{i}", {"A": "k0", "B": "k1"}, "A") for i in range(4)]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cfg = trainer.TrainConfig(name="t", model_path="unused", train_path=str(tmp_path / "train.jsonl"),
                              out_dir=str(tmp_path / "out"), lora=False, grad_checkpointing=False, bf16=False,
                              token_budget=4000, grad_accum=1, save_every=100, max_steps=1, sort_window=7)
    trainer.train(cfg, tokenizer=tok, model=tiny_model)
    assert seen and set(seen) == {7}
    assert trainer.TrainConfig.sort_window == 50  # Stage 0 configs keep the old window


def test_exclude_from_keeps_intent_pools_disjoint(tmp_path):
    intent = [dec(f"banking77:{i}", "abc"[i % 3], source="banking77", family="intent", state=f"Utterance {i}.")
              for i in range(400)]
    reframed = [dec(f"{d.id}:mo", d.gold, source="banking77", family="many_options_intent", state=d.state,
                    metadata={"orig_id": d.id}) for d in intent]
    write_jsonl(tmp_path / "intent.jsonl", intent)
    write_jsonl(tmp_path / "mo.jsonl", reframed)
    res = build_mix(_recipe([
        {"name": "intent", "files": [str(tmp_path / "intent.jsonl")], "rows": 150},
        {"name": "mo", "files": [str(tmp_path / "mo.jsonl")], "rows": 150, "dev_share": 0.05, "dev_min_rows": 20,
         "exclude_from": ["intent"]},
    ]), NO_BENCH)
    rows = res.train + res.dev
    intent_states = {d.state for d in rows if d.family == "intent"}
    mo_states = {d.state for d in rows if d.family == "many_options_intent"}
    assert intent_states and mo_states and not (intent_states & mo_states)
    assert res.report["blocks"]["mo"]["excluded"] == len(intent_states)
    with pytest.raises(ValueError, match="earlier block"):
        build_mix(_recipe([{"name": "mo", "files": [str(tmp_path / "mo.jsonl")], "rows": 5, "exclude_from": ["x"]}]),
                  NO_BENCH)
    blocks = {b["name"]: b for b in _load_json("recipes/stage1.json")["blocks"]}
    assert blocks["many_options_intent"]["exclude_from"] == ["intent"]


def test_split_follows_keeps_padded_rows_with_their_originals(tmp_path):
    orig = [dec(f"rv2:{i}", "abc"[i % 3], family="returns_v2", cluster=f"rv2:{i // 2}") for i in range(400)]
    padded = [dec(f"pad:{d.id}", d.gold, family="long_padding", state=f"Long filler text. {d.state}",
                  cluster=f"pad:{d.cluster_id}", metadata={"orig_id": d.id}) for d in orig]
    write_jsonl(tmp_path / "rv2.jsonl", orig)
    write_jsonl(tmp_path / "pad.jsonl", padded)
    res = build_mix(_recipe([
        {"name": "rv2", "files": [str(tmp_path / "rv2.jsonl")], "rows": 300, "dev_share": 0.1, "dev_min_rows": 20},
        {"name": "pad", "files": [str(tmp_path / "pad.jsonl")], "rows": 400, "dev_share": 0.1, "dev_min_rows": 20,
         "split_follows": ["rv2"]},
    ]), NO_BENCH)
    split_of = {d.id: d.split for d in res.train + res.dev if d.family == "returns_v2"}
    pads = [d for d in res.train + res.dev if d.family == "long_padding"]
    assert pads and any(d.split == "dev" for d in pads)
    for d in pads:
        o = split_of.get(d.metadata["orig_id"])
        if o == "dev":
            raise AssertionError(f"{d.id} kept although its original is held out")
        if o == "train":
            assert d.split == "train"
    assert res.report["blocks"]["pad"]["excluded"] == sum(s == "dev" for s in split_of.values())
    blocks = {b["name"]: b for b in _load_json("recipes/stage1.json")["blocks"]}
    assert "returns_v2" in blocks["long_padding"]["split_follows"]


def test_long_families_get_and_give_no_examples(tmp_path):
    short = [dec(f"s:{i}", "abc"[i % 3], family="short") for i in range(60)]
    long_ = [dec(f"l:{i}", "abc"[i % 3], family="public_long", state=f"Long contract {i}.") for i in range(60)]
    write_jsonl(tmp_path / "p.jsonl", short + long_)
    recipe = _recipe([{"name": "x", "files": [str(tmp_path / "p.jsonl")], "rows": 1000}], examples_rate=1.0,
                     examples_exclude_families=["public_long"])
    out = tmp_path / "mix"
    write_mix(build_mix(recipe, NO_BENCH), recipe, out)
    rows = [json.loads(l) for l in (out / "train.chat.jsonl").read_text().splitlines()]
    with_examples = [r for r in rows if sum(m["role"] == "user" for m in r["messages"]) > 1]
    assert with_examples
    for r in rows:
        users = [m for m in r["messages"] if m["role"] == "user"]
        if "Long contract" in users[-1]["content"]:
            assert len(users) == 1  # a long row gets no worked examples
        else:
            assert "Long contract" not in json.dumps(users[:-1])  # and is never an example for another row
    s1 = _load_json("recipes/stage1.json")
    assert set(s1["examples_exclude_families"]) == LONG_FAMILIES


def test_report_gold_distribution(tmp_path):
    rows = [dec(f"m:{i}", "a" if i % 3 else "c", family="many_options") for i in range(30)]
    write_jsonl(tmp_path / "m.jsonl", rows)
    res = build_mix(_recipe([{"name": "m", "files": [str(tmp_path / "m.jsonl")], "rows": 30, "report_gold": True}]),
                    NO_BENCH)
    assert res.report["blocks"]["m"]["train_gold"] == {"a": 20, "c": 10}
    blocks = {b["name"]: b for b in _load_json("recipes/stage1.json")["blocks"]}
    assert blocks["many_options"]["report_gold"] and blocks["returns_v2"]["report_gold"]


def test_state_key_is_the_unit_for_clusterless_documents():
    a = dec("p:1", state="Same contract.")
    b = dec("p:2", state="Same contract.", question="Other clause?")
    assert state_key(a) == state_key(b)
