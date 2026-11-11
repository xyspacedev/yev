import json
from pathlib import Path

from yev import cli
from yev.generators import padding, prompt_template
from yev.mix import build_mix, write_mix
from yev.schema import read_jsonl

from test_mix import NO_BENCH, setup_pool

ROOT = Path(__file__).resolve().parent.parent
C = json.loads((ROOT / "recipes/stage1c.json").read_text())
D = json.loads((ROOT / "recipes/stage1d.json").read_text())
TEMPLATE = {"rate": 0.6, "tokens": [120, 400], "max_tokens": 6000, "exclude_sources": ["decidebench"]}


def count(t):
    return max(1, len(t) // 4)


def test_stage1d_is_stage1c_with_prompt_templates():
    assert D["name"] == "stage1d"
    assert {k: v for k, v in D.items() if k not in ("name", "prompt_template")} == {k: v for k, v in C.items() if k != "name"}
    spec = prompt_template.parse_spec(D["prompt_template"])
    assert spec["exclude_families"] == {"many_options"} and spec["exclude_sources"] == {"decidebench"}
    assert spec["max_tokens"] < padding.MAX_TRAIN_TOKENS


def _chats(out, split):
    return [json.loads(line) for line in (out / f"{split}.chat.jsonl").read_text().splitlines()]


def test_templated_mix_differs_from_the_plain_mix_only_in_states(tmp_path):
    recipe = setup_pool(tmp_path)
    plain = tmp_path / "plain"
    write_mix(build_mix(recipe, NO_BENCH), recipe, plain)
    templated = tmp_path / "templated"
    report = write_mix(build_mix(recipe, NO_BENCH), {**recipe, "prompt_template": TEMPLATE}, templated,
                       fingerprints=NO_BENCH, count=count)
    for split in ("train", "dev", "calibration"):
        a, b = list(read_jsonl(plain / f"{split}.jsonl")), list(read_jsonl(templated / f"{split}.jsonl"))
        assert [(d.id, d.gold, d.question, d.options, d.split) for d in a] == [(d.id, d.gold, d.question, d.options, d.split) for d in b]
        changed = {x.id for x, y in zip(a, b) if x.state != y.state}
        assert len(changed) == report["prompt_template"][split]["templated"]
        assert all(x.source != "decidebench" for x in a if x.id in changed)
        ca, cb = _chats(plain, split), _chats(templated, split)
        for x, y in zip(ca, cb):
            assert {k: v for k, v in x.items() if k != "messages"} == {k: v for k, v in y.items() if k != "messages"}
            assert len(x["messages"]) == len(y["messages"])
            assert [m["content"] for m in x["messages"] if m["role"] != "user"] == [m["content"] for m in y["messages"] if m["role"] != "user"]
            for mx, my in zip(x["messages"], y["messages"]):
                if mx["role"] == "user":
                    ux, uy = json.loads(mx["content"]), json.loads(my["content"])
                    assert (ux["question"], ux["options"]) == (uy["question"], uy["options"])
                    assert (ux["state"] != uy["state"]) == (x["id"] in changed)
    train = report["prompt_template"]["train"]
    assert 0.4 * train["rows"] < train["templated"] < 0.8 * train["rows"]
    assert train["wrapper_tokens"] > 0 and report["prompt_template"]["dev"]["excluded"] == 5


def test_mix_cli_needs_a_tokenizer_for_prompt_templates(tmp_path, monkeypatch):
    recipe = {**setup_pool(tmp_path), "prompt_template": TEMPLATE}
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    monkeypatch.setattr(cli, "load_fingerprints", lambda: NO_BENCH)

    def fake(allow_heuristic=False):
        if not allow_heuristic:
            raise padding.TokenizerUnavailable("tokenizer not found")
        return padding.heuristic_tokens

    monkeypatch.setattr(padding, "load_token_counter", fake)
    args = ["mix", "--recipe", str(tmp_path / "recipe.json"), "--out", str(tmp_path / "out")]
    assert cli.main(args) == 2 and not (tmp_path / "out").exists()
    assert cli.main([*args, "--allow-heuristic"]) == 0
    assert "prompt_template" in json.loads((tmp_path / "out" / "mix_report.json").read_text())
    recipe["prompt_template"] = {"rate": 2}
    (tmp_path / "recipe.json").write_text(json.dumps(recipe))
    assert cli.main([*args, "--allow-heuristic"]) == 2
