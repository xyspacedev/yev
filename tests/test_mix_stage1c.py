import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = json.loads((ROOT / "recipes/stage1b.json").read_text())
C = json.loads((ROOT / "recipes/stage1c.json").read_text())
bb = {b["name"]: b for b in B["blocks"]}
cb = {b["name"]: b for b in C["blocks"]}

NEW_ROWS = {
    "agent_injection_pair": 1200, "agent_overreach": 1120, "agent_benign_multistep": 500, "trace_audit": 500,
    "disclosure_gate": 810, "grounded_offer": 500, "persona_style_rule": 450, "count_threshold_rule": 1000,
    "judge_planted_error": 600, "planted_note": 600, "policy_amendment": 555,
    "invoice_line": 900, "payment_terms": 400, "invoice_action": 1300, "cs_scales_actions": 900,
    "security_context": 850,
}
OTHER_OPUS = ["access_request", "agent_action", "claim_evidence", "expenses", "intent", "severity", "sentiment"]


def test_stage1c_recipe_extends_stage1b():
    assert C["name"] == "stage1c"
    for k in ("seed", "examples_rate", "dev_files", "calibration"):
        assert C[k] == B[k]
    assert [b["name"] for b in C["blocks"]][: len(B["blocks"])] == [b["name"] for b in B["blocks"]]
    changed = {"public_long", "long_workflow", "synthetic_opus", "skill_atlas", "numbered_scales"}
    for name, b in bb.items():
        if name not in changed:
            assert cb[name] == b, name
    assert set(cb) - set(bb) == set(NEW_ROWS)

    def rest(name, *drop):
        return {k: v for k, v in cb[name].items() if k not in drop} == {k: v for k, v in bb[name].items() if k not in drop}

    assert cb["public_long"]["files"] == ["data/stage1c-pool/public_long.jsonl"] and rest("public_long", "files")
    assert cb["long_workflow"]["files"] == ["data/stage1b-pool/long_workflow.jsonl",
                                            "data/stage1c-pool/long_workflow_more.jsonl"] and rest("long_workflow", "files")
    assert cb["numbered_scales"]["rows"] == 2500 and rest("numbered_scales", "rows")
    so = cb["synthetic_opus"]
    assert so["repeat"] == 2 and so["repeat_filter"] == {"family": ["moderation", "routing", "returns"]}
    assert so["repeat_extra"] == [{"repeat": 1, "repeat_filter": {"family": OTHER_OPUS}}]
    assert rest("synthetic_opus", "repeat", "repeat_filter", "repeat_extra")
    sa = cb["skill_atlas"]
    assert sa["repeat"] == 0.5 and sa["repeat_filter"] == {"family": ["skill_filing"]}
    assert rest("skill_atlas", "repeat", "repeat_filter")
    assert "repeat" not in cb["synthetic_rules"] and "synthetic_opus_replay" not in cb
    assert C["examples_exclude_families"] == B["examples_exclude_families"] + ["trace_audit", "invoice_line", "policy_amendment"]


def test_stage1c_new_blocks_have_calibration_share():
    for name, rows in NEW_ROWS.items():
        b = cb[name]
        files = (["data/stage1c-pool/agent_overreach.jsonl", "data/stage1c-pool/agent_overreach_rules.jsonl"]
                 if name == "agent_overreach" else [f"data/stage1c-pool/{name}.jsonl"])
        assert b["files"] == files
        assert b["rows"] == rows
        assert (b["dev_share"], b["dev_min_rows"]) == (0.05, 60)
        assert (b["calibration_share"], b["calibration_min_rows"]) == (0.05, 40)


def test_stage1c_config_matches_stage1b_except_paths():
    b = json.loads((ROOT / "configs/stage1/stage1b.json").read_text())
    c = json.loads((ROOT / "configs/stage1/stage1c.json").read_text())
    assert c["name"] == "stage1c"
    assert c["train_path"] == "/home/ubuntu/yev/data/mix/stage1c/train.chat.jsonl"
    assert c["out_dir"] == "/home/ubuntu/runs/stage1c"
    for k in ("name", "train_path", "out_dir"):
        b.pop(k), c.pop(k)
    assert b == c


def test_stage1c_pool_files_are_unique_and_named_after_families():
    from yev.cli import RULE_GENERATORS
    from yev.generators.llm.targets import PLANNED_FAMILIES

    seen: dict[str, str] = {}
    for b in C["blocks"]:
        for f in b["files"]:
            assert f not in seen, (f, seen.get(f), b["name"])
            seen[f] = b["name"]
    rule_families = {d.family for gen in RULE_GENERATORS.values() for d in gen(2, 0)}
    allowed = set(PLANNED_FAMILIES) | rule_families | {"agent_overreach_rules", "public_long", "long_workflow_more"}
    pool = [f for f in seen if f.startswith("data/stage1c-pool/")]
    assert pool
    for f in pool:
        stem = Path(f).name.removesuffix(".jsonl")
        assert Path(f).name.endswith(".jsonl") and stem in allowed, f
