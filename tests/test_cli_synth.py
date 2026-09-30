from jeff import cli
from jeff.schema import read_jsonl


def test_gen_rules_writes_valid_clusters(tmp_path):
    out = tmp_path / "rules" / "returns.jsonl"
    assert cli.main(["gen-rules", "--family", "returns", "--clusters", "5", "--seed", "7", "--out", str(out)]) == 0
    rows = list(read_jsonl(out))
    assert len({r.cluster_id for r in rows}) == 5
    assert {r.source for r in rows} == {"synthetic_rules"}


def test_gen_rules_rejects_unknown_family(tmp_path, capsys):
    import pytest

    with pytest.raises(SystemExit):
        cli.main(["gen-rules", "--family", "nope", "--clusters", "1", "--out", str(tmp_path / "x.jsonl")])
