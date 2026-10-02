import pytest

from yev import licences
from yev.sources.registry import SOURCES, by_name


def test_names_unique_and_all_sources_licence_clean():
    names = [s.name for s in SOURCES]
    assert len(names) == len(set(names))
    for s in SOURCES:
        licences.check(s.hf_id, s.licence, config=s.config, split=s.split)


def test_label_universe_sources_declare_a_label_column():
    for name in ["banking77", "clinc150", "go_emotions", "it_support"]:
        assert by_name(name).label_column


def test_by_name_unknown_raises():
    with pytest.raises(KeyError):
        by_name("nope")


@pytest.mark.network
def test_fetch_maps_classlabel_to_names():
    from itertools import islice

    from yev.sources.base import fetch

    rows = list(islice(fetch(by_name("banking77")), 3))
    assert all(isinstance(r["label"], str) for r in rows)


def test_snli_cf_uses_parquet_branch_and_agent_actions_uses_train_file():
    snli = by_name("snli_cf")
    assert (snli.config, snli.revision) == ("default", "refs/convert/parquet")
    aa = by_name("agent_actions")
    assert (aa.hf_id, aa.data_files, aa.licence) == ("karanxa/agent-action-safety-dataset", "train.jsonl", "apache-2.0")
