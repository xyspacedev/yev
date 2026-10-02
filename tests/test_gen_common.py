from yev import licences
from yev.generators.common import (
    EDIT_TYPES,
    SYNTH_LICENCE,
    keep_valid_clusters,
    make_cluster,
    token_edit_size,
)
from yev.generators.specs import SYNTHETIC_SPECS
from yev.schema import Option

OPTS = [Option("allow", "Allow it."), Option("deny", "Deny it.")]


def test_token_edit_size_counts_changed_tokens():
    assert token_edit_size("returned on day 30", "returned on day 30") == 0
    assert token_edit_size("returned on day 30", "returned on day 31") == 1
    assert token_edit_size("the item is sealed", "the item is not sealed") == 1
    assert token_edit_size("Staging DB", "production db") == 1


def test_make_cluster_ids_provenance_and_edit_types():
    ds = make_cluster(
        cluster_id="c1", family="f", source="synthetic_rules", qtype="choice", question="What now?",
        members=[("base state", OPTS, "allow", None), ("edited state", OPTS, "deny", "negation")],
    )
    assert [d.id for d in ds] == ["c1:0", "c1:1"]
    assert {d.cluster_id for d in ds} == {"c1"}
    assert [d.edit_type for d in ds] == [None, "negation"]
    assert {d.licence for d in ds} == {SYNTH_LICENCE}
    for d in ds:
        d.validate()


def test_keep_valid_clusters_requires_two_distinct_golds():
    same = make_cluster(cluster_id="a", family="f", source="synthetic_rules", qtype="choice", question="q?",
                        members=[("s1", OPTS, "allow", None), ("s2", OPTS, "allow", "negation")])
    mixed = make_cluster(cluster_id="b", family="f", source="synthetic_rules", qtype="choice", question="q?",
                         members=[("s1", OPTS, "allow", None), ("s2", OPTS, "deny", "negation")])
    lone = make_cluster(cluster_id="c", family="f", source="synthetic_rules", qtype="choice", question="q?",
                        members=[("s1", OPTS, "deny", None)])
    kept = keep_valid_clusters(same + mixed + lone)
    assert {d.cluster_id for d in kept} == {"b"}


def test_edit_types_and_specs_are_licence_clean():
    assert "policy_edit" in EDIT_TYPES and "injection" in EDIT_TYPES and len(EDIT_TYPES) == 9
    assert {s.name for s in SYNTHETIC_SPECS} == {"synthetic_rules", "synthetic_opus"}
    for s in SYNTHETIC_SPECS:
        licences.check(s.hf_id, s.licence, config=s.config, split=s.split, data_files=s.data_files)


def test_spread_into_parts_is_balanced_and_separates_clusters():
    import random

    from yev.generators.common import spread_into_parts

    rng = random.Random(0)
    items, owner = [], []
    for c in range(300):
        for j in range(rng.choice([2, 3, 4])):
            items.append((c, j))
            owner.append(f"c{c}")
    parts = spread_into_parts(items, owner, 100, random.Random(1))
    target = -(-len(items) // 100)
    assert len(parts) == target
    assert sum(len(p) for p in parts) == len(items)
    for p in parts:
        assert abs(len(p) - len(items) / target) <= 4
        assert len({c for c, _ in p}) == len(p)
