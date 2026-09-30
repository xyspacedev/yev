from collections import Counter

from jeff.generators.common import EDIT_TYPES
from jeff.generators.llm.families import DOMAINS, FAMILIES
from jeff.generators.llm.plan import plan_batches, render_writer_prompt


def test_plan_covers_families_evenly_and_is_deterministic():
    batches = plan_batches(200, per_batch=10, seed=0)
    assert batches == plan_batches(200, per_batch=10, seed=0)
    assert sum(b["n_clusters"] for b in batches) == 200
    counts = Counter(b["family"] for b in batches)
    assert set(counts) == set(FAMILIES)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len({b["batch_id"] for b in batches}) == len(batches)
    for b in batches:
        assert b["qtype"] == FAMILIES[b["family"]][0]
        assert len(b["domains"]) == b["n_clusters"] and set(b["domains"]) <= set(DOMAINS)
        assert set(b["edit_types"]) <= set(EDIT_TYPES)
        assert b["policy_edit_share"] == (0.3 if FAMILIES[b["family"]][2] else 0.0)


def test_writer_prompt_contains_contract():
    b = plan_batches(20, per_batch=10, seed=0)[0]
    text = render_writer_prompt(b, "/tmp/out/batch-000.jsonl")
    for needle in ["/tmp/out/batch-000.jsonl", b["family"], '"variants"', '"policy_variants"', "15 tokens",
                   "one JSON object per line", "fictional", *b["domains"]]:
        assert needle in text
    assert "decidebench" not in text.lower()


def test_domains_are_plentiful_and_unique():
    assert len(DOMAINS) >= 60 and len(set(DOMAINS)) == len(DOMAINS)
