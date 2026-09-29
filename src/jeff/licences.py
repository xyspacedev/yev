"""Which licences and datasets may be trained on (spec §2, rules 3-4)."""

from __future__ import annotations

# Commercially usable. `oanc-mixed` is MultiNLI's own statement (OANC plus CC-BY-SA-3.0 fiction).
ALLOWED = frozenset(
    {"apache-2.0", "mit", "cc0-1.0", "cc-by-3.0", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "oanc-mixed"}
)

# Non-commercial, unclear, proprietary-output, evaluation-only, or licence-forbids-training.
FORBIDDEN_DATASETS = frozenset(
    {
        # non-commercial or unclear licence
        "facebook/anli", "allenai/scifact", "lmsys/toxic-chat", "pku-alignment/beavertails",
        "kiddothe2b/contract-nli", "takala/financial_phrasebank", "salesforce/apigen-mt-5k",
        "ehovy/race", "cais/mmlu", "yelp/yelp_review_full", "mteb/amazon_reviews_multi",
        "setfit/amazon_reviews_multi_en", "setfit/sst5", "fancyzhx/ag_news", "ag_news",
        "fever/fever", "tobi-bueck/customer-support-tickets", "deeppalov/hwu64",
        "yueliu1999/guardreasonertrain",
        # proprietary-model outputs
        "karanxa/agent-action-safety-dataset",
        # licence forbids training
        "nvidia/nemotron-aiq-agentic-safety-dataset-1.0",
        # evaluation-only
        "choyiny/decidebench", "gorilla-llm/berkeley-function-calling-leaderboard",
        "ai-safety-institute/agentharm", "thu-coai/agent-safetybench", "normster/rules",
        "google/ifeval", "nguha/legalbench",
    }
)

# (hf_id, config) pairs that are forbidden even though other configs are fine.
FORBIDDEN_CONFIGS = frozenset(
    {
        ("montehoover/dynabench", "DynaBench"),  # the benchmark config
        ("montehoover/dynabench", "DynaBenchSafetyMix"),  # contains BeaverTails and ToxicChat
    }
)


class LicenceError(ValueError):
    """A source may not be used for training."""


def is_allowed(licence: str) -> bool:
    return licence in ALLOWED


def check(hf_id: str, licence: str, *, config: str | None = None, split: str = "train") -> None:
    hf_id_lower = hf_id.lower()
    if hf_id_lower in FORBIDDEN_DATASETS:
        raise LicenceError(f"{hf_id} is forbidden for training")
    if (hf_id_lower, config) in FORBIDDEN_CONFIGS:
        raise LicenceError(f"{hf_id} config {config} is forbidden for training")
    # Check compound splits: split on "+", strip whitespace, remove slice notation, require all are "train"
    for part in split.split("+"):
        part = part.strip().split("[")[0]
        if part != "train":
            raise LicenceError(f"{hf_id}: only train splits may be used, got split {split!r}")
    if not is_allowed(licence):
        raise LicenceError(f"{hf_id}: licence {licence!r} is not commercially usable")
