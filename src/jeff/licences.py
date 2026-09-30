"""Which licences and datasets may be trained on (spec §2, rules 3-4)."""

from __future__ import annotations

import re

# Commercially usable. `oanc-mixed` is MultiNLI's own statement (OANC plus CC-BY-SA-3.0 fiction).
ALLOWED = frozenset(
    {"apache-2.0", "mit", "cc0-1.0", "cc-by-3.0", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "oanc-mixed"}
)

# Non-commercial, unclear, proprietary-output, evaluation-only, or licence-forbids-training.
FORBIDDEN_DATASETS = frozenset(
    {
        # non-commercial or unclear licence
        "facebook/anli", "allenai/scifact", "lmsys/toxic-chat", "PKU-Alignment/BeaverTails",
        "kiddothe2b/contract-nli", "takala/financial_phrasebank", "Salesforce/APIGen-MT-5k",
        "ehovy/race", "cais/mmlu", "Yelp/yelp_review_full", "mteb/amazon_reviews_multi",
        "SetFit/amazon_reviews_multi_en", "SetFit/sst5", "fancyzhx/ag_news", "ag_news",
        "fever/fever", "Tobi-Bueck/customer-support-tickets", "DeepPavlov/hwu64",
        "yueliu1999/GuardReasonerTrain",
        # licence forbids training
        "nvidia/Nemotron-AIQ-Agentic-Safety-Dataset-1.0",
        # evaluation-only
        "choyiny/decidebench", "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "ai-safety-institute/AgentHarm", "thu-coai/Agent-SafetyBench", "normster/RuLES",
        "google/IFEval", "nguha/legalbench",
    }
)

# (hf_id, config) pairs that are forbidden even though other configs are fine.
FORBIDDEN_CONFIGS = frozenset(
    {
        ("montehoover/DynaBench", "DynaBench"),  # the benchmark config
        ("montehoover/DynaBench", "DynaBenchSafetyMix"),  # contains BeaverTails and ToxicChat
    }
)

# Derived lowercase versions for case-insensitive comparison
_FORBIDDEN_DATASETS_LC = frozenset(x.lower() for x in FORBIDDEN_DATASETS)
_FORBIDDEN_CONFIGS_LC = frozenset((h.lower(), c) for h, c in FORBIDDEN_CONFIGS)


class LicenceError(ValueError):
    """A source may not be used for training."""


def is_allowed(licence: str) -> bool:
    return licence in ALLOWED


def check(hf_id: str, licence: str, *, config: str | None = None, split: str = "train", data_files: str | None = None) -> None:
    hf_id_lower = hf_id.lower()
    if hf_id_lower in _FORBIDDEN_DATASETS_LC:
        raise LicenceError(f"{hf_id} is forbidden for training")
    if (hf_id_lower, config) in _FORBIDDEN_CONFIGS_LC:
        raise LicenceError(f"{hf_id} config {config} is forbidden for training")
    # Check compound splits: split on "+", strip whitespace, remove slice notation, require all are "train"
    for part in split.split("+"):
        part = part.strip().split("[")[0]
        if part != "train":
            raise LicenceError(f"{hf_id}: only train splits may be used, got split {split!r}")
    if data_files and re.search(r"(^|[^a-z])(val|valid|validation|dev|test)([^a-z]|$)", data_files.lower()):
        raise LicenceError(f"{hf_id}: only train splits may be used, got split-like data_files {data_files!r}")
    if not is_allowed(licence):
        raise LicenceError(f"{hf_id}: licence {licence!r} is not commercially usable")
