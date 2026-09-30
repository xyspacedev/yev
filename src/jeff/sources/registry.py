"""Every public source we train on, with its licence and pool size."""

from __future__ import annotations

from jeff.sources.base import SourceSpec
from jeff.sources.agent_actions import convert_agent_action_safety
from jeff.sources.fast_decisions import FAST_DECISIONS_CONFIGS, convert_fast_decisions
from jeff.sources.intent import convert_go_emotions, intent_converter
from jeff.sources.moderation import convert_aegis, convert_civil_comments, convert_dynabench
from jeff.sources.nli import convert_multi_nli, convert_negation, convert_snli_cf, convert_vitaminc, convert_wanli
from jeff.sources.replay import convert_boolq, convert_multiple_choice
from jeff.sources.rules import convert_eikos, convert_ruletaker
from jeff.sources.triage import convert_cvss, convert_help_desk, convert_it_support

SOURCES: list[SourceSpec] = [
    # evidence
    SourceSpec("vitaminc", "tals/vitaminc", None, "train", "cc-by-sa-3.0", convert_vitaminc, 12000),
    SourceSpec("wanli", "alisawuffles/WANLI", None, "train", "cc-by-4.0", convert_wanli, 6000),
    SourceSpec("multi_nli", "nyu-mll/multi_nli", None, "train", "oanc-mixed", convert_multi_nli, 4000),
    SourceSpec("snli_cf", "sagnikrayc/snli-cf-kaushik", "default", "train", "cc-by-4.0", convert_snli_cf, 3000,
               revision="refs/convert/parquet"),
    SourceSpec("negation", "jinaai/negation-dataset", None, "train", "apache-2.0", convert_negation, 2000),
    # moderation
    SourceSpec("dynabench", "montehoover/DynaBench", "DynaBenchTrain", "train", "mit", convert_dynabench, 12000),
    SourceSpec("aegis", "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "train", "cc-by-4.0", convert_aegis, 6000),
    SourceSpec("civil_comments", "google/civil_comments", None, "train", "cc0-1.0", convert_civil_comments, 2000,
               max_scan=300_000),
    # rule application
    SourceSpec("eikos", "caiovicentino1/eikos-decisions", "core", "train", "cc-by-4.0", convert_eikos, 12000),
    SourceSpec("ruletaker", "tasksource/ruletaker", None, "train", "apache-2.0", convert_ruletaker, 3000,
               max_scan=100_000),
    # action review (Gemini-written, Apache-2.0; spec §2 rule 7)
    SourceSpec("agent_actions", "karanxa/agent-action-safety-dataset", None, "train", "apache-2.0",
               convert_agent_action_safety, 10000, data_files="train.jsonl"),
    # intent and sentiment
    SourceSpec("banking77", "legacy-datasets/banking77", None, "train", "cc-by-4.0",
               intent_converter("text", "label"), 4000, label_column="label"),
    SourceSpec("clinc150", "clinc/clinc_oos", "plus", "train", "cc-by-3.0",
               intent_converter("text", "intent", oos_label="oos"), 4000, label_column="intent"),
    SourceSpec("go_emotions", "google-research-datasets/go_emotions", "simplified", "train", "apache-2.0",
               convert_go_emotions, 2000, label_column="labels"),
    # triage
    SourceSpec("help_desk", "tasksource/help-desk-tickets", None, "train", "cc-by-4.0", convert_help_desk, 400),
    SourceSpec("cvss", "AgileRLArena/vulnerability-scores-cvss-v3", None, "train", "cc-by-4.0", convert_cvss, 3000,
               max_scan=200_000),
    SourceSpec("it_support", "tasksource/it-support-tickets", None, "train", "cc-by-4.0", convert_it_support, 2000,
               label_column="label"),
    # replay
    SourceSpec("boolq", "google/boolq", None, "train", "cc-by-sa-3.0", convert_boolq, 4000),
    SourceSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "train", "cc-by-sa-4.0", convert_multiple_choice, 1000),
    SourceSpec("arc_easy", "allenai/ai2_arc", "ARC-Easy", "train", "cc-by-sa-4.0", convert_multiple_choice, 1000),
    SourceSpec("csqa", "tau/commonsense_qa", None, "train", "mit", convert_multiple_choice, 2000),
]

# fastino/fast-decisions: published dev split (test is private); trained on at the user's request.
SOURCES += [
    SourceSpec(f"fastdec_{config}", "fastino/fast-decisions", config, "train", "apache-2.0",
               convert_fast_decisions, 1000)
    for config in FAST_DECISIONS_CONFIGS
]


# Sources read from a local checkout rather than the Hub. Each writes its own files (several
# families plus attribution), so build-public runs them only when named in --only.
LOCAL_SOURCES = ("skill_atlas",)


def by_name(name: str) -> SourceSpec:
    for spec in SOURCES:
        if spec.name == name:
            return spec
    raise KeyError(name)
