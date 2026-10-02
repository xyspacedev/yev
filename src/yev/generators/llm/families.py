"""What the Opus writers generate: families (spec §4.1) and the domains they are set in."""

from __future__ import annotations

# name -> (question type, what a decision in this family looks like, whether a written policy is in the options)
FAMILIES: dict[str, tuple[str, str, bool]] = {
    "agent_action": ("choice", "An AI agent proposes an action (a tool call, command, payment or message). A written policy in the options decides whether it runs, needs a person's sign-off, or is refused.", True),
    "returns": ("choice", "A customer's return or refund request, judged against a written returns policy with time windows, conditions and exceptions.", True),
    "moderation": ("choice", "A user's post or message, judged against a written community policy: keep it, restrict or label it, or remove it.", True),
    "expenses": ("choice", "An employee expense claim, judged against a written expense policy: approve, approve in part, or reject.", True),
    "access_request": ("choice", "A request for access to a system or dataset, judged against a written access policy.", True),
    "claim_evidence": ("choice", "A claim and a short evidence passage: the evidence supports it, contradicts it, or does not settle it.", False),
    "routing": ("choice", "An assistant must pick which tool or team should handle a request; one option is 'none of these fits'.", False),
    "intent": ("choice", "A customer message and a set of possible intents; one option is 'none of these'.", False),
    "sentiment": ("score", "A customer review written naturally, with no rules or checklists. The options are tone levels from very negative to very positive, and the rating follows the reviewer's overall tone.", False),
    "severity": ("score", "A bug report or incident, rated on an ordered four-point severity scale.", False),
}

DOMAINS: tuple[str, ...] = (
    "regional airline", "veterinary clinic", "public library", "bike-share scheme", "craft brewery",
    "municipal water utility", "online bookstore", "hotel chain", "university registrar", "dental practice",
    "car rental agency", "grocery delivery app", "coworking space", "insurance broker", "ski resort",
    "food bank", "museum gift shop", "indie game studio", "solar installer", "pharmacy chain",
    "freight forwarder", "language-learning app", "wedding planner", "city parking authority", "robotics startup",
    "orchestra box office", "home-cleaning marketplace", "credit union", "podcast network", "garden centre",
    "telehealth provider", "furniture maker", "esports league", "ferry operator", "pet insurance firm",
    "school district IT", "open-source foundation", "used-car dealer", "farmers' cooperative", "art supply store",
    "climbing gym", "payroll provider", "tax preparation service", "recycling contractor", "streaming service",
    "wine subscription club", "medical device maker", "event ticketing platform", "property manager", "courier startup",
    "fitness tracker brand", "print shop", "theme park", "legal aid clinic", "mobile carrier",
    "data-labelling vendor", "cloud backup service", "toy manufacturer", "bakery chain", "research lab",
    "charity shop network", "airport lounge operator", "e-bike retailer", "translation agency", "cosmetics brand",
)

# Family 6, long_workflow: long documents written by Opus, each carrying several typed questions.
# topic name -> (question type, what the question asks). A question's `name` is its topic name.
DOC_TOPICS: dict[str, tuple[str, str]] = {
    "next_action": ("choice", "What the person or agent handling the case should do next, given everything in the document so far."),
    "policy_compliance": ("choice", "Whether a specific action taken or proposed in the document follows the written policy or procedure quoted in the document."),
    "escalation": ("choice", "Which team, role or person the case should go to now, under the routing rules stated in the document."),
    "root_cause": ("choice", "Which cause the evidence in the document supports; one option says the document does not settle it."),
    "claim_support": ("choice", "Whether the document supports, contradicts, or does not settle a specific claim someone makes about the case."),
    "tool_choice": ("choice", "Which tool, command or step an agent should use next, given its instructions and the outputs so far."),
    "error_detection": ("choice", "Which step, message or entry in the document is the first one that is wrong (a bad calculation, a broken rule, a misread output)."),
    "resolution_status": ("choice", "The true current state of the issue or task at the end of the document, as opposed to what someone claimed."),
    "commitment_check": ("choice", "What was actually promised or agreed, by whom, and by when, after later messages corrected earlier ones."),
    "numeric_reasoning": ("choice", "A quantity that must be worked out from figures spread across the document (amount owed, hours of downtime, units left); options are candidate values."),
    "customer_sentiment": ("score", "The tone of the customer or user at the end of the document, on a numbered scale defined in the options."),
    "severity": ("score", "How severe the incident or problem is, on a numbered scale whose levels are defined in the options."),
    "risk": ("score", "How risky a proposed action is, on a numbered scale whose levels are defined in the options."),
}

# kind -> (what the document is, the question topics it should carry)
DOC_KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "support_thread": (
        "An email or chat thread between a customer and a support team over several days, with internal notes, "
        "order or account details, and the policy excerpts the agents quote.",
        ("next_action", "policy_compliance", "escalation", "claim_support", "resolution_status",
         "numeric_reasoning", "customer_sentiment"),
    ),
    "incident_report": (
        "A post-incident report or a live incident channel log: timeline, alerts, actions taken, impact figures, "
        "and the on-call runbook excerpt the responders follow.",
        ("severity", "root_cause", "next_action", "claim_support", "numeric_reasoning", "policy_compliance",
         "escalation", "error_detection"),
    ),
    "agent_trace": (
        "The step-by-step log of an AI agent working a task: the operator's instructions and rules, each tool "
        "call with its arguments, each tool output, and the agent's messages to the user.",
        ("tool_choice", "error_detection", "policy_compliance", "risk", "next_action", "claim_support",
         "resolution_status"),
    ),
    "multi_turn_transcript": (
        "A long multi-turn conversation between a user and an assistant, or between two staff members, in which "
        "requirements change and earlier statements get corrected.",
        ("next_action", "claim_support", "commitment_check", "customer_sentiment", "resolution_status",
         "numeric_reasoning", "risk"),
    ),
    "ops_handover": (
        "A shift or project handover note: open items, a status table, pending decisions, owners and deadlines, "
        "plus the procedure the incoming shift must follow.",
        ("next_action", "commitment_check", "escalation", "risk", "numeric_reasoning", "policy_compliance",
         "resolution_status"),
    ),
}
