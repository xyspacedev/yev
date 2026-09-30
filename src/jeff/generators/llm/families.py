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
