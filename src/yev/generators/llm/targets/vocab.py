"""Shared vocabulary for planned families: tool names, staff roles, personas, payload kinds, internal systems.

Every entry is invented for this project. None is taken from any benchmark, dataset or real product; later tasks
may extend these lists under the same rule.
"""

from __future__ import annotations

# Tool names an assistant in a log can call (snake_case, as they appear in [TOOL] lines).
TOOL_NAMES: tuple[str, ...] = (
    "ledger_peek", "parcel_locator", "roster_board", "refund_desk", "inventory_count", "calendar_slotter",
    "contact_card", "invoice_vault", "ticket_scribe", "badge_issuer", "quote_builder", "note_pinboard",
    "shipment_relabel", "meter_reader", "policy_shelf", "file_courier",
)

# Staff roles that appear in scenes and logs.
STAFF_ROLES: tuple[str, ...] = (
    "front-desk coordinator", "shift lead", "billing clerk", "returns specialist", "site supervisor",
    "facilities planner", "accounts assistant", "dispatch officer", "member services agent", "records keeper",
    "duty manager", "procurement assistant",
)

# Short persona sketches for the person an assistant is helping.
PERSONAS: tuple[str, ...] = (
    "a retired teacher booking a repair visit", "a night-shift nurse checking a delivery",
    "a first-year student disputing a late fee", "a small bakery owner reordering supplies",
    "a volunteer coordinator moving an event date", "a freelance translator chasing an unpaid invoice",
    "a parent changing a pickup address", "a warehouse temp asking about overtime pay",
    "a landlord reporting a broken meter", "a hobby beekeeper returning damaged equipment",
)

# Kinds of payload a tool result or pasted text can carry.
PAYLOAD_KINDS: tuple[str, ...] = (
    "pasted email footer", "shared spreadsheet cell", "calendar invite description", "customer review text",
    "scanned receipt note", "chat transcript excerpt", "web page snippet", "support form comment",
    "file attachment caption", "product listing blurb",
)

# Names of internal company systems mentioned in scenes.
INTERNAL_SYSTEMS: tuple[str, ...] = (
    "Larkspur CRM", "Tidewell Billing", "Copperline Rota", "Brambleworks Inventory", "Quillstone Ticketing",
    "Marrowgate Payroll", "Fernhollow Records", "Sablecrest Dispatch",
)
