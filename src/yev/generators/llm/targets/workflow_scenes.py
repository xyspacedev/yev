"""Scene tables for the workflow-decision planned families (targets.workflow and targets.workflow_invoice).

Every trade, item, cue, record and rule is invented for this project and written in our own words; none is taken
from a benchmark. Placeholders in braces are filled per group by the planners.
"""

from __future__ import annotations

# --- invoice_line ---------------------------------------------------------------------------------

# Units an invoice line can be priced in: singular, plural, our record source, the other unit our systems also log
# for the same work (singular, plural, source), and the vendor's own paperwork.
UNIT_KINDS: dict[str, dict] = {
    "visit": {"sg": "visit", "pl": "visits", "src": "our site visit log",
              "alt": ("hour on site", "hours on site", "our site sign-in register"),
              "vendor": "the vendor's job sheets"},
    "hour": {"sg": "hour", "pl": "hours", "src": "our approved timesheets",
             "alt": ("day on site", "days on site", "our site sign-in register"),
             "vendor": "the vendor's own timesheet"},
    "day": {"sg": "day", "pl": "days", "src": "our site sign-in register",
            "alt": ("hour", "hours", "our approved timesheets"),
            "vendor": "the vendor's attendance sheet"},
    "goods": {"src": "our goods-in receiving log",
              "alt": ("delivery", "deliveries", "our gatehouse delivery log"),
              "vendor": "the vendor's delivery notes"},
}
SECOND_OWN_SOURCE = "our project lead's monthly sign-off sheet"   # the other own record when two of ours disagree

# Each trade: four priced order items (order wording, a different invoice wording for the same thing, unit kind,
# singular and plural for goods, rate range, quantity range); a part the rate of one item already includes; a
# similar-sounding extra the same item's text leaves out; two plainly separate extras; one point the order leaves
# open. Rates are whole currency units; amounts are printed by the planner.
TRADES: tuple[dict, ...] = (
    {
        "trade": "building services maintenance",
        "items": (
            ("Monthly planned maintenance visit to the rooftop air-handling units",
             "Scheduled upkeep call, rooftop air handlers", "visit", None, None, (180, 320), (2, 5)),
            ("Three-monthly test of the emergency lighting on every floor",
             "Emergency light inspection round (quarterly)", "visit", None, None, (220, 400), (1, 2)),
            ("Maintenance engineer time on site, weekdays", "Technician attendance, Monday to Friday",
             "hour", None, None, (48, 72), (12, 40)),
            ("Treatment fluid for the chilled-water loop", "Cooling circuit dosing chemical",
             "goods", "litre", "litres", (6, 11), (40, 120)),
        ),
        "part": (0, "pleated air filters for each unit", "Pleated air filters, 592 x 592 mm", "filter", "filters",
                 (14, 26), (8, 24)),
        "near": (0, "repairs to the units",
                 "Repair to rooftop air-handling unit 2, replacement drive motor and fitting labour",
                 "hour", None, None, (55, 80), (4, 10)),
        "extras": (
            ("Supply and fitting of a carbon monoxide alarm in plant room B", "goods", "alarm", "alarms",
             (90, 160), (1, 3)),
            ("Survey of the basement pipework for lagging condition", "day", None, None, (420, 640), (1, 2)),
        ),
        "open": (2, "call-outs outside weekday working hours", "Engineer call-out, Saturday night",
                 "visit", None, None, (240, 380), (1, 2)),
    },
    {
        "trade": "commercial cleaning",
        "items": (
            ("Nightly office cleaning, floors one to three", "Evening clean of levels 1-3", "visit",
             None, None, (140, 260), (18, 23)),
            ("Washroom hygiene service, all washrooms", "Toilet and washroom sanitary care",
             "visit", None, None, (60, 120), (4, 8)),
            ("Day porter cover at reception", "Daytime janitor on duty, front desk area",
             "hour", None, None, (19, 28), (60, 160)),
            ("Paper hand towels, boxed", "Folded hand-drying paper, cases", "goods", "box", "boxes",
             (18, 34), (10, 40)),
        ),
        "part": (1, "refills for the hand soap dispensers", "Hand soap refill cartridges, 800 ml", "cartridge",
                 "cartridges", (4, 9), (20, 60)),
        "near": (0, "window cleaning", "Office window cleaning, floors one to three, inside and out",
                 "visit", None, None, (300, 520), (1, 2)),
        "extras": (
            ("Deep clean of the staff kitchen after a burst pipe", "hour", None, None, (26, 40), (6, 14)),
            ("Carpet shampoo of the boardroom", "visit", None, None, (180, 300), (1, 1)),
        ),
        "open": (2, "porter cover on public holidays", "Day porter cover, bank holiday Monday",
                 "hour", None, None, (28, 42), (6, 10)),
    },
    {
        "trade": "IT support",
        "items": (
            ("Remote helpdesk support, business hours", "Service desk assistance by phone and remote session",
             "hour", None, None, (55, 90), (20, 70)),
            ("On-site engineer visit for desk moves", "Field technician attendance for workstation relocation",
             "visit", None, None, (150, 260), (2, 6)),
            ("Laptop docking stations, model DS-40", "USB-C desk docks (DS-40)", "goods", "dock", "docks",
             (110, 190), (4, 15)),
            ("Monthly patching of the office servers", "Server update and patch run (monthly)", "visit",
             None, None, (300, 520), (1, 1)),
        ),
        "part": (2, "a power adapter with each dock", "Power adapters for DS-40 docks", "adapter", "adapters",
                 (18, 35), (4, 15)),
        "near": (3, "server hardware replacement",
                 "Office server maintenance, swap of failed storage drives in server 2",
                 "hour", None, None, (70, 110), (3, 8)),
        "extras": (
            ("Setting up a guest wifi network on the second floor", "hour", None, None, (65, 95), (4, 12)),
            ("Recovery of deleted files from a former employee's mailbox", "hour", None, None, (70, 105), (2, 6)),
        ),
        "open": (0, "support calls made after 18:00", "Helpdesk support, evening calls after 18:00",
                 "hour", None, None, (75, 120), (3, 9)),
    },
    {
        "trade": "catering supplies",
        "items": (
            ("Fresh milk, two-litre bottles", "Semi-skimmed milk, 2 L", "goods", "bottle", "bottles",
             (2, 4), (60, 200)),
            ("Coffee beans, medium roast, one-kilo bags", "Whole bean coffee, 1 kg, house roast", "goods", "bag",
             "bags", (16, 28), (10, 40)),
            ("Weekly service of the bean-to-cup coffee machines", "Coffee machine upkeep visit (weekly)",
             "visit", None, None, (45, 90), (4, 5)),
            ("Fruit boxes for the staff kitchens", "Seasonal fruit crates, kitchen delivery", "goods", "box",
             "boxes", (22, 38), (8, 20)),
        ),
        "part": (2, "descaling tablets used at each service", "Descaler tablets for coffee machines, tubs",
                 "tub", "tubs", (9, 16), (4, 10)),
        "near": (2, "repairs to the machines",
                 "Coffee machine repair, replacement brewing unit for machine 3", "visit", None, None,
                 (160, 280), (1, 1)),
        "extras": (
            ("Catering platters for the open day buffet", "goods", "platter", "platters", (35, 60), (6, 20)),
            ("Hire of a water chiller for the summer fair", "day", None, None, (40, 75), (2, 4)),
        ),
        "open": (0, "oat and soy milk alternatives", "Oat drink, 1 L cartons", "goods", "carton", "cartons",
                 (2, 4), (24, 80)),
    },
    {
        "trade": "grounds maintenance",
        "items": (
            ("Fortnightly grass cutting of the front and rear lawns", "Lawn mowing, front and back (every two weeks)",
             "visit", None, None, (120, 220), (2, 3)),
            ("Gardener time for bed weeding and pruning", "Planting bed upkeep labour", "hour", None, None,
             (24, 38), (10, 40)),
            ("Bark mulch for the planting beds, bulk bags", "Wood chip ground cover, bulk sacks", "goods", "bag",
             "bags", (55, 90), (2, 8)),
            ("Litter pick of the car park and paths", "Car park and footpath tidy round", "visit", None, None,
             (40, 80), (4, 9)),
        ),
        "part": (0, "removal of the grass cuttings", "Green waste removal, lawn clippings", "load", "loads",
                 (35, 60), (2, 3)),
        "near": (1, "tree work", "Tree surgery, crown reduction of the two oaks by the rear lawn",
                 "day", None, None, (520, 800), (1, 2)),
        "extras": (
            ("Gritting of the car park during the cold snap", "visit", None, None, (90, 150), (2, 5)),
            ("Repair of the broken bench by the main entrance", "hour", None, None, (30, 45), (2, 5)),
        ),
        "open": (3, "emptying of the outdoor bins", "Outdoor bin emptying round", "visit", None, None,
                 (35, 70), (4, 9)),
    },
    {
        "trade": "security guarding",
        "items": (
            ("Night guard cover at the main gate", "Overnight gatehouse officer", "hour", None, None, (21, 32),
             (120, 300)),
            ("Mobile patrol visits to the warehouse yard", "Drive-by security checks, warehouse yard", "visit",
             None, None, (25, 55), (20, 60)),
            ("Key holding and alarm response", "Alarm call attendance and key holding service", "visit", None,
             None, (60, 110), (1, 4)),
            ("Event steward cover", "Crowd marshal staffing for events", "day", None, None, (180, 260), (1, 4)),
        ),
        "part": (0, "the guard's radio and body camera", "Two-way radio and body camera hire for gate officer",
                 "set", "sets", (8, 15), (20, 31)),
        "near": (1, "patrols inside the buildings",
                 "Mobile patrol visits inside the warehouse buildings, all floors", "visit", None, None,
                 (35, 70), (10, 30)),
        "extras": (
            ("Security escort for the cash collection on payday", "visit", None, None, (85, 140), (1, 2)),
            ("Fitting of a padlock and chain to the side gate", "goods", "lock set", "lock sets", (40, 75), (1, 2)),
        ),
        "open": (0, "gate cover on Sundays", "Gatehouse officer, Sunday day shift", "hour", None, None,
                 (26, 38), (8, 12)),
    },
)

VENDOR_WORDS: tuple[str, ...] = (
    "Hollowmere", "Brackenfield", "Ostrander", "Tamsworth", "Gildenrow", "Pellbridge", "Corvane", "Whitlock-Ames",
    "Ferrisdale", "Quarrington",
)
VENDOR_SUFFIXES: tuple[str, ...] = ("Services Ltd", "& Partners", "Group", "Facilities", "Supply Co.", "Contracts")
OTHER_ORGS: tuple[str, ...] = (
    "Marlpit Dental Group", "Norrington Hall Hotel", "Ashdown Printing Works", "Fallowfield Primary School",
    "Kestrel Lane Surgery", "Brightmoor Self Storage", "Juniper Court Apartments", "Orchard Row Library",
)
STREETS: tuple[str, ...] = (
    "Tanner Street", "Holloway Road", "Mill Lane", "Station Approach", "Cobden Way", "Larch Avenue",
    "Wharf Road", "Beacon Hill", "Quarry Close", "Sedgefield Park",
)
MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)
OWNER_ROLES: tuple[str, ...] = ("facilities manager", "project owner", "office manager", "site lead",
                                "operations manager")
PAYMENT_TERMS: tuple[str, ...] = ("Net 30", "Net 45", "Due 30 days from invoice date", "Net 14")
COVER_NOTES: tuple[str, ...] = (
    "sends the invoice for the month and asks that payment follow the usual terms",
    "attaches the invoice, thanks the team for the work this month and gives a phone number for queries",
    "sends the invoice and mentions that the vendor's office will be closed for two days next week",
)
QUIET_MESSAGES: tuple[str, ...] = (
    "a note from our finance lead reminding the team that invoices must quote the purchase order number",
    "an email from the vendor's account manager announcing a new contact address for remittance advice, "
    "with the bank details unchanged",
    "a message from our reception asking whether the vendor's staff need visitor passes next month",
    "a reminder from our accounts system that the vendor's insurance certificate is on file until next year",
)

# --- cs_scales_actions ----------------------------------------------------------------------------

# What the customer's problem is (the chat is about this; nothing here sets a level).
CS_PROBLEMS: tuple[str, ...] = (
    "an order that was marked delivered but never arrived", "an account they cannot sign in to",
    "a booking that shows the wrong date", "a device that stopped working after an update",
    "a refund that has not reached their account", "a subscription that was renewed at the wrong plan",
    "a voucher code that the checkout keeps rejecting", "a replacement part that was sent in the wrong size",
)
CS_EVENTS: tuple[str, ...] = (
    "a family trip", "an inspection at their premises", "the end of their billing cycle", "a product launch",
    "their move to a new flat", "a board meeting", "their daughter's birthday party", "a trade fair",
)
CS_HARMS: tuple[str, ...] = (
    "their online shop cannot take a single payment while this lasts",
    "their whole team is locked out and cannot work at all until it is fixed",
    "orders from their own customers are being turned away every few minutes",
    "their clinic's booking screen is down and patients are waiting at the desk",
)
# Urgency cues by level. {n} is a day count printed as a word or figure, {event} and {harm} come from the tables.
URGENCY_CUES: dict[int, tuple[str, ...]] = {
    0: ("says there is no hurry at all and that nothing depends on when it is sorted",
        "says they are only asking so they know for next time and are happy to wait as long as it takes"),
    1: ("says they would like it sorted within the next {n} days, before {event}",
        "mentions that {event} is {n} days away and they would like it fixed before then"),
    2: ("says they need it sorted within {n} days, before {event}",
        "explains that {event} is in {n} days and it has to be fixed before that"),
    3: ("says they need it sorted by tomorrow morning, before {event}",
        "says it has to be fixed today, because {event} is this evening"),
    4: ("says that right now {harm}",
        "says the problem is costing them right now: {harm}"),
}
URGENCY_DAYS: dict[int, tuple[int, int]] = {1: (10, 30), 2: (3, 7)}   # level 3 is today or tomorrow
NTH: tuple[str, ...] = ("third", "fourth", "fifth")
# Frustration cues by level; {nth} is third, fourth or fifth.
FRUSTRATION_CUES: dict[int, tuple[str, ...]] = {
    0: ("stays calm and friendly throughout, makes no complaint about the service, and this is their first time "
        "getting in touch about it",),
    1: ("stays polite throughout but remarks once that the problem is a nuisance; this is their first time "
        "getting in touch about it",),
    2: ("says plainly that they are unhappy with how this has gone and that this is their second time getting in "
        "touch about it, while staying civil",),
    3: ("says this is their {nth} time getting in touch about it, writes sharply and says they are running out of "
        "patience, but never threatens to leave or cancel",),
    4: ("says this is their {nth} time getting in touch about it, writes in anger with capital letters and "
        "exclamation marks, and says they will cancel and take their business elsewhere unless it is fixed",),
}
CS_OWED: tuple[str, ...] = (
    "a photo of the item's label", "the order confirmation email", "a screenshot of the error message",
    "the serial number from the back of the device", "a copy of their bank statement line",
)

# Conditions the team rules can test, as (name, wording). Each is a plain fact of the chat.
CS_CONDITIONS: dict[str, str] = {
    "person": "the customer has asked to speak to a staff member",
    "owes": "the customer still has to send us something we asked for",
    "unknown": "the customer reports a payment they do not recognise",
    "u2": "the customer's time limit is seven days away or less, or they are losing something right now",
    "u3": "the customer needs it today or by tomorrow, or they are losing something right now",
    "u4": "the customer is losing something right now while the problem lasts",
    "f3": "this is at least the customer's third time getting in touch about the problem",
    "f4": "the customer threatens to cancel or take their business elsewhere",
}
# How the rules word each action (different from the option descriptions).
CS_ACTION_PHRASES: dict[str, str] = {
    "flag_for_review": "send the ticket to the supervisor review queue",
    "set_pending": "park the ticket as waiting on the customer",
    "handoff_human": "transfer the chat to a staff member",
    "raise_priority": "bump the ticket to the urgent lane",
    "reply_close": "answer and close the ticket",
}
# Three policy variants: ordered (condition, action) rules, first match wins, then a fallback action.
CS_POLICIES: dict[str, tuple[tuple[tuple[str, str], ...], str]] = {
    "people_first": ((("person", "handoff_human"), ("f3", "handoff_human"), ("unknown", "flag_for_review"),
                      ("u3", "raise_priority"), ("owes", "set_pending")), "reply_close"),
    "risk_first": ((("unknown", "flag_for_review"), ("u2", "raise_priority"), ("f4", "handoff_human"),
                    ("owes", "set_pending"), ("person", "handoff_human")), "reply_close"),
    "self_serve": ((("owes", "set_pending"), ("u4", "raise_priority"), ("f4", "handoff_human"),
                    ("unknown", "flag_for_review")), "reply_close"),
}

# --- security_context -----------------------------------------------------------------------------

# Alert kinds. `label` is the record kind that can cover the activity; `user`, `place`, `time` and `status` are its
# field labels (status None: the record has none); `alert_place` is how the alert labels the place; `faults` are the
# single-field faults the kind allows.
SEC_KINDS: tuple[dict, ...] = (
    {"kind": "change", "rule": "Configuration change outside a maintenance window", "alert_place": "Host",
     "label": "Change ticket", "prefix": "CHG", "user": "Requested by", "place": "Host", "time": "Window",
     "status": "Status", "activity": "{user} pushed a new configuration to {place}",
     "faults": ("cancelled", "submitted", "closed_before", "other_user", "other_place")},
    {"kind": "travel", "rule": "Sign-in from a country not seen before for this account",
     "alert_place": "Sign-in country", "label": "Travel request", "prefix": "TRV", "user": "Traveller",
     "place": "Destination", "time": "Dates", "status": "Status",
     "activity": "{user} signed in to the staff portal from {place}",
     "faults": ("cancelled", "submitted", "closed_before", "other_user", "other_place")},
    {"kind": "badge", "rule": "Console login at a server room terminal after hours", "alert_place": "Room",
     "label": "Badge log", "prefix": "BDG", "user": "Person", "place": "Door", "time": "Presence", "status": None,
     "activity": "{user} logged in at the console terminal inside {place}",
     "faults": ("badge_absent", "other_user", "other_place")},
    {"kind": "access", "rule": "Bulk export from a restricted data store", "alert_place": "System",
     "label": "Access approval", "prefix": "ACC", "user": "Person", "place": "System", "time": "Valid",
     "status": "Status", "activity": "{user} exported 48,000 rows from {place}",
     "faults": ("cancelled", "submitted", "closed_before", "other_user", "other_place")},
)
SEC_PLACES: dict[str, tuple[str, ...]] = {
    "change": ("edge-fw-{n:02d}", "core-sw-{n:02d}", "vpn-gw-{n:02d}"),
    "travel": ("Portugal", "Chile", "Vietnam", "Kenya", "Iceland", "Canada", "Japan", "Morocco"),
    "badge": ("server room {n}A", "server room {n}B", "comms room {n}C"),
    "access": ("payroll-db-{n:02d}", "claims-store-{n:02d}", "member-data-{n:02d}"),
}
SEC_PURPOSES: dict[str, tuple[str, ...]] = {
    "change": ("rule update for the new office link", "removal of an old partner tunnel", "logging change"),
    "travel": ("client workshop", "conference talk", "site visit to a supplier"),
    "access": ("quarterly audit extract", "migration test", "regulator data request"),
}
SEC_STATUS_OK = "approved"
SEC_STATUS_FAULT = {"cancelled": "cancelled", "submitted": "submitted, waiting for approval"}
SEC_TEAMS: tuple[str, ...] = ("network operations", "finance systems", "member services IT", "data platform",
                              "field engineering", "service desk")
SEC_BANDS: tuple[str, ...] = ("low", "medium", "high", "critical")
# How the playbook words each action (different from the option descriptions).
SEC_ACTION_PHRASES: dict[str, str] = {
    "close_explained": "close it with no further action",
    "notify_user": "ask the user named in the alert to confirm it was them",
    "open_investigation": "open a case for an analyst",
    "contain_now": "lock the account or cut the host off the network at once",
}
# Playbook variants: (accounted for?, band) -> action.
SEC_PLAYBOOKS: dict[str, dict[tuple[bool, str], str]] = {
    "standard": {(True, "low"): "close_explained", (True, "medium"): "close_explained",
                 (True, "high"): "notify_user", (True, "critical"): "notify_user",
                 (False, "low"): "notify_user", (False, "medium"): "open_investigation",
                 (False, "high"): "open_investigation", (False, "critical"): "contain_now"},
    "strict": {(True, "low"): "close_explained", (True, "medium"): "close_explained",
               (True, "high"): "close_explained", (True, "critical"): "notify_user",
               (False, "low"): "open_investigation", (False, "medium"): "open_investigation",
               (False, "high"): "contain_now", (False, "critical"): "contain_now"},
    "cautious": {(True, "low"): "close_explained", (True, "medium"): "notify_user",
                 (True, "high"): "notify_user", (True, "critical"): "notify_user",
                 (False, "low"): "notify_user", (False, "medium"): "notify_user",
                 (False, "high"): "open_investigation", (False, "critical"): "contain_now"},
}
SEC_DEFINITION = (
    "An alert counts as accounted for only when one single record covers the activity: the record names the same "
    "user and the same host, room, destination or system as the alert, its time span includes the time of the "
    "activity, and, where the record has a status, that status is approved. Records that cover only part of this "
    "do not count."
)
