"""Routing and category questions with 7-10 options (Stage 1, family 5).

WorkflowEvals choice questions list up to 9 options, while Stage 0 trained on at most 6. Two sources:

- `generate`: four fixed taxonomies (support queues, incident categories, document types, expense
  categories), ten categories each. Every category's option text names its trigger keywords, and the
  state (a 2-3 sentence request rendered from templates) contains a trigger phrase of at most one
  category, so the gold follows from the text alone: the category whose keyword appears, or "none of
  these fits" when no listed category's keyword appears. Clusters swap the trigger phrase (one short
  edit) or swap the gold category out of the option list; every cluster has a "none of these fits"
  member (a keyword of an unlisted category, no keyword, or the matching option swapped out).
- `reframe_intents`: existing intent rows (Banking77, CLINC150 pool) re-asked with 7-10 candidate
  intents: the original options plus label-space siblings that share words with the gold label.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from yev.generators.common import keep_valid_clusters, make_cluster
from yev.schema import Decision, Option

SOURCE = "synthetic_rules"
FAMILY = "many_options"
INTENT_FAMILY = "many_options_intent"
NONE_KEY = "none_of_these"
NONE_OPTION = Option(NONE_KEY, "None of these fits.")
NONE_RATE = 0.1  # share of clusters whose base row matches no listed category


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    keywords: tuple[str, ...]
    triggers: tuple[tuple[str, str], ...]  # (phrase for the state, the keyword it contains)


@dataclass(frozen=True)
class Taxonomy:
    question: tuple[str, ...]
    subject: str  # "the request mentions" ...
    templates: tuple[str, ...]
    none_triggers: tuple[str, ...]


def _cat(key: str, label: str, keywords: tuple[str, ...], *triggers: tuple[str, str]) -> Category:
    for phrase, kw in triggers:
        assert kw in keywords and kw.lower() in phrase.lower(), (key, phrase, kw)
    return Category(key, label, keywords, triggers)


TAXONOMIES: dict[str, list[Category]] = {
    "support_queue": [
        _cat("billing", "Billing", ("refund", "invoice", "overcharged"),
             ("a refund for a duplicate payment", "refund"),
             ("an invoice that lists the wrong company name", "invoice"),
             ("being overcharged for the premium plan", "overcharged")),
        _cat("account_access", "Account access", ("password", "locked out", "two-factor"),
             ("a password that no longer works", "password"),
             ("being locked out after too many attempts", "locked out"),
             ("two-factor codes that never arrive", "two-factor")),
        _cat("shipping", "Shipping", ("tracking number", "parcel", "courier"),
             ("a tracking number that shows no updates", "tracking number"),
             ("a parcel left at the wrong address", "parcel"),
             ("a courier who never showed up", "courier")),
        _cat("returns", "Returns", ("return label", "exchange"),
             ("a return label for a jacket that does not fit", "return label"),
             ("an exchange for a different size", "exchange")),
        _cat("technical", "Technical support", ("error message", "keeps freezing", "crash"),
             ("an error message when opening the app", "error message"),
             ("an app that keeps freezing on startup", "keeps freezing"),
             ("a crash every time a photo is uploaded", "crash")),
        _cat("privacy", "Privacy", ("data deletion", "personal data", "GDPR"),
             ("a data deletion request", "data deletion"),
             ("how their personal data is stored", "personal data"),
             ("a GDPR access request", "GDPR")),
        _cat("sales", "Sales", ("quote", "bulk order"),
             ("a quote for forty office seats", "quote"),
             ("a bulk order for the spring catalogue", "bulk order")),
        _cat("product_feedback", "Product feedback", ("feature request", "suggestion"),
             ("a feature request for dark mode", "feature request"),
             ("a suggestion to add a wishlist", "suggestion")),
        _cat("partnerships", "Partnerships", ("reseller", "affiliate"),
             ("becoming a reseller in Portugal", "reseller"),
             ("the affiliate programme terms", "affiliate")),
        _cat("cancellations", "Cancellations", ("cancellation", "cancelling"),
             ("cancelling the yearly subscription", "cancelling"),
             ("a cancellation that was never confirmed", "cancellation")),
    ],
    "incident_category": [
        _cat("phishing", "Phishing", ("suspicious link", "phishing", "spoofed sender"),
             ("an email with a suspicious link asking staff to confirm payroll details", "suspicious link"),
             ("a phishing message pretending to be the CEO", "phishing"),
             ("a spoofed sender address on a payment request", "spoofed sender")),
        _cat("malware", "Malware", ("ransomware", "malware", "trojan"),
             ("ransomware notes appearing on a shared file server", "ransomware"),
             ("antivirus flagging malware on a reception PC", "malware"),
             ("a trojan found in a downloaded installer", "trojan")),
        _cat("data_leak", "Data leak", ("exposed bucket", "leaked credentials", "data leak"),
             ("an exposed bucket containing customer exports", "exposed bucket"),
             ("leaked credentials posted on a paste site", "leaked credentials"),
             ("a data leak through a misaddressed mailing list", "data leak")),
        _cat("outage", "Service outage", ("outage", "unreachable"),
             ("an outage of the customer portal since 9am", "outage"),
             ("the VPN gateway being unreachable from all offices", "unreachable")),
        _cat("lost_device", "Lost or stolen device", ("lost laptop", "stolen phone", "missing badge"),
             ("a lost laptop left on a train", "lost laptop"),
             ("a stolen phone with company email on it", "stolen phone"),
             ("a missing badge that still opens the server room", "missing badge")),
        _cat("access_violation", "Access violation", ("unauthorized access", "privilege escalation"),
             ("unauthorized access to the HR folder by a contractor", "unauthorized access"),
             ("a privilege escalation on the build server", "privilege escalation")),
        _cat("ddos", "Denial of service", ("traffic flood", "DDoS", "botnet"),
             ("a traffic flood hitting the login page", "traffic flood"),
             ("a DDoS attack on the public API", "DDoS"),
             ("botnet requests from thousands of addresses", "botnet")),
        _cat("physical", "Physical security", ("tailgating", "broken door lock", "CCTV"),
             ("tailgating through the loading-dock door", "tailgating"),
             ("a broken door lock on the records room", "broken door lock"),
             ("CCTV footage showing an unknown visitor", "CCTV")),
        _cat("insider", "Insider threat", ("disgruntled employee", "USB drive"),
             ("a disgruntled employee downloading the client list", "disgruntled employee"),
             ("files copied to a USB drive before a resignation", "USB drive")),
        _cat("vulnerability", "Vulnerability", ("unpatched", "CVE", "zero-day"),
             ("an unpatched VPN appliance", "unpatched"),
             ("a new CVE affecting the web framework", "CVE"),
             ("a zero-day in the mail gateway", "zero-day")),
    ],
    "document_type": [
        _cat("invoice", "Invoice", ("invoice number", "amount due", "remit-to address"),
             ("an invoice number and an amount due", "invoice number"),
             ("a remit-to address and payment terms", "remit-to address")),
        _cat("contract", "Contract", ("governing law", "signature blocks", "termination clause"),
             ("a governing law clause", "governing law"),
             ("signature blocks for both parties", "signature blocks"),
             ("a termination clause and a term of three years", "termination clause")),
        _cat("resume", "Resume", ("work experience", "career objective"),
             ("a work experience section and a list of skills", "work experience"),
             ("a career objective at the top", "career objective")),
        _cat("receipt", "Receipt", ("cashier", "thank you for your purchase"),
             ("a cashier name and the change given", "cashier"),
             ("the line 'thank you for your purchase'", "thank you for your purchase")),
        _cat("payslip", "Payslip", ("gross pay", "net pay", "tax withheld"),
             ("gross pay and net pay for March", "gross pay"),
             ("tax withheld and a pension deduction", "tax withheld")),
        _cat("purchase_order", "Purchase order", ("PO number", "requested delivery date"),
             ("a PO number and a ship-to address", "PO number"),
             ("a requested delivery date for 200 units", "requested delivery date")),
        _cat("meeting_minutes", "Meeting minutes", ("attendees", "action items", "motion carried"),
             ("a list of attendees and action items", "attendees"),
             ("a motion carried by six votes to two", "motion carried")),
        _cat("bank_statement", "Bank statement", ("opening balance", "closing balance", "statement period"),
             ("an opening balance and a closing balance", "opening balance"),
             ("a statement period of 1 to 31 May", "statement period")),
        _cat("medical_record", "Medical record", ("diagnosis", "patient history"),
             ("a diagnosis and a list of prescribed drugs", "diagnosis"),
             ("a patient history section", "patient history")),
        _cat("nda", "Non-disclosure agreement", ("confidential information", "non-disclosure"),
             ("a definition of confidential information", "confidential information"),
             ("a non-disclosure clause lasting five years", "non-disclosure")),
    ],
    "expense_category": [
        _cat("travel", "Travel", ("airfare", "train ticket", "rental car"),
             ("airfare to the Lisbon office", "airfare"),
             ("a return train ticket to Leeds", "train ticket"),
             ("a rental car for the site visit", "rental car")),
        _cat("lodging", "Lodging", ("hotel", "Airbnb"),
             ("two nights at a hotel near the venue", "hotel"),
             ("an Airbnb for the offsite week", "Airbnb")),
        _cat("meals", "Meals", ("dinner", "lunch", "catering"),
             ("a team dinner after the launch", "dinner"),
             ("catering for the board meeting", "catering"),
             ("lunch with a visiting auditor", "lunch")),
        _cat("software", "Software", ("licence", "SaaS"),
             ("an annual licence for a diagramming tool", "licence"),
             ("a SaaS seat for the design team", "SaaS")),
        _cat("office_supplies", "Office supplies", ("toner", "printer paper", "stationery"),
             ("toner for the shared printer", "toner"),
             ("a box of printer paper", "printer paper"),
             ("stationery for the new starters", "stationery")),
        _cat("training", "Training", ("course fee", "conference ticket", "workshop"),
             ("a course fee for a project management certificate", "course fee"),
             ("a conference ticket for a security summit", "conference ticket"),
             ("a two-day workshop on negotiation", "workshop")),
        _cat("equipment", "Equipment", ("monitor", "headset", "docking station"),
             ("a second monitor for home working", "monitor"),
             ("a replacement headset", "headset"),
             ("a docking station", "docking station")),
        _cat("client_gifts", "Client gifts", ("gift basket", "client gift"),
             ("a gift basket for a customer's anniversary", "gift basket"),
             ("a client gift for closing the deal", "client gift")),
        _cat("telecom", "Telecom", ("phone bill", "roaming", "mobile plan"),
             ("the monthly phone bill", "phone bill"),
             ("roaming charges during a trip abroad", "roaming"),
             ("an upgrade to the mobile plan", "mobile plan")),
        _cat("mileage", "Mileage", ("mileage", "kilometres driven"),
             ("mileage for driving to a supplier", "mileage"),
             ("320 kilometres driven to a customer site", "kilometres driven")),
    ],
}

SPECS: dict[str, Taxonomy] = {
    "support_queue": Taxonomy(
        ("Which support queue should this request be routed to?", "Where should this ticket be routed?"),
        "the request mentions",
        ("Message from {name}: they need help with {trigger}. Their order number is {ref}.",
         "{name} wrote to the help desk on {day} about {trigger}. They have been a customer since {year}.",
         "Chat summary: the customer ({name}, account {ref}) asked about {trigger}. "
         "They would like a reply by {day}."),
        ("the opening hours of the downtown showroom", "whether the company sponsors local charity runs",
         "a job application sent last month", "the history of the company logo"),
    ),
    "incident_category": Taxonomy(
        ("Which incident category does this report belong to?", "How should this security report be categorised?"),
        "the report mentions",
        ("Report {ref} filed by {name} in {dept} on {day}: {trigger}. The reporter is available until 5pm.",
         "{name} from {dept} called the security desk to report {trigger}. Ticket {ref} was opened.",
         "Security queue, item {ref}. Summary: {trigger}. Raised by {name} ({dept}) on {day}."),
        ("a printer jam on the third floor", "a request for a second desk phone",
         "a question about the holiday rota", "a flickering light in the car park"),
    ),
    "document_type": Taxonomy(
        ("What type of document is this?", "Which document type should this file be tagged with?"),
        "the document contains",
        ("A {pages}-page PDF from {name} was uploaded on {day}. It contains {trigger}.",
         "Scanned file {ref}, received from {name}. The first page shows {trigger}. It is {pages} pages long.",
         "Mailroom log for {day}: one document from {name}, {pages} pages, with {trigger}."),
        ("a hand-drawn map of a garden", "a recipe for lemon cake", "a poem written for a retirement party",
         "a seating plan for a wedding"),
    ),
    "expense_category": Taxonomy(
        ("Which expense category should this line be filed under?", "What category does this expense go in?"),
        "the line mentions",
        ("Expense line from {name}, dated {day}: {trigger}, {amount}.",
         "{name} submitted a claim for {amount} on {day}. The receipt describes {trigger}.",
         "Claim {ref} ({name}, {dept}): {amount} for {trigger}. Approved by the line manager."),
        ("a parking fine", "a personal gym membership", "a donation to a school fundraiser",
         "a dry-cleaning bill"),
    ),
}

NAMES = ["Priya Nair", "Tomasz Kowal", "Amara Okafor", "Lena Fischer", "Diego Ramos", "Hana Sato",
         "Yusuf Demir", "Claire Dubois", "Sven Larsen", "Mei Lin", "Owen Price", "Fatima Zahra"]
DEPTS = ["finance", "operations", "marketing", "HR", "engineering", "legal", "customer success", "facilities"]
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


def _pattern(kw: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)


def classify(state: str, taxonomy: str) -> tuple[str, str] | None:
    """The (category, keyword) whose keyword appears in the state; None when no keyword appears.

    Raises when keywords of two categories appear, which the templates must never produce.
    """
    hits = {}
    for cat in TAXONOMIES[taxonomy]:
        for kw in cat.keywords:
            if _pattern(kw).search(state):
                hits.setdefault(cat.key, kw)
    if len(hits) > 1:
        raise ValueError(f"state matches several categories {sorted(hits)}: {state!r}")
    return next(iter(hits.items()), None)


def gold_for(state: str, taxonomy: str, option_keys) -> str:
    hit = classify(state, taxonomy)
    return hit[0] if hit and hit[0] in option_keys else NONE_KEY


def category_option(cat: Category, subject: str) -> Option:
    quoted = [f'"{k}"' for k in cat.keywords]
    listed = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]
    return Option(cat.key, f"{cat.label}: {subject} {listed}.")


def _slots(rng: random.Random) -> dict:
    return {"name": rng.choice(NAMES), "ref": str(rng.randint(10000, 999999)), "dept": rng.choice(DEPTS),
            "day": f"{rng.randint(1, 28)} {rng.choice(MONTH_NAMES)}", "year": rng.randint(2012, 2025),
            "pages": rng.randint(2, 30), "amount": f"${rng.randint(8, 1900)}.{rng.randint(0, 99):02d}"}


def taxonomy_cluster(rng: random.Random, cid: str, taxonomy: str) -> list[Decision]:
    cats, spec = TAXONOMIES[taxonomy], SPECS[taxonomy]
    by_key = {c.key: c for c in cats}
    n_options = rng.randint(7, 10)
    listed = rng.sample(cats, n_options - 1)
    unlisted = [c for c in cats if c not in listed]
    options = [category_option(c, spec.subject) for c in listed] + [NONE_OPTION]
    rng.shuffle(options)
    template, slots = rng.choice(spec.templates), _slots(rng)

    def trig(cat: Category | None) -> str:
        return rng.choice(cat.triggers)[0] if cat else rng.choice(spec.none_triggers)

    base_cat = None if rng.random() < NONE_RATE else rng.choice(listed)
    base_trigger = trig(base_cat)
    others = [c for c in listed if c is not base_cat]
    swap_in, second = rng.sample(others, 2)
    rows = [(base_trigger, options, None), (trig(swap_in), options, "entity_swap")]
    # Every cluster has a "none of these fits" member, so matching words alone never solves a cluster.
    if base_cat is None:
        if rng.random() < 0.5:  # a keyword of a category that is not on the list
            rows.append((trig(rng.choice(unlisted)), options, "entity_swap"))
    else:
        repl = category_option(rng.choice(unlisted), spec.subject)
        none_rows = [(trig(rng.choice(unlisted)), options, "entity_swap"),
                     (trig(None), options, "entity_swap"),
                     # Same text; the matching category's option is replaced by an unlisted one.
                     (base_trigger, [repl if o.key == base_cat.key else o for o in options], "policy_edit")]
        rows.append(rng.choices(none_rows, (3, 2, 2))[0])
    if rng.random() < 0.5:
        rows.append((trig(second), options, "entity_swap"))
    question = rng.choice(spec.question)
    members, metas = [], []
    for trigger, opts, edit in rows:
        state = template.format(trigger=trigger, **slots)
        hit = classify(state, taxonomy)
        keys = [o.key for o in opts]
        members.append((state, opts, gold_for(state, taxonomy, keys), edit))
        metas.append({"taxonomy": taxonomy, "category": hit[0] if hit else None, "keyword": hit[1] if hit else None,
                      "label": by_key[hit[0]].label if hit else None})
    out = make_cluster(cluster_id=cid, family=FAMILY, source=SOURCE, qtype="choice", question=question,
                       members=members)
    for d, m in zip(out, metas):
        d.metadata = m
    return out


def generate(n_clusters: int, seed: int) -> list[Decision]:
    rng = random.Random(f"many_options:{seed}")
    names = list(TAXONOMIES)
    out: list[Decision] = []
    for k in range(n_clusters):
        taxonomy = names[k % len(names)] if k < len(names) else rng.choice(names)
        out.extend(taxonomy_cluster(rng, f"many_options:{seed}:{k}", taxonomy))
    return keep_valid_clusters(out)


# ---------------------------------------------------------------- intent rows, reframed

_STOP = {"my", "a", "an", "the", "or", "to", "of", "is", "not", "and", "for", "in", "on", "me", "i", "you",
         "do", "how", "what", "why", "can", "be"}


def _label_words(key: str) -> set[str]:
    """Content words of a label, with a plural "s" dropped (reservations ~ reservation)."""
    words = {w for w in key.split("_") if w and w not in _STOP}
    return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in words}


def _refuse(r: Decision) -> str | None:
    if r.split != "pool":
        return f"split is {r.split!r}, not 'pool'"
    if "test" in r.source.lower().split("_") or r.source.lower().endswith("test"):
        return f"source {r.source!r} is a test split"
    meta = r.metadata or {}
    for k in ("split", "hf_split", "source_split"):
        if str(meta.get(k, "")).lower() == "test":
            return f"metadata {k} is 'test'"
    return None


def reframe_intents(rows: list[Decision], n_options: range = range(7, 11), seed: int = 0) -> list[Decision]:
    """Each intent row re-asked with 7-10 candidate intents from its own source's label space.

    The original options are kept; the extra candidates are labels of the same source, preferring those
    that share a word with the gold label (e.g. card_arrival -> card_delivery_estimate). Rows whose
    label space is too small to reach the drawn size get as many as the space holds, and are dropped
    when that is under 7. Rows not in the pool split, or drawn from a test split, are refused.
    """
    for r in rows:
        why = _refuse(r)
        if why:
            raise ValueError(f"{r.id}: refusing to reframe: {why}")
    spaces: dict[str, dict[str, Option]] = {}
    for r in rows:
        space = spaces.setdefault(r.source, {})
        for o in r.options:
            if o.key != NONE_KEY:
                space.setdefault(o.key, o)
    rng = random.Random(f"many_options_intent:{seed}")
    sizes = list(n_options)
    out: list[Decision] = []
    for r in rows:
        space = spaces[r.source]
        target = rng.choice(sizes)
        present = set(r.keys)
        gold_words = _label_words(r.gold) if r.gold != NONE_KEY else set()
        pool = sorted(k for k in space if k not in present)
        rng.shuffle(pool)
        pool.sort(key=lambda k: -len(_label_words(k) & gold_words))  # stable: ties stay shuffled
        extra = pool[: max(0, target - len(r.options))]
        options = list(r.options) + [space[k] for k in extra]
        if len(options) < min(sizes):
            continue
        rng.shuffle(options)
        out.append(Decision(
            id=f"{r.id}:mo", type="choice", state=r.state, question=r.question, options=options, gold=r.gold,
            family=INTENT_FAMILY, source=r.source, licence=r.licence, split="pool", weight=r.weight,
            metadata={**(r.metadata or {}), "orig_id": r.id, "orig_family": r.family},
        ))
    return out
