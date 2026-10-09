"""Review packets for judge_planted_error (Plan 6, Group C): a task, acceptance criteria, source material and two
outputs built in code, one meeting every criterion and one with a single planted defect.

Each builder returns a JudgeCase. Code writes every output in full (a query, a function, a summary, a JSON
extraction or a customer reply), so the label never depends on the writer: the writer copies the output and writes
only the surroundings (the requester's framing and, for summaries, extractions and replies, the passage or message
from the facts given). Flaw kinds: join_key, off_by_one, dropped_field, unsupported_claim, wrong_shape, criterion.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta

from yev.generators.llm.targets.agent_trace import person

__all__ = ["JudgeCase", "BUILDERS", "KINDS", "REVIEW_ROLES", "SILENT", "day"]


@dataclass(frozen=True)
class JudgeCase:
    kind: str
    flaw: str
    team: str            # who the work is for, e.g. "reporting team"
    thing: str           # what is under review, e.g. "SQL query"
    task: str
    criteria: tuple[str, ...]
    source: str          # the scene's source-material paragraph(s)
    correct: str
    flawed: str


def day(d: date) -> str:
    return f"{d.day} {d:%B} {d.year}"


def _start(rng: random.Random) -> date:
    return date(2025, 1, 1) + timedelta(days=rng.randrange(0, 600))


# --- SQL ------------------------------------------------------------------------------------------

SQL_TABLES: tuple[dict, ...] = (
    {"parent": "members", "pk": "member_id", "name": "full_name", "group": "branch_code",
     "groups": ("NW2", "SE1", "CT4", "HB7"), "child": "visits", "ck": "visit_id", "date": "visit_date",
     "value": "minutes", "vtype": "INTEGER", "total": "total_minutes", "pnoun": "member", "pnouns": "members",
     "cnoun": "visit", "cnouns": "visits", "vnoun": "minutes"},
    {"parent": "drivers", "pk": "driver_id", "name": "driver_name", "group": "depot",
     "groups": ("EASTGATE", "RIVERSIDE", "MILLBROOK"), "child": "deliveries", "ck": "delivery_id",
     "date": "delivered_on", "value": "distance_km", "vtype": "REAL", "total": "total_km", "pnoun": "driver",
     "pnouns": "drivers", "cnoun": "delivery", "cnouns": "deliveries", "vnoun": "distance in kilometres"},
    {"parent": "customers", "pk": "customer_id", "name": "customer_name", "group": "region",
     "groups": ("north", "south", "west", "central"), "child": "orders", "ck": "order_id", "date": "ordered_on",
     "value": "amount", "vtype": "REAL", "total": "total_amount", "pnoun": "customer", "pnouns": "customers",
     "cnoun": "order", "cnouns": "orders", "vnoun": "order amount"},
    {"parent": "volunteers", "pk": "volunteer_id", "name": "volunteer_name", "group": "team",
     "groups": ("kitchen", "warehouse", "outreach"), "child": "shifts", "ck": "shift_id", "date": "shift_date",
     "value": "hours", "vtype": "REAL", "total": "total_hours", "pnoun": "volunteer", "pnouns": "volunteers",
     "cnoun": "shift", "cnouns": "shifts", "vnoun": "hours"},
    {"parent": "vendors", "pk": "vendor_id", "name": "vendor_name", "group": "category",
     "groups": ("print", "catering", "cleaning"), "child": "invoices", "ck": "invoice_id", "date": "issued_on",
     "value": "amount_due", "vtype": "REAL", "total": "total_due", "pnoun": "vendor", "pnouns": "vendors",
     "cnoun": "invoice", "cnouns": "invoices", "vnoun": "amount due"},
)


def _sql(rng: random.Random, domain: str) -> JudgeCase:
    t = rng.choice(SQL_TABLES)
    g = rng.choice(t["groups"])
    s = _start(rng)
    e = s + timedelta(days=rng.randint(6, 40))
    pa = t["parent"][0]
    ca = t["child"][0] if t["child"][0] != pa else t["child"][:2]
    iso = lambda d: f"'{d.isoformat()}'"  # noqa: E731
    one = timedelta(days=1)
    style = rng.choice(("closed", "between", "halfopen"))
    col = f"{ca}.{t['date']}"
    if style == "closed":
        good = f"{col} >= {iso(s)}\n  AND {col} <= {iso(e)}"
        bad = rng.choice((f"{col} > {iso(s)}\n  AND {col} <= {iso(e)}", f"{col} >= {iso(s)}\n  AND {col} < {iso(e)}"))
    elif style == "between":
        good = f"{col} BETWEEN {iso(s)} AND {iso(e)}"
        bad = rng.choice((f"{col} BETWEEN {iso(s + one)} AND {iso(e)}", f"{col} BETWEEN {iso(s)} AND {iso(e - one)}"))
    else:
        good = f"{col} >= {iso(s)}\n  AND {col} < {iso(e + one)}"
        bad = rng.choice((f"{col} >= {iso(s)}\n  AND {col} < {iso(e)}",
                          f"{col} > {iso(s)}\n  AND {col} < {iso(e + one)}"))

    def query(select: str = f"{pa}.{t['name']}", on: str = f"{ca}.{t['pk']}", dates: str = good,
              order: str = " DESC") -> str:
        return (f"SELECT {select}, SUM({ca}.{t['value']}) AS {t['total']}\n"
                f"FROM {t['parent']} AS {pa}\n"
                f"JOIN {t['child']} AS {ca} ON {on} = {pa}.{t['pk']}\n"
                f"WHERE {pa}.{t['group']} = '{g}'\n"
                f"  AND {dates}\n"
                f"GROUP BY {pa}.{t['pk']}, {pa}.{t['name']}\n"
                f"ORDER BY {t['total']}{order};")

    flaw = rng.choice(("join_key", "off_by_one", "criterion", "wrong_shape"))
    flawed = {"join_key": lambda: query(on=f"{ca}.{t['ck']}"),
              "off_by_one": lambda: query(dates=bad),
              "criterion": lambda: query(order=rng.choice(("", " ASC"))),
              "wrong_shape": lambda: query(select=f"{pa}.{t['pk']}, {pa}.{t['name']}")}[flaw]()
    criteria = [
        f"Each {t['cnoun']} is counted for the {t['pnoun']} it belongs to, as recorded in {t['child']}.{t['pk']}.",
        f"Only {t['cnouns']} dated from {day(s)} to {day(e)}, both days included, are counted.",
        f"Only {t['pnouns']} whose {t['group']} is '{g}' appear in the result.",
        f"The result has exactly two columns, in this order: {t['name']}, then {t['total']}.",
        f"Rows are sorted by {t['total']} from largest to smallest.",
    ]
    if rng.random() < 0.5:
        criteria.append(rng.choice((f"There is one row for each {t['pnoun']} with at least one counted "
                                    f"{t['cnoun']}.", "The answer is a single SELECT statement that runs on SQLite.")))
    rng.shuffle(criteria)
    schema = (f"CREATE TABLE {t['parent']} ({t['pk']} INTEGER PRIMARY KEY, {t['name']} TEXT, {t['group']} TEXT);\n"
              f"CREATE TABLE {t['child']} ({t['ck']} INTEGER PRIMARY KEY, {t['pk']} INTEGER REFERENCES "
              f"{t['parent']}({t['pk']}), {t['date']} TEXT, {t['value']} {t['vtype']});")
    source = (f"Schema (copy as given):\n{schema}\n\n"
              f"Below the schema, the packet notes that {t['child']}.{t['date']} holds a calendar date written "
              f"YYYY-MM-DD with no time part, and that {t['child']}.{t['pk']} is the id of the {t['pnoun']} the "
              f"{t['cnoun']} belongs to.")
    task = (f"Write an SQLite query that reports, for each {t['pnoun']} whose {t['group']} is '{g}', the total "
            f"{t['vnoun']} over their {t['cnouns']} dated from {day(s)} to {day(e)}.")
    return JudgeCase("sql", flaw, "reporting team", "SQL query", task, tuple(criteria), source, query(), flawed)


# --- Python functions -----------------------------------------------------------------------------

def _labels(rng: random.Random) -> dict:
    fn = rng.choice(("ticket_labels", "bin_labels", "seat_labels", "locker_labels", "crate_labels"))
    px = rng.choice(("BX", "QT", "LK", "RM", "PX", "DW", "HN"))
    things = {"ticket_labels": "tickets", "bin_labels": "storage bins", "seat_labels": "seats",
              "locker_labels": "lockers", "crate_labels": "crates"}[fn]
    body = ('    """Labels for items 1 to n."""\n    out = []\n    for i in range(1, {stop}):\n'
            '        out.append(f"{{prefix}}-{{i{pad}}}")\n    return {ret}')

    def code(stop="n + 1", pad=":03d", ret="out"):
        return f"def {fn}(prefix, n):\n" + body.format(stop=stop, pad=pad, ret=ret)

    n = rng.randint(2, 4)
    example = f'{fn}("{px}", {n}) returns [' + ", ".join(f'"{px}-{i:03d}"' for i in range(1, n + 1)) + "]"
    return {
        "fn": fn, "args": "prefix, n", "code": code, "example": example,
        "task": f"Write a Python function {fn}(prefix, n) that builds the labels for n numbered {things}, where n "
                f"is a whole number from 0 to 999.",
        "criteria": [
            "It returns a list of strings.",
            "The list has one label for every whole number from 1 to n, both included, in increasing order.",
            f'Each label is the prefix, a hyphen and the number padded with zeros to three digits; for example, '
            f'number 7 with prefix "{px}" gives "{px}-007".',
            "When n is 0 it returns an empty list.",
        ],
        "flaws": {"off_by_one": code(stop="n"), "wrong_shape": code(ret='", ".join(out)'),
                  "criterion": code(pad="")},
    }


def _band(rng: random.Random) -> dict:
    fn, noun = rng.choice((("readings_in_band", "sensor readings"), ("scores_in_range", "test scores"),
                           ("weights_within", "parcel weights"), ("temps_in_band", "fridge temperatures")))

    def code(test="low <= v <= high", ret="kept"):
        return (f"def {fn}(values, low, high):\n    \"\"\"Values from low to high, in their original order.\"\"\"\n"
                f"    kept = []\n    for v in values:\n        if {test}:\n            kept.append(v)\n"
                f"    return {ret}")

    vals = [rng.randint(1, 30) for _ in range(rng.randint(5, 7))]
    low, high = sorted(rng.sample(sorted(set(vals)), 2)) if len(set(vals)) > 1 else (1, 30)
    example = f"{fn}({vals}, {low}, {high}) returns {[v for v in vals if low <= v <= high]}"
    return {
        "fn": fn, "args": "values, low, high", "code": code, "example": example,
        "task": f"Write a Python function {fn}(values, low, high) that picks out the {noun} in the list values "
                f"that fall within the band from low to high. low is never greater than high.",
        "criteria": [
            "It returns a new list holding the values it keeps.",
            "A value is kept when it is at least low and at most high, so values equal to low or to high are kept.",
            "Kept values stay in the order they have in values.",
            "It leaves the list it is given unchanged.",
        ],
        "flaws": {"off_by_one": code(test=rng.choice(("low < v <= high", "low <= v < high"))),
                  "wrong_shape": code(ret="len(kept)"), "criterion": code(ret="sorted(kept)")},
    }


def _tail(rng: random.Random) -> dict:
    fn, noun = rng.choice((("last_lines", "a log file"), ("tail_of_log", "a job log"),
                           ("final_entries", "a shift diary"), ("latest_notes", "a notes file")))

    def code(split="text.splitlines()", ret="lines[-n:]"):
        return (f"def {fn}(text, n):\n    \"\"\"The last n lines of text.\"\"\"\n    lines = {split}\n"
                f"    return {ret}")

    words = rng.sample(("alpha", "bravo", "delta", "echo", "kilo", "lima", "oscar", "tango"), 4)
    n = rng.randint(1, 3)
    example = f'{fn}("' + "\\n".join(words) + f'", {n}) returns ' + json.dumps(words[-n:])
    return {
        "fn": fn, "args": "text, n", "code": code, "example": example,
        "task": f"Write a Python function {fn}(text, n) that returns the last n lines of the text of {noun}. n is "
                f"always at least 1.",
        "criteria": [
            "It returns a list of strings, one string per line.",
            "The list holds the last n lines of text in their original order; if text has fewer than n lines, it "
            "holds all of them.",
            "The returned strings do not include line endings.",
        ],
        "flaws": {"off_by_one": code(ret="lines[-n - 1:]"), "wrong_shape": code(ret='"\\n".join(lines[-n:])'),
                  "criterion": code(split="text.splitlines(keepends=True)")},
    }


def _function(rng: random.Random, domain: str) -> JudgeCase:
    f = rng.choice((_labels, _band, _tail))(rng)
    flaw = rng.choice(sorted(f["flaws"]))
    criteria = [f"The function is called {f['fn']} and takes these arguments, in this order: {f['args']}.",
                *f["criteria"], "It imports nothing."]
    source = f"Example (copy as given):\n{f['example']}"
    return JudgeCase("function", flaw, "internal tools team", "Python function", f["task"], tuple(criteria[:6]),
                     source, f["code"](), f["flaws"][flaw])


# --- summaries ------------------------------------------------------------------------------------

SITE_NAMES: tuple[str, ...] = ("Quayside", "Harbour Road", "Millbank", "Station Square", "Elm Park", "Castle Hill",
                               "Northgate", "Riverside Court", "Fenwick Lane", "Copper Street")
SERVICES: tuple[str, ...] = ("the parcel collection counter", "the walk-in help desk", "the key-cutting service",
                             "the photo printing kiosk", "the equipment returns desk", "the visitor badge office")
ROUTES: tuple[str, ...] = ("the harbour shuttle", "the park-and-ride bus", "the airport link coach",
                           "the river ferry", "the campus loop bus", "the hospital minibus")
PRODUCTS: tuple[str, ...] = ("Brindle 2-litre travel kettle", "Wexmoor folding step stool", "Lumen desk fan",
                             "Orrin cordless hand blender", "Tamsin children's scooter", "Pellow USB bike light")
REASONS: tuple[str, ...] = ("the handle can work loose", "the hinge can snap under load", "the battery can overheat",
                            "a sharp edge was found on the base", "the clip can break off")
HOURS: tuple[str, ...] = ("8:00 to 18:00 on weekdays", "9:00 to 17:30 Monday to Saturday", "7:30 to 15:30 on weekdays",
                          "10:00 to 19:00 every day")
TIMES: tuple[str, ...] = ("6:10", "6:25", "6:40", "7:05", "22:15", "22:50", "23:20", "23:45")
# kind -> silent topic -> (the clause a flawed summary adds about it, the words that carry the claim)
SILENT: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "move": {"parking": ("and parking there is free", ("parking",)),
             "fees or prices": ("with no change to the fees", ("fees",)),
             "staffing": ("with the same staff as before", ("staff",))},
    "timetable": {"fares": ("and fares stay the same", ("fares",)),
                  "step-free access": ("with step-free boarding on every trip", ("step", "boarding")),
                  "Sunday running": ("including on Sundays", ("sundays",))},
    "recall": {"replacements": ("and a replacement will be posted out", ("replacement",)),
               "vouchers or compensation": ("plus a voucher for the trouble", ("voucher",)),
               "home collection": ("or have it collected from home", ("collected", "home"))},
}


def _contact(rng: random.Random, domain: str) -> str:
    return f"{rng.choice(('changes', 'help', 'notices', 'updates'))}@{domain.split()[0].lower()}-service.co"


def _summary(rng: random.Random, domain: str) -> JudgeCase:
    kind = rng.choice(sorted(SILENT))
    contact = _contact(rng, domain)
    d = day(_start(rng))
    if kind == "move":
        service = rng.choice(SERVICES)
        old, new = rng.sample(SITE_NAMES, 2)
        hours = rng.choice(HOURS)
        req = [f"{service[0].upper()}{service[1:]} is moving from {old} to {new} on {d}.",
               f"At {new} it will be open {hours}."]
        opt = [f"Appointments already booked for after {d} move to {new} automatically.",
               f"Questions about the move can be sent to {contact}."]
        need = [f'It names the new site ("{new}") and the date of the move ("{d}").',
                f'It gives the opening hours at the new site ("{hours}").']
        doc = f"a notice to customers about {service}"
    elif kind == "timetable":
        route = rng.choice(ROUTES)
        nf, of = rng.sample((10, 12, 15, 20, 30), 2)
        first, last = rng.choice(TIMES[:4]), rng.choice(TIMES[4:])
        req = [f"From {d}, {route} will run every {nf} minutes instead of every {of} minutes.",
               f"The first departure of the day will be at {first} and the last at {last}."]
        opt = ["The change follows a review of passenger numbers over the past year.",
               f"Comments on the new timetable can be sent to {contact}."]
        need = [f'It gives the start date ("{d}") and the new frequency ("every {nf} minutes").',
                f'It gives the new first and last departure times ("{first}" and "{last}").']
        doc = f"a timetable notice for {route}"
    else:
        product, reason = rng.choice(PRODUCTS), rng.choice(REASONS)
        d2 = day(_start(rng) + timedelta(days=700))
        where = f"any {domain} counter"
        req = [f"The {product} sold between {d} and {d2} is being recalled because {reason}.",
               f"Customers should stop using it and bring it back to {where} for a full refund."]
        opt = ["No receipt is needed to return it.", f"Questions about the recall can be sent to {contact}."]
        need = [f'It names the product ("{product}") and the dates it was sold ("{d}" and "{d2}").',
                f'It says where to bring it back ("{where}").']
        doc = f"a product recall notice about the {product}"
    silent = rng.choice(sorted(SILENT[kind]))
    clause = SILENT[kind][silent][0]
    keep = rng.sample(opt, rng.randint(0, 1))
    correct = [*req, *keep]
    flaw = rng.choice(("unsupported_claim", "dropped_field", "criterion"))
    k = len(correct) if flaw == "criterion" else len(correct) + rng.choice((0, 1))
    if flaw == "unsupported_claim":
        flawed = [req[0], f"{req[1][:-1]}, {clause}.", *keep]
    elif flaw == "dropped_field":
        flawed = [s for s in correct if s != req[1]]   # req[1]'s quoted values appear in no other sentence
    else:
        flawed = [*correct, next(s for s in opt if s not in keep)]
    facts = [*req, *opt]
    criteria = [f"The summary is at most {k} sentences long.", *need,
                "Everything the summary says is stated in the passage."]
    if rng.random() < 0.5:
        criteria.append("It is written as running prose, with no bullet points or headings.")
    rng.shuffle(criteria)
    source = (f"Passage: write it under \"Source material:\" as {doc}, 150 to 450 words long, in the voice of the "
              f"{domain}. It states these facts, with every name, date, time and figure exactly as given:\n"
              + "\n".join(f"- {x}" for x in facts)
              + f"\n\nIt says nothing at all about {silent}, and it gives no other dates, times or figures.")
    task = f"Summarise the passage below for the {domain}'s weekly staff bulletin."
    return JudgeCase("summary", flaw, "communications team", "summary", task, tuple(criteria), source,
                     " ".join(correct), " ".join(flawed))


# --- JSON extraction ------------------------------------------------------------------------------

ORDER_ITEMS: tuple[str, ...] = ("A4 copier paper, box of 5 reams", "black gel pens, pack of 12", "printer toner TX-40",
                                "cable ties, bag of 100", "hand soap refill, 5 litres", "packing tape, roll",
                                "lever arch files, pack of 10", "nitrile gloves, box of 200", "bin liners, roll of 50",
                                "whiteboard markers, pack of 4")
ROOMS: tuple[str, ...] = ("Larch Room", "Willow Room", "Boardroom 2", "Studio B", "Hazel Suite", "Garden Room",
                          "Training Room 1", "Rowan Room")


def _extraction(rng: random.Random, domain: str) -> JudgeCase:
    if rng.random() < 0.5:
        noun, keys, wrap = "order line", ("item", "quantity", "unit_price"), "items"
        rows = []
        for item in rng.sample(ORDER_ITEMS, rng.randint(3, 5)):
            price = rng.randint(1, 60) + rng.choice((5, 15, 25, 35, 45, 65, 85, 95)) / 100
            rows.append({"item": item, "quantity": rng.randint(1, 40), "unit_price": round(price, 2)})
        lines = [f'- [{r["item"]}] x [{r["quantity"]}] at [{r["unit_price"]:.2f}] each' for r in rows]
        numeric = ("quantity", "unit_price")
        type_rule = "quantity and unit_price are JSON numbers, not strings, and unit_price has no currency sign."
        who = "a customer placing a stationery and supplies order"
    else:
        noun, keys, wrap = "booking", ("room", "date", "attendees"), "bookings"
        rows = []
        for room in rng.sample(ROOMS, rng.randint(3, 5)):
            rows.append({"room": room, "date": day(_start(rng)), "attendees": rng.randint(3, 60)})
        lines = [f'- [{r["room"]}] on [{r["date"]}] for [{r["attendees"]}] people' for r in rows]
        numeric = ("attendees",)
        type_rule = "attendees is a JSON number, not a string, and date is copied as the message writes it."
        who = "an event organiser booking rooms"
    flaw = rng.choice(("dropped_field", "wrong_shape", "unsupported_claim"))
    bad = json.loads(json.dumps(rows))
    i = rng.randrange(len(bad))
    if flaw == "dropped_field":
        del bad[i][rng.choice(keys[1:])]
    elif flaw == "wrong_shape":
        if rng.random() < 0.5:
            bad = {wrap: bad}
        else:
            k = rng.choice(numeric)
            bad[i][k] = f"{bad[i][k]:.2f}" if isinstance(bad[i][k], float) else str(bad[i][k])
    else:
        k = keys[1] if noun == "order line" else "attendees"
        used = {str(r[k]) for r in rows}
        bad[i][k] = rng.choice([v for v in range(2, 70) if str(v) not in used and abs(v - bad[i][k]) <= 9])
    criteria = [
        f"The output is one JSON array with one object per {noun} in the message, in the order the message gives "
        f"them.",
        "Each object has exactly the keys " + ", ".join(f'"{k}"' for k in keys[:-1]) + f' and "{keys[-1]}".',
        type_rule,
        "Every value is copied from the message, and nothing is added that the message does not give.",
    ]
    if rng.random() < 0.5:
        criteria.append(f"The {keys[0]} value is the name exactly as the message writes it.")
    rng.shuffle(criteria)
    source = (f"Message: write it under \"Source material:\" as an email from {who} to the {domain}, 80 to 300 "
              f"words long. It lists exactly these {len(rows)} {noun}s, in this order, with every name, date and "
              f"figure exactly as given:\n" + "\n".join(lines)
              + f"\n\nThe square brackets only mark the values; leave them out. The email mentions no other "
              f"{noun}s, quantities, dates or prices.")
    task = f"Turn the {noun}s in the message below into JSON for the {domain}'s booking and ordering system."
    dump = lambda x: json.dumps(x, indent=2, ensure_ascii=False)  # noqa: E731
    return JudgeCase("extraction", flaw, "orders desk", "JSON extraction", task, tuple(criteria), source,
                     dump(rows), dump(bad))


# --- customer replies -----------------------------------------------------------------------------

PROBLEMS: tuple[tuple[str, str], ...] = (
    ("table lamp", "arrived with a cracked shade"), ("kettle", "arrived with a dented lid"),
    ("rucksack", "arrived with a broken zip"), ("bookcase", "arrived with two shelves missing"),
    ("coffee grinder", "arrived with a split hopper"), ("rain jacket", "arrived two sizes too small"),
    ("garden hose", "arrived with a leaking connector"), ("desk chair", "arrived with a bent wheel base"),
)


def _reply(rng: random.Random, domain: str) -> JudgeCase:
    customer, agent, staffer = person(rng), person(rng), person(rng)
    first = customer.split(" ")[0]
    item, problem = rng.choice(PROBLEMS)
    order = f"{rng.choice(('KT', 'QV', 'HM', 'RX', 'BL'))}-{rng.randint(10000, 99999)}"
    amount = f"£{rng.randint(12, 240)}.{rng.choice(('00', '49', '50', '95', '99'))}"
    record = [f"Customer: {customer}", f"Order number: {order}", f"Item: {item}",
              f"Problem: {problem}, checked from the customer's photo by {staffer}",
              f"Decision: refund of {amount} approved", "Refund date: not yet set; the payments team will confirm it",
              "Replacement: none, the customer asked for a refund only", "Vouchers or discounts: none approved",
              f"Handled by: {agent}"]
    thanks = f"Thank you for writing to us about order {order}."
    refund = f"We have approved a refund of {amount}, and the payments team will confirm the date it will be paid."

    def letter(thanks=thanks, extra=""):
        middle = f"{thanks} I am sorry that your {item} {problem}. {refund}" + (f" {extra}" if extra else "")
        return f"Hello {first},\n\n{middle}\n\nKind regards,\n{agent}"

    flaw = rng.choice(("unsupported_claim", "dropped_field", "criterion"))
    if flaw == "unsupported_claim":
        flawed = letter(extra=rng.choice(("The refund should reach your account within three working days.",
                                          f"A replacement {item} has already been dispatched to you.")))
    elif flaw == "dropped_field":
        flawed = letter(thanks="Thank you for writing to us about your order.")
    else:
        code = f"{rng.choice(('SORRY', 'THANKS', 'CARE'))}{rng.randint(10, 30)}"
        flawed = letter(extra=f"As an apology please take a 10% discount on your next order with the code {code}.")
    criteria = [
        f'It greets the customer by first name ("{first}").',
        f'It gives the order number ("{order}").',
        f'It states the refund amount exactly as the case record shows it ("{amount}").',
        "It says nothing about the order, the item or the refund that the case record does not show.",
        "It offers no voucher, discount or credit.",
        f'It ends with the agent\'s name ("{agent}") as the sign-off.',
    ]
    rng.shuffle(criteria)
    source = ("Case record (copy as given):\n" + "\n".join(f"- {x}" for x in record)
              + f"\n\nCustomer message: write it under \"Source material:\", after the case record, as {customer}'s "
              f"message to the {domain}, 60 to 250 words long: they describe the problem and ask what happens next. "
              f"It adds no fact the case record does not show.")
    task = f"Write the reply to {customer}'s message below, using the case record."
    return JudgeCase("reply", flaw, "customer support team", "customer reply", task, tuple(criteria), source,
                     letter(), flawed)


BUILDERS = {"sql": _sql, "function": _function, "summary": _summary, "extraction": _extraction, "reply": _reply}
KINDS: tuple[str, ...] = tuple(BUILDERS)
REVIEW_ROLES: tuple[str, ...] = ("quality reviewer", "team lead", "senior analyst", "operations manager",
                                 "support lead", "release checker")
