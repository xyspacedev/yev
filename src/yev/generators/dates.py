"""Shared date helpers for generators whose labels depend on calendar arithmetic."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

STYLES = ("iso", "long", "us", "slash_dmy")


def elapsed_days(a: date, b: date) -> int:
    """Calendar days from a to b (negative when b is earlier)."""
    return (b - a).days


def is_business_day(d: date, holidays: Iterable[date] = ()) -> bool:
    return d.weekday() < 5 and d not in set(holidays)


def business_days(a: date, b: date, holidays: Iterable[date] = ()) -> int:
    """Weekdays strictly after a, up to and including b, that are not holidays (0 when b <= a)."""
    hol = set(holidays)
    return sum(1 for k in range(1, elapsed_days(a, b) + 1) if is_business_day(a + timedelta(days=k), hol))


def add_business_days(a: date, n: int, holidays: Iterable[date] = ()) -> date:
    """The n-th business day after a (n >= 1), so that business_days(a, result) == n."""
    if n < 1:
        raise ValueError("n must be at least 1")
    hol = set(holidays)
    d, left = a, n
    while left:
        d += timedelta(days=1)
        if is_business_day(d, hol):
            left -= 1
    return d


def fmt(d: date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "long":
        return f"{d.day} {d:%B %Y}"
    if style == "us":
        return f"{d:%B} {d.day}, {d.year}"
    if style == "slash_dmy":
        return f"{d:%d/%m/%Y}"
    raise ValueError(f"unknown date style {style!r}")


def last_day_of_month(year: int, month: int) -> date:
    first_next = date(year + month // 12, month % 12 + 1, 1)
    return first_next - timedelta(days=1)
