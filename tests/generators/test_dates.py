from datetime import date, timedelta

import pytest

from yev.generators.dates import STYLES, add_business_days, business_days, elapsed_days, fmt


def test_elapsed_days_across_feb29_and_year_end():
    assert elapsed_days(date(2024, 2, 28), date(2024, 3, 1)) == 2
    assert elapsed_days(date(2023, 2, 28), date(2023, 3, 1)) == 1
    assert elapsed_days(date(2024, 12, 30), date(2025, 1, 2)) == 3
    assert elapsed_days(date(2025, 1, 2), date(2024, 12, 30)) == -3


def test_business_days_skips_weekends():
    fri, mon = date(2025, 1, 3), date(2025, 1, 6)
    assert business_days(fri, mon) == 1  # Saturday and Sunday skipped, Monday counted
    assert business_days(mon, date(2025, 1, 13)) == 5  # Monday to next Monday
    assert business_days(mon, mon) == 0
    assert business_days(fri, date(2025, 1, 5)) == 0  # Friday to Sunday: only weekend days
    assert business_days(mon, date(2025, 1, 10)) == 4  # start day excluded, end day included
    assert business_days(mon, date(2025, 1, 13), holidays=[date(2025, 1, 8)]) == 4
    assert business_days(mon, date(2025, 1, 13), holidays=[date(2025, 1, 11)]) == 5  # weekend holiday: no effect


def test_add_business_days_inverts_business_days():
    start = date(2024, 12, 20)
    hol = [date(2024, 12, 25), date(2024, 12, 26), date(2025, 1, 1)]
    for n in range(1, 30):
        d = add_business_days(start, n, hol)
        assert d.weekday() < 5 and d not in hol
        assert business_days(start, d, hol) == n
        assert business_days(start, d - timedelta(days=1), hol) == n - 1


def test_fmt_styles():
    d = date(2025, 3, 1)
    assert fmt(d, "iso") == "2025-03-01"
    assert fmt(d, "long") == "1 March 2025"
    assert fmt(d, "us") == "March 1, 2025"
    assert fmt(d, "slash_dmy") == "01/03/2025"
    assert set(STYLES) >= {"iso", "long", "us", "slash_dmy"}
    with pytest.raises(ValueError):
        fmt(d, "nope")
