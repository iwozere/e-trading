"""Unit tests for `EarningsSource`."""

from datetime import date
from unittest.mock import MagicMock

from src.portfolio.management.earnings_source import EarningsSource


def test_empty_ticker_list_short_circuits():
    calendar = MagicMock()
    source = EarningsSource(calendar=calendar)

    events = source.get_upcoming_events([], date(2026, 8, 20), 21)

    assert events == []
    calendar.get_earnings_within_days.assert_not_called()


def test_maps_calendar_dict_to_events_sorted_by_date_then_ticker():
    calendar = MagicMock()
    calendar.get_earnings_within_days.return_value = {
        "BBB": date(2026, 8, 21),
        "AAA": date(2026, 8, 21),
        "CCC": date(2026, 8, 20),
    }
    source = EarningsSource(calendar=calendar)

    events = source.get_upcoming_events(["aaa", "bbb", "ccc"], date(2026, 8, 20), 21)

    assert [(e.ticker, e.earnings_date) for e in events] == [
        ("CCC", date(2026, 8, 20)),
        ("AAA", date(2026, 8, 21)),
        ("BBB", date(2026, 8, 21)),
    ]
    # Session is always "unknown" today (see module docstring / Tasks.md).
    assert all(e.session == "unknown" for e in events)
    calendar.get_earnings_within_days.assert_called_once_with(["AAA", "BBB", "CCC"], date(2026, 8, 20), 21)


def test_calendar_exception_returns_empty_list():
    calendar = MagicMock()
    calendar.get_earnings_within_days.side_effect = RuntimeError("boom")
    source = EarningsSource(calendar=calendar)

    assert source.get_upcoming_events(["AAA"], date(2026, 8, 20), 21) == []
