"""Tests for the planner's request windows (spec §6.5, plan WP-4)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.data.fmp_pipeline.planner.windowing import (
    EPOCH,
    intraday_window_days,
    intraday_windows,
    is_possibly_truncated,
    split_window,
    trading_days,
    weekdays,
    year_windows,
)


def test_weekdays_counts_inclusive_and_handles_empty_ranges():
    assert weekdays(date(2026, 10, 12), date(2026, 10, 18)) == 5  # Mon..Sun
    assert weekdays(date(2026, 10, 17), date(2026, 10, 18)) == 0  # weekend
    assert weekdays(date(2026, 10, 14), date(2026, 10, 14)) == 1
    assert weekdays(date(2026, 10, 15), date(2026, 10, 14)) == 0
    assert trading_days(date(2025, 1, 1), date(2025, 12, 31)) == pytest.approx(252, abs=1)


def test_year_windows_follow_a_fixed_calendar_grid():
    windows = year_windows(date(2000, 1, 1), date(2026, 10, 14), 5)
    assert windows[0] == (date(2000, 1, 1), date(2004, 12, 31))
    assert windows[1] == (date(2005, 1, 1), date(2009, 12, 31))
    assert windows[-1] == (date(2025, 1, 1), date(2026, 10, 14))
    assert len(windows) == 6
    for (_, hi), (lo, _) in zip(windows, windows[1:]):
        assert lo == hi + timedelta(days=1)


def test_year_windows_start_mid_year_and_edge_cases():
    assert year_windows(date(2003, 6, 15), date(2009, 3, 1), 5) == [
        (date(2003, 6, 15), date(2007, 12, 31)), (date(2008, 1, 1), date(2009, 3, 1))]
    assert year_windows(date(2027, 1, 1), date(2026, 10, 14), 5) == []
    assert year_windows(date(2026, 1, 1), date(2026, 10, 14), 1) == [(date(2026, 1, 1), date(2026, 10, 14))]
    with pytest.raises(ValueError):
        year_windows(date(2000, 1, 1), date(2026, 1, 1), 0)


@pytest.mark.parametrize("max_rows,bars,safety,expected", [
    (1000, 7, 1.25, 114), (5000, 7, 1.25, 571), (10, 7, 1.25, 1), (7, 7, 1.0, 1), (700, 7, 1.0, 100)])
def test_intraday_window_days(max_rows: int, bars: int, safety: float, expected: int):
    assert intraday_window_days(max_rows, bars, safety) == expected


def test_intraday_window_days_rejects_nonsense():
    with pytest.raises(ValueError):
        intraday_window_days(0, 7)
    with pytest.raises(ValueError):
        intraday_window_days(1000, 0)


def test_intraday_windows_cover_the_range_on_a_stable_grid():
    as_of = date(2026, 10, 14)
    windows = intraday_windows(date(2024, 3, 6), date(2024, 9, 30), as_of, 20)
    assert windows[0][0] <= date(2024, 3, 6) <= windows[0][1]
    assert windows[-1][0] <= date(2024, 9, 30) <= windows[-1][1]
    for lo, hi in windows:
        assert lo.weekday() < 5 and hi.weekday() < 5
        assert weekdays(lo, hi) == 20
        assert weekdays(EPOCH, lo - timedelta(days=1)) % 20 == 0  # starts on a grid boundary
    for (_, hi), (lo, _) in zip(windows, windows[1:]):
        assert weekdays(hi, lo) == 2  # consecutive weekdays: no gap, no overlap
    # Moving the start inside the same grid block yields the same windows (task ids stay stable).
    assert intraday_windows(date(2024, 3, 8), date(2024, 9, 30), as_of, 20)[0] == windows[0]


def test_intraday_windows_cap_at_as_of_and_handle_empty_ranges():
    as_of = date(2026, 10, 14)
    windows = intraday_windows(date(2026, 9, 1), date(2030, 1, 1), as_of, 100)
    assert windows[-1][1] == as_of
    assert intraday_windows(date(2026, 10, 20), date(2026, 12, 1), as_of, 10) == []
    assert intraday_windows(date(2026, 5, 1), date(2026, 4, 1), as_of, 10) == []


def test_split_window():
    assert split_window(date(2026, 1, 1), date(2026, 1, 10)) == (
        (date(2026, 1, 1), date(2026, 1, 5)), (date(2026, 1, 6), date(2026, 1, 10)))
    assert split_window(date(2026, 1, 1), date(2026, 1, 2)) == (
        (date(2026, 1, 1), date(2026, 1, 1)), (date(2026, 1, 2), date(2026, 1, 2)))
    assert split_window(date(2026, 1, 1), date(2026, 1, 1)) is None


def test_is_possibly_truncated_at_95_percent():
    assert is_possibly_truncated(950, 1000)
    assert is_possibly_truncated(1000, 1000)
    assert not is_possibly_truncated(949, 1000)
    assert is_possibly_truncated(5, 10, ratio=0.5)
