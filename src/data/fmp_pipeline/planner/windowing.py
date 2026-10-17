"""
Request windows for date-ranged endpoints (spec §6.5).

Windows sit on **fixed grids** so that re-planning yields the same ``from``/``to`` (and therefore the same
``task_id``) no matter when the plan runs or how a symbol's known dates have moved:

- Daily: calendar-year blocks of ``years`` years starting at ``daily_from``; the last ``to`` is the dump's ``as_of``.
- Intraday: blocks of ``n`` weekdays counted from ``EPOCH`` (a Monday). A symbol gets the grid windows that overlap
  its first/last known daily bars; windows are not clipped to the symbol, only ``to`` is capped at ``as_of``.

Holidays are not removed from the weekday count, so a window holds slightly fewer bars than its weekday count
implies, which errs on the safe side of the row cap.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

EPOCH = date(1990, 1, 1)  # a Monday
#: Trading days per weekday: about 252 sessions in 261 weekdays (US holidays).
TRADING_PER_WEEKDAY = 252 / 261

Window = tuple[date, date]


def weekdays(start: date, end: date) -> int:
    """Return the number of Monday–Friday dates in ``start`` … ``end`` inclusive (0 if ``end < start``)."""
    if end < start:
        return 0
    return _weekday_index(end + timedelta(days=1)) - _weekday_index(start)


def trading_days(start: date, end: date) -> float:
    """Estimate US trading sessions in ``start`` … ``end`` inclusive."""
    return weekdays(start, end) * TRADING_PER_WEEKDAY


def year_windows(start: date, as_of: date, years: int) -> list[Window]:
    """
    Split ``start`` … ``as_of`` into blocks of ``years`` calendar years.

    Args:
        start: First date (the configured ``daily_from``); the first window starts here.
        as_of: Last date of the dump; the last window ends here.
        years: Years per window (≥ 1).

    Returns:
        Windows in date order; empty if ``start > as_of``.
    """
    if years < 1:
        raise ValueError("years must be >= 1")
    windows: list[Window] = []
    year = start.year
    while True:
        lo = start if year == start.year else date(year, 1, 1)
        if lo > as_of:
            return windows
        hi = min(date(year + years - 1, 12, 31), as_of)
        windows.append((lo, hi))
        year += years


def intraday_window_days(max_rows: int, bars_per_day: int, safety_factor: float = 1.25) -> int:
    """Return ``floor(max_rows / bars_per_day / safety_factor)`` weekdays per window, at least 1 (spec §6.5.2)."""
    if max_rows < 1 or bars_per_day < 1 or safety_factor <= 0:
        raise ValueError("max_rows, bars_per_day and safety_factor must be positive")
    return max(1, math.floor(max_rows / bars_per_day / safety_factor))


def _weekday_index(day: date) -> int:
    """Weekdays in ``EPOCH`` … ``day`` exclusive (negative before ``EPOCH``); integer maths, as ``EPOCH`` is a Monday."""
    weeks, rest = divmod((day - EPOCH).days, 7)
    return weeks * 5 + min(rest, 5)


def _weekday_at(index: int) -> date:
    """The weekday with ``_weekday_index`` equal to ``index``."""
    weeks, rest = divmod(index, 5)
    return EPOCH + timedelta(days=weeks * 7 + rest)


def intraday_windows(first: date, last: date, as_of: date, days: int) -> list[Window]:
    """
    Return the grid windows of ``days`` weekdays that overlap ``first`` … ``last``.

    Args:
        first: First date to cover (first daily bar, or the configured earliest intraday date if later).
        last: Last date to cover (last daily bar, plus a margin for delisted symbols).
        as_of: Dump date; caps every ``to``.
        days: Weekdays per window.

    Returns:
        Windows in date order; empty if the range is empty or starts after ``as_of``.
    """
    last = min(last, as_of)
    if last < first:
        return []
    windows: list[Window] = []
    k = _weekday_index(first) // days
    while True:
        lo = _weekday_at(k * days)
        if lo > last:
            return windows
        hi = min(_weekday_at((k + 1) * days - 1), as_of)
        windows.append((lo, hi))
        k += 1


def split_window(start: date, end: date) -> Optional[tuple[Window, Window]]:
    """Halve a window by calendar days; ``None`` if it is a single day and cannot be split."""
    if end <= start:
        return None
    mid = start + timedelta(days=(end - start).days // 2)
    return (start, mid), (mid + timedelta(days=1), end)


def is_possibly_truncated(rows: int, max_rows: int, ratio: float = 0.95) -> bool:
    """Return whether a response with ``rows`` rows reached ``ratio`` of the row cap (spec §6.5.3)."""
    return rows >= math.ceil(ratio * max_rows)
