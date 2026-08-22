"""
P21 Momentum — NYSE trading-calendar wrapper.

Thin wrapper over ``pandas_market_calendars`` (XNYS). Per
``docs/pipeline-specification.md`` §1: "Do not derive trading days
heuristically from weekends — holidays will break the logic."
"""

from __future__ import annotations

from datetime import date

import pandas_market_calendars as mcal

_XNYS = mcal.get_calendar("XNYS")


def _valid_days(start: date, end: date) -> list[date]:
    days = _XNYS.valid_days(start_date=start, end_date=end)
    return [d.date() for d in days]


def is_trading_day(d: date) -> bool:
    """Return True if ``d`` is an NYSE trading day."""
    return len(_valid_days(d, d)) == 1


def trading_days(start: date, end: date) -> list[date]:
    """Return every NYSE trading day in [start, end], inclusive, ascending.

    Public wrapper over the module's internal calendar fetch — used by the
    backtest harness (backtest/p21_momentum/) to iterate a historical range
    without reaching into the private ``_valid_days`` helper.
    """
    return _valid_days(start, end)


def last_trading_day_of_month(year: int, month: int) -> date:
    """Return the last NYSE trading day of the given month."""
    if month == 12:
        next_month_start = date(year + 1, 1, 1)
    else:
        next_month_start = date(year, month + 1, 1)
    month_start = date(year, month, 1)
    days = _valid_days(month_start, next_month_start)
    if not days:
        raise ValueError(f"No trading days found for {year}-{month:02d}")
    last_day = max(d for d in days if d < next_month_start)
    return last_day


def first_trading_day_of_month(year: int, month: int) -> date:
    """Return the first NYSE trading day of the given month."""
    if month == 12:
        month_end = date(year, 12, 31)
    else:
        month_end = date(year, month + 1, 1)
    month_start = date(year, month, 1)
    days = _valid_days(month_start, month_end)
    if not days:
        raise ValueError(f"No trading days found for {year}-{month:02d}")
    return min(d for d in days if d < month_end or d == month_end)


def is_last_trading_day_of_month(d: date) -> bool:
    """Return True if ``d`` is the last NYSE trading day of its month."""
    if not is_trading_day(d):
        return False
    return d == last_trading_day_of_month(d.year, d.month)


def is_first_trading_day_of_month(d: date) -> bool:
    """Return True if ``d`` is the first NYSE trading day of its month."""
    if not is_trading_day(d):
        return False
    return d == first_trading_day_of_month(d.year, d.month)
