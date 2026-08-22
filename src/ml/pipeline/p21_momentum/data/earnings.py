"""
P21 Momentum — Earnings calendar (F6 blackout, docs/pipeline-specification.md §5).

**This is the one place the pipeline talks to yfinance directly**, rather
than through ``src/data/downloader/``. No downloader class in this repo
exposes ``get_earnings_dates()`` (spec §2), so a small isolated helper lives
here instead of forking the shared downloader layer for one call.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import yfinance as yf

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def next_earnings_date(ticker: str) -> Optional[date]:
    """
    Return the next upcoming earnings date for ticker, or None if unknown.

    Args:
        ticker: yfinance-normalized symbol.

    Returns:
        The earliest future earnings date yfinance reports, or None if the
        call fails or no upcoming date is available. A None here is treated
        by F6 as "not excludable on earnings grounds" — the pipeline must
        not stall a rebalance on a single ticker's flaky earnings-date
        lookup (matches F4's "pass on missing data" philosophy).
    """
    try:
        t = yf.Ticker(ticker)
        dates_df = t.get_earnings_dates(limit=8)
        if dates_df is None or dates_df.empty:
            return None
        today = date.today()
        future = [d.date() for d in dates_df.index if d.date() >= today]
        if not future:
            return None
        return min(future)
    except Exception:
        _logger.warning("Could not fetch earnings date for %s — treating as unknown", ticker)
        return None
