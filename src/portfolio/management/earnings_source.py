"""
Earnings-date source for held tickers.

Wraps `EarningsCalendar` (P05 AI Selector) — already FMP-backed with monthly
caching and, unlike P20 Kestrel's calendar, parameterized by an arbitrary
ticker list rather than bound to Kestrel's own watchlist tables. See
``docs/brainstorm.md`` "Reuse plan" for the full comparison.

Session (BMO/AMC) detection is not implemented: `EarningsCalendar` doesn't
currently surface it, so every event comes back with `session="unknown"`,
which `earnings_window.resolve_anchor_utc` treats as the conservative
market-open anchor. See ``docs/Tasks.md`` for the follow-up.
"""

from datetime import date
from typing import Iterable, List, Optional

from src.ml.pipeline.p05_ai_selector.signals.earnings_calendar import EarningsCalendar
from src.notification.logger import setup_logger
from src.portfolio.management.earnings_window import EarningsEvent

_logger = setup_logger(__name__)


class EarningsSource:
    """Resolves upcoming earnings events for held tickers."""

    def __init__(self, calendar: Optional[EarningsCalendar] = None) -> None:
        self._calendar = calendar or EarningsCalendar()

    def get_upcoming_events(
        self,
        tickers: Iterable[str],
        as_of_date: date,
        window_days: int,
    ) -> List[EarningsEvent]:
        """
        Return one `EarningsEvent` per ticker with an earnings date within
        `window_days` of `as_of_date`.

        Args:
            tickers: Held ticker symbols.
            as_of_date: Reference date.
            window_days: Lookahead window in days.

        Returns:
            List of `EarningsEvent` (session always "unknown" today — see
            module docstring), sorted by earnings date then ticker.
        """
        ticker_list = sorted({t.upper() for t in tickers})
        if not ticker_list:
            return []

        try:
            dates_by_ticker = self._calendar.get_earnings_within_days(ticker_list, as_of_date, window_days)
        except Exception:
            _logger.exception("EarningsSource: calendar lookup failed")
            return []

        events = [EarningsEvent(ticker=ticker, earnings_date=earnings_date) for ticker, earnings_date in dates_by_ticker.items()]
        return sorted(events, key=lambda e: (e.earnings_date, e.ticker))
