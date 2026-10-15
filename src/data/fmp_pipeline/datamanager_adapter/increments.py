"""
The increment layer's read interface (spec §9.2 layer 2, §10). WP-9 implements it over ``increments/``.

Increments are yfinance bars and events newer than the vintage, stored **raw** like the vintage. The adapter
appends increment bars strictly after the security's last FMP bar and unions the events (FMP wins on a date both
have). Until WP-9 lands, ``NoIncrements`` is used and a request past the vintage's last bar returns only what the
vintage has.
"""

from __future__ import annotations

from datetime import date
from typing import Optional, Protocol

import pandas as pd

EVENT_COLUMNS_SPLITS = ["date", "numerator", "denominator"]
EVENT_COLUMNS_DIVIDENDS = ["date", "amount"]


class IncrementSource(Protocol):
    """What the adapter reads from the increment layer."""

    def bars(self, security_id: int, symbol: str, timeframe: str, after: Optional[date]) -> Optional[pd.DataFrame]:
        """
        Raw bars strictly after ``after`` (all bars when ``None``).

        Returns:
            Frame shaped like ``FmpStore.prices`` (``open`` … ``volume``, ``flags``, ``source``), or ``None``.
        """
        ...

    def events(self, security_id: int, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits ``[date, numerator, denominator]`` and dividends ``[date, amount]`` (raw basis)."""
        ...


class NoIncrements:
    """The empty increment layer (before WP-9)."""

    def bars(self, security_id: int, symbol: str, timeframe: str, after: Optional[date]) -> Optional[pd.DataFrame]:
        """No bars."""
        return None

    def events(self, security_id: int, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """No events."""
        return pd.DataFrame(columns=EVENT_COLUMNS_SPLITS), pd.DataFrame(columns=EVENT_COLUMNS_DIVIDENDS)
