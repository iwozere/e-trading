"""
P21 Momentum backtest — missing-data handling (spec §14.4).

Lives here, not in the live pipeline's ``data/prices.py``, because live data
does not have this class of gap: the live pipeline always fetches a
contiguous recent window from a currently-listed ticker. A frozen 2005-2026
panel can have holiday-calendar quirks, brief data-vendor gaps, or (rarely,
under Option A survivorship bias — see ``docs/pipeline-specification.md``
§14.3) an apparent mid-history disappearance.

Rules (spec §14.4):
- Forward-fill prices for a maximum of 3 trading days; beyond that, treat
  the name as untradeable that month.
- Never back-fill.
- A name that disappears mid-month: liquidate at the last available price
  with a -30% haircut if the disappearance is performance-related,
  at last price otherwise; log EXIT_DELISTED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

MAX_FORWARD_FILL_DAYS = 3
DELISTING_HAIRCUT_PCT = -0.30


@dataclass(slots=True)
class ForwardFillResult:
    """Outcome of forward_fill_gaps() for one ticker's adjusted-close series."""

    series: pd.Series
    untradeable_dates: list  # dates where the gap exceeded MAX_FORWARD_FILL_DAYS


def forward_fill_gaps(adj_close: pd.Series, max_days: int = MAX_FORWARD_FILL_DAYS) -> ForwardFillResult:
    """
    Forward-fill missing trading-day values for up to max_days, never back-fill.

    Args:
        adj_close: Series indexed by a DatetimeIndex of NYSE trading days
            (already reindexed to the full trading calendar, with NaN where
            data is missing). Ascending by date.
        max_days: Spec §14.4 (default 3).

    Returns:
        ForwardFillResult with the filled series (NaN preserved wherever the
        gap exceeded max_days — never silently filled past the limit) and
        the list of dates that remain untradeable.
    """
    filled = adj_close.ffill(limit=max_days)
    untradeable = filled[filled.isna()].index.tolist()
    return ForwardFillResult(series=filled, untradeable_dates=untradeable)


@dataclass(slots=True)
class DelistingEvent:
    """A mid-month disappearance requiring liquidation (spec §14.4)."""

    ticker: str
    last_price: float
    liquidation_price: float
    performance_related: bool


def detect_and_liquidate_delisting(
    ticker: str,
    adj_close: pd.Series,
    as_of_date,
    performance_related: bool = True,
    haircut_pct: float = DELISTING_HAIRCUT_PCT,
) -> Optional[DelistingEvent]:
    """
    Detect whether a held ticker has disappeared as of as_of_date, and if so, price its liquidation.

    Args:
        ticker: Symbol being checked.
        adj_close: Full available series for the ticker (may end before
            as_of_date if the name has disappeared from the frozen panel).
        as_of_date: The date being evaluated (typically a rebalance date).
        performance_related: Whether the disappearance is a
            performance-related delisting (CRSP convention: apply the
            haircut) vs. a non-performance event, e.g. an acquisition at a
            fixed price (no haircut — spec is explicit these are different).
        haircut_pct: Spec §14.4 (default -30%).

    Returns:
        DelistingEvent if the series has no data at or after as_of_date
        (i.e. the name has disappeared), else None.
    """
    if adj_close.empty:
        return None
    last_available = adj_close.dropna()
    if last_available.empty:
        return None
    last_date = last_available.index[-1]
    if last_date >= pd.Timestamp(as_of_date):
        return None  # still has data at/after as_of_date -> not delisted

    last_price = float(last_available.iloc[-1])
    liquidation_price = last_price * (1 + haircut_pct) if performance_related else last_price
    return DelistingEvent(
        ticker=ticker,
        last_price=last_price,
        liquidation_price=liquidation_price,
        performance_related=performance_related,
    )
