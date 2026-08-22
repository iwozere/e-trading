"""
P21 Momentum — Universe construction.

Wraps ``src.util.tickers_list.get_sp500_constituents_with_sector()`` and
writes the run's ``universe.json`` (docs/pipeline-specification.md §3).

**Known gap, flagged rather than silently fixed:** this always returns the
*current* S&P 500 membership. That's correct for the live pipeline
(constituents are always "current" there by construction) but is exactly
Option A from spec §14.3 when reused by the backtest harness — applying
today's constituents to historical dates introduces survivorship bias. This
module has no time dimension; the survivorship-bias banner and any
point-in-time membership logic belong in the backtest harness
(``backtest/p21_momentum/``), not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.notification.logger import setup_logger
from src.util.tickers_list import get_sp500_constituents_with_sector

_logger = setup_logger(__name__)


@dataclass(slots=True)
class UniverseConstituent:
    """One S&P 500 constituent as of the universe fetch."""

    ticker: str
    sector: str


def fetch_universe() -> List[UniverseConstituent]:
    """
    Fetch current S&P 500 constituents with sector.

    Returns:
        List of UniverseConstituent, one per constituent. Tickers are
        already yfinance-normalized (dots -> hyphens) by
        get_sp500_constituents_with_sector().

    Raises:
        Exception: propagated from the Wikipedia fetch; callers should
            treat a failure here as fatal (spec §13: "Constituent list
            loaded >= 450 names" is an ABORT-level gate).
    """
    df: pd.DataFrame = get_sp500_constituents_with_sector()
    constituents = [
        UniverseConstituent(ticker=str(row.ticker), sector=str(row.sector)) for row in df.itertuples(index=False)
    ]
    _logger.info("Fetched %d S&P 500 constituents", len(constituents))
    return constituents


def universe_to_json(constituents: List[UniverseConstituent], as_of: str) -> dict:
    """
    Build the ``universe.json`` payload for a run.

    Args:
        constituents: Result of fetch_universe().
        as_of: Signal date, ISO format (YYYY-MM-DD).

    Returns:
        Dict ready for json.dump — {"as_of": ..., "constituents": [...]}.
    """
    return {
        "as_of": as_of,
        "count": len(constituents),
        "constituents": [{"ticker": c.ticker, "sector": c.sector} for c in constituents],
    }
