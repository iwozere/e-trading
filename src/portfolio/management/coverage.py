"""
Stop-loss coverage classification.

Pure functions: no I/O, easy to unit test.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class CoverageStatus(str, Enum):
    """Classification of a position's stop-loss protection."""

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    UNCOVERED = "uncovered"


@dataclass(frozen=True)
class CoverageRow:
    """
    Coverage result for one held ticker.

    Attributes:
        ticker: Ticker symbol.
        position_qty: Current held quantity.
        protected_qty: Working protective-order quantity for this ticker,
            capped at `position_qty` — excess protective quantity doesn't
            make the position "more than covered".
        status: Classification.
    """

    ticker: str
    position_qty: float
    protected_qty: float
    status: CoverageStatus


def classify(position_qty: float, protective_qty: float) -> CoverageStatus:
    """
    Classify stop-loss coverage for one ticker.

    Args:
        position_qty: Current held quantity (> 0).
        protective_qty: Working protective-order quantity (>= 0).

    Returns:
        `CoverageStatus`.
    """
    if protective_qty <= 0:
        return CoverageStatus.UNCOVERED
    if protective_qty >= position_qty:
        return CoverageStatus.COVERED
    return CoverageStatus.PARTIALLY_COVERED


def evaluate(positions: Dict[str, float], protective_qty_by_symbol: Dict[str, float]) -> List[CoverageRow]:
    """
    Evaluate coverage for every ticker in `positions`.

    Args:
        positions: {ticker: quantity} for tickers to evaluate.
        protective_qty_by_symbol: {ticker: working protective-order qty}
            (e.g. from `IBKROpenOrdersFeed.protective_order_qty`). Tickers
            absent from this dict are treated as 0 (no protection).

    Returns:
        One `CoverageRow` per ticker in `positions`, sorted by ticker.
    """
    rows = []
    for ticker, qty in positions.items():
        protected = protective_qty_by_symbol.get(ticker, 0.0)
        rows.append(
            CoverageRow(
                ticker=ticker,
                position_qty=qty,
                protected_qty=min(protected, qty),
                status=classify(qty, protected),
            )
        )
    return sorted(rows, key=lambda r: r.ticker)
