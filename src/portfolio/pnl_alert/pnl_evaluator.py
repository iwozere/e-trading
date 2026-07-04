"""
PnL evaluator.

Pure function that takes merged holdings + current prices and returns the
rows that qualify for the alert digest, sorted by PnL% descending.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.position_aggregator import Holding

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class AlertRow:
    """
    One row in the alert digest.

    Attributes:
        symbol: Ticker symbol.
        avg_price: Average buy price used for PnL calculation.
        current_price: Latest close price used.
        quantity: Position size (1 for watchlist-only entries).
        pnl_abs: Absolute PnL in USD = `(current - avg) * quantity`.
        pnl_pct: Fractional PnL = `(current - avg) / avg`. 0.10 means +10%.
        source: "ibkr" or "watchlist".
    """

    symbol: str
    avg_price: float
    current_price: float
    quantity: float
    pnl_abs: float
    pnl_pct: float
    source: str


def evaluate(
    holdings: Iterable[Holding],
    prices: Dict[str, float],
    threshold_pct: float,
) -> List[AlertRow]:
    """
    Compute PnL per holding and return rows meeting the threshold.

    Args:
        holdings: Merged holdings from the aggregator.
        prices: Mapping of `{symbol: current_price}`.
        threshold_pct: Inclusive threshold. A row is kept iff
            `pnl_pct >= threshold_pct`.

    Returns:
        List of `AlertRow`, sorted by `pnl_pct` descending. Ties are broken by
        `pnl_abs` descending, then by symbol ascending.
    """
    if threshold_pct <= 0:
        raise ValueError(f"threshold_pct must be > 0, got {threshold_pct}")

    rows: List[AlertRow] = []
    missing_prices: List[str] = []

    for holding in holdings:
        price = prices.get(holding.symbol)
        if price is None:
            missing_prices.append(holding.symbol)
            continue

        if holding.avg_price <= 0:
            _logger.warning(
                "Ignoring holding %s with non-positive avg_price: %s",
                holding.symbol,
                holding.avg_price,
            )
            continue

        pnl_pct = (price - holding.avg_price) / holding.avg_price
        pnl_abs = (price - holding.avg_price) * holding.quantity

        if pnl_pct < threshold_pct:
            continue

        rows.append(
            AlertRow(
                symbol=holding.symbol,
                avg_price=holding.avg_price,
                current_price=price,
                quantity=holding.quantity,
                pnl_abs=pnl_abs,
                pnl_pct=pnl_pct,
                source=holding.source,
            )
        )

    rows.sort(key=lambda r: (-r.pnl_pct, -r.pnl_abs, r.symbol))

    if missing_prices:
        _logger.warning(
            "No current price for %d symbols (excluded from alert): %s",
            len(missing_prices),
            missing_prices,
        )

    _logger.info(
        "Evaluated PnL: %d rows above %.2f%% threshold",
        len(rows),
        threshold_pct * 100,
    )
    return rows
