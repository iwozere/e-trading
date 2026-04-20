"""
Position aggregator.

Merges live IBKR positions with the user's YAML watchlist into a single list
of holdings used by the PnL evaluator.
"""

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from src.notification.logger import setup_logger
from src.portfolio.pnl_alert.watchlist_loader import WatchlistEntry

_logger = setup_logger(__name__)


SOURCE_IBKR = "ibkr"
SOURCE_WATCHLIST = "watchlist"


@dataclass(frozen=True)
class Holding:
    """
    Unified view of a position used for PnL evaluation.

    Attributes:
        symbol: Ticker symbol, always uppercase.
        avg_price: Average buy price in USD (> 0).
        quantity: Position size. Defaults to 1 for watchlist-only entries.
        source: Either "ibkr" or "watchlist".
    """

    symbol: str
    avg_price: float
    quantity: float
    source: str


@dataclass(frozen=True)
class RawIbkrPosition:
    """
    Minimal IBKR position view used by the aggregator.

    Decoupled from `ib_insync` so the aggregator is easy to unit-test.

    Attributes:
        symbol: Ticker symbol.
        avg_price: Average cost (IBKR `avgCost`).
        quantity: Signed position size.
        sec_type: IBKR security type (e.g. "STK", "OPT").
    """

    symbol: str
    avg_price: float
    quantity: float
    sec_type: str


def fetch_raw_ibkr_positions(broker: Any) -> List[RawIbkrPosition]:
    """
    Extract raw IBKR positions from a connected `IBKRBroker`.

    Reads `broker.ib.positions()` directly so we retain the security type
    (`contract.secType`), which is not surfaced by `Position.metadata`.

    Args:
        broker: A connected `IBKRBroker` (or compatible object exposing `.ib`).

    Returns:
        List of `RawIbkrPosition`. Empty list on failure (warning logged).
    """
    try:
        ib = getattr(broker, "ib", None)
        if ib is None:
            _logger.warning("IBKR broker has no 'ib' attribute; skipping IBKR positions")
            return []

        raw = ib.positions()
    except Exception:
        _logger.exception("Failed to read IBKR positions")
        return []

    out: List[RawIbkrPosition] = []
    for pos in raw:
        try:
            contract = pos.contract
            symbol = getattr(contract, "symbol", None) or getattr(contract, "localSymbol", None)
            if not symbol:
                _logger.debug("Skipping IBKR position with unresolved symbol: %s", pos)
                continue

            sec_type = getattr(contract, "secType", "") or ""
            quantity = float(pos.position)
            avg_price = float(pos.avgCost)

            out.append(
                RawIbkrPosition(
                    symbol=str(symbol).upper(),
                    avg_price=avg_price,
                    quantity=quantity,
                    sec_type=sec_type.upper(),
                )
            )
        except (AttributeError, TypeError, ValueError):
            _logger.exception("Skipping malformed IBKR position: %s", pos)

    _logger.info("Fetched %d raw IBKR positions", len(out))
    return out


def merge_holdings(
    ibkr_positions: Iterable[RawIbkrPosition],
    watchlist: Iterable[WatchlistEntry],
    stk_only: bool = True,
) -> Tuple[List[Holding], List[str]]:
    """
    Merge IBKR positions and the watchlist into a unified list of holdings.

    Rules:
        - Non-STK IBKR positions are dropped when `stk_only=True`.
        - IBKR positions with quantity <= 0 or avg_price <= 0 are dropped.
        - If a symbol exists in both sources, the IBKR entry wins and a
          warning is appended to `conflicts`.

    Args:
        ibkr_positions: Raw IBKR positions (from `fetch_raw_ibkr_positions`).
        watchlist: Parsed watchlist entries.
        stk_only: If True, filter out non-stock IBKR positions.

    Returns:
        Tuple of `(holdings, conflict_symbols)`.
    """
    holdings_by_symbol: dict[str, Holding] = {}
    skipped_non_stk: List[str] = []

    for pos in ibkr_positions:
        if stk_only and pos.sec_type and pos.sec_type != "STK":
            skipped_non_stk.append(f"{pos.symbol}[{pos.sec_type}]")
            continue

        if pos.quantity <= 0 or pos.avg_price <= 0:
            _logger.debug(
                "Skipping IBKR position with non-positive qty/price: %s qty=%s avg=%s",
                pos.symbol, pos.quantity, pos.avg_price,
            )
            continue

        holdings_by_symbol[pos.symbol] = Holding(
            symbol=pos.symbol,
            avg_price=pos.avg_price,
            quantity=pos.quantity,
            source=SOURCE_IBKR,
        )

    if skipped_non_stk:
        _logger.info(
            "Filtered out %d non-STK IBKR positions: %s",
            len(skipped_non_stk), skipped_non_stk,
        )

    conflicts: List[str] = []
    for entry in watchlist:
        symbol = entry.symbol.upper()
        if symbol in holdings_by_symbol:
            ibkr = holdings_by_symbol[symbol]
            _logger.warning(
                "Symbol %s present in both IBKR (avg=%.4f) and watchlist (avg=%.4f); IBKR wins",
                symbol, ibkr.avg_price, entry.avg_price,
            )
            conflicts.append(symbol)
            continue

        holdings_by_symbol[symbol] = Holding(
            symbol=symbol,
            avg_price=entry.avg_price,
            quantity=1.0,
            source=SOURCE_WATCHLIST,
        )

    holdings = sorted(holdings_by_symbol.values(), key=lambda h: h.symbol)
    _logger.info(
        "Aggregated %d holdings (ibkr=%d, watchlist=%d, conflicts=%d)",
        len(holdings),
        sum(1 for h in holdings if h.source == SOURCE_IBKR),
        sum(1 for h in holdings if h.source == SOURCE_WATCHLIST),
        len(conflicts),
    )
    return holdings, conflicts


async def aggregate_holdings(
    broker: Optional[Any],
    watchlist: Iterable[WatchlistEntry],
    stk_only: bool = True,
) -> Tuple[List[Holding], List[str]]:
    """
    High-level helper: pull IBKR positions (best-effort) and merge with the watchlist.

    Args:
        broker: Connected `IBKRBroker`, or None to skip IBKR entirely.
        watchlist: Parsed watchlist entries.
        stk_only: If True, filter non-STK IBKR positions.

    Returns:
        Tuple of `(holdings, conflict_symbols)`.
    """
    if broker is None:
        _logger.info("IBKR disabled; using watchlist only")
        return merge_holdings([], watchlist, stk_only=stk_only)

    ibkr_positions = fetch_raw_ibkr_positions(broker)
    return merge_holdings(ibkr_positions, watchlist, stk_only=stk_only)
