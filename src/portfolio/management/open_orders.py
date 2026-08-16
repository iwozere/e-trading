"""
Live IBKR open-orders fetch, for stop-loss coverage checking.

Read-only. Connects to the **live** Gateway (never paper — real stops only
exist on the live account) and reports total working protective-order
quantity per symbol.

IBKR Flex Query cannot report open orders — Flex only covers account
*activity* (positions, trades, cash, corporate actions), not a live
order-book snapshot. This has to come from a live Gateway/TWS API session
(`reqAllOpenOrders`). See ``docs/brainstorm.md`` "Key finding" for detail.

Setup requirement: for a manually-placed stop (via the TWS/Gateway GUI) to be
visible here, the connecting clientId must be configured as the account's
Master API Client ID (Gateway/TWS -> Configure -> API -> Settings). Without
that, `reqAllOpenOrders()` only sees orders placed by this same API session,
and every manually-set stop would be (wrongly) reported as missing.
"""

from collections import defaultdict
from typing import Dict, Iterable

from src.notification.logger import setup_logger
from src.portfolio.management.config import PROTECTIVE_ORDER_TYPES

_logger = setup_logger(__name__)


class IBKROpenOrdersFeed:
    """Read-only live-IBKR open-orders lookup."""

    def __init__(self, host: str, port: int, client_id: int) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib = None

    def connect(self, attempts: int = 2, backoff_seconds: float = 3.0) -> bool:
        """
        Connect read-only to the live Gateway, retrying a few times.

        `readonly=True` so this can never place, modify, or cancel an order
        even if called wrong — it only ever looks.
        """
        try:
            from src.common.asyncio_compat import ensure_event_loop

            ensure_event_loop()  # Py3.14: must run before import ib_async/ib_insync
            try:
                from ib_async import IB  # type: ignore[import-not-found]
            except ImportError:
                from ib_insync import IB  # type: ignore[import-not-found]
        except Exception:
            _logger.warning("ib_async/ib_insync unavailable — open-orders feed disabled")
            return False

        for attempt in range(1, attempts + 1):
            ib = IB()
            try:
                ib.connect(self.host, self.port, clientId=self.client_id, timeout=15, readonly=True)
                self._ib = ib
                return True
            except Exception as e:
                _logger.warning(
                    "IBKR open-orders feed connect %s:%s attempt %d/%d failed (%s: %s)",
                    self.host,
                    self.port,
                    attempt,
                    attempts,
                    type(e).__name__,
                    e,
                )
                try:
                    ib.disconnect()
                except Exception:
                    pass
                if attempt < attempts:
                    try:
                        ib.sleep(backoff_seconds)
                    except Exception:
                        import time

                        time.sleep(backoff_seconds)
        return False

    def protective_order_qty(self, symbols: Iterable[str]) -> Dict[str, float]:
        """
        Sum working protective-order (STP/STP LMT/TRAIL/TRAIL LIMIT SELL)
        quantity per symbol.

        Args:
            symbols: Symbols to report on (others are ignored even if they
                have open orders).

        Returns:
            {symbol: total working protective qty}. Symbols with no
            protective orders are simply absent — treat missing as 0.
        """
        if self._ib is None:
            return {}

        symbol_set = {s.upper() for s in symbols}
        try:
            trades = self._ib.reqAllOpenOrders()
        except Exception as e:
            _logger.warning("reqAllOpenOrders failed: %s: %s", type(e).__name__, e)
            return {}

        totals: Dict[str, float] = defaultdict(float)
        for trade in trades:
            try:
                contract = trade.contract
                order = trade.order
                symbol = (getattr(contract, "symbol", "") or "").upper()
                if symbol not in symbol_set:
                    continue
                # A protective order for a long position is a resting SELL
                # of one of the stop types — a stray BUY-side STP/TRAIL
                # (e.g. protecting a short) must not count as coverage here.
                if order.action != "SELL" or order.orderType not in PROTECTIVE_ORDER_TYPES:
                    continue
                # Prefer remaining (post-partial-fill) quantity when known;
                # fall back to the order's original total quantity.
                remaining = getattr(getattr(trade, "orderStatus", None), "remaining", None)
                qty = float(remaining) if remaining else float(order.totalQuantity)
                totals[symbol] += qty
            except (AttributeError, TypeError, ValueError):
                _logger.debug("Skipping malformed open order/trade: %s", trade)

        return dict(totals)

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None


def fetch_protective_qty(feed: IBKROpenOrdersFeed, symbols: Iterable[str]) -> tuple[bool, Dict[str, float]]:
    """
    Connect, fetch, and disconnect in one call.

    `ib_insync`/`ib_async` ties its session to a single event loop per
    thread, so the whole connect/fetch/disconnect lifecycle must run on the
    same thread — this is the one function `runner.py` offloads via
    `asyncio.to_thread`, rather than making three separate offloaded calls
    that could land on different worker threads.

    Args:
        feed: An (unconnected) `IBKROpenOrdersFeed`.
        symbols: Symbols to report on.

    Returns:
        `(connected, protective_qty_by_symbol)`. `protective_qty_by_symbol`
        is `{}` when `connected` is False.
    """
    if not feed.connect():
        return False, {}
    try:
        return True, feed.protective_order_qty(symbols)
    finally:
        feed.disconnect()
