"""
P19 intraday feed — IBKR delayed `reqMktData` snapshots.

One streaming market-data line per watchlist name (≤ ~100, the IBKR line budget),
in **delayed** mode (`reqMarketDataType(3)`, free). Each line carries last price,
day open/high/low, prev close, and **cumulative day volume** — everything the
triggers and RVOL-so-far need, with **no per-poll historical requests** (so the
~60/10-min pacing limit never applies).

A poll = subscribe → let ticks settle briefly → read each ticker's fields. The feed
is gateway-guarded: ``connect()`` returns False (logged) when the Gateway is
unreachable, so the loop degrades instead of crashing.
"""

import math
from typing import Any, Dict, List

from src.notification.logger import setup_logger
from src.ml.pipeline.p19_penny_intraday.config import P19FeedConfig

_logger = setup_logger(__name__)


def _num(v: Any) -> float:
    """Coerce ib_async's nan/None tick values to 0.0."""
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


class IBKRIntradayFeed:
    """Delayed IBKR market-data snapshots for a list of tickers."""

    def __init__(self, feed_config: P19FeedConfig) -> None:
        self.cfg = feed_config
        self._ib = None

    def connect(self) -> bool:
        try:
            try:
                from ib_async import IB
            except ImportError:
                from ib_insync import IB
        except Exception:
            _logger.warning("ib_async/ib_insync unavailable — intraday feed disabled")
            return False

        ib = IB()
        try:
            ib.connect(self.cfg.ibkr_host, self.cfg.ibkr_port,
                       clientId=self.cfg.ibkr_client_id, timeout=15, readonly=True)
        except Exception as e:
            _logger.warning("IBKR feed connect %s:%s failed (%s: %s)",
                            self.cfg.ibkr_host, self.cfg.ibkr_port, type(e).__name__, e)
            return False
        ib.reqMarketDataType(self.cfg.ibkr_market_data_type)   # 3 = delayed
        self._ib = ib
        return True

    def snapshot(self, tickers: List[str], settle_seconds: float = 12.0) -> Dict[str, Dict[str, Any]]:
        """
        Subscribe to ``tickers``, wait (adaptively) for ticks, and return raw quote
        dicts: ``{ticker: {last, open, high, low, prev_close, volume}}``.

        Waits up to ``settle_seconds``, returning early once ~80% of names have a
        price — many delayed subscriptions need several seconds to populate.
        """
        if self._ib is None:
            return {}
        from_ib = self._ib
        try:
            from ib_async import Stock
        except ImportError:
            from ib_insync import Stock

        subs = {}
        for sym in tickers:
            try:
                subs[sym] = from_ib.reqMktData(Stock(sym, "SMART", "USD"), "", False, False)
            except Exception:
                _logger.debug("reqMktData failed for %s", sym)

        # Adaptive settle: poll until most tickers have a price or the budget runs out.
        waited = 0.0
        target = max(1, int(0.8 * len(subs)))
        while waited < settle_seconds:
            from_ib.sleep(0.5)
            waited += 0.5
            ready = sum(1 for t in subs.values()
                        if _num(getattr(t, "last", None)) or _num(getattr(t, "close", None)))
            if ready >= target:
                break

        out: Dict[str, Dict[str, Any]] = {}
        for sym, t in subs.items():
            last = _num(getattr(t, "last", None)) or _num(getattr(t, "close", None))
            out[sym] = {
                "last": last,
                "open": _num(getattr(t, "open", None)),
                "high": _num(getattr(t, "high", None)),
                "low": _num(getattr(t, "low", None)),
                "prev_close": _num(getattr(t, "close", None)),
                "volume": _num(getattr(t, "volume", None)),
            }
            try:
                from_ib.cancelMktData(t.contract)
            except Exception:
                pass
        return out

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            except Exception:
                pass
            self._ib = None
