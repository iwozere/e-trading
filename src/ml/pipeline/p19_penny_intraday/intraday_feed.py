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
from typing import TYPE_CHECKING, Any, Dict, List

from src.ml.pipeline.p19_penny_intraday.config import P19FeedConfig
from src.notification.logger import setup_logger

if TYPE_CHECKING:
    # Annotation-only; type-check against ib_insync's types -- see
    # src/trading/broker/ibkr_utils.py for why. The real (runtime-selected)
    # import happens in connect() below.
    from ib_insync import IB

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
        self._ib: "IB | None" = None

    def connect(self, attempts: int = 2, backoff_seconds: float = 3.0) -> bool:
        """
        Connect to the Gateway, retrying a few times.

        For unattended multi-week running, a single retry smooths over the daily
        Gateway re-auth/restart and the "Re-login required" churn — a transient
        failure on one attempt usually succeeds on the next.
        """
        try:
            from src.common.asyncio_compat import ensure_event_loop

            ensure_event_loop()  # Py3.14: must run before import ib_async/ib_insync
            if TYPE_CHECKING:
                # Type-check against ib_insync's types -- see ibkr_utils.py for why.
                from ib_insync import IB
            else:
                try:  # prefer maintained ib_async
                    from ib_async import IB  # type: ignore[import-not-found]
                except ImportError:
                    from ib_insync import IB  # type: ignore[import-not-found]
        except Exception:
            _logger.warning("ib_async/ib_insync unavailable — intraday feed disabled")
            return False

        for attempt in range(1, attempts + 1):
            ib = IB()
            try:
                ib.connect(
                    self.cfg.ibkr_host, self.cfg.ibkr_port, clientId=self.cfg.ibkr_client_id, timeout=15, readonly=True
                )
                ib.reqMarketDataType(self.cfg.ibkr_market_data_type)  # 3 = delayed
                self._ib = ib
                return True
            except Exception as e:
                _logger.warning(
                    "IBKR feed connect %s:%s attempt %d/%d failed (%s: %s)",
                    self.cfg.ibkr_host,
                    self.cfg.ibkr_port,
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
        if TYPE_CHECKING:
            # Type-check against ib_insync's types -- see ibkr_utils.py for why.
            from ib_insync import Stock
        else:
            try:  # prefer maintained ib_async
                from ib_async import Stock  # type: ignore[import-not-found]
            except ImportError:
                from ib_insync import Stock  # type: ignore[import-not-found]

        # reqMktData() keys its subscription table by contract hash, and a Contract
        # only hashes once it carries a conId — so bare Stock(sym, "SMART", "USD")
        # objects must be qualified (conId populated) before they're usable here.
        contracts = {sym: Stock(sym, "SMART", "USD") for sym in tickers}
        try:
            from_ib.qualifyContracts(*contracts.values())
        except Exception as e:
            _logger.warning("qualifyContracts failed for %d tickers: %s: %s", len(contracts), type(e).__name__, e)

        subs = {}
        failures: List[str] = []
        first_failure: Exception | None = None
        for sym, contract in contracts.items():
            if not contract.conId:
                failures.append(sym)
                continue
            try:
                subs[sym] = from_ib.reqMktData(contract, "", False, False)
            except Exception as e:
                failures.append(sym)
                if first_failure is None:
                    first_failure = e

        if failures:
            detail = (
                f"; first failure ({failures[0]}): {type(first_failure).__name__}: {first_failure}"
                if first_failure is not None
                else " (unqualified — no conId; IBKR couldn't resolve the contract)"
            )
            _logger.warning(
                "reqMktData failed for %d/%d tickers (%s)%s",
                len(failures),
                len(tickers),
                ", ".join(failures[:10]) + ("..." if len(failures) > 10 else ""),
                detail,
            )

        # Adaptive settle: poll until most tickers have a price or the budget runs out.
        waited = 0.0
        target = max(1, int(0.8 * len(subs)))
        while waited < settle_seconds:
            from_ib.sleep(0.5)
            waited += 0.5
            ready = sum(1 for t in subs.values() if _num(getattr(t, "last", None)) or _num(getattr(t, "close", None)))
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
                if t.contract is not None:
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
