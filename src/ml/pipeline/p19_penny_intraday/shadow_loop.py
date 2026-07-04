"""
P19 shadow loop (Phase 1) — stateless `run-once` that logs, never alerts.

Each invocation: load the day's watchlist → take one delayed IBKR snapshot of every
name → compute %-move / RVOL-so-far → append to the SQLite shadow store. A separate
`eod-backfill` fills O/H/L/C after the close. Designed to be driven by a short
market-hours cron; idempotent and crash-safe (no in-memory state between runs).

The feed, store, and EOD OHLC fetcher are injectable so the orchestration is testable
without a live Gateway.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.metrics import compute_signal
from src.ml.pipeline.p19_penny_intraday.models.watchlist_entry import WatchlistEntry
from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_OUTPUT_DIR = "results/p19_penny_intraday"

# WatchlistEntry fields we can rehydrate from watchlist.json.
_ENTRY_FIELDS = {
    "ticker",
    "source",
    "tier",
    "explosive",
    "company_name",
    "prior_close",
    "avg_volume_30d",
    "float_shares",
    "market_cap",
    "dilution_penalty",
    "short_interest_pct_float",
    "has_catalyst",
    "catalyst_signals",
}

OhlcFetcher = Callable[[str, str], Dict[str, float] | None]


class ShadowLoop:
    def __init__(
        self,
        config: P19Config,
        target_date: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        feed: Any = None,
        store: ShadowStore | None = None,
    ) -> None:
        self.cfg = config
        self.target_date = target_date
        self.output_dir = output_dir
        if feed is None:
            from src.ml.pipeline.p19_penny_intraday.intraday_feed import IBKRIntradayFeed

            feed = IBKRIntradayFeed(config.feed_config)
        self._feed = feed
        self._store = store or ShadowStore(os.path.join(output_dir, "shadow.sqlite"))

    # ── Watchlist ──────────────────────────────────────────────────────────

    def _load_watchlist(self) -> List[WatchlistEntry]:
        path = Path(self.output_dir) / self.target_date / "watchlist.json"
        if not path.exists():
            _logger.warning("No watchlist at %s — run build-watchlist first", path)
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries: List[WatchlistEntry] = []
        for d in payload.get("entries", []):
            entries.append(WatchlistEntry(**{k: v for k, v in d.items() if k in _ENTRY_FIELDS}))
        return entries

    # ── One poll (shadow) ──────────────────────────────────────────────────

    def run_once(self) -> Dict[str, Any]:
        entries = self._load_watchlist()
        if not entries:
            return {"date": self.target_date, "polled": 0, "logged": 0, "reason": "no watchlist"}

        if not self._feed.connect():
            return {"date": self.target_date, "polled": len(entries), "logged": 0, "reason": "feed unavailable"}
        try:
            quotes = self._feed.snapshot([e.ticker for e in entries])
        finally:
            self._feed.disconnect()

        now = datetime.now(UTC)
        lot = self.cfg.feed_config.ibkr_volume_lot_size
        signals = [
            compute_signal(e, quotes[e.ticker], now, lot)
            for e in entries
            if e.ticker in quotes and (quotes[e.ticker].get("last") or 0) > 0
        ]
        logged = self._store.append_many(self.target_date, signals)
        _logger.info(
            "Shadow poll %s: polled=%d quotes=%d logged=%d", self.target_date, len(entries), len(quotes), logged
        )
        return {"date": self.target_date, "polled": len(entries), "quotes": len(quotes), "logged": logged}

    # ── EOD backfill ───────────────────────────────────────────────────────

    def eod_backfill(self, ohlc_fetcher: OhlcFetcher | None = None) -> Dict[str, Any]:
        fetcher = ohlc_fetcher or self._default_ohlc_fetcher
        tickers = self._store.tickers_for_date(self.target_date)
        updated = 0
        for t in tickers:
            ohlc = fetcher(t, self.target_date)
            if ohlc:
                updated += self._store.update_eod(self.target_date, t, ohlc)
        _logger.info("EOD backfill %s: %d tickers, %d rows updated", self.target_date, len(tickers), updated)
        return {"date": self.target_date, "tickers": len(tickers), "rows_updated": updated}

    @staticmethod
    def _default_ohlc_fetcher(ticker: str, date: str) -> Dict[str, float] | None:
        """Day OHLC via DataManager (cached in DATA_CACHE_DIR); best-effort."""
        try:
            from src.data.data_manager import DataManager

            d = datetime.strptime(date, "%Y-%m-%d")
            df = DataManager().get_ohlcv(ticker, "1d", d, d + timedelta(days=1))
            if df is None or df.empty:
                return None
            row = df.iloc[-1]
            return {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
        except Exception:
            _logger.debug("EOD OHLC fetch failed for %s", ticker)
            return None
