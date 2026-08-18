"""
P19 shadow loop (Phase 1) — stateless `run-once` that logs, never alerts.

Each invocation: load the day's watchlist → take one delayed IBKR snapshot of every
name → compute %-move / RVOL-so-far → append to the SQLite shadow store. A separate
`eod-backfill` fills O/H/L/C after the close. Designed to be driven by a short
market-hours cron; idempotent and crash-safe (no in-memory state between runs).

The feed, store, and EOD OHLC fetcher are injectable so the orchestration is testable
without a live Gateway.
"""

import os
from datetime import UTC, datetime, timedelta
from typing import Optional, Any, Callable, Dict, List

from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.metrics import classify_momentum_tier, compute_same_day_labels, compute_signal
from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.ml.pipeline.p19_penny_intraday.models.structural_profile import StructuralProfile
from src.ml.pipeline.p19_penny_intraday.sentiment_cache import SentimentCache
from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore
from src.ml.pipeline.p19_penny_intraday.structural.cache import StructuralProfileCache
from src.ml.pipeline.p19_penny_intraday.watchlist_builder import load_watchlist
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_OUTPUT_DIR = "results/p19_penny_intraday"

OhlcFetcher = Callable[[str, str], Dict[str, float] | None]
SentimentFetcher = Callable[[List[str]], Dict[str, Any]]


def _default_sentiment_fetch(tickers: List[str]) -> Dict[str, Any]:
    """
    Batch sentiment via the shared cross-pipeline collector (spec §10). Errors
    inside individual providers are already handled by
    ``collect_sentiment_batch_sync`` itself (per-provider circuit breakers) —
    this only guards the call boundary.
    """
    from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch_sync

    result = collect_sentiment_batch_sync(tickers, output_format="dict")
    return result if isinstance(result, dict) else {}


class ShadowLoop:
    def __init__(
        self,
        config: P19Config,
        target_date: str,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        feed: Optional[Any] = None,
        store: ShadowStore | None = None,
        structural_cache: Optional[StructuralProfileCache] = None,
        sentiment_cache: Optional[SentimentCache] = None,
        sentiment_fetcher: Optional[SentimentFetcher] = None,
    ) -> None:
        self.cfg = config
        self.target_date = target_date
        self.output_dir = output_dir
        if feed is None:
            from src.ml.pipeline.p19_penny_intraday.intraday_feed import IBKRIntradayFeed

            feed = IBKRIntradayFeed(config.feed_config)
        self._feed = feed
        self._store = store or ShadowStore(os.path.join(output_dir, "shadow.sqlite"))
        # Read-only lookup of the pre-market Layer 0 profile (spec §12.1 —
        # denormalised as a point-in-time snapshot, never a live join). Missing
        # for a name that hasn't been profiled yet — that's fine, the row still
        # gets logged (decision #7), just with structural_grade="".
        self._structural_cache = structural_cache or StructuralProfileCache(
            ttl_days=config.structural_config.profile_cache_ttl_days
        )
        self._sentiment_cache = sentiment_cache or SentimentCache(os.path.join(output_dir, "sentiment_cache.json"))
        self._sentiment_fetcher = sentiment_fetcher or _default_sentiment_fetch

    # ── One poll (shadow) ──────────────────────────────────────────────────

    def run_once(self) -> Dict[str, Any]:
        entries = load_watchlist(self.output_dir, self.target_date)
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
        sentiment_by_ticker = self._fetch_sentiment([e.ticker for e in entries])
        for s in signals:
            self._apply_momentum_and_structural(s)
            self._apply_sentiment(s, sentiment_by_ticker)
        logged = self._store.append_many(self.target_date, signals)
        _logger.info(
            "Shadow poll %s: polled=%d quotes=%d logged=%d", self.target_date, len(entries), len(quotes), logged
        )
        return {"date": self.target_date, "polled": len(entries), "quotes": len(quotes), "logged": logged}

    def _apply_momentum_and_structural(self, signal: IntradaySignal) -> None:
        """
        Mutates `signal` in place: momentum_score/momentum_tier (spec §16 item
        5 — the "simulated trigger point", log-only, no Disposition Engine yet)
        and the structural-axis denormalisation from the cached Layer 0 profile
        (spec §12.1). A name with no cached profile yet just logs with
        structural_grade="" — never blocks or drops the poll (decision #7:
        shadow logging happens for every watchlist name regardless of grade).
        """
        score, tier = classify_momentum_tier(signal, self.cfg.trigger_config)
        signal.momentum_score = score
        signal.momentum_tier = tier

        profile: Optional[StructuralProfile] = self._structural_cache.load(signal.ticker)
        if profile is None:
            return
        signal.structural_grade = profile.grade
        signal.dilution_urgency = profile.dilution_urgency
        signal.insider_conviction = profile.insider_conviction
        signal.runway_quarters = profile.runway_quarters
        signal.disqualifiers = list(profile.disqualifiers)
        signal.structural_coverage = profile.coverage
        signal.is_fpi = profile.is_fpi

    # ── Sentiment (spec §10, context only — never a trigger) ────────────────

    def _fetch_sentiment(self, tickers: List[str]) -> Dict[str, Any]:
        """Batch fetch, throttled by ``SentimentCache`` (see its docstring for
        why per-poll would be far too frequent for a multi-provider fetch)."""
        if not tickers:
            return {}
        if self._sentiment_cache.is_fresh():
            return self._sentiment_cache.data()
        try:
            data = self._sentiment_fetcher(tickers)
        except Exception:
            _logger.warning("Sentiment batch fetch failed for %d tickers", len(tickers))
            return self._sentiment_cache.data()  # serve stale-but-something over nothing
        self._sentiment_cache.save(data)
        return data

    @staticmethod
    def _apply_sentiment(signal: IntradaySignal, sentiment_by_ticker: Dict[str, Any]) -> None:
        feat = sentiment_by_ticker.get(signal.ticker)
        if not feat:
            return
        out: Dict[str, float] = {}
        for key in ("mentions_24h", "sentiment_score_24h", "mentions_growth_7d"):
            val = feat.get(key)
            if val is not None:
                try:
                    out[key] = float(val)
                except (TypeError, ValueError):
                    continue
        if out:
            signal.sentiment = out

    # ── EOD backfill ───────────────────────────────────────────────────────

    def eod_backfill(self, ohlc_fetcher: OhlcFetcher | None = None) -> Dict[str, Any]:
        fetcher = ohlc_fetcher or self._default_ohlc_fetcher
        tickers = self._store.tickers_for_date(self.target_date)
        updated = 0
        for t in tickers:
            ohlc = fetcher(t, self.target_date)
            if ohlc:
                updated += self._store.update_eod(self.target_date, t, ohlc)

        labelled = self._backfill_same_day_labels()
        _logger.info(
            "EOD backfill %s: %d tickers, %d rows updated, %d names labelled",
            self.target_date,
            len(tickers),
            updated,
            labelled,
        )
        return {"date": self.target_date, "tickers": len(tickers), "rows_updated": updated, "labelled": labelled}

    def _backfill_same_day_labels(self) -> int:
        """
        Fills high_time/close_retention/mae_from_alert/mfe_from_alert (spec
        §12.2) for every name that now has an EOD close but no labels yet —
        i.e., every name update_eod just touched, this run or a prior one.
        """
        labelled = 0
        for ticker in self._store.tickers_for_date_needing_labels(self.target_date):
            polls = self._store.polls_for_date_ticker(self.target_date, ticker)
            eod = self._store.get_eod(self.target_date, ticker)
            if eod is None:
                continue
            labels = compute_same_day_labels(polls, eod)
            self._store.update_same_day_labels(self.target_date, ticker, labels)
            labelled += 1
        return labelled

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
