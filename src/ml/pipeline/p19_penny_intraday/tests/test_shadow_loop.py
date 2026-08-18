"""Tests for the P19 shadow loop orchestration (fake feed + real SQLite store)."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.sentiment_cache import SentimentCache
from src.ml.pipeline.p19_penny_intraday.shadow_loop import ShadowLoop
from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore
from src.ml.pipeline.p19_penny_intraday.structural.cache import StructuralProfileCache


class FakeFeed:
    def __init__(self, quotes, ok=True):
        self.quotes, self.ok = quotes, ok
        self.connected = False

    def connect(self):
        self.connected = self.ok
        return self.ok

    def snapshot(self, tickers, settle_seconds=5.0):
        return {t: self.quotes[t] for t in tickers if t in self.quotes}

    def disconnect(self):
        self.connected = False


def _watchlist(tmp_path, date, entries):
    d = tmp_path / date
    d.mkdir(parents=True)
    (d / "watchlist.json").write_text(json.dumps({"date": date, "entries": entries}))


def _entry(ticker, source="p17", tier="B", avg_vol=1_000_000, prior=2.0):
    return {
        "ticker": ticker,
        "source": source,
        "tier": tier,
        "avg_volume_30d": avg_vol,
        "prior_close": prior,
        "catalyst_signals": [],
    }


def _loop(tmp_path, date, feed, sentiment_fetcher=None):
    store = ShadowStore(str(tmp_path / "s.sqlite"))
    # Always inject a tmp-path structural cache -- the default path is a real
    # repo directory (results/p19_penny_intraday/structural_cache) and must
    # never be touched by tests.
    structural_cache = StructuralProfileCache(str(tmp_path / "structural_cache"))
    sentiment_cache = SentimentCache(str(tmp_path / "sentiment_cache.json"))
    # Always inject a no-op sentiment_fetcher -- the default one makes real
    # multi-provider network calls (collect_sentiment_batch_sync), which must
    # never run from a test.
    return (
        ShadowLoop(
            P19Config.create_default(),
            date,
            output_dir=str(tmp_path),
            feed=feed,
            store=store,
            structural_cache=structural_cache,
            sentiment_cache=sentiment_cache,
            sentiment_fetcher=sentiment_fetcher or (lambda tickers: {}),
        ),
        store,
    )


def test_run_once_logs_quotes(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA"), _entry("BBB", source="gapper", tier="", avg_vol=0, prior=0)])
    feed = FakeFeed(
        {
            "AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000},
            "BBB": {"last": 1.0, "open": 0.9, "high": 1.1, "low": 0.8, "prev_close": 0.85, "volume": 500},
        }
    )
    loop, store = _loop(tmp_path, date, feed)
    summary = loop.run_once()
    assert summary["logged"] == 2 and summary["polled"] == 2
    assert store.count(date) == 2


def test_run_once_skips_zero_price(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA"), _entry("DEAD")])
    feed = FakeFeed(
        {
            "AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000},
            "DEAD": {"last": 0.0, "open": 0.0, "high": 0.0, "low": 0.0, "prev_close": 0.0, "volume": 0},
        }
    )
    loop, store = _loop(tmp_path, date, feed)
    assert loop.run_once()["logged"] == 1  # DEAD (last=0) skipped


def test_run_once_feed_unavailable(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    loop, store = _loop(tmp_path, date, FakeFeed({}, ok=False))
    res = loop.run_once()
    assert res["logged"] == 0 and res["reason"] == "feed unavailable"


def test_run_once_no_watchlist(tmp_path):
    loop, _ = _loop(tmp_path, "2026-06-24", FakeFeed({}))
    assert loop.run_once()["reason"] == "no watchlist"


# ── Phase 3: sentiment attach (spec §10, context only) ───────────────────────


def test_sentiment_feature_flows_through_to_the_stored_row(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    fetched = {
        "AAA": {"mentions_24h": 42, "sentiment_score_24h": 0.35, "mentions_growth_7d": 1.5, "unrelated_field": "x"}
    }
    loop, store = _loop(tmp_path, date, feed, sentiment_fetcher=lambda tickers: fetched)
    loop.run_once()
    row = store._conn.execute("SELECT sentiment FROM shadow_log WHERE ticker='AAA'").fetchone()
    assert row[0] == "mentions_24h=42.0;sentiment_score_24h=0.35;mentions_growth_7d=1.5"


def test_no_sentiment_data_for_ticker_leaves_field_empty(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    loop, store = _loop(tmp_path, date, feed, sentiment_fetcher=lambda tickers: {})
    loop.run_once()
    row = store._conn.execute("SELECT sentiment FROM shadow_log WHERE ticker='AAA'").fetchone()
    assert row[0] in (None, "")


def test_sentiment_fetch_failure_does_not_abort_the_poll(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})

    def _boom(tickers):
        raise RuntimeError("provider down")

    loop, store = _loop(tmp_path, date, feed, sentiment_fetcher=_boom)
    summary = loop.run_once()  # must not raise
    assert summary["logged"] == 1


def test_sentiment_fetch_is_throttled_by_the_cache_ttl(tmp_path):
    """A second poll within the TTL must not re-invoke the batch fetcher."""
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    calls = []

    def _fetch(tickers):
        calls.append(tickers)
        return {"AAA": {"mentions_24h": 5}}

    loop, _store = _loop(tmp_path, date, feed, sentiment_fetcher=_fetch)
    loop.run_once()
    loop.run_once()
    assert len(calls) == 1


def test_eod_backfill_updates_rows(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    loop, store = _loop(tmp_path, date, feed)
    loop.run_once()
    res = loop.eod_backfill(ohlc_fetcher=lambda t, d: {"open": 2.5, "high": 3.5, "low": 2.3, "close": 3.1})
    assert res["rows_updated"] == 1
    assert store.tickers_for_date(date) == []  # backfilled → no longer pending


# ── v2: momentum tier + structural denormalisation ──────────────────────────


def test_run_once_sets_momentum_tier_on_every_row(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    # Explosive move: well past the default trigger thresholds.
    feed = FakeFeed({"AAA": {"last": 5.0, "open": 2.5, "high": 5.2, "low": 2.4, "prev_close": 2.0, "volume": 300000}})
    loop, store = _loop(tmp_path, date, feed)
    loop.run_once()
    row = store._conn.execute("SELECT momentum_tier, momentum_score FROM shadow_log").fetchone()
    assert row[0] in ("T1", "T2", "T3")
    assert row[1] > 0


def test_run_once_denormalises_cached_structural_profile(tmp_path):
    from datetime import date as _date

    from src.ml.pipeline.p19_penny_intraday.models.structural_profile import StructuralProfile

    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    loop, store = _loop(tmp_path, date, feed)
    loop._structural_cache.save(
        StructuralProfile(
            ticker="AAA",
            as_of=_date(2026, 6, 24),
            grade="D",
            dilution_urgency=80.0,
            disqualifiers=["reverse split within 24mo (N1)"],
            coverage=0.9,
        )
    )
    loop.run_once()
    row = store._conn.execute("SELECT structural_grade, dilution_urgency, disqualifiers FROM shadow_log").fetchone()
    assert row[0] == "D"
    assert row[1] == 80.0
    assert "reverse split" in row[2]


def test_run_once_denormalises_is_fpi(tmp_path):
    """StructuralSignals.md §2 -- FPI status must be tracked per row, not just
    folded into the grade/coverage, or calibration confounds two populations."""
    from datetime import date as _date

    from src.ml.pipeline.p19_penny_intraday.models.structural_profile import StructuralProfile

    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    loop, store = _loop(tmp_path, date, feed)
    loop._structural_cache.save(StructuralProfile(ticker="AAA", as_of=_date(2026, 6, 24), grade="C", is_fpi=True))
    loop.run_once()
    row = store._conn.execute("SELECT is_fpi FROM shadow_log").fetchone()
    assert row[0] == 1


def test_run_once_without_cached_profile_still_logs(tmp_path):
    """Decision #7: shadow logging happens for every name, profiled or not."""
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    loop, store = _loop(tmp_path, date, feed)
    summary = loop.run_once()
    assert summary["logged"] == 1
    row = store._conn.execute("SELECT structural_grade FROM shadow_log").fetchone()
    assert row[0] == ""


# ── v2: same-day outcome labels ──────────────────────────────────────────────


def test_eod_backfill_writes_same_day_labels(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    # Explosive first poll (crosses T1+) followed by a fade.
    feed = FakeFeed({"AAA": {"last": 5.0, "open": 2.5, "high": 5.2, "low": 2.4, "prev_close": 2.0, "volume": 300000}})
    loop, store = _loop(tmp_path, date, feed)
    loop.run_once()
    res = loop.eod_backfill(ohlc_fetcher=lambda t, d: {"open": 2.5, "high": 5.2, "low": 2.4, "close": 3.0})
    assert res["labelled"] == 1
    row = store._conn.execute(
        "SELECT close_retention, mae_from_alert, mfe_from_alert FROM shadow_log WHERE ticker='AAA'"
    ).fetchone()
    close_retention, mae, mfe = row
    assert close_retention is not None and 0.0 < close_retention < 1.0  # faded but held some of the move
    assert mae is not None and mae <= 0.0
    assert mfe is not None and mfe >= 0.0


def test_eod_backfill_labels_survive_when_no_trigger_crossed(tmp_path):
    """No T1+ poll that day -> close_retention still fills, MAE/MFE stay None."""
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 2.55, "open": 2.5, "high": 2.6, "low": 2.4, "prev_close": 2.5, "volume": 1000}})
    loop, store = _loop(tmp_path, date, feed)
    loop.run_once()
    loop.eod_backfill(ohlc_fetcher=lambda t, d: {"open": 2.5, "high": 2.6, "low": 2.4, "close": 2.55})
    row = store._conn.execute(
        "SELECT close_retention, mae_from_alert, mfe_from_alert FROM shadow_log WHERE ticker='AAA'"
    ).fetchone()
    assert row[0] is not None
    assert row[1] is None and row[2] is None
