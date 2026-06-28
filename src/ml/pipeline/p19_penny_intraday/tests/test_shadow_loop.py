"""Tests for the P19 shadow loop orchestration (fake feed + real SQLite store)."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.shadow_loop import ShadowLoop
from src.ml.pipeline.p19_penny_intraday.shadow_store import ShadowStore


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
    return {"ticker": ticker, "source": source, "tier": tier,
            "avg_volume_30d": avg_vol, "prior_close": prior, "catalyst_signals": []}


def _loop(tmp_path, date, feed):
    store = ShadowStore(str(tmp_path / "s.sqlite"))
    return ShadowLoop(P19Config.create_default(), date, output_dir=str(tmp_path),
                      feed=feed, store=store), store


def test_run_once_logs_quotes(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA"), _entry("BBB", source="gapper", tier="", avg_vol=0, prior=0)])
    feed = FakeFeed({
        "AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000},
        "BBB": {"last": 1.0, "open": 0.9, "high": 1.1, "low": 0.8, "prev_close": 0.85, "volume": 500},
    })
    loop, store = _loop(tmp_path, date, feed)
    summary = loop.run_once()
    assert summary["logged"] == 2 and summary["polled"] == 2
    assert store.count(date) == 2


def test_run_once_skips_zero_price(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA"), _entry("DEAD")])
    feed = FakeFeed({
        "AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000},
        "DEAD": {"last": 0.0, "open": 0.0, "high": 0.0, "low": 0.0, "prev_close": 0.0, "volume": 0},
    })
    loop, store = _loop(tmp_path, date, feed)
    assert loop.run_once()["logged"] == 1            # DEAD (last=0) skipped


def test_run_once_feed_unavailable(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    loop, store = _loop(tmp_path, date, FakeFeed({}, ok=False))
    res = loop.run_once()
    assert res["logged"] == 0 and res["reason"] == "feed unavailable"


def test_run_once_no_watchlist(tmp_path):
    loop, _ = _loop(tmp_path, "2026-06-24", FakeFeed({}))
    assert loop.run_once()["reason"] == "no watchlist"


def test_eod_backfill_updates_rows(tmp_path):
    date = "2026-06-24"
    _watchlist(tmp_path, date, [_entry("AAA")])
    feed = FakeFeed({"AAA": {"last": 3.0, "open": 2.5, "high": 3.2, "low": 2.4, "prev_close": 2.0, "volume": 3000}})
    loop, store = _loop(tmp_path, date, feed)
    loop.run_once()
    res = loop.eod_backfill(ohlc_fetcher=lambda t, d: {"open": 2.5, "high": 3.5, "low": 2.3, "close": 3.1})
    assert res["rows_updated"] == 1
    assert store.tickers_for_date(date) == []        # backfilled → no longer pending
