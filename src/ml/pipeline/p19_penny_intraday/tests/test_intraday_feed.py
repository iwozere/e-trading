"""Tests for IBKRIntradayFeed.snapshot()'s contract-qualification and failure surfacing."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p19_penny_intraday.config import P19FeedConfig
from src.ml.pipeline.p19_penny_intraday.intraday_feed import IBKRIntradayFeed


class FakeTicker:
    """Stand-in for ib_async's Ticker: pre-populated, no further ticks arrive."""

    def __init__(self, contract, last=1.23):
        self.contract = contract
        self.last = last
        self.close = last
        self.open = last
        self.high = last
        self.low = last
        self.volume = 100


class FakeIB:
    """Minimal stand-in for ib_async.IB covering qualifyContracts + reqMktData."""

    def __init__(self, unresolvable=(), raising=()):
        self.unresolvable = set(unresolvable)
        self.raising = set(raising)
        self.cancelled = []

    def qualifyContracts(self, *contracts):
        qualified = []
        for c in contracts:
            if c.symbol not in self.unresolvable:
                c.conId = hash(c.symbol) % 100000 or 1  # any nonzero id
                qualified.append(c)
        return qualified

    def reqMktData(self, contract, *args):
        del args
        if contract.symbol in self.raising:
            raise RuntimeError(f"pacing violation for {contract.symbol}")
        return FakeTicker(contract)

    def sleep(self, seconds):
        del seconds

    def cancelMktData(self, contract):
        self.cancelled.append(contract.symbol)


def _feed(ib):
    feed = IBKRIntradayFeed(P19FeedConfig())
    feed._ib = ib
    return feed


def test_snapshot_skips_unqualified_contracts(caplog):
    ib = FakeIB(unresolvable={"DGNX"})
    feed = _feed(ib)
    with caplog.at_level(logging.WARNING):
        out = feed.snapshot(["AAA", "DGNX"], settle_seconds=0.5)

    assert "AAA" in out and out["AAA"]["last"] == 1.23
    assert "DGNX" not in out
    assert any("unqualified" in r.message for r in caplog.records)


def test_snapshot_surfaces_reqmktdata_exception(caplog):
    ib = FakeIB(raising={"BBB"})
    feed = _feed(ib)
    with caplog.at_level(logging.WARNING):
        out = feed.snapshot(["AAA", "BBB"], settle_seconds=0.5)

    assert "AAA" in out
    assert "BBB" not in out
    warning = next(r.message for r in caplog.records if "reqMktData failed" in r.message)
    assert "BBB" in warning and "RuntimeError" in warning


def test_snapshot_all_qualified_logs_nothing(caplog):
    ib = FakeIB()
    feed = _feed(ib)
    with caplog.at_level(logging.WARNING):
        out = feed.snapshot(["AAA", "BBB"], settle_seconds=0.5)

    assert set(out) == {"AAA", "BBB"}
    assert not any("failed" in r.message for r in caplog.records)


def test_snapshot_no_connection_returns_empty():
    feed = IBKRIntradayFeed(P19FeedConfig())
    assert feed.snapshot(["AAA"]) == {}
