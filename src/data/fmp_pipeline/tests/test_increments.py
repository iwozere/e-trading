"""Tests for the yfinance increment layer (spec §10): raw conversion, strictly-after appends, gaps, single writer."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from yfinance.exceptions import YFRateLimitError

from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.datamanager_adapter.adapter import FmpOhlcvAdapter
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.errors import DumpLocked
from src.data.fmp_pipeline.fsutil import exclusive_lock
from src.data.fmp_pipeline.increments import ingest as ingest_module
from src.data.fmp_pipeline.increments.ingest import IncrementIngester, NotTheWriter, check_writer, to_raw_bars
from src.data.fmp_pipeline.increments.store import IncrementStore
from src.data.fmp_pipeline.planner.config import IncrementsConfig, PlanConfig
from src.data.fmp_pipeline.tests.vintage_fixtures import make_vintage

NY = "America/New_York"
HOST = "pi5"
CONFIG = PlanConfig(dump_id="v", increments=IncrementsConfig(writer_host="PI5", batch_size=2, pause_s=0.5))
TODAY = date(2025, 3, 7)
SPLIT = "2025-03-05"  # 2:1, after the vintage ends (2025-02-28): yfinance restates the bars before it


def yf_daily(raw_closes: dict[str, float], split: Optional[tuple[str, float]] = None,
             dividends: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """A yfinance ``auto_adjust=False`` frame for raw closes: bars before a split restated by its ratio."""
    index = pd.DatetimeIndex(sorted(raw_closes)).tz_localize(NY)
    ratio = np.array([split[1] if split and d < pd.Timestamp(split[0]).tz_localize(NY) else 1.0 for d in index])
    close = np.array([raw_closes[d.strftime("%Y-%m-%d")] for d in index]) / ratio
    frame = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Adj Close": close,
                          "Volume": 1000 * ratio, "Dividends": 0.0, "Stock Splits": 0.0}, index=index)
    if split:
        frame.loc[pd.Timestamp(split[0]).tz_localize(NY), "Stock Splits"] = split[1]
    for day, amount in (dividends or {}).items():
        when = pd.Timestamp(day).tz_localize(NY)
        frame.loc[when, "Dividends"] = amount / (split[1] if split and when < pd.Timestamp(split[0]).tz_localize(NY)
                                                 else 1.0)
    return frame


def yf_hourly(day_closes: dict[str, float], split: Optional[tuple[str, float]] = None) -> pd.DataFrame:
    """yfinance hourly bars (09:30 … 15:30 local) for raw closes, restated before a split like the daily ones."""
    rows = {}
    for day, close in day_closes.items():
        ratio = split[1] if split and day < split[0] else 1.0
        for hour in range(9, 16):
            rows[pd.Timestamp(f"{day} {hour:02d}:30").tz_localize(NY)] = close / ratio
    index = pd.DatetimeIndex(sorted(rows))
    values = np.array([rows[i] for i in index])
    return pd.DataFrame({"Open": values, "High": values, "Low": values, "Close": values, "Adj Close": values,
                         "Volume": 10.0, "Dividends": 0.0, "Stock Splits": 0.0}, index=index)


# Feb 24-28 equal the vintage's raw closes (100 + business-day index), so the overlap verifies cleanly.
AAA_RAW = {"2025-02-24": 137.0, "2025-02-25": 138.0, "2025-02-26": 139.0, "2025-02-27": 140.0, "2025-02-28": 141.0,
           "2025-03-03": 150.0, "2025-03-04": 152.0, "2025-03-05": 76.0, "2025-03-06": 77.0, "2025-03-07": 78.0}


class FakeYahoo:
    """Serves synthetic yfinance frames; records requests."""

    def __init__(self, daily: dict[str, pd.DataFrame], hourly: Optional[dict[str, pd.DataFrame]] = None) -> None:
        self.daily = daily
        self.hourly = hourly or {}
        self.calls: list[tuple[tuple[str, ...], str, datetime, datetime]] = []
        self.fail: list[Exception] = []

    def __call__(self, symbols: list[str], interval: str, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        self.calls.append((tuple(symbols), interval, start, end))
        if self.fail:
            raise self.fail.pop(0)
        source = self.daily if interval == "1d" else self.hourly
        out = {}
        for symbol in symbols:
            frame = source.get(symbol)
            if frame is not None:
                local = pd.DatetimeIndex(frame.index).tz_localize(None)
                out[symbol] = frame[(local >= pd.Timestamp(start)) & (local < pd.Timestamp(end))]
        return out


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_vintage(tmp_path)
    return tmp_path


def _ingester(root: Path, fetch: Any, today: date = TODAY, sleeps: Optional[list[float]] = None,
              config: PlanConfig = CONFIG) -> IncrementIngester:
    recorded = sleeps if sleeps is not None else []
    return IncrementIngester(root, config, fetch, today=today, sleep=recorded.append, hostname=HOST)


def test_writer_check():
    assert check_writer("pi5", "PI5") == "PI5"
    with pytest.raises(NotTheWriter, match="not set"):
        check_writer("", "pi5")
    with pytest.raises(NotTheWriter, match="increment writer is 'laptop'"):
        check_writer("laptop", "pi5")


def test_to_raw_bars_undoes_the_split_and_keeps_the_time_axis():
    daily = to_raw_bars(yf_daily(AAA_RAW, (SPLIT, 2.0)), "1d", NY)
    assert daily.loc["2025-03-04", "close"] == 152.0 and daily.loc["2025-03-05", "close"] == 76.0
    assert daily.loc["2025-03-04", "volume"] == 1000.0 and pd.DatetimeIndex(daily.index).tz is None  # raw volume
    hourly = to_raw_bars(yf_hourly({"2025-03-04": 152.0}, (SPLIT, 2.0)), "1h", NY,
                         pd.DataFrame({"date": [SPLIT], "numerator": [2.0], "denominator": [1.0]}))
    assert hourly.index[0] == pd.Timestamp("2025-03-04 14:30") and hourly["close"].iloc[0] == 152.0
    assert hourly["date_et"].iloc[0] == pd.Timestamp("2025-03-04")
    naive = yf_hourly({"2025-03-04": 1.0})
    naive.index = pd.DatetimeIndex(naive.index).tz_localize(None)
    with pytest.raises(ValueError, match="time zone"):
        to_raw_bars(naive, "1h", NY)


def test_ingest_appends_raw_bars_strictly_after_the_vintage_across_a_split_gap(root: Path):
    yahoo = FakeYahoo({"AAA": yf_daily(AAA_RAW, (SPLIT, 2.0), {"2025-03-04": 0.5})},
                      {"AAA": yf_hourly({"2025-03-04": 152.0, "2025-03-06": 77.0}, (SPLIT, 2.0))})
    sleeps: list[float] = []
    report = _ingester(root, yahoo, sleeps=sleeps).run()
    daily = report.timeframes["1d"]
    assert report.stop_reason == "done" and report.increment_seq == 1
    assert (daily.candidates, daily.with_data, daily.updated, daily.new_bars) == (4, 1, 1, 5)
    # batches are grouped by start date: [AAA, BRK.B], [NEWC], [XYZ] for 1d, [AAA] for 1h
    assert daily.events_added == 2 and daily.mismatched == {} and sleeps == [0.5] * 4
    aaa_call = next(c for c in yahoo.calls if "AAA" in c[0] and c[1] == "1d")
    assert aaa_call[0] == ("AAA", "BRK-B") and aaa_call[2] == datetime(2025, 2, 23)  # yf symbols; 5-day overlap
    assert aaa_call[3] == datetime(2025, 3, 8)  # through today (end exclusive)
    assert yahoo.calls[0][2] == datetime(2024, 12, 28)  # XYZ's group starts earliest (last bar 2025-01-02)

    store = FmpStore(root)
    increments = IncrementStore(root)
    aaa = store.resolve("AAA")
    assert aaa is not None
    bars = increments.bars(aaa.security_id, "AAA", "1d", None)
    assert bars is not None and list(bars["close"]) == [150.0, 152.0, 76.0, 77.0, 78.0]  # raw: the split shows
    assert set(bars["source"]) == {"yfinance"} and list(bars["volume"]) == [1000] * 5
    splits, dividends = increments.events(aaa.security_id, "AAA")
    assert list(splits["numerator"]) == [2.0] and list(dividends["amount"]) == [0.5]  # dividend on raw basis
    table = pq.read_table(increments.path("1d", aaa.security_id))
    assert set(table.column("vintage").to_pylist()) == {"v"} and table.column("ingested_at").null_count == 0

    hourly = increments.bars(aaa.security_id, "AAA", "1h", None)
    assert hourly is not None and report.timeframes["1h"].candidates == 1  # only AAA has hourly history
    assert hourly["close"].iloc[0] == 152.0 and hourly.index[0] == pd.Timestamp("2025-03-04 14:30")

    adapter = FmpOhlcvAdapter(store, increments)
    split_view = adapter.get_ohlcv("AAA", "1d", datetime(2025, 2, 27), datetime(2025, 3, 7), "split",
                                   include_source=True)
    assert split_view is not None
    assert list(split_view["source"]) == ["fmp", "fmp", "yfinance", "yfinance", "yfinance", "yfinance", "yfinance"]
    np.testing.assert_allclose(split_view["close"].tolist(), [70.0, 70.5, 75.0, 76.0, 76.0, 77.0, 78.0])

    meta = increments.meta()
    assert meta["increment_seq"] == 1 and meta["writer_host"] == HOST and meta["last_run"]["stop_reason"] == "done"


def test_second_run_is_idempotent_and_overlap_only_verifies(root: Path):
    yahoo = FakeYahoo({"AAA": yf_daily(AAA_RAW)})
    _ingester(root, yahoo).run(timeframes=["1d"])
    increments = IncrementStore(root)
    sid = FmpStore(root).resolve("AAA").security_id  # type: ignore[union-attr]
    before = increments.path("1d", sid).read_bytes()
    # yfinance revised two stored bars inside the overlap window (5 days before the last stored bar, 2025-03-07)
    changed = dict(AAA_RAW, **{"2025-03-04": 160.0, "2025-03-06": 80.0, "2025-02-27": 150.0})
    report = _ingester(root, FakeYahoo({"AAA": yf_daily(changed)})).run(timeframes=["1d"])
    assert report.timeframes["1d"].new_bars == 0 and report.increment_seq == 1  # nothing written, seq unchanged
    assert report.timeframes["1d"].mismatched == {"AAA": 2}
    assert increments.path("1d", sid).read_bytes() == before  # stored bars are never overwritten


def test_hourly_gap_warning_and_730_day_window(root: Path):
    later = date(2027, 3, 1)
    yahoo = FakeYahoo({}, {"AAA": yf_hourly({"2026-12-01": 10.0})})
    report = _ingester(root, yahoo, today=later).run(timeframes=["1h"])
    assert report.timeframes["1h"].gap_warnings == {"AAA": (later - date(2025, 2, 3)).days}
    assert yahoo.calls[0][2] == datetime(2025, 3, 2)  # today - 729 days, not the (lost) last stored bar


def test_rate_limit_waits_once_then_stops(root: Path):
    yahoo = FakeYahoo({"AAA": yf_daily(AAA_RAW)})
    yahoo.fail = [YFRateLimitError(), YFRateLimitError()]
    sleeps: list[float] = []
    report = _ingester(root, yahoo, sleeps=sleeps).run(timeframes=["1d", "1h"])
    assert report.stop_reason == "rate_limited" and sleeps == [60.0] and "1h" not in report.timeframes
    yahoo.fail = [YFRateLimitError()]
    report = _ingester(root, yahoo).run(timeframes=["1d"])
    assert report.stop_reason == "done" and report.timeframes["1d"].new_bars == 5  # resumed


def test_symbol_selection_limit_and_lock(root: Path):
    yahoo = FakeYahoo({"AAA": yf_daily(AAA_RAW)})
    report = _ingester(root, yahoo).run(symbols=["AAA", "NOPE", "OLDC"], timeframes=["1d"])
    assert report.timeframes["1d"].candidates == 2  # AAA and NEWC (OLDC resolves to it); NOPE unknown
    report = _ingester(root, yahoo).run(timeframes=["1d"], limit=1)
    assert report.timeframes["1d"].candidates == 1
    with exclusive_lock(root / "locks" / "increments.lock"), pytest.raises(DumpLocked):
        _ingester(root, yahoo).run()
    with pytest.raises(NotTheWriter):
        IncrementIngester(root, PlanConfig(dump_id="v"), yahoo, today=TODAY, hostname=HOST).run()


def test_delisted_securities_are_skipped(tmp_path: Path):
    make_vintage(tmp_path, segment=True)  # XYZ splits into a delisted and an active segment
    yahoo = FakeYahoo({})
    report = _ingester(tmp_path, yahoo).run(symbols=["XYZ"], timeframes=["1d"])
    assert report.timeframes["1d"].candidates == 1  # resolved as of today: the active listing only
    store = FmpStore(tmp_path)
    assert [s.symbol for s in store.securities() if s.alive].count("XYZ") == 1
    report = _ingester(tmp_path, yahoo).run(timeframes=["1d"])
    assert report.timeframes["1d"].candidates == 4  # the delisted XYZ segment is never ingested


def test_store_handles_missing_files(tmp_path: Path):
    increments = IncrementStore(tmp_path)
    assert increments.bars(1, "X", "1d", None) is None and increments.bars(1, "X", "5m", None) is None
    assert increments.last_bar("1d", 1) is None and increments.meta() == {}
    splits, dividends = increments.events(1, "X")
    assert splits.empty and dividends.empty


def test_cli_ingest(root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory):
    config = tmp_path_factory.mktemp("cfg") / "fmp.yaml"
    config.write_text("fmp:\n  increments: {writer_host: elsewhere}\n", encoding="utf-8")
    args = ["ingest-increments", "--config", str(config), "--data-root", str(root), "--timeframes", "1d"]
    assert cli.main(args) == cli.EXIT_USAGE
    monkeypatch.setattr(ingest_module.socket, "gethostname", lambda: "elsewhere")
    monkeypatch.setattr(ingest_module, "default_fetch", lambda: FakeYahoo({"AAA": yf_daily(AAA_RAW)}))
    monkeypatch.setattr(ingest_module.time, "sleep", lambda s: None)
    assert cli.main(args) == cli.EXIT_OK
    assert cli.main([*args[:3], "--data-root", str(root / "nothing")]) == cli.EXIT_USAGE
