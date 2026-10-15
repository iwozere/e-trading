"""Tests for the DataManager adapter (spec §9): symbol resolution, adjustment modes, contract, caching, batch."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.data.fmp_pipeline.adjust.adjust import adjust
from src.data.fmp_pipeline.datamanager_adapter.adapter import FmpOhlcvAdapter, adjust_mode, empty_frame
from src.data.fmp_pipeline.datamanager_adapter.factory import load_adapter
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import AmbiguousSymbol, FmpStore
from src.data.fmp_pipeline.tests.vintage_fixtures import DAYS, DIVIDEND_DAY, SPLIT_DAY, make_vintage


@pytest.fixture(scope="module")
def vintage_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("vintage")
    make_vintage(root)
    return root


@pytest.fixture
def adapter(vintage_root: Path) -> FmpOhlcvAdapter:
    return FmpOhlcvAdapter(FmpStore(vintage_root))


START, END = datetime(2025, 1, 1), datetime(2025, 3, 1)


def test_store_resolves_symbols(vintage_root: Path):
    store = FmpStore(vintage_root)
    aaa = store.resolve("AAA")
    assert aaa is not None and aaa.first_price_date == date(2025, 1, 2) and store.vintage == "v"
    brk = store.resolve("BRK.B")
    assert brk is not None and store.resolve("BRK-B") == brk  # yf_symbol and separator swap
    newc = store.resolve("NEWC")
    assert store.resolve("OLDC") == newc and store.resolve("OLDC", date(2025, 1, 6)) == newc  # through history
    assert store.resolve("OLDC", date(2025, 6, 2)) is None and store.resolve("NOPE") is None
    assert store.security(aaa.security_id) == aaa
    assert store.price_path(aaa.security_id, "4h") is None and store.price_path(aaa.security_id, "1w") is None


def test_store_without_vintage(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no active FMP vintage"):
        FmpStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="has no"):
        FmpStore(tmp_path, "missing")


def test_reused_ticker_is_ambiguous_without_as_of(tmp_path: Path):
    make_vintage(tmp_path, segment=True)
    store = FmpStore(tmp_path)
    with pytest.raises(AmbiguousSymbol, match="pass as_of"):
        store.resolve("XYZ")
    old, new = store.resolve("XYZ", date(2024, 6, 3)), store.resolve("XYZ", date(2025, 1, 2))
    assert old is not None and new is not None and old != new
    adapter = FmpOhlcvAdapter(store)
    frame = adapter.get_ohlcv("XYZ", "1d", datetime(2020, 1, 1), END, as_of=date(2024, 6, 3))
    assert frame is not None and list(frame["close"]) == [5.0]  # bars clipped to the old listing


def test_legacy_is_total_and_matches_adjust(adapter: FmpOhlcvAdapter):
    legacy = adapter.get_ohlcv("AAA", "1d", START, END)
    total = adapter.get_ohlcv("AAA", "1d", START, END, "total")
    raw = adapter.get_ohlcv("AAA", "1d", START, END, "raw")
    split = adapter.get_ohlcv("AAA", "1d", START, END, "split")
    assert legacy is not None and total is not None and raw is not None and split is not None
    pd.testing.assert_frame_equal(legacy, total)
    assert raw["close"].iloc[0] == 400.0 and split["close"].iloc[0] == 100.0  # 4:1 split
    splits = pd.DataFrame({"date": [SPLIT_DAY], "numerator": [4], "denominator": [1]})
    dividends = pd.DataFrame({"date": [DIVIDEND_DAY], "amount": [2.0]})
    expected = adjust(raw.astype({"volume": float}), splits, dividends, "total")
    np.testing.assert_allclose(total["close"], expected["close"])
    assert total["close"].iloc[0] < split["close"].iloc[0]  # dividend drag before the ex-date
    assert split["volume"].iloc[0] == 4000  # volume x ratio, kept int64


def test_output_contract_and_range(adapter: FmpOhlcvAdapter):
    frame = adapter.get_ohlcv("AAA", "1d", datetime(2025, 1, 3), datetime(2025, 1, 6))
    assert frame is not None
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    index = pd.DatetimeIndex(frame.index)
    assert index.name == "timestamp" and index.tz is None and str(index.dtype) == "datetime64[us]"
    assert list(index.strftime("%Y-%m-%d")) == ["2025-01-03", "2025-01-06"]  # inclusive at both ends
    assert str(frame["volume"].dtype) == "int64" and str(frame["close"].dtype) == "float64"
    aware = adapter.get_ohlcv("AAA", "1d", pd.Timestamp("2025-01-03", tz="UTC"), pd.Timestamp("2025-01-06", tz="UTC"))
    assert aware is not None and len(aware) == 2
    empty = adapter.get_ohlcv("AAA", "1d", datetime(2030, 1, 1), datetime(2030, 2, 1))
    assert empty is not None and empty.empty  # FMP knows the symbol: no fall-through to legacy
    assert list(empty_frame().columns) == list(frame.columns)


def test_opt_in_columns_and_exclude_suspect(adapter: FmpOhlcvAdapter):
    frame = adapter.get_ohlcv("AAA", "1d", START, END, include_source=True, include_flags=True)
    assert frame is not None and list(frame.columns)[-2:] == ["source", "flags"]
    assert set(frame["source"]) == {"fmp"}
    flagged = int((frame["flags"] != 0).sum())
    clean = adapter.get_ohlcv("AAA", "1d", START, END, exclude_suspect=True)
    assert clean is not None and len(clean) == len(frame) - flagged


def test_not_served_returns_none(adapter: FmpOhlcvAdapter):
    assert adapter.get_ohlcv("NOPE", "1d", START, END) is None
    assert adapter.get_ohlcv("AAA", "5m", START, END) is None  # timeframe not served
    assert adapter.get_ohlcv("BRK.B", "1h", START, END) is None  # no 1h file in the vintage
    with pytest.raises(ValueError, match="adjust must be"):
        adapter.get_ohlcv("AAA", "1d", START, END, "weird")
    assert adjust_mode("legacy") == "total" and adjust_mode("raw") == "raw"


def test_intraday_uses_exchange_local_dates_and_utc_index(adapter: FmpOhlcvAdapter):
    raw = adapter.get_ohlcv("AAA", "1h", START, END, "raw")
    split = adapter.get_ohlcv("AAA", "1h", START, END, "split")
    assert raw is not None and split is not None and len(raw) == 14
    assert raw.index[0] == pd.Timestamp("2025-01-31 14:30")  # 09:30 EST as tz-naive UTC, like the legacy files
    assert raw["close"].iloc[0] == 400.0 and split["close"].iloc[0] == 100.0 and split["close"].iloc[-1] == 100.0


def test_adjusted_cache_and_increments(vintage_root: Path):
    class Increments:
        def __init__(self) -> None:
            self.dividend = pd.DataFrame(columns=["date", "amount"])

        def bars(self, security_id: int, symbol: str, timeframe: str, after: Optional[date]) -> pd.DataFrame:
            index = pd.DatetimeIndex(["2025-02-28", "2025-03-03"], name="timestamp")  # first one overlaps FMP
            return pd.DataFrame({"open": [1.0, 150.0], "high": [1.0, 150.0], "low": [1.0, 150.0],
                                 "close": [1.0, 150.0], "volume": [1, 5], "flags": [0, 0],
                                 "source": ["yfinance", "yfinance"]}, index=index)

        def events(self, security_id: int, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
            return pd.DataFrame(columns=["date", "numerator", "denominator"]), self.dividend

    increments = Increments()
    adapter = FmpOhlcvAdapter(FmpStore(vintage_root), increments, cache_entries=2)
    frame = adapter.get_ohlcv("AAA", "1d", START, datetime(2025, 3, 31), "raw", include_source=True)
    assert frame is not None and frame.index[-1] == pd.Timestamp("2025-03-03")
    assert frame["source"].iloc[-1] == "yfinance" and frame["close"].iloc[-2] != 1.0  # FMP bar kept on overlap
    before = adapter.get_ohlcv("AAA", "1d", START, END, "total")
    assert len(adapter._cache) == 2
    increments.dividend = pd.DataFrame({"date": ["2025-03-03", DIVIDEND_DAY], "amount": [1.0, 99.0]})
    after = adapter.get_ohlcv("AAA", "1d", START, END, "total")
    assert before is not None and after is not None
    assert after["close"].iloc[0] < before["close"].iloc[0]  # new dividend: new cache key, new factors
    c_prev = 100.0 + len(DAYS) - 1  # raw close of 2025-02-28, the last bar before the new ex-date
    np.testing.assert_allclose(after["close"].iloc[-1] / before["close"].iloc[-1], 1 - 1.0 / c_prev, rtol=1e-9)
    # the increment's 99.0 on the FMP ex-date is ignored (FMP wins), so the first bar moves by the same factor
    np.testing.assert_allclose(after["close"].iloc[0] / before["close"].iloc[0], 1 - 1.0 / c_prev, rtol=1e-9)
    assert len(adapter._cache) == 2  # LRU bound


def test_batch_matches_single_reads(adapter: FmpOhlcvAdapter):
    batch = adapter.get_ohlcv_batch(["AAA", "NOPE", "BRK-B"], "1d", datetime(2025, 1, 15), datetime(2025, 2, 20))
    assert list(batch) == ["AAA", "BRK-B"]
    for symbol, frame in batch.items():
        single = adapter.get_ohlcv(symbol, "1d", datetime(2025, 1, 15), datetime(2025, 2, 20))
        assert single is not None
        pd.testing.assert_frame_equal(frame, single)
    hourly = adapter.get_ohlcv_batch(["AAA", "BRK.B"], "1h", START, END)
    assert list(hourly) == ["AAA"]


def test_factory(tmp_path: Path, vintage_root: Path, caplog: pytest.LogCaptureFixture):
    config = tmp_path / "fmp.yaml"
    config.write_text("fmp:\n  adapter: {enabled: false}\n", encoding="utf-8")
    assert load_adapter(config) is None
    config.write_text(f"fmp:\n  adapter: {{enabled: true, data_root: '{tmp_path.as_posix()}'}}\n", encoding="utf-8")
    assert load_adapter(config) is None  # enabled, no vintage: logged, not raised
    config.write_text(f"fmp:\n  adapter: {{enabled: true, data_root: '{vintage_root.as_posix()}', "
                      f"timeframes: [1d]}}\n", encoding="utf-8")
    adapter = load_adapter(config)
    assert adapter is not None and adapter.timeframes == ("1d",) and adapter.vintage == "v"
    config.write_text("fmp:\n  adapter: {policy: legacy_first}\n", encoding="utf-8")
    assert load_adapter(config) is None
    assert load_adapter(tmp_path / "missing.yaml") is None
