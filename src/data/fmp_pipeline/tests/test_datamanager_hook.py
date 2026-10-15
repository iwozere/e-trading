"""Tests for the DataManager hook (spec §9.2) and the golden-output snapshot (plan WP-8)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from src.data import data_manager as dm_module
from src.data.data_manager import DataManager
from src.data.fmp_pipeline import cli
from src.data.fmp_pipeline.datamanager_adapter.golden import (
    Snapshot,
    cached_series,
    capture,
    compare,
    frame_hash,
    offline_fetch,
)

START, END = datetime(2025, 1, 1), datetime(2025, 2, 1)


def _frame(close: float) -> pd.DataFrame:
    index = pd.DatetimeIndex(["2025-01-02", "2025-01-03"], name="timestamp")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1}, index=index)


class StubAdapter:
    """Serves FMP-only symbols; records calls."""

    vintage = "v"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def get_ohlcv(self, symbol: str, timeframe: str, start: Any, end: Any, adjust: str, include_source: bool,
                  include_flags: bool, exclude_suspect: bool, as_of: Any) -> Optional[pd.DataFrame]:
        self.calls.append((symbol, adjust, include_source, include_flags, exclude_suspect, as_of))
        return _frame(1.0) if symbol.startswith("F") else None

    def get_ohlcv_batch(self, symbols: list[str], timeframe: str, start: Any, end: Any) -> dict[str, pd.DataFrame]:
        return {s: _frame(1.0) for s in symbols if s.startswith("F")}


def _manager(adapter: Any) -> DataManager:
    manager = DataManager.__new__(DataManager)  # no providers, no cache: only the hook is under test
    manager.fmp_adapter = adapter
    manager._get_ohlcv_legacy = lambda *args: _frame(2.0)  # type: ignore[method-assign]
    manager._get_ohlcv_batch_legacy = lambda symbols, *args: {s: _frame(2.0) for s in symbols}  # type: ignore
    return manager


def test_fmp_served_symbols_come_from_the_adapter():
    adapter = StubAdapter()
    manager = _manager(adapter)
    assert manager.get_ohlcv("FOO", "1d", START, END)["close"].iloc[0] == 1.0
    assert manager.get_ohlcv("BAR", "1d", START, END)["close"].iloc[0] == 2.0  # legacy path
    frame = manager.get_ohlcv("FOO", "1d", START, END, adjust="raw", include_flags=True, as_of=START.date())
    assert frame is not None and adapter.calls[-1] == ("FOO", "raw", False, True, False, START.date())
    assert manager.get_ohlcv("FOO", "1d", START, END, force_refresh=True)["close"].iloc[0] == 2.0
    assert len(adapter.calls) == 3  # force_refresh never asked the adapter


def test_fmp_only_options_raise_on_the_legacy_path():
    for manager in (_manager(StubAdapter()), _manager(None)):
        with pytest.raises(ValueError, match="need the FMP vintage"):
            manager.get_ohlcv("BAR", "1d", START, END, adjust="raw")
        with pytest.raises(ValueError, match="need the FMP vintage"):
            manager.get_ohlcv("BAR", "1d", START, END, exclude_suspect=True)
        legacy = manager.get_ohlcv("BAR", "1d", START, END, include_source=True)
        assert list(legacy["source"]) == ["legacy", "legacy"]
    with pytest.raises(ValueError, match="start_date"):
        _manager(None).get_ohlcv("BAR", "1d", END, START)


def test_batch_merges_in_request_order():
    manager = _manager(StubAdapter())
    result = manager.get_ohlcv_batch(["BAR", "FOO", "BAZ", "FOO"], "1d", START, END)
    assert list(result) == ["BAR", "FOO", "BAZ"]
    assert result["FOO"]["close"].iloc[0] == 1.0 and result["BAR"]["close"].iloc[0] == 2.0
    only_fmp = manager.get_ohlcv_batch(["FOO"], "1d", START, END)
    assert list(only_fmp) == ["FOO"]
    assert manager.get_ohlcv_batch(["FOO"], "1d", START, END, force_refresh=True)["FOO"]["close"].iloc[0] == 2.0
    assert _manager(None).get_ohlcv_batch(["FOO"], "1d", START, END)["FOO"]["close"].iloc[0] == 2.0


def test_adapter_loader_never_raises(monkeypatch: pytest.MonkeyPatch):
    import src.data.fmp_pipeline.datamanager_adapter.factory as factory

    def boom() -> None:
        raise RuntimeError("broken install")

    monkeypatch.setattr(factory, "load_adapter", boom)
    assert dm_module._load_fmp_adapter() is None
    monkeypatch.setattr(factory, "load_adapter", lambda: "adapter")
    assert dm_module._load_fmp_adapter() == "adapter"


def test_frame_hash_sees_every_observable_change():
    base = _frame(1.0)
    variants = [base.assign(close=1.5), base.astype({"volume": float}), base.rename_axis("time"),
                base[["open", "high", "low", "volume", "close"]], base.iloc[:1]]
    hashes = {frame_hash(base)} | {frame_hash(v) for v in variants}
    assert len(hashes) == 1 + len(variants) and frame_hash(base) == frame_hash(_frame(1.0))


def test_capture_compare_and_snapshot_file(tmp_path: Path):
    ohlcv = tmp_path / "ohlcv"
    for symbol, timeframe in (("AAA", "1d"), ("AAA", "1h"), ("BBB", "1d")):
        (ohlcv / symbol / timeframe).mkdir(parents=True)
        (ohlcv / symbol / timeframe / "2025.csv.gz").write_bytes(b"")
    (ohlcv / "EMPTY" / "1d").mkdir(parents=True)
    pairs = cached_series(ohlcv)
    assert pairs == [("AAA", "1d"), ("AAA", "1h"), ("BBB", "1d")] and cached_series(tmp_path / "none") == []

    def fetch(symbol: str, timeframe: str) -> pd.DataFrame:
        if symbol == "BBB":
            raise RuntimeError("no cache")
        return _frame(1.0 if timeframe == "1d" else 3.0)

    baseline = capture(pairs, fetch, {"note": "x"})
    assert baseline.entries["BBB/1d"] == {"error": "RuntimeError: no cache"}
    path = tmp_path / "golden" / "hashes.json"
    baseline.save(path)
    loaded = Snapshot.load(path)
    assert loaded.entries == baseline.entries and loaded.meta == {"note": "x"}
    assert compare(loaded, capture(pairs, fetch)).clean

    def changed(symbol: str, timeframe: str) -> pd.DataFrame:
        return _frame(9.0) if symbol == "AAA" and timeframe == "1h" else fetch(symbol, timeframe)

    result = compare(loaded, capture(pairs[1:] + [("CCC", "1d")], lambda s, t: changed(s, t) if s != "CCC"
                                     else _frame(1.0)))
    assert (result.changed, result.missing, result.added, result.same) == (["AAA/1h"], ["AAA/1d"], ["CCC/1d"], 1)
    path.write_text('{"version": 99}', encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        Snapshot.load(path)


def test_offline_fetch_disables_providers():
    class Selector:
        def get_provider_with_failover(self, *args: Any) -> list[str]:
            return ["yahoo"]

    class Manager:
        provider_selector = Selector()

        def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
            assert self.provider_selector.get_provider_with_failover(symbol, timeframe) == []
            return _frame(1.0)

    assert len(offline_fetch(Manager())("AAA", "1d")) == 2


def test_cli_golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ohlcv = tmp_path / "cache" / "ohlcv" / "AAA" / "1d"
    ohlcv.mkdir(parents=True)
    (ohlcv / "2025.csv.gz").write_bytes(b"")
    closes = {"value": 1.0}

    class FakeManager:
        def __init__(self, cache_dir: str) -> None:
            self.provider_selector = type("S", (), {})()
            self.fmp_adapter = None

        def get_ohlcv(self, *args: Any) -> pd.DataFrame:
            return _frame(closes["value"])

    monkeypatch.setattr(dm_module, "DataManager", FakeManager)
    args = ["golden", "compare", "--cache-dir", str(tmp_path / "cache"), "--data-root", str(tmp_path)]
    assert cli.main(args) == cli.EXIT_USAGE  # no baseline yet
    args[1] = "capture"
    assert cli.main(args + ["--symbols", "AAA"]) == cli.EXIT_OK
    args[1] = "compare"
    assert cli.main(args) == cli.EXIT_OK
    closes["value"] = 2.0
    assert cli.main(args) == cli.EXIT_STOPPED
    assert (tmp_path / "reports" / "golden" / "ohlcv_hashes_latest.json").exists()
