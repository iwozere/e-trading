"""Tests for raw price files (spec §7.3, §8.4): field mapping by endpoint, dedup, windows, DST-safe time, clipping."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.builders.identity import IdentityResult, SecurityRecord
from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings
from src.data.fmp_pipeline.builders.prices import (
    DAILY_ENDPOINT,
    DAILY_SCHEMA,
    INTRADAY_SCHEMA,
    PriceBuilder,
    PriceStats,
    bucket_path,
    parse_daily,
    parse_intraday,
)
from src.data.fmp_pipeline.planner.config import BuildConfig
from src.data.fmp_pipeline.tests.archive_fixtures import daily_rows, put


def _record(body: Any, params: dict[str, Any], fetched_at: str = "2026-10-14T10:00:00+00:00") -> dict[str, Any]:
    return {"request": {"endpoint": DAILY_ENDPOINT, "params": params}, "fetched_at": fetched_at, "body": body,
            "sha256": "x"}


def test_daily_maps_adj_fields_dedups_and_drops_out_of_window():
    first = _record(daily_rows("A", {"2025-01-02": 10.0, "2025-01-03": 11.0, "2024-12-31": 9.0}),
                    {"symbol": "A", "from": "2025-01-01", "to": "2025-12-31"})
    later = _record(daily_rows("A", {"2025-01-03": 12.0}) + [{"date": "junk"}],
                    {"symbol": "A", "from": "2025-01-03", "to": "2025-01-03"}, "2026-10-15T10:00:00+00:00")
    stats = PriceStats()
    bars = parse_daily([later, first], stats)
    assert list(bars.key.astype(str)) == ["2025-01-02", "2025-01-03"]
    assert list(bars.c) == [10.0, 12.0] and bars.h[0] == 10.0 * 1.01  # raw prices from adj* fields (probe F3)
    assert (stats.out_of_window, stats.duplicates, stats.conflicts, stats.unparseable) == (1, 1, 1, 1)
    empty = parse_daily([], PriceStats())
    assert len(empty) == 0


def test_intraday_converts_local_time_per_bar_across_dst():
    rows = [{"date": "2025-03-07 09:30:00", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 5},
            {"date": "2025-03-10 09:30:00", "open": 2, "high": 2, "low": 2, "close": 2, "volume": 5},
            {"date": "2025-03-09 02:30:00", "open": 3, "high": 3, "low": 3, "close": 3, "volume": 5},  # nonexistent
            {"date": "2025-11-02 01:30:00", "open": 4, "high": 4, "low": 4, "close": 4, "volume": 5},  # ambiguous
            {"date": "garbage", "open": 5, "high": 5, "low": 5, "close": 5, "volume": 5}]
    stats = PriceStats()
    bars = parse_intraday([_record(rows, {"symbol": "A"})], stats, "America/New_York", "start", 60)
    stamps = [str(t) for t in bars.key.astype("datetime64[m]")]
    assert stamps == ["2025-03-07T14:30", "2025-03-10T13:30"]  # EST then EDT: never a fixed offset
    assert list(bars.dates.astype(str)) == ["2025-03-07", "2025-03-10"] and stats.unparseable == 3
    end = parse_intraday([_record(rows[:1], {"symbol": "A"})], PriceStats(), "America/New_York", "end", 60)
    assert str(end.key[0].astype("datetime64[m]")) == "2025-03-07T13:30"  # end label -> start of the bar


def _security(sid: int, symbol: str, valid_from: Any = None, valid_to: Any = None) -> SecurityRecord:
    return SecurityRecord(key=f"{symbol}{sid}", symbol=symbol, security_id=sid, valid_from=valid_from,
                          valid_to=valid_to)


def test_builder_writes_files_clips_segments_and_reports_jumps(tmp_path: Path):
    archive = RawArchive(tmp_path / "raw", "d")
    closes = {f"2025-01-{d:02d}": 100.0 for d in range(2, 30)}
    closes.update({f"2025-02-{d:02d}": 25.0 for d in range(3, 10)})  # unexplained 4:1 drop
    put(archive, DAILY_ENDPOINT, "AAA", daily_rows("AAA", closes), {"from": "2025-01-01", "to": "2025-12-31"})
    put(archive, DAILY_ENDPOINT, "XYZ", daily_rows("XYZ", {"2008-06-27": 5.0, "2008-06-30": 5.1, "2010-01-04": 9.0}),
        {"from": "2005-01-01", "to": "2010-12-31"})
    put(archive, DAILY_ENDPOINT, "GONE", daily_rows("GONE", {"2025-01-02": 1.0}))
    put(archive, "historical-chart/1hour", "AAA",
        [{"date": "2025-01-02 09:30:00", "open": 1, "high": 1.2, "low": 0.9, "close": 1.1, "volume": 7}])
    identity = IdentityResult([_security(1, "AAA"), _security(2, "XYZ", None, date(2008, 6, 27)),
                               _security(3, "XYZ", date(2008, 7, 1), None)])
    for s in identity.securities:
        identity.by_symbol.setdefault(s.symbol, []).append(s)
    builder = PriceBuilder(archive.dump_dir, tmp_path / "out", identity, BuildConfig(), "d", ParquetSettings(),
                           {}, {})
    daily = builder.build_daily()
    assert (daily.securities, daily.unresolved_symbols, daily.unassigned) == (3, 1, 1)  # 2008-06-30 in no segment
    aaa = pq.read_table(tmp_path / "out" / bucket_path("1d_raw", 1))
    assert aaa.schema.equals(DAILY_SCHEMA) and aaa.num_rows == len(closes)
    assert identity.securities[0].first_price_date == date(2025, 1, 2)
    assert identity.securities[0].last_price_date == date(2025, 2, 9)
    assert [r[2] for r in builder.jump_rows] == ["2025-02-03"] and daily.flags["JUMP_NO_SPLIT"] == 1
    assert builder.missing_splits_csv().decode().splitlines() == [
        "symbol,security_id,date,implied_ratio,matched_ratio", "AAA,1,2025-02-03,4.0,4.0"]
    assert pq.read_table(tmp_path / "out" / bucket_path("1d_raw", 3)).column("close").to_pylist() == [9.0]
    hourly = builder.build_intraday("1hour")
    assert hourly is not None and hourly.bars == 1
    table = pq.read_table(tmp_path / "out" / bucket_path("1h_raw", 1))
    assert table.schema.equals(INTRADAY_SCHEMA)
    assert table.column("ts_utc").to_pylist() == [datetime(2025, 1, 2, 14, 30, tzinfo=timezone.utc)]
    assert builder.build_intraday("5min") is None and builder.build_intraday("2hour") is None
    assert bucket_path("1d_raw", 300) == "prices/1d_raw/2c/300.parquet"
    assert np.all(aaa.column("security_id").to_numpy() == 1)


def test_missing_volume_is_null_not_zero(tmp_path: Path):
    archive = RawArchive(tmp_path / "raw", "d")
    rows = daily_rows("AAA", {"2025-01-02": 10.0})
    rows[0]["volume"] = None
    put(archive, DAILY_ENDPOINT, "AAA", rows)
    identity = IdentityResult([_security(1, "AAA")], by_symbol={"AAA": [_security(1, "AAA")]})
    PriceBuilder(archive.dump_dir, tmp_path, identity, BuildConfig(), "d", ParquetSettings(), {}, {}).build_daily()
    assert pq.read_table(tmp_path / bucket_path("1d_raw", 1)).column("volume").to_pylist() == [None]
