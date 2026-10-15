"""
Raw price files (spec §7.1, §7.3): ``prices/1d_raw/<bucket>/<security_id>.parquet`` and
``prices/<interval>_raw/<bucket>/<security_id>.parquet``, one security at a time.

**Field mapping by endpoint** (probe F3): ``historical-price-eod/non-split-adjusted`` names its *raw* prices
``adjOpen`` … ``adjClose``; they are stored as ``open`` … ``close``. Never map by field name alone.

Per symbol: newest successful response per request key; rows outside their request's ``from``/``to`` dropped
(F7); one bar per date (per timestamp for intraday), the newest response winning, conflicts counted; bars sorted.
For a segmented reused ticker each bar goes to the security whose range covers its date (clipping only assigns
bars; the archive keeps the full response). Bars no segment covers are dropped and counted.

**Intraday time** (spec §8.4): the ``date`` string is local time in ``build.intraday_tz`` (default
``America/New_York``, [VERIFY] by the paid probe), converted per bar with ``zoneinfo`` rules, never a fixed
offset. With ``build.intraday_label: end`` the interval is subtracted so ``ts_utc`` is always the bar **start**.
Local times that do not exist or are ambiguous (DST changes, never inside a US session) are dropped and counted.

``<bucket>`` is ``security_id % 256`` as two hex digits: stable, and about 120 files per directory for 30k
securities.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from src.data.fmp_pipeline.archive.lists import body_rows, iter_symbol_records
from src.data.fmp_pipeline.archive.reader import request_key
from src.data.fmp_pipeline.builders.flags import FLAG_NAMES, compute_flags
from src.data.fmp_pipeline.builders.identity import IdentityResult, SecurityRecord
from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings, write_table
from src.data.fmp_pipeline.builders.records import in_window, parse_date
from src.data.fmp_pipeline.planner.config import BuildConfig

DAILY_ENDPOINT = "historical-price-eod/non-split-adjusted"
DAILY_FIELDS = (("open", "adjOpen"), ("high", "adjHigh"), ("low", "adjLow"), ("close", "adjClose"))
INTRADAY_FIELDS = (("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"))
INTRADAY_MINUTES = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1hour": 60, "4hour": 240}
INTRADAY_DIR = {"1min": "1m", "5min": "5m", "15min": "15m", "30min": "30m", "1hour": "1h", "4hour": "4h"}
SOURCE = "fmp"
PRICE_SCHEMA_COMMON = [pa.field("open", pa.float64()), pa.field("high", pa.float64()), pa.field("low", pa.float64()),
                       pa.field("close", pa.float64()), pa.field("volume", pa.int64()), pa.field("flags", pa.int32()),
                       pa.field("source", pa.string()), pa.field("vintage", pa.string())]
DAILY_SCHEMA = pa.schema([pa.field("security_id", pa.int64()), pa.field("date", pa.date32()), *PRICE_SCHEMA_COMMON])
INTRADAY_SCHEMA = pa.schema([pa.field("security_id", pa.int64()), pa.field("ts_utc", pa.timestamp("us", tz="UTC")),
                             pa.field("date_et", pa.date32()), *PRICE_SCHEMA_COMMON])


def bucket_path(kind: str, security_id: int) -> str:
    """Relative path of a security's price file: ``prices/<kind>/<bucket>/<security_id>.parquet``."""
    return f"prices/{kind}/{security_id % 256:02x}/{security_id}.parquet"


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return np.nan
    return float(value)


@dataclass
class PriceStats:
    """Counters for one price kind, summed over securities."""

    securities: int = 0
    bars: int = 0
    responses: int = 0
    out_of_window: int = 0
    duplicates: int = 0
    conflicts: int = 0
    unparseable: int = 0
    unassigned: int = 0
    unresolved_symbols: int = 0
    flags: Counter[str] = field(default_factory=Counter)
    flagged_bars: int = 0


@dataclass
class Bars:
    """One symbol's bars, sorted; ``key`` is the date (daily) or the UTC bar start (intraday)."""

    key: np.ndarray
    dates: np.ndarray
    o: np.ndarray
    h: np.ndarray
    low: np.ndarray
    c: np.ndarray
    v: np.ndarray

    def __len__(self) -> int:
        return len(self.key)

    def take(self, mask: np.ndarray) -> "Bars":
        """Return the bars where ``mask`` is true."""
        return Bars(*(arr[mask] for arr in (self.key, self.dates, self.o, self.h, self.low, self.c, self.v)))


def _unique_rows(records: list[dict[str, Any]], key_of: Any, stats: PriceStats,
                 fields: tuple[tuple[str, str], ...]) -> dict[Any, tuple[float, ...]]:
    """``{key: (o, h, l, c, v)}``; newest response wins per key, rows outside their window dropped."""
    out: dict[Any, tuple[float, ...]] = {}
    for record in sorted(records, key=lambda r: (r["fetched_at"], request_key(r))):
        stats.responses += 1
        for row in body_rows(record.get("body")):
            key = key_of(row)
            if key is None:
                stats.unparseable += 1
                continue
            if not in_window(record, parse_date(row.get("date"))):
                stats.out_of_window += 1
                continue
            values = tuple(_number(row.get(src)) for _, src in fields) + (_number(row.get("volume")),)
            if key in out:
                stats.duplicates += 1
                if not np.allclose(out[key], values, equal_nan=True):
                    stats.conflicts += 1
            out[key] = values
    return out


def parse_daily(records: list[dict[str, Any]], stats: PriceStats) -> Bars:
    """Turn one symbol's daily responses into sorted bars."""
    rows = _unique_rows(records, lambda r: parse_date(r.get("date")), stats, DAILY_FIELDS)
    keys = sorted(rows)
    values = np.array([rows[k] for k in keys], dtype=float).reshape(len(keys), 5)
    dates = np.array(keys, dtype="datetime64[D]")
    return Bars(dates, dates, *values.T)


def parse_intraday(records: list[dict[str, Any]], stats: PriceStats, tz: str, label: str, minutes: int) -> Bars:
    """Turn one symbol's intraday responses into sorted bars keyed by the UTC bar start."""
    rows = _unique_rows(records, lambda r: r.get("date") if isinstance(r.get("date"), str) else None, stats,
                        INTRADAY_FIELDS)
    texts = sorted(rows)
    local = pd.DatetimeIndex(pd.to_datetime(pd.Series(texts, dtype=object), format="%Y-%m-%d %H:%M:%S",
                                            errors="coerce"))
    if label == "end":
        local = local - pd.Timedelta(minutes=minutes)
    zoned = local.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    ok = ~np.asarray(zoned.isna(), dtype=bool)
    stats.unparseable += int((~ok).sum())
    values = np.array([rows[t] for t in texts], dtype=float).reshape(len(texts), 5)[ok]
    zoned = zoned[ok]
    utc = zoned.tz_convert("UTC").tz_localize(None).values.astype("datetime64[us]")
    local_dates = zoned.tz_localize(None).values.astype("datetime64[D]")
    order = np.argsort(utc, kind="stable")
    keys, unique_idx = np.unique(utc[order], return_index=True)
    stats.duplicates += len(order) - len(unique_idx)
    pick = order[unique_idx]
    return Bars(keys, local_dates[pick], *values[pick].T)


def _volume(v: np.ndarray) -> pa.Array:
    mask = ~np.isfinite(v)
    return pa.array(np.where(mask, 0, np.round(v)).astype(np.int64), mask=mask)


def daily_table(security_id: int, bars: Bars, flags: np.ndarray, vintage: str, source: str = SOURCE) -> pa.Table:
    """Arrow table of ``prices_1d_raw`` for one security (``source`` is ``yfinance`` for increments)."""
    n = len(bars)
    return pa.table([pa.array(np.full(n, security_id, dtype=np.int64)), pa.array(bars.key, pa.date32()),
                     pa.array(bars.o), pa.array(bars.h), pa.array(bars.low), pa.array(bars.c), _volume(bars.v),
                     pa.array(flags, pa.int32()), pa.array([source] * n, pa.string()),
                     pa.array([vintage] * n, pa.string())], schema=DAILY_SCHEMA)


def intraday_table(security_id: int, bars: Bars, flags: np.ndarray, vintage: str, source: str = SOURCE) -> pa.Table:
    """Arrow table of ``prices_1h_raw`` (or another interval) for one security."""
    n = len(bars)
    return pa.table([pa.array(np.full(n, security_id, dtype=np.int64)),
                     pa.array(bars.key, pa.timestamp("us")).cast(pa.timestamp("us", tz="UTC")),
                     pa.array(bars.dates, pa.date32()), pa.array(bars.o), pa.array(bars.h), pa.array(bars.low),
                     pa.array(bars.c), _volume(bars.v), pa.array(flags, pa.int32()),
                     pa.array([source] * n, pa.string()), pa.array([vintage] * n, pa.string())],
                    schema=INTRADAY_SCHEMA)


@dataclass
class OutputFile:
    """A written price file."""

    path: str
    security_id: int
    rows: int


class PriceBuilder:
    """Builds the price files of one vintage."""

    def __init__(self, dump_dir: Path, out_dir: Path, identity: IdentityResult, config: BuildConfig, vintage: str,
                 settings: ParquetSettings, split_dates: dict[int, np.ndarray],
                 dividend_dates: dict[int, np.ndarray]) -> None:
        """
        Args:
            dump_dir: ``<raw_root>/<dump_id>``.
            out_dir: Vintage directory being built.
            identity: Securities and resolver.
            config: Build settings (flags, intraday time zone and label).
            vintage: Value of the ``vintage`` column (the dump id).
            settings: Parquet settings.
            split_dates: Split dates per ``security_id``.
            dividend_dates: Dividend ex-dates per ``security_id``.
        """
        self.dump_dir = dump_dir
        self.out_dir = out_dir
        self.identity = identity
        self.config = config
        self.vintage = vintage
        self.settings = settings
        self.split_dates = split_dates
        self.dividend_dates = dividend_dates
        self.files: list[OutputFile] = []
        self.jump_rows: list[tuple[str, int, str, float, float]] = []
        self.flagged_by_security: dict[int, tuple[int, int]] = {}

    def _segments(self, symbol: str) -> list[SecurityRecord]:
        return self.identity.by_symbol.get(symbol, [])

    def _split(self, symbol: str, bars: Bars, stats: PriceStats) -> list[tuple[SecurityRecord, Bars]]:
        segments = self._segments(symbol)
        if len(segments) <= 1:
            return [(segments[0], bars)] if segments else []
        parts = []
        assigned = np.zeros(len(bars), dtype=bool)
        for segment in segments:
            mask = np.ones(len(bars), dtype=bool)
            if segment.valid_from is not None:
                mask &= bars.dates >= np.datetime64(segment.valid_from, "D")
            if segment.valid_to is not None:
                mask &= bars.dates <= np.datetime64(segment.valid_to, "D")
            assigned |= mask
            parts.append((segment, bars.take(mask)))
        stats.unassigned += int((~assigned).sum())
        return parts

    def build_daily(self) -> PriceStats:
        """Write every security's daily file; fills ``first_price_date``/``last_price_date``."""
        stats = PriceStats()
        for symbol, records in iter_symbol_records(self.dump_dir, DAILY_ENDPOINT):
            self.build_daily_symbol(symbol, records, stats)
        return stats

    def build_daily_symbol(self, symbol: str, records: list[dict[str, Any]], stats: PriceStats) -> None:
        """Write the daily file(s) of one symbol (one per segment); also used by V-REPRO to rebuild a subset."""
        bars = parse_daily(records, stats)
        if not self._segments(symbol):
            stats.unresolved_symbols += 1
            return
        for security, part in self._split(symbol, bars, stats):
            if len(part) == 0:
                continue
            sid = security.security_id
            flags, jumps = compute_flags(part.dates, part.o, part.h, part.low, part.c, part.v, self.config.flags,
                                         daily=True, split_dates=self.split_dates.get(sid),
                                         dividend_dates=self.dividend_dates.get(sid))
            for jump in jumps:
                self.jump_rows.append((symbol, sid, str(part.dates[jump.index]), round(jump.implied_ratio, 6),
                                       jump.matched_ratio))
            security.first_price_date = part.dates[0].astype(date)
            security.last_price_date = part.dates[-1].astype(date)
            self._write(bucket_path("1d_raw", sid), daily_table(sid, part, flags, self.vintage), sid, stats, flags)
            self.flagged_by_security[sid] = (int((flags != 0).sum()), len(part))

    def build_intraday(self, interval: str) -> Optional[PriceStats]:
        """Write every security's intraday file of ``interval``; ``None`` if nothing of it is archived."""
        endpoint = f"historical-chart/{interval}"
        if interval not in INTRADAY_MINUTES or not (self.dump_dir / "historical-chart" / interval).is_dir():
            return None
        stats = PriceStats()
        kind = f"{INTRADAY_DIR[interval]}_raw"
        for symbol, records in iter_symbol_records(self.dump_dir, endpoint):
            bars = parse_intraday(records, stats, self.config.intraday_tz, self.config.intraday_label,
                                  INTRADAY_MINUTES[interval])
            if not self._segments(symbol):
                stats.unresolved_symbols += 1
                continue
            for security, part in self._split(symbol, bars, stats):
                if len(part) == 0:
                    continue
                sid = security.security_id
                flags, _ = compute_flags(part.dates, part.o, part.h, part.low, part.c, part.v, self.config.flags,
                                         daily=False)
                self._write(bucket_path(kind, sid), intraday_table(sid, part, flags, self.vintage), sid, stats, flags)
        return stats

    def _write(self, rel: str, table: pa.Table, sid: int, stats: PriceStats, flags: np.ndarray) -> None:
        rows = write_table(self.out_dir / rel, table, self.settings)
        self.files.append(OutputFile(rel, sid, rows))
        stats.securities += 1
        stats.bars += rows
        stats.flagged_bars += int((flags != 0).sum())
        for bit, name in FLAG_NAMES.items():
            stats.flags[name] += int(((flags & bit) != 0).sum())

    def missing_splits_csv(self) -> bytes:
        """``missing_splits_candidates.csv`` (spec §7.5): one row per ``JUMP_NO_SPLIT`` hit."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["symbol", "security_id", "date", "implied_ratio", "matched_ratio"])
        writer.writerows(sorted(self.jump_rows))
        return buffer.getvalue().encode("utf-8")
