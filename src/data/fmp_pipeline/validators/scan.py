"""
One pass over every price file of the vintage (daily and every intraday interval), collecting what V-SCHEMA, V-DUP,
V-ORDER, V-OHLC, V-CAL, V-FLAGS, V-IDENT and V-INTRADAY-ALIGN need, so each file is read once instead of once per check.

Expected trading days and session times come from the NYSE calendar (``pandas_market_calendars``, already a
project dependency): weekends, holidays and early closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.flags import FLAG_NAMES
from src.data.fmp_pipeline.builders.prices import DAILY_SCHEMA, INTRADAY_DIR, INTRADAY_SCHEMA
from src.data.fmp_pipeline.builders.validate import SECURITIES
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

CALENDAR_START, CALENDAR_END = "1985-01-01", "2035-12-31"
SESSION_OPEN_MINUTES = 9 * 60 + 30
#: Intraday timeframe (vintage folder prefix) -> bar length in minutes.
INTRADAY_MINUTES = {folder: int(interval[:-4]) * 60 if interval.endswith("hour") else int(interval[:-3])
                    for interval, folder in INTRADAY_DIR.items()}


@dataclass(frozen=True)
class Calendar:
    """NYSE sessions: dates, and the local close time (minutes after midnight) of each, as a dict and as an array
    aligned with ``sessions``."""

    sessions: np.ndarray
    close_minutes: dict[date, int]
    closes: np.ndarray

    def expected_between(self, first: np.datetime64, last: np.datetime64) -> np.ndarray:
        """Session dates from ``first`` to ``last`` inclusive."""
        lo, hi = np.searchsorted(self.sessions, [first, last], side="left")
        if hi < len(self.sessions) and self.sessions[hi] == last:
            hi += 1
        return self.sessions[lo:hi]


@lru_cache(maxsize=1)
def nyse_calendar() -> Calendar:
    """The NYSE calendar (cached; building it takes about a second)."""
    import pandas_market_calendars as mcal

    schedule = mcal.get_calendar("NYSE").schedule(CALENDAR_START, CALENDAR_END)
    closes = pd.DatetimeIndex(schedule["market_close"]).tz_convert("America/New_York")
    sessions = pd.DatetimeIndex(schedule.index).to_numpy(dtype="datetime64[D]")
    close_array = np.asarray(closes.hour * 60 + closes.minute, dtype=np.int64)
    minutes = {pd.Timestamp(s).date(): int(c) for s, c in zip(sessions, close_array)}
    return Calendar(sessions, minutes, close_array)


@dataclass
class SeriesStats:
    """What the scan learned about one price file."""

    security_id: int
    symbol: str
    tier: str
    kind: str
    rows: int = 0
    schema_ok: bool = True
    null_keys: int = 0
    foreign_rows: int = 0
    duplicates: int = 0
    order_violations: int = 0
    ohlc_violations: int = 0
    flagged: int = 0
    flag_counts: dict[str, int] = field(default_factory=dict)
    first: Optional[date] = None
    last: Optional[date] = None
    expected_days: int = 0
    missing_days: int = 0
    max_gap_days: int = 0
    max_gap_at: Optional[date] = None
    days: int = 0
    bad_bar_count_days: int = 0
    bad_first_bar_days: int = 0
    outside_session_bars: int = 0


@dataclass
class PriceScan:
    """Stats per price file plus the securities table; ``intraday`` is keyed by timeframe, then security."""

    daily: dict[int, SeriesStats] = field(default_factory=dict)
    intraday: dict[str, dict[int, SeriesStats]] = field(default_factory=dict)

    def intraday_series(self) -> list[SeriesStats]:
        """Every intraday series, all timeframes."""
        return [stats for by_sid in self.intraday.values() for stats in by_sid.values()]
    securities: pd.DataFrame = field(default_factory=pd.DataFrame)


def _ohlc_violations(df: pd.DataFrame) -> int:
    o, h, low, c, v = (df[col].to_numpy(dtype=float) for col in ("open", "high", "low", "close", "volume"))
    with np.errstate(invalid="ignore"):
        bad = (np.isnan(np.vstack([o, h, low, c])).any(axis=0) | (np.vstack([o, h, low, c]) <= 0).any(axis=0)
               | (low > np.fmin(o, c)) | (h < np.fmax(o, c)) | (v < 0))
    return int(bad.sum())


def _common(stats: SeriesStats, path: Path, schema_expected: object, key: str) -> Optional[pd.DataFrame]:
    try:
        table = pq.read_table(path)
    except (OSError, ValueError) as exc:
        _logger.warning("Unreadable price file %s: %s", path, exc)
        stats.schema_ok = False
        return None
    stats.schema_ok = table.schema.equals(schema_expected)
    df = table.to_pandas()
    stats.rows = len(df)
    stats.null_keys = int(df[key].isna().sum() + df["security_id"].isna().sum())
    stats.foreign_rows = int((df["security_id"] != stats.security_id).sum())
    keys = df[key]
    stats.duplicates = int(keys.duplicated().sum())
    values = keys.to_numpy()
    stats.order_violations = int((values[1:] <= values[:-1]).sum()) if len(values) > 1 else 0
    stats.ohlc_violations = _ohlc_violations(df)
    flags = df["flags"].to_numpy(dtype=np.int64)
    stats.flagged = int((flags != 0).sum())
    stats.flag_counts = {name: int(((flags & bit) != 0).sum()) for bit, name in FLAG_NAMES.items()}
    return df


def _daily(stats: SeriesStats, path: Path, calendar: Calendar) -> None:
    df = _common(stats, path, DAILY_SCHEMA, "date")
    if df is None or df.empty:
        return
    dates = np.unique(pd.to_datetime(df["date"]).to_numpy(dtype="datetime64[D]"))
    stats.first, stats.last = pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()
    expected = calendar.expected_between(dates[0], dates[-1])
    stats.expected_days = len(expected)
    stats.missing_days = int(len(np.setdiff1d(expected, dates, assume_unique=True)))
    if len(dates) > 1:
        gaps = np.diff(dates).astype(int)
        i = int(np.argmax(gaps))
        stats.max_gap_days, stats.max_gap_at = int(gaps[i]), pd.Timestamp(dates[i]).date()


def _intraday(stats: SeriesStats, path: Path, calendar: Calendar, tz: str, interval_minutes: int) -> None:
    """
    Session alignment per exchange-local day, vectorised (a 5-minute file holds ~6k days).

    A regular day expects ceil(session minutes / interval) bars from 09:30 ET; days that are not NYSE sessions only
    count their bars as outside the session.
    """
    df = _common(stats, path, INTRADAY_SCHEMA, "ts_utc")
    if df is None or df.empty:
        return
    local = pd.DatetimeIndex(pd.to_datetime(df["ts_utc"], utc=True)).tz_convert(tz)
    minutes = np.asarray(local.hour * 60 + local.minute, dtype=np.int64)
    days = local.tz_localize(None).to_numpy(dtype="datetime64[D]")
    unique_days, inverse = np.unique(days, return_inverse=True)
    stats.first, stats.last = pd.Timestamp(unique_days[0]).date(), pd.Timestamp(unique_days[-1]).date()
    stats.days = len(unique_days)
    position = np.minimum(np.searchsorted(calendar.sessions, unique_days), len(calendar.sessions) - 1)
    is_session = calendar.sessions[position] == unique_days
    day_close = np.where(is_session, calendar.closes[position], SESSION_OPEN_MINUTES)
    inside = (minutes >= SESSION_OPEN_MINUTES) & (minutes < day_close[inverse])
    stats.outside_session_bars = int(len(minutes) - inside.sum())
    inside_count = np.bincount(inverse, weights=inside, minlength=len(unique_days))
    opened = np.bincount(inverse, weights=inside & (minutes == SESSION_OPEN_MINUTES), minlength=len(unique_days)) > 0
    expected = -(-(day_close - SESSION_OPEN_MINUTES) // interval_minutes)
    stats.bad_bar_count_days = int((is_session & (inside_count != expected)).sum())
    stats.bad_first_bar_days = int((is_session & ~opened).sum())


def scan_prices(store: FmpStore) -> PriceScan:
    """
    Read every daily and intraday price file of the vintage once.

    Args:
        store: The vintage.

    Returns:
        Stats per security and kind, and the securities table.
    """
    calendar = nyse_calendar()
    securities = pq.read_table(store.dir / SECURITIES).to_pandas()
    scan = PriceScan(securities=securities)
    tiers = dict(zip(securities["security_id"], securities["tier"]))
    timeframes = [tf for tf in INTRADAY_MINUTES if (store.dir / "prices" / f"{tf}_raw").is_dir()]
    for count, info in enumerate(store.securities(), 1):
        sid = info.security_id
        tier = str(tiers.get(sid, "E"))
        daily_path = store.price_path(sid, "1d")
        if daily_path is not None:
            stats = SeriesStats(sid, info.symbol, tier, "1d")
            _daily(stats, daily_path, calendar)
            scan.daily[sid] = stats
        for timeframe in timeframes:
            path = store.price_path(sid, timeframe)
            if path is not None:
                stats = SeriesStats(sid, info.symbol, tier, timeframe)
                _intraday(stats, path, calendar, store.intraday_tz, INTRADAY_MINUTES[timeframe])
                scan.intraday.setdefault(timeframe, {})[sid] = stats
        if count % 2000 == 0:
            _logger.info("Price scan: %d securities read", count)
    _logger.info("Price scan: %d daily files; intraday %s", len(scan.daily),
                 {tf: len(by_sid) for tf, by_sid in scan.intraday.items()})
    return scan
