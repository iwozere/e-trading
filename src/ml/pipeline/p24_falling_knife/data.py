"""
P24 Falling Knife — price data (spec §3).

Backtest bars come from the read-only FMP vintage: raw daily bars plus split/dividend events, split-adjusted at read
time with the shared ``adjust()`` (signals use split-adjusted prices, spec §3.2). The raw open/close/volume are kept
next to the adjusted columns because the price floor and the cost tier depend on the price actually traded. Bars the
FMP builder flagged as bad prints are dropped; ``JUMP_NO_SPLIT`` bars are kept and marked ``jump_no_split`` (a price jump
without a split is usually a missing corporate action, spec §3.3).
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.data.fmp_pipeline.adjust.adjust import adjust
from src.data.fmp_pipeline.builders.flags import JUMP_NO_SPLIT, OHLC_VIOLATION, SPIKE_REVERTED
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.ml.pipeline.p24_falling_knife.config import DAILY, WEEKLY, UniverseParams
from src.ml.pipeline.p24_falling_knife.universe import eligibility_mask, median_dollar_volume
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
_SKIP_FLAGS = OHLC_VIOLATION | SPIKE_REVERTED


def load_daily_batch(store: FmpStore, security_ids: Iterable[int], start: date, end: date) -> Dict[int, pd.DataFrame]:
    """
    Read and split-adjust the daily bars of many securities.

    Args:
        store: Open FMP vintage.
        security_ids: Securities to read.
        start: First date (inclusive).
        end: Last date (inclusive).

    Returns:
        ``{security_id: frame}`` with ``open … volume`` (split-adjusted), ``open_raw``, ``close_raw``, ``volume_raw``
        and ``jump_no_split``; securities without bars are absent.
    """
    raw = store.bulk_daily(security_ids, start, end)
    events = store.events_many(raw.keys())
    out: Dict[int, pd.DataFrame] = {}
    for sid, frame in raw.items():
        prepared = prepare_daily(frame, *events.get(sid, (None, None)))
        if prepared.empty:
            _logger.debug("Security %s has no usable daily bars after flag filtering", sid)
            continue
        out[sid] = prepared
    return out


def prepare_daily(frame: pd.DataFrame, splits: Optional[pd.DataFrame], dividends: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Turn one raw vintage frame into P24 daily bars.

    Args:
        frame: ``FmpStore`` daily frame (raw prices, ``flags``).
        splits: Split events in ``adjust()`` form, or None.
        dividends: Dividend events in ``adjust()`` form, or None.

    Returns:
        Bars sorted by date; empty if nothing usable is left.
    """
    frame = frame.sort_index()
    flags = frame["flags"].fillna(0).astype(np.int64) if "flags" in frame.columns else pd.Series(0, index=frame.index)
    keep = (flags & _SKIP_FLAGS) == 0
    keep &= frame["close"].notna() & (frame["close"] > 0)
    frame, flags = frame[keep], flags[keep]
    if frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    adjusted = adjust(frame[PRICE_COLUMNS].astype(float), splits, dividends, mode="split")
    adjusted["open_raw"] = frame["open"].astype(float)
    adjusted["close_raw"] = frame["close"].astype(float)
    adjusted["volume_raw"] = frame["volume"].astype(float)
    adjusted["jump_no_split"] = (flags & JUMP_NO_SPLIT) != 0
    return adjusted


@lru_cache(maxsize=4)
def _sessions(start: date, end: date) -> pd.DatetimeIndex:
    days = mcal.get_calendar("XNYS").valid_days(start_date=start, end_date=end)
    return pd.DatetimeIndex(days).tz_localize(None).normalize()


def to_weekly(daily: pd.DataFrame, as_of: Optional[date] = None) -> pd.DataFrame:
    """
    Resample daily bars to W-FRI weeks; each week is labelled with its last actual session.

    The week containing ``as_of`` is dropped when it still has XNYS sessions after ``as_of`` (an incomplete week). A
    delisted security's final short week is kept: no more bars will come.

    Args:
        daily: P24 daily bars.
        as_of: Date the data is complete up to (vintage date / today's close); None keeps every week.

    Returns:
        Weekly bars with the same columns.
    """
    if daily.empty:
        return daily.copy()
    key = pd.DatetimeIndex(daily.index).to_period("W-FRI")
    grouped = daily.groupby(key)
    agg: Dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    optional = {"open_raw": "first", "close_raw": "last", "volume_raw": "sum", "jump_no_split": "max", "eligible": "last",
                "dollar_volume_20": "last"}
    agg.update({column: how for column, how in optional.items() if column in daily.columns})
    weekly = grouped.agg(agg)  # type: ignore[arg-type]
    weekly.index = pd.DatetimeIndex(grouped.apply(lambda part: part.index.max()))
    if "jump_no_split" in weekly.columns:
        weekly["jump_no_split"] = weekly["jump_no_split"].astype(bool)
    if as_of is not None and _last_week_incomplete(weekly.index[-1], as_of):
        weekly = weekly.iloc[:-1]
    return weekly


def _last_week_incomplete(last_bar: pd.Timestamp, as_of: date) -> bool:
    friday = last_bar if last_bar.weekday() == 4 else last_bar + pd.offsets.Week(weekday=4)
    if friday.date() <= as_of:
        return False
    return len(_sessions(as_of + timedelta(days=1), friday.date())) > 0


def series_for_timeframe(daily: pd.DataFrame, timeframe: str, universe: UniverseParams,
                         as_of: Optional[date] = None) -> pd.DataFrame:
    """
    Daily or weekly bars with the point-in-time ``eligible`` column (computed on daily bars, spec §3.4).

    Args:
        daily: P24 daily bars.
        timeframe: ``"daily"`` or ``"weekly"``.
        universe: Universe filter parameters.
        as_of: Data completeness date (weekly only).

    Returns:
        Bars with an ``eligible`` boolean column.

    Raises:
        ValueError: Unknown timeframe.
    """
    frame = daily.copy()
    frame["eligible"] = eligibility_mask(frame, universe)
    frame["dollar_volume_20"] = median_dollar_volume(frame, universe)
    if timeframe == DAILY:
        return frame
    if timeframe == WEEKLY:
        weekly = to_weekly(frame, as_of)
        weekly["eligible"] = weekly["eligible"].astype(bool)
        return weekly
    raise ValueError(f"unknown timeframe {timeframe!r}")
