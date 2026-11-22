"""
Survivor-bias study — month-end panel from the EODHD store (docs/Design.md, "Monthly panel").

One row per stock and calendar month in which it traded: last bar date, total-return and raw close, 63-session median
dollar volume, bar count, the 12-1 momentum inputs, and the return to the next month-end with the delisting flag. The
store's 54M daily rows are read one price bucket at a time in a process pool.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data.eodhd_pipeline import config as eodhd_config
from src.data.eodhd_pipeline.store import EodhdStore
from src.data.fmp_pipeline.adjust.adjust import adjust
from src.ml.research.survivor_bias import config
from src.ml.research.survivor_bias.delisting import classify_end
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]
PANEL_COLUMNS = ["security_id", "month", "last_date", "close_tr", "close_raw", "dollar_volume", "bars", "mom_ret",
                 "mom_vol", "ret_next", "ret_vendor", "suspect", "next_terminal", "end_kind"]


@dataclass(frozen=True)
class StockInfo:
    """What the panel needs to know about one stock besides its bars."""

    security_id: int
    symbol: str
    is_active: bool
    last_date: Optional[date]


def sources_agree(ours: pd.Series, vendor: pd.Series) -> pd.Series:
    """
    True where two returns agree within ``MAX_SOURCE_GAP`` (as growth factors), or the vendor value is missing.

    Args:
        ours: Return from our adjustment.
        vendor: The same return from EODHD's adjusted close.

    Returns:
        Boolean series on ``ours.index``.
    """
    gap = ((1.0 + ours) / (1.0 + vendor) - 1.0).abs()
    return (gap <= config.MAX_SOURCE_GAP) | vendor.isna()


def bad_print_days(close_raw: pd.Series, splits: Optional[pd.DataFrame]) -> pd.Series:
    """
    Days whose raw close is a data error (docs/Design.md, Amendment 2).

    A day is bad when the raw close moves 3× up or to a third and returns within 1.5× of the previous close within
    ``SPIKE_REVERT_BARS`` bars (a spike), or moves ``IMPOSSIBLE_MOVE``× either way on a day without a split.

    Args:
        close_raw: Raw daily closes, ascending.
        splits: Splits in ``adjust()`` form, or None.

    Returns:
        Boolean series on ``close_raw.index``.
    """
    prev = close_raw.shift(1)
    move = close_raw / prev
    big = (move > config.SPIKE_MOVE) | (move < 1.0 / config.SPIKE_MOVE)
    reverts = pd.Series(False, index=close_raw.index)
    for k in range(1, config.SPIKE_REVERT_BARS + 1):
        back = close_raw.shift(-k) / prev
        reverts |= (back >= 1.0 / config.SPIKE_BACK) & (back <= config.SPIKE_BACK)
    split_days = set(pd.to_datetime(splits["date"])) if splits is not None and not splits.empty else set()
    no_split = ~pd.Series(close_raw.index.isin(list(split_days)), index=close_raw.index)
    impossible = ((move > config.IMPOSSIBLE_MOVE) | (move < 1.0 / config.IMPOSSIBLE_MOVE)) & no_split
    return ((big & reverts) | impossible).fillna(False).astype(bool)


def daily_features(frame: pd.DataFrame, splits: Optional[pd.DataFrame], dividends: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Daily total-return close, raw close, median dollar volume and momentum inputs of one stock.

    Args:
        frame: Raw daily bars (``open`` … ``volume``) on an ascending ``DatetimeIndex``.
        splits: Splits in ``adjust()`` form, or None.
        dividends: Dividends in ``adjust()`` form, or None.

    Returns:
        Frame with ``close_tr``, ``close_raw``, ``dollar_volume``, ``bars``, ``mom_ret``, ``mom_vol``; empty if no bar
        has a positive close.
    """
    frame = frame.sort_index()
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    if frame.empty:
        return pd.DataFrame(columns=["close_tr", "close_raw", "dollar_volume", "bars", "mom_ret", "mom_vol"])
    total = adjust(frame[PRICE_COLUMNS].astype(float), splits, dividends, mode="total")
    out = pd.DataFrame(index=frame.index)
    out["close_tr"] = total["close"].astype(float)
    out["close_raw"] = frame["close"].astype(float)
    volume = frame["volume"].astype(float)
    out["dollar_volume"] = (out["close_raw"] * volume.where(volume > 0)).rolling(
        config.DOLLAR_VOLUME_BARS, min_periods=config.DOLLAR_VOLUME_BARS // 2).median()
    out["bars"] = np.arange(1, len(out) + 1)
    vendor = frame["closeadj_vendor"].astype(float) if "closeadj_vendor" in frame.columns else pd.Series(np.nan, index=frame.index)
    out["close_vendor"] = vendor.where(vendor > 0)
    momentum = out["close_tr"].shift(config.SKIP_BARS) / out["close_tr"].shift(config.LOOKBACK_BARS) - 1.0
    vendor_momentum = out["close_vendor"].shift(config.SKIP_BARS) / out["close_vendor"].shift(config.LOOKBACK_BARS) - 1.0
    agree = sources_agree(momentum, vendor_momentum)
    out["bad_day"] = bad_print_days(out["close_raw"], splits)
    clean_window = out["bad_day"].astype(float).rolling(config.LOOKBACK_BARS + 1, min_periods=1).sum() == 0
    out["mom_ret"] = momentum.where((out["bars"] >= config.MIN_MOMENTUM_BARS) & agree & clean_window)
    weekly = out["close_tr"].resample("W-FRI").last().dropna().pct_change()
    skip_weeks = round(config.SKIP_BARS / 5)
    vol = weekly.rolling(config.WEEKS_IN_WINDOW, min_periods=config.MIN_WEEKLY_RETURNS).std() * math.sqrt(52)
    vol = vol.shift(skip_weeks)
    out["mom_vol"] = vol.reindex(out.index, method="ffill").where(out["bars"] >= config.MIN_MOMENTUM_BARS)
    return out


def month_rows(daily: pd.DataFrame, end_kind: str) -> pd.DataFrame:
    """
    Month-end rows of one stock with the return to the next month-end.

    Args:
        daily: Output of ``daily_features``.
        end_kind: How the stock's history ends (``classify_end``).

    Returns:
        Frame with ``PANEL_COLUMNS`` except ``security_id``.
    """
    if daily.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS[1:])
    periods = pd.DatetimeIndex(daily.index).to_period("M")
    last = daily.groupby(periods).tail(1).copy()
    index = pd.DatetimeIndex(last.index)
    months = index.to_period("M")
    last["last_date"] = index.date
    last["month"] = months.to_timestamp()
    ordinal = np.asarray(months.asi8, dtype=np.int64)
    consecutive = np.append(ordinal[1:] - ordinal[:-1] == 1, False)
    close = last["close_tr"].to_numpy(dtype=float)
    ret_next = np.full(len(last), np.nan)
    ret_next[:-1] = close[1:] / close[:-1] - 1.0
    ret_next = np.where(consecutive, ret_next, np.nan)
    vendor = last["close_vendor"].to_numpy(dtype=float) if "close_vendor" in last.columns else np.full(len(last), np.nan)
    ret_vendor = np.full(len(last), np.nan)
    ret_vendor[:-1] = vendor[1:] / vendor[:-1] - 1.0
    ret_vendor = np.where(consecutive, ret_vendor, np.nan)
    terminal = np.zeros(len(last), dtype=bool)
    if end_kind != config.LIVE:
        terminal[-1] = True
        ret_next[-1] = 0.0
        ret_vendor[-1] = 0.0
        if len(last) > 1 and consecutive[-2]:
            terminal[-2] = True
    bad_month = daily["bad_day"].groupby(periods).any().to_numpy(dtype=bool) if "bad_day" in daily.columns else np.zeros(len(last), dtype=bool)
    suspect = bad_month.copy()
    suspect[:-1] |= bad_month[1:]
    last["ret_next"] = ret_next
    last["ret_vendor"] = ret_vendor
    last["suspect"] = suspect
    last["next_terminal"] = terminal
    last["end_kind"] = end_kind
    return last.reset_index(drop=True)[PANEL_COLUMNS[1:]]


def stock_panel(info: StockInfo, frame: pd.DataFrame, events: Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]],
                store_date: date) -> Tuple[pd.DataFrame, str]:
    """
    Panel rows and end kind of one stock.

    Args:
        info: Stock reference data.
        frame: Raw daily bars.
        events: ``(splits, dividends)`` in ``adjust()`` form.
        store_date: Latest date in the store.

    Returns:
        ``(rows, end_kind)``.
    """
    daily = daily_features(frame, *events)
    if daily.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS), config.LIVE
    last_date = info.last_date or daily.index[-1].date()
    end_kind = classify_end(info.symbol, info.is_active, last_date, store_date, daily["close_raw"], daily["close_tr"])
    rows = month_rows(daily, end_kind)
    rows.insert(0, "security_id", info.security_id)
    return rows, end_kind


_STORE: Optional[EodhdStore] = None


def _init_worker(data_root: str) -> None:
    global _STORE
    _STORE = EodhdStore(Path(data_root))


def _bucket_panel(infos: Sequence[StockInfo], store_date: date) -> List[pd.DataFrame]:
    if _STORE is None:
        raise RuntimeError("worker store not initialised")
    ids = [i.security_id for i in infos]
    bars = _STORE.bulk_daily(ids, config.DATA_START, store_date, vendor_adjusted=True)
    events = _STORE.events_many(bars.keys())
    parts = []
    for info in infos:
        frame = bars.get(info.security_id)
        if frame is None:
            continue
        rows, _ = stock_panel(info, frame, events.get(info.security_id, (None, None)), store_date)
        if not rows.empty:
            parts.append(rows)
    return parts


def build_panel(store: EodhdStore, infos: Sequence[StockInfo], workers: Optional[int] = None) -> pd.DataFrame:
    """
    Month-end panel of many stocks.

    Args:
        store: Open EODHD store (its root is passed to the workers).
        infos: Stocks to include.
        workers: Process count (default CPU count − 1); 1 runs in-process.

    Returns:
        Frame with ``PANEL_COLUMNS``.

    Raises:
        ValueError: The store has no price date.
    """
    store_date = store.as_of
    if store_date is None:
        raise ValueError("store has no price dates")
    buckets: Dict[int, List[StockInfo]] = {}
    for info in infos:
        buckets.setdefault(info.security_id % eodhd_config.PRICE_BUCKETS, []).append(info)
    groups = [buckets[k] for k in sorted(buckets)]
    parts: List[pd.DataFrame] = []
    n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if n_workers == 1:
        _init_worker(str(store.data_root))
        for group in groups:
            parts.extend(_bucket_panel(group, store_date))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker, initargs=(str(store.data_root),)) as pool:
            futures = [pool.submit(_bucket_panel, group, store_date) for group in groups]
            for done, future in enumerate(as_completed(futures), 1):
                parts.extend(future.result())
                if done % 8 == 0 or done == len(futures):
                    _logger.info("Survivor panel: %d of %d buckets done", done, len(futures))
    if not parts:
        return pd.DataFrame(columns=PANEL_COLUMNS)
    panel = pd.concat(parts, ignore_index=True)
    return panel.sort_values(["month", "security_id"]).reset_index(drop=True)
