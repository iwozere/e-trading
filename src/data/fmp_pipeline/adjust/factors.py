"""
Split and dividend adjustment factors (spec §8.2) — pure, vectorized, no I/O.

Definitions, for a bar at date ``t``:

- ``split_adj_factor(t) = PRODUCT over splits with S > t of (1 / ratio_S)``, where
  ``ratio = numerator / denominator`` and ``S`` is the first trading day at the new share count.
- ``div_adj_factor(t) = PRODUCT over dividends with D > t of DF(D)``, with
  ``DF(D) = 1 - d / C_prev``; ``d`` is the cash amount on the raw basis of ``D`` and ``C_prev`` is the raw close
  of the last bar before ``D``.

Two clarifications of §8.2 made here (recorded in ``docs/decisions.md``):

1. ``C_prev`` is restated on the share basis of ``D`` before computing ``DF``. When a split falls between
   ``C_prev``'s bar and ``D`` (the common case being a dividend whose ex-date *is* the split date), ``d`` is
   quoted per post-split share while ``C_prev`` is a pre-split price; dividing one by the other would
   overstate the dividend by the split ratio.
2. A dividend with no earlier bar gets ``DF = 1``. It could only affect bars before it, and there are none.

Comparisons are made on calendar dates (the bar timestamp normalized to midnight, wall-clock time for
tz-aware indexes), so the same code serves daily and intraday bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

SPLIT_COLUMNS = ("date", "numerator", "denominator")
DIVIDEND_COLUMNS = ("date", "amount")


@dataclass(frozen=True)
class SplitEvents:
    """Validated splits, sorted by date: ``dates`` (datetime64[ns]) and ``ratios`` (numerator / denominator)."""

    dates: np.ndarray
    ratios: np.ndarray


@dataclass(frozen=True)
class DividendEvents:
    """Validated dividends, sorted by date: ex-``dates`` (datetime64[ns]) and raw-basis cash ``amounts``."""

    dates: np.ndarray
    amounts: np.ndarray


def bar_dates(index: pd.Index) -> np.ndarray:
    """
    Return the calendar date of every bar as a tz-naive ``datetime64[ns]`` array.

    Args:
        index: A ``DatetimeIndex``; tz-aware indexes are reduced to their local wall-clock date.

    Returns:
        Array of dates normalized to midnight, same length and order as ``index``.

    Raises:
        TypeError: If ``index`` is not a ``DatetimeIndex``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(index).__name__}")
    if index.tz is not None:
        index = index.tz_localize(None)
    return index.normalize().to_numpy(dtype="datetime64[ns]")


def normalize_splits(splits: Optional[pd.DataFrame]) -> SplitEvents:
    """
    Validate, deduplicate and sort a splits table.

    Args:
        splits: Frame with columns ``date``, ``numerator``, ``denominator``; ``None`` or empty means no splits.

    Returns:
        The splits as sorted arrays. Exact duplicate rows are collapsed to one.

    Raises:
        ValueError: On missing columns, a non-positive or non-finite ratio, or two different ratios on one date.
    """
    if splits is None or splits.empty:
        return SplitEvents(np.array([], dtype="datetime64[ns]"), np.array([], dtype=float))
    _require_columns(splits, SPLIT_COLUMNS, "splits")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(splits["date"]).dt.normalize(),
            "ratio": splits["numerator"].astype(float) / splits["denominator"].astype(float),
        }
    )
    bad = frame[~np.isfinite(frame["ratio"]) | (frame["ratio"] <= 0)]
    if not bad.empty:
        raise ValueError(f"invalid split ratio on {list(bad['date'].dt.date)}")
    frame = frame.drop_duplicates().sort_values("date", kind="stable")
    conflicts = frame[frame["date"].duplicated(keep=False)]
    if not conflicts.empty:
        raise ValueError(f"conflicting split ratios on {sorted(set(conflicts['date'].dt.date))}")
    return SplitEvents(frame["date"].to_numpy(dtype="datetime64[ns]"), frame["ratio"].to_numpy(dtype=float))


def normalize_dividends(dividends: Optional[pd.DataFrame]) -> DividendEvents:
    """
    Validate, deduplicate and sort a dividends table.

    Two *different* amounts on one ex-date (e.g. regular plus special) are both kept; only exact duplicate rows
    are collapsed.

    Args:
        dividends: Frame with columns ``date`` (ex-date) and ``amount`` (raw basis); ``None`` or empty means none.

    Returns:
        The dividends as sorted arrays.

    Raises:
        ValueError: On missing columns or a negative or non-finite amount.
    """
    if dividends is None or dividends.empty:
        return DividendEvents(np.array([], dtype="datetime64[ns]"), np.array([], dtype=float))
    _require_columns(dividends, DIVIDEND_COLUMNS, "dividends")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dividends["date"]).dt.normalize(),
            "amount": dividends["amount"].astype(float),
        }
    )
    bad = frame[~np.isfinite(frame["amount"]) | (frame["amount"] < 0)]
    if not bad.empty:
        raise ValueError(f"invalid dividend amount on {list(bad['date'].dt.date)}")
    frame = frame.drop_duplicates().sort_values("date", kind="stable")
    return DividendEvents(frame["date"].to_numpy(dtype="datetime64[ns]"), frame["amount"].to_numpy(dtype=float))


def split_adj_factor(dates: np.ndarray, splits: SplitEvents) -> np.ndarray:
    """
    Compute ``split_adj_factor`` for each date: the product of ``1 / ratio`` over splits strictly after it.

    Args:
        dates: Bar dates (``datetime64[ns]``), any order.
        splits: Sorted splits from ``normalize_splits``.

    Returns:
        Float array aligned with ``dates``.
    """
    return _suffix_product_after(dates, splits.dates, 1.0 / splits.ratios)


def dividend_factors(
    splits: SplitEvents,
    dividends: DividendEvents,
    close_dates: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Compute ``DF(D)`` for every dividend.

    Args:
        splits: Sorted splits, used to restate ``C_prev`` on the share basis of the ex-date.
        dividends: Sorted dividends.
        close_dates: Sorted calendar dates of the raw closes used for ``C_prev`` lookups.
        closes: Raw closes aligned with ``close_dates``; non-finite values are skipped.

    Returns:
        Float array aligned with ``dividends.dates``.

    Raises:
        ValueError: If a ``C_prev`` is not positive, or a dividend is at least as large as ``C_prev``.
    """
    finite = np.isfinite(closes)
    close_dates = close_dates[finite]
    closes = closes[finite]
    prev_idx = np.searchsorted(close_dates, dividends.dates, side="left") - 1
    has_prev = prev_idx >= 0
    factors = np.ones(len(dividends.dates), dtype=float)
    if not has_prev.any():
        return factors

    prev_dates = close_dates[prev_idx[has_prev]]
    c_prev = closes[prev_idx[has_prev]]
    if (c_prev <= 0).any():
        raise ValueError("non-positive raw close before a dividend ex-date")
    # Splits with C_prev's date < S <= D change the share basis between the two prices.
    basis = split_adj_factor(prev_dates, splits) / split_adj_factor(dividends.dates[has_prev], splits)
    df = 1.0 - dividends.amounts[has_prev] / (c_prev * basis)
    if (df <= 0).any():
        bad = dividends.dates[has_prev][df <= 0]
        raise ValueError(f"dividend not smaller than the previous close on {list(pd.to_datetime(bad).date)}")
    factors[has_prev] = df
    return factors


def adjustment_factors(
    prices_raw: pd.DataFrame,
    splits: Optional[pd.DataFrame] = None,
    dividends: Optional[pd.DataFrame] = None,
    reference_closes: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute the per-bar split and dividend adjustment factors for a raw price series.

    Events dated after the last bar are ignored: they do not affect existing bars until a bar exists after them
    (spec §8.2).

    Args:
        prices_raw: Raw bars on an ascending ``DatetimeIndex`` with a ``close`` column.
        splits: Splits table (see ``normalize_splits``).
        dividends: Dividends table on the raw basis (see ``normalize_dividends``).
        reference_closes: Optional raw **daily** closes used for ``C_prev``. Pass these for intraday bars so that
            the previous close is the session close, not the last (possibly extended-hours) bar.

    Returns:
        Frame on ``prices_raw.index`` with float columns ``split_factor`` and ``div_factor``.

    Raises:
        ValueError: If the index is not ascending, or on invalid events (see the normalizers).
    """
    if not prices_raw.index.is_monotonic_increasing:
        raise ValueError("prices_raw index must be sorted ascending")
    dates = bar_dates(prices_raw.index)
    split_events = normalize_splits(splits)
    dividend_events = normalize_dividends(dividends)
    if len(dates) == 0:
        return pd.DataFrame({"split_factor": [], "div_factor": []}, index=prices_raw.index, dtype=float)

    last = dates[-1]
    keep = split_events.dates <= last
    split_events = SplitEvents(split_events.dates[keep], split_events.ratios[keep])
    keep = dividend_events.dates <= last
    dividend_events = DividendEvents(dividend_events.dates[keep], dividend_events.amounts[keep])

    if reference_closes is not None:
        ref = reference_closes.sort_index()
        close_dates, closes = bar_dates(ref.index), ref.to_numpy(dtype=float)
    else:
        close_dates, closes = dates, prices_raw["close"].to_numpy(dtype=float)

    df_per_event = dividend_factors(split_events, dividend_events, close_dates, closes)
    return pd.DataFrame(
        {
            "split_factor": split_adj_factor(dates, split_events),
            "div_factor": _suffix_product_after(dates, dividend_events.dates, df_per_event),
        },
        index=prices_raw.index,
    )


def _suffix_product_after(dates: np.ndarray, event_dates: np.ndarray, event_factors: np.ndarray) -> np.ndarray:
    """Return, for each date, the product of ``event_factors`` over events dated strictly after it."""
    if len(event_dates) == 0:
        return np.ones(len(dates), dtype=float)
    suffix = np.append(np.cumprod(event_factors[::-1])[::-1], 1.0)
    return suffix[np.searchsorted(event_dates, dates, side="right")]


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    """Raise ``ValueError`` if ``frame`` lacks any of ``columns``."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}")
