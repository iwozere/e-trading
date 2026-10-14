"""
Convert yfinance bars and actions back to the raw basis of the FMP store (spec §10.1–§10.2).

yfinance, even with ``auto_adjust=False``, returns ``Close``/``Open``/``High``/``Low``, ``Volume`` and
``Dividends`` restated for every split up to *download time*. Appending those bars to a raw series after a new
split would introduce a jump of exactly the split ratio, so every increment is un-adjusted here first.

Unlike ``adjust()``, splits dated after the last bar are **not** ignored: yfinance has already applied them, so
they must be undone too. That is what makes the long-gap case (§10.2.5) safe.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.fmp_pipeline.adjust.factors import bar_dates, normalize_splits, split_adj_factor

_YF_PRICE_COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}


def extract_yfinance_splits(df_yf: pd.DataFrame) -> pd.DataFrame:
    """
    Return the splits reported in a yfinance ``Stock Splits`` column.

    Args:
        df_yf: yfinance history with actions enabled.

    Returns:
        Frame with columns ``date``, ``numerator``, ``denominator`` (``denominator`` is always 1; yfinance gives a
        single float ratio, e.g. ``4.0`` or ``0.1`` for a 1-for-10 reverse split). Empty if the column is absent.
    """
    if "Stock Splits" not in df_yf.columns:
        return pd.DataFrame(columns=["date", "numerator", "denominator"])
    events = df_yf.loc[df_yf["Stock Splits"].fillna(0) != 0, "Stock Splits"]
    return pd.DataFrame(
        {
            "date": pd.DatetimeIndex(bar_dates(events.index)),
            "numerator": events.to_numpy(dtype=float),
            "denominator": 1.0,
        }
    ).reset_index(drop=True)


def extract_yfinance_dividends(df_yf: pd.DataFrame, splits: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Return the dividends reported in a yfinance ``Dividends`` column, restated on the raw basis of each ex-date.

    Args:
        df_yf: yfinance history with actions enabled.
        splits: All splits known at download time; defaults to those in ``df_yf``.

    Returns:
        Frame with columns ``date`` (ex-date) and ``amount`` (raw basis). Empty if the column is absent.
    """
    if "Dividends" not in df_yf.columns:
        return pd.DataFrame(columns=["date", "amount"])
    events = df_yf.loc[df_yf["Dividends"].fillna(0) != 0, "Dividends"]
    dates = bar_dates(events.index)
    split_events = normalize_splits(extract_yfinance_splits(df_yf) if splits is None else splits)
    amounts = events.to_numpy(dtype=float) / split_adj_factor(dates, split_events)
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "amount": amounts})


def yfinance_to_raw(df_yf: pd.DataFrame, splits: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Un-adjust yfinance OHLCV bars to raw prices and raw volume.

    ``close_raw = close_yf / split_adj_factor`` and ``volume_raw = volume_yf * split_adj_factor``, where the factor
    covers every split dated after the bar — inside the downloaded window or after it.

    Args:
        df_yf: yfinance history downloaded with ``auto_adjust=False`` (``Open``/``High``/``Low``/``Close``/``Volume``,
            optionally ``Dividends`` and ``Stock Splits``). Lowercase column names are accepted too.
        splits: All splits known at download time (columns ``date``, ``numerator``, ``denominator``). Defaults to
            the ``Stock Splits`` column of ``df_yf``; pass it explicitly when the window ends before today.

    Returns:
        Frame on the same index with float columns ``open``, ``high``, ``low``, ``close``, ``volume`` (those present).
        ``Adj Close`` and the action columns are dropped.

    Raises:
        ValueError: If no price column is present, or on invalid splits.
    """
    renamed = df_yf.rename(columns=_YF_PRICE_COLUMNS)
    columns = [c for c in _YF_PRICE_COLUMNS.values() if c in renamed.columns]
    if not any(c in columns for c in ("open", "high", "low", "close")):
        raise ValueError("df_yf has no price columns")
    out = renamed[columns].astype(float)
    if out.empty:
        return out

    split_events = normalize_splits(extract_yfinance_splits(df_yf) if splits is None else splits)
    factor = split_adj_factor(bar_dates(out.index), split_events)
    for column in columns:
        if column == "volume":
            out[column] = out[column] * factor
        else:
            out[column] = out[column] / factor
    return out
