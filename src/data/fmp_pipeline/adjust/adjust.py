"""
Read-time price adjustment (spec §8.3): ``adjust(prices_raw, splits, dividends, mode)``.

The store holds raw prices plus event tables (D3); every adjusted view is computed here. Pure and vectorized,
no I/O. Factor definitions live in ``src.data.fmp_pipeline.adjust.factors``.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd

from src.data.fmp_pipeline.adjust.factors import factor_arrays

AdjustMode = Literal["raw", "split", "total"]
PRICE_COLUMNS = ("open", "high", "low", "close")
_MODES = ("raw", "split", "total")


def adjust(
    prices_raw: pd.DataFrame,
    splits: Optional[pd.DataFrame] = None,
    dividends: Optional[pd.DataFrame] = None,
    mode: AdjustMode = "total",
    reference_closes: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Return an adjusted copy of a raw OHLCV series.

    - ``raw``: unchanged copy.
    - ``split``: prices multiplied by ``split_adj_factor`` (yfinance ``Close`` semantics); volume divided by it.
    - ``total``: prices additionally multiplied by ``div_adj_factor`` (yfinance ``Adj Close`` semantics); volume
      is split-adjusted only.

    Price columns among ``open``/``high``/``low``/``close`` and ``volume`` are adjusted when present; every other
    column is passed through untouched.

    Args:
        prices_raw: Raw bars on an ascending ``DatetimeIndex``; ``close`` is required unless ``mode`` is ``raw``.
        splits: Columns ``date``, ``numerator``, ``denominator``.
        dividends: Columns ``date`` (ex-date) and ``amount`` (cash per share on the raw basis of the ex-date).
        mode: One of ``raw``, ``split``, ``total``.
        reference_closes: Optional raw daily closes for the dividend ``C_prev`` lookup (use for intraday bars).

    Returns:
        A new frame with the same index and columns; adjusted columns are float.

    Raises:
        ValueError: On an unknown mode, an unsorted index, or invalid events.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    out = prices_raw.copy()
    if mode == "raw" or prices_raw.empty:
        return out

    split_factor, div_factor = factor_arrays(prices_raw, splits, dividends, reference_closes)
    price_factor = split_factor * div_factor if mode == "total" else split_factor

    for column in PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column].to_numpy(dtype=float) * price_factor
    if "volume" in out.columns:
        out["volume"] = out["volume"].to_numpy(dtype=float) / split_factor
    return out
