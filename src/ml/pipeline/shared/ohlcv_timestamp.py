"""
Bridge for **legacy column-based** OHLCV steps.

The canonical shape at the data layer is documented in ``src.data.ohlcv_contract``: a tz-naive
``DatetimeIndex`` plus OHLCV columns—**no** ``timestamp`` column. That is what you get whether
rows came from cache or from a provider after ``DataManager`` normalization.

Use ``coerce_ohlcv_timestamp_column`` only at the boundary where a stage still sorts on or masks
by a ``timestamp`` column (e.g. TRF volume correction). Prefer this over duplicating cache-vs-fetch
logic.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def coerce_ohlcv_timestamp_column(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """If ``df`` has a DatetimeIndex but no ``timestamp`` column, add one from the index."""
    if df is None or df.empty:
        return df
    if "timestamp" in df.columns:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy().reset_index()
        first = out.columns[0]
        out = out.rename(columns={first: "timestamp"})
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        return out
    return df
