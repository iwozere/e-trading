"""
Canonical in-memory OHLCV shape from ``DataManager.get_ohlcv`` / ``get_ohlcv_batch`` and from
``UnifiedCache`` reads for the same symbol/timeframe/range. **Cached and freshly downloaded rows
are merged into this single representation** in ``DataManager._normalize_ohlcv``.

Contract:

- **Index:** ``DatetimeIndex``, timezone-naive (bar open time).
- **Columns:** ``open``, ``high``, ``low``, ``close``, ``volume`` (lowercase). No ``timestamp``
  column—the index is the time axis.

Pipeline or strategy code that still assumes a ``timestamp`` column should call
``coerce_ohlcv_timestamp_column`` (``src.ml.pipeline.shared.ohlcv_timestamp``) at that boundary
instead of treating cache vs network differently.

``validate_normalized_ohlcv`` is for tests and diagnostics, not for hot-path assertions.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def validate_normalized_ohlcv(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if ``df`` does not match the normalized OHLCV contract."""
    if df is None:
        raise ValueError("OHLCV frame is None")
    if df.empty:
        return
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"Expected DatetimeIndex, got {type(df.index).__name__}")
    if df.index.tz is not None:
        raise ValueError("Expected tz-naive DatetimeIndex")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "timestamp" in df.columns:
        raise ValueError("Normalized OHLCV must not have a 'timestamp' column (time is the index)")
