"""Synthetic price fixtures for the P24 tests."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# (bars, end_price) legs of the textbook knife: warm-up drift, top, first fall to support S3, a bounce that is
# large enough to make S3 a support (> 8%) but too small to reset the peak (< 25%), a sharp fall, a slowing fall to
# the signal low, then the rebound.
KNIFE_LEGS: Tuple[Tuple[int, float], ...] = (
    (330, 100.0),
    (18, 72.0),
    (8, 80.0),
    (10, 58.0),
    (10, 49.0),
    (60, 75.0),
)

# Same knife without its rebound, but with a crash into S3 (deep CCI at point 3); exit tests append their own tail.
DEEP_KNIFE_BASE: Tuple[Tuple[int, float], ...] = (
    (330, 100.0),
    (14, 80.0),
    (4, 66.0),
    (8, 74.0),
    (12, 58.0),
    (10, 49.0),
)

# A second, more violent leg of the same decline after the first signal (re-arm case, like VMW trade 2).
TWO_LEG_KNIFE: Tuple[Tuple[int, float], ...] = KNIFE_LEGS[:5] + (
    (12, 56.0),
    (10, 50.0),
    (6, 55.5),
    (10, 36.0),
    (10, 32.0),
    (60, 50.0),
)


def path_from_legs(legs: Sequence[Tuple[int, float]], start: float = 60.0, noise: float = 0.004,
                   seed: int = 7) -> np.ndarray:
    """
    Build a close path from geometric legs plus small deterministic noise.

    Args:
        legs: ``(bars, end_price)`` per leg.
        start: First close.
        noise: Daily noise standard deviation (fraction).
        seed: RNG seed.

    Returns:
        Close array.
    """
    rng = np.random.default_rng(seed)
    closes: List[float] = [start]
    for bars, end in legs:
        begin = closes[-1]
        steps = np.geomspace(begin, end, bars + 1)[1:]
        closes.extend(float(c) for c in steps)
    path = np.asarray(closes)
    wiggle = np.exp(rng.normal(0.0, noise, size=path.size))
    wiggle[0] = 1.0
    return path * wiggle


def bars_from_closes(closes: np.ndarray, start: str = "2015-01-02", freq: str = "B", range_frac: float = 0.012,
                     volume: float = 1_000_000.0) -> pd.DataFrame:
    """
    OHLCV frame from closes: open = previous close, high/low = close band widened to include the open.

    Args:
        closes: Close prices.
        start: First date.
        freq: Bar frequency.
        range_frac: Half-width of the high/low band around the close.
        volume: Constant volume.

    Returns:
        OHLCV frame on a ``DatetimeIndex``.
    """
    index = pd.date_range(start, periods=len(closes), freq=freq)
    close = pd.Series(closes, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(close * (1 + range_frac), open_)
    low = np.minimum(close * (1 - range_frac), open_)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": np.full(len(closes), volume)}, index=index)


def knife_bars(legs: Sequence[Tuple[int, float]] = KNIFE_LEGS, seed: int = 7) -> pd.DataFrame:
    """Return the textbook knife as an OHLCV frame."""
    return bars_from_closes(path_from_legs(legs, seed=seed))
