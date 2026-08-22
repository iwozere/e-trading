"""
P21 Momentum — Signal computation (docs/pipeline-specification.md §4).

**Critical (spec §4):** ranking uses ``signal`` (risk-adjusted), not
``raw_return``. This error fails silently and costs roughly the entire
factor premium — see test_signal.py's regression guard.

**Critical (spec §4):** the lookback window ends at ``-SKIP_RECENT``, not
``-1``. Including the most recent month imports short-term reversal into the
portfolio and consistently degrades results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.ml.pipeline.p21_momentum.config import LOOKBACK_START, MIN_HISTORY, MIN_VOL, MIN_WEEKLY_BARS, SKIP_RECENT


@dataclass(slots=True)
class SignalResult:
    """Output of compute_signal() for one ticker."""

    raw_return: float
    vol: float
    signal: float  # RANK ON THIS FIELD, not raw_return


def compute_signal(adj_close: pd.Series) -> Optional[SignalResult]:
    """
    Compute the 12-1 month risk-adjusted momentum signal for one ticker.

    Args:
        adj_close: Split/dividend-adjusted daily close series (§10.1),
            ascending by date, most recent value last.

    Returns:
        SignalResult, or None if the ticker is ineligible: insufficient
        history (< MIN_HISTORY bars), insufficient weekly observations for a
        stable vol estimate (< MIN_WEEKLY_BARS), or near-zero volatility
        (< MIN_VOL — guards against division by ~0).
    """
    if len(adj_close) < MIN_HISTORY:
        return None  # IPO / insufficient history

    window = adj_close.iloc[-LOOKBACK_START:-SKIP_RECENT]  # e.g. 231 bars

    raw_return = float(window.iloc[-1] / window.iloc[0] - 1.0)

    weekly = window.resample("W-FRI").last().pct_change().dropna()
    if len(weekly) < MIN_WEEKLY_BARS:
        return None
    vol = float(weekly.std(ddof=1) * math.sqrt(52))

    if vol < MIN_VOL:  # guard against division by ~0
        return None

    return SignalResult(raw_return=raw_return, vol=vol, signal=raw_return / vol)
