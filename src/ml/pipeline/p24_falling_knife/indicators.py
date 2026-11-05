"""
P24 Falling Knife — indicators (spec §4.1).

ADX, +DI, −DI and CCI come from TA-Lib. PPO is computed here with true EMAs: TA-Lib's ``PPO`` defaults to a simple
moving average (``matype=0``) and must not be used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from src.ml.pipeline.p24_falling_knife.config import TimeframeParams

INDICATOR_COLUMNS = ("adx", "plus_di", "minus_di", "ppo", "ppo_signal", "ppo_hist", "cci")


def ema(values: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average with ``alpha = 2 / (span + 1)``, seeded at the first value.

    Args:
        values: Input series.
        span: EMA span.

    Returns:
        EMA series of the same index.
    """
    return values.ewm(span=span, adjust=False).mean()


def ppo(close: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    """
    Percentage Price Oscillator in percent: ``100 * (EMA_fast − EMA_slow) / EMA_slow``.

    The first ``slow − 1`` values are set to NaN: an EMA seeded at the first close is not meaningful before it has seen
    a full slow window. The signal line starts after its own warm-up.

    Args:
        close: Close prices.
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal-line EMA span.

    Returns:
        Frame with ``ppo``, ``ppo_signal``, ``ppo_hist``.
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = 100.0 * (fast_ema - slow_ema) / slow_ema
    line.iloc[: slow - 1] = np.nan
    sig = ema(line.dropna(), signal).reindex(line.index)
    sig.iloc[: slow - 1 + signal - 1] = np.nan
    return pd.DataFrame({"ppo": line, "ppo_signal": sig, "ppo_hist": line - sig}, index=close.index)


def compute_indicators(bars: pd.DataFrame, params: TimeframeParams) -> pd.DataFrame:
    """
    Add the P24 indicator columns to an OHLC frame.

    Args:
        bars: Frame with ``high``, ``low``, ``close`` (any extra columns are kept).
        params: Timeframe parameters (periods).

    Returns:
        Copy of ``bars`` with ``INDICATOR_COLUMNS`` added.
    """
    out = bars.copy()
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)
    out["adx"] = talib.ADX(high, low, close, timeperiod=params.adx_period)
    out["plus_di"] = talib.PLUS_DI(high, low, close, timeperiod=params.adx_period)
    out["minus_di"] = talib.MINUS_DI(high, low, close, timeperiod=params.adx_period)
    out["cci"] = talib.CCI(high, low, close, timeperiod=params.cci_period)
    ppo_frame = ppo(out["close"].astype(np.float64), params.ppo_fast, params.ppo_slow, params.ppo_signal)
    for column in ("ppo", "ppo_signal", "ppo_hist"):
        out[column] = ppo_frame[column]
    return out
