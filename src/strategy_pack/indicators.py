"""Small OHLCV helpers (SuperTrend logic aligned with ``run_plotter``)."""

from __future__ import annotations

import pandas as pd


def atr_rma(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def supertrend_line_and_direction(df: pd.DataFrame, period: int, multiplier: float) -> tuple[pd.Series, pd.Series]:
    """
    SuperTrend line and trend direction (+1 bullish, -1 bearish).
    Ported from ``src/backtester/plotter/run_plotter.py::_calculate_supertrend`` with direction output.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    if len(df) == 0:
        return supertrend, direction

    direction.iloc[0] = 1 if close.iloc[0] >= hl2.iloc[0] else -1
    supertrend.iloc[0] = lower_band.iloc[0] if direction.iloc[0] == 1 else upper_band.iloc[0]

    for i in range(1, len(df)):
        if pd.isna(atr.iloc[i]):
            direction.iloc[i] = direction.iloc[i - 1]
            supertrend.iloc[i] = supertrend.iloc[i - 1]
            continue

        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        ub = float(upper_band.iloc[i])
        lb = float(lower_band.iloc[i])
        if direction.iloc[i] == 1 and lb < float(lower_band.iloc[i - 1]):
            lower_band.iloc[i] = lower_band.iloc[i - 1]
            lb = float(lower_band.iloc[i])
        if direction.iloc[i] == -1 and ub > float(upper_band.iloc[i - 1]):
            upper_band.iloc[i] = upper_band.iloc[i - 1]
            ub = float(upper_band.iloc[i])

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = float(lower_band.iloc[i])
        else:
            supertrend.iloc[i] = float(upper_band.iloc[i])

    return supertrend, direction


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()
