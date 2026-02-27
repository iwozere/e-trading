import pandas as pd
import numpy as np
from typing import Optional
from numba import njit

@njit
def _find_first_barrier_hit(prices, start_idx, tpl_bars, barrier_up, barrier_down):
    """Numba-optimized search for first barrier hit."""
    for j in range(1, tpl_bars + 1):
        if start_idx + j >= len(prices):
            break
        future_price = prices[start_idx + j]
        if future_price >= barrier_up:
            return 1
        elif future_price <= barrier_down:
            return -1
    return 0

def get_triple_barrier_labels(
    ohlcv: pd.DataFrame,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    tpl_bars: int = 12,
    atr_period: int = 14
) -> pd.Series:
    """
    Implements Triple Barrier Method for labeling.
    - Upper Barrier: Profit Take (based on ATR)
    - Lower Barrier: Stop Loss (based on ATR)
    - Vertical Barrier: Time Limit (Time Path Limit)

    Returns: 1 (Buy/High), -1 (Sell/Low), 0 (Hold/None)
    """
    # 1. Calculate Volatility (ATR)
    # Using TA-Lib for parity but keeping manual fallback for visibility
    try:
        import talib
        atr = talib.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=atr_period)
    except ImportError:
        high_low = ohlcv['high'] - ohlcv['low']
        high_close = np.abs(ohlcv['high'] - ohlcv['close'].shift(1))
        low_close = np.abs(ohlcv['low'] - ohlcv['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()

    prices = ohlcv['close'].values
    atr_values = atr.values
    labels_arr = np.zeros(len(prices))

    # 2. Optimized iteration
    for i in range(len(ohlcv) - 1):
        if np.isnan(atr_values[i]):
            continue

        start_price = prices[i]
        barrier_up = start_price + pt_mult * atr_values[i]
        barrier_down = start_price - sl_mult * atr_values[i]

        labels_arr[i] = _find_first_barrier_hit(prices, i, tpl_bars, barrier_up, barrier_down)

    return pd.Series(labels_arr, index=ohlcv.index, name="label").astype(int)
