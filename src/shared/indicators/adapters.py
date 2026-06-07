"""
Shared TA-Lib Adapters
----------------------

Thin wrappers around TA-Lib functions to ensure consistent behavior
between research (vectorbt) and production (backtrader) environments.

ARCHITECTURE NOTE — two TA-Lib adapter layers (SHARED-1)
---------------------------------------------------------
There are currently **two** TA-Lib adapter implementations in the repo:

* ``src/shared/indicators/adapters.py`` (this file) — research-focused wrappers
  that accept Series, DataFrame, and numpy arrays; used by the VectorBT pipeline.

* ``src/indicators/adapters/ta_lib_adapter.py`` — production-grade adapter with
  strict registry-based input validation; used by the live-trading indicator
  service and the Backtrader backtesting engine.

The long-term goal is to **consolidate to a single adapter** so that research
and production compute indicators identically and divergence is structurally
impossible.  Until that refactor is done, keep the two sets of wrappers
functionally consistent — any parameter change in one must be mirrored in the
other to avoid silent divergence in strategy signals.
"""

import talib
import numpy as np
import pandas as pd
from typing import Union, Dict

class RSI:
    @staticmethod
    def compute(close: Union[pd.Series, pd.DataFrame, np.ndarray], window: int) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        Compute Relative Strength Index (RSI).
        Handles Series, DataFrame, and numpy arrays (1D/2D).
        """
        if isinstance(close, pd.DataFrame):
            return close.apply(lambda x: talib.RSI(x.values, timeperiod=window))
        elif isinstance(close, np.ndarray):
            if close.ndim == 1:
                return talib.RSI(close, timeperiod=window)
            else:
                # 2D array: apply along axis 0 (columns)
                return np.apply_along_axis(lambda x: talib.RSI(x, timeperiod=window), 0, close)

        # Assume Series
        return pd.Series(talib.RSI(close.values, timeperiod=window), index=close.index)

class BBANDS:
    @staticmethod
    def compute(close: Union[pd.Series, pd.DataFrame, np.ndarray], window: int, nbdevup: float, nbdevdn: float) -> Union[pd.DataFrame, Dict[str, Union[pd.DataFrame, np.ndarray]]]:
        """
        Compute Bollinger Bands.
        Handles Series, DataFrame, and numpy arrays.
        """
        if isinstance(close, pd.DataFrame):
            # For vectorbt broadcasting, returning a dict of DataFrames is often best
            upper = close.apply(lambda x: talib.BBANDS(x.values, window, nbdevup, nbdevdn, 0)[0])
            middle = close.apply(lambda x: talib.BBANDS(x.values, window, nbdevup, nbdevdn, 0)[1])
            lower = close.apply(lambda x: talib.BBANDS(x.values, window, nbdevup, nbdevdn, 0)[2])
            return {
                'upperband': upper,
                'middleband': middle,
                'lowerband': lower
            }
        elif isinstance(close, np.ndarray):
            if close.ndim == 1:
                u, m, l = talib.BBANDS(close, window, nbdevup, nbdevdn, 0)
                return {'upperband': u, 'middleband': m, 'lowerband': l}
            else:
                # 2D array
                u = np.apply_along_axis(lambda x: talib.BBANDS(x, window, nbdevup, nbdevdn, 0)[0], 0, close)
                m = np.apply_along_axis(lambda x: talib.BBANDS(x, window, nbdevup, nbdevdn, 0)[1], 0, close)
                l = np.apply_along_axis(lambda x: talib.BBANDS(x, window, nbdevup, nbdevdn, 0)[2], 0, close)
                return {'upperband': u, 'middleband': m, 'lowerband': l}

        # Assume Series
        upper, middle, lower = talib.BBANDS(
            close.values,
            timeperiod=window,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=0
        )
        return pd.DataFrame({
            'upperband': upper,
            'middleband': middle,
            'lowerband': lower
        }, index=close.index)

class SMA:
    @staticmethod
    def compute(close: Union[pd.Series, pd.DataFrame, np.ndarray], window: int) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        if isinstance(close, pd.DataFrame):
            return close.apply(lambda x: talib.SMA(x.values, timeperiod=window))
        elif isinstance(close, np.ndarray):
            if close.ndim == 1:
                return talib.SMA(close, timeperiod=window)
            return np.apply_along_axis(lambda x: talib.SMA(x, timeperiod=window), 0, close)
        return pd.Series(talib.SMA(close.values, timeperiod=window), index=close.index)

class EMA:
    @staticmethod
    def compute(close: Union[pd.Series, pd.DataFrame, np.ndarray], window: int) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        if isinstance(close, pd.DataFrame):
            return close.apply(lambda x: talib.EMA(x.values, timeperiod=window))
        elif isinstance(close, np.ndarray):
            if close.ndim == 1:
                return talib.EMA(close, timeperiod=window)
            return np.apply_along_axis(lambda x: talib.EMA(x, timeperiod=window), 0, close)
        return pd.Series(talib.EMA(close.values, timeperiod=window), index=close.index)

class ADX:
    @staticmethod
    def compute(high: Union[pd.Series, pd.DataFrame], low: Union[pd.Series, pd.DataFrame], close: Union[pd.Series, pd.DataFrame], window: int) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute Average Directional Index (ADX).
        """
        if isinstance(close, pd.DataFrame):
            # Alignment check: Ensure all inputs are DataFrames with matching columns
            res = {}
            for col in close.columns:
                res[col] = talib.ADX(high[col].values, low[col].values, close[col].values, timeperiod=window)
            return pd.DataFrame(res, index=close.index)

        # Assume Series
        return pd.Series(
            talib.ADX(high.values, low.values, close.values, timeperiod=window),
            index=close.index
        )

class ATR:
    @staticmethod
    def compute(high: Union[pd.Series, pd.DataFrame], low: Union[pd.Series, pd.DataFrame], close: Union[pd.Series, pd.DataFrame], window: int) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute Average True Range (ATR).
        Handles multi-symbol DataFrames.
        """
        if isinstance(close, pd.DataFrame):
            res = {}
            for col in close.columns:
                res[col] = talib.ATR(high[col].values, low[col].values, close[col].values, timeperiod=window)
            return pd.DataFrame(res, index=close.index)

        # Assume Series
        return pd.Series(
            talib.ATR(high.values, low.values, close.values, timeperiod=window),
            index=close.index
        )
