"""
Shared TA-Lib Adapters
----------------------

Thin wrappers around TA-Lib functions to ensure consistent behavior
between research (vectorbt) and production (backtrader) environments.
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

class ATR:
    @staticmethod
    def compute(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
        # ATR usually takes 1D inputs in our context, scale to 2D if needed later
        return pd.Series(
            talib.ATR(high.values, low.values, close.values, timeperiod=window),
            index=close.index
        )
