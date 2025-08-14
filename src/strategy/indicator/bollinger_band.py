"""
Backtrader-Compatible Bollinger Bands Indicator Wrapper

This module provides a Backtrader-compatible wrapper for Bollinger Bands indicator.
Supports multiple backends: bt, bt-talib, talib.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class BollingerBandIndicator(bt.Indicator):
    """
    Bollinger Bands indicator wrapper for Backtrader.

    Parameters:
    -----------
    period : int
        The period for Bollinger Bands calculation (default: 20)
    devfactor : float
        The standard deviation factor (default: 2.0)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')
    line_names : tuple
        Custom line names for the indicator (default: None, uses 'upper', 'middle', 'lower')

    Note on TA-Lib vs Backtrader calculation:
    - When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
    - When using the 'talib' type, the entire indicator array is precalculated in __init__ and values are assigned in next().
    - For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
    """
    lines = ("upper", "middle", "lower")
    params = (
        ("period", 20),
        ("devfactor", 2.0),
        ("indicator_type", "bt"),
        ("line_names", None),
    )

    def __init__(self):
        # Handle custom line names
        if self.p.line_names is not None:
            self.lines = type(self.lines)(*self.p.line_names)
            self._line_names = self.p.line_names
        else:
            self._line_names = ("upper", "middle", "lower")
        super().__init__()
        self.addminperiod(self.p.period)
        self._backend = self.p.indicator_type
        self._talib_result = None
        try:
            if self._backend == "bt":
                self._impl = bt.indicators.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.devfactor)
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.BBANDS(self.data.close, timeperiod=self.p.period, nbdevup=self.p.devfactor, nbdevdn=self.p.devfactor)
            elif self._backend == "talib":
                try:
                    import talib
                    import numpy as np
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                close = np.array(self.data.close.get(size=len(self.data)))
                upper, middle, lower = talib.BBANDS(close, timeperiod=self.p.period, nbdevup=self.p.devfactor, nbdevdn=self.p.devfactor)
                self._talib_result = (upper, middle, lower)
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.exception("Error initializing BollingerBandIndicator: Falling back to bt.indicators.BollingerBands")
            self._impl = bt.indicators.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.devfactor)
            self._backend = "bt"

    def next(self):
        upper, middle, lower = self._line_names
        if self._backend == "bt":
            self.lines[upper][0] = self._impl.lines.top[0]
            self.lines[middle][0] = self._impl.lines.mid[0]
            self.lines[lower][0] = self._impl.lines.bot[0]
        elif self._backend == "bt-talib":
            self.lines[upper][0] = self._impl.lines.upperband[0]
            self.lines[middle][0] = self._impl.lines.middleband[0]
            self.lines[lower][0] = self._impl.lines.lowerband[0]
        elif self._backend == "talib":
            if self._talib_result is not None and len(self) - 1 < len(self._talib_result[0]):
                self.lines[upper][0] = self._talib_result[0][len(self) - 1]
                self.lines[middle][0] = self._talib_result[1][len(self) - 1]
                self.lines[lower][0] = self._talib_result[2][len(self) - 1]
            else:
                self.lines[upper][0] = float("nan")
                self.lines[middle][0] = float("nan")
                self.lines[lower][0] = float("nan")