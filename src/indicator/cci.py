"""
Backtrader-Compatible CCI Indicator Wrapper

This module provides a Backtrader-compatible wrapper for CCI indicator.
Supports multiple backends: bt, bt-talib, talib.

Note on TA-Lib vs Backtrader calculation:
- When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
- When using the 'talib' type, the entire indicator array is precalculated in __init__ and values are assigned in next().
- For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class CciIndicator(bt.Indicator):
    """
    CCI indicator wrapper for Backtrader.

    Parameters:
    -----------
    period : int
        The period for CCI calculation (default: 20)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')
    line_names : tuple
        Custom line names for the indicator (default: None)
    """
    lines = ("cci",)
    params = (
        ("period", 20),
        ("indicator_type", "bt"),
        ("line_names", None),
    )

    def __init__(self):
        # Handle custom line names
        if self.p.line_names is not None:
            self.lines = type(self.lines)(*self.p.line_names)
            self._line_names = self.p.line_names
        else:
            self._line_names = ("cci",)
        super().__init__()
        self.addminperiod(self.p.period)
        self._backend = self.p.indicator_type
        self._talib_result = None
        try:
            if self._backend == "bt":
                self._impl = bt.indicators.CCI(self.data, period=self.p.period)
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.CCI(self.data.high, self.data.low, self.data.close, timeperiod=self.p.period)
            elif self._backend == "talib":
                try:
                    import talib
                    import numpy as np
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                high = np.array(self.data.high.get(size=len(self.data)))
                low = np.array(self.data.low.get(size=len(self.data)))
                close = np.array(self.data.close.get(size=len(self.data)))
                self._talib_result = talib.CCI(high, low, close, timeperiod=self.p.period)
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.error(f"Error initializing CciIndicator: {e}. Falling back to bt.indicators.CCI", exc_info=e)
            self._impl = bt.indicators.CCI(self.data, period=self.p.period)
            self._backend = "bt"

    def next(self):
        (cci,) = self._line_names
        if self._backend in ["bt", "bt-talib"]:
            self.lines[cci][0] = self._impl[0]
        elif self._backend == "talib":
            if self._talib_result is not None and len(self) - 1 < len(self._talib_result):
                self.lines[cci][0] = self._talib_result[len(self) - 1]
            else:
                self.lines[cci][0] = float("nan") 