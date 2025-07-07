"""
Backtrader-Compatible OBV Indicator Wrapper

This module provides a Backtrader-compatible wrapper for OBV indicator.
Supports multiple backends: bt, bt-talib, talib.

Note on TA-Lib vs Backtrader calculation:
- When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
- When using the 'talib' type, the entire indicator array is recalculated on every `next()` call, as TA-Lib expects array inputs and does not support incremental updates. This is less efficient for large datasets or live trading.
- For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class ObvIndicator(bt.Indicator):
    """
    OBV indicator wrapper for Backtrader.

    Parameters:
    -----------
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt-talib', 'talib')
    line_names : tuple, optional
        Custom names for the indicator's lines (default: None, uses default names)

    Note on TA-Lib vs Backtrader calculation:
    - When using the 'talib' type, the entire indicator array is precalculated in __init__ and values are assigned in next().
    - For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
    """
    lines = ("obv",)
    params = (
        ("indicator_type", "talib"),
        ("line_names", None),
    )

    def __init__(self):
        # Handle custom line names
        if self.p.line_names is not None:
            self.lines = type(self.lines)(*self.p.line_names)
            self._line_names = self.p.line_names
        else:
            self._line_names = ("obv",)
        super().__init__()
        self._backend = self.p.indicator_type
        self._talib_result = None
        try:
            if self._backend == "bt":
                raise NotImplementedError("Backtrader does not have a built-in OBV indicator. Use 'talib' or 'bt-talib'.")
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.OBV(self.data.close, self.data.volume)
            elif self._backend == "talib":
                try:
                    import talib
                    import numpy as np
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                close = np.array(self.data.close.get(size=len(self.data)))
                volume = np.array(self.data.volume.get(size=len(self.data)))
                self._talib_result = talib.OBV(close, volume)
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.error("Error initializing ObvIndicator: %s.", e, exc_info=True)
            raise

    def next(self):
        (obv,) = self._line_names
        if self._backend == "bt-talib":
            self.lines[obv][0] = self._impl[0]
        elif self._backend == "talib":
            if self._talib_result is not None and len(self) - 1 < len(self._talib_result):
                self.lines[obv][0] = self._talib_result[len(self) - 1]
            else:
                self.lines[obv][0] = float("nan") 