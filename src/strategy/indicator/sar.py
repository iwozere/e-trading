"""
Backtrader-Compatible SAR Indicator Wrapper

This module provides a Backtrader-compatible wrapper for SAR (Parabolic SAR) indicator.
Supports multiple backends: bt, bt-talib, talib.

Note on TA-Lib vs Backtrader calculation:
- When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
- When using the 'talib' type, the entire indicator array is precalculated in __init__ and values are assigned in next().
- For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class SarIndicator(bt.Indicator):
    """
    SAR indicator wrapper for Backtrader.

    Parameters:
    -----------
    acceleration : float
        The acceleration factor (default: 0.02)
    maximum : float
        The maximum value for acceleration (default: 0.2)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')
    line_names : tuple
        Custom line names for the indicator (default: None)
    """
    lines = ("sar",)
    params = (
        ("acceleration", 0.02),
        ("maximum", 0.2),
        ("indicator_type", "bt"),
        ("line_names", None),
    )

    def __init__(self):
        # Handle custom line names
        if self.p.line_names is not None:
            self.lines = type(self.lines)(*self.p.line_names)
            self._line_names = self.p.line_names
        else:
            self._line_names = ("sar",)
        super().__init__()
        self._backend = self.p.indicator_type
        self._talib_result = None
        try:
            if self._backend == "bt":
                self._impl = bt.indicators.ParabolicSAR(
                    self.data,
                    af=self.p.acceleration,
                    afmax=self.p.maximum,
                )
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.SAR(
                    self.data.high, self.data.low,
                    acceleration=self.p.acceleration, maximum=self.p.maximum
                )
            elif self._backend == "talib":
                try:
                    import talib
                    import numpy as np
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                high = np.array(self.data.high.get(size=len(self.data)))
                low = np.array(self.data.low.get(size=len(self.data)))
                self._talib_result = talib.SAR(high, low, acceleration=self.p.acceleration, maximum=self.p.maximum)
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.error("Error initializing SarIndicator: %s. Falling back to bt.indicators.ParabolicSAR", e, exc_info=True)
            self._impl = bt.indicators.ParabolicSAR(
                self.data,
                af=self.p.acceleration,
                afmax=self.p.maximum,
            )
            self._backend = "bt"

    def next(self):
        (sar,) = self._line_names
        if self._backend in ["bt", "bt-talib"]:
            self.lines[sar][0] = self._impl[0]
        elif self._backend == "talib":
            if self._talib_result is not None and len(self) - 1 < len(self._talib_result):
                self.lines[sar][0] = self._talib_result[len(self) - 1]
            else:
                self.lines[sar][0] = float("nan") 