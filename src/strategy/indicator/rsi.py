"""
Backtrader-Compatible RSI Indicator Wrapper

This module provides a Backtrader-compatible wrapper for RSI indicator.
Supports multiple backends: bt, bt-talib, talib.

Note on TA-Lib vs Backtrader calculation:
- When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
- When using the 'talib' type, the entire indicator array is recalculated on every `next()` call, as TA-Lib expects array inputs and does not support incremental updates. This is less efficient for large datasets or live trading.
- For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
"""

import backtrader as bt
from src.notification.logger import setup_logger
import numpy as np  # Added for TA-Lib compatibility

logger = setup_logger(__name__)

class RsiIndicator(bt.Indicator):
    """
    RSI indicator wrapper for Backtrader.

    Parameters:
    -----------
    period : int
        The period for RSI calculation (default: 14)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')
    line_names : tuple, optional
        Custom names for the indicator lines (default: None, uses default names)
    """
    lines = ("rsi",)
    params = (
        ("period", 14),
        ("indicator_type", "bt"),
    )

    def __init__(self):
        super().__init__()
        self.addminperiod(self.p.period)
        self._backend = self.p.indicator_type
        try:
            if self._backend == "bt":
                self._impl = bt.indicators.RSI(self.data.close, period=self.p.period)
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.RSI(self.data.close, timeperiod=self.p.period)
            elif self._backend == "talib":
                try:
                    import talib
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                # talib expects numpy arrays, but Backtrader provides line objects
                # We'll update in next()
                self._talib = talib
                self._talib_cache = []
                self._impl = None
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.error("Error initializing RsiIndicator: %s. Falling back to bt.indicators.RSI", e, exc_info=True)
            self._impl = bt.indicators.RSI(self.data.close, period=self.p.period)
            self._backend = "bt"

    def next(self):
        if self._backend in ["bt", "bt-talib"]:
            self.lines.rsi[0] = self._impl[0]
        elif self._backend == "talib":
            # Build up a cache of closes for talib
            self._talib_cache.append(float(self.data.close[0]))
            if len(self._talib_cache) >= self.p.period:
                arr = np.array(self._talib_cache[-len(self._talib_cache):])
                rsi_arr = self._talib.RSI(
                    arr, timeperiod=self.p.period
                )
                self.lines.rsi[0] = rsi_arr[-1] if rsi_arr is not None else float("nan")
            else:
                self.lines.rsi[0] = float("nan")
