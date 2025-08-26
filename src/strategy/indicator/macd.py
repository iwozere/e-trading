"""
Backtrader-Compatible MACD Indicator Wrapper

This module provides a Backtrader-compatible wrapper for MACD indicator.
Supports multiple backends: bt, bt-talib, talib.

Note on TA-Lib vs Backtrader calculation:
- When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
- When using the 'talib' type, the entire indicator array is recalculated on every `next()` call, as TA-Lib expects array inputs and does not support incremental updates. This is less efficient for large datasets or live trading.
- For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class MacdIndicator(bt.Indicator):
    """
    MACD indicator wrapper for Backtrader.

    Parameters:
    -----------
    fast_period : int
        The fast EMA period (default: 12)
    slow_period : int
        The slow EMA period (default: 26)
    signal_period : int
        The signal line EMA period (default: 9)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')
    """
    lines = ("macd", "signal", "histo")
    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("indicator_type", "bt"),
    )

    def __init__(self):
        super().__init__()
        self.addminperiod(self.p.slow_period + self.p.signal_period)
        self._backend = self.p.indicator_type
        self._talib_result = None
        try:
            if self._backend == "bt":
                self._impl = bt.indicators.MACD(
                    self.data.close,
                    period_me1=self.p.fast_period,
                    period_me2=self.p.slow_period,
                    period_signal=self.p.signal_period,
                )
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.MACD(
                    self.data.close,
                    fastperiod=self.p.fast_period,
                    slowperiod=self.p.slow_period,
                    signalperiod=self.p.signal_period,
                )
            elif self._backend == "talib":
                try:
                    import talib
                    import numpy as np
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                close = np.array(self.data.close.get(size=len(self.data)))
                macd, signal, hist = talib.MACD(
                    close,
                    fastperiod=self.p.fast_period,
                    slowperiod=self.p.slow_period,
                    signalperiod=self.p.signal_period,
                )
                self._talib_result = (macd, signal, hist)
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.exception("Error initializing MacdIndicator: Falling back to bt.indicators.MACD")
            self._impl = bt.indicators.MACD(
                self.data.close,
                period_me1=self.p.fast_period,
                period_me2=self.p.slow_period,
                period_signal=self.p.signal_period,
            )
            self._backend = "bt"

    def next(self):
        if self._backend == "bt":
            self.lines.macd[0] = self._impl.lines.macd[0] if hasattr(self._impl.lines, "macd") else float("nan")
            self.lines.signal[0] = self._impl.lines.signal[0] if hasattr(self._impl.lines, "signal") else float("nan")
            self.lines.histo[0] = self._impl.lines.histo[0] if hasattr(self._impl.lines, "histo") else float("nan")
        elif self._backend == "bt-talib":
            self.lines.macd[0] = self._impl.lines.macd[0]
            self.lines.signal[0] = self._impl.lines.macdsignal[0]
            self.lines.histo[0] = self._impl.lines.macdhist[0]
        elif self._backend == "talib":
            if self._talib_result is not None and len(self) - 1 < len(self._talib_result[0]):
                self.lines.macd[0] = self._talib_result[0][len(self) - 1]
                self.lines.signal[0] = self._talib_result[1][len(self) - 1]
                self.lines.histo[0] = self._talib_result[2][len(self) - 1]
            else:
                self.lines.macd[0] = float("nan")
                self.lines.signal[0] = float("nan")
                self.lines.histo[0] = float("nan")
