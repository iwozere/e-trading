"""
Backtrader-Compatible Stochastic Indicator Wrapper

This module provides a Backtrader-compatible wrapper for Stochastic indicator.
Supports multiple backends: bt, bt-talib, talib.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class StochasticIndicator(bt.Indicator):
    """
    Stochastic Oscillator indicator wrapper for Backtrader.

    Parameters:
    -----------
    k_period : int
        The period for %K calculation (default: 14)
    d_period : int
        The period for %D calculation (default: 3)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')
    line_names : tuple
        Custom line names for the indicator (default: None)

    Note on TA-Lib vs Backtrader calculation:
    - When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
    - When using the 'talib' type, the entire indicator array is precalculated in __init__ and values are assigned in next().
    - For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
    """
    lines = ("k", "d")
    params = (
        ("k_period", 14),
        ("d_period", 3),
        ("indicator_type", "bt"),
        ("line_names", None),
    )

    def __init__(self):
        # Handle custom line names
        if self.p.line_names is not None:
            self.lines = type(self.lines)(*self.p.line_names)
            self._line_names = self.p.line_names
        else:
            self._line_names = ("k", "d")
        super().__init__()
        self.addminperiod(self.p.k_period)
        self._backend = self.p.indicator_type
        self._talib_result = None
        try:
            if self._backend == "bt":
                self._impl = bt.indicators.Stochastic(
                    self.data,
                    period=self.p.k_period,
                    period_dfast=self.p.d_period,
                )
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.STOCH(
                    self.data.high, self.data.low, self.data.close,
                    fastk_period=self.p.k_period, slowk_period=self.p.d_period, slowd_period=self.p.d_period
                )
            elif self._backend == "talib":
                try:
                    import talib
                    import numpy as np
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                high = np.array(self.data.high.get(size=len(self.data)))
                low = np.array(self.data.low.get(size=len(self.data)))
                close = np.array(self.data.close.get(size=len(self.data)))
                k, d = talib.STOCH(high, low, close, fastk_period=self.p.k_period, slowk_period=self.p.d_period, slowd_period=self.p.d_period)
                self._talib_result = (k, d)
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.exception("Error initializing StochasticIndicator: Falling back to bt.indicators.Stochastic")
            self._impl = bt.indicators.Stochastic(
                self.data,
                period=self.p.k_period,
                period_dfast=self.p.d_period,
            )
            self._backend = "bt"

    def next(self):
        k, d = self._line_names
        if self._backend == "bt":
            self.lines[k][0] = self._impl.lines.percK[0]
            self.lines[d][0] = self._impl.lines.percD[0]
        elif self._backend == "bt-talib":
            self.lines[k][0] = self._impl.lines.slowk[0]
            self.lines[d][0] = self._impl.lines.slowd[0]
        elif self._backend == "talib":
            if self._talib_result is not None and len(self) - 1 < len(self._talib_result[0]):
                self.lines[k][0] = self._talib_result[0][len(self) - 1]
                self.lines[d][0] = self._talib_result[1][len(self) - 1]
            else:
                self.lines[k][0] = float("nan")
                self.lines[d][0] = float("nan")
