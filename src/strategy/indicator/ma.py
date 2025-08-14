"""
Backtrader-Compatible MA Indicator Wrapper

This module provides a Backtrader-compatible wrapper for Moving Average (MA) indicator.
Supports multiple backends: bt, bt-talib, talib. Supports SMA and EMA.

Note on TA-Lib vs Backtrader calculation:
- When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
- When using the 'talib' type, the entire indicator array is precalculated in __init__ and values are assigned in next().
- For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class MaIndicator(bt.Indicator):
    """
    Moving Average indicator wrapper for Backtrader.

    Parameters:
    -----------
    period : int
        The period for MA calculation (default: 20)
    ma_type : str
        The type of moving average ('sma' or 'ema', default: 'sma')
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')

    Note on TA-Lib vs Backtrader calculation:
    - When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
    - When using the 'talib' type, the entire indicator array is precalculated in __init__ and values are assigned in next().
    - For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
    """
    lines = ("sma", "ema")
    params = (
        ("period", 20),
        ("ma_type", "sma"),
        ("indicator_type", "bt"),
    )

    def __init__(self):
        super().__init__()
        self.addminperiod(self.p.period)
        self._backend = self.p.indicator_type
        self._ma_type = self.p.ma_type
        self._talib_result = None
        try:
            if self._backend == "bt":
                if self._ma_type == "sma":
                    self._impl = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.period)
                elif self._ma_type == "ema":
                    self._impl = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.period)
                else:
                    raise ValueError(f"Unknown ma_type: {self._ma_type}")
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                if self._ma_type == "sma":
                    self._impl = bt.talib.SMA(self.data.close, timeperiod=self.p.period)
                elif self._ma_type == "ema":
                    self._impl = bt.talib.EMA(self.data.close, timeperiod=self.p.period)
                else:
                    raise ValueError(f"Unknown ma_type: {self._ma_type}")
            elif self._backend == "talib":
                try:
                    import talib
                    import numpy as np
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                close = np.array(self.data.close.get(size=len(self.data)))
                if self._ma_type == "sma":
                    self._talib_result = talib.SMA(close, timeperiod=self.p.period)
                elif self._ma_type == "ema":
                    self._talib_result = talib.EMA(close, timeperiod=self.p.period)
                else:
                    raise ValueError(f"Unknown ma_type: {self._ma_type}")
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.exception("Error initializing MaIndicator: Falling back to bt.indicators.SimpleMovingAverage")
            self._impl = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.period)
            self._backend = "bt"
            self._ma_type = "sma"

    def next(self):
        if self._backend in ["bt", "bt-talib"]:
            if self._ma_type == "sma":
                self.lines.sma[0] = self._impl[0]
            elif self._ma_type == "ema":
                self.lines.ema[0] = self._impl[0]
        elif self._backend == "talib":
            if self._talib_result is not None and len(self) - 1 < len(self._talib_result):
                if self._ma_type == "sma":
                    self.lines.sma[0] = self._talib_result[len(self) - 1]
                elif self._ma_type == "ema":
                    self.lines.ema[0] = self._talib_result[len(self) - 1]
            else:
                if self._ma_type == "sma":
                    self.lines.sma[0] = float("nan")
                elif self._ma_type == "ema":
                    self.lines.ema[0] = float("nan")