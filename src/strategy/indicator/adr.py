"""
Backtrader-Compatible ADR Indicator Wrapper

This module provides a Backtrader-compatible wrapper for ADR (Average Daily Range) indicator.
Supports bt and custom calculation. TA-Lib does not provide ADR.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class AdrIndicator(bt.Indicator):
    """
    ADR indicator wrapper for Backtrader.

    Parameters:
    -----------
    period : int
        The period for ADR calculation (default: 14)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'talib')
    line_names : tuple or None
        Custom names for the indicator lines (default: None, uses 'adr' as the only line)

    Note on TA-Lib vs Backtrader calculation:
    - When using Backtrader's built-in indicators ('bt' or 'bt-talib' types), the indicator is calculated incrementally and efficiently, updating only with each new bar.
    - When using the 'talib' type, the entire indicator array is recalculated on every `next()` call, as TA-Lib expects array inputs and does not support incremental updates. This is less efficient for large datasets or live trading.
    - For backtesting, where the full dataset is loaded in memory, it is recommended to pre-calculate indicators (e.g., with pandas or TA-Lib) and feed them as custom data lines for best performance, especially during optimization.
    """
    lines = ("adr",)
    params = (
        ("period", 14),
        ("indicator_type", "bt"),
        ("line_names", None),
    )

    def __init__(self):
        # Handle custom line names
        if self.p.line_names is not None:
            self.lines = type(self.lines)(*self.p.line_names)
            self._line_names = self.p.line_names
        else:
            self._line_names = ("adr",)
        super().__init__()
        self.addminperiod(self.p.period)
        self._backend = self.p.indicator_type
        if self._backend not in ["bt", "talib"]:
            raise ValueError(f"Unknown indicator_type: {self._backend}")
        self._ranges = []

    def next(self):
        (adr,) = self._line_names
        # ADR = average of (high - low) over period
        self._ranges.append(float(self.data.high[0]) - float(self.data.low[0]))
        if len(self._ranges) > self.p.period:
            self._ranges.pop(0)
        if len(self._ranges) == self.p.period:
            self.lines[adr][0] = sum(self._ranges) / self.p.period
        else:
            self.lines[adr][0] = float("nan") 
