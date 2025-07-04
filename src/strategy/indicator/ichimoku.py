"""
Backtrader-Compatible Ichimoku Indicator Wrapper

This module provides a Backtrader-compatible wrapper for Ichimoku indicator.
Supports multiple backends: bt, bt-talib, talib.
"""

import backtrader as bt
from src.notification.logger import setup_logger

logger = setup_logger(__name__)

class IchimokuIndicator(bt.Indicator):
    """
    Ichimoku indicator wrapper for Backtrader.

    Parameters:
    -----------
    tenkan_period : int
        The period for Tenkan-sen (default: 9)
    kijun_period : int
        The period for Kijun-sen (default: 26)
    senkou_span_b_period : int
        The period for Senkou Span B (default: 52)
    indicator_type : str
        The type of indicator to use (default: 'bt', values: 'bt', 'bt-talib', 'talib')
    line_names : tuple
        Custom names for the indicator lines (default: None)
    """
    lines = ("tenkan", "kijun", "senkou_a", "senkou_b", "chikou")
    params = (
        ("tenkan_period", 9),
        ("kijun_period", 26),
        ("senkou_span_b_period", 52),
        ("indicator_type", "bt"),
        ("line_names", None),
    )

    def __init__(self):
        # Handle custom line names
        if self.p.line_names is not None:
            self.lines = type(self.lines)(*self.p.line_names)
            self._line_names = self.p.line_names
        else:
            self._line_names = ("tenkan", "kijun", "senkou_a", "senkou_b", "chikou")
        super().__init__()
        self.addminperiod(self.p.senkou_span_b_period)
        self._backend = self.p.indicator_type
        try:
            if self._backend == "bt":
                self._impl = bt.indicators.Ichimoku(
                    self.data,
                    tenkan=self.p.tenkan_period,
                    kijun=self.p.kijun_period,
                    senkou=self.p.senkou_span_b_period,
                )
            elif self._backend == "bt-talib":
                if not hasattr(bt, "talib"):
                    raise ImportError("Backtrader TA-Lib integration (bt.talib) not available.")
                self._impl = bt.talib.ICHIMOKU(
                    self.data,
                    tenkan=self.p.tenkan_period,
                    kijun=self.p.kijun_period,
                    senkou=self.p.senkou_span_b_period,
                )
            elif self._backend == "talib":
                try:
                    import talib
                except ImportError:
                    raise ImportError("TA-Lib is not installed.")
                self._talib = talib
                self._talib_cache = {"high": [], "low": [], "close": []}
                self._impl = None
            else:
                raise ValueError(f"Unknown indicator_type: {self._backend}")
        except Exception as e:
            logger.error(f"Error initializing IchimokuIndicator: {e}. Falling back to bt.indicators.Ichimoku", exc_info=e)
            self._impl = bt.indicators.Ichimoku(
                self.data,
                tenkan=self.p.tenkan_period,
                kijun=self.p.kijun_period,
                senkou=self.p.senkou_span_b_period,
            )
            self._backend = "bt"

    def next(self):
        tenkan, kijun, senkou_a, senkou_b, chikou = self._line_names
        if self._backend in ["bt", "bt-talib"]:
            self.lines[tenkan][0] = getattr(self._impl.lines, "tenkan_sen", float("nan"))[0] if hasattr(self._impl.lines, "tenkan_sen") else float("nan")
            self.lines[kijun][0] = getattr(self._impl.lines, "kijun_sen", float("nan"))[0] if hasattr(self._impl.lines, "kijun_sen") else float("nan")
            self.lines[senkou_a][0] = getattr(self._impl.lines, "senkou_span_a", float("nan"))[0] if hasattr(self._impl.lines, "senkou_span_a") else float("nan")
            self.lines[senkou_b][0] = getattr(self._impl.lines, "senkou_span_b", float("nan"))[0] if hasattr(self._impl.lines, "senkou_span_b") else float("nan")
            self.lines[chikou][0] = getattr(self._impl.lines, "chikou_span", float("nan"))[0] if hasattr(self._impl.lines, "chikou_span") else float("nan")
        elif self._backend == "talib":
            self.lines[tenkan][0] = float("nan")
            self.lines[kijun][0] = float("nan")
            self.lines[senkou_a][0] = float("nan")
            self.lines[senkou_b][0] = float("nan")
            self.lines[chikou][0] = float("nan") 