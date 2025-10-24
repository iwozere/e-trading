"""
Simplified Backtrader Indicator Wrappers for Unified Service

This module provides simplified Backtrader indicator wrappers that use
only the unified indicator service without fallback mechanisms.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from typing import Dict, Any
import asyncio

from src.indicators.adapters.backtrader_adapter import BacktraderIndicatorWrapper
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class UnifiedRSIIndicator(BacktraderIndicatorWrapper):
    """RSI indicator using unified service"""

    lines = ("rsi",)
    params = (
        ("period", 14),
        ("backend", "bt"),
        ("use_unified_service", True),
    )

    def _init_fallback(self):
        """Initialize - simplified version uses only unified service"""
        self.addminperiod(self.p.period)

    def _get_indicator_name(self) -> str:
        return "rsi"

    def _get_indicator_params(self) -> Dict[str, Any]:
        return {"period": self.p.period}

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to RSI line"""
        if "rsi" in results and len(results["rsi"]) > 0:
            latest_value = results["rsi"].iloc[-1]
            if not pd.isna(latest_value):
                self.lines.rsi[0] = float(latest_value)
            else:
                self.lines.rsi[0] = float("nan")
        else:
            self.lines.rsi[0] = float("nan")

    def _use_fallback(self):
        """Fallback not implemented in simplified version"""
        self.lines.rsi[0] = float("nan")


class UnifiedBollingerBandsIndicator(BacktraderIndicatorWrapper):
    """Bollinger Bands indicator using unified service"""

    lines = ("upper", "middle", "lower")
    params = (
        ("period", 20),
        ("devfactor", 2.0),
        ("backend", "bt"),
        ("use_unified_service", True),
    )

    def _init_fallback(self):
        """Initialize - simplified version uses only unified service"""
        self.addminperiod(self.p.period)

    def _get_indicator_name(self) -> str:
        return "bollinger_bands"

    def _get_indicator_params(self) -> Dict[str, Any]:
        return {
            "period": self.p.period,
            "devfactor": self.p.devfactor
        }

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to Bollinger Bands lines"""
        if all(key in results for key in ["upper", "middle", "lower"]):
            upper_val = results["upper"].iloc[-1] if len(results["upper"]) > 0 else float("nan")
            middle_val = results["middle"].iloc[-1] if len(results["middle"]) > 0 else float("nan")
            lower_val = results["lower"].iloc[-1] if len(results["lower"]) > 0 else float("nan")

            self.lines.upper[0] = float(upper_val) if not pd.isna(upper_val) else float("nan")
            self.lines.middle[0] = float(middle_val) if not pd.isna(middle_val) else float("nan")
            self.lines.lower[0] = float(lower_val) if not pd.isna(lower_val) else float("nan")
        else:
            self.lines.upper[0] = float("nan")
            self.lines.middle[0] = float("nan")
            self.lines.lower[0] = float("nan")

    def _use_fallback(self):
        """Fallback not implemented in simplified version"""
        self.lines.upper[0] = float("nan")
        self.lines.middle[0] = float("nan")
        self.lines.lower[0] = float("nan")


class UnifiedMACDIndicator(BacktraderIndicatorWrapper):
    """MACD indicator using unified service"""

    lines = ("macd", "signal", "histogram")
    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("backend", "bt"),
        ("use_unified_service", True),
    )

    def _init_fallback(self):
        """Initialize - simplified version uses only unified service"""
        self.addminperiod(self.p.slow_period + self.p.signal_period)

    def _get_indicator_name(self) -> str:
        return "macd"

    def _get_indicator_params(self) -> Dict[str, Any]:
        return {
            "fast_period": self.p.fast_period,
            "slow_period": self.p.slow_period,
            "signal_period": self.p.signal_period
        }

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to MACD lines"""
        if all(key in results for key in ["macd", "signal", "histogram"]):
            macd_val = results["macd"].iloc[-1] if len(results["macd"]) > 0 else float("nan")
            signal_val = results["signal"].iloc[-1] if len(results["signal"]) > 0 else float("nan")
            hist_val = results["histogram"].iloc[-1] if len(results["histogram"]) > 0 else float("nan")

            self.lines.macd[0] = float(macd_val) if not pd.isna(macd_val) else float("nan")
            self.lines.signal[0] = float(signal_val) if not pd.isna(signal_val) else float("nan")
            self.lines.histogram[0] = float(hist_val) if not pd.isna(hist_val) else float("nan")
        else:
            self.lines.macd[0] = float("nan")
            self.lines.signal[0] = float("nan")
            self.lines.histogram[0] = float("nan")

    def _use_fallback(self):
        """Fallback not implemented in simplified version"""
        self.lines.macd[0] = float("nan")
        self.lines.signal[0] = float("nan")
        self.lines.histogram[0] = float("nan")


class UnifiedATRIndicator(BacktraderIndicatorWrapper):
    """ATR indicator using unified service"""

    lines = ("atr",)
    params = (
        ("period", 14),
        ("backend", "bt"),
        ("use_unified_service", True),
    )

    def _init_fallback(self):
        """Initialize - simplified version uses only unified service"""
        self.addminperiod(self.p.period)

    def _get_indicator_name(self) -> str:
        return "atr"

    def _get_indicator_params(self) -> Dict[str, Any]:
        return {"period": self.p.period}

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to ATR line"""
        if "atr" in results and len(results["atr"]) > 0:
            latest_value = results["atr"].iloc[-1]
            if not pd.isna(latest_value):
                self.lines.atr[0] = float(latest_value)
            else:
                self.lines.atr[0] = float("nan")
        else:
            self.lines.atr[0] = float("nan")

    def _use_fallback(self):
        """Fallback not implemented in simplified version"""
        self.lines.atr[0] = float("nan")


class UnifiedSMAIndicator(BacktraderIndicatorWrapper):
    """Simple Moving Average indicator using unified service"""

    lines = ("sma",)
    params = (
        ("period", 20),
        ("backend", "bt"),
        ("use_unified_service", True),
    )

    def _init_fallback(self):
        """Initialize - simplified version uses only unified service"""
        self.addminperiod(self.p.period)

    def _get_indicator_name(self) -> str:
        return "sma"

    def _get_indicator_params(self) -> Dict[str, Any]:
        return {"period": self.p.period}

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to SMA line"""
        if "sma" in results and len(results["sma"]) > 0:
            latest_value = results["sma"].iloc[-1]
            if not pd.isna(latest_value):
                self.lines.sma[0] = float(latest_value)
            else:
                self.lines.sma[0] = float("nan")
        else:
            self.lines.sma[0] = float("nan")

    def _use_fallback(self):
        """Fallback not implemented in simplified version"""
        self.lines.sma[0] = float("nan")


class UnifiedEMAIndicator(BacktraderIndicatorWrapper):
    """Exponential Moving Average indicator using unified service"""

    lines = ("ema",)
    params = (
        ("period", 20),
        ("backend", "bt"),
        ("use_unified_service", True),
    )

    def _init_fallback(self):
        """Initialize - simplified version uses only unified service"""
        self.addminperiod(self.p.period)

    def _get_indicator_name(self) -> str:
        return "ema"

    def _get_indicator_params(self) -> Dict[str, Any]:
        return {"period": self.p.period}

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to EMA line"""
        if "ema" in results and len(results["ema"]) > 0:
            latest_value = results["ema"].iloc[-1]
            if not pd.isna(latest_value):
                self.lines.ema[0] = float(latest_value)
            else:
                self.lines.ema[0] = float("nan")
        else:
            self.lines.ema[0] = float("nan")

    def _use_fallback(self):
        """Fallback not implemented in simplified version"""
        self.lines.ema[0] = float("nan")


class UnifiedSuperTrendIndicator(BacktraderIndicatorWrapper):
    """SuperTrend indicator using unified service"""

    lines = ("super_trend", "direction")
    params = (
        ("length", 10),
        ("multiplier", 3.0),
        ("backend", "bt"),
        ("use_unified_service", True),
    )

    def _init_fallback(self):
        """Initialize - simplified version uses only unified service"""
        self.addminperiod(self.p.length)

    def _get_indicator_name(self) -> str:
        return "super_trend"

    def _get_indicator_params(self) -> Dict[str, Any]:
        return {"length": self.p.length, "multiplier": self.p.multiplier}

    def _map_unified_results(self, results: Dict[str, pd.Series]):
        """Map unified service results to SuperTrend lines"""
        if "value" in results and len(results["value"]) > 0:
            latest_value = results["value"].iloc[-1]
            if not pd.isna(latest_value):
                self.lines.super_trend[0] = float(latest_value)
            else:
                self.lines.super_trend[0] = float("nan")
        else:
            self.lines.super_trend[0] = float("nan")

        if "trend" in results and len(results["trend"]) > 0:
            latest_trend = results["trend"].iloc[-1]
            if not pd.isna(latest_trend):
                self.lines.direction[0] = float(latest_trend)
            else:
                self.lines.direction[0] = float("nan")
        else:
            self.lines.direction[0] = float("nan")

    def _use_fallback(self):
        """Fallback not implemented in simplified version"""
        self.lines.super_trend[0] = float("nan")
        self.lines.direction[0] = float("nan")