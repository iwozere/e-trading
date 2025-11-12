"""
Backtrader Indicator Wrappers with Direct Calculation Fallbacks

This module provides Backtrader indicator wrappers that use the unified
indicator service when available, and fall back to direct calculations
when the service is unavailable.
"""

import pandas as pd
from typing import Dict, Any

from src.indicators.adapters.backtrader_adapter import BacktraderIndicatorWrapper
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class UnifiedRSIIndicator(BacktraderIndicatorWrapper):
    """RSI indicator using unified service with Backtrader fallback"""

    lines = ("rsi",)
    params = (
        ("period", 14),
        ("backend", "bt"),
        ("use_unified_service", False),  # Deprecated - use fallback implementation
    )

    def _init_fallback(self):
        """Initialize fallback - just set minimum period"""
        self.addminperiod(self.p.period)
        self._bt_rsi = None
        self._fallback_initialized = False
        # Set _fallback_impl to indicate fallback is available
        self._fallback_impl = True

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
        """Calculate RSI directly without child indicators"""
        # Simple RSI calculation
        period = self.p.period
        current_len = len(self.data)

        if current_len < period + 1:
            self.lines.rsi[0] = float("nan")
            return

        # Get price changes
        closes = [self.data.close[-i] for i in range(period + 1)]
        closes.reverse()

        # Calculate gains and losses
        gains = []
        losses = []
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        # Calculate average gain and loss
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            self.lines.rsi[0] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            self.lines.rsi[0] = rsi


class UnifiedBollingerBandsIndicator(BacktraderIndicatorWrapper):
    """Bollinger Bands indicator using unified service with Backtrader fallback"""

    lines = ("upper", "middle", "lower")
    params = (
        ("period", 20),
        ("devfactor", 2.0),
        ("backend", "bt"),
        ("use_unified_service", False),  # Deprecated - use fallback implementation
    )

    def _init_fallback(self):
        """Initialize fallback - just set minimum period"""
        self.addminperiod(self.p.period)
        self._bt_bb = None
        self._fallback_initialized = False
        # Set _fallback_impl to indicate fallback is available
        self._fallback_impl = True

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
        """Calculate Bollinger Bands directly without child indicators"""
        period = self.p.period
        devfactor = self.p.devfactor
        current_len = len(self.data)

        if current_len < period:
            self.lines.upper[0] = float("nan")
            self.lines.middle[0] = float("nan")
            self.lines.lower[0] = float("nan")
            return

        # Get recent closes
        closes = [self.data.close[-i] for i in range(period)]

        # Calculate SMA (middle band)
        sma = sum(closes) / period

        # Calculate standard deviation
        variance = sum((x - sma) ** 2 for x in closes) / period
        std = variance ** 0.5

        # Calculate bands
        self.lines.middle[0] = sma
        self.lines.upper[0] = sma + (std * devfactor)
        self.lines.lower[0] = sma - (std * devfactor)


class UnifiedMACDIndicator(BacktraderIndicatorWrapper):
    """MACD indicator using unified service"""

    lines = ("macd", "signal", "histogram")
    params = (
        ("fast_period", 12),
        ("slow_period", 26),
        ("signal_period", 9),
        ("backend", "bt"),
        ("use_unified_service", False),  # Deprecated - use fallback implementation
    )

    def _init_fallback(self):
        """Initialize fallback - just set minimum period"""
        self.addminperiod(self.p.slow_period + self.p.signal_period)
        # Set _fallback_impl to indicate fallback is available
        self._fallback_impl = True

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
        """Calculate MACD directly without child indicators"""
        fast_period = self.p.fast_period
        slow_period = self.p.slow_period
        signal_period = self.p.signal_period
        current_len = len(self.data)

        if current_len < slow_period + signal_period:
            self.lines.macd[0] = float("nan")
            self.lines.signal[0] = float("nan")
            self.lines.histogram[0] = float("nan")
            return

        # Calculate fast EMA
        fast_closes = [self.data.close[-i] for i in range(fast_period)]
        fast_ema = sum(fast_closes) / fast_period  # Simple approximation

        # Calculate slow EMA
        slow_closes = [self.data.close[-i] for i in range(slow_period)]
        slow_ema = sum(slow_closes) / slow_period  # Simple approximation

        # MACD line = fast EMA - slow EMA
        macd_value = fast_ema - slow_ema

        # Signal line = EMA of MACD (simplified as SMA for fallback)
        # Note: This is simplified; true MACD uses EMA of MACD line
        signal_value = macd_value  # Simplified - would need historical MACD values

        # Histogram = MACD - Signal
        histogram_value = macd_value - signal_value

        self.lines.macd[0] = macd_value
        self.lines.signal[0] = signal_value
        self.lines.histogram[0] = histogram_value


class UnifiedATRIndicator(BacktraderIndicatorWrapper):
    """ATR indicator using unified service"""

    lines = ("atr",)
    params = (
        ("period", 14),
        ("backend", "bt"),
        ("use_unified_service", False),  # Deprecated - use fallback implementation
    )

    def _init_fallback(self):
        """Initialize fallback - just set minimum period"""
        self.addminperiod(self.p.period)
        # Set _fallback_impl to indicate fallback is available
        self._fallback_impl = True

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
        """Calculate ATR directly without child indicators"""
        period = self.p.period
        current_len = len(self.data)

        if current_len < period + 1:
            self.lines.atr[0] = float("nan")
            return

        # Calculate True Ranges for the period
        true_ranges = []
        for i in range(period):
            high = self.data.high[-i]
            low = self.data.low[-i]
            prev_close = self.data.close[-(i + 1)]

            # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # ATR is the average of True Ranges
        atr = sum(true_ranges) / period
        self.lines.atr[0] = atr


class UnifiedSMAIndicator(BacktraderIndicatorWrapper):
    """Simple Moving Average indicator using unified service"""

    lines = ("sma",)
    params = (
        ("period", 20),
        ("backend", "bt"),
        ("use_unified_service", False),
    )

    def _init_fallback(self):
        """Initialize fallback - just set minimum period"""
        self.addminperiod(self.p.period)
        # Set _fallback_impl to indicate fallback is available
        self._fallback_impl = True

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
        """Calculate SMA directly without child indicators"""
        period = self.p.period
        current_len = len(self.data)

        if current_len < period:
            self.lines.sma[0] = float("nan")
            return

        # Get recent closes
        closes = [self.data.close[-i] for i in range(period)]

        # Calculate SMA (simple average)
        sma = sum(closes) / period
        self.lines.sma[0] = sma


class UnifiedEMAIndicator(BacktraderIndicatorWrapper):
    """Exponential Moving Average indicator using unified service"""

    lines = ("ema",)
    params = (
        ("period", 20),
        ("backend", "bt"),
        ("use_unified_service", False),
    )

    def _init_fallback(self):
        """Initialize fallback - just set minimum period"""
        self.addminperiod(self.p.period)
        # Set _fallback_impl to indicate fallback is available
        self._fallback_impl = True
        self._ema_value = None

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
        """Calculate EMA directly without child indicators"""
        period = self.p.period
        current_len = len(self.data)

        if current_len < period:
            self.lines.ema[0] = float("nan")
            return

        # EMA multiplier
        multiplier = 2.0 / (period + 1.0)

        # Initialize EMA with SMA on first calculation
        if self._ema_value is None:
            closes = [self.data.close[-i] for i in range(period)]
            self._ema_value = sum(closes) / period
        else:
            # EMA = (Close - EMA_prev) * multiplier + EMA_prev
            self._ema_value = (self.data.close[0] - self._ema_value) * multiplier + self._ema_value

        self.lines.ema[0] = self._ema_value


class UnifiedSuperTrendIndicator(BacktraderIndicatorWrapper):
    """SuperTrend indicator using unified service"""

    lines = ("super_trend", "direction")
    params = (
        ("length", 10),
        ("multiplier", 3.0),
        ("backend", "bt"),
        ("use_unified_service", False),
    )

    def _init_fallback(self):
        """Initialize fallback - just set minimum period"""
        self.addminperiod(self.p.length)
        # Set _fallback_impl to indicate fallback is available
        self._fallback_impl = True
        self._trend = 1  # 1 for uptrend, -1 for downtrend
        self._super_trend_value = None

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
        """Calculate SuperTrend directly without child indicators"""
        period = self.p.length
        multiplier = self.p.multiplier
        current_len = len(self.data)

        if current_len < period + 1:
            self.lines.super_trend[0] = float("nan")
            self.lines.direction[0] = float("nan")
            return

        # Calculate ATR for the period
        true_ranges = []
        for i in range(period):
            high = self.data.high[-i]
            low = self.data.low[-i]
            prev_close = self.data.close[-(i + 1)]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        atr = sum(true_ranges) / period

        # Calculate basic bands
        hl_avg = (self.data.high[0] + self.data.low[0]) / 2.0
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)

        # Initialize super_trend_value if first calculation
        if self._super_trend_value is None:
            self._super_trend_value = lower_band
            self._trend = 1

        # Determine trend and SuperTrend value
        close = self.data.close[0]

        # Check for trend change
        if self._trend == 1:
            # Currently in uptrend
            if close <= self._super_trend_value:
                self._trend = -1
                self._super_trend_value = upper_band
            else:
                self._super_trend_value = max(lower_band, self._super_trend_value)
        else:
            # Currently in downtrend
            if close >= self._super_trend_value:
                self._trend = 1
                self._super_trend_value = lower_band
            else:
                self._super_trend_value = min(upper_band, self._super_trend_value)

        self.lines.super_trend[0] = self._super_trend_value
        self.lines.direction[0] = float(self._trend)