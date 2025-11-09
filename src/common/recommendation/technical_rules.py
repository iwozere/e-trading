"""
Technical indicator recommendation rules.

This module contains all the logic for generating recommendations
based on technical indicators like RSI, MACD, Bollinger Bands, etc.
"""

from typing import Dict, Tuple

from .types import RecommendationType


class TechnicalRecommendationRules:
    """Rules for technical indicator recommendations."""

    @staticmethod
    def get_rsi_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get RSI recommendation."""
        if value < 30:
            return RecommendationType.STRONG_BUY, 0.9, "Oversold - Strong buy signal"
        elif value < 40:
            return RecommendationType.BUY, 0.7, "Approaching oversold - Buy opportunity"
        elif value > 70:
            return RecommendationType.STRONG_SELL, 0.9, "Overbought - Strong sell signal"
        elif value > 60:
            return RecommendationType.SELL, 0.7, "Approaching overbought - Sell opportunity"
        else:
            return RecommendationType.HOLD, 0.5, "Neutral zone - No clear signal"

    @staticmethod
    def get_bollinger_recommendation(close: float, bb_upper: float, bb_middle: float, bb_lower: float) -> Tuple[RecommendationType, float, str]:
        """Get Bollinger Bands recommendation."""
        band_width = bb_upper - bb_lower
        if band_width <= 0:
            return RecommendationType.HOLD, 0.5, "Bollinger Bands too narrow or invalid"

        position = (close - bb_lower) / band_width

        if close <= bb_lower:
            return RecommendationType.STRONG_BUY, 0.9, "Price at or below lower band - Oversold"
        elif close >= bb_upper:
            return RecommendationType.STRONG_SELL, 0.9, "Price at or above upper band - Overbought"
        elif position < 0.2:
            return RecommendationType.BUY, 0.7, f"Price near lower band ({position:.1%} position) - Potential buy"
        elif position > 0.8:
            return RecommendationType.SELL, 0.7, f"Price near upper band ({position:.1%} position) - Potential sell"
        elif position < 0.4:
            return RecommendationType.BUY, 0.6, f"Price in lower range ({position:.1%} position) - Slight buy bias"
        elif position > 0.6:
            return RecommendationType.SELL, 0.6, f"Price in upper range ({position:.1%} position) - Slight sell bias"
        else:
            return RecommendationType.HOLD, 0.5, f"Price in middle range ({position:.1%} position) - Neutral"

    @staticmethod
    def get_macd_recommendation(macd: float, signal: float, macd_hist: float) -> Tuple[RecommendationType, float, str]:
        """Get MACD recommendation."""
        if macd is None or signal is None or macd_hist is None:
            return RecommendationType.HOLD, 0.5, "Insufficient MACD data"

        if macd > signal and macd_hist > 0:
            if macd_hist > 0.5:
                return RecommendationType.STRONG_BUY, 0.8, "Strong bullish MACD crossover"
            else:
                return RecommendationType.BUY, 0.6, "Bullish MACD crossover"
        elif macd < signal and macd_hist < 0:
            if macd_hist < -0.5:
                return RecommendationType.STRONG_SELL, 0.8, "Strong bearish MACD crossover"
            else:
                return RecommendationType.SELL, 0.6, "Bearish MACD crossover"
        else:
            return RecommendationType.HOLD, 0.5, "MACD signals neutral"

    @staticmethod
    def get_stochastic_recommendation(k: float, d: float) -> Tuple[RecommendationType, float, str]:
        """Get Stochastic recommendation."""
        if k < 20 and d < 20:
            return RecommendationType.STRONG_BUY, 0.9, "Both K and D in oversold territory"
        elif k > 80 and d > 80:
            return RecommendationType.STRONG_SELL, 0.9, "Both K and D in overbought territory"
        elif k < 30 and k > d:
            return RecommendationType.BUY, 0.7, "K crossing above D in oversold area"
        elif k > 70 and k < d:
            return RecommendationType.SELL, 0.7, "K crossing below D in overbought area"
        else:
            return RecommendationType.HOLD, 0.5, "Stochastic in neutral zone"

    @staticmethod
    def get_adx_recommendation(adx: float, plus_di: float, minus_di: float) -> Tuple[RecommendationType, float, str]:
        """Get ADX recommendation."""
        if adx is None or plus_di is None or minus_di is None:
            return RecommendationType.HOLD, 0.5, "Insufficient ADX data"

        if adx > 25:
            if plus_di > minus_di:
                return RecommendationType.BUY, 0.8, f"Strong uptrend (ADX: {adx:.1f})"
            else:
                return RecommendationType.SELL, 0.8, f"Strong downtrend (ADX: {adx:.1f})"
        else:
            return RecommendationType.HOLD, 0.5, "Weak trend - Sideways market"

    @staticmethod
    def get_sma_recommendation(current_price: float, sma: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get improved SMA recommendation with trend and distance analysis."""
        if current_price is None or sma is None:
            return RecommendationType.HOLD, 0.5, "Insufficient data"

        distance_pct = ((current_price - sma) / sma) * 100
        ma_trend = context.get('ma_trend', 'unknown') if context else 'unknown'
        fast_ma = context.get('fast_ma', None) if context else None
        slow_ma = context.get('slow_ma', None) if context else None

        if fast_ma is not None and slow_ma is not None:
            if fast_ma > slow_ma:
                ma_crossover = "bullish"
            elif fast_ma < slow_ma:
                ma_crossover = "bearish"
            else:
                ma_crossover = "neutral"
        else:
            ma_crossover = "unknown"

        if distance_pct >= 5 and ma_trend == 'up':
            return RecommendationType.STRONG_BUY, 0.8, f"Price {distance_pct:.1f}% above rising MA - Strong uptrend"
        elif distance_pct <= -5 and ma_trend == 'down':
            return RecommendationType.STRONG_SELL, 0.8, f"Price {distance_pct:.1f}% below falling MA - Strong downtrend"
        elif distance_pct > 2 and ma_trend == 'up':
            return RecommendationType.BUY, 0.7, f"Price {distance_pct:.1f}% above rising MA - Uptrend"
        elif distance_pct < -2 and ma_trend == 'down':
            return RecommendationType.SELL, 0.7, f"Price {distance_pct:.1f}% below falling MA - Downtrend"
        elif ma_crossover == "bullish" and current_price > sma and distance_pct <= 2:
            return RecommendationType.BUY, 0.7, f"Bullish MA crossover - Price {distance_pct:.1f}% above MA"
        elif ma_crossover == "bearish" and current_price < sma and distance_pct >= -2:
            return RecommendationType.SELL, 0.7, f"Bearish MA crossover - Price {distance_pct:.1f}% below MA"
        elif current_price > sma:
            return RecommendationType.BUY, 0.6, f"Price above MA ({distance_pct:.1f}%) - Weak bullish"
        else:
            return RecommendationType.SELL, 0.6, f"Price below MA ({distance_pct:.1f}%) - Weak bearish"

    @staticmethod
    def get_cci_recommendation(cci: float) -> Tuple[RecommendationType, float, str]:
        """Get CCI recommendation."""
        if cci <= -100:
            return RecommendationType.BUY, 0.7, "CCI in oversold territory"
        elif cci >= 100:
            return RecommendationType.SELL, 0.7, "CCI in overbought territory"
        else:
            return RecommendationType.HOLD, 0.5, "CCI in neutral zone"

    @staticmethod
    def get_mfi_recommendation(mfi: float) -> Tuple[RecommendationType, float, str]:
        """Get MFI recommendation."""
        if mfi <= 20:
            return RecommendationType.BUY, 0.7, "MFI in oversold territory"
        elif mfi >= 80:
            return RecommendationType.SELL, 0.7, "MFI in overbought territory"
        else:
            return RecommendationType.HOLD, 0.5, "MFI in neutral zone"

    @staticmethod
    def get_obv_recommendation(obv: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get OBV recommendation."""
        if obv is None:
            return RecommendationType.HOLD, 0.5, "Insufficient OBV data"

        if context and 'obv_prev' in context:
            obv_change = obv - context['obv_prev']
            if obv_change > 0:
                return RecommendationType.BUY, 0.6, "OBV increasing - Accumulation"
            elif obv_change < 0:
                return RecommendationType.SELL, 0.6, "OBV decreasing - Distribution"
            else:
                return RecommendationType.HOLD, 0.5, "OBV stable - No clear signal"
        else:
            if obv > 0:
                return RecommendationType.BUY, 0.5, "Positive OBV - Accumulation"
            else:
                return RecommendationType.HOLD, 0.5, "OBV neutral - No clear signal"

    @staticmethod
    def get_adr_recommendation(adr: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get ADR (Average Daily Range) recommendation."""
        if adr is None:
            return RecommendationType.HOLD, 0.5, "Insufficient ADR data"

        if context and 'current_price' in context:
            adr_percentage = (adr / context['current_price']) * 100
            if adr_percentage > 5:
                return RecommendationType.HOLD, 0.7, f"High volatility ({adr_percentage:.1f}%) - Exercise caution"
            elif adr_percentage > 3:
                return RecommendationType.HOLD, 0.6, f"Moderate volatility ({adr_percentage:.1f}%) - Normal trading"
            else:
                return RecommendationType.HOLD, 0.5, f"Low volatility ({adr_percentage:.1f}%) - Stable price action"
        else:
            return RecommendationType.HOLD, 0.5, f"ADR: {adr:.2f} - Volatility indicator"

    @staticmethod
    def get_williams_r_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Williams %R."""
        if value is None:
            return RecommendationType.HOLD, 0.0, "No Williams %R data available"

        if value > -20:
            return RecommendationType.SELL, 0.8, "Overbought - Sell signal"
        elif value < -80:
            return RecommendationType.BUY, 0.8, "Oversold - Buy signal"
        else:
            return RecommendationType.HOLD, 0.5, "Neutral zone"

    @staticmethod
    def get_roc_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Rate of Change (ROC)."""
        if value is None:
            return RecommendationType.HOLD, 0.0, "No ROC data available"

        if value > 2.0:
            return RecommendationType.BUY, 0.7, f"Strong positive momentum ({value:.2f}%)"
        elif value < -2.0:
            return RecommendationType.SELL, 0.7, f"Strong negative momentum ({value:.2f}%)"
        else:
            return RecommendationType.HOLD, 0.5, f"Neutral momentum ({value:.2f}%)"

    @staticmethod
    def get_atr_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Average True Range (ATR)."""
        if value is None:
            return RecommendationType.HOLD, 0.0, "No ATR data available"

        if value > 5.0:
            return RecommendationType.HOLD, 0.6, f"High volatility ({value:.2f}) - Caution"
        elif value < 1.0:
            return RecommendationType.HOLD, 0.6, f"Low volatility ({value:.2f}) - Stable"
        else:
            return RecommendationType.HOLD, 0.5, f"Normal volatility ({value:.2f})"

    @staticmethod
    def get_di_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Directional Indicators (PLUS_DI, MINUS_DI)."""
        if value is None:
            return RecommendationType.HOLD, 0.0, "No DI data available"

        if value > 25:
            return RecommendationType.BUY, 0.7, f"Strong directional movement ({value:.1f})"
        elif value < 15:
            return RecommendationType.SELL, 0.7, f"Weak directional movement ({value:.1f})"
        else:
            return RecommendationType.HOLD, 0.5, f"Moderate directional movement ({value:.1f})"


__all__ = ["TechnicalRecommendationRules"]
