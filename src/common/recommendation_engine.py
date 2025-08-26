"""
Unified recommendation engine for indicators.

This module consolidates all recommendation logic from scattered locations
into a single, consistent recommendation engine for both technical and fundamental indicators.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd

from src.models.indicators import (
    Recommendation, RecommendationType, IndicatorCategory,
    IndicatorResult, IndicatorSet, CompositeRecommendation
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


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
        if close <= bb_lower:
            return RecommendationType.STRONG_BUY, 0.9, "Price at or below lower band - Oversold"
        elif close >= bb_upper:
            return RecommendationType.STRONG_SELL, 0.9, "Price at or above upper band - Overbought"
        elif close < bb_middle:
            return RecommendationType.BUY, 0.6, "Price below middle band - Potential buy"
        else:
            return RecommendationType.HOLD, 0.5, "Price above middle band - Neutral"

    @staticmethod
    def get_macd_recommendation(macd: float, signal: float, macd_hist: float) -> Tuple[RecommendationType, float, str]:
        """Get MACD recommendation."""
        # Handle None values
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
        # Handle None values
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
    def get_sma_recommendation(current_price: float, sma: float) -> Tuple[RecommendationType, float, str]:
        """Get SMA recommendation."""
        if current_price is None or sma is None:
            return RecommendationType.HOLD, 0.5, "Insufficient data"

        if current_price > sma:
            return RecommendationType.BUY, 0.6, "Price above moving average"
        else:
            return RecommendationType.SELL, 0.6, "Price below moving average"

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

        # OBV is cumulative, so we need to look at the trend
        # For now, we'll use a simple approach based on the magnitude
        # In a real implementation, you'd want to compare with previous values
        if context and 'obv_prev' in context:
            obv_change = obv - context['obv_prev']
            if obv_change > 0:
                return RecommendationType.BUY, 0.6, "OBV increasing - Accumulation"
            elif obv_change < 0:
                return RecommendationType.SELL, 0.6, "OBV decreasing - Distribution"
            else:
                return RecommendationType.HOLD, 0.5, "OBV stable - No clear signal"
        else:
            # Without trend data, use magnitude-based approach
            if obv > 0:
                return RecommendationType.BUY, 0.5, "Positive OBV - Accumulation"
            else:
                return RecommendationType.HOLD, 0.5, "OBV neutral - No clear signal"

    @staticmethod
    def get_adr_recommendation(adr: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get ADR (Average Daily Range) recommendation."""
        if adr is None:
            return RecommendationType.HOLD, 0.5, "Insufficient ADR data"

        # ADR indicates volatility - higher ADR means more volatility
        # This is more informational than directional
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


class FundamentalRecommendationRules:
    """Rules for fundamental indicator recommendations."""

    @staticmethod
    def get_pe_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get P/E ratio recommendation."""
        if value <= 15:
            return RecommendationType.STRONG_BUY, 0.8, "Low P/E - Undervalued"
        elif value <= 25:
            return RecommendationType.BUY, 0.6, "Reasonable P/E - Good value"
        elif value <= 35:
            return RecommendationType.HOLD, 0.5, "Moderate P/E - Fair value"
        else:
            return RecommendationType.SELL, 0.7, "High P/E - Potentially overvalued"

    @staticmethod
    def get_pb_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get P/B ratio recommendation."""
        if value <= 1.5:
            return RecommendationType.STRONG_BUY, 0.8, "Low P/B - Undervalued"
        elif value <= 3.0:
            return RecommendationType.BUY, 0.6, "Reasonable P/B - Good value"
        elif value <= 5.0:
            return RecommendationType.HOLD, 0.5, "Moderate P/B - Fair value"
        else:
            return RecommendationType.SELL, 0.7, "High P/B - Potentially overvalued"

    @staticmethod
    def get_ps_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get P/S ratio recommendation."""
        if value <= 1.0:
            return RecommendationType.STRONG_BUY, 0.8, "Low P/S - Undervalued"
        elif value <= 3.0:
            return RecommendationType.BUY, 0.6, "Reasonable P/S - Good value"
        elif value <= 5.0:
            return RecommendationType.HOLD, 0.5, "Moderate P/S - Fair value"
        else:
            return RecommendationType.SELL, 0.7, "High P/S - Potentially overvalued"

    @staticmethod
    def get_peg_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get PEG ratio recommendation."""
        if value <= 1.0:
            return RecommendationType.STRONG_BUY, 0.8, "Low PEG - Undervalued"
        elif value <= 1.5:
            return RecommendationType.BUY, 0.6, "Reasonable PEG - Good value"
        elif value <= 2.0:
            return RecommendationType.HOLD, 0.5, "Moderate PEG - Fair value"
        else:
            return RecommendationType.SELL, 0.7, "High PEG - Potentially overvalued"

    @staticmethod
    def get_roe_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get ROE recommendation."""
        if value >= 0.15:
            return RecommendationType.STRONG_BUY, 0.8, "High ROE - Excellent profitability"
        elif value >= 0.10:
            return RecommendationType.BUY, 0.6, "Good ROE - Strong profitability"
        elif value >= 0.05:
            return RecommendationType.HOLD, 0.5, "Moderate ROE - Average profitability"
        else:
            return RecommendationType.SELL, 0.7, "Low ROE - Poor profitability"

    @staticmethod
    def get_roa_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get ROA recommendation."""
        if value >= 0.05:
            return RecommendationType.STRONG_BUY, 0.8, "High ROA - Excellent efficiency"
        elif value >= 0.03:
            return RecommendationType.BUY, 0.6, "Good ROA - Strong efficiency"
        elif value >= 0.01:
            return RecommendationType.HOLD, 0.5, "Moderate ROA - Average efficiency"
        else:
            return RecommendationType.SELL, 0.7, "Low ROA - Poor efficiency"

    @staticmethod
    def get_debt_equity_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Debt/Equity recommendation."""
        if value <= 0.5:
            return RecommendationType.STRONG_BUY, 0.8, "Low debt - Strong financial position"
        elif value <= 1.0:
            return RecommendationType.BUY, 0.6, "Moderate debt - Good financial position"
        elif value <= 2.0:
            return RecommendationType.HOLD, 0.5, "High debt - Moderate financial position"
        else:
            return RecommendationType.SELL, 0.7, "Very high debt - Poor financial position"

    @staticmethod
    def get_current_ratio_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Current Ratio recommendation."""
        if value >= 2.0:
            return RecommendationType.STRONG_BUY, 0.8, "High liquidity - Strong short-term position"
        elif value >= 1.5:
            return RecommendationType.BUY, 0.6, "Good liquidity - Strong short-term position"
        elif value >= 1.0:
            return RecommendationType.HOLD, 0.5, "Adequate liquidity - Moderate short-term position"
        else:
            return RecommendationType.SELL, 0.7, "Low liquidity - Poor short-term position"

    @staticmethod
    def get_quick_ratio_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Quick Ratio recommendation."""
        if value >= 1.0:
            return RecommendationType.STRONG_BUY, 0.8, "High quick ratio - Strong immediate liquidity"
        elif value >= 0.8:
            return RecommendationType.BUY, 0.6, "Good quick ratio - Strong immediate liquidity"
        elif value >= 0.5:
            return RecommendationType.HOLD, 0.5, "Adequate quick ratio - Moderate immediate liquidity"
        else:
            return RecommendationType.SELL, 0.7, "Low quick ratio - Poor immediate liquidity"

    @staticmethod
    def get_margin_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Operating/Profit Margin recommendation."""
        if value >= 0.15:
            return RecommendationType.STRONG_BUY, 0.8, "High margin - Excellent profitability"
        elif value >= 0.10:
            return RecommendationType.BUY, 0.6, "Good margin - Strong profitability"
        elif value >= 0.05:
            return RecommendationType.HOLD, 0.5, "Moderate margin - Average profitability"
        else:
            return RecommendationType.SELL, 0.7, "Low margin - Poor profitability"

    @staticmethod
    def get_growth_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Revenue/Net Income Growth recommendation."""
        if value >= 0.10:
            return RecommendationType.STRONG_BUY, 0.8, "High growth - Excellent growth prospects"
        elif value >= 0.05:
            return RecommendationType.BUY, 0.6, "Good growth - Strong growth prospects"
        elif value >= 0.02:
            return RecommendationType.HOLD, 0.5, "Moderate growth - Average growth prospects"
        else:
            return RecommendationType.SELL, 0.7, "Low growth - Poor growth prospects"

    @staticmethod
    def get_fcf_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Free Cash Flow recommendation."""
        if value > 0:
            return RecommendationType.STRONG_BUY, 0.8, "Positive FCF - Strong cash generation"
        elif value >= -1000000:  # Small negative FCF
            return RecommendationType.HOLD, 0.5, "Slight negative FCF - Monitor closely"
        else:
            return RecommendationType.SELL, 0.7, "Negative FCF - Poor cash generation"

    @staticmethod
    def get_dividend_yield_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Dividend Yield recommendation."""
        if value >= 4.0:
            return RecommendationType.STRONG_BUY, 0.8, "High dividend yield - Strong income potential"
        elif value >= 2.0:
            return RecommendationType.BUY, 0.6, "Good dividend yield - Strong income potential"
        elif value >= 1.0:
            return RecommendationType.HOLD, 0.5, "Moderate dividend yield - Average income potential"
        else:
            return RecommendationType.SELL, 0.7, "Low dividend yield - Poor income potential"

    @staticmethod
    def get_payout_ratio_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Payout Ratio recommendation."""
        if value <= 0.50:
            return RecommendationType.STRONG_BUY, 0.8, "Low payout ratio - Sustainable dividends"
        elif value <= 0.75:
            return RecommendationType.BUY, 0.6, "Moderate payout ratio - Sustainable dividends"
        elif value <= 1.0:
            return RecommendationType.HOLD, 0.5, "High payout ratio - Monitor sustainability"
        else:
            return RecommendationType.SELL, 0.7, "Very high payout ratio - Unsustainable dividends"

    @staticmethod
    def get_growth_recommendation(value: float) -> Tuple[RecommendationType, float, str]:
        """Get Revenue/Net Income Growth recommendation."""
        if value >= 0.10:
            return RecommendationType.STRONG_BUY, 0.8, "High growth - Excellent growth prospects"
        elif value >= 0.05:
            return RecommendationType.BUY, 0.6, "Good growth - Strong growth prospects"
        elif value >= 0.02:
            return RecommendationType.HOLD, 0.5, "Moderate growth - Average growth prospects"
        else:
            return RecommendationType.SELL, 0.7, "Low growth - Poor growth prospects"


class RecommendationEngine:
    """Unified recommendation engine for all indicators."""

    def __init__(self):
        """Initialize the recommendation engine."""
        self.technical_rules = TechnicalRecommendationRules()
        self.fundamental_rules = FundamentalRecommendationRules()

        # Mapping of indicator names to their recommendation functions
        self.technical_functions = {
            "RSI": self._get_rsi_recommendation_wrapper,
            "BB_UPPER": self._get_bollinger_recommendation_wrapper,
            "BB_MIDDLE": self._get_bollinger_recommendation_wrapper,
            "BB_LOWER": self._get_bollinger_recommendation_wrapper,
            "MACD": self._get_macd_recommendation_wrapper,
            "MACD_SIGNAL": self._get_macd_recommendation_wrapper,
            "MACD_HISTOGRAM": self._get_macd_recommendation_wrapper,
            "STOCH_K": self._get_stochastic_recommendation_wrapper,
            "STOCH_D": self._get_stochastic_recommendation_wrapper,
            "ADX": self._get_adx_recommendation_wrapper,
            "PLUS_DI": self._get_adx_recommendation_wrapper,
            "MINUS_DI": self._get_adx_recommendation_wrapper,
            "SMA_50": self._get_sma_recommendation_wrapper,
            "SMA_200": self._get_sma_recommendation_wrapper,
            "CCI": self._get_cci_recommendation_wrapper,
            "MFI": self._get_mfi_recommendation_wrapper,
            "OBV": self._get_obv_recommendation_wrapper,
            "ADR": self._get_adr_recommendation_wrapper,
        }

        self.fundamental_functions = {
            "PE_RATIO": self.fundamental_rules.get_pe_recommendation,
            "FORWARD_PE": self.fundamental_rules.get_pe_recommendation,
            "PB_RATIO": self.fundamental_rules.get_pb_recommendation,
            "PS_RATIO": self.fundamental_rules.get_ps_recommendation,
            "PEG_RATIO": self.fundamental_rules.get_peg_recommendation,
            "ROE": self.fundamental_rules.get_roe_recommendation,
            "ROA": self.fundamental_rules.get_roa_recommendation,
            "DEBT_TO_EQUITY": self.fundamental_rules.get_debt_equity_recommendation,
            "CURRENT_RATIO": self.fundamental_rules.get_current_ratio_recommendation,
            "QUICK_RATIO": self.fundamental_rules.get_quick_ratio_recommendation,
            "OPERATING_MARGIN": self.fundamental_rules.get_margin_recommendation,
            "PROFIT_MARGIN": self.fundamental_rules.get_margin_recommendation,
            "REVENUE_GROWTH": self.fundamental_rules.get_growth_recommendation,
            "NET_INCOME_GROWTH": self.fundamental_rules.get_growth_recommendation,
            "FREE_CASH_FLOW": self.fundamental_rules.get_fcf_recommendation,
            "DIVIDEND_YIELD": self.fundamental_rules.get_dividend_yield_recommendation,
            "PAYOUT_RATIO": self.fundamental_rules.get_payout_ratio_recommendation,
        }

    def get_recommendation(self, indicator: str, value: float, context: Dict = None) -> Recommendation:
        """
        Get recommendation for any indicator type.

        Args:
            indicator: Indicator name
            value: Indicator value
            context: Additional context (e.g., current price for technical indicators)

        Returns:
            Recommendation object
        """
        try:
            # Determine if it's a technical or fundamental indicator
            if indicator in self.technical_functions:
                rec_type, confidence, reason = self.technical_functions[indicator](value, context)
                category = IndicatorCategory.TECHNICAL
            elif indicator in self.fundamental_functions:
                rec_type, confidence, reason = self.fundamental_functions[indicator](value)
                category = IndicatorCategory.FUNDAMENTAL
            else:
                # Default recommendation for unknown indicators
                rec_type = RecommendationType.HOLD
                confidence = 0.5
                reason = f"No specific rules for {indicator}"
                category = IndicatorCategory.TECHNICAL  # Default to technical

            return Recommendation(
                recommendation=rec_type,
                confidence=confidence,
                reason=reason,
                threshold_used=value,
                context=context
            )

        except Exception as e:
            _logger.exception("Error getting recommendation for %s: %s", indicator, e)
            return Recommendation(
                recommendation=RecommendationType.HOLD,
                confidence=0.0,
                reason=f"Error calculating recommendation: {str(e)}",
                threshold_used=value,
                context=context
            )

    def get_legacy_recommendation(self, indicator: str, value: float, context: Dict = None) -> Tuple[str, str]:
        """
        Get recommendation in legacy format (Tuple[str, str]) for backward compatibility.
        """
        recommendation = self.get_recommendation(indicator, value, context)

        # Convert RecommendationType to string
        if recommendation.recommendation == RecommendationType.STRONG_BUY:
            rec_str = "BUY"
        elif recommendation.recommendation == RecommendationType.BUY:
            rec_str = "BUY"
        elif recommendation.recommendation == RecommendationType.STRONG_SELL:
            rec_str = "SELL"
        elif recommendation.recommendation == RecommendationType.SELL:
            rec_str = "SELL"
        else:
            rec_str = "HOLD"

        return rec_str, recommendation.reason

    def get_composite_recommendation(self, indicator_set: IndicatorSet) -> CompositeRecommendation:
        """
        Get overall recommendation based on all indicators.

        Args:
            indicator_set: Collection of indicators for a ticker

        Returns:
            Composite recommendation
        """
        try:
            all_indicators = indicator_set.get_all_indicators()

            if not all_indicators:
                return CompositeRecommendation(
                    recommendation=RecommendationType.HOLD,
                    confidence=0.0,
                    reasoning="No indicators available",
                    contributing_indicators=[]
                )

            # Calculate scores by category
            technical_scores = []
            fundamental_scores = []
            contributing_indicators = []

            for name, indicator in all_indicators.items():
                if indicator.recommendation.recommendation in [RecommendationType.STRONG_BUY, RecommendationType.BUY]:
                    score = indicator.recommendation.confidence
                    contributing_indicators.append(name)
                elif indicator.recommendation.recommendation in [RecommendationType.STRONG_SELL, RecommendationType.SELL]:
                    score = -indicator.recommendation.confidence
                    contributing_indicators.append(name)
                else:
                    score = 0

                if indicator.category == IndicatorCategory.TECHNICAL:
                    technical_scores.append(score)
                else:
                    fundamental_scores.append(score)

            # Calculate average scores
            technical_score = sum(technical_scores) / len(technical_scores) if technical_scores else 0
            fundamental_score = sum(fundamental_scores) / len(fundamental_scores) if fundamental_scores else 0

            # Weighted composite score (can be adjusted)
            composite_score = (technical_score * 0.4 + fundamental_score * 0.6)

            # Determine overall recommendation
            if composite_score >= 0.3:
                recommendation = RecommendationType.STRONG_BUY
                reasoning = f"Strong positive signals from {len(contributing_indicators)} indicators"
            elif composite_score >= 0.1:
                recommendation = RecommendationType.BUY
                reasoning = f"Positive signals from {len(contributing_indicators)} indicators"
            elif composite_score <= -0.3:
                recommendation = RecommendationType.STRONG_SELL
                reasoning = f"Strong negative signals from {len(contributing_indicators)} indicators"
            elif composite_score <= -0.1:
                recommendation = RecommendationType.SELL
                reasoning = f"Negative signals from {len(contributing_indicators)} indicators"
            else:
                recommendation = RecommendationType.HOLD
                reasoning = f"Mixed signals from {len(contributing_indicators)} indicators"

            return CompositeRecommendation(
                recommendation=recommendation,
                confidence=abs(composite_score),
                reasoning=reasoning,
                contributing_indicators=contributing_indicators,
                technical_score=technical_score,
                fundamental_score=fundamental_score,
                composite_score=composite_score
            )

        except Exception as e:
            _logger.exception("Error calculating composite recommendation: %s", e)
            return CompositeRecommendation(
                recommendation=RecommendationType.HOLD,
                confidence=0.0,
                reasoning=f"Error calculating composite recommendation: {str(e)}",
                contributing_indicators=[]
            )

    # Wrapper methods for indicators that need additional context
    def _get_bollinger_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Bollinger Bands recommendation."""
        if context and 'current_price' in context and 'bb_upper' in context and 'bb_lower' in context:
            return self.technical_rules.get_bollinger_recommendation(
                context['current_price'], context['bb_upper'], value, context['bb_lower']
            )
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for Bollinger Bands"

    def _get_macd_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for MACD recommendation."""
        if context and 'macd_signal' in context and 'macd_histogram' in context:
            return self.technical_rules.get_macd_recommendation(
                value, context['macd_signal'], context['macd_histogram']
            )
        elif context and 'macd_signal' in context and 'macd_hist' in context:
            # Handle legacy naming convention
            return self.technical_rules.get_macd_recommendation(
                value, context['macd_signal'], context['macd_hist']
            )
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for MACD"

    def _get_stochastic_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Stochastic recommendation."""
        if context and 'stoch_d' in context:
            return self.technical_rules.get_stochastic_recommendation(value, context['stoch_d'])
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for Stochastic"

    def _get_adx_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for ADX recommendation."""
        if context and 'plus_di' in context and 'minus_di' in context:
            return self.technical_rules.get_adx_recommendation(
                value, context['plus_di'], context['minus_di']
            )
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for ADX"

    def _get_sma_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for SMA recommendation."""
        if context and 'current_price' in context:
            return self.technical_rules.get_sma_recommendation(context['current_price'], value)
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for SMA"

    def _get_rsi_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for RSI recommendation."""
        return TechnicalRecommendationRules.get_rsi_recommendation(value)

    def _get_cci_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for CCI recommendation."""
        return TechnicalRecommendationRules.get_cci_recommendation(value)

    def _get_mfi_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for MFI recommendation."""
        return TechnicalRecommendationRules.get_mfi_recommendation(value)

    def _get_obv_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for OBV recommendation."""
        return TechnicalRecommendationRules.get_obv_recommendation(value, context)

    def _get_adr_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for ADR recommendation."""
        return TechnicalRecommendationRules.get_adr_recommendation(value, context)
