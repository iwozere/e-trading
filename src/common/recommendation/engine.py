"""
Unified recommendation engine for indicators.

This module provides the main RecommendationEngine class that coordinates
all recommendation logic for both technical and fundamental indicators.
"""

from typing import Dict, Tuple

from .types import (
    Recommendation, RecommendationType, IndicatorCategory,
    IndicatorSet, CompositeRecommendation
)
from .technical_rules import TechnicalRecommendationRules
from .fundamental_rules import FundamentalRecommendationRules
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class RecommendationEngine:
    """Unified recommendation engine for all indicators."""

    def __init__(self):
        """Initialize the recommendation engine."""
        self.technical_rules = TechnicalRecommendationRules()
        self.fundamental_rules = FundamentalRecommendationRules()

        self._recommendation_methods = {
            "RSI": self._get_rsi_recommendation,
            "MACD": self._get_macd_recommendation,
            "BB_UPPER": self._get_bollinger_bands_recommendation,
            "BB_MIDDLE": self._get_bollinger_bands_recommendation,
            "BB_LOWER": self._get_bollinger_bands_recommendation,
            "SMA_FAST": self._get_sma_recommendation_wrapper,
            "SMA_SLOW": self._get_sma_recommendation_wrapper,
            "EMA_FAST": self._get_sma_recommendation_wrapper,
            "EMA_SLOW": self._get_sma_recommendation_wrapper,
            "ADX": self._get_adx_recommendation,
            "PLUS_DI": self._get_di_recommendation,
            "MINUS_DI": self._get_di_recommendation,
            "STOCH_K": self._get_stochastic_recommendation,
            "STOCH_D": self._get_stochastic_recommendation,
            "OBV": self._get_obv_recommendation,
            "ADR": self._get_adr_recommendation,
            "CCI": self._get_cci_recommendation_wrapper,
            "MFI": self._get_mfi_recommendation_wrapper,
            "WILLIAMS_R": self._get_williams_r_recommendation,
            "ROC": self._get_roc_recommendation,
            "ATR": self._get_atr_recommendation,
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
            if indicator in self._recommendation_methods:
                rec_type, confidence, reason = self._recommendation_methods[indicator](value, context)
                category = IndicatorCategory.TECHNICAL
            elif indicator in self.fundamental_functions:
                rec_type, confidence, reason = self.fundamental_functions[indicator](value)
                category = IndicatorCategory.FUNDAMENTAL
            else:
                rec_type = RecommendationType.HOLD
                confidence = 0.5
                reason = f"No specific rules for {indicator}"
                category = IndicatorCategory.TECHNICAL

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
        """Get recommendation in legacy format (Tuple[str, str]) for backward compatibility."""
        recommendation = self.get_recommendation(indicator, value, context)

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

            technical_score = sum(technical_scores) / len(technical_scores) if technical_scores else 0
            fundamental_score = sum(fundamental_scores) / len(fundamental_scores) if fundamental_scores else 0
            composite_score = (technical_score * 0.4 + fundamental_score * 0.6)

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

    def _get_bollinger_bands_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Bollinger Bands recommendation."""
        if context and 'current_price' in context and 'bb_upper' in context and 'bb_lower' in context:
            return self.technical_rules.get_bollinger_recommendation(
                context['current_price'], context['bb_upper'], value, context['bb_lower']
            )
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for Bollinger Bands"

    def _get_macd_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for MACD recommendation."""
        if context and 'macd_signal' in context and 'macd_histogram' in context:
            return self.technical_rules.get_macd_recommendation(
                value, context['macd_signal'], context['macd_histogram']
            )
        elif context and 'macd_signal' in context and 'macd_hist' in context:
            return self.technical_rules.get_macd_recommendation(
                value, context['macd_signal'], context['macd_hist']
            )
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for MACD"

    def _get_stochastic_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for Stochastic recommendation."""
        if context and 'stoch_d' in context:
            return self.technical_rules.get_stochastic_recommendation(value, context['stoch_d'])
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for Stochastic"

    def _get_adx_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
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
            return self.technical_rules.get_sma_recommendation(context['current_price'], value, context)
        else:
            return RecommendationType.HOLD, 0.5, "Insufficient context for SMA"

    def _get_rsi_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for RSI recommendation."""
        return TechnicalRecommendationRules.get_rsi_recommendation(value)

    def _get_cci_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for CCI recommendation."""
        return TechnicalRecommendationRules.get_cci_recommendation(value)

    def _get_mfi_recommendation_wrapper(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for MFI recommendation."""
        return TechnicalRecommendationRules.get_mfi_recommendation(value)

    def _get_obv_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for OBV recommendation."""
        return TechnicalRecommendationRules.get_obv_recommendation(value, context)

    def _get_adr_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Wrapper for ADR recommendation."""
        return TechnicalRecommendationRules.get_adr_recommendation(value, context)

    def _get_di_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Directional Indicators (PLUS_DI, MINUS_DI)."""
        return TechnicalRecommendationRules.get_di_recommendation(value)

    def _get_williams_r_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Williams %R."""
        return TechnicalRecommendationRules.get_williams_r_recommendation(value)

    def _get_roc_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Rate of Change (ROC)."""
        return TechnicalRecommendationRules.get_roc_recommendation(value)

    def _get_atr_recommendation(self, value: float, context: Dict = None) -> Tuple[RecommendationType, float, str]:
        """Get recommendation for Average True Range (ATR)."""
        return TechnicalRecommendationRules.get_atr_recommendation(value)


__all__ = ["RecommendationEngine"]
