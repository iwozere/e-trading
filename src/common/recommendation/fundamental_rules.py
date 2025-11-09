"""
Fundamental indicator recommendation rules.

This module contains all the logic for generating recommendations
based on fundamental indicators like P/E ratio, ROE, debt ratios, etc.
"""

from typing import Tuple

from .types import RecommendationType


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
        elif value >= -1000000:
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


__all__ = ["FundamentalRecommendationRules"]
