"""
Unified data models for indicator system.

This module provides the core data structures for the unified indicator system,
including indicator results, indicator sets, and recommendation structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class RecommendationType(Enum):
    """Types of recommendations."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    WEAK_BUY = "WEAK_BUY"
    WEAK_SELL = "WEAK_SELL"


class IndicatorCategory(Enum):
    """Categories of indicators."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


@dataclass
class Recommendation:
    """A recommendation for an indicator."""
    recommendation: RecommendationType
    confidence: float  # 0.0 to 1.0
    reason: str
    threshold_used: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class IndicatorResult:
    """Result of a single indicator calculation."""
    name: str
    value: float
    recommendation: Recommendation
    category: IndicatorCategory
    last_updated: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IndicatorSet:
    """Collection of indicators for a ticker."""
    ticker: str
    technical_indicators: Dict[str, IndicatorResult] = field(default_factory=dict)
    fundamental_indicators: Dict[str, IndicatorResult] = field(default_factory=dict)
    composite_score: float = 0.0
    overall_recommendation: Optional[Recommendation] = None
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_indicator(self, name: str) -> Optional[IndicatorResult]:
        """Get an indicator by name from either category."""
        return (
            self.technical_indicators.get(name) or
            self.fundamental_indicators.get(name)
        )

    def get_indicators_by_category(self, category: IndicatorCategory) -> Dict[str, IndicatorResult]:
        """Get all indicators of a specific category."""
        if category == IndicatorCategory.TECHNICAL:
            return self.technical_indicators
        elif category == IndicatorCategory.FUNDAMENTAL:
            return self.fundamental_indicators
        else:
            return {}

    def add_indicator(self, indicator: IndicatorResult):
        """Add an indicator to the appropriate category."""
        if indicator.category == IndicatorCategory.TECHNICAL:
            self.technical_indicators[indicator.name] = indicator
        elif indicator.category == IndicatorCategory.FUNDAMENTAL:
            self.fundamental_indicators[indicator.name] = indicator

    def get_all_indicators(self) -> Dict[str, IndicatorResult]:
        """Get all indicators as a single dictionary."""
        all_indicators = {}
        all_indicators.update(self.technical_indicators)
        all_indicators.update(self.fundamental_indicators)
        return all_indicators


@dataclass
class CompositeRecommendation:
    """Overall recommendation based on multiple indicators."""
    recommendation: RecommendationType
    confidence: float
    reasoning: str
    contributing_indicators: List[str]
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    composite_score: float = 0.0


@dataclass
class IndicatorCalculationRequest:
    """Request for indicator calculation."""
    ticker: str
    indicators: List[str]
    timeframe: str = "1d"
    period: str = "2y"
    provider: Optional[str] = None
    force_refresh: bool = False
    include_recommendations: bool = True


@dataclass
class BatchIndicatorRequest:
    """Request for batch indicator calculation."""
    tickers: List[str]
    indicators: List[str]
    timeframe: str = "1d"
    period: str = "2y"
    provider: Optional[str] = None
    force_refresh: bool = False
    include_recommendations: bool = True
    max_concurrent: int = 10


@dataclass
class CacheEntry:
    """Cache entry for indicator data."""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics for indicator calculations."""
    operation: str
    duration: float
    cache_hit: bool
    indicators_calculated: int
    tickers_processed: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Constants for indicator names
INDICATOR_DESCRIPTIONS = {
    "RSI": "Relative Strength Index",
    "MACD": "Moving Average Convergence Divergence",
    "MACD_SIGNAL": "MACD Signal Line",
    "MACD_HISTOGRAM": "MACD Histogram",
    "BB_UPPER": "Bollinger Bands Upper",
    "BB_MIDDLE": "Bollinger Bands Middle",
    "BB_LOWER": "Bollinger Bands Lower",
    "SMA_FAST": "Simple Moving Average Fast",
    "SMA_SLOW": "Simple Moving Average Slow",
    "EMA_FAST": "Exponential Moving Average Fast",
    "EMA_SLOW": "Exponential Moving Average Slow",
    "ADX": "Average Directional Index",
    "PLUS_DI": "Plus Directional Indicator",
    "MINUS_DI": "Minus Directional Indicator",
    "ATR": "Average True Range",
    "STOCH_K": "Stochastic %K",
    "STOCH_D": "Stochastic %D",
    "WILLIAMS_R": "Williams %R",
    "CCI": "Commodity Channel Index",
    "ROC": "Rate of Change",
    "MFI": "Money Flow Index",
    "OBV": "On-Balance Volume",
    "ADR": "Average Daily Range"
}

FUNDAMENTAL_INDICATORS = {
    "PE_RATIO": "Price-to-Earnings Ratio",
    "FORWARD_PE": "Forward P/E Ratio",
    "PB_RATIO": "Price-to-Book Ratio",
    "PS_RATIO": "Price-to-Sales Ratio",
    "PEG_RATIO": "Price/Earnings-to-Growth Ratio",
    "ROE": "Return on Equity",
    "ROA": "Return on Assets",
    "DEBT_TO_EQUITY": "Debt-to-Equity Ratio",
    "CURRENT_RATIO": "Current Ratio",
    "QUICK_RATIO": "Quick Ratio",
    "OPERATING_MARGIN": "Operating Margin",
    "PROFIT_MARGIN": "Profit Margin",
    "REVENUE_GROWTH": "Revenue Growth",
    "NET_INCOME_GROWTH": "Net Income Growth",
    "FREE_CASH_FLOW": "Free Cash Flow",
    "DIVIDEND_YIELD": "Dividend Yield",
    "PAYOUT_RATIO": "Payout Ratio",
    "BETA": "Beta",
    "MARKET_CAP": "Market Capitalization",
    "ENTERPRISE_VALUE": "Enterprise Value",
    "EV_TO_EBITDA": "Enterprise Value to EBITDA",
}

ALL_INDICATORS = {**INDICATOR_DESCRIPTIONS, **FUNDAMENTAL_INDICATORS}
