"""
Unified data models for indicator system.

This module provides the core data structures for the unified indicator system,
including indicator results, indicator sets, and recommendation structures.
All models use Pydantic for validation and serialization.
"""

from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict

# Import comprehensive type definitions
from src.indicators.types import (
    TickerSymbol, IndicatorName, TimeFrame, Period, ProviderName,
    RecommendationLiteral, IndicatorCategoryLiteral, FillMethodLiteral,
    IndicatorParameters, CalculationContext
)


class RecommendationType(str, Enum):
    """Types of recommendations."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"
    WEAK_BUY = "WEAK_BUY"
    WEAK_SELL = "WEAK_SELL"


class IndicatorCategory(str, Enum):
    """Categories of indicators."""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"


class Recommendation(BaseModel):
    """A recommendation for an indicator."""
    model_config = ConfigDict(use_enum_values=True)

    recommendation: RecommendationType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reason: str = Field(min_length=1, description="Explanation for the recommendation")
    threshold_used: Optional[float] = Field(default=None, description="Threshold value used for recommendation")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context data")


class IndicatorValue(BaseModel):
    """Simple indicator value container."""
    name: str = Field(min_length=1, description="Indicator name")
    value: Union[float, Dict[str, float]] = Field(description="Indicator value(s)")
    source: Optional[str] = Field(default=None, description="Data source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class IndicatorResult(BaseModel):
    """Result of a single indicator calculation."""
    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(min_length=1, description="Indicator name")
    value: Union[float, Dict[str, float]] = Field(description="Indicator value(s)")
    recommendation: Optional[Recommendation] = Field(default=None, description="Trading recommendation")
    category: IndicatorCategory = Field(description="Indicator category")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last calculation time")
    source: str = Field(min_length=1, description="Calculation source/backend")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CompositeRecommendation(BaseModel):
    """Overall recommendation based on multiple indicators."""
    model_config = ConfigDict(use_enum_values=True)

    recommendation: RecommendationType
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score")
    reasoning: str = Field(min_length=1, description="Explanation for composite recommendation")
    contributing_indicators: List[str] = Field(description="Indicators that contributed to recommendation")
    technical_score: float = Field(default=0.0, description="Technical indicators score")
    fundamental_score: float = Field(default=0.0, description="Fundamental indicators score")
    composite_score: float = Field(default=0.0, description="Overall composite score")


class IndicatorSet(BaseModel):
    """Collection of indicators for a ticker."""
    model_config = ConfigDict(use_enum_values=True)

    ticker: str = Field(min_length=1, description="Stock ticker symbol")
    technical_indicators: Dict[str, IndicatorResult] = Field(default_factory=dict)
    fundamental_indicators: Dict[str, IndicatorResult] = Field(default_factory=dict)
    composite_score: float = Field(default=0.0, description="Overall composite score")
    overall_recommendation: Optional[CompositeRecommendation] = Field(default=None)
    last_updated: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

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


# Request models
class FillNASpec(BaseModel):
    """Specification for handling missing data."""
    method: Optional[Literal["ffill", "bfill", "zero"]] = Field(default=None)
    limit: Optional[int] = Field(default=None, ge=1)


class WarmupSpec(BaseModel):
    """Specification for indicator warmup period."""
    min_bars: int = Field(default=0, ge=0, description="Minimum bars required")
    mask_to_nan: bool = Field(default=True, description="Mask insufficient data to NaN")


class IndicatorSpec(BaseModel):
    """Specification for a single indicator calculation."""
    name: str = Field(min_length=1, description="Canonical indicator name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")
    input_map: Dict[str, str] = Field(default_factory=dict, description="Input column mapping")
    output: Union[str, Dict[str, str]] = Field(description="Output column name(s)")
    depends_on: List[str] = Field(default_factory=list, description="Dependencies on other indicators")
    timeframe: Optional[str] = Field(default=None, description="Override timeframe for this indicator")


class IndicatorBatchConfig(BaseModel):
    """Configuration for batch indicator calculation."""
    timeframe: Optional[str] = Field(default=None, description="Default timeframe")
    fillna: Optional[FillNASpec] = Field(default=None)
    warmup: Optional[WarmupSpec] = Field(default=None)
    dropna_after: bool = Field(default=False, description="Drop NaN values after calculation")
    indicators: List[IndicatorSpec] = Field(description="List of indicators to calculate")

    @field_validator("indicators")
    @classmethod
    def unique_outputs(cls, v):
        """Validate that output column names are unique."""
        seen = set()
        for spec in v:
            outs = spec.output if isinstance(spec.output, dict) else {"value": spec.output}
            for name in outs.values():
                if name in seen:
                    raise ValueError(f"Duplicate output column: {name}")
                seen.add(name)
        return v


class IndicatorCalculationRequest(BaseModel):
    """Request for indicator calculation."""
    ticker: TickerSymbol = Field(min_length=1, description="Stock ticker symbol")
    indicators: List[IndicatorName] = Field(min_length=1, description="List of indicator names")
    timeframe: TimeFrame = Field(default="1d", description="Data timeframe")
    period: Period = Field(default="2y", description="Data period")
    provider: Optional[ProviderName] = Field(default=None, description="Data provider")
    force_refresh: bool = Field(default=False, description="Force cache refresh")
    include_recommendations: bool = Field(default=True, description="Include trading recommendations")

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker symbol format."""
        if not v.isupper():
            v = v.upper()
        if not v.replace('.', '').replace('-', '').isalnum():
            raise ValueError("Ticker must contain only alphanumeric characters, dots, and hyphens")
        return v

    @field_validator('indicators')
    @classmethod
    def validate_indicators(cls, v):
        """Validate indicator names."""
        # Use lazy import to avoid circular dependency
        try:
            from src.indicators.constants import validate_indicator_name
            for indicator in v:
                if not validate_indicator_name(indicator):
                    raise ValueError(f"Unknown indicator: {indicator}")
        except ImportError:
            # Fallback validation if constants module not available
            # Basic validation - check against known indicators
            known_indicators = set(TECHNICAL_INDICATORS.keys()) | set(FUNDAMENTAL_INDICATORS.keys())
            for indicator in v:
                canonical = get_canonical_name(indicator)
                if canonical not in known_indicators:
                    raise ValueError(f"Unknown indicator: {indicator}")
        return v


class BatchIndicatorRequest(BaseModel):
    """Request for batch indicator calculation."""
    tickers: List[TickerSymbol] = Field(min_length=1, description="List of ticker symbols")
    indicators: List[IndicatorName] = Field(min_length=1, description="List of indicator names")
    timeframe: TimeFrame = Field(default="1d", description="Data timeframe")
    period: Period = Field(default="2y", description="Data period")
    provider: Optional[ProviderName] = Field(default=None, description="Data provider")
    force_refresh: bool = Field(default=False, description="Force cache refresh")
    include_recommendations: bool = Field(default=True, description="Include trading recommendations")
    max_concurrent: int = Field(default=10, ge=1, le=50, description="Maximum concurrent requests")

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        """Validate ticker symbols."""
        validated = []
        for ticker in v:
            if not ticker.isupper():
                ticker = ticker.upper()
            if not ticker.replace('.', '').replace('-', '').isalnum():
                raise ValueError(f"Invalid ticker format: {ticker}")
            validated.append(ticker)
        return validated

    @field_validator('indicators')
    @classmethod
    def validate_indicators(cls, v):
        """Validate indicator names."""
        # Use lazy import to avoid circular dependency
        try:
            from src.indicators.constants import validate_indicator_name
            for indicator in v:
                if not validate_indicator_name(indicator):
                    raise ValueError(f"Unknown indicator: {indicator}")
        except ImportError:
            # Fallback validation if constants module not available
            # Basic validation - check against known indicators
            known_indicators = set(TECHNICAL_INDICATORS.keys()) | set(FUNDAMENTAL_INDICATORS.keys())
            for indicator in v:
                canonical = get_canonical_name(indicator)
                if canonical not in known_indicators:
                    raise ValueError(f"Unknown indicator: {indicator}")
        return v


class TickerIndicatorsRequest(BaseModel):
    """Request for ticker-based computation (tech + fundamentals)."""
    ticker: TickerSymbol = Field(min_length=1, description="Stock ticker symbol")
    timeframe: TimeFrame = Field(default="1d", description="OHLCV timeframe")
    period: Period = Field(default="1y", description="Data period")
    provider: Optional[ProviderName] = Field(default=None, description="Data provider")
    indicators: List[IndicatorName] = Field(min_length=1, description="Indicator names from registry")
    fillna: Optional[FillNASpec] = Field(default=None)
    warmup: Optional[WarmupSpec] = Field(default=None)
    include_recommendations: bool = Field(default=True)
    force_refresh: bool = Field(default=False)

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        """Validate ticker symbol format."""
        if not v.isupper():
            v = v.upper()
        if not v.replace('.', '').replace('-', '').isalnum():
            raise ValueError("Ticker must contain only alphanumeric characters, dots, and hyphens")
        return v

    @field_validator('indicators')
    @classmethod
    def validate_indicators(cls, v):
        """Validate indicator names."""
        # Use lazy import to avoid circular dependency
        try:
            from src.indicators.constants import validate_indicator_name
            for indicator in v:
                if not validate_indicator_name(indicator):
                    raise ValueError(f"Unknown indicator: {indicator}")
        except ImportError:
            # Fallback validation if constants module not available
            # Basic validation - check against known indicators
            known_indicators = set(TECHNICAL_INDICATORS.keys()) | set(FUNDAMENTAL_INDICATORS.keys())
            for indicator in v:
                canonical = get_canonical_name(indicator)
                if canonical not in known_indicators:
                    raise ValueError(f"Unknown indicator: {indicator}")
        return v


# Result containers
class IndicatorResultSet(BaseModel):
    """Result container for indicator calculations."""
    ticker: Optional[str] = Field(default=None, description="Ticker symbol")
    technical: Dict[str, IndicatorValue] = Field(default_factory=dict)
    fundamental: Dict[str, IndicatorValue] = Field(default_factory=dict)
    overall: Optional[Dict[str, Any]] = Field(default=None, description="Overall metrics")


class CacheEntry(BaseModel):
    """Cache entry for indicator data."""
    key: str = Field(min_length=1, description="Cache key")
    data: Any = Field(description="Cached data")
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: datetime = Field(description="Expiration time")
    access_count: int = Field(default=0, ge=0, description="Access count")
    last_accessed: datetime = Field(default_factory=datetime.now)


class PerformanceMetrics(BaseModel):
    """Performance metrics for indicator calculations."""
    operation: str = Field(min_length=1, description="Operation name")
    duration: float = Field(ge=0.0, description="Duration in seconds")
    cache_hit: bool = Field(description="Whether cache was hit")
    indicators_calculated: int = Field(ge=0, description="Number of indicators calculated")
    tickers_processed: int = Field(ge=0, description="Number of tickers processed")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Type aliases for better readability
OutputName = Union[str, Dict[str, str]]

# Constants for indicator names and descriptions
# Technical indicators (canonical names in lowercase)
TECHNICAL_INDICATORS = {
    "rsi": "Relative Strength Index",
    "macd": "Moving Average Convergence Divergence",
    "bbands": "Bollinger Bands",
    "stoch": "Stochastic Oscillator",
    "adx": "Average Directional Index",
    "plus_di": "Plus Directional Indicator",
    "minus_di": "Minus Directional Indicator",
    "sma": "Simple Moving Average",
    "ema": "Exponential Moving Average",
    "cci": "Commodity Channel Index",
    "roc": "Rate of Change",
    "mfi": "Money Flow Index",
    "williams_r": "Williams %R",
    "atr": "Average True Range",
    "obv": "On-Balance Volume",
    "adr": "Average Daily Range",
    "aroon": "Aroon Oscillator",
    "ichimoku": "Ichimoku Cloud",
    "sar": "Parabolic SAR",
    "super_trend": "Super Trend",
    "ad": "Accumulation/Distribution Line",
    "adosc": "Chaikin A/D Oscillator",
    "bop": "Balance of Power"
}

# Fundamental indicators (canonical names in lowercase)
FUNDAMENTAL_INDICATORS = {
    "pe_ratio": "Price-to-Earnings Ratio",
    "forward_pe": "Forward P/E Ratio",
    "pb_ratio": "Price-to-Book Ratio",
    "ps_ratio": "Price-to-Sales Ratio",
    "peg_ratio": "Price/Earnings-to-Growth Ratio",
    "roe": "Return on Equity",
    "roa": "Return on Assets",
    "debt_to_equity": "Debt-to-Equity Ratio",
    "current_ratio": "Current Ratio",
    "quick_ratio": "Quick Ratio",
    "operating_margin": "Operating Margin",
    "profit_margin": "Profit Margin",
    "revenue_growth": "Revenue Growth",
    "net_income_growth": "Net Income Growth",
    "free_cash_flow": "Free Cash Flow",
    "dividend_yield": "Dividend Yield",
    "payout_ratio": "Payout Ratio",
    "beta": "Beta",
    "market_cap": "Market Capitalization",
    "enterprise_value": "Enterprise Value",
    "ev_to_ebitda": "Enterprise Value to EBITDA"
}

# Legacy indicator names for backward compatibility
LEGACY_INDICATOR_NAMES = {
    # Technical indicators - legacy uppercase names
    "RSI": "rsi",
    "MACD": "macd",
    "MACD_SIGNAL": "macd",  # Multi-output indicator
    "MACD_HISTOGRAM": "macd",  # Multi-output indicator
    "BB_UPPER": "bbands",  # Multi-output indicator
    "BB_MIDDLE": "bbands",  # Multi-output indicator
    "BB_LOWER": "bbands",  # Multi-output indicator
    "SMA_FAST": "sma",
    "SMA_SLOW": "sma",
    "SMA_50": "sma",
    "SMA_200": "sma",
    "EMA_FAST": "ema",
    "EMA_SLOW": "ema",
    "EMA_12": "ema",
    "EMA_26": "ema",
    "ADX": "adx",
    "PLUS_DI": "plus_di",
    "MINUS_DI": "minus_di",
    "ATR": "atr",
    "STOCH_K": "stoch",  # Multi-output indicator
    "STOCH_D": "stoch",  # Multi-output indicator
    "WILLIAMS_R": "williams_r",
    "Williams_R": "williams_r",
    "CCI": "cci",
    "ROC": "roc",
    "MFI": "mfi",
    "OBV": "obv",
    "ADR": "adr",
    "AROON_UP": "aroon",  # Multi-output indicator
    "AROON_DOWN": "aroon",  # Multi-output indicator
    "ICHIMOKU": "ichimoku",
    "SAR": "sar",
    "SUPER_TREND": "super_trend",
    "AD": "ad",
    "ADOSC": "adosc",
    "BOP": "bop",

    # Fundamental indicators - legacy uppercase names
    "PE_RATIO": "pe_ratio",
    "FORWARD_PE": "forward_pe",
    "PB_RATIO": "pb_ratio",
    "PS_RATIO": "ps_ratio",
    "PEG_RATIO": "peg_ratio",
    "ROE": "roe",
    "ROA": "roa",
    "DEBT_TO_EQUITY": "debt_to_equity",
    "CURRENT_RATIO": "current_ratio",
    "QUICK_RATIO": "quick_ratio",
    "OPERATING_MARGIN": "operating_margin",
    "PROFIT_MARGIN": "profit_margin",
    "REVENUE_GROWTH": "revenue_growth",
    "NET_INCOME_GROWTH": "net_income_growth",
    "FREE_CASH_FLOW": "free_cash_flow",
    "DIVIDEND_YIELD": "dividend_yield",
    "PAYOUT_RATIO": "payout_ratio",
    "BETA": "beta",
    "MARKET_CAP": "market_cap",
    "ENTERPRISE_VALUE": "enterprise_value",
    "EV_TO_EBITDA": "ev_to_ebitda"
}

# Combined indicators dictionary
ALL_INDICATORS = {**TECHNICAL_INDICATORS, **FUNDAMENTAL_INDICATORS}

# Legacy combined dictionary for backward compatibility
INDICATOR_DESCRIPTIONS = {
    **{k.upper(): v for k, v in TECHNICAL_INDICATORS.items()},
    **{legacy: ALL_INDICATORS[canonical] for legacy, canonical in LEGACY_INDICATOR_NAMES.items() if canonical in TECHNICAL_INDICATORS}
}

FUNDAMENTAL_INDICATORS_LEGACY = {
    **{k.upper(): v for k, v in FUNDAMENTAL_INDICATORS.items()},
    **{legacy: ALL_INDICATORS[canonical] for legacy, canonical in LEGACY_INDICATOR_NAMES.items() if canonical in FUNDAMENTAL_INDICATORS}
}

ALL_INDICATORS_LEGACY = {**INDICATOR_DESCRIPTIONS, **FUNDAMENTAL_INDICATORS_LEGACY}


def get_canonical_name(indicator_name: str) -> str:
    """Get canonical name for an indicator, handling legacy names."""
    # First check if it's already canonical (lowercase)
    if indicator_name.lower() in ALL_INDICATORS:
        return indicator_name.lower()

    # Check legacy mapping
    canonical = LEGACY_INDICATOR_NAMES.get(indicator_name)
    if canonical:
        return canonical

    # Default to lowercase
    return indicator_name.lower()


def get_indicator_description(indicator_name: str) -> Optional[str]:
    """Get description for an indicator by name."""
    canonical_name = get_canonical_name(indicator_name)
    return ALL_INDICATORS.get(canonical_name)


# Re-export everything for backward compatibility
__all__ = [
    # Core models
    "RecommendationType",
    "IndicatorCategory",
    "Recommendation",
    "IndicatorValue",
    "IndicatorResult",
    "CompositeRecommendation",
    "IndicatorSet",

    # Request models
    "FillNASpec",
    "WarmupSpec",
    "IndicatorSpec",
    "IndicatorBatchConfig",
    "IndicatorCalculationRequest",
    "BatchIndicatorRequest",
    "TickerIndicatorsRequest",

    # Result models
    "IndicatorResultSet",
    "CacheEntry",
    "PerformanceMetrics",

    # Type aliases
    "OutputName",

    # Constants and utilities
    "TECHNICAL_INDICATORS",
    "FUNDAMENTAL_INDICATORS",
    "ALL_INDICATORS",
    "LEGACY_INDICATOR_NAMES",
    "INDICATOR_DESCRIPTIONS",
    "FUNDAMENTAL_INDICATORS_LEGACY",
    "ALL_INDICATORS_LEGACY",
    "get_canonical_name",
    "get_indicator_description"
]
