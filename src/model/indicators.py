"""
Backward compatibility module for indicator models.

This module provides backward compatibility by importing all indicator models
from the src.indicators module where they now reside.
"""

# Import all models from the indicators module for backward compatibility
from src.indicators.models import (
    # Core models
    RecommendationType,
    IndicatorCategory,
    Recommendation,
    IndicatorValue,
    IndicatorResult,
    CompositeRecommendation,
    IndicatorSet,

    # Request models
    FillNASpec,
    WarmupSpec,
    IndicatorSpec,
    IndicatorBatchConfig,
    IndicatorCalculationRequest,
    BatchIndicatorRequest,
    TickerIndicatorsRequest,

    # Result models
    IndicatorResultSet,
    CacheEntry,
    PerformanceMetrics,

    # Type aliases
    OutputName,

    # Constants and utilities
    TECHNICAL_INDICATORS,
    FUNDAMENTAL_INDICATORS,
    ALL_INDICATORS,
    LEGACY_INDICATOR_NAMES,
    INDICATOR_DESCRIPTIONS,
    FUNDAMENTAL_INDICATORS_LEGACY,
    ALL_INDICATORS_LEGACY,
    get_canonical_name,
    get_indicator_description
)

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