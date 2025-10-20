"""
Unified indicator models - imports from consolidated model module.

This module now imports all models from the unified src.model.indicators module
to maintain backward compatibility while consolidating all data structures.
"""

# Import all models from the unified module
from src.model.indicators import (
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
