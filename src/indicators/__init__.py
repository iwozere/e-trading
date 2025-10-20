"""
Unified Indicators Module

This module provides a consolidated interface for all indicator calculations,
including technical and fundamental analysis with multiple backend support.
"""

from src.indicators.service import UnifiedIndicatorService, IndicatorService, get_unified_indicator_service
from src.indicators.registry import INDICATOR_META, get_canonical_name, get_indicator_meta
from src.indicators.config_manager import UnifiedConfigManager, get_config_manager
from src.indicators.recommendation_engine import RecommendationEngine
from src.indicators.models import (
    IndicatorBatchConfig, IndicatorResultSet, IndicatorSpec,
    IndicatorValue, TickerIndicatorsRequest
)

__all__ = [
    # Main service classes
    "UnifiedIndicatorService",
    "IndicatorService",  # Backward compatibility
    "get_unified_indicator_service",

    # Registry and metadata
    "INDICATOR_META",
    "get_canonical_name",
    "get_indicator_meta",

    # Configuration management
    "UnifiedConfigManager",
    "get_config_manager",

    # Recommendation engine
    "RecommendationEngine",

    # Data models
    "IndicatorBatchConfig",
    "IndicatorResultSet",
    "IndicatorSpec",
    "IndicatorValue",
    "TickerIndicatorsRequest",
]