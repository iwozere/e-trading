"""
Recommendation types and data models.

These types are re-exported from src.indicators.models for convenience.
Eventually, these types should be moved here to avoid circular dependencies.
"""

from src.indicators.models import (
    CompositeRecommendation,
    IndicatorCategory,
    IndicatorSet,
    Recommendation,
    RecommendationType,
)

__all__ = [
    "Recommendation",
    "RecommendationType",
    "IndicatorCategory",
    "IndicatorSet",
    "CompositeRecommendation",
]
