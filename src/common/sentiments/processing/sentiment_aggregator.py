# src/common/sentiments/processing/sentiment_aggregator.py
"""
Sentiment aggregation and weighting strategies for multi-source sentiment analysis.

This module provides sophisticated sentiment aggregation with:
- Multi-source sentiment combination
- Quality-based weighting strategies
- Confidence interval calculation
- Temporal sentiment analysis
- Adaptive weighting based on data quality
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import math
import statistics
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

@dataclass
class SourceSentiment:
    """Sentiment data from a single source."""
    source_name: str
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    data_quality: str  # 'excellent', 'good', 'fair', 'poor'
    sample_size: int
    timestamp: datetime
    raw_data: Dict[str, Any]

@dataclass
class AggregatedSentiment:
    """Final aggregated sentiment result."""
    final_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    quality_score: float  # 0.0 to 1.0
    source_breakdown: Dict[str, Dict[str, float]]
    weighted_sources: Dict[str, float]
    confidence_interval: Tuple[float, float]
    temporal_trend: Optional[str]  # 'improving', 'declining', 'stable'
    aggregation_method: str
    metadata: Dict[str, Any]

class SentimentAggregator:
    """
    Advanced sentiment aggregation system for multi-source sentiment analysis.

    Features:
    - Quality-based source weighting
    - Confidence interval calculation
    - Temporal trend analysis
    - Adaptive weighting strategies
    - Outlier detection and handling
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the sentiment aggregator.

        Args:
            config: Configuration dictionary with aggregation parameters
        """
        self.config = config or {}

        # Aggregation method
        self.default_method = self.config.get("aggregation_method", "weighted_average")

        # Quality weights
        self.quality_weights = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "poor": 0.3
        }
        self.quality_weights.update(self.config.get("quality_weights", {}))

        # Source base weights (can be overridden by quality)
        self.source_weights = {
            "stocktwits": 0.25,
            "reddit": 0.25,
            "twitter": 0.20,
            "news": 0.15,
            "discord": 0.10,
            "huggingface": 0.05  # ML enhancement, not primary source
        }
        self.source_weights.update(self.config.get("source_weights", {}))

        # Confidence calculation parameters
        self.min_confidence = self.config.get("min_confidence", 0.1)
        self.confidence_boost_threshold = self.config.get("confidence_boost_threshold", 3)
        self.sample_size_weight = self.config.get("sample_size_weight", 0.2)

        # Temporal analysis parameters
        self.trend_window_hours = self.config.get("trend_window_hours", 6)
        self.trend_threshold = self.config.get("trend_threshold", 0.1)

        # Outlier detection
        self.outlier_threshold = self.config.get("outlier_threshold", 2.0)  # Standard deviations
        self.enable_outlier_removal = self.config.get("enable_outlier_removal", True)

    def aggregate_sentiment(self, sources: List[SourceSentiment],
                          method: Optional[str] = None) -> AggregatedSentiment:
        """
        Aggregate sentiment from multiple sources.

        Args:
            sources: List of sentiment data from different sources
            method: Aggregation method to use (overrides default)

        Returns:
            AggregatedSentiment with comprehensive analysis
        """
        if not sources:
            return self._create_empty_result()

        method = method or self.default_method

        # Filter and validate sources
        valid_sources = self._validate_sources(sources)
        if not valid_sources:
            return self._create_empty_result()

        # Remove outliers if enabled
        if self.enable_outlier_removal and len(valid_sources) > 2:
            valid_sources = self._remove_outliers(valid_sources)

        # Calculate weights for each source
        source_weights = self._calculate_source_weights(valid_sources)

        # Aggregate based on method
        if method == "weighted_average":
            final_score = self._weighted_average_aggregation(valid_sources, source_weights)
        elif method == "median":
            final_score = self._median_aggregation(valid_sources)
        elif method == "confidence_weighted":
            final_score = self._confidence_weighted_aggregation(valid_sources)
        elif method == "quality_weighted":
            final_score = self._quality_weighted_aggregation(valid_sources)
        else:
            _logger.warning("Unknown aggregation method: %s, using weighted_average", method)
            final_score = self._weighted_average_aggregation(valid_sources, source_weights)

        # Calculate confidence
        confidence = self._calculate_confidence(valid_sources, source_weights)

        # Calculate quality score
        quality_score = self._calculate_quality_score(valid_sources)

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(valid_sources, final_score)

        # Analyze temporal trend
        temporal_trend = self._analyze_temporal_trend(valid_sources)

        # Create source breakdown
        source_breakdown = self._create_source_breakdown(valid_sources, source_weights)

        # Create metadata
        metadata = {
            "total_sources": len(sources),
            "valid_sources": len(valid_sources),
            "total_sample_size": sum(s.sample_size for s in valid_sources),
            "aggregation_timestamp": datetime.now(timezone.utc).isoformat(),
            "outliers_removed": len(sources) - len(valid_sources) if self.enable_outlier_removal else 0
        }

        return AggregatedSentiment(
            final_score=final_score,
            confidence=confidence,
            quality_score=quality_score,
            source_breakdown=source_breakdown,
            weighted_sources={s.source_name: source_weights.get(s.source_name, 0.0) for s in valid_sources},
            confidence_interval=confidence_interval,
            temporal_trend=temporal_trend,
            aggregation_method=method,
            metadata=metadata
        )

    def _validate_sources(self, sources: List[SourceSentiment]) -> List[SourceSentiment]:
        """Validate and filter sources."""
        valid_sources = []

        for source in sources:
            # Check for valid sentiment score
            if not (-1.0 <= source.sentiment_score <= 1.0):
                _logger.debug("Invalid sentiment score for %s: %f", source.source_name, source.sentiment_score)
                continue

            # Check for valid confidence
            if not (0.0 <= source.confidence <= 1.0):
                _logger.debug("Invalid confidence for %s: %f", source.source_name, source.confidence)
                continue

            # Check for minimum sample size
            min_sample_size = self.config.get("min_sample_size", 1)
            if source.sample_size < min_sample_size:
                _logger.debug("Insufficient sample size for %s: %d", source.source_name, source.sample_size)
                continue

            valid_sources.append(source)

        return valid_sources

    def _remove_outliers(self, sources: List[SourceSentiment]) -> List[SourceSentiment]:
        """Remove outlier sentiment scores using statistical methods."""
        if len(sources) <= 2:
            return sources

        scores = [s.sentiment_score for s in sources]
        mean_score = statistics.mean(scores)

        try:
            std_dev = statistics.stdev(scores)
        except statistics.StatisticsError:
            # All scores are the same
            return sources

        if std_dev == 0:
            return sources

        # Remove sources that are more than threshold standard deviations away
        filtered_sources = []
        for source in sources:
            z_score = abs(source.sentiment_score - mean_score) / std_dev
            if z_score <= self.outlier_threshold:
                filtered_sources.append(source)
            else:
                _logger.debug("Removing outlier source %s with z-score %.2f",
                            source.source_name, z_score)

        # Ensure we don't remove too many sources
        if len(filtered_sources) < max(2, len(sources) // 2):
            _logger.debug("Too many outliers detected, keeping original sources")
            return sources

        return filtered_sources

    def _calculate_source_weights(self, sources: List[SourceSentiment]) -> Dict[str, float]:
        """Calculate dynamic weights for each source."""
        weights = {}

        for source in sources:
            # Start with base weight
            base_weight = self.source_weights.get(source.source_name, 0.1)

            # Apply quality multiplier
            quality_multiplier = self.quality_weights.get(source.data_quality, 0.5)

            # Apply confidence multiplier
            confidence_multiplier = 0.5 + (source.confidence * 0.5)  # 0.5 to 1.0 range

            # Apply sample size multiplier
            sample_size_multiplier = min(2.0, 1.0 + math.log10(max(1, source.sample_size)) * 0.1)

            # Calculate final weight
            final_weight = base_weight * quality_multiplier * confidence_multiplier * sample_size_multiplier
            weights[source.source_name] = final_weight

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}

        return weights

    def _weighted_average_aggregation(self, sources: List[SourceSentiment],
                                    weights: Dict[str, float]) -> float:
        """Aggregate using weighted average."""
        weighted_sum = 0.0
        total_weight = 0.0

        for source in sources:
            weight = weights.get(source.source_name, 0.0)
            weighted_sum += source.sentiment_score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def _median_aggregation(self, sources: List[SourceSentiment]) -> float:
        """Aggregate using median (robust to outliers)."""
        scores = [s.sentiment_score for s in sources]
        return statistics.median(scores)

    def _confidence_weighted_aggregation(self, sources: List[SourceSentiment]) -> float:
        """Aggregate using confidence-based weighting."""
        weighted_sum = 0.0
        total_weight = 0.0

        for source in sources:
            weight = source.confidence
            weighted_sum += source.sentiment_score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def _quality_weighted_aggregation(self, sources: List[SourceSentiment]) -> float:
        """Aggregate using quality-based weighting."""
        weighted_sum = 0.0
        total_weight = 0.0

        for source in sources:
            weight = self.quality_weights.get(source.data_quality, 0.5)
            weighted_sum += source.sentiment_score * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0

    def _calculate_confidence(self, sources: List[SourceSentiment],
                            weights: Dict[str, float]) -> float:
        """Calculate overall confidence in the aggregated result."""
        if not sources:
            return 0.0

        # Base confidence from source count
        source_count_factor = min(1.0, len(sources) / self.confidence_boost_threshold)

        # Weighted average of source confidences
        weighted_confidence = 0.0
        total_weight = 0.0

        for source in sources:
            weight = weights.get(source.source_name, 0.0)
            weighted_confidence += source.confidence * weight
            total_weight += weight

        if total_weight > 0:
            avg_confidence = weighted_confidence / total_weight
        else:
            avg_confidence = 0.0

        # Sample size factor
        total_samples = sum(s.sample_size for s in sources)
        sample_factor = min(1.0, math.log10(max(1, total_samples)) / 3.0)  # Log scale

        # Quality factor
        quality_scores = [self.quality_weights.get(s.data_quality, 0.5) for s in sources]
        avg_quality = sum(quality_scores) / len(quality_scores)

        # Combine factors
        final_confidence = (
            source_count_factor * 0.3 +
            avg_confidence * 0.4 +
            sample_factor * 0.2 +
            avg_quality * 0.1
        )

        return max(self.min_confidence, min(1.0, final_confidence))

    def _calculate_quality_score(self, sources: List[SourceSentiment]) -> float:
        """Calculate overall data quality score."""
        if not sources:
            return 0.0

        quality_scores = []
        for source in sources:
            base_quality = self.quality_weights.get(source.data_quality, 0.5)

            # Adjust for sample size
            sample_adjustment = min(1.2, 1.0 + math.log10(max(1, source.sample_size)) * 0.05)

            # Adjust for confidence
            confidence_adjustment = 0.8 + (source.confidence * 0.2)

            adjusted_quality = base_quality * sample_adjustment * confidence_adjustment
            quality_scores.append(adjusted_quality)

        return min(1.0, sum(quality_scores) / len(quality_scores))

    def _calculate_confidence_interval(self, sources: List[SourceSentiment],
                                     final_score: float) -> Tuple[float, float]:
        """Calculate confidence interval for the aggregated sentiment."""
        if len(sources) < 2:
            # Wide interval for single source
            margin = 0.3
            return (max(-1.0, final_score - margin), min(1.0, final_score + margin))

        scores = [s.sentiment_score for s in sources]

        try:
            std_dev = statistics.stdev(scores)
        except statistics.StatisticsError:
            # All scores are the same
            return (final_score, final_score)

        # Calculate margin of error (simplified confidence interval)
        # Using t-distribution approximation for small samples
        n = len(scores)
        if n < 30:
            t_value = 2.0  # Approximate t-value for 95% confidence
        else:
            t_value = 1.96  # Z-value for 95% confidence

        margin_of_error = t_value * (std_dev / math.sqrt(n))

        lower_bound = max(-1.0, final_score - margin_of_error)
        upper_bound = min(1.0, final_score + margin_of_error)

        return (lower_bound, upper_bound)

    def _analyze_temporal_trend(self, sources: List[SourceSentiment]) -> Optional[str]:
        """Analyze temporal trend in sentiment."""
        if len(sources) < 2:
            return None

        # Sort sources by timestamp
        sorted_sources = sorted(sources, key=lambda s: s.timestamp)

        # Split into early and late periods
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.trend_window_hours)

        early_sources = [s for s in sorted_sources if s.timestamp < cutoff_time]
        late_sources = [s for s in sorted_sources if s.timestamp >= cutoff_time]

        if not early_sources or not late_sources:
            return None

        # Calculate average sentiment for each period
        early_avg = sum(s.sentiment_score for s in early_sources) / len(early_sources)
        late_avg = sum(s.sentiment_score for s in late_sources) / len(late_sources)

        # Determine trend
        difference = late_avg - early_avg

        if abs(difference) < self.trend_threshold:
            return "stable"
        elif difference > 0:
            return "improving"
        else:
            return "declining"

    def _create_source_breakdown(self, sources: List[SourceSentiment],
                               weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Create detailed breakdown of source contributions."""
        breakdown = {}

        for source in sources:
            breakdown[source.source_name] = {
                "sentiment_score": source.sentiment_score,
                "confidence": source.confidence,
                "weight": weights.get(source.source_name, 0.0),
                "sample_size": source.sample_size,
                "quality_score": self.quality_weights.get(source.data_quality, 0.5),
                "data_quality": source.data_quality
            }

        return breakdown

    def _create_empty_result(self) -> AggregatedSentiment:
        """Create empty aggregated sentiment result."""
        return AggregatedSentiment(
            final_score=0.0,
            confidence=0.0,
            quality_score=0.0,
            source_breakdown={},
            weighted_sources={},
            confidence_interval=(0.0, 0.0),
            temporal_trend=None,
            aggregation_method="none",
            metadata={"total_sources": 0, "valid_sources": 0}
        )

    def create_source_sentiment(self, source_name: str, sentiment_score: float,
                              confidence: float, data_quality: str,
                              sample_size: int, raw_data: Optional[Dict] = None) -> SourceSentiment:
        """
        Helper method to create SourceSentiment objects.

        Args:
            source_name: Name of the sentiment source
            sentiment_score: Sentiment score (-1.0 to 1.0)
            confidence: Confidence in the score (0.0 to 1.0)
            data_quality: Quality rating ('excellent', 'good', 'fair', 'poor')
            sample_size: Number of samples used for this sentiment
            raw_data: Optional raw data from the source

        Returns:
            SourceSentiment object
        """
        return SourceSentiment(
            source_name=source_name,
            sentiment_score=max(-1.0, min(1.0, sentiment_score)),
            confidence=max(0.0, min(1.0, confidence)),
            data_quality=data_quality,
            sample_size=max(0, sample_size),
            timestamp=datetime.now(timezone.utc),
            raw_data=raw_data or {}
        )

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about aggregation configuration."""
        return {
            "default_method": self.default_method,
            "quality_weights": self.quality_weights,
            "source_weights": self.source_weights,
            "config": self.config
        }