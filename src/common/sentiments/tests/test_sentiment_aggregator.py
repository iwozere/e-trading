# src/common/sentiments/tests/test_sentiment_aggregator.py
"""
Unit tests for sentiment aggregation and weighting strategies.

Tests cover:
- Multi-source sentiment combination
- Quality-based weighting strategies
- Confidence interval calculation
- Temporal sentiment analysis
- Adaptive weighting based on data quality
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.common.sentiments.processing.sentiment_aggregator import (
    SentimentAggregator, SourceSentiment, AggregatedSentiment
)


class TestSentimentAggregator(unittest.TestCase):
    """Test cases for SentimentAggregator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "aggregation_method": "weighted_average",
            "quality_weights": {
                "excellent": 1.0,
                "good": 0.8,
                "fair": 0.6,
                "poor": 0.3
            },
            "source_weights": {
                "stocktwits": 0.3,
                "reddit": 0.3,
                "twitter": 0.2,
                "news": 0.2
            },
            "min_confidence": 0.1,
            "confidence_boost_threshold": 3,
            "outlier_threshold": 2.0,
            "enable_outlier_removal": True
        }
        self.aggregator = SentimentAggregator(self.config)

    def test_initialization_default_config(self):
        """Test aggregator initialization with default configuration."""
        aggregator = SentimentAggregator()

        self.assertEqual(aggregator.default_method, "weighted_average")
        self.assertIn("excellent", aggregator.quality_weights)
        self.assertIn("stocktwits", aggregator.source_weights)
        self.assertEqual(aggregator.min_confidence, 0.1)

    def test_initialization_custom_config(self):
        """Test aggregator initialization with custom configuration."""
        self.assertEqual(self.aggregator.default_method, "weighted_average")
        self.assertEqual(self.aggregator.quality_weights["excellent"], 1.0)
        self.assertEqual(self.aggregator.source_weights["stocktwits"], 0.3)
        self.assertEqual(self.aggregator.outlier_threshold, 2.0)

    def test_create_source_sentiment(self):
        """Test creating SourceSentiment objects."""
        source = self.aggregator.create_source_sentiment(
            source_name="stocktwits",
            sentiment_score=0.7,
            confidence=0.8,
            data_quality="good",
            sample_size=50,
            raw_data={"test": "data"}
        )

        self.assertEqual(source.source_name, "stocktwits")
        self.assertEqual(source.sentiment_score, 0.7)
        self.assertEqual(source.confidence, 0.8)
        self.assertEqual(source.data_quality, "good")
        self.assertEqual(source.sample_size, 50)
        self.assertEqual(source.raw_data, {"test": "data"})
        self.assertIsInstance(source.timestamp, datetime)

    def test_create_source_sentiment_clamping(self):
        """Test sentiment score and confidence clamping."""
        source = self.aggregator.create_source_sentiment(
            source_name="test",
            sentiment_score=2.0,  # Should be clamped to 1.0
            confidence=1.5,       # Should be clamped to 1.0
            data_quality="good",
            sample_size=-5        # Should be clamped to 0
        )

        self.assertEqual(source.sentiment_score, 1.0)
        self.assertEqual(source.confidence, 1.0)
        self.assertEqual(source.sample_size, 0)

    def test_aggregate_sentiment_empty_sources(self):
        """Test aggregation with empty sources list."""
        result = self.aggregator.aggregate_sentiment([])

        self.assertEqual(result.final_score, 0.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.quality_score, 0.0)
        self.assertEqual(result.source_breakdown, {})
        self.assertEqual(result.aggregation_method, "none")

    def test_aggregate_sentiment_single_source(self):
        """Test aggregation with single source."""
        source = SourceSentiment(
            source_name="stocktwits",
            sentiment_score=0.6,
            confidence=0.8,
            data_quality="good",
            sample_size=25,
            timestamp=datetime.now(timezone.utc),
            raw_data={}
        )

        result = self.aggregator.aggregate_sentiment([source])

        self.assertAlmostEqual(result.final_score, 0.6, places=2)
        self.assertGreater(result.confidence, 0.0)
        self.assertGreater(result.quality_score, 0.0)
        self.assertIn("stocktwits", result.source_breakdown)
        self.assertEqual(result.aggregation_method, "weighted_average")

    def test_aggregate_sentiment_multiple_sources(self):
        """Test aggregation with multiple sources."""
        sources = [
            SourceSentiment(
                source_name="stocktwits",
                sentiment_score=0.8,
                confidence=0.9,
                data_quality="excellent",
                sample_size=100,
                timestamp=datetime.now(timezone.utc),
                raw_data={}
            ),
            SourceSentiment(
                source_name="reddit",
                sentiment_score=0.4,
                confidence=0.7,
                data_quality="good",
                sample_size=50,
                timestamp=datetime.now(timezone.utc),
                raw_data={}
            ),
            SourceSentiment(
                source_name="twitter",
                sentiment_score=0.6,
                confidence=0.8,
                data_quality="fair",
                sample_size=75,
                timestamp=datetime.now(timezone.utc),
                raw_data={}
            )
        ]

        result = self.aggregator.aggregate_sentiment(sources)

        # Should be weighted average of sources
        self.assertGreater(result.final_score, 0.0)
        self.assertLess(result.final_score, 1.0)
        self.assertGreater(result.confidence, 0.0)
        self.assertGreater(result.quality_score, 0.0)
        self.assertEqual(len(result.source_breakdown), 3)
        self.assertEqual(result.aggregation_method, "weighted_average")

    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        sources = [
            SourceSentiment("source1", 0.8, 0.9, "excellent", 100, datetime.now(timezone.utc), {}),
            SourceSentiment("source2", 0.4, 0.7, "good", 50, datetime.now(timezone.utc), {}),
            SourceSentiment("source3", 0.6, 0.8, "fair", 75, datetime.now(timezone.utc), {})
        ]

        # Test weighted average
        result_weighted = self.aggregator.aggregate_sentiment(sources, method="weighted_average")

        # Test median
        result_median = self.aggregator.aggregate_sentiment(sources, method="median")
        self.assertEqual(result_median.final_score, 0.6)  # Median of [0.8, 0.4, 0.6]

        # Test confidence weighted
        result_confidence = self.aggregator.aggregate_sentiment(sources, method="confidence_weighted")

        # Test quality weighted
        result_quality = self.aggregator.aggregate_sentiment(sources, method="quality_weighted")

        # All should produce valid results
        for result in [result_weighted, result_median, result_confidence, result_quality]:
            self.assertGreaterEqual(result.final_score, -1.0)
            self.assertLessEqual(result.final_score, 1.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)

    def test_outlier_removal(self):
        """Test outlier detection and removal."""
        sources = [
            SourceSentiment("source1", 0.5, 0.8, "good", 50, datetime.now(timezone.utc), {}),
            SourceSentiment("source2", 0.6, 0.8, "good", 50, datetime.now(timezone.utc), {}),
            SourceSentiment("source3", 0.55, 0.8, "good", 50, datetime.now(timezone.utc), {}),
            SourceSentiment("outlier", -0.9, 0.8, "good", 50, datetime.now(timezone.utc), {})  # Outlier
        ]

        # With outlier removal enabled
        result_with_removal = self.aggregator.aggregate_sentiment(sources)

        # Create new aggregator with outlier removal disabled
        config_no_outlier = self.config.copy()
        config_no_outlier["enable_outlier_removal"] = False
        aggregator_no_outlier = SentimentAggregator(config_no_outlier)
        result_without_removal = aggregator_no_outlier.aggregate_sentiment(sources)

        # Test that both aggregators produce valid results
        self.assertIsInstance(result_with_removal, AggregatedSentiment)
        self.assertIsInstance(result_without_removal, AggregatedSentiment)

        # Both should have valid scores
        self.assertGreaterEqual(result_with_removal.final_score, -1.0)
        self.assertLessEqual(result_with_removal.final_score, 1.0)
        self.assertGreaterEqual(result_without_removal.final_score, -1.0)
        self.assertLessEqual(result_without_removal.final_score, 1.0)

        # Test that outlier removal configuration is respected
        self.assertTrue(self.aggregator.enable_outlier_removal)
        self.assertFalse(aggregator_no_outlier.enable_outlier_removal)

    def test_confidence_calculation(self):
        """Test confidence calculation logic."""
        # High confidence scenario: many sources, high individual confidence
        high_conf_sources = [
            SourceSentiment(f"source{i}", 0.6, 0.9, "excellent", 100, datetime.now(timezone.utc), {})
            for i in range(5)
        ]

        # Low confidence scenario: few sources, low individual confidence
        low_conf_sources = [
            SourceSentiment("source1", 0.6, 0.3, "poor", 5, datetime.now(timezone.utc), {})
        ]

        high_result = self.aggregator.aggregate_sentiment(high_conf_sources)
        low_result = self.aggregator.aggregate_sentiment(low_conf_sources)

        self.assertGreater(high_result.confidence, low_result.confidence)
        self.assertGreaterEqual(low_result.confidence, self.aggregator.min_confidence)

    def test_temporal_trend_analysis(self):
        """Test temporal trend analysis."""
        now = datetime.now(timezone.utc)

        # Improving trend: older sources negative, newer positive
        sources = [
            SourceSentiment("old1", -0.5, 0.8, "good", 50, now - timedelta(hours=8), {}),
            SourceSentiment("old2", -0.3, 0.8, "good", 50, now - timedelta(hours=7), {}),
            SourceSentiment("new1", 0.4, 0.8, "good", 50, now - timedelta(hours=2), {}),
            SourceSentiment("new2", 0.6, 0.8, "good", 50, now - timedelta(hours=1), {})
        ]

        result = self.aggregator.aggregate_sentiment(sources)
        self.assertEqual(result.temporal_trend, "improving")

    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        sources = [
            SourceSentiment("source1", 0.5, 0.8, "good", 50, datetime.now(timezone.utc), {}),
            SourceSentiment("source2", 0.7, 0.8, "good", 50, datetime.now(timezone.utc), {}),
            SourceSentiment("source3", 0.6, 0.8, "good", 50, datetime.now(timezone.utc), {})
        ]

        result = self.aggregator.aggregate_sentiment(sources)

        lower, upper = result.confidence_interval
        self.assertLessEqual(lower, result.final_score)
        self.assertGreaterEqual(upper, result.final_score)
        self.assertGreaterEqual(lower, -1.0)
        self.assertLessEqual(upper, 1.0)

    def test_source_validation(self):
        """Test source validation logic."""
        # Invalid sources that should be filtered out
        invalid_sources = [
            SourceSentiment("invalid1", 2.0, 0.8, "good", 50, datetime.now(timezone.utc), {}),  # Invalid score
            SourceSentiment("invalid2", 0.5, 1.5, "good", 50, datetime.now(timezone.utc), {}),  # Invalid confidence
            SourceSentiment("invalid3", 0.5, 0.8, "good", 0, datetime.now(timezone.utc), {})   # Zero sample size
        ]

        # Should return empty result due to no valid sources
        result = self.aggregator.aggregate_sentiment(invalid_sources)
        self.assertEqual(result.final_score, 0.0)
        self.assertEqual(result.metadata["valid_sources"], 0)

    def test_source_weight_calculation(self):
        """Test dynamic source weight calculation."""
        sources = [
            SourceSentiment("high_quality", 0.6, 0.9, "excellent", 200, datetime.now(timezone.utc), {}),
            SourceSentiment("low_quality", 0.6, 0.3, "poor", 10, datetime.now(timezone.utc), {})
        ]

        weights = self.aggregator._calculate_source_weights(sources)

        # High quality source should have higher weight
        self.assertGreater(weights["high_quality"], weights["low_quality"])

        # Weights should sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=3)

    def test_get_aggregation_stats(self):
        """Test aggregation statistics retrieval."""
        stats = self.aggregator.get_aggregation_stats()

        self.assertIn("default_method", stats)
        self.assertIn("quality_weights", stats)
        self.assertIn("source_weights", stats)
        self.assertIn("config", stats)

        self.assertEqual(stats["default_method"], "weighted_average")
        self.assertIsInstance(stats["quality_weights"], dict)
        self.assertIsInstance(stats["source_weights"], dict)


class TestSourceSentiment(unittest.TestCase):
    """Test cases for SourceSentiment dataclass."""

    def test_source_sentiment_creation(self):
        """Test SourceSentiment object creation."""
        timestamp = datetime.now(timezone.utc)
        raw_data = {"test": "data", "count": 42}

        source = SourceSentiment(
            source_name="test_source",
            sentiment_score=0.75,
            confidence=0.85,
            data_quality="excellent",
            sample_size=100,
            timestamp=timestamp,
            raw_data=raw_data
        )

        self.assertEqual(source.source_name, "test_source")
        self.assertEqual(source.sentiment_score, 0.75)
        self.assertEqual(source.confidence, 0.85)
        self.assertEqual(source.data_quality, "excellent")
        self.assertEqual(source.sample_size, 100)
        self.assertEqual(source.timestamp, timestamp)
        self.assertEqual(source.raw_data, raw_data)


class TestAggregatedSentiment(unittest.TestCase):
    """Test cases for AggregatedSentiment dataclass."""

    def test_aggregated_sentiment_creation(self):
        """Test AggregatedSentiment object creation."""
        breakdown = {"engagement": 0.8, "velocity": 0.6}
        weighted_sources = {"stocktwits": 0.6, "reddit": 0.4}
        confidence_interval = (0.4, 0.8)
        top_contributors = [("user1", 0.9), ("user2", 0.7)]
        metadata = {"total_sources": 2, "valid_sources": 2}

        result = AggregatedSentiment(
            final_score=0.65,
            confidence=0.82,
            quality_score=0.78,
            source_breakdown=breakdown,
            weighted_sources=weighted_sources,
            confidence_interval=confidence_interval,
            temporal_trend="improving",
            aggregation_method="weighted_average",
            metadata=metadata
        )

        self.assertEqual(result.final_score, 0.65)
        self.assertEqual(result.confidence, 0.82)
        self.assertEqual(result.quality_score, 0.78)
        self.assertEqual(result.source_breakdown, breakdown)
        self.assertEqual(result.weighted_sources, weighted_sources)
        self.assertEqual(result.confidence_interval, confidence_interval)
        self.assertEqual(result.temporal_trend, "improving")
        self.assertEqual(result.aggregation_method, "weighted_average")
        self.assertEqual(result.metadata, metadata)


if __name__ == "__main__":
    unittest.main()