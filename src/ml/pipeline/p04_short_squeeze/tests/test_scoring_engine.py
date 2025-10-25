"""
Unit tests for the scoring engine.

Tests metric normalization, scoring algorithms, and score validation.
"""

import unittest
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.scoring_engine import ScoringEngine
from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, TransientMetrics, Candidate, ScoredCandidate
)
from src.ml.pipeline.p04_short_squeeze.config.data_classes import (
    ScoringConfig, DeepScanWeights
)
from src.data.db.models.model_short_squeeze import CandidateSource


class TestScoringEngine(unittest.TestCase):
    """Test cases for ScoringEngine class."""

    def setUp(self):
        """Set up test fixtures."""
        self.scoring_config = ScoringConfig(
            normalization_method="minmax",
            score_bounds=(0.0, 1.0),
            weight_validation=True
        )

        self.deep_scan_weights = DeepScanWeights(
            volume_spike=0.35,
            sentiment_24h=0.25,
            call_put_ratio=0.20,
            borrow_fee=0.20
        )

        self.scoring_engine = ScoringEngine(self.scoring_config, self.deep_scan_weights)

        # Sample structural metrics
        self.structural_metrics = StructuralMetrics(
            short_interest_pct=0.25,
            days_to_cover=8.5,
            float_shares=50_000_000,
            avg_volume_14d=1_000_000,
            market_cap=500_000_000
        )

        # Sample transient metrics
        self.transient_metrics = TransientMetrics(
            volume_spike=3.5,
            call_put_ratio=2.1,
            sentiment_24h=0.6,
            borrow_fee_pct=0.15
        )

        # Sample candidate
        self.candidate = Candidate(
            ticker="TSLA",
            screener_score=0.75,
            structural_metrics=self.structural_metrics,
            last_updated=datetime.now(),
            source=CandidateSource.SCREENER
        )

    def test_scoring_engine_initialization(self):
        """Test scoring engine initialization and validation."""
        # Valid configuration should work
        engine = ScoringEngine(self.scoring_config, self.deep_scan_weights)
        self.assertIsNotNone(engine)

        # Invalid weights should raise error
        invalid_weights = DeepScanWeights(
            volume_spike=0.5,
            sentiment_24h=0.3,
            call_put_ratio=0.3,
            borrow_fee=0.2  # Total = 1.3, should fail validation
        )

        with self.assertRaises(ValueError) as context:
            ScoringEngine(self.scoring_config, invalid_weights)

        self.assertIn("weights must sum to 1.0", str(context.exception))

    def test_calculate_squeeze_score(self):
        """Test squeeze score calculation."""
        score = self.scoring_engine.calculate_squeeze_score(
            self.structural_metrics,
            self.transient_metrics
        )

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Score should be reasonable for good metrics
        self.assertGreater(score, 0.3)  # Should be above minimum for decent metrics

    def test_calculate_structural_score(self):
        """Test structural score calculation."""
        structural_score = self.scoring_engine._calculate_structural_score(self.structural_metrics)

        # Score should be between 0 and 1
        self.assertGreaterEqual(structural_score, 0.0)
        self.assertLessEqual(structural_score, 1.0)

        # Test with high short interest
        high_si_metrics = StructuralMetrics(
            short_interest_pct=0.45,  # Very high
            days_to_cover=12.0,       # High
            float_shares=30_000_000,  # Smaller float
            avg_volume_14d=2_000_000, # High volume
            market_cap=1_000_000_000
        )

        high_score = self.scoring_engine._calculate_structural_score(high_si_metrics)
        self.assertGreater(high_score, structural_score)

    def test_extract_transient_metrics(self):
        """Test transient metrics extraction."""
        metrics_dict = self.scoring_engine._extract_transient_metrics(self.transient_metrics)

        expected_keys = ['volume_spike', 'sentiment_24h', 'call_put_ratio', 'borrow_fee']
        self.assertEqual(set(metrics_dict.keys()), set(expected_keys))

        self.assertEqual(metrics_dict['volume_spike'], 3.5)
        self.assertEqual(metrics_dict['sentiment_24h'], 0.6)
        self.assertEqual(metrics_dict['call_put_ratio'], 2.1)
        self.assertEqual(metrics_dict['borrow_fee'], 0.15)

    def test_extract_transient_metrics_with_none_values(self):
        """Test transient metrics extraction with None values."""
        metrics_with_none = TransientMetrics(
            volume_spike=2.0,
            call_put_ratio=None,
            sentiment_24h=0.3,
            borrow_fee_pct=None
        )

        metrics_dict = self.scoring_engine._extract_transient_metrics(metrics_with_none)

        self.assertEqual(metrics_dict['call_put_ratio'], 0.0)
        self.assertEqual(metrics_dict['borrow_fee'], 0.0)

    def test_minmax_normalize_volume_spike(self):
        """Test min-max normalization for volume spike."""
        # Test normal range
        self.assertAlmostEqual(
            self.scoring_engine._minmax_normalize('volume_spike', 1.0), 0.0
        )
        self.assertAlmostEqual(
            self.scoring_engine._minmax_normalize('volume_spike', 10.0), 1.0
        )
        self.assertAlmostEqual(
            self.scoring_engine._minmax_normalize('volume_spike', 5.5), 0.5
        )

        # Test clamping
        self.assertEqual(
            self.scoring_engine._minmax_normalize('volume_spike', 0.5), 0.0
        )
        self.assertEqual(
            self.scoring_engine._minmax_normalize('volume_spike', 15.0), 1.0
        )

    def test_minmax_normalize_sentiment(self):
        """Test min-max normalization for sentiment."""
        # Sentiment is already normalized [-1, 1] -> [0, 1]
        self.assertAlmostEqual(
            self.scoring_engine._minmax_normalize('sentiment_24h', -1.0), 0.0
        )
        self.assertAlmostEqual(
            self.scoring_engine._minmax_normalize('sentiment_24h', 1.0), 1.0
        )
        self.assertAlmostEqual(
            self.scoring_engine._minmax_normalize('sentiment_24h', 0.0), 0.5
        )

    def test_sigmoid_normalize(self):
        """Test sigmoid normalization."""
        # Test volume spike sigmoid
        sigmoid_config = ScoringConfig(
            normalization_method="sigmoid",
            score_bounds=(0.0, 1.0),
            weight_validation=True
        )
        sigmoid_engine = ScoringEngine(sigmoid_config, self.deep_scan_weights)

        # Values around center should be near 0.5
        center_val = sigmoid_engine._sigmoid_normalize('volume_spike', 3.0)
        self.assertGreater(center_val, 0.4)
        self.assertLess(center_val, 0.6)

        # Higher values should approach 1
        high_val = sigmoid_engine._sigmoid_normalize('volume_spike', 10.0)
        self.assertGreater(high_val, 0.8)

    def test_normalize_metrics(self):
        """Test complete metrics normalization."""
        metrics = {
            'volume_spike': 3.5,
            'sentiment_24h': 0.6,
            'call_put_ratio': 2.1,
            'borrow_fee': 0.15
        }

        normalized = self.scoring_engine.normalize_metrics(metrics)

        # All normalized values should be between 0 and 1
        for key, value in normalized.items():
            self.assertGreaterEqual(value, 0.0, f"{key} normalized value below 0")
            self.assertLessEqual(value, 1.0, f"{key} normalized value above 1")

    def test_apply_weights(self):
        """Test weight application to normalized metrics."""
        normalized_metrics = {
            'volume_spike': 0.6,
            'sentiment_24h': 0.7,
            'call_put_ratio': 0.5,
            'borrow_fee': 0.4
        }

        weighted_score = self.scoring_engine._apply_weights(normalized_metrics)

        # Calculate expected score manually
        expected = (
            0.35 * 0.6 +  # volume_spike
            0.25 * 0.7 +  # sentiment_24h
            0.20 * 0.5 +  # call_put_ratio
            0.20 * 0.4    # borrow_fee
        )

        self.assertAlmostEqual(weighted_score, expected, places=3)

    def test_validate_score_bounds(self):
        """Test score bounds validation."""
        # Normal score should pass through
        self.assertEqual(self.scoring_engine._validate_score_bounds(0.5), 0.5)

        # Score below minimum should be clamped
        self.assertEqual(self.scoring_engine._validate_score_bounds(-0.1), 0.0)

        # Score above maximum should be clamped
        self.assertEqual(self.scoring_engine._validate_score_bounds(1.5), 1.0)

    def test_score_candidate(self):
        """Test complete candidate scoring."""
        scored_candidate = self.scoring_engine.score_candidate(
            self.candidate,
            self.transient_metrics
        )

        self.assertIsInstance(scored_candidate, ScoredCandidate)
        self.assertEqual(scored_candidate.candidate, self.candidate)
        self.assertEqual(scored_candidate.transient_metrics, self.transient_metrics)

        # Score should be valid
        self.assertGreaterEqual(scored_candidate.squeeze_score, 0.0)
        self.assertLessEqual(scored_candidate.squeeze_score, 1.0)

    def test_get_score_breakdown(self):
        """Test detailed score breakdown."""
        breakdown = self.scoring_engine.get_score_breakdown(
            self.structural_metrics,
            self.transient_metrics
        )

        # Check required keys
        required_keys = [
            'final_score', 'structural_score', 'transient_score',
            'raw_transient_metrics', 'normalized_transient_metrics',
            'weights', 'combination_weights'
        ]

        for key in required_keys:
            self.assertIn(key, breakdown)

        # Check score validity
        self.assertGreaterEqual(breakdown['final_score'], 0.0)
        self.assertLessEqual(breakdown['final_score'], 1.0)

        # Check weights
        self.assertEqual(breakdown['weights']['volume_spike'], 0.35)
        self.assertEqual(breakdown['combination_weights']['transient'], 0.6)
        self.assertEqual(breakdown['combination_weights']['structural'], 0.4)

    def test_scoring_consistency(self):
        """Test that scoring is consistent across multiple calls."""
        score1 = self.scoring_engine.calculate_squeeze_score(
            self.structural_metrics,
            self.transient_metrics
        )

        score2 = self.scoring_engine.calculate_squeeze_score(
            self.structural_metrics,
            self.transient_metrics
        )

        self.assertEqual(score1, score2)

    def test_scoring_with_extreme_values(self):
        """Test scoring with extreme metric values."""
        # Extreme high values
        extreme_high_transient = TransientMetrics(
            volume_spike=50.0,  # Very high
            call_put_ratio=10.0,  # Very high
            sentiment_24h=1.0,  # Maximum
            borrow_fee_pct=1.0  # Very high
        )

        high_score = self.scoring_engine.calculate_squeeze_score(
            self.structural_metrics,
            extreme_high_transient
        )

        # Extreme low values
        extreme_low_transient = TransientMetrics(
            volume_spike=0.5,  # Very low
            call_put_ratio=0.1,  # Very low
            sentiment_24h=-1.0,  # Minimum
            borrow_fee_pct=0.0  # Zero
        )

        low_score = self.scoring_engine.calculate_squeeze_score(
            self.structural_metrics,
            extreme_low_transient
        )

        # High extreme should score higher than low extreme
        self.assertGreater(high_score, low_score)

        # Both should be within bounds
        self.assertGreaterEqual(high_score, 0.0)
        self.assertLessEqual(high_score, 1.0)
        self.assertGreaterEqual(low_score, 0.0)
        self.assertLessEqual(low_score, 1.0)

    def test_invalid_normalization_method(self):
        """Test invalid normalization method returns minimum score."""
        invalid_config = ScoringConfig(
            normalization_method="invalid_method",
            score_bounds=(0.0, 1.0),
            weight_validation=True
        )

        invalid_engine = ScoringEngine(invalid_config, self.deep_scan_weights)

        # Should return minimum score on error instead of raising
        score = invalid_engine.calculate_squeeze_score(
            self.structural_metrics,
            self.transient_metrics
        )

        # Should return minimum score (0.0) on error
        self.assertEqual(score, 0.0)


if __name__ == '__main__':
    unittest.main()