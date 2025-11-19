"""
Unit tests for sentiment integration in p04_short_squeeze pipeline.

Tests cover:
- TransientMetrics with enhanced sentiment fields
- Scoring engine with virality and mention growth
- Daily deep scan sentiment batch collection
- Configuration loading and validation
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
from dataclasses import dataclass

from src.ml.pipeline.p04_short_squeeze.core.models import TransientMetrics
from src.ml.pipeline.p04_short_squeeze.core.scoring_engine import ScoringEngine
from src.ml.pipeline.p04_short_squeeze.config.data_classes import (
    ScoringConfig, DeepScanWeights, SentimentConfig,
    SentimentProviders, SentimentBatching, SentimentWeights
)


# ============================================================================
# Model Tests
# ============================================================================

class TestTransientMetricsEnhanced:
    """Test enhanced TransientMetrics with sentiment fields."""

    def test_transient_metrics_with_enhanced_sentiment(self):
        """Test creating TransientMetrics with all sentiment fields."""
        metrics = TransientMetrics(
            volume_spike=3.5,
            call_put_ratio=2.0,
            sentiment_24h=0.7,
            borrow_fee_pct=15.5,
            mentions_24h=150,
            mentions_growth_7d=2.5,
            virality_index=0.75,
            bot_pct=0.15,
            sentiment_data_quality={'stocktwits': 'ok', 'reddit': 'ok'}
        )

        assert metrics.volume_spike == 3.5
        assert metrics.sentiment_24h == 0.7
        assert metrics.mentions_24h == 150
        assert metrics.mentions_growth_7d == 2.5
        assert metrics.virality_index == 0.75
        assert metrics.bot_pct == 0.15
        assert metrics.sentiment_data_quality['stocktwits'] == 'ok'

    def test_transient_metrics_default_sentiment_fields(self):
        """Test TransientMetrics with default sentiment values."""
        metrics = TransientMetrics(
            volume_spike=2.0,
            call_put_ratio=None,
            sentiment_24h=0.0,
            borrow_fee_pct=None
        )

        assert metrics.mentions_24h == 0
        assert metrics.mentions_growth_7d is None
        assert metrics.virality_index == 0.0
        assert metrics.bot_pct == 0.0
        assert metrics.sentiment_data_quality == {}

    def test_transient_metrics_validation_virality_range(self):
        """Test virality index must be in [0, 1] range."""
        with pytest.raises(ValueError, match="Virality index must be between 0 and 1"):
            TransientMetrics(
                volume_spike=1.0,
                sentiment_24h=0.0,
                virality_index=1.5  # Invalid: > 1.0
            )

    def test_transient_metrics_validation_bot_pct_range(self):
        """Test bot percentage must be in [0, 1] range."""
        with pytest.raises(ValueError, match="Bot percentage must be between 0 and 1"):
            TransientMetrics(
                volume_spike=1.0,
                sentiment_24h=0.0,
                bot_pct=-0.1  # Invalid: < 0
            )

    def test_transient_metrics_validation_mentions_positive(self):
        """Test mentions count must be non-negative."""
        with pytest.raises(ValueError, match="Mentions count must be non-negative"):
            TransientMetrics(
                volume_spike=1.0,
                sentiment_24h=0.0,
                mentions_24h=-10  # Invalid: negative
            )

    def test_transient_metrics_validation_growth_floor(self):
        """Test mentions growth cannot be less than -1 (100% shrinkage)."""
        with pytest.raises(ValueError, match="Mentions growth must be >= -1"):
            TransientMetrics(
                volume_spike=1.0,
                sentiment_24h=0.0,
                mentions_growth_7d=-1.5  # Invalid: < -1
            )


# ============================================================================
# Scoring Engine Tests
# ============================================================================

class TestScoringEngineEnhanced:
    """Test scoring engine with enhanced sentiment metrics."""

    @pytest.fixture
    def scoring_config(self):
        """Create test scoring configuration."""
        return ScoringConfig(
            normalization_method="minmax",
            score_bounds=(0.0, 1.0),
            weight_validation=True
        )

    @pytest.fixture
    def deep_scan_weights(self):
        """Create test deep scan weights."""
        return DeepScanWeights(
            volume_spike=0.35,
            sentiment_24h=0.25,
            call_put_ratio=0.20,
            borrow_fee=0.20
        )

    @pytest.fixture
    def scoring_engine(self, scoring_config, deep_scan_weights):
        """Create scoring engine instance."""
        return ScoringEngine(scoring_config, deep_scan_weights)

    def test_calculate_virality_score_no_bots(self, scoring_engine):
        """Test virality score with no bot activity."""
        score = scoring_engine.calculate_virality_score_with_bot_penalty(
            virality_index=0.8,
            bot_pct=0.0
        )

        assert score == 0.8  # No penalty

    def test_calculate_virality_score_with_bots(self, scoring_engine):
        """Test virality score with bot penalty."""
        score = scoring_engine.calculate_virality_score_with_bot_penalty(
            virality_index=0.8,
            bot_pct=0.5  # 50% bots
        )

        # With 50% bots, penalty = 1.0 - (0.5 * 0.6) = 0.7
        # Score = 0.8 * 0.7 = 0.56
        assert abs(score - 0.56) < 0.01

    def test_calculate_virality_score_max_bots(self, scoring_engine):
        """Test virality score with maximum bot activity."""
        score = scoring_engine.calculate_virality_score_with_bot_penalty(
            virality_index=1.0,
            bot_pct=1.0  # 100% bots
        )

        # Max 60% penalty: 1.0 * (1.0 - 0.6) = 0.4
        assert abs(score - 0.4) < 0.01

    def test_apply_mentions_growth_boost_no_growth(self, scoring_engine):
        """Test no boost for low growth."""
        base_score = 0.5
        boosted = scoring_engine.apply_mentions_growth_boost(
            base_score=base_score,
            mentions_growth_7d=0.5  # 50% growth (below threshold)
        )

        assert boosted == base_score  # No boost

    def test_apply_mentions_growth_boost_moderate(self, scoring_engine):
        """Test moderate boost for 2x growth."""
        base_score = 0.5
        boosted = scoring_engine.apply_mentions_growth_boost(
            base_score=base_score,
            mentions_growth_7d=2.0  # 200% growth
        )

        # 2x growth = +5% boost
        expected = 0.5 * 1.05
        assert abs(boosted - expected) < 0.01

    def test_apply_mentions_growth_boost_high(self, scoring_engine):
        """Test high boost for 5x growth."""
        base_score = 0.6
        boosted = scoring_engine.apply_mentions_growth_boost(
            base_score=base_score,
            mentions_growth_7d=5.0  # 500% growth
        )

        # 5x growth = +10% boost
        expected = 0.6 * 1.10
        assert abs(boosted - expected) < 0.01

    def test_apply_mentions_growth_boost_extreme(self, scoring_engine):
        """Test extreme boost for 10x+ growth."""
        base_score = 0.7
        boosted = scoring_engine.apply_mentions_growth_boost(
            base_score=base_score,
            mentions_growth_7d=15.0  # 1500% growth
        )

        # 10x+ growth = +15% boost
        expected = 0.7 * 1.15
        assert abs(boosted - expected) < 0.01

    def test_apply_mentions_growth_boost_capped_at_one(self, scoring_engine):
        """Test boost is capped at 1.0."""
        base_score = 0.95
        boosted = scoring_engine.apply_mentions_growth_boost(
            base_score=base_score,
            mentions_growth_7d=20.0  # Extreme growth
        )

        # Should cap at 1.0, not exceed
        assert boosted <= 1.0

    def test_normalize_virality_index(self, scoring_engine):
        """Test virality index normalization (already normalized)."""
        metrics = {'virality_index': 0.75}
        normalized = scoring_engine._minmax_normalize('virality_index', 0.75)

        assert normalized == 0.75  # Already in [0, 1]

    def test_normalize_mentions_growth(self, scoring_engine):
        """Test mentions growth normalization."""
        # 0 growth should map to 0.5 (middle of range)
        normalized_zero = scoring_engine._minmax_normalize('mentions_growth_7d', 0.0)
        assert abs(normalized_zero - 0.0909) < 0.01  # (0 - (-1)) / (10 - (-1)) ≈ 0.09

        # 5x growth
        normalized_5x = scoring_engine._minmax_normalize('mentions_growth_7d', 5.0)
        assert abs(normalized_5x - 0.545) < 0.01  # (5 - (-1)) / (10 - (-1)) ≈ 0.545

        # 10x growth (max expected)
        normalized_10x = scoring_engine._minmax_normalize('mentions_growth_7d', 10.0)
        assert normalized_10x == 1.0

    def test_normalize_bot_pct(self, scoring_engine):
        """Test bot percentage normalization (already normalized)."""
        normalized = scoring_engine._minmax_normalize('bot_pct', 0.5)

        assert normalized == 0.5  # Already in [0, 1]


# ============================================================================
# Configuration Tests
# ============================================================================

class TestSentimentConfiguration:
    """Test sentiment configuration classes."""

    def test_sentiment_config_defaults(self):
        """Test SentimentConfig with default values."""
        config = SentimentConfig()

        assert config.providers.stocktwits is True
        assert config.providers.reddit_pushshift is True
        assert config.providers.hf_enabled is False
        assert config.batching.concurrency == 8
        assert config.weights.stocktwits == 0.4
        assert config.cache.enabled is True
        assert config.cache.ttl_seconds == 1800

    def test_sentiment_config_custom(self):
        """Test SentimentConfig with custom values."""
        config = SentimentConfig(
            providers=SentimentProviders(
                stocktwits=False,
                hf_enabled=True
            ),
            batching=SentimentBatching(
                concurrency=16,
                batch_size=100
            ),
            weights=SentimentWeights(
                stocktwits=0.5,
                reddit=0.5
            )
        )

        assert config.providers.stocktwits is False
        assert config.providers.hf_enabled is True
        assert config.batching.concurrency == 16
        assert config.batching.batch_size == 100
        assert config.weights.stocktwits == 0.5


# ============================================================================
# Integration Tests (with mocks)
# ============================================================================

@pytest.mark.asyncio
class TestDailyDeepScanSentiment:
    """Test daily deep scan with sentiment integration."""

    @pytest.fixture
    def mock_sentiment_features(self):
        """Mock SentimentFeatures response."""
        @dataclass
        class MockSentimentFeatures:
            ticker: str
            mentions_24h: int
            unique_authors_24h: int
            mentions_growth_7d: float
            positive_ratio_24h: float
            sentiment_score_24h: float
            sentiment_normalized: float
            virality_index: float
            bot_pct: float
            data_quality: dict
            raw_payload: dict

        return MockSentimentFeatures(
            ticker='AAPL',
            mentions_24h=150,
            unique_authors_24h=80,
            mentions_growth_7d=2.5,
            positive_ratio_24h=0.7,
            sentiment_score_24h=0.65,
            sentiment_normalized=0.65,
            virality_index=0.75,
            bot_pct=0.15,
            data_quality={'stocktwits': 'ok', 'reddit': 'ok'},
            raw_payload={}
        )

    @patch('src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan.collect_sentiment_batch')
    async def test_batch_sentiment_collection(self, mock_collect, mock_sentiment_features):
        """Test batch sentiment collection."""
        from src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan import DailyDeepScan
        from src.ml.pipeline.p04_short_squeeze.core.models import Candidate, StructuralMetrics
        from src.ml.pipeline.p04_short_squeeze.config.data_classes import DeepScanConfig

        # Setup mock
        mock_collect.return_value = {'AAPL': mock_sentiment_features}

        # Create scanner with sentiment config
        scanner = DailyDeepScan(
            fmp_downloader=MagicMock(),
            finnhub_downloader=MagicMock(),
            config=DeepScanConfig(),
            sentiment_config=SentimentConfig()
        )

        # Create test candidate
        candidate = Candidate(
            ticker='AAPL',
            screener_score=0.6,
            structural_metrics=StructuralMetrics(
                short_interest_pct=0.2,
                days_to_cover=5.0,
                float_shares=1000000,
                avg_volume_14d=500000,
                market_cap=1000000000
            ),
            last_updated=datetime.now()
        )

        # Call batch collection
        metrics = {}
        sentiment_map = scanner._collect_batch_sentiment([candidate], metrics)

        # Verify sentiment was collected
        assert 'AAPL' in sentiment_map
        assert sentiment_map['AAPL'].mentions_24h == 150
        assert sentiment_map['AAPL'].virality_index == 0.75

    def test_fallback_to_finnhub_sentiment(self):
        """Test fallback to Finnhub when sentiment module unavailable."""
        from src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan import DailyDeepScan
        from src.ml.pipeline.p04_short_squeeze.config.data_classes import DeepScanConfig

        # Create scanner WITHOUT sentiment config (should use Finnhub)
        scanner = DailyDeepScan(
            fmp_downloader=MagicMock(),
            finnhub_downloader=MagicMock(),
            config=DeepScanConfig(),
            sentiment_config=None  # No sentiment config
        )

        assert scanner.use_enhanced_sentiment is False

    def test_transient_metrics_from_sentiment_features(self, mock_sentiment_features):
        """Test creating TransientMetrics from sentiment features."""
        from src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan import DailyDeepScan
        from src.ml.pipeline.p04_short_squeeze.config.data_classes import DeepScanConfig

        scanner = DailyDeepScan(
            fmp_downloader=MagicMock(),
            finnhub_downloader=MagicMock(),
            config=DeepScanConfig(),
            sentiment_config=SentimentConfig()
        )

        # Mock volume spike
        with patch.object(scanner, '_calculate_volume_spike_ratio', return_value=3.0):
            with patch.object(scanner, '_get_call_put_ratio', return_value=2.0):
                with patch.object(scanner, '_get_borrow_fee_percentage', return_value=15.0):
                    metrics = {}
                    transient = scanner.calculate_transient_metrics(
                        'AAPL', metrics, mock_sentiment_features
                    )

                    assert transient is not None
                    assert transient.sentiment_24h == 0.65
                    assert transient.mentions_24h == 150
                    assert transient.mentions_growth_7d == 2.5
                    assert transient.virality_index == 0.75
                    assert transient.bot_pct == 0.15
                    assert transient.sentiment_data_quality['stocktwits'] == 'ok'


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
