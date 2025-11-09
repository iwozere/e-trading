"""
Unit tests for Daily Deep Scan module.

Tests the daily deep scan functionality including transient metrics calculation,
scoring, and data storage.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, date

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan import DailyDeepScan, DeepScanResults, create_daily_deep_scan
from src.ml.pipeline.p04_short_squeeze.config.data_classes import DeepScanConfig, DeepScanMetrics, DeepScanWeights
from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, TransientMetrics, Candidate, ScoredCandidate, CandidateSource
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestDailyDeepScan(unittest.TestCase):
    """Test cases for Daily Deep Scan."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fmp_downloader = Mock()
        self.mock_finnhub_downloader = Mock()

        self.deep_scan_config = DeepScanConfig(
            batch_size=5,
            api_delay_seconds=0.1,
            metrics=DeepScanMetrics(
                volume_lookback_days=14,
                sentiment_lookback_hours=24,
                options_min_volume=100
            ),
            scoring=DeepScanWeights(
                volume_spike=0.35,
                sentiment_24h=0.25,
                call_put_ratio=0.20,
                borrow_fee=0.20
            )
        )

        self.daily_deep_scan = DailyDeepScan(
            self.mock_fmp_downloader,
            self.mock_finnhub_downloader,
            self.deep_scan_config
        )

    def create_test_candidate(self, ticker='TEST', score=0.7):
        """Create a test candidate for testing."""
        structural_metrics = StructuralMetrics(
            short_interest_pct=0.2,
            days_to_cover=10.0,
            float_shares=50_000_000,
            avg_volume_14d=500_000,
            market_cap=1_000_000_000
        )

        return Candidate(
            ticker=ticker,
            screener_score=score,
            structural_metrics=structural_metrics,
            last_updated=datetime.now(),
            source=CandidateSource.SCREENER
        )

    def test_calculate_volume_spike_ratio_success(self):
        """Test successful volume spike calculation."""
        import pandas as pd

        # Mock OHLCV data with volume spike
        volumes = [100_000] * 14 + [500_000]  # Last day has 5x volume spike
        mock_df = pd.DataFrame({
            'volume': volumes,
            'timestamp': pd.date_range(start='2024-01-01', periods=15, freq='D')
        })
        self.mock_fmp_downloader.get_ohlcv.return_value = mock_df

        # Calculate volume spike
        volume_spike = self.daily_deep_scan._calculate_volume_spike_ratio('TEST')

        # Assertions
        self.assertIsNotNone(volume_spike)
        self.assertAlmostEqual(volume_spike, 5.0, places=1)  # 500k / 100k = 5.0

    def test_calculate_volume_spike_ratio_insufficient_data(self):
        """Test volume spike calculation with insufficient data."""
        import pandas as pd

        # Mock insufficient data
        mock_df = pd.DataFrame({
            'volume': [100_000] * 5,  # Only 5 days
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='D')
        })
        self.mock_fmp_downloader.get_ohlcv.return_value = mock_df

        # Calculate volume spike
        volume_spike = self.daily_deep_scan._calculate_volume_spike_ratio('TEST')

        # Should return None for insufficient data
        self.assertIsNone(volume_spike)

    def test_get_sentiment_score_success(self):
        """Test successful sentiment score retrieval."""
        # Mock sentiment data
        mock_sentiment = {
            'sentiment_score_24h': 0.6,  # Positive sentiment
            'bullish_percent_24h': 70,
            'bearish_percent_24h': 30
        }
        self.mock_finnhub_downloader.aggregate_24h_sentiment.return_value = mock_sentiment

        # Get sentiment score
        sentiment_score = self.daily_deep_scan._get_sentiment_score('TEST')

        # Assertions
        self.assertIsNotNone(sentiment_score)
        self.assertEqual(sentiment_score, 0.6)

    def test_get_sentiment_score_no_data(self):
        """Test sentiment score with no data."""
        # Mock no sentiment data
        self.mock_finnhub_downloader.aggregate_24h_sentiment.return_value = None

        # Get sentiment score
        sentiment_score = self.daily_deep_scan._get_sentiment_score('TEST')

        # Should return None
        self.assertIsNone(sentiment_score)

    def test_get_call_put_ratio_success(self):
        """Test successful call/put ratio calculation."""
        # Mock options data
        mock_options = {
            'total_call_volume': 1000,
            'total_put_volume': 500,
            'call_put_volume_ratio': 2.0
        }
        self.mock_finnhub_downloader.get_options_data.return_value = mock_options
        self.mock_finnhub_downloader.calculate_call_put_ratio.return_value = 2.0

        # Get call/put ratio
        call_put_ratio = self.daily_deep_scan._get_call_put_ratio('TEST')

        # Assertions
        self.assertIsNotNone(call_put_ratio)
        self.assertEqual(call_put_ratio, 2.0)

    def test_get_call_put_ratio_no_data(self):
        """Test call/put ratio with no options data."""
        # Mock no options data
        self.mock_finnhub_downloader.get_options_data.return_value = None

        # Get call/put ratio
        call_put_ratio = self.daily_deep_scan._get_call_put_ratio('TEST')

        # Should return None
        self.assertIsNone(call_put_ratio)

    def test_get_borrow_fee_percentage_success(self):
        """Test successful borrow fee retrieval."""
        # Mock borrow rates data
        mock_borrow = {
            'fee_rate_percentage': 5.5,  # 5.5% borrow fee
            'available_shares': 1000000
        }
        self.mock_finnhub_downloader.get_borrow_rates_data.return_value = mock_borrow

        # Get borrow fee
        borrow_fee = self.daily_deep_scan._get_borrow_fee_percentage('TEST')

        # Assertions
        self.assertIsNotNone(borrow_fee)
        self.assertEqual(borrow_fee, 5.5)

    def test_get_borrow_fee_percentage_no_data(self):
        """Test borrow fee with no data."""
        # Mock no borrow data
        self.mock_finnhub_downloader.get_borrow_rates_data.return_value = None

        # Get borrow fee
        borrow_fee = self.daily_deep_scan._get_borrow_fee_percentage('TEST')

        # Should return None
        self.assertIsNone(borrow_fee)

    def test_calculate_transient_metrics_success(self):
        """Test successful transient metrics calculation."""
        # Mock all data sources
        import pandas as pd

        # Mock volume data
        volumes = [100_000] * 14 + [300_000]  # 3x volume spike
        mock_df = pd.DataFrame({
            'volume': volumes,
            'timestamp': pd.date_range(start='2024-01-01', periods=15, freq='D')
        })
        self.mock_fmp_downloader.get_ohlcv.return_value = mock_df

        # Mock sentiment data
        self.mock_finnhub_downloader.aggregate_24h_sentiment.return_value = {
            'sentiment_score_24h': 0.4
        }

        # Mock options data
        self.mock_finnhub_downloader.get_options_data.return_value = {
            'total_call_volume': 1500,
            'total_put_volume': 500
        }
        self.mock_finnhub_downloader.calculate_call_put_ratio.return_value = 3.0

        # Mock borrow rates data
        self.mock_finnhub_downloader.get_borrow_rates_data.return_value = {
            'fee_rate_percentage': 8.0
        }

        # Calculate transient metrics
        metrics_dict = {'valid_volume_data': 0, 'valid_sentiment_data': 0,
                       'valid_options_data': 0, 'valid_borrow_rates': 0,
                       'api_calls_fmp': 0, 'api_calls_finnhub': 0}
        transient_metrics = self.daily_deep_scan.calculate_transient_metrics('TEST', metrics_dict)

        # Assertions
        self.assertIsNotNone(transient_metrics)
        self.assertAlmostEqual(transient_metrics.volume_spike, 3.0, places=1)
        self.assertEqual(transient_metrics.sentiment_24h, 0.4)
        self.assertEqual(transient_metrics.call_put_ratio, 3.0)
        self.assertEqual(transient_metrics.borrow_fee_pct, 8.0)

        # Check metrics tracking
        self.assertEqual(metrics_dict['valid_volume_data'], 1)
        self.assertEqual(metrics_dict['valid_sentiment_data'], 1)
        self.assertEqual(metrics_dict['valid_options_data'], 1)
        self.assertEqual(metrics_dict['valid_borrow_rates'], 1)

    def test_calculate_transient_metrics_partial_data(self):
        """Test transient metrics calculation with partial data."""
        # Mock only volume data (others return None)
        import pandas as pd

        volumes = [100_000] * 14 + [200_000]  # 2x volume spike
        mock_df = pd.DataFrame({
            'volume': volumes,
            'timestamp': pd.date_range(start='2024-01-01', periods=15, freq='D')
        })
        self.mock_fmp_downloader.get_ohlcv.return_value = mock_df

        # Mock no other data
        self.mock_finnhub_downloader.aggregate_24h_sentiment.return_value = None
        self.mock_finnhub_downloader.get_options_data.return_value = None
        self.mock_finnhub_downloader.get_borrow_rates_data.return_value = None

        # Calculate transient metrics
        metrics_dict = {'valid_volume_data': 0, 'valid_sentiment_data': 0,
                       'valid_options_data': 0, 'valid_borrow_rates': 0,
                       'api_calls_fmp': 0, 'api_calls_finnhub': 0}
        transient_metrics = self.daily_deep_scan.calculate_transient_metrics('TEST', metrics_dict)

        # Assertions
        self.assertIsNotNone(transient_metrics)
        self.assertAlmostEqual(transient_metrics.volume_spike, 2.0, places=1)
        self.assertEqual(transient_metrics.sentiment_24h, 0.0)  # Default neutral
        self.assertIsNone(transient_metrics.call_put_ratio)
        self.assertIsNone(transient_metrics.borrow_fee_pct)

    def test_calculate_preliminary_squeeze_score(self):
        """Test preliminary squeeze score calculation."""
        # Create test transient metrics
        transient_metrics = TransientMetrics(
            volume_spike=3.0,      # 3x volume spike
            call_put_ratio=2.5,    # 2.5 call/put ratio
            sentiment_24h=0.6,     # Positive sentiment
            borrow_fee_pct=7.0     # 7% borrow fee
        )

        # Calculate preliminary score
        screener_score = 0.7
        squeeze_score = self.daily_deep_scan._calculate_preliminary_squeeze_score(
            screener_score, transient_metrics
        )

        # Assertions
        self.assertGreaterEqual(squeeze_score, 0.0)
        self.assertLessEqual(squeeze_score, 1.0)
        self.assertGreater(squeeze_score, screener_score)  # Should be higher due to good transient metrics

    def test_scan_candidate_success(self):
        """Test successful candidate scanning."""
        # Create test candidate
        candidate = self.create_test_candidate('TEST', 0.6)

        # Mock transient metrics calculation
        with patch.object(self.daily_deep_scan, 'calculate_transient_metrics') as mock_calc:
            mock_transient = TransientMetrics(
                volume_spike=2.5,
                call_put_ratio=1.8,
                sentiment_24h=0.3,
                borrow_fee_pct=4.5
            )
            mock_calc.return_value = mock_transient

            # Scan candidate
            metrics_dict = {'successful_scans': 0, 'failed_scans': 0}
            scored_candidate = self.daily_deep_scan._scan_candidate(candidate, metrics_dict)

            # Assertions
            self.assertIsNotNone(scored_candidate)
            self.assertIsInstance(scored_candidate, ScoredCandidate)
            self.assertEqual(scored_candidate.candidate, candidate)
            self.assertEqual(scored_candidate.transient_metrics, mock_transient)
            self.assertGreaterEqual(scored_candidate.squeeze_score, 0.0)
            self.assertLessEqual(scored_candidate.squeeze_score, 1.0)

    def test_scan_candidate_no_transient_data(self):
        """Test candidate scanning with no transient data."""
        # Create test candidate
        candidate = self.create_test_candidate('TEST', 0.6)

        # Mock failed transient metrics calculation
        with patch.object(self.daily_deep_scan, 'calculate_transient_metrics') as mock_calc:
            mock_calc.return_value = None

            # Scan candidate
            metrics_dict = {'successful_scans': 0, 'failed_scans': 0}
            scored_candidate = self.daily_deep_scan._scan_candidate(candidate, metrics_dict)

            # Should return None
            self.assertIsNone(scored_candidate)

    def test_process_batch(self):
        """Test batch processing of candidates."""
        # Create test candidates
        candidates = [
            self.create_test_candidate('TEST1', 0.7),
            self.create_test_candidate('TEST2', 0.8),
            self.create_test_candidate('TEST3', 0.6)
        ]

        # Mock successful scanning for first two, failure for third
        def mock_scan_candidate(candidate, metrics):
            if candidate.ticker in ['TEST1', 'TEST2']:
                transient_metrics = TransientMetrics(
                    volume_spike=2.0, call_put_ratio=1.5,
                    sentiment_24h=0.2, borrow_fee_pct=3.0
                )
                return ScoredCandidate(
                    candidate=candidate,
                    transient_metrics=transient_metrics,
                    squeeze_score=0.75,
                    alert_level=None
                )
            return None

        with patch.object(self.daily_deep_scan, '_scan_candidate', side_effect=mock_scan_candidate):
            # Process batch
            metrics_dict = {'successful_scans': 0, 'failed_scans': 0}
            scored_candidates = self.daily_deep_scan._process_batch(candidates, metrics_dict)

            # Assertions
            self.assertEqual(len(scored_candidates), 2)  # Only TEST1 and TEST2 should succeed
            self.assertEqual(metrics_dict['successful_scans'], 2)
            self.assertEqual(metrics_dict['failed_scans'], 1)

    @patch('src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan.session_scope')
    def test_store_results(self, mock_session_scope):
        """Test storing results in database."""
        # Mock database session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Create test scored candidates
            candidate = self.create_test_candidate('TEST', 0.7)
            transient_metrics = TransientMetrics(
                volume_spike=2.0, call_put_ratio=1.5,
                sentiment_24h=0.2, borrow_fee_pct=3.0
            )
            scored_candidate = ScoredCandidate(
                candidate=candidate,
                transient_metrics=transient_metrics,
                squeeze_score=0.8,
                alert_level=None
            )

            # Test storing results
            scan_date = date.today()
            self.daily_deep_scan._store_results([scored_candidate], scan_date, {})

            # Verify service was called
            mock_service.save_deep_scan_results.assert_called_once()

    def test_factory_function(self):
        """Test the factory function."""
        deep_scan = create_daily_deep_scan(
            self.mock_fmp_downloader,
            self.mock_finnhub_downloader,
            self.deep_scan_config
        )

        self.assertIsInstance(deep_scan, DailyDeepScan)
        self.assertEqual(deep_scan.fmp_downloader, self.mock_fmp_downloader)
        self.assertEqual(deep_scan.finnhub_downloader, self.mock_finnhub_downloader)
        self.assertEqual(deep_scan.config, self.deep_scan_config)

    @patch('src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan.session_scope')
    def test_run_deep_scan_with_candidates(self, mock_session_scope):
        """Test complete deep scan run with provided candidates."""
        # Mock database operations
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.daily_deep_scan.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Create test candidates
            candidates = [
                self.create_test_candidate('TEST1', 0.7),
                self.create_test_candidate('TEST2', 0.8)
            ]

            # Mock successful processing
            with patch.object(self.daily_deep_scan, '_process_batch') as mock_process:
                mock_scored = [
                    ScoredCandidate(
                        candidate=candidates[0],
                        transient_metrics=TransientMetrics(2.0, 1.5, 0.2, 3.0),
                        squeeze_score=0.75,
                        alert_level=None
                    ),
                    ScoredCandidate(
                        candidate=candidates[1],
                        transient_metrics=TransientMetrics(3.0, 2.0, 0.4, 5.0),
                        squeeze_score=0.85,
                        alert_level=None
                    )
                ]
                mock_process.return_value = mock_scored

                # Run deep scan
                results = self.daily_deep_scan.run_deep_scan(candidates)

                # Assertions
                self.assertIsInstance(results, DeepScanResults)
                self.assertEqual(results.candidates_processed, 2)
                self.assertEqual(len(results.scored_candidates), 2)
                self.assertIn('duration_seconds', results.runtime_metrics)

    def test_run_deep_scan_no_candidates(self):
        """Test deep scan run with no candidates."""
        # Run deep scan with empty candidate list
        results = self.daily_deep_scan.run_deep_scan([])

        # Assertions
        self.assertIsInstance(results, DeepScanResults)
        self.assertEqual(results.candidates_processed, 0)
        self.assertEqual(len(results.scored_candidates), 0)


if __name__ == '__main__':
    unittest.main()