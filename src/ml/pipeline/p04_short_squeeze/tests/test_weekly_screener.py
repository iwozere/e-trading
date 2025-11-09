"""
Unit tests for Weekly Screener module.

Tests the weekly screening functionality including scoring algorithms, filtering, and data storage.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.weekly_screener import WeeklyScreener, ScreenerResults, create_weekly_screener
from src.ml.pipeline.p04_short_squeeze.config.data_classes import ScreenerConfig, ScreenerFilters, ScreenerWeights
from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics, Candidate, CandidateSource
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestWeeklyScreener(unittest.TestCase):
    """Test cases for Weekly Screener."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_fmp_downloader = Mock()
        self.screener_config = ScreenerConfig(
            filters=ScreenerFilters(
                si_percent_min=0.15,
                days_to_cover_min=5.0,
                float_max=100_000_000,
                top_k_candidates=10
            ),
            scoring=ScreenerWeights(
                short_interest_pct=0.4,
                days_to_cover=0.3,
                float_ratio=0.2,
                volume_consistency=0.1
            )
        )

        self.weekly_screener = WeeklyScreener(self.mock_fmp_downloader, self.screener_config)

    def test_extract_structural_metrics_success(self):
        """Test successful extraction of structural metrics."""
        # Mock API data
        short_data = {
            'shortInterest': 10_000_000  # 10M shares short
        }

        float_data = {
            'sharesOutstanding': 50_000_000,  # 50M shares outstanding
            'floatShares': 40_000_000,        # 40M float shares
            'marketCap': 1_000_000_000        # $1B market cap
        }

        volume_data = {
            'avg_volume': 500_000,  # 500k average volume
            'days_of_data': 14
        }

        # Extract metrics
        metrics = self.weekly_screener._extract_structural_metrics(
            'TEST', short_data, float_data, volume_data
        )

        # Assertions
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics.short_interest_pct, 0.2, places=3)  # 10M/50M = 20%
        self.assertAlmostEqual(metrics.days_to_cover, 20.0, places=1)      # 10M/500k = 20 days
        self.assertEqual(metrics.float_shares, 40_000_000)
        self.assertEqual(metrics.avg_volume_14d, 500_000)
        self.assertEqual(metrics.market_cap, 1_000_000_000)

    def test_extract_structural_metrics_invalid_data(self):
        """Test extraction with invalid data."""
        # Test with zero shares outstanding
        short_data = {'shortInterest': 10_000_000}
        float_data = {'sharesOutstanding': 0, 'floatShares': 0, 'marketCap': 1_000_000_000}
        volume_data = {'avg_volume': 500_000}

        metrics = self.weekly_screener._extract_structural_metrics(
            'TEST', short_data, float_data, volume_data
        )
        self.assertIsNone(metrics)

        # Test with zero market cap
        float_data = {'sharesOutstanding': 50_000_000, 'floatShares': 40_000_000, 'marketCap': 0}
        metrics = self.weekly_screener._extract_structural_metrics(
            'TEST', short_data, float_data, volume_data
        )
        self.assertIsNone(metrics)

    def test_meets_minimum_criteria(self):
        """Test minimum criteria checking."""
        # Create test metrics that meet criteria
        good_metrics = StructuralMetrics(
            short_interest_pct=0.20,      # 20% > 15% minimum
            days_to_cover=10.0,           # 10 days > 5 days minimum
            float_shares=50_000_000,      # 50M < 100M maximum
            avg_volume_14d=500_000,
            market_cap=1_000_000_000
        )
        self.assertTrue(self.weekly_screener._meets_minimum_criteria(good_metrics))

        # Test metrics that don't meet SI criteria
        low_si_metrics = StructuralMetrics(
            short_interest_pct=0.10,      # 10% < 15% minimum
            days_to_cover=10.0,
            float_shares=50_000_000,
            avg_volume_14d=500_000,
            market_cap=1_000_000_000
        )
        self.assertFalse(self.weekly_screener._meets_minimum_criteria(low_si_metrics))

        # Test metrics that don't meet DTC criteria
        low_dtc_metrics = StructuralMetrics(
            short_interest_pct=0.20,
            days_to_cover=3.0,            # 3 days < 5 days minimum
            float_shares=50_000_000,
            avg_volume_14d=500_000,
            market_cap=1_000_000_000
        )
        self.assertFalse(self.weekly_screener._meets_minimum_criteria(low_dtc_metrics))

        # Test metrics with too large float
        large_float_metrics = StructuralMetrics(
            short_interest_pct=0.20,
            days_to_cover=10.0,
            float_shares=150_000_000,     # 150M > 100M maximum
            avg_volume_14d=500_000,
            market_cap=1_000_000_000
        )
        self.assertFalse(self.weekly_screener._meets_minimum_criteria(large_float_metrics))

    def test_calculate_screener_score(self):
        """Test screener score calculation."""
        # Create test metrics
        metrics = StructuralMetrics(
            short_interest_pct=0.25,      # 25% short interest
            days_to_cover=10.0,           # 10 days to cover
            float_shares=50_000_000,      # 50M float
            avg_volume_14d=1_000_000,     # 1M average volume
            market_cap=1_000_000_000      # $1B market cap
        )

        score = self.weekly_screener.calculate_screener_score(metrics)

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Score should be reasonable for good metrics
        self.assertGreater(score, 0.3)  # Should be above minimum threshold

    def test_normalize_metrics(self):
        """Test metrics normalization."""
        metrics = StructuralMetrics(
            short_interest_pct=0.25,      # 25%
            days_to_cover=10.0,           # 10 days
            float_shares=50_000_000,      # 50M shares
            avg_volume_14d=1_000_000,     # 1M volume
            market_cap=1_000_000_000
        )

        normalized = self.weekly_screener._normalize_metrics(metrics)

        # All normalized values should be between 0 and 1
        for key, value in normalized.items():
            self.assertGreaterEqual(value, 0.0, f"{key} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{key} should be <= 1")

        # Check specific normalizations
        self.assertAlmostEqual(normalized['short_interest_pct'], 0.5, places=2)  # 25% / 50% cap = 0.5
        self.assertAlmostEqual(normalized['days_to_cover'], 0.5, places=2)       # 10 / 20 cap = 0.5

    def test_filter_candidates(self):
        """Test candidate filtering."""
        # Create test candidates with different scores
        candidates = []
        for i, score in enumerate([0.8, 0.5, 0.2, 0.7, 0.1]):
            metrics = StructuralMetrics(
                short_interest_pct=0.2, days_to_cover=10.0, float_shares=50_000_000,
                avg_volume_14d=500_000, market_cap=1_000_000_000
            )
            candidate = Candidate(
                ticker=f'TEST{i}', screener_score=score, structural_metrics=metrics,
                last_updated=datetime.now(), source=CandidateSource.SCREENER
            )
            candidates.append(candidate)

        # Filter candidates (should remove those with score < 0.3)
        filtered = self.weekly_screener._filter_candidates(candidates)

        # Should keep candidates with scores >= 0.3
        self.assertEqual(len(filtered), 3)  # 0.8, 0.5, 0.7
        scores = [c.screener_score for c in filtered]
        self.assertIn(0.8, scores)
        self.assertIn(0.5, scores)
        self.assertIn(0.7, scores)
        self.assertNotIn(0.2, scores)
        self.assertNotIn(0.1, scores)

    def test_select_top_candidates(self):
        """Test top candidate selection."""
        # Create test candidates
        candidates = []
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for i, score in enumerate(scores):
            metrics = StructuralMetrics(
                short_interest_pct=0.2, days_to_cover=10.0, float_shares=50_000_000,
                avg_volume_14d=500_000, market_cap=1_000_000_000
            )
            candidate = Candidate(
                ticker=f'TEST{i}', screener_score=score, structural_metrics=metrics,
                last_updated=datetime.now(), source=CandidateSource.SCREENER
            )
            candidates.append(candidate)

        # Select top 5 candidates
        self.screener_config.filters.top_k_candidates = 5
        top_candidates = self.weekly_screener._select_top_candidates(candidates)

        # Should return top 5 by score
        self.assertEqual(len(top_candidates), 5)

        # Should be sorted by score (descending)
        scores = [c.screener_score for c in top_candidates]
        self.assertEqual(scores, [0.9, 0.8, 0.7, 0.6, 0.5])

    @patch('src.ml.pipeline.p04_short_squeeze.core.weekly_screener.session_scope')
    def test_store_results(self, mock_session_scope):
        """Test storing results in database."""
        # Mock database session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.core.weekly_screener.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Create test candidates
            metrics = StructuralMetrics(
                short_interest_pct=0.2, days_to_cover=10.0, float_shares=50_000_000,
                avg_volume_14d=500_000, market_cap=1_000_000_000
            )
            candidate = Candidate(
                ticker='TEST', screener_score=0.8, structural_metrics=metrics,
                last_updated=datetime.now(), source=CandidateSource.SCREENER
            )

            # Test storing results
            self.weekly_screener._store_results([candidate], {}, {})

            # Verify service was called
            mock_service.save_screener_results.assert_called_once()

    def test_get_average_volume(self):
        """Test average volume calculation."""
        import pandas as pd

        # Mock OHLCV data
        mock_df = pd.DataFrame({
            'volume': [100_000, 200_000, 150_000, 300_000, 250_000] * 4  # 20 days
        })
        self.mock_fmp_downloader.get_ohlcv.return_value = mock_df

        # Get average volume
        volume_data = self.weekly_screener._get_average_volume('TEST')

        # Assertions
        self.assertIsNotNone(volume_data)
        self.assertIn('avg_volume', volume_data)
        self.assertEqual(volume_data['days_of_data'], 14)  # Should use last 14 days

        # Check average calculation
        expected_avg = mock_df['volume'].tail(14).mean()
        self.assertAlmostEqual(volume_data['avg_volume'], expected_avg, places=0)

    def test_get_average_volume_insufficient_data(self):
        """Test average volume with insufficient data."""
        import pandas as pd

        # Mock insufficient data (less than 7 days)
        mock_df = pd.DataFrame({
            'volume': [100_000, 200_000, 150_000]  # Only 3 days
        })
        self.mock_fmp_downloader.get_ohlcv.return_value = mock_df

        # Get average volume
        volume_data = self.weekly_screener._get_average_volume('TEST')

        # Should return None for insufficient data
        self.assertIsNone(volume_data)

    def test_screen_ticker_success(self):
        """Test successful ticker screening."""
        # Mock API responses
        self.mock_fmp_downloader.get_short_interest_data.return_value = {
            'shortInterest': 10_000_000
        }
        self.mock_fmp_downloader.get_float_shares_data.return_value = {
            'sharesOutstanding': 50_000_000,
            'floatShares': 40_000_000,
            'marketCap': 1_000_000_000
        }

        # Mock volume data
        import pandas as pd
        mock_df = pd.DataFrame({
            'volume': [500_000] * 20
        })
        self.mock_fmp_downloader.get_ohlcv.return_value = mock_df

        # Screen ticker
        metrics = {'api_calls_made': 0, 'successful_fetches': 0, 'failed_fetches': 0,
                  'valid_short_interest': 0, 'valid_float_data': 0}
        candidate = self.weekly_screener._screen_ticker('TEST', metrics)

        # Assertions
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.ticker, 'TEST')
        self.assertGreater(candidate.screener_score, 0)
        self.assertEqual(metrics['successful_fetches'], 1)
        self.assertEqual(metrics['api_calls_made'], 3)

    def test_screen_ticker_no_data(self):
        """Test ticker screening with no data."""
        # Mock no data responses
        self.mock_fmp_downloader.get_short_interest_data.return_value = None

        # Screen ticker
        metrics = {'api_calls_made': 0, 'successful_fetches': 0, 'failed_fetches': 0,
                  'valid_short_interest': 0, 'valid_float_data': 0}
        candidate = self.weekly_screener._screen_ticker('TEST', metrics)

        # Should return None
        self.assertIsNone(candidate)

    def test_factory_function(self):
        """Test the factory function."""
        screener = create_weekly_screener(self.mock_fmp_downloader, self.screener_config)

        self.assertIsInstance(screener, WeeklyScreener)
        self.assertEqual(screener.fmp_downloader, self.mock_fmp_downloader)
        self.assertEqual(screener.config, self.screener_config)

    def test_run_screener_integration(self):
        """Test the complete screener run integration."""
        # Mock successful API responses for multiple tickers
        def mock_short_interest(ticker):
            return {'shortInterest': 5_000_000} if ticker in ['GOOD1', 'GOOD2'] else None

        def mock_float_data(ticker):
            return {
                'sharesOutstanding': 25_000_000,
                'floatShares': 20_000_000,
                'marketCap': 500_000_000
            } if ticker in ['GOOD1', 'GOOD2'] else None

        def mock_ohlcv(ticker, interval, start_date, end_date):
            import pandas as pd
            return pd.DataFrame({'volume': [400_000] * 20}) if ticker in ['GOOD1', 'GOOD2'] else None

        self.mock_fmp_downloader.get_short_interest_data.side_effect = mock_short_interest
        self.mock_fmp_downloader.get_float_shares_data.side_effect = mock_float_data
        self.mock_fmp_downloader.get_ohlcv.side_effect = mock_ohlcv

        # Mock database storage
        with patch('src.ml.pipeline.p04_short_squeeze.core.weekly_screener.session_scope'):
            with patch('src.ml.pipeline.p04_short_squeeze.core.weekly_screener.ShortSqueezeService'):
                # Run screener
                universe = ['GOOD1', 'GOOD2', 'BAD1', 'BAD2']
                results = self.weekly_screener.run_screener(universe)

                # Assertions
                self.assertIsInstance(results, ScreenerResults)
                self.assertEqual(results.total_universe, 4)
                self.assertEqual(results.candidates_found, 2)  # Only GOOD1 and GOOD2 should pass
                self.assertEqual(len(results.top_candidates), 2)


if __name__ == '__main__':
    unittest.main()