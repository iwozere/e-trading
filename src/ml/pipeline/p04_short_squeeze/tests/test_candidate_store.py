"""
Unit tests for Candidate Store module.

Tests candidate storage and retrieval operations, lifecycle management,
and data validation functionality.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, date
from decimal import Decimal

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.data.candidate_store import CandidateStore
from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, Candidate, CandidateSource
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestCandidateStore(unittest.TestCase):
    """Test cases for Candidate Store."""

    def setUp(self):
        """Set up test fixtures."""
        self.candidate_store = CandidateStore()

        # Sample structural metrics
        self.sample_structural_metrics = StructuralMetrics(
            short_interest_pct=0.25,
            days_to_cover=5.5,
            float_shares=100000000,
            avg_volume_14d=5000000,
            market_cap=1000000000
        )

        # Sample candidate data
        self.sample_candidate_data = {
            'ticker': 'TSLA',
            'short_interest_pct': Decimal('0.25'),
            'days_to_cover': Decimal('5.5'),
            'float_shares': 100000000,
            'avg_volume_14d': 5000000,
            'market_cap': 1000000000,
            'screener_score': Decimal('0.85'),
            'raw_payload': {'test': True},
            'data_quality': Decimal('0.95')
        }

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_store_screener_snapshot_success(self, mock_session_scope):
        """Test successful storage of screener snapshot."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.save_screener_results.return_value = 2

            # Test data
            candidates = [self.sample_candidate_data, self.sample_candidate_data.copy()]
            run_date = date.today()

            # Execute
            result = self.candidate_store.store_screener_snapshot(candidates, run_date)

            # Verify
            self.assertEqual(result, 2)
            mock_service.save_screener_results.assert_called_once_with(candidates, run_date)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_retrieve_screener_snapshot_latest(self, mock_session_scope):
        """Test retrieval of latest screener snapshot."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        expected_results = [
            {
                'ticker': 'TSLA',
                'screener_score': 0.85,
                'short_interest_pct': 0.25,
                'run_date': date.today()
            }
        ]

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.get_top_candidates_by_screener_score.return_value = expected_results

            # Execute
            result = self.candidate_store.retrieve_screener_snapshot(limit=10)

            # Verify
            self.assertEqual(result, expected_results)
            mock_service.get_top_candidates_by_screener_score.assert_called_once_with(10)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_retrieve_screener_snapshot_specific_date(self, mock_session_scope):
        """Test retrieval of screener snapshot for specific date."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository structure
        mock_repo = Mock()
        mock_service.repo = mock_repo

        # Mock snapshot objects
        mock_snapshot = Mock()
        mock_snapshot.ticker = 'TSLA'
        mock_snapshot.screener_score = Decimal('0.85')
        mock_snapshot.short_interest_pct = Decimal('0.25')
        mock_snapshot.days_to_cover = Decimal('5.5')
        mock_snapshot.float_shares = 100000000
        mock_snapshot.avg_volume_14d = 5000000
        mock_snapshot.market_cap = 1000000000
        mock_snapshot.run_date = date.today()
        mock_snapshot.data_quality = Decimal('0.95')

        mock_repo.screener_snapshots.get_top_candidates.return_value = [mock_snapshot]

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            run_date = date.today()
            result = self.candidate_store.retrieve_screener_snapshot(run_date=run_date, limit=10)

            # Verify
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['ticker'], 'TSLA')
            self.assertEqual(result[0]['screener_score'], 0.85)
            mock_repo.screener_snapshots.get_top_candidates.assert_called_once_with(run_date, 10)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_store_deep_scan_results_success(self, mock_session_scope):
        """Test successful storage of deep scan results."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.save_deep_scan_results.return_value = 3

            # Test data
            results = [
                {
                    'ticker': 'TSLA',
                    'volume_spike': 2.5,
                    'sentiment_24h': 0.6,
                    'squeeze_score': 0.75
                }
            ]
            scan_date = date.today()

            # Execute
            result = self.candidate_store.store_deep_scan_results(results, scan_date)

            # Verify
            self.assertEqual(result, 3)
            mock_service.save_deep_scan_results.assert_called_once_with(results, scan_date)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_get_active_candidates(self, mock_session_scope):
        """Test retrieval of active candidates."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        expected_tickers = ['TSLA', 'GME', 'AMC']

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.get_candidates_for_deep_scan.return_value = expected_tickers

            # Execute
            result = self.candidate_store.get_active_candidates()

            # Verify
            self.assertEqual(result, expected_tickers)
            mock_service.get_candidates_for_deep_scan.assert_called_once()

    def test_create_candidate_success(self):
        """Test successful candidate creation."""
        # Execute
        candidate = self.candidate_store.create_candidate(
            ticker='TSLA',
            screener_score=0.85,
            structural_metrics=self.sample_structural_metrics,
            source=CandidateSource.SCREENER
        )

        # Verify
        self.assertIsInstance(candidate, Candidate)
        self.assertEqual(candidate.ticker, 'TSLA')
        self.assertEqual(candidate.screener_score, 0.85)
        self.assertEqual(candidate.source, CandidateSource.SCREENER)
        self.assertIsInstance(candidate.last_updated, datetime)

    def test_create_candidate_invalid_ticker(self):
        """Test candidate creation with invalid ticker."""
        with self.assertRaises(ValueError):
            self.candidate_store.create_candidate(
                ticker='',
                screener_score=0.85,
                structural_metrics=self.sample_structural_metrics
            )

    def test_create_candidate_invalid_score(self):
        """Test candidate creation with invalid score."""
        with self.assertRaises(ValueError):
            self.candidate_store.create_candidate(
                ticker='TSLA',
                screener_score=1.5,  # Invalid score > 1
                structural_metrics=self.sample_structural_metrics
            )

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_get_candidate_history(self, mock_session_scope):
        """Test retrieval of candidate history."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        expected_history = {
            'ticker': 'TSLA',
            'screener_history': [],
            'metrics_history': [],
            'alert_history': []
        }

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.get_ticker_analysis.return_value = expected_history

            # Execute
            result = self.candidate_store.get_candidate_history('TSLA', days=30)

            # Verify
            self.assertEqual(result, expected_history)
            mock_service.get_ticker_analysis.assert_called_once_with('TSLA', 30)

    def test_validate_candidate_data_valid(self):
        """Test validation of valid candidate data."""
        valid_data = {
            'ticker': 'TSLA',
            'screener_score': 0.85,
            'structural_metrics': {
                'short_interest_pct': 0.25,
                'days_to_cover': 5.5,
                'float_shares': 100000000,
                'avg_volume_14d': 5000000,
                'market_cap': 1000000000
            }
        }

        is_valid, errors = self.candidate_store.validate_candidate_data(valid_data)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_candidate_data_missing_ticker(self):
        """Test validation with missing ticker."""
        invalid_data = {
            'screener_score': 0.85
        }

        is_valid, errors = self.candidate_store.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Missing required field: ticker", errors)

    def test_validate_candidate_data_invalid_score(self):
        """Test validation with invalid score."""
        invalid_data = {
            'ticker': 'TSLA',
            'screener_score': 1.5  # Invalid score
        }

        is_valid, errors = self.candidate_store.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Screener score must be a number between 0 and 1", errors)

    def test_validate_candidate_data_empty_ticker(self):
        """Test validation with empty ticker."""
        invalid_data = {
            'ticker': '',
            'screener_score': 0.85
        }

        is_valid, errors = self.candidate_store.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Ticker must be a non-empty string", errors)

    def test_validate_candidate_data_long_ticker(self):
        """Test validation with ticker too long."""
        invalid_data = {
            'ticker': 'VERYLONGTICKER',  # Too long
            'screener_score': 0.85
        }

        is_valid, errors = self.candidate_store.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Ticker must be 10 characters or less", errors)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_batch_store_candidates_with_validation(self, mock_session_scope):
        """Test batch storage of candidates with validation."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.save_screener_results.return_value = 1

            # Test data - mix of valid and invalid
            candidates = [
                {
                    'ticker': 'TSLA',
                    'screener_score': 0.85
                },
                {
                    'ticker': '',  # Invalid - empty ticker
                    'screener_score': 0.75
                },
                {
                    'ticker': 'GME',
                    'screener_score': 1.5  # Invalid - score > 1
                }
            ]

            # Execute
            stored_count, errors = self.candidate_store.batch_store_candidates(
                candidates, date.today(), validate=True
            )

            # Verify
            self.assertEqual(stored_count, 1)  # Only 1 valid candidate
            self.assertEqual(len(errors), 2)  # 2 validation errors
            mock_service.save_screener_results.assert_called_once()

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_cleanup_expired_candidates(self, mock_session_scope):
        """Test cleanup of expired candidates."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        expected_cleanup_stats = {
            'snapshots_deleted': 10,
            'metrics_deleted': 5,
            'alerts_deleted': 3
        }

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.cleanup_old_data.return_value = expected_cleanup_stats

            # Execute
            result = self.candidate_store.cleanup_expired_candidates(days_to_keep=90)

            # Verify
            self.assertEqual(result, expected_cleanup_stats)
            mock_service.cleanup_old_data.assert_called_once_with(90)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_get_candidate_statistics(self, mock_session_scope):
        """Test retrieval of candidate statistics."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        expected_stats = {
            'latest_screener_run': date.today(),
            'active_adhoc_candidates': 5,
            'recent_alerts_7d': 3,
            'todays_deep_scan_count': 15,
            'status': 'healthy'
        }

        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.get_pipeline_statistics.return_value = expected_stats

            # Execute
            result = self.candidate_store.get_candidate_statistics()

            # Verify
            self.assertIn('store_type', result)
            self.assertEqual(result['store_type'], 'CandidateStore')
            self.assertIn('last_updated', result)
            mock_service.get_pipeline_statistics.assert_called_once()


if __name__ == '__main__':
    unittest.main()