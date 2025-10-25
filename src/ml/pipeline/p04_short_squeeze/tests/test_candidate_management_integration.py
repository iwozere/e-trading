"""
Integration tests for Candidate Management System.

Tests the integration between CandidateStore and AdHocManager,
including end-to-end workflows and data consistency.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta
from decimal import Decimal

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.data.candidate_store import CandidateStore
from src.ml.pipeline.p04_short_squeeze.data.adhoc_manager import AdHocManager
from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, Candidate, AdHocCandidate, CandidateSource
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestCandidateManagementIntegration(unittest.TestCase):
    """Integration test cases for Candidate Management System."""

    def setUp(self):
        """Set up test fixtures."""
        self.candidate_store = CandidateStore()
        self.adhoc_manager = AdHocManager(default_ttl_days=7)

        # Sample data
        self.sample_structural_metrics = StructuralMetrics(
            short_interest_pct=0.25,
            days_to_cover=5.5,
            float_shares=100000000,
            avg_volume_14d=5000000,
            market_cap=1000000000
        )

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_combined_candidate_retrieval(self, mock_adhoc_session, mock_store_session):
        """Test that active candidates include both screener and ad-hoc candidates."""
        # Mock sessions
        mock_store_session.return_value.__enter__.return_value = Mock()
        mock_adhoc_session.return_value.__enter__.return_value = Mock()

        # Mock CandidateStore returning screener candidates
        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_store_service:
            mock_store_service.return_value.get_candidates_for_deep_scan.return_value = ['TSLA', 'GME', 'AMC']

            # Mock AdHocManager returning ad-hoc candidates
            mock_adhoc_candidates = [
                AdHocCandidate(ticker='NVDA', reason='Manual addition', active=True),
                AdHocCandidate(ticker='AAPL', reason='Unusual activity', active=True)
            ]

            with patch.object(self.adhoc_manager, 'get_active_candidates') as mock_get_adhoc:
                mock_get_adhoc.return_value = mock_adhoc_candidates

                # Execute
                store_candidates = self.candidate_store.get_active_candidates()
                adhoc_tickers = self.adhoc_manager.get_candidates_for_deep_scan()

                # Verify
                self.assertEqual(store_candidates, ['TSLA', 'GME', 'AMC'])
                self.assertEqual(adhoc_tickers, ['NVDA', 'AAPL'])

                # Combined list should include all unique candidates
                all_candidates = list(set(store_candidates + adhoc_tickers))
                self.assertEqual(len(all_candidates), 5)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_candidate_lifecycle_workflow(self, mock_adhoc_session, mock_store_session):
        """Test complete candidate lifecycle from addition to expiration."""
        # Mock sessions
        mock_store_session.return_value.__enter__.return_value = Mock()
        mock_adhoc_session.return_value.__enter__.return_value = Mock()

        # Step 1: Add ad-hoc candidate
        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_adhoc_service:
            mock_adhoc_service.return_value.add_adhoc_candidate.return_value = True

            result = self.adhoc_manager.add_candidate('TSLA', 'Unusual volume spike')
            self.assertTrue(result)

        # Step 2: Verify candidate is active
        mock_candidate = Mock()
        mock_candidate.ticker = 'TSLA'
        mock_candidate.reason = 'Unusual volume spike'
        mock_candidate.active = True
        mock_candidate.first_seen = datetime.now()
        mock_candidate.expires_at = datetime.now() + timedelta(days=7)
        mock_candidate.promoted_by_screener = False

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_adhoc_service:
            mock_repo = Mock()
            mock_adhoc_service.return_value.repo = mock_repo
            mock_repo.adhoc_candidates.get_active_candidates.return_value = [mock_candidate]

            active_candidates = self.adhoc_manager.get_active_candidates()
            self.assertEqual(len(active_candidates), 1)
            self.assertEqual(active_candidates[0].ticker, 'TSLA')

        # Step 3: Candidate gets promoted by screener
        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_adhoc_service:
            mock_repo = Mock()
            mock_adhoc_service.return_value.repo = mock_repo
            mock_repo.adhoc_candidates.promote_by_screener.return_value = True

            result = self.adhoc_manager.promote_by_screener('TSLA')
            self.assertTrue(result)

        # Step 4: Store screener results including the promoted candidate
        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_store_service:
            mock_store_service.return_value.save_screener_results.return_value = 1

            screener_data = [{
                'ticker': 'TSLA',
                'screener_score': Decimal('0.85'),
                'short_interest_pct': Decimal('0.25'),
                'days_to_cover': Decimal('5.5')
            }]

            result = self.candidate_store.store_screener_snapshot(screener_data, date.today())
            self.assertEqual(result, 1)

        # Step 5: Expire old candidates
        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_adhoc_service:
            mock_adhoc_service.return_value.expire_adhoc_candidates.return_value = ['OLD_TICKER']

            expired = self.adhoc_manager.expire_candidates()
            self.assertEqual(expired, ['OLD_TICKER'])

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    def test_data_consistency_validation(self, mock_session_scope):
        """Test data consistency between different storage operations."""
        # Mock session
        mock_session_scope.return_value.__enter__.return_value = Mock()

        # Test data validation in CandidateStore
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

        # Test data validation in AdHocManager
        adhoc_data = {
            'ticker': 'TSLA',
            'reason': 'Volume spike detected',
            'ttl_days': 7
        }

        is_valid, errors = self.adhoc_manager.validate_candidate_data(adhoc_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Test ticker normalization consistency
        candidate = self.candidate_store.create_candidate(
            ticker='tsla',  # lowercase
            screener_score=0.85,
            structural_metrics=self.sample_structural_metrics
        )
        self.assertEqual(candidate.ticker, 'TSLA')  # Should be normalized to uppercase

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_error_handling_integration(self, mock_adhoc_session, mock_store_session):
        """Test error handling across both components."""
        # Mock sessions
        mock_store_session.return_value.__enter__.return_value = Mock()
        mock_adhoc_session.return_value.__enter__.return_value = Mock()

        # Test CandidateStore error handling
        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_store_service:
            mock_store_service.return_value.save_screener_results.side_effect = Exception("Database error")

            # Should handle exception gracefully in batch operation
            candidates = [{'ticker': 'TSLA', 'screener_score': 0.85}]
            stored_count, errors = self.candidate_store.batch_store_candidates(
                candidates, date.today(), validate=False
            )

            self.assertEqual(stored_count, 0)
            self.assertTrue(len(errors) > 0)
            self.assertIn("Failed to store candidates", errors[0])

        # Test AdHocManager error handling
        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_adhoc_service:
            mock_adhoc_service.return_value.add_adhoc_candidate.return_value = False

            # Should handle failed addition gracefully in bulk operation
            candidates_data = [
                {'ticker': 'TSLA', 'reason': 'Test'},
                {'ticker': 'GME', 'reason': 'Test'}
            ]

            with patch.object(self.adhoc_manager, 'add_candidate') as mock_add:
                mock_add.side_effect = [True, False]  # First succeeds, second fails

                added_count, errors = self.adhoc_manager.bulk_add_candidates(candidates_data)

                self.assertEqual(added_count, 1)
                self.assertEqual(len(errors), 1)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_performance_statistics_integration(self, mock_adhoc_session, mock_store_session):
        """Test integration of performance statistics from both components."""
        # Mock sessions
        mock_store_session.return_value.__enter__.return_value = Mock()
        mock_adhoc_session.return_value.__enter__.return_value = Mock()

        # Mock CandidateStore statistics
        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_store_service:
            mock_store_service.return_value.get_pipeline_statistics.return_value = {
                'latest_screener_run': date.today(),
                'active_adhoc_candidates': 5,
                'recent_alerts_7d': 3,
                'todays_deep_scan_count': 15,
                'status': 'healthy'
            }

            store_stats = self.candidate_store.get_candidate_statistics()

            self.assertIn('store_type', store_stats)
            self.assertEqual(store_stats['store_type'], 'CandidateStore')
            self.assertEqual(store_stats['active_adhoc_candidates'], 5)

        # Mock AdHocManager statistics
        mock_candidates = [
            AdHocCandidate(
                ticker='TSLA',
                reason='Test',
                first_seen=datetime.now() - timedelta(days=5),
                active=True,
                promoted_by_screener=True
            ),
            AdHocCandidate(
                ticker='GME',
                reason='Test',
                first_seen=datetime.now() - timedelta(days=10),
                active=True,
                promoted_by_screener=False
            )
        ]

        with patch.object(self.adhoc_manager, 'get_active_candidates') as mock_get_active:
            with patch.object(self.adhoc_manager, 'get_expiring_candidates') as mock_get_expiring:
                mock_get_active.return_value = mock_candidates
                mock_get_expiring.return_value = []

                adhoc_stats = self.adhoc_manager.get_statistics()

                self.assertEqual(adhoc_stats['total_active'], 2)
                self.assertEqual(adhoc_stats['promoted_by_screener'], 1)

        # Verify statistics are consistent
        self.assertEqual(store_stats['active_adhoc_candidates'], 5)  # From store perspective
        self.assertEqual(adhoc_stats['total_active'], 2)  # From adhoc manager perspective
        # Note: These might differ in real scenarios due to timing and different data sources

    def test_candidate_object_creation_consistency(self):
        """Test that candidate objects are created consistently across components."""
        # Test CandidateStore candidate creation
        store_candidate = self.candidate_store.create_candidate(
            ticker='TSLA',
            screener_score=0.85,
            structural_metrics=self.sample_structural_metrics,
            source=CandidateSource.SCREENER
        )

        self.assertIsInstance(store_candidate, Candidate)
        self.assertEqual(store_candidate.ticker, 'TSLA')
        self.assertEqual(store_candidate.screener_score, 0.85)
        self.assertEqual(store_candidate.source, CandidateSource.SCREENER)

        # Test AdHocCandidate creation (through mock data)
        adhoc_candidate = AdHocCandidate(
            ticker='TSLA',
            reason='Manual addition',
            active=True
        )

        self.assertIsInstance(adhoc_candidate, AdHocCandidate)
        self.assertEqual(adhoc_candidate.ticker, 'TSLA')
        self.assertTrue(adhoc_candidate.active)

        # Both should normalize ticker to uppercase
        self.assertEqual(store_candidate.ticker, adhoc_candidate.ticker)

    @patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.session_scope')
    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_ttl_expiration_workflow(self, mock_adhoc_session, mock_store_session):
        """Test TTL expiration workflow integration."""
        # Mock sessions
        mock_store_session.return_value.__enter__.return_value = Mock()
        mock_adhoc_session.return_value.__enter__.return_value = Mock()

        # Step 1: Add candidates with different TTLs
        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_adhoc_service:
            mock_adhoc_service.return_value.add_adhoc_candidate.return_value = True

            # Add short TTL candidate
            result1 = self.adhoc_manager.add_candidate('SHORT_TTL', 'Test', ttl_days=1)
            self.assertTrue(result1)

            # Add long TTL candidate
            result2 = self.adhoc_manager.add_candidate('LONG_TTL', 'Test', ttl_days=30)
            self.assertTrue(result2)

        # Step 2: Mock expiration process
        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_adhoc_service:
            # Simulate that only short TTL candidate expired
            mock_adhoc_service.return_value.expire_adhoc_candidates.return_value = ['SHORT_TTL']

            expired_tickers = self.adhoc_manager.expire_candidates()

            self.assertEqual(len(expired_tickers), 1)
            self.assertIn('SHORT_TTL', expired_tickers)
            self.assertNotIn('LONG_TTL', expired_tickers)

        # Step 3: Verify cleanup affects candidate store statistics
        with patch('src.ml.pipeline.p04_short_squeeze.data.candidate_store.ShortSqueezeService') as mock_store_service:
            mock_store_service.return_value.cleanup_old_data.return_value = {
                'snapshots_deleted': 5,
                'metrics_deleted': 3,
                'alerts_deleted': 2
            }

            cleanup_stats = self.candidate_store.cleanup_expired_candidates(days_to_keep=90)

            self.assertEqual(cleanup_stats['snapshots_deleted'], 5)
            self.assertEqual(cleanup_stats['metrics_deleted'], 3)
            self.assertEqual(cleanup_stats['alerts_deleted'], 2)


if __name__ == '__main__':
    unittest.main()