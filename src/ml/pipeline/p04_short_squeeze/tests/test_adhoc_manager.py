"""
Unit tests for Ad-hoc Candidate Manager module.

Tests ad-hoc candidate lifecycle management, TTL expiration functionality,
and integration with deep scan processing.
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.data.adhoc_manager import AdHocManager
from src.ml.pipeline.p04_short_squeeze.core.models import AdHocCandidate
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestAdHocManager(unittest.TestCase):
    """Test cases for Ad-hoc Candidate Manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.adhoc_manager = AdHocManager(default_ttl_days=7)

        # Sample ad-hoc candidate data
        self.sample_candidate = AdHocCandidate(
            ticker='TSLA',
            reason='Unusual volume spike detected',
            first_seen=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7),
            active=True,
            promoted_by_screener=False
        )

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_add_candidate_success(self, mock_session_scope):
        """Test successful addition of ad-hoc candidate."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.add_adhoc_candidate.return_value = True

            # Execute
            result = self.adhoc_manager.add_candidate('TSLA', 'Unusual volume spike', ttl_days=10)

            # Verify
            self.assertTrue(result)
            mock_service.add_adhoc_candidate.assert_called_once_with('TSLA', 'Unusual volume spike', 10)

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_add_candidate_default_ttl(self, mock_session_scope):
        """Test addition of ad-hoc candidate with default TTL."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.add_adhoc_candidate.return_value = True

            # Execute
            result = self.adhoc_manager.add_candidate('GME', 'Social media buzz')

            # Verify
            self.assertTrue(result)
            mock_service.add_adhoc_candidate.assert_called_once_with('GME', 'Social media buzz', 7)

    def test_add_candidate_empty_ticker(self):
        """Test addition with empty ticker."""
        result = self.adhoc_manager.add_candidate('', 'Some reason')
        self.assertFalse(result)

    def test_add_candidate_empty_reason(self):
        """Test addition with empty reason."""
        result = self.adhoc_manager.add_candidate('TSLA', '')
        self.assertFalse(result)

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_remove_candidate_success(self, mock_session_scope):
        """Test successful removal of ad-hoc candidate."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.remove_adhoc_candidate.return_value = True

            # Execute
            result = self.adhoc_manager.remove_candidate('TSLA')

            # Verify
            self.assertTrue(result)
            mock_service.remove_adhoc_candidate.assert_called_once_with('TSLA')

    def test_remove_candidate_empty_ticker(self):
        """Test removal with empty ticker."""
        result = self.adhoc_manager.remove_candidate('')
        self.assertFalse(result)

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_get_active_candidates(self, mock_session_scope):
        """Test retrieval of active ad-hoc candidates."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository and model
        mock_repo = Mock()
        mock_service.repo = mock_repo

        mock_model = Mock()
        mock_model.ticker = 'TSLA'
        mock_model.reason = 'Test reason'
        mock_model.first_seen = datetime.now()
        mock_model.expires_at = datetime.now() + timedelta(days=7)
        mock_model.active = True
        mock_model.promoted_by_screener = False

        mock_repo.adhoc_candidates.get_active_candidates.return_value = [mock_model]

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            result = self.adhoc_manager.get_active_candidates()

            # Verify
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], AdHocCandidate)
            self.assertEqual(result[0].ticker, 'TSLA')
            self.assertEqual(result[0].reason, 'Test reason')
            self.assertTrue(result[0].active)

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_get_candidate_found(self, mock_session_scope):
        """Test retrieval of specific ad-hoc candidate."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository and model
        mock_repo = Mock()
        mock_service.repo = mock_repo

        mock_model = Mock()
        mock_model.ticker = 'TSLA'
        mock_model.reason = 'Test reason'
        mock_model.first_seen = datetime.now()
        mock_model.expires_at = datetime.now() + timedelta(days=7)
        mock_model.active = True
        mock_model.promoted_by_screener = False

        mock_repo.adhoc_candidates.get_candidate.return_value = mock_model

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            result = self.adhoc_manager.get_candidate('TSLA')

            # Verify
            self.assertIsNotNone(result)
            self.assertIsInstance(result, AdHocCandidate)
            self.assertEqual(result.ticker, 'TSLA')
            mock_repo.adhoc_candidates.get_candidate.assert_called_once_with('TSLA')

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_get_candidate_not_found(self, mock_session_scope):
        """Test retrieval of non-existent ad-hoc candidate."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository
        mock_repo = Mock()
        mock_service.repo = mock_repo
        mock_repo.adhoc_candidates.get_candidate.return_value = None

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            result = self.adhoc_manager.get_candidate('NONEXISTENT')

            # Verify
            self.assertIsNone(result)

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_activate_candidate_success(self, mock_session_scope):
        """Test successful activation of ad-hoc candidate."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository and model
        mock_repo = Mock()
        mock_service.repo = mock_repo

        mock_model = Mock()
        mock_model.ticker = 'TSLA'
        mock_model.reason = 'Test reason'
        mock_model.active = False  # Currently inactive

        mock_repo.adhoc_candidates.get_candidate.return_value = mock_model
        mock_service.add_adhoc_candidate.return_value = True

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            result = self.adhoc_manager.activate_candidate('TSLA')

            # Verify
            self.assertTrue(result)
            mock_service.add_adhoc_candidate.assert_called_once_with('TSLA', 'Test reason', 7)

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_activate_candidate_already_active(self, mock_session_scope):
        """Test activation of already active candidate."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository and model
        mock_repo = Mock()
        mock_service.repo = mock_repo

        mock_model = Mock()
        mock_model.ticker = 'TSLA'
        mock_model.active = True  # Already active

        mock_repo.adhoc_candidates.get_candidate.return_value = mock_model

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            result = self.adhoc_manager.activate_candidate('TSLA')

            # Verify
            self.assertTrue(result)
            # Should not call add_adhoc_candidate since already active
            mock_service.add_adhoc_candidate.assert_not_called()

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_expire_candidates(self, mock_session_scope):
        """Test expiration of ad-hoc candidates."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        expected_expired = ['TSLA', 'GME']

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service
            mock_service.expire_adhoc_candidates.return_value = expected_expired

            # Execute
            result = self.adhoc_manager.expire_candidates()

            # Verify
            self.assertEqual(result, expected_expired)
            mock_service.expire_adhoc_candidates.assert_called_once()

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_extend_ttl_success(self, mock_session_scope):
        """Test successful TTL extension."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository and model
        mock_repo = Mock()
        mock_service.repo = mock_repo

        mock_model = Mock()
        mock_model.ticker = 'TSLA'
        mock_model.active = True
        mock_model.expires_at = datetime.now() + timedelta(days=3)

        mock_repo.adhoc_candidates.get_candidate.return_value = mock_model

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            result = self.adhoc_manager.extend_ttl('TSLA', 5)

            # Verify
            self.assertTrue(result)
            # Check that expires_at was updated
            self.assertIsNotNone(mock_model.expires_at)

    def test_extend_ttl_invalid_days(self):
        """Test TTL extension with invalid days."""
        result = self.adhoc_manager.extend_ttl('TSLA', 0)
        self.assertFalse(result)

        result = self.adhoc_manager.extend_ttl('TSLA', -5)
        self.assertFalse(result)

    @patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.session_scope')
    def test_promote_by_screener(self, mock_session_scope):
        """Test marking candidate as promoted by screener."""
        # Mock session and service
        mock_session = Mock()
        mock_service = Mock()
        mock_session_scope.return_value.__enter__.return_value = mock_session

        # Mock repository
        mock_repo = Mock()
        mock_service.repo = mock_repo
        mock_repo.adhoc_candidates.promote_by_screener.return_value = True

        with patch('src.ml.pipeline.p04_short_squeeze.data.adhoc_manager.ShortSqueezeService') as mock_service_class:
            mock_service_class.return_value = mock_service

            # Execute
            result = self.adhoc_manager.promote_by_screener('TSLA')

            # Verify
            self.assertTrue(result)
            mock_repo.adhoc_candidates.promote_by_screener.assert_called_once_with('TSLA')

    def test_get_candidates_for_deep_scan(self):
        """Test retrieval of candidates for deep scan."""
        # Mock get_active_candidates
        with patch.object(self.adhoc_manager, 'get_active_candidates') as mock_get_active:
            mock_candidates = [
                AdHocCandidate(ticker='TSLA', reason='Test', active=True),
                AdHocCandidate(ticker='GME', reason='Test', active=True)
            ]
            mock_get_active.return_value = mock_candidates

            # Execute
            result = self.adhoc_manager.get_candidates_for_deep_scan()

            # Verify
            self.assertEqual(result, ['TSLA', 'GME'])

    def test_bulk_add_candidates_success(self):
        """Test bulk addition of candidates."""
        candidates_data = [
            {'ticker': 'TSLA', 'reason': 'Volume spike', 'ttl_days': 10},
            {'ticker': 'GME', 'reason': 'Social buzz', 'ttl_days': 5}
        ]

        with patch.object(self.adhoc_manager, 'add_candidate') as mock_add:
            mock_add.return_value = True

            # Execute
            added_count, errors = self.adhoc_manager.bulk_add_candidates(candidates_data)

            # Verify
            self.assertEqual(added_count, 2)
            self.assertEqual(len(errors), 0)
            self.assertEqual(mock_add.call_count, 2)

    def test_bulk_add_candidates_with_errors(self):
        """Test bulk addition with some invalid candidates."""
        candidates_data = [
            {'ticker': 'TSLA', 'reason': 'Volume spike'},  # Valid
            {'ticker': '', 'reason': 'Empty ticker'},      # Invalid - empty ticker
            {'ticker': 'GME', 'reason': ''}                # Invalid - empty reason
        ]

        with patch.object(self.adhoc_manager, 'add_candidate') as mock_add:
            mock_add.return_value = True

            # Execute
            added_count, errors = self.adhoc_manager.bulk_add_candidates(candidates_data)

            # Verify
            self.assertEqual(added_count, 1)  # Only 1 valid candidate
            self.assertEqual(len(errors), 2)  # 2 validation errors
            mock_add.assert_called_once_with('TSLA', 'Volume spike', 7)

    def test_get_expiring_candidates(self):
        """Test retrieval of expiring candidates."""
        now = datetime.now()

        # Mock active candidates with different expiration dates
        mock_candidates = [
            AdHocCandidate(
                ticker='TSLA',
                reason='Test',
                expires_at=now + timedelta(days=1),  # Expiring soon
                active=True
            ),
            AdHocCandidate(
                ticker='GME',
                reason='Test',
                expires_at=now + timedelta(days=10),  # Not expiring soon
                active=True
            ),
            AdHocCandidate(
                ticker='AMC',
                reason='Test',
                expires_at=now + timedelta(days=2),  # Expiring soon
                active=True
            )
        ]

        with patch.object(self.adhoc_manager, 'get_active_candidates') as mock_get_active:
            mock_get_active.return_value = mock_candidates

            # Execute
            result = self.adhoc_manager.get_expiring_candidates(days_ahead=3)

            # Verify
            self.assertEqual(len(result), 2)  # TSLA and AMC expire within 3 days
            tickers = [c.ticker for c in result]
            self.assertIn('TSLA', tickers)
            self.assertIn('AMC', tickers)
            self.assertNotIn('GME', tickers)

    def test_get_statistics(self):
        """Test retrieval of ad-hoc candidate statistics."""
        now = datetime.now()

        # Mock active candidates
        mock_candidates = [
            AdHocCandidate(
                ticker='TSLA',
                reason='Test',
                first_seen=now - timedelta(days=5),
                expires_at=now + timedelta(days=1),
                active=True,
                promoted_by_screener=True
            ),
            AdHocCandidate(
                ticker='GME',
                reason='Test',
                first_seen=now - timedelta(days=10),
                expires_at=now + timedelta(days=5),
                active=True,
                promoted_by_screener=False
            )
        ]

        with patch.object(self.adhoc_manager, 'get_active_candidates') as mock_get_active:
            with patch.object(self.adhoc_manager, 'get_expiring_candidates') as mock_get_expiring:
                mock_get_active.return_value = mock_candidates
                mock_get_expiring.return_value = [mock_candidates[0]]  # Only TSLA expiring

                # Execute
                result = self.adhoc_manager.get_statistics()

                # Verify
                self.assertEqual(result['total_active'], 2)
                self.assertEqual(result['promoted_by_screener'], 1)
                self.assertEqual(result['expiring_within_3_days'], 1)
                self.assertEqual(result['average_age_days'], 7.5)  # (5 + 10) / 2
                self.assertEqual(result['default_ttl_days'], 7)

    def test_validate_candidate_data_valid(self):
        """Test validation of valid candidate data."""
        valid_data = {
            'ticker': 'TSLA',
            'reason': 'Volume spike detected',
            'ttl_days': 10
        }

        is_valid, errors = self.adhoc_manager.validate_candidate_data(valid_data)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_candidate_data_missing_ticker(self):
        """Test validation with missing ticker."""
        invalid_data = {
            'reason': 'Volume spike detected'
        }

        is_valid, errors = self.adhoc_manager.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Ticker is required", errors)

    def test_validate_candidate_data_missing_reason(self):
        """Test validation with missing reason."""
        invalid_data = {
            'ticker': 'TSLA'
        }

        is_valid, errors = self.adhoc_manager.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("Reason is required", errors)

    def test_validate_candidate_data_invalid_ttl(self):
        """Test validation with invalid TTL."""
        invalid_data = {
            'ticker': 'TSLA',
            'reason': 'Volume spike',
            'ttl_days': -5  # Invalid negative TTL
        }

        is_valid, errors = self.adhoc_manager.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("TTL days must be a positive integer", errors)

    def test_validate_candidate_data_excessive_ttl(self):
        """Test validation with excessive TTL."""
        invalid_data = {
            'ticker': 'TSLA',
            'reason': 'Volume spike',
            'ttl_days': 500  # Too high
        }

        is_valid, errors = self.adhoc_manager.validate_candidate_data(invalid_data)

        self.assertFalse(is_valid)
        self.assertIn("TTL days cannot exceed 365", errors)


if __name__ == '__main__':
    unittest.main()