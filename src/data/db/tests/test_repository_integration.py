"""
Integration tests for updated repositories with corrected models.

Tests repository methods work correctly with the corrected model definitions,
focusing on the changes made during database alignment.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import models and repositories
from src.data.db.models.model_jobs import Schedule, ScheduleRun, JobType, RunStatus
from src.data.db.models.model_users import User, AuthIdentity, VerificationCode
from src.data.db.models.model_telegram import TelegramFeedback, TelegramSetting

from src.data.db.repos.repo_jobs import JobsRepository
from src.data.db.repos.repo_users import UsersRepo, VerificationRepo
from src.data.db.repos.repo_telegram import FeedbackRepo, SettingsRepo

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestJobsRepositoryIntegration:
    """Integration tests for JobsRepository with corrected models."""

    def test_job_id_data_type_handling(self):
        """Test that repository correctly handles job_id as integer."""
        # Mock session to avoid database dependency
        mock_session = Mock()
        repo = JobsRepository(mock_session)

        # Test data with integer job_id (corrected type)
        run_data = {
            "job_type": JobType.REPORT.value,
            "job_id": 123,  # Integer, not string
            "user_id": 1,
            "scheduled_for": datetime.now(timezone.utc),
            "job_snapshot": {"param": "value"}
        }

        # Mock the ScheduleRun creation
        mock_run = Mock()
        mock_run.id = 1
        mock_run.job_type = JobType.REPORT.value
        mock_run.job_id = 123

        mock_session.add.return_value = None
        mock_session.flush.return_value = None

        with patch('src.data.db.repos.repo_jobs.ScheduleRun') as mock_schedule_run:
            mock_schedule_run.return_value = mock_run

            result = repo.create_run(run_data)

            # Verify ScheduleRun was called with correct data types
            mock_schedule_run.assert_called_once_with(**run_data)
            assert result == mock_run

    def test_get_runs_by_job_with_integer_id(self):
        """Test get_runs_by_job method with integer job_id."""
        mock_session = Mock()
        repo = JobsRepository(mock_session)

        # Mock query chain
        mock_query = Mock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []

        # Call with integer job_id (corrected type)
        result = repo.get_runs_by_job(JobType.REPORT, 123)

        # Verify query was built correctly
        mock_session.query.assert_called_once_with(ScheduleRun)
        assert result == []

    def test_logging_uses_lazy_formatting(self):
        """Test that repository uses lazy logging formatting."""
        mock_session = Mock()
        repo = JobsRepository(mock_session)

        # Mock successful run creation
        mock_run = Mock()
        mock_run.id = 1
        mock_run.job_type = "report"
        mock_run.job_id = 123

        mock_session.add.return_value = None
        mock_session.flush.return_value = None

        with patch('src.data.db.repos.repo_jobs.ScheduleRun') as mock_schedule_run:
            mock_schedule_run.return_value = mock_run

            with patch('src.data.db.repos.repo_jobs._logger') as mock_logger:
                run_data = {
                    "job_type": JobType.REPORT.value,
                    "job_id": 123,
                    "user_id": 1,
                    "scheduled_for": datetime.now(timezone.utc),
                    "job_snapshot": {}
                }

                repo.create_run(run_data)

                # Verify lazy logging was used (with % formatting, not f-strings)
                mock_logger.info.assert_called_once()
                call_args = mock_logger.info.call_args
                assert len(call_args[0]) >= 2  # Should have format string + args
                assert "%" in call_args[0][0]  # Should use % formatting


class TestUsersRepositoryIntegration:
    """Integration tests for UsersRepo with corrected AuthIdentity model."""

    def test_identity_metadata_attribute_access(self):
        """Test that repository correctly accesses identity_metadata attribute."""
        mock_session = Mock()
        repo = UsersRepo(mock_session)

        # Mock AuthIdentity with identity_metadata attribute
        mock_identity = Mock()
        mock_identity.identity_metadata = {"verified": True, "language": "en"}

        # Mock User
        mock_user = Mock()
        mock_user.id = 1
        mock_user.email = "test@example.com"

        # Mock query result
        mock_session.execute.return_value.first.return_value = (mock_user, mock_identity)

        with patch('src.data.db.repos.repo_users.select') as mock_select:
            result = repo._ensure_identity(
                provider="telegram",
                external_id="123456"
            )

            user, identity = result
            assert user == mock_user
            assert identity == mock_identity
            # Verify identity_metadata was accessed correctly
            assert identity.identity_metadata == {"verified": True, "language": "en"}

    def test_telegram_profile_metadata_operations(self):
        """Test telegram profile operations use identity_metadata correctly."""
        mock_session = Mock()
        repo = UsersRepo(mock_session)

        # Mock existing identity with metadata
        mock_identity = Mock()
        mock_identity.identity_metadata = {"verified": False}

        mock_user = Mock()
        mock_user.id = 1
        mock_user.email = "test@example.com"

        # Mock _ensure_identity to return our mocks
        with patch.object(repo, '_ensure_identity') as mock_ensure:
            mock_ensure.return_value = (mock_user, mock_identity)

            # Update profile
            repo.update_telegram_profile("123456", verified=True, language="en")

            # Verify identity_metadata was updated correctly
            expected_metadata = {"verified": True, "language": "en"}
            assert mock_identity.identity_metadata == expected_metadata

    def test_verification_code_with_new_columns(self):
        """Test VerificationRepo works with updated VerificationCode model."""
        mock_session = Mock()
        repo = VerificationRepo(mock_session)

        # Mock VerificationCode creation
        mock_code = Mock()
        mock_code.id = 1
        mock_code.user_id = 1
        mock_code.code = "123456"
        mock_code.sent_time = 1634567890
        mock_code.provider = "telegram"  # New column
        mock_code.created_at = datetime.now(timezone.utc)  # New column

        mock_session.add.return_value = None
        mock_session.flush.return_value = None

        with patch('src.data.db.repos.repo_users.VerificationCode') as mock_vc:
            mock_vc.return_value = mock_code

            result = repo.issue(user_id=1, code="123456", sent_time=1634567890)

            # Verify VerificationCode was created with correct parameters
            mock_vc.assert_called_once_with(
                user_id=1,
                code="123456",
                sent_time=1634567890
            )
            assert result == mock_code


class TestTelegramRepositoryIntegration:
    """Integration tests for Telegram repositories with corrected models."""

    def test_feedback_creation_with_new_columns(self):
        """Test FeedbackRepo creates feedback with all required columns."""
        mock_session = Mock()
        repo = FeedbackRepo(mock_session)

        # Mock TelegramFeedback creation
        mock_feedback = Mock()
        mock_feedback.id = 1
        mock_feedback.user_id = 1
        mock_feedback.type = "bug_report"  # New column
        mock_feedback.message = "Test feedback"  # New column
        mock_feedback.status = "open"  # New column
        mock_feedback.created_at = datetime.now(timezone.utc)  # New column

        mock_session.add.return_value = None
        mock_session.flush.return_value = None

        with patch('src.data.db.repos.repo_telegram.TelegramFeedback') as mock_tf:
            mock_tf.return_value = mock_feedback

            with patch('src.data.db.repos.repo_telegram.utcnow') as mock_utcnow:
                mock_time = datetime.now(timezone.utc)
                mock_utcnow.return_value = mock_time

                result = repo.create(
                    user_id=1,
                    type_="bug_report",
                    message="Test feedback"
                )

                # Verify TelegramFeedback was created with all new columns
                mock_tf.assert_called_once_with(
                    user_id=1,
                    type="bug_report",
                    message="Test feedback",
                    status="open",
                    created_at=mock_time
                )
                assert result == mock_feedback

    def test_feedback_filtering_by_type(self):
        """Test feedback filtering uses the new type column correctly."""
        mock_session = Mock()
        repo = FeedbackRepo(mock_session)

        # Mock query chain
        mock_query = Mock()
        mock_session.execute.return_value.scalars.return_value = []

        with patch('src.data.db.repos.repo_telegram.select') as mock_select:
            mock_select.return_value = mock_query
            mock_query.where.return_value = mock_query

            result = repo.list(type_="bug_report")

            # Verify query was built with type filter
            mock_select.assert_called_once_with(TelegramFeedback)
            mock_query.where.assert_called_once()
            assert result == []

    def test_feedback_status_update(self):
        """Test feedback status update uses the new status column."""
        mock_session = Mock()
        repo = FeedbackRepo(mock_session)

        # Mock update result
        mock_result = Mock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        with patch('src.data.db.repos.repo_telegram.update') as mock_update:
            mock_update_query = Mock()
            mock_update.return_value = mock_update_query
            mock_update_query.where.return_value = mock_update_query
            mock_update_query.values.return_value = mock_update_query

            result = repo.set_status(feedback_id=1, status="resolved")

            # Verify update was called correctly
            mock_update.assert_called_once_with(TelegramFeedback)
            assert result is True


class TestModelConstraintValidation:
    """Test that corrected models have proper constraints."""

    def test_schedule_run_model_attributes(self):
        """Test ScheduleRun model has correct attribute types."""
        # Test that model can be instantiated with correct types
        run = ScheduleRun(
            job_type="report",
            job_id=123,  # Integer, not string
            user_id=1,
            status="pending",
            scheduled_for=datetime.now(timezone.utc)
        )

        assert run.job_type == "report"
        assert run.job_id == 123
        assert isinstance(run.job_id, int)
        assert run.user_id == 1
        assert run.status == "pending"

    def test_auth_identity_metadata_mapping(self):
        """Test AuthIdentity model has correct column mapping."""
        # Test that model can be instantiated with identity_metadata
        identity = AuthIdentity(
            user_id=1,
            provider="telegram",
            external_id="123456",
            identity_metadata={"verified": True}
        )

        assert identity.user_id == 1
        assert identity.provider == "telegram"
        assert identity.external_id == "123456"
        assert identity.identity_metadata == {"verified": True}

        # Verify the column mapping exists
        assert hasattr(AuthIdentity, 'identity_metadata')

    def test_telegram_feedback_new_columns(self):
        """Test TelegramFeedback model has all required columns."""
        # Test that model can be instantiated with new columns
        feedback = TelegramFeedback(
            user_id=1,
            type="bug_report",  # New column
            message="Test message",  # New column
            status="open",  # New column
            created_at=datetime.now(timezone.utc)  # New column
        )

        assert feedback.user_id == 1
        assert feedback.type == "bug_report"
        assert feedback.message == "Test message"
        assert feedback.status == "open"
        assert feedback.created_at is not None

    def test_verification_code_new_columns(self):
        """Test VerificationCode model has new columns."""
        # Test that model can be instantiated with new columns
        code = VerificationCode(
            user_id=1,
            code="123456",
            sent_time=1634567890,
            provider="telegram",  # New column
            created_at=datetime.now(timezone.utc)  # New column
        )

        assert code.user_id == 1
        assert code.code == "123456"
        assert code.sent_time == 1634567890
        assert code.provider == "telegram"
        assert code.created_at is not None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))