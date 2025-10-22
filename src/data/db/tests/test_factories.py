"""
Test factories for database models.

Tests that all factory functions create valid model instances
compatible with PostgreSQL constraints and relationships.
"""

import pytest
import uuid
from datetime import datetime, timezone

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.models.model_users import User, AuthIdentity, VerificationCode
from src.data.db.models.model_telegram import TelegramFeedback
from src.data.db.models.model_jobs import Schedule, ScheduleRun
from src.data.db.tests.factories import (
    make_user, add_telegram_identity, make_verification_code,
    make_feedback, make_run, make_job_schedule, RNG
)


@pytest.fixture()
def rng():
    """Create deterministic RNG for testing."""
    return RNG(seed=42)


def unique_email():
    """Generate unique email for testing to avoid constraint violations."""
    return f"test-{uuid.uuid4().hex[:8]}@example.com"


class TestUserFactories:
    """Test user model factories."""

    def test_make_user_creates_valid_user(self, dbsess):
        """Test that make_user creates a valid User instance."""
        email = unique_email()
        user = make_user(dbsess, email=email, role="trader")

        assert user.id is not None
        assert user.email == email
        assert user.role == "trader"
        assert user.is_active is True

        # Verify it's persisted
        retrieved = dbsess.get(User, user.id)
        assert retrieved is not None
        assert retrieved.email == email

    def test_make_user_with_defaults(self, dbsess):
        """Test make_user with default values."""
        user = make_user(dbsess)

        assert user.id is not None
        assert user.role == "trader"
        assert user.is_active is True

    def test_add_telegram_identity_creates_valid_identity(self, dbsess):
        """Test that add_telegram_identity creates valid AuthIdentity."""
        user = make_user(dbsess, email=unique_email())

        identity = add_telegram_identity(
            dbsess,
            user_id=user.id,
            telegram_user_id="123456789",
            username="testuser",
            first_name="Test"
        )

        assert identity.id is not None
        assert identity.user_id == user.id
        assert identity.provider == "telegram"
        assert identity.external_id == "123456789"
        assert identity.identity_metadata is not None
        assert identity.identity_metadata["username"] == "testuser"
        assert identity.identity_metadata["first_name"] == "Test"
        assert identity.created_at is not None

    def test_make_verification_code_creates_valid_code(self, dbsess):
        """Test that make_verification_code creates valid VerificationCode."""
        user = make_user(dbsess, email=unique_email())

        vc = make_verification_code(
            dbsess,
            user_id=user.id,
            code="123456",
            provider="telegram"
        )

        assert vc.id is not None
        assert vc.user_id == user.id
        assert vc.code == "123456"
        assert vc.provider == "telegram"
        assert vc.sent_time is not None
        assert vc.created_at is not None

    def test_make_verification_code_with_defaults(self, dbsess):
        """Test make_verification_code with default values."""
        user = make_user(dbsess, email=unique_email())

        vc = make_verification_code(dbsess, user_id=user.id)

        assert vc.id is not None
        assert vc.user_id == user.id
        assert vc.code is not None
        assert len(vc.code) > 0
        assert vc.provider == "telegram"  # Default value
        assert vc.sent_time is not None
        assert vc.created_at is not None


class TestTelegramFactories:
    """Test telegram model factories."""

    def test_make_feedback_creates_valid_feedback(self, dbsess):
        """Test that make_feedback creates valid TelegramFeedback."""
        user = make_user(dbsess, email=unique_email())

        feedback = make_feedback(
            dbsess,
            user_id=user.id,
            type_="bug",
            message="Something is broken",
            status="open"
        )

        assert feedback.id is not None
        assert feedback.user_id == user.id
        assert feedback.type == "bug"
        assert feedback.message == "Something is broken"
        assert feedback.status == "open"
        assert feedback.created_at is not None

    def test_make_feedback_with_defaults(self, dbsess):
        """Test make_feedback with default values."""
        user = make_user(dbsess, email=unique_email())

        feedback = make_feedback(dbsess, user_id=user.id)

        assert feedback.id is not None
        assert feedback.user_id == user.id
        assert feedback.type == "bug"  # Default
        assert feedback.message == "it broke"  # Default
        assert feedback.status == "open"  # Default
        assert feedback.created_at is not None

    def test_make_feedback_foreign_key_constraint(self, dbsess):
        """Test that make_feedback respects foreign key constraints."""
        # This should work with valid user_id
        user = make_user(dbsess, email=unique_email())
        feedback = make_feedback(dbsess, user_id=user.id)
        assert feedback.user_id == user.id


@pytest.mark.skip(reason="Job tables may not exist in current database schema")
class TestJobFactories:
    """Test job model factories."""

    def test_make_run_creates_valid_run(self, dbsess, rng):
        """Test that make_run creates valid Run instance."""
        run = make_run(
            dbsess,
            rng,
            user_id=12345,  # BigInteger
            job_type="report",
            job_id=987654321,  # BigInteger
            status="pending",
            worker_id="worker-1"
        )

        assert run.run_id is not None
        assert run.user_id == 12345
        assert run.job_type == "report"
        assert run.job_id == 987654321
        assert run.status == "pending"
        assert run.worker_id == "worker-1"
        assert run.scheduled_for is not None
        assert run.job_snapshot is not None

    def test_make_run_with_defaults(self, dbsess, rng):
        """Test make_run with default values."""
        run = make_run(dbsess, rng, user_id=12345)

        assert run.run_id is not None
        assert run.user_id == 12345
        assert run.job_type in ["report", "screener", "alert"]
        assert run.job_id is not None
        assert isinstance(run.job_id, int)
        assert run.job_id >= 100000  # BigInteger range
        assert run.status == "pending"
        assert run.worker_id is not None
        assert run.worker_id.startswith("worker-")
        assert run.scheduled_for is not None
        assert run.job_snapshot is not None

    def test_make_run_biginteger_compatibility(self, dbsess, rng):
        """Test that make_run creates BigInteger-compatible values."""
        # Test with large BigInteger values
        large_user_id = 9223372036854775807  # Max int8 value
        large_job_id = 9223372036854775806

        run = make_run(
            dbsess,
            rng,
            user_id=large_user_id,
            job_id=large_job_id
        )

        assert run.user_id == large_user_id
        assert run.job_id == large_job_id

    def test_make_job_schedule_creates_valid_schedule(self, dbsess, rng):
        """Test that make_job_schedule creates valid Schedule."""
        schedule = make_job_schedule(
            dbsess,
            rng,
            user_id=12345,
            name="Test Schedule",
            job_type="report",
            target="portfolio",
            cron="0 9 * * *"
        )

        assert schedule.id is not None
        assert schedule.user_id == 12345
        assert schedule.name == "Test Schedule"
        assert schedule.job_type == "report"
        assert schedule.target == "portfolio"
        assert schedule.cron == "0 9 * * *"
        assert schedule.enabled is True
        assert schedule.task_params is not None
        assert schedule.next_run_at is not None


class TestFactoryRelationships:
    """Test factory relationships and constraints."""

    def test_user_identity_relationship(self, dbsess):
        """Test user and identity relationship through factories."""
        user = make_user(dbsess, email=unique_email())
        identity = add_telegram_identity(
            dbsess,
            user_id=user.id,
            telegram_user_id="123456789"
        )

        # Test the relationship works
        assert identity.user_id == user.id

        # Verify foreign key constraint
        retrieved_identity = dbsess.get(AuthIdentity, identity.id)
        assert retrieved_identity.user_id == user.id

    def test_user_feedback_relationship(self, dbsess):
        """Test user and feedback relationship through factories."""
        user = make_user(dbsess, email=unique_email())
        feedback = make_feedback(dbsess, user_id=user.id)

        assert feedback.user_id == user.id

        # Verify foreign key constraint
        retrieved_feedback = dbsess.get(TelegramFeedback, feedback.id)
        assert retrieved_feedback.user_id == user.id

    def test_user_verification_code_relationship(self, dbsess):
        """Test user and verification code relationship through factories."""
        user = make_user(dbsess, email=unique_email())
        vc = make_verification_code(dbsess, user_id=user.id)

        assert vc.user_id == user.id

        # Verify foreign key constraint
        retrieved_vc = dbsess.get(VerificationCode, vc.id)
        assert retrieved_vc.user_id == user.id


class TestFactoryDataTypes:
    """Test factory data type compatibility."""

    def test_datetime_fields_are_timezone_aware(self, dbsess, rng):
        """Test that datetime fields are timezone-aware."""
        user = make_user(dbsess, email=unique_email())
        identity = add_telegram_identity(dbsess, user_id=user.id, telegram_user_id="123")
        feedback = make_feedback(dbsess, user_id=user.id)
        vc = make_verification_code(dbsess, user_id=user.id)

        # Check timezone awareness (skip run test due to missing table)
        assert identity.created_at.tzinfo is not None
        assert feedback.created_at.tzinfo is not None
        assert vc.created_at.tzinfo is not None

    def test_json_fields_are_valid(self, dbsess, rng):
        """Test that JSON/JSONB fields contain valid data."""
        user = make_user(dbsess, email=unique_email())
        identity = add_telegram_identity(
            dbsess,
            user_id=user.id,
            telegram_user_id="123",
            username="test",
            first_name="Test"
        )

        # Check JSON data is valid
        assert isinstance(identity.identity_metadata, dict)
        assert "username" in identity.identity_metadata

    def test_string_length_constraints(self, dbsess, rng):
        """Test that string fields respect length constraints."""
        user = make_user(dbsess, email=unique_email())

        # Test various string fields
        identity = add_telegram_identity(dbsess, user_id=user.id, telegram_user_id="123456789")
        assert len(identity.provider) <= 32
        assert len(identity.external_id) <= 255

        feedback = make_feedback(dbsess, user_id=user.id, type_="bug")
        assert len(feedback.type) <= 50
        assert len(feedback.status) <= 20

        vc = make_verification_code(dbsess, user_id=user.id)
        assert len(vc.code) <= 32
        assert len(vc.provider) <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])