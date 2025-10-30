"""
Unit tests for Telegram models.

Tests the updated telegram models including TelegramFeedback with all columns,
constraint enforcement, and model relationships.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# Import models
from src.data.db.models.model_users import Base as UsersBase, User
from src.data.db.models.model_telegram import (
    Base as TelegramBase,
    TelegramFeedback,
    TelegramSetting,
    TelegramBroadcastLog,
    TelegramCommandAudit
)

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


# ------------------------ Test Database Setup ------------------------
@pytest.fixture()
def engine():
    """Create in-memory SQLite database with foreign keys enabled for testing."""
    # Using SQLite for unit tests as it's faster and doesn't require external dependencies
    # The models use func.now() which works with both SQLite and PostgreSQL
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)

    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()

    # Create all tables
    UsersBase.metadata.create_all(eng)
    TelegramBase.metadata.create_all(eng)
    return eng


@pytest.fixture()
def dbsess(engine):
    """Create database session for testing."""
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    s = Session()
    try:
        yield s
        s.rollback()
    finally:
        s.close()


# ------------------------ Helper Functions ------------------------
def _create_user(dbsess, email: str = "test@example.com") -> int:
    """Create a test user and return user_id."""
    user = User(email=email)
    dbsess.add(user)
    dbsess.flush()
    return user.id


# ------------------------ TelegramFeedback Tests ------------------------
def test_telegram_feedback_creation_with_all_columns(dbsess):
    """Test TelegramFeedback model creation with all columns."""
    user_id = _create_user(dbsess)

    feedback = TelegramFeedback(
        user_id=user_id,
        type="bug_report",
        message="Something is not working correctly",
        status="open"
    )

    dbsess.add(feedback)
    dbsess.flush()

    assert feedback.id is not None
    assert feedback.user_id == user_id
    assert feedback.type == "bug_report"
    assert feedback.message == "Something is not working correctly"
    assert feedback.status == "open"
    assert feedback.created_at is not None  # Should have default value


def test_telegram_feedback_with_minimal_data(dbsess):
    """Test TelegramFeedback creation with only required fields."""
    user_id = _create_user(dbsess)

    feedback = TelegramFeedback(user_id=user_id)
    dbsess.add(feedback)
    dbsess.flush()

    assert feedback.id is not None
    assert feedback.user_id == user_id
    assert feedback.type is None
    assert feedback.message is None
    assert feedback.status is None
    assert feedback.created_at is not None  # Should have default value


def test_telegram_feedback_foreign_key_constraint(dbsess):
    """Test that foreign key constraint is enforced."""
    # Try to create feedback with non-existent user_id
    feedback = TelegramFeedback(
        user_id=999999,  # Non-existent user
        type="test",
        message="test message"
    )

    dbsess.add(feedback)

    with pytest.raises(IntegrityError):
        dbsess.flush()


def test_telegram_feedback_cascade_delete(dbsess):
    """Test that feedback is deleted when user is deleted (CASCADE)."""
    user_id = _create_user(dbsess)

    feedback = TelegramFeedback(
        user_id=user_id,
        type="test",
        message="test message"
    )
    dbsess.add(feedback)
    dbsess.flush()

    feedback_id = feedback.id

    # Delete the user
    user = dbsess.get(User, user_id)
    dbsess.delete(user)
    dbsess.commit()  # Commit to ensure cascade takes effect

    # Feedback should be deleted due to CASCADE
    deleted_feedback = dbsess.get(TelegramFeedback, feedback_id)
    assert deleted_feedback is None


def test_telegram_feedback_type_length_constraint(dbsess):
    """Test that type field respects String(50) constraint."""
    user_id = _create_user(dbsess)

    # Test with exactly 50 characters
    long_type = "a" * 50
    feedback = TelegramFeedback(
        user_id=user_id,
        type=long_type,
        message="test"
    )
    dbsess.add(feedback)
    dbsess.flush()

    assert feedback.type == long_type


def test_telegram_feedback_status_length_constraint(dbsess):
    """Test that status field respects String(20) constraint."""
    user_id = _create_user(dbsess)

    # Test with exactly 20 characters
    long_status = "a" * 20
    feedback = TelegramFeedback(
        user_id=user_id,
        status=long_status,
        message="test"
    )
    dbsess.add(feedback)
    dbsess.flush()

    assert feedback.status == long_status


# ------------------------ TelegramSetting Tests ------------------------
def test_telegram_setting_creation(dbsess):
    """Test TelegramSetting model creation."""
    setting = TelegramSetting(
        key="theme",
        value="dark"
    )

    dbsess.add(setting)
    dbsess.flush()

    assert setting.key == "theme"
    assert setting.value == "dark"


def test_telegram_setting_primary_key_constraint(dbsess):
    """Test that primary key constraint is enforced."""
    setting1 = TelegramSetting(key="theme", value="dark")
    setting2 = TelegramSetting(key="theme", value="light")

    dbsess.add(setting1)
    dbsess.flush()

    dbsess.add(setting2)

    with pytest.raises(IntegrityError):
        dbsess.flush()


def test_telegram_setting_key_length_constraint(dbsess):
    """Test that key field respects String(100) constraint."""
    # Test with exactly 100 characters
    long_key = "a" * 100
    setting = TelegramSetting(
        key=long_key,
        value="test"
    )
    dbsess.add(setting)
    dbsess.flush()

    assert setting.key == long_key


def test_telegram_setting_null_value(dbsess):
    """Test that value can be null."""
    setting = TelegramSetting(key="test_key", value=None)
    dbsess.add(setting)
    dbsess.flush()

    assert setting.key == "test_key"
    assert setting.value is None


# ------------------------ TelegramBroadcastLog Tests ------------------------
def test_telegram_broadcast_log_creation(dbsess):
    """Test TelegramBroadcastLog model creation."""
    log = TelegramBroadcastLog(
        message="Test broadcast message",
        sent_by="admin",
        success_count=5,
        total_count=10
    )

    dbsess.add(log)
    dbsess.flush()

    assert log.id is not None
    assert log.message == "Test broadcast message"
    assert log.sent_by == "admin"
    assert log.success_count == 5
    assert log.total_count == 10
    assert log.created_at is not None  # Should have default value


def test_telegram_broadcast_log_with_minimal_data(dbsess):
    """Test TelegramBroadcastLog creation with only required fields."""
    log = TelegramBroadcastLog(
        message="Test message",
        sent_by="user"
    )

    dbsess.add(log)
    dbsess.flush()

    assert log.id is not None
    assert log.message == "Test message"
    assert log.sent_by == "user"
    assert log.success_count is None
    assert log.total_count is None
    assert log.created_at is not None


def test_telegram_broadcast_log_sent_by_length_constraint(dbsess):
    """Test that sent_by field respects String(255) constraint."""
    # Test with exactly 255 characters
    long_sent_by = "a" * 255
    log = TelegramBroadcastLog(
        message="test",
        sent_by=long_sent_by
    )
    dbsess.add(log)
    dbsess.flush()

    assert log.sent_by == long_sent_by


# ------------------------ TelegramCommandAudit Tests ------------------------
def test_telegram_command_audit_creation(dbsess):
    """Test TelegramCommandAudit model creation."""
    audit = TelegramCommandAudit(
        telegram_user_id="123456789",
        command="/start",
        full_message="/start hello",
        is_registered_user=True,
        user_email="test@example.com",
        success=True,
        response_time_ms=150
    )

    dbsess.add(audit)
    dbsess.flush()

    assert audit.id is not None
    assert audit.telegram_user_id == "123456789"
    assert audit.command == "/start"
    assert audit.full_message == "/start hello"
    assert audit.is_registered_user is True
    assert audit.user_email == "test@example.com"
    assert audit.success is True
    assert audit.response_time_ms == 150
    assert audit.created_at is not None


def test_telegram_command_audit_with_minimal_data(dbsess):
    """Test TelegramCommandAudit creation with only required fields."""
    audit = TelegramCommandAudit(
        telegram_user_id="123456789",
        command="/help"
    )

    dbsess.add(audit)
    dbsess.flush()

    assert audit.id is not None
    assert audit.telegram_user_id == "123456789"
    assert audit.command == "/help"
    assert audit.full_message is None
    assert audit.is_registered_user is None
    assert audit.user_email is None
    assert audit.success is None
    assert audit.error_message is None
    assert audit.response_time_ms is None
    assert audit.created_at is not None


def test_telegram_command_audit_with_error(dbsess):
    """Test TelegramCommandAudit creation with error information."""
    audit = TelegramCommandAudit(
        telegram_user_id="123456789",
        command="/invalid",
        success=False,
        error_message="Command not found",
        response_time_ms=50
    )

    dbsess.add(audit)
    dbsess.flush()

    assert audit.success is False
    assert audit.error_message == "Command not found"
    assert audit.response_time_ms == 50


def test_telegram_command_audit_indexes_exist(dbsess):
    """Test that indexes are properly defined (basic validation)."""
    # This test validates that the model can be created without index errors
    # The actual index creation is tested in DDL generation tests

    audit = TelegramCommandAudit(
        telegram_user_id="test_user",
        command="/test"
    )

    dbsess.add(audit)
    dbsess.flush()

    # Query using indexed fields to ensure they work
    result = dbsess.query(TelegramCommandAudit).filter(
        TelegramCommandAudit.telegram_user_id == "test_user"
    ).first()

    assert result is not None
    assert result.telegram_user_id == "test_user"


# ------------------------ Model Relationship Tests ------------------------
def test_telegram_feedback_user_relationship(dbsess):
    """Test that TelegramFeedback properly relates to User."""
    user_id = _create_user(dbsess, "feedback@example.com")

    feedback = TelegramFeedback(
        user_id=user_id,
        type="suggestion",
        message="Great app!"
    )

    dbsess.add(feedback)
    dbsess.flush()

    # Verify the relationship works
    user = dbsess.get(User, user_id)
    assert user is not None
    assert user.email == "feedback@example.com"

    # Verify feedback references correct user
    assert feedback.user_id == user.id


def test_multiple_feedbacks_per_user(dbsess):
    """Test that a user can have multiple feedbacks."""
    user_id = _create_user(dbsess)

    feedback1 = TelegramFeedback(
        user_id=user_id,
        type="bug",
        message="Bug report 1"
    )

    feedback2 = TelegramFeedback(
        user_id=user_id,
        type="feature",
        message="Feature request"
    )

    dbsess.add_all([feedback1, feedback2])
    dbsess.flush()

    # Query all feedbacks for the user
    feedbacks = dbsess.query(TelegramFeedback).filter(
        TelegramFeedback.user_id == user_id
    ).all()

    assert len(feedbacks) == 2
    messages = {f.message for f in feedbacks}
    assert "Bug report 1" in messages
    assert "Feature request" in messages


# ------------------------ Table Structure Tests ------------------------
def test_telegram_models_table_names(dbsess):
    """Test that all telegram models have correct table names."""
    assert TelegramFeedback.__tablename__ == "telegram_feedbacks"
    assert TelegramSetting.__tablename__ == "telegram_settings"
    assert TelegramBroadcastLog.__tablename__ == "telegram_broadcast_logs"
    assert TelegramCommandAudit.__tablename__ == "telegram_command_audits"


def test_telegram_models_in_metadata(dbsess):
    """Test that all telegram models are registered in metadata."""
    tables = TelegramBase.metadata.tables

    expected_tables = {
        "telegram_feedbacks",
        "telegram_settings",
        "telegram_broadcast_logs",
        "telegram_command_audits"
    }

    actual_tables = set(tables.keys())

    # Check that all expected tables are present
    missing_tables = expected_tables - actual_tables
    assert not missing_tables, f"Missing telegram tables: {missing_tables}"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))