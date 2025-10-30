"""
Tests for Users Service

Tests the service layer for user operations.
Uses monkeypatch to mock database_service for isolated testing.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Services under test
from src.data.db.services import database_service as ds
from src.data.db.services import users_service as us

# Models for table creation
from src.data.db.models.model_users import Base as UsersBase


@pytest.fixture()
def setup_test_db(monkeypatch):
    """
    Setup PostgreSQL test database and monkeypatch database_service.
    """
    import os
    test_db_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://postgres:password@localhost:5432/etrading_test"
    )

    eng = create_engine(test_db_url, future=True)
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False, future=True, expire_on_commit=False)

    monkeypatch.setattr(ds, "engine", eng, raising=False)
    monkeypatch.setattr(ds, "SessionLocal", SessionLocal, raising=False)
    ds._db_service_singleton = None  # reset singleton

    # Create the necessary tables
    UsersBase.metadata.create_all(eng)

    yield eng, SessionLocal


def test_ensure_user_for_telegram_creates_user(setup_test_db):
    """Test that ensure_user_for_telegram creates a new user."""
    eng, SessionLocal = setup_test_db
    telegram_id = "123456789"

    # Should return user_id after creation
    user_id = us.ensure_user_for_telegram(telegram_id, {"email": "test@example.com"})
    assert isinstance(user_id, int)
    assert user_id > 0

    # Should be able to get the user
    user = us.get_user_by_telegram_id(telegram_id)
    assert user is not None
    assert user.id == user_id
    assert user.email == "test@example.com"


def test_get_user_by_telegram_id_nonexistent(setup_test_db):
    """Test getting non-existent user returns None."""
    eng, SessionLocal = setup_test_db

    user = us.get_user_by_telegram_id("nonexistent")
    assert user is None


def test_get_telegram_profile_flow(setup_test_db):
    """Test getting and updating telegram profile."""
    eng, SessionLocal = setup_test_db
    telegram_id = "profile_test"

    # Initially no profile
    profile = us.get_telegram_profile(telegram_id)
    assert profile is None

    # Create user
    user_id = us.ensure_user_for_telegram(telegram_id, {"email": "profile@example.com"})

    # Now should have basic profile
    profile = us.get_telegram_profile(telegram_id)
    assert profile is not None
    assert profile["user_id"] == user_id
    assert profile["telegram_user_id"] == telegram_id
    assert profile["email"] == "profile@example.com"
    assert profile["verified"] is None  # Not set yet
    assert profile["approved"] is None  # Not set yet


def test_update_telegram_profile(setup_test_db):
    """Test updating telegram profile metadata."""
    eng, SessionLocal = setup_test_db
    telegram_id = "update_test"

    # Create user first
    us.ensure_user_for_telegram(telegram_id)

    # Update profile
    us.update_telegram_profile(
        telegram_id,
        verified=True,
        approved=False,
        language="en",
        is_admin=True,
        max_alerts=15,
        max_schedules=8,
        email="updated@example.com"
    )

    # Verify updates
    profile = us.get_telegram_profile(telegram_id)
    assert profile["verified"] is True
    assert profile["approved"] is False
    assert profile["language"] == "en"
    assert profile["is_admin"] is True
    assert profile["max_alerts"] == 15
    assert profile["max_schedules"] == 8
    assert profile["email"] == "updated@example.com"


def test_list_telegram_users_dto(setup_test_db):
    """Test listing telegram users as DTOs."""
    eng, SessionLocal = setup_test_db

    # Create multiple users
    us.ensure_user_for_telegram("user1", {"email": "user1@example.com"})
    us.ensure_user_for_telegram("user2", {"email": "user2@example.com"})
    us.update_telegram_profile("user1", verified=True, approved=True)
    us.update_telegram_profile("user2", verified=False, approved=False)

    # List users
    users = us.list_telegram_users_dto()
    assert len(users) == 2

    # Check structure
    user_ids = {u["telegram_user_id"] for u in users}
    assert {"user1", "user2"} == user_ids

    # Verify data
    user1_data = next(u for u in users if u["telegram_user_id"] == "user1")
    assert user1_data["email"] == "user1@example.com"
    assert user1_data["verified"] is True
    assert user1_data["approved"] is True


def test_list_pending_telegram_approvals(setup_test_db):
    """Test listing users pending approval."""
    eng, SessionLocal = setup_test_db

    # Create users with different states
    us.ensure_user_for_telegram("approved_user")
    us.ensure_user_for_telegram("pending_user")
    us.ensure_user_for_telegram("unverified_user")

    us.update_telegram_profile("approved_user", verified=True, approved=True)
    us.update_telegram_profile("pending_user", verified=True, approved=False)
    us.update_telegram_profile("unverified_user", verified=False, approved=False)

    # Get pending approvals
    pending = us.list_pending_telegram_approvals()

    # Should only include verified but not approved
    assert len(pending) == 1
    assert pending[0]["telegram_user_id"] == "pending_user"


def test_get_admin_telegram_user_ids(setup_test_db):
    """Test getting admin user IDs."""
    eng, SessionLocal = setup_test_db

    # Create users
    us.ensure_user_for_telegram("admin1")
    us.ensure_user_for_telegram("admin2")
    us.ensure_user_for_telegram("regular")

    us.update_telegram_profile("admin1", is_admin=True)
    us.update_telegram_profile("admin2", is_admin=True)
    us.update_telegram_profile("regular", is_admin=False)

    # Get admin IDs
    admin_ids = us.get_admin_telegram_user_ids()

    assert len(admin_ids) == 2
    assert set(admin_ids) == {"admin1", "admin2"}


def test_list_users_for_broadcast(setup_test_db):
    """Test getting users formatted for broadcast."""
    eng, SessionLocal = setup_test_db

    # Create users with different states
    us.ensure_user_for_telegram("broadcast1", {"email": "b1@example.com"})
    us.ensure_user_for_telegram("broadcast2", {"email": "b2@example.com"})

    us.update_telegram_profile("broadcast1", verified=True, approved=True)
    us.update_telegram_profile("broadcast2", verified=True, approved=False)

    # Get broadcast list
    broadcast_users = us.list_users_for_broadcast()

    assert len(broadcast_users) == 2

    # Check structure - should be compact DTO
    for user in broadcast_users:
        assert "telegram_user_id" in user
        assert "email" in user
        assert "approved" in user
        assert "verified" in user
        # Should be string format for telegram_user_id
        assert isinstance(user["telegram_user_id"], str)

    # Check specific data
    user1_data = next(u for u in broadcast_users if u["telegram_user_id"] == "broadcast1")
    assert user1_data["email"] == "b1@example.com"
    assert user1_data["approved"] is True
    assert user1_data["verified"] is True


def test_service_functions_handle_string_and_int_ids(setup_test_db):
    """Test that service functions handle both string and int telegram IDs."""
    eng, SessionLocal = setup_test_db

    # Test with string ID
    user_id_str = us.ensure_user_for_telegram("123456", {"email": "string@example.com"})

    # Test with int ID (should be converted to string internally)
    user_id_int = us.ensure_user_for_telegram(123456, {"email": "int@example.com"})

    # Should be the same user (same external_id after string conversion)
    assert user_id_str == user_id_int

    # Verify we can get the user with both formats
    user_by_str = us.get_user_by_telegram_id("123456")
    user_by_int = us.get_user_by_telegram_id(123456)

    assert user_by_str is not None
    assert user_by_int is not None
    assert user_by_str.id == user_by_int.id


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))