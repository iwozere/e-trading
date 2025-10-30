"""
Tests for UsersRepo

Tests the repository layer for user and authentication identity operations.
Covers the renamed tables: usr_users, usr_auth_identities, usr_verification_codes.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Models
from src.data.db.models.model_users import Base as UsersBase, User, AuthIdentity, VerificationCode

# Repository under test
from src.data.db.repos.repo_users import UsersRepo


# Use the shared engine and dbsess fixtures from conftest.py


def test_ensure_user_for_telegram_creates_new_user(dbsess):
    """Test that ensure_user_for_telegram creates a new user and identity."""
    repo = UsersRepo(dbsess)
    telegram_id = "123456789"

    # Should not exist initially
    user = repo.get_user_by_telegram_id(telegram_id)
    assert user is None

    # Create user
    created_user = repo.ensure_user_for_telegram(telegram_id, {"email": "test@example.com"})
    assert created_user is not None
    assert created_user.email == "test@example.com"

    # Should now exist
    found_user = repo.get_user_by_telegram_id(telegram_id)
    assert found_user is not None
    assert found_user.id == created_user.id


def test_ensure_user_for_telegram_returns_existing_user(dbsess):
    """Test that ensure_user_for_telegram returns existing user."""
    repo = UsersRepo(dbsess)
    telegram_id = "987654321"

    # Create user first time
    user1 = repo.ensure_user_for_telegram(telegram_id, {"email": "first@example.com"})

    # Create user second time - should return same user
    user2 = repo.ensure_user_for_telegram(telegram_id, {"email": "second@example.com"})

    assert user1.id == user2.id
    assert user1.email == "first@example.com"  # Email should not change


def test_update_telegram_profile_metadata(dbsess):
    """Test updating telegram profile metadata."""
    repo = UsersRepo(dbsess)
    telegram_id = "555666777"

    # Create user
    user = repo.ensure_user_for_telegram(telegram_id)

    # Update profile with metadata
    repo.update_telegram_profile(
        telegram_id,
        verified=True,
        approved=False,
        language="en",
        is_admin=True,
        max_alerts=10,
        max_schedules=5
    )

    # Get profile and verify
    profile = repo.get_telegram_profile(telegram_id)
    assert profile is not None
    assert profile["verified"] is True
    assert profile["approved"] is False
    assert profile["language"] == "en"
    assert profile["is_admin"] is True
    assert profile["max_alerts"] == 10
    assert profile["max_schedules"] == 5


def test_update_telegram_profile_email(dbsess):
    """Test updating user email through telegram profile."""
    repo = UsersRepo(dbsess)
    telegram_id = "111222333"

    # Create user
    user = repo.ensure_user_for_telegram(telegram_id)
    assert user.email is None

    # Update email
    repo.update_telegram_profile(telegram_id, email="updated@example.com")

    # Verify email was updated
    profile = repo.get_telegram_profile(telegram_id)
    assert profile["email"] == "updated@example.com"

    # Verify user object was updated
    updated_user = repo.get_user_by_telegram_id(telegram_id)
    assert updated_user.email == "updated@example.com"


def test_get_telegram_profile_nonexistent_user(dbsess):
    """Test getting profile for non-existent user returns None."""
    repo = UsersRepo(dbsess)
    profile = repo.get_telegram_profile("nonexistent")
    assert profile is None


def test_list_telegram_users_dto(dbsess):
    """Test listing all telegram users as DTOs."""
    repo = UsersRepo(dbsess)

    # Create multiple users
    repo.ensure_user_for_telegram("user1", {"email": "user1@example.com"})
    repo.ensure_user_for_telegram("user2", {"email": "user2@example.com"})
    repo.update_telegram_profile("user1", verified=True, approved=True)
    repo.update_telegram_profile("user2", verified=True, approved=False)

    # List users
    users = repo.list_telegram_users_dto()
    assert len(users) == 2

    # Check structure
    user_ids = {u["telegram_user_id"] for u in users}
    assert {"user1", "user2"} == user_ids

    # Check specific user data
    user1_data = next(u for u in users if u["telegram_user_id"] == "user1")
    assert user1_data["email"] == "user1@example.com"
    assert user1_data["verified"] is True
    assert user1_data["approved"] is True


def test_list_pending_telegram_approvals(dbsess):
    """Test listing users pending approval."""
    repo = UsersRepo(dbsess)

    # Create users with different approval states
    repo.ensure_user_for_telegram("verified_approved")
    repo.ensure_user_for_telegram("verified_pending")
    repo.ensure_user_for_telegram("unverified")

    repo.update_telegram_profile("verified_approved", verified=True, approved=True)
    repo.update_telegram_profile("verified_pending", verified=True, approved=False)
    repo.update_telegram_profile("unverified", verified=False, approved=False)

    # Get pending approvals
    pending = repo.list_pending_telegram_approvals()

    # Should only include verified but not approved users
    assert len(pending) == 1
    assert pending[0]["telegram_user_id"] == "verified_pending"
    assert pending[0]["verified"] is True
    assert pending[0]["approved"] is False


def test_get_admin_telegram_user_ids(dbsess):
    """Test getting admin telegram user IDs."""
    repo = UsersRepo(dbsess)

    # Create users with different admin states
    repo.ensure_user_for_telegram("admin1")
    repo.ensure_user_for_telegram("admin2")
    repo.ensure_user_for_telegram("regular_user")

    repo.update_telegram_profile("admin1", is_admin=True)
    repo.update_telegram_profile("admin2", is_admin=True)
    repo.update_telegram_profile("regular_user", is_admin=False)

    # Get admin IDs
    admin_ids = repo.get_admin_telegram_user_ids()

    assert len(admin_ids) == 2
    assert set(admin_ids) == {"admin1", "admin2"}


def test_identity_metadata_initialization(dbsess):
    """Test that identity metadata is properly initialized."""
    repo = UsersRepo(dbsess)
    telegram_id = "metadata_test"

    # Create user
    user = repo.ensure_user_for_telegram(telegram_id)

    # Get the identity directly to check metadata initialization
    identity = repo._get_identity(provider="telegram", external_id=telegram_id)
    assert identity is not None
    assert identity.identity_metadata == {}


def test_multiple_providers_same_external_id(dbsess):
    """Test that same external ID can exist for different providers."""
    repo = UsersRepo(dbsess)
    external_id = "12345"

    # Create telegram identity
    telegram_user = repo.ensure_user_for_telegram(external_id)

    # Create another identity with same external_id but different provider
    # (This would be done manually since we only have telegram methods in this repo)
    other_identity = AuthIdentity(
        user_id=telegram_user.id,
        provider="github",
        external_id=external_id,
        identity_metadata={}
    )
    dbsess.add(other_identity)
    dbsess.flush()

    # Both should exist independently
    telegram_identity = repo._get_identity(provider="telegram", external_id=external_id)
    github_identity = repo._get_identity(provider="github", external_id=external_id)

    assert telegram_identity is not None
    assert github_identity is not None
    assert telegram_identity.id != github_identity.id
    assert telegram_identity.user_id == github_identity.user_id


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))