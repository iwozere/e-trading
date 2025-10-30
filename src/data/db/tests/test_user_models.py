"""
User Model Tests

Tests for AuthIdentity and VerificationCode models to validate
the corrected column mappings and constraints.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.data.db.models.model_users import User, AuthIdentity, VerificationCode


@pytest.fixture
def test_user(dbsess):
    """Create a test user for relationship testing."""
    user = User(
        email="test@example.com",
        role="trader",
        is_active=True
    )
    dbsess.add(user)
    dbsess.commit()
    return user


class TestAuthIdentityModel:
    """Test AuthIdentity model with corrected column mapping."""

    def test_auth_identity_creation(self, dbsess, test_user):
        """Test creating AuthIdentity with correct column mapping."""
        auth_identity = AuthIdentity(
            user_id=test_user.id,
            provider="telegram",
            external_id="123456789",
            identity_metadata={"username": "testuser", "first_name": "Test"}
        )

        dbsess.add(auth_identity)
        dbsess.commit()

        # Verify the record was created
        assert auth_identity.id is not None
        assert auth_identity.user_id == test_user.id
        assert auth_identity.provider == "telegram"
        assert auth_identity.external_id == "123456789"
        assert auth_identity.identity_metadata == {"username": "testuser", "first_name": "Test"}
        assert auth_identity.created_at is not None

    def test_auth_identity_metadata_column_mapping(self, dbsess, test_user):
        """Test that identity_metadata attribute maps to metadata database column."""
        auth_identity = AuthIdentity(
            user_id=test_user.id,
            provider="github",
            external_id="987654321",
            identity_metadata={"login": "testuser", "avatar_url": "https://example.com/avatar.jpg"}
        )

        dbsess.add(auth_identity)
        dbsess.commit()

        # Query directly to verify column mapping
        result = dbsess.execute(
            text("SELECT metadata FROM usr_auth_identities WHERE id = :id"),
            {"id": auth_identity.id}
        ).fetchone()

        # The database column should contain the metadata
        assert result is not None
        metadata_dict = result[0]
        assert metadata_dict["login"] == "testuser"
        assert metadata_dict["avatar_url"] == "https://example.com/avatar.jpg"

    def test_auth_identity_convenience_methods(self, dbsess, test_user):
        """Test meta_get and meta_set convenience methods."""
        auth_identity = AuthIdentity(
            user_id=test_user.id,
            provider="telegram",
            external_id="111222333",
            identity_metadata={"username": "testuser"}
        )

        dbsess.add(auth_identity)
        dbsess.commit()

        # Test meta_get
        assert auth_identity.meta_get("username") == "testuser"
        assert auth_identity.meta_get("nonexistent", "default") == "default"

        # Test meta_set
        auth_identity.meta_set("first_name", "Test")
        dbsess.commit()

        # Verify the metadata was updated
        assert auth_identity.identity_metadata["first_name"] == "Test"
        assert auth_identity.identity_metadata["username"] == "testuser"

    def test_auth_identity_unique_constraint(self, dbsess, test_user):
        """Test unique constraint on provider and external_id."""
        # Create first identity
        auth1 = AuthIdentity(
            user_id=test_user.id,
            provider="telegram",
            external_id="123456789"
        )
        dbsess.add(auth1)
        dbsess.commit()

        # Try to create duplicate
        auth2 = AuthIdentity(
            user_id=test_user.id,
            provider="telegram",
            external_id="123456789"  # Same provider and external_id
        )
        dbsess.add(auth2)

        with pytest.raises(IntegrityError):
            dbsess.commit()

    def test_auth_identity_foreign_key_relationship(self, dbsess, test_user):
        """Test foreign key relationship with User."""
        auth_identity = AuthIdentity(
            user_id=test_user.id,
            provider="telegram",
            external_id="123456789"
        )

        dbsess.add(auth_identity)
        dbsess.commit()

        # Verify the relationship
        assert auth_identity.user_id == test_user.id

        # Test cascade delete
        auth_id = auth_identity.id
        dbsess.delete(test_user)
        dbsess.commit()

        # The auth identity should be deleted due to CASCADE
        deleted_auth = dbsess.get(AuthIdentity, auth_id)
        assert deleted_auth is None


class TestVerificationCodeModel:
    """Test VerificationCode model with all columns."""

    def test_verification_code_creation(self, dbsess, test_user):
        """Test creating VerificationCode with all columns."""
        verification_code = VerificationCode(
            user_id=test_user.id,
            code="ABC123",
            sent_time=1234567890,
            provider="telegram"
        )

        dbsess.add(verification_code)
        dbsess.commit()

        # Verify the record was created
        assert verification_code.id is not None
        assert verification_code.user_id == test_user.id
        assert verification_code.code == "ABC123"
        assert verification_code.sent_time == 1234567890
        assert verification_code.provider == "telegram"
        assert verification_code.created_at is not None

    def test_verification_code_default_provider(self, dbsess, test_user):
        """Test default provider value."""
        verification_code = VerificationCode(
            user_id=test_user.id,
            code="XYZ789",
            sent_time=1234567890
            # provider not specified, should default to 'telegram'
        )

        dbsess.add(verification_code)
        dbsess.commit()

        # Verify default provider
        assert verification_code.provider == "telegram"

    def test_verification_code_foreign_key_relationship(self, dbsess, test_user):
        """Test foreign key relationship with User."""
        verification_code = VerificationCode(
            user_id=test_user.id,
            code="DEF456",
            sent_time=1234567890,
            provider="email"
        )

        dbsess.add(verification_code)
        dbsess.commit()

        # Verify the relationship
        assert verification_code.user_id == test_user.id

        # Test cascade delete
        code_id = verification_code.id
        dbsess.delete(test_user)
        dbsess.commit()

        # The verification code should be deleted due to CASCADE
        deleted_code = dbsess.get(VerificationCode, code_id)
        assert deleted_code is None

    def test_verification_code_multiple_providers(self, dbsess, test_user):
        """Test creating verification codes with different providers."""
        # Create telegram verification code
        telegram_code = VerificationCode(
            user_id=test_user.id,
            code="TEL123",
            sent_time=1234567890,
            provider="telegram"
        )

        # Create email verification code
        email_code = VerificationCode(
            user_id=test_user.id,
            code="EML456",
            sent_time=1234567891,
            provider="email"
        )

        dbsess.add_all([telegram_code, email_code])
        dbsess.commit()

        # Verify both were created
        assert telegram_code.id is not None
        assert email_code.id is not None
        assert telegram_code.provider == "telegram"
        assert email_code.provider == "email"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v", "-rA"]))