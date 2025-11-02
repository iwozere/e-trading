"""
Test data factories for User models.

Provides factory functions to create test data for User, AuthIdentity, and VerificationCode models.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional


class UserFactory:
    """Factory for creating User test data."""

    @staticmethod
    def create_data(
        email: Optional[str] = None,
        role: str = "trader",
        is_active: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Create user data dictionary matching actual User model."""
        return {
            "email": email,
            "role": role,  # Must be one of: admin, trader, viewer
            "is_active": is_active,
            **kwargs
        }

    @staticmethod
    def admin_user(email: str = "admin@example.com") -> Dict[str, Any]:
        """Create an admin user."""
        return UserFactory.create_data(
            email=email,
            role="admin",
            is_active=True
        )

    @staticmethod
    def regular_user(email: str = "user@example.com") -> Dict[str, Any]:
        """Create a regular user."""
        return UserFactory.create_data(
            email=email,
            role="trader",
            is_active=True
        )

    @staticmethod
    def inactive_user(email: str = "inactive@example.com") -> Dict[str, Any]:
        """Create an inactive user."""
        return UserFactory.create_data(
            email=email,
            role="trader",
            is_active=False
        )


class AuthIdentityFactory:
    """Factory for creating AuthIdentity test data."""

    @staticmethod
    def create_data(
        user_id: int,
        provider: str = "telegram",
        provider_user_id: str = "123456789",
        provider_username: Optional[str] = None,
        provider_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create auth identity data dictionary."""
        return {
            "user_id": user_id,
            "provider": provider,
            "provider_user_id": provider_user_id,
            "provider_username": provider_username or f"user_{provider_user_id}",
            "provider_data": provider_data or {},
            **kwargs
        }

    @staticmethod
    def telegram_identity(
        user_id: int,
        telegram_id: int = 123456789,
        username: str = "testuser",
        first_name: str = "Test",
        last_name: str = "User"
    ) -> Dict[str, Any]:
        """Create a Telegram auth identity."""
        return AuthIdentityFactory.create_data(
            user_id=user_id,
            provider="telegram",
            provider_user_id=str(telegram_id),
            provider_username=username,
            provider_data={
                "telegram_id": telegram_id,
                "username": username,
                "first_name": first_name,
                "last_name": last_name,
                "is_bot": False
            }
        )

    @staticmethod
    def google_identity(
        user_id: int,
        google_id: str = "google_123456",
        email: str = "test@example.com"
    ) -> Dict[str, Any]:
        """Create a Google auth identity."""
        return AuthIdentityFactory.create_data(
            user_id=user_id,
            provider="google",
            provider_user_id=google_id,
            provider_username=email,
            provider_data={
                "email": email,
                "email_verified": True
            }
        )


class VerificationCodeFactory:
    """Factory for creating VerificationCode test data."""

    @staticmethod
    def create_data(
        user_id: int,
        code: str = "123456",
        purpose: str = "login",
        expires_at: Optional[datetime] = None,
        used_at: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create verification code data dictionary."""
        return {
            "user_id": user_id,
            "code": code,
            "purpose": purpose,
            "expires_at": expires_at or (datetime.now(timezone.utc) + timedelta(minutes=15)),
            "used_at": used_at,
            **kwargs
        }

    @staticmethod
    def active_code(user_id: int, code: str = "123456", purpose: str = "login") -> Dict[str, Any]:
        """Create an active verification code."""
        return VerificationCodeFactory.create_data(
            user_id=user_id,
            code=code,
            purpose=purpose,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            used_at=None
        )

    @staticmethod
    def expired_code(user_id: int, code: str = "999999", purpose: str = "login") -> Dict[str, Any]:
        """Create an expired verification code."""
        return VerificationCodeFactory.create_data(
            user_id=user_id,
            code=code,
            purpose=purpose,
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
            used_at=None
        )

    @staticmethod
    def used_code(user_id: int, code: str = "000000", purpose: str = "login") -> Dict[str, Any]:
        """Create a used verification code."""
        return VerificationCodeFactory.create_data(
            user_id=user_id,
            code=code,
            purpose=purpose,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            used_at=datetime.now(timezone.utc) - timedelta(minutes=5)
        )


# Convenient aliases
user_factory = UserFactory()
auth_identity_factory = AuthIdentityFactory()
verification_code_factory = VerificationCodeFactory()
