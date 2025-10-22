#!/usr/bin/env python3
"""
Unit Tests for Authentication and Authorization
---------------------------------------------

Tests for JWT token management, password hashing, role-based access control,
and authentication utilities.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, status
from pathlib import Path
import sys
import jwt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.api.auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    authenticate_user,
    get_current_user,
    require_admin,
    require_trader_or_admin,
    log_user_action,
    AuthenticationError,
    AuthorizationError,
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS
)
from src.data.db.models.model_users import User


class TestTokenManagement:
    """Test cases for JWT token creation and verification."""

    def test_create_access_token_default_expiry(self):
        """Test creating access token with default expiry."""
        data = {"sub": "123", "username": "testuser"}
        token = create_access_token(data)

        # Verify token structure
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify payload
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "123"
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"
        assert "exp" in payload

        # Verify expiry is approximately correct (within 1 minute)
        exp_time = datetime.fromtimestamp(payload["exp"], timezone.utc)
        expected_exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        time_diff = abs((exp_time - expected_exp).total_seconds())
        assert time_diff < 60  # Within 1 minute

    def test_create_access_token_custom_expiry(self):
        """Test creating access token with custom expiry."""
        data = {"sub": "123", "username": "testuser"}
        custom_expiry = timedelta(hours=2)
        token = create_access_token(data, custom_expiry)

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp_time = datetime.fromtimestamp(payload["exp"], timezone.utc)
        expected_exp = datetime.now(timezone.utc) + custom_expiry
        time_diff = abs((exp_time - expected_exp).total_seconds())
        assert time_diff < 60  # Within 1 minute

    def test_create_refresh_token(self):
        """Test creating refresh token."""
        data = {"sub": "123", "username": "testuser"}
        token = create_refresh_token(data)

        # Verify token structure
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode and verify payload
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == "123"
        assert payload["username"] == "testuser"
        assert payload["type"] == "refresh"
        assert "exp" in payload

        # Verify expiry is approximately correct
        exp_time = datetime.fromtimestamp(payload["exp"], timezone.utc)
        expected_exp = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        time_diff = abs((exp_time - expected_exp).total_seconds())
        assert time_diff < 3600  # Within 1 hour

    def test_verify_token_valid(self):
        """Test verifying valid token."""
        data = {"sub": "123", "username": "testuser"}
        token = create_access_token(data)

        payload = verify_token(token)
        assert payload["sub"] == "123"
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"

    def test_verify_token_expired(self):
        """Test verifying expired token."""
        data = {"sub": "123", "username": "testuser"}
        # Create token that expires immediately
        expired_token = create_access_token(data, timedelta(seconds=-1))

        with pytest.raises(AuthenticationError, match="Token has expired"):
            verify_token(expired_token)

    def test_verify_token_invalid(self):
        """Test verifying invalid token."""
        invalid_token = "invalid.token.here"

        with pytest.raises(AuthenticationError, match="Invalid token"):
            verify_token(invalid_token)

    def test_verify_token_wrong_secret(self):
        """Test verifying token with wrong secret."""
        # Create token with different secret
        wrong_token = jwt.encode(
            {"sub": "123", "username": "testuser", "type": "access"},
            "wrong-secret",
            algorithm=ALGORITHM
        )

        with pytest.raises(AuthenticationError, match="Invalid token"):
            verify_token(wrong_token)


class TestUserAuthentication:
    """Test cases for user authentication."""

    @patch('src.api.auth.webui_app_service')
    def test_authenticate_user_by_email_success(self, mock_service):
        """Test successful authentication by email."""
        # Mock database session and user
        mock_db = Mock()
        mock_user = Mock(spec=User)
        mock_user.is_active = True
        mock_user.verify_password.return_value = True
        mock_user.last_login = None

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        result = authenticate_user(mock_db, "test@example.com", "password")

        assert result == mock_user
        mock_user.verify_password.assert_called_once_with("password")
        mock_db.commit.assert_called_once()
        assert mock_user.last_login is not None

    @patch('src.api.auth.webui_app_service')
    @patch('src.api.auth.get_user_by_telegram_id')
    @patch('src.api.auth.get_user_by_email')
    def test_authenticate_user_by_telegram_id_success(self, mock_get_by_email, mock_get_by_telegram, mock_service):
        """Test successful authentication by telegram ID."""
        mock_db = Mock()
        mock_user = Mock(spec=User)
        mock_user.is_active = True
        mock_user.verify_password.return_value = True
        mock_user.last_login = None

        # First query (by email) returns None, second query (by telegram_user_id) returns user
        mock_get_by_email.return_value = None
        mock_get_by_telegram.return_value = mock_user

        result = authenticate_user(mock_db, "123456789", "password")

        assert result == mock_user
        mock_get_by_email.assert_called_once_with(mock_db, "123456789")
        mock_get_by_telegram.assert_called_once_with("123456789")
        mock_user.verify_password.assert_called_once_with("password")

    @patch('src.api.auth.webui_app_service')
    def test_authenticate_user_not_found(self, mock_service):
        """Test authentication when user not found."""
        mock_db = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = authenticate_user(mock_db, "nonexistent@example.com", "password")

        assert result is None

    @patch('src.api.auth.webui_app_service')
    def test_authenticate_user_inactive(self, mock_service):
        """Test authentication with inactive user."""
        mock_db = Mock()
        mock_user = Mock(spec=User)
        mock_user.is_active = False

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        result = authenticate_user(mock_db, "test@example.com", "password")

        assert result is None

    @patch('src.api.auth.webui_app_service')
    def test_authenticate_user_wrong_password(self, mock_service):
        """Test authentication with wrong password."""
        mock_db = Mock()
        mock_user = Mock(spec=User)
        mock_user.is_active = True
        mock_user.verify_password.return_value = False

        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        result = authenticate_user(mock_db, "test@example.com", "wrongpassword")

        assert result is None
        mock_user.verify_password.assert_called_once_with("wrongpassword")


class TestGetCurrentUser:
    """Test cases for get_current_user dependency."""

    @patch('src.api.auth.get_database_service')
    def test_get_current_user_success(self, mock_get_db_service):
        """Test successful user retrieval from token."""
        # Mock token
        token_data = {"sub": "123", "username": "testuser", "type": "access"}
        token = create_access_token(token_data)

        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.credentials = token

        # Mock database service and user
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()
        mock_user = Mock(spec=User)
        mock_user.id = 123
        mock_user.email = "testuser@trading-system.local"
        mock_user.role = "trader"
        mock_user.is_active = True
        mock_user.created_at = datetime.now(timezone.utc)
        mock_user.updated_at = datetime.now(timezone.utc)
        mock_user.last_login = datetime.now(timezone.utc)

        mock_get_db_service.return_value = mock_db_service
        mock_db_service.uow.return_value.__enter__ = Mock(return_value=mock_uow)
        mock_db_service.uow.return_value.__exit__ = Mock(return_value=None)
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        result = get_current_user(mock_credentials)

        assert result.id == 123
        assert result.email == "testuser@trading-system.local"
        assert result.role == "trader"
        assert result.is_active is True

    def test_get_current_user_invalid_token(self):
        """Test get_current_user with invalid token."""
        mock_credentials = Mock()
        mock_credentials.credentials = "invalid.token.here"

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(mock_credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token" in str(exc_info.value.detail)

    def test_get_current_user_wrong_token_type(self):
        """Test get_current_user with wrong token type."""
        # Create refresh token instead of access token
        token_data = {"sub": "123", "username": "testuser"}
        token = create_refresh_token(token_data)

        mock_credentials = Mock()
        mock_credentials.credentials = token

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(mock_credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token type" in str(exc_info.value.detail)

    @patch('src.api.auth.get_database_service')
    def test_get_current_user_user_not_found(self, mock_get_db_service):
        """Test get_current_user when user not found in database."""
        token_data = {"sub": "999", "username": "nonexistent", "type": "access"}
        token = create_access_token(token_data)

        mock_credentials = Mock()
        mock_credentials.credentials = token

        # Mock database service returning no user
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()

        mock_get_db_service.return_value = mock_db_service
        mock_db_service.uow.return_value.__enter__ = Mock(return_value=mock_uow)
        mock_db_service.uow.return_value.__exit__ = Mock(return_value=None)
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(mock_credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "User not found" in str(exc_info.value.detail)

    @patch('src.api.auth.get_database_service')
    def test_get_current_user_inactive_user(self, mock_get_db_service):
        """Test get_current_user with inactive user."""
        token_data = {"sub": "123", "username": "testuser", "type": "access"}
        token = create_access_token(token_data)

        mock_credentials = Mock()
        mock_credentials.credentials = token

        # Mock database service with inactive user
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()
        mock_user = Mock(spec=User)
        mock_user.is_active = False

        mock_get_db_service.return_value = mock_db_service
        mock_db_service.uow.return_value.__enter__ = Mock(return_value=mock_uow)
        mock_db_service.uow.return_value.__exit__ = Mock(return_value=None)
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        with pytest.raises(HTTPException) as exc_info:
            get_current_user(mock_credentials)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "User account is inactive" in str(exc_info.value.detail)


class TestRoleBasedAccess:
    """Test cases for role-based access control."""

    def test_require_admin_success(self, mock_admin_user):
        """Test require_admin with admin user."""
        result = require_admin(mock_admin_user)
        assert result == mock_admin_user

    def test_require_admin_failure(self, mock_trader_user):
        """Test require_admin with non-admin user."""
        with pytest.raises(HTTPException) as exc_info:
            require_admin(mock_trader_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Access denied" in str(exc_info.value.detail)
        assert "admin" in str(exc_info.value.detail)

    def test_require_trader_or_admin_with_admin(self, mock_admin_user):
        """Test require_trader_or_admin with admin user."""
        result = require_trader_or_admin(mock_admin_user)
        assert result == mock_admin_user

    def test_require_trader_or_admin_with_trader(self, mock_trader_user):
        """Test require_trader_or_admin with trader user."""
        result = require_trader_or_admin(mock_trader_user)
        assert result == mock_trader_user

    def test_require_trader_or_admin_failure(self, mock_viewer_user):
        """Test require_trader_or_admin with viewer user."""
        with pytest.raises(HTTPException) as exc_info:
            require_trader_or_admin(mock_viewer_user)

        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Access denied" in str(exc_info.value.detail)
        assert "trader" in str(exc_info.value.detail)
        assert "admin" in str(exc_info.value.detail)


class TestUserActionLogging:
    """Test cases for user action logging."""

    @patch('src.api.auth.webui_app_service')
    def test_log_user_action_success(self, mock_service, mock_admin_user):
        """Test successful user action logging."""
        mock_service.log_user_action.return_value = 1

        log_user_action(
            user=mock_admin_user,
            action="login",
            resource_type="authentication",
            resource_id="session_123",
            details={"method": "jwt"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

        mock_service.log_user_action.assert_called_once_with(
            user_id=mock_admin_user.id,
            action="login",
            resource_type="authentication",
            resource_id="session_123",
            details={"method": "jwt"},
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

    @patch('src.api.auth.webui_app_service')
    @patch('src.api.auth._logger')
    def test_log_user_action_failure(self, mock_logger, mock_service, mock_admin_user):
        """Test user action logging failure handling."""
        mock_service.log_user_action.side_effect = Exception("Database error")

        # Should not raise exception, just log error
        log_user_action(
            user=mock_admin_user,
            action="login"
        )

        mock_logger.error.assert_called_once()
        assert "Failed to log user action" in str(mock_logger.error.call_args)

    @patch('src.api.auth.webui_app_service')
    def test_log_user_action_minimal_params(self, mock_service, mock_admin_user):
        """Test user action logging with minimal parameters."""
        mock_service.log_user_action.return_value = 1

        log_user_action(user=mock_admin_user, action="test_action")

        mock_service.log_user_action.assert_called_once_with(
            user_id=mock_admin_user.id,
            action="test_action",
            resource_type=None,
            resource_id=None,
            details=None,
            ip_address=None,
            user_agent=None
        )


class TestAuthenticationErrors:
    """Test cases for custom authentication errors."""

    def test_authentication_error(self):
        """Test AuthenticationError exception."""
        error = AuthenticationError("Test authentication error")
        assert str(error) == "Test authentication error"
        assert isinstance(error, Exception)

    def test_authorization_error(self):
        """Test AuthorizationError exception."""
        error = AuthorizationError("Test authorization error")
        assert str(error) == "Test authorization error"
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__])