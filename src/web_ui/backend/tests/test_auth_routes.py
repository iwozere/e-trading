#!/usr/bin/env python3
"""
Unit Tests for Authentication Routes
----------------------------------

Tests for authentication endpoints including:
- User login and logout
- Token refresh
- Password validation
- Session management
"""

import pytest
from unittest.mock import patch, Mock
from fastapi import status
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.web_ui.backend.auth import create_access_token, create_refresh_token


class TestAuthenticationRoutes:
    """Test cases for authentication route endpoints."""

    @patch('src.web_ui.backend.auth_routes.webui_app_service')
    def test_login_success(self, mock_service, client):
        """Test successful user login."""
        # Mock successful authentication
        mock_user_data = {
            "id": 1,
            "email": "admin@test.com",
            "role": "admin",
            "is_active": True
        }
        mock_service.authenticate_user.return_value = mock_user_data

        login_data = {
            "username": "admin",
            "password": "admin"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["email"] == "admin@test.com"
        assert data["user"]["role"] == "admin"

        mock_service.authenticate_user.assert_called_once_with("admin", "admin")

    @patch('src.web_ui.backend.auth_routes.webui_app_service')
    def test_login_invalid_credentials(self, mock_service, client):
        """Test login with invalid credentials."""
        # Mock failed authentication
        mock_service.authenticate_user.return_value = None

        login_data = {
            "username": "admin",
            "password": "wrongpassword"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid credentials" in data["detail"]

    def test_login_missing_username(self, client):
        """Test login with missing username."""
        login_data = {
            "password": "admin"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 422  # Validation error

    def test_login_missing_password(self, client):
        """Test login with missing password."""
        login_data = {
            "username": "admin"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 422  # Validation error

    def test_login_empty_credentials(self, client):
        """Test login with empty credentials."""
        login_data = {
            "username": "",
            "password": ""
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 400
        data = response.json()
        assert "Username and password are required" in data["detail"]

    @patch('src.web_ui.backend.auth_routes.webui_app_service')
    def test_login_service_error(self, mock_service, client):
        """Test login when authentication service fails."""
        # Mock service error
        mock_service.authenticate_user.side_effect = Exception("Database connection failed")

        login_data = {
            "username": "admin",
            "password": "admin"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 500
        data = response.json()
        assert "Authentication service error" in data["detail"]

    @patch('src.web_ui.backend.auth_routes.verify_token')
    @patch('src.web_ui.backend.auth_routes.get_database_service')
    def test_refresh_token_success(self, mock_get_db_service, mock_verify_token, client):
        """Test successful token refresh."""
        # Mock token verification
        mock_verify_token.return_value = {
            "sub": "1",
            "username": "admin",
            "type": "refresh"
        }

        # Mock database service and user
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()
        mock_user = Mock()

        mock_user.id = 1
        mock_user.email = "admin@test.com"
        mock_user.role = "admin"
        mock_user.is_active = True

        mock_get_db_service.return_value = mock_db_service
        mock_db_service.uow.return_value.__enter__.return_value = mock_uow
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        # Create a valid refresh token
        refresh_token = create_refresh_token({"sub": "1", "username": "admin"})

        refresh_data = {
            "refresh_token": refresh_token
        }

        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_refresh_token_missing(self, client):
        """Test token refresh with missing refresh token."""
        refresh_data = {}

        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 422  # Validation error

    @patch('src.web_ui.backend.auth_routes.verify_token')
    def test_refresh_token_invalid(self, mock_verify_token, client):
        """Test token refresh with invalid refresh token."""
        from src.web_ui.backend.auth import AuthenticationError
        mock_verify_token.side_effect = AuthenticationError("Invalid token")

        refresh_data = {
            "refresh_token": "invalid.token.here"
        }

        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid refresh token" in data["detail"]

    @patch('src.web_ui.backend.auth_routes.verify_token')
    def test_refresh_token_wrong_type(self, mock_verify_token, client):
        """Test token refresh with access token instead of refresh token."""
        # Mock token verification returning access token type
        mock_verify_token.return_value = {
            "sub": "1",
            "username": "admin",
            "type": "access"  # Wrong type
        }

        # Create an access token (wrong type)
        access_token = create_access_token({"sub": "1", "username": "admin"})

        refresh_data = {
            "refresh_token": access_token
        }

        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "Invalid token type" in data["detail"]

    @patch('src.web_ui.backend.auth_routes.verify_token')
    @patch('src.web_ui.backend.auth_routes.get_database_service')
    def test_refresh_token_user_not_found(self, mock_get_db_service, mock_verify_token, client):
        """Test token refresh when user no longer exists."""
        # Mock token verification
        mock_verify_token.return_value = {
            "sub": "999",
            "username": "nonexistent",
            "type": "refresh"
        }

        # Mock database service returning no user
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()

        mock_get_db_service.return_value = mock_db_service
        mock_db_service.uow.return_value.__enter__.return_value = mock_uow
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        refresh_token = create_refresh_token({"sub": "999", "username": "nonexistent"})

        refresh_data = {
            "refresh_token": refresh_token
        }

        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "User not found" in data["detail"]

    @patch('src.web_ui.backend.auth_routes.verify_token')
    @patch('src.web_ui.backend.auth_routes.get_database_service')
    def test_refresh_token_inactive_user(self, mock_get_db_service, mock_verify_token, client):
        """Test token refresh with inactive user."""
        # Mock token verification
        mock_verify_token.return_value = {
            "sub": "1",
            "username": "admin",
            "type": "refresh"
        }

        # Mock database service with inactive user
        mock_db_service = Mock()
        mock_uow = Mock()
        mock_session = Mock()
        mock_user = Mock()

        mock_user.is_active = False  # Inactive user

        mock_get_db_service.return_value = mock_db_service
        mock_db_service.uow.return_value.__enter__.return_value = mock_uow
        mock_uow.s = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = mock_user

        refresh_token = create_refresh_token({"sub": "1", "username": "admin"})

        refresh_data = {
            "refresh_token": refresh_token
        }

        response = client.post("/auth/refresh", json=refresh_data)

        assert response.status_code == 401
        data = response.json()
        assert "User account is inactive" in data["detail"]

    def test_logout_success(self, authenticated_client_admin):
        """Test successful logout."""
        response = authenticated_client_admin.post("/auth/logout")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out"

    def test_logout_unauthenticated(self, client):
        """Test logout without authentication."""
        response = client.post("/auth/logout")

        assert response.status_code == 403  # No authorization header

    @patch('src.web_ui.backend.auth_routes.log_user_action')
    @patch('src.web_ui.backend.auth_routes.webui_app_service')
    def test_login_with_audit_logging(self, mock_service, mock_log_action, client):
        """Test login with audit logging."""
        # Mock successful authentication
        mock_user_data = {
            "id": 1,
            "email": "admin@test.com",
            "role": "admin",
            "is_active": True
        }
        mock_service.authenticate_user.return_value = mock_user_data

        login_data = {
            "username": "admin",
            "password": "admin"
        }

        response = client.post("/auth/login", json=login_data)

        assert response.status_code == 200

        # Verify audit logging was called
        mock_log_action.assert_called()
        call_args = mock_log_action.call_args[1]  # Get keyword arguments
        assert call_args["action"] == "login"
        assert call_args["resource_type"] == "authentication"

    @patch('src.web_ui.backend.auth_routes.log_user_action')
    def test_logout_with_audit_logging(self, mock_log_action, authenticated_client_admin, mock_admin_user):
        """Test logout with audit logging."""
        response = authenticated_client_admin.post("/auth/logout")

        assert response.status_code == 200

        # Verify audit logging was called
        mock_log_action.assert_called()
        call_args = mock_log_action.call_args[1]  # Get keyword arguments
        assert call_args["action"] == "logout"
        assert call_args["resource_type"] == "authentication"


class TestAuthenticationValidation:
    """Test cases for authentication input validation."""

    def test_login_data_validation_types(self, client):
        """Test login data type validation."""
        # Test with non-string username
        login_data = {
            "username": 123,
            "password": "admin"
        }

        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 422

        # Test with non-string password
        login_data = {
            "username": "admin",
            "password": 123
        }

        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 422

    def test_refresh_token_validation(self, client):
        """Test refresh token validation."""
        # Test with non-string refresh token
        refresh_data = {
            "refresh_token": 123
        }

        response = client.post("/auth/refresh", json=refresh_data)
        assert response.status_code == 422

    def test_login_username_length_limits(self, client):
        """Test username length validation."""
        # Test with very long username
        long_username = "a" * 1000
        login_data = {
            "username": long_username,
            "password": "admin"
        }

        response = client.post("/auth/login", json=login_data)
        # Should handle gracefully (either validation error or authentication failure)
        assert response.status_code in [400, 401, 422]

    @patch('src.web_ui.backend.auth_routes.webui_app_service')
    def test_login_special_characters(self, mock_service, client):
        """Test login with special characters in credentials."""
        mock_service.authenticate_user.return_value = None

        # Test with special characters
        login_data = {
            "username": "admin@test.com",
            "password": "p@ssw0rd!#$"
        }

        response = client.post("/auth/login", json=login_data)

        # Should handle special characters properly
        assert response.status_code == 401  # Invalid credentials (expected)
        mock_service.authenticate_user.assert_called_once_with("admin@test.com", "p@ssw0rd!#$")


if __name__ == "__main__":
    pytest.main([__file__])