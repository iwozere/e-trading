"""
Unit tests for password change and password reset endpoints.
"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.api.auth import get_current_user
from src.api.main import app
from src.api.rate_limiter import limiter
from src.data.db.models.model_users import AuthIdentity, User, VerificationCode


@pytest.fixture(autouse=True)
def bypass_rate_limit(monkeypatch):
    """Disable the rate limiter for unit tests."""
    monkeypatch.setattr(limiter, "enabled", False)


def _make_user(id: int = 10, email: str = "user@test.com", role: str = "trader") -> User:
    u = User(id=id, email=email, role=role, is_active=True)
    u.set_password("oldpassword")
    return u


def _make_db_mock(user=None, ident=None, verification_code=None) -> Mock:
    """Build a mock get_database_service() whose uow() session returns mocks."""
    session = MagicMock()
    
    def query_side_effect(model):
        q = MagicMock()
        if model == User:
            q.filter.return_value.first.return_value = user
        elif model == AuthIdentity:
            q.filter.return_value.first.return_value = ident
        elif model == VerificationCode:
            # Handle both list (.all()) and single (.first()) retrievals
            q.filter.return_value.order_by.return_value.all.return_value = [verification_code] if verification_code else []
            q.filter.return_value.order_by.return_value.first.return_value = verification_code
            q.filter.return_value.first.return_value = verification_code
        return q
        
    session.query.side_effect = query_side_effect
    session.query.return_value.filter.return_value.delete.return_value = 1

    uow_ctx = MagicMock()
    uow_ctx.__enter__ = Mock(return_value=MagicMock(s=session))
    uow_ctx.__exit__ = Mock(return_value=None)

    db = Mock()
    db.uow.return_value = uow_ctx
    return db


def _make_notify_mock():
    """Async context manager mock for NotificationServiceClient."""
    notify_client = AsyncMock()
    notify_client.send_notification = AsyncMock(return_value=True)
    cls = MagicMock()
    cls.return_value.__aenter__ = AsyncMock(return_value=notify_client)
    cls.return_value.__aexit__ = AsyncMock(return_value=None)
    return cls, notify_client


class TestChangePassword:
    @patch("src.api.auth_routes.get_database_service")
    def test_change_password_success(self, mock_get_db):
        """Test successful password change for authenticated non-special user."""
        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        mock_get_db.return_value = _make_db_mock(user=user)

        client = TestClient(app)
        response = client.post(
            "/auth/change-password",
            json={"current_password": "oldpassword", "new_password": "newpassword"},
            headers={"Authorization": "Bearer fake-token"}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"detail": "Password updated successfully"}
        assert user.verify_password("newpassword")

        app.dependency_overrides.clear()

    def test_change_password_special_user_blocked(self):
        """Test that special users are blocked from changing password."""
        special_user = _make_user(email="admin@trading-system.local", role="admin")
        app.dependency_overrides[get_current_user] = lambda: special_user

        client = TestClient(app)
        response = client.post(
            "/auth/change-password",
            json={"current_password": "admin", "new_password": "newpassword"},
            headers={"Authorization": "Bearer fake-token"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "not permitted" in response.json()["detail"]

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.get_database_service")
    def test_change_password_incorrect_current(self, mock_get_db):
        """Test changing password with incorrect current password."""
        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        mock_get_db.return_value = _make_db_mock(user=user)

        client = TestClient(app)
        response = client.post(
            "/auth/change-password",
            json={"current_password": "wrongpassword", "new_password": "newpassword"},
            headers={"Authorization": "Bearer fake-token"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Incorrect current password" in response.json()["detail"]

        app.dependency_overrides.clear()


class TestResetPassword:
    @patch("src.api.auth_routes.NotificationServiceClient")
    @patch("src.api.auth_routes.get_database_service")
    def test_reset_request_email_success(self, mock_get_db, mock_notify_cls):
        """Test password reset request via email successfully sends code."""
        user = _make_user(email="user@test.com")
        mock_get_db.return_value = _make_db_mock(user=user)
        
        mock_cls, mock_client = _make_notify_mock()
        mock_notify_cls.side_effect = mock_cls.side_effect
        mock_notify_cls.return_value = mock_cls.return_value

        client = TestClient(app)
        response = client.post(
            "/auth/reset-password/request",
            json={"identity": "user@test.com"}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["channel"] == "email"
        assert "Verification code sent" in response.json()["detail"]

    @patch("src.api.auth_routes.NotificationServiceClient")
    @patch("src.api.auth_routes.get_database_service")
    def test_reset_request_telegram_success(self, mock_get_db, mock_notify_cls):
        """Test password reset request via telegram successfully sends code."""
        user = _make_user(email="user@test.com")
        ident = Mock(spec=AuthIdentity)
        ident.user_id = user.id
        ident.provider = "telegram"
        ident.external_id = "12345678"

        mock_get_db.return_value = _make_db_mock(user=user, ident=ident)
        
        mock_cls, mock_client = _make_notify_mock()
        mock_notify_cls.side_effect = mock_cls.side_effect
        mock_notify_cls.return_value = mock_cls.return_value

        client = TestClient(app)
        response = client.post(
            "/auth/reset-password/request",
            json={"identity": "telegram_12345678"}
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["channel"] == "telegram"

    @patch("src.api.auth_routes.get_database_service")
    def test_reset_request_special_user_blocked(self, mock_get_db):
        """Test that password reset is blocked for special users."""
        special_user = _make_user(email="admin@trading-system.local")
        mock_get_db.return_value = _make_db_mock(user=special_user)

        client = TestClient(app)
        response = client.post(
            "/auth/reset-password/request",
            json={"identity": "admin@trading-system.local"}
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not permitted" in response.json()["detail"]

    @patch("src.api.auth_routes.get_database_service")
    def test_confirm_reset_success(self, mock_get_db):
        """Test successful confirm of password reset with valid code."""
        user = _make_user()
        code_row = VerificationCode(
            user_id=user.id,
            code="123456",
            sent_time=int(time.time()),
            provider="reset_email"
        )

        mock_get_db.return_value = _make_db_mock(user=user, verification_code=code_row)

        client = TestClient(app)
        response = client.post(
            "/auth/reset-password/confirm",
            json={
                "identity": "user@test.com",
                "code": "123456",
                "new_password": "brandnewpassword"
            }
        )

        assert response.status_code == status.HTTP_200_OK
        assert "Password reset successfully" in response.json()["detail"]
        assert user.verify_password("brandnewpassword")

    @patch("src.api.auth_routes.get_database_service")
    def test_confirm_reset_invalid_code(self, mock_get_db):
        """Test password reset confirm fails with invalid code."""
        user = _make_user()
        code_row = VerificationCode(
            user_id=user.id,
            code="123456",
            sent_time=int(time.time()),
            provider="reset_email"
        )

        mock_get_db.return_value = _make_db_mock(user=user, verification_code=code_row)

        client = TestClient(app)
        response = client.post(
            "/auth/reset-password/confirm",
            json={
                "identity": "user@test.com",
                "code": "654321",
                "new_password": "brandnewpassword"
            }
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid or expired" in response.json()["detail"]
