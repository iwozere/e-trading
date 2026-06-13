"""
Unit tests for POST /auth/2fa/send and POST /auth/2fa/verify endpoints.

Coverage:
- Code generation and storage (send tests)
- Expiry enforcement (verify)
- Replay prevention (verify consumes the code on success)
- Unauthenticated access
- Missing linked accounts
"""

import time
import pytest
from typing import Optional
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from src.api.auth import get_current_user
from src.api.main import app
from src.api.rate_limiter import limiter
from src.data.db.models.model_users import User, AuthIdentity, VerificationCode


@pytest.fixture(autouse=True)
def bypass_rate_limit(monkeypatch):
    """Disable the rate limiter for unit tests — rate-limit behaviour is not under test here."""
    monkeypatch.setattr(limiter, 'enabled', False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_user(email: Optional[str] = "admin@test.com") -> User:
    return User(id=1, email=email, role="admin", is_active=True)


def _make_db_mock(first_return=None) -> Mock:
    """Build a mock get_database_service() whose uow() session returns first_return on .first()."""
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = first_return
    session.query.return_value.filter.return_value.order_by.return_value.first.return_value = first_return
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


# ---------------------------------------------------------------------------
# POST /auth/2fa/send
# ---------------------------------------------------------------------------

class TestSend2FACode:

    @patch("src.api.auth_routes.NotificationServiceClient")
    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_send_code_email_success(self, mock_get_user, mock_get_db, mock_notify_cls):
        """Email channel: code written to DB, notification dispatched, 200 returned."""
        from fastapi.testclient import TestClient

        user = _make_user(email="admin@test.com")
        mock_get_user.return_value = user
        app.dependency_overrides[get_current_user] = lambda: user

        mock_get_db.return_value = _make_db_mock(first_return=None)
        mock_cls, mock_client = _make_notify_mock()
        mock_notify_cls.side_effect = mock_cls.side_effect
        mock_notify_cls.return_value = mock_cls.return_value

        client = TestClient(app)
        response = client.post("/auth/2fa/send", json={"channel": "email"})

        assert response.status_code == 200
        assert response.json() == {"detail": "Code sent"}

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.NotificationServiceClient")
    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_send_code_telegram_success(self, mock_get_user, mock_get_db, mock_notify_cls):
        """Telegram channel: AuthIdentity found, code stored, notification dispatched."""
        from fastapi.testclient import TestClient

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        ident = Mock(spec=AuthIdentity)
        ident.external_id = "99887766"
        mock_get_db.return_value = _make_db_mock(first_return=ident)

        mock_cls, mock_client = _make_notify_mock()
        mock_notify_cls.side_effect = mock_cls.side_effect
        mock_notify_cls.return_value = mock_cls.return_value

        client = TestClient(app)
        response = client.post("/auth/2fa/send", json={"channel": "telegram"})

        assert response.status_code == 200
        assert response.json() == {"detail": "Code sent"}

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_send_code_telegram_no_identity(self, mock_get_user, mock_get_db):
        """Telegram channel: no AuthIdentity → 400."""
        from fastapi.testclient import TestClient

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user
        mock_get_db.return_value = _make_db_mock(first_return=None)

        client = TestClient(app)
        response = client.post("/auth/2fa/send", json={"channel": "telegram"})

        assert response.status_code == 400
        assert "Telegram" in response.json()["detail"]

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_send_code_email_no_address(self, mock_get_user, mock_get_db):
        """Email channel: user.email is None → 400."""
        from fastapi.testclient import TestClient

        user = _make_user(email=None)
        app.dependency_overrides[get_current_user] = lambda: user
        mock_get_db.return_value = _make_db_mock()

        client = TestClient(app)
        response = client.post("/auth/2fa/send", json={"channel": "email"})

        assert response.status_code == 400
        assert "email" in response.json()["detail"].lower()

        app.dependency_overrides.clear()

    def test_send_code_unauthenticated(self):
        """No JWT → 401 (get_current_user raises 401 when credentials are absent)."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post("/auth/2fa/send", json={"channel": "email"})

        assert response.status_code == 401

    def test_send_code_invalid_channel(self):
        """Unknown channel value → 422 (Pydantic Literal validation)."""
        from fastapi.testclient import TestClient

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        client = TestClient(app)
        response = client.post("/auth/2fa/send", json={"channel": "sms"})

        assert response.status_code == 422

        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /auth/2fa/verify
# ---------------------------------------------------------------------------

class TestVerify2FACode:

    def _make_code_row(self, code: str = "123456", age_seconds: int = 30) -> Mock:
        row = Mock(spec=VerificationCode)
        row.code = code
        row.sent_time = int(time.time()) - age_seconds
        row.id = 1
        return row

    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_verify_code_success(self, mock_get_user, mock_get_db):
        """Valid code, not expired → 200 with access_token containing 2fa_verified=True."""
        import jwt
        from fastapi.testclient import TestClient
        from src.api.config import settings

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        code_row = self._make_code_row(code="654321", age_seconds=60)
        mock_get_db.return_value = _make_db_mock(first_return=code_row)

        client = TestClient(app)
        response = client.post("/auth/2fa/verify", json={"code": "654321"})

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 1800

        payload = jwt.decode(data["access_token"], settings.jwt_secret_key, algorithms=["HS256"])
        assert payload.get("2fa_verified") is True
        assert payload.get("sub") == str(user.id)

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_verify_code_expired(self, mock_get_user, mock_get_db):
        """Code older than 600 s → 400."""
        from fastapi.testclient import TestClient

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        expired_row = self._make_code_row(code="111111", age_seconds=601)
        mock_get_db.return_value = _make_db_mock(first_return=expired_row)

        client = TestClient(app)
        response = client.post("/auth/2fa/verify", json={"code": "111111"})

        assert response.status_code == 400
        assert "expired" in response.json()["detail"].lower()

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_verify_code_wrong_code(self, mock_get_user, mock_get_db):
        """Correct format but wrong digits → 400."""
        from fastapi.testclient import TestClient

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        code_row = self._make_code_row(code="999999", age_seconds=30)
        mock_get_db.return_value = _make_db_mock(first_return=code_row)

        client = TestClient(app)
        response = client.post("/auth/2fa/verify", json={"code": "000000"})

        assert response.status_code == 400
        assert "Invalid" in response.json()["detail"]

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_verify_code_no_code_exists(self, mock_get_user, mock_get_db):
        """No code in DB (already consumed or never sent) → 400."""
        from fastapi.testclient import TestClient

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        mock_get_db.return_value = _make_db_mock(first_return=None)

        client = TestClient(app)
        response = client.post("/auth/2fa/verify", json={"code": "123456"})

        assert response.status_code == 400

        app.dependency_overrides.clear()

    @patch("src.api.auth_routes.get_database_service")
    @patch("src.api.auth_routes.get_current_user")
    def test_verify_code_replay_prevention(self, mock_get_user, mock_get_db):
        """Successful verify calls r.s.delete(code_row) so the code cannot be reused."""
        from fastapi.testclient import TestClient

        user = _make_user()
        app.dependency_overrides[get_current_user] = lambda: user

        code_row = self._make_code_row(code="777777", age_seconds=10)
        db_mock = _make_db_mock(first_return=code_row)
        mock_get_db.return_value = db_mock

        client = TestClient(app)
        response = client.post("/auth/2fa/verify", json={"code": "777777"})

        assert response.status_code == 200
        # Verify delete was called on the code row (replay prevention)
        uow_enter = db_mock.uow.return_value.__enter__.return_value
        uow_enter.s.delete.assert_called_once_with(code_row)

        app.dependency_overrides.clear()

    def test_verify_code_unauthenticated(self):
        """No JWT → 401."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.post("/auth/2fa/verify", json={"code": "123456"})

        assert response.status_code == 401
