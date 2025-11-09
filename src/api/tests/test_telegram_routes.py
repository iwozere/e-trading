#!/usr/bin/env python3
"""
Unit Tests for Telegram Bot Management API Routes
------------------------------------------------

Tests for the Telegram bot management endpoints including:
- User management (list, verify, approve, reset)
- Alert management (list, toggle, delete, config)
- Schedule management (list, toggle, delete, update)
- Broadcast messaging
- Audit logging and statistics
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.api.main import app
from src.api.models import User
from src.api.auth import require_admin

# Create test client
client = TestClient(app)

# Mock user for authentication
mock_admin_user = User(
    id=1,
    email="admin@test.com",
    role="admin",
    is_active=True
)

mock_regular_user = User(
    id=2,
    email="user@test.com",
    role="trader",
    is_active=True
)

class TestTelegramUserManagement:
    """Test cases for Telegram user management endpoints."""

    def setup_method(self):
        """Set up test dependencies."""
        app.dependency_overrides.clear()

    def teardown_method(self):
        """Clean up test dependencies."""
        app.dependency_overrides.clear()

    @patch('src.api.telegram_routes.telegram_app_service')
    def test_get_telegram_users_all(self, mock_service):
        """Test getting all Telegram users."""
        # Mock authentication by overriding the dependency
        app.dependency_overrides[require_admin] = lambda: mock_admin_user

        # Mock database response
        mock_users = [
            {
                'telegram_user_id': '123456789',
                'email': 'test@example.com',
                'verified': True,
                'approved': False,
                'language': 'en',
                'is_admin': False,
                'max_alerts': 5,
                'max_schedules': 5
            },
            {
                'telegram_user_id': '987654321',
                'email': 'user2@example.com',
                'verified': False,
                'approved': False,
                'language': 'en',
                'is_admin': False,
                'max_alerts': 5,
                'max_schedules': 5
            }
        ]
        mock_service.get_users_list.return_value = mock_users

        # Make request
        response = client.get("/api/telegram/users")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data['data']) == 2
        assert data['data'][0]['telegram_user_id'] == '123456789'
        assert data['data'][0]['verified'] is True
        assert data['data'][1]['verified'] is False
        mock_service.get_users_list.assert_called_once()



    @patch('src.api.telegram_routes.telegram_app_service')
    def test_get_telegram_users_verified_filter(self, mock_service):
        """Test getting verified Telegram users only."""
        # Mock authentication
        app.dependency_overrides[require_admin] = lambda: mock_admin_user

        # Mock database response
        mock_users = [
            {
                'telegram_user_id': '123456789',
                'email': 'test@example.com',
                'verified': True,
                'approved': False,
                'language': 'en',
                'is_admin': False,
                'max_alerts': 5,
                'max_schedules': 5
            },
            {
                'telegram_user_id': '987654321',
                'email': 'user2@example.com',
                'verified': False,
                'approved': False,
                'language': 'en',
                'is_admin': False,
                'max_alerts': 5,
                'max_schedules': 5
            }
        ]
        mock_service.get_users_list.return_value = mock_users

        # Make request with filter
        response = client.get("/api/telegram/users?filter=verified")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1  # Only verified user
        assert data[0]['telegram_user_id'] == '123456789'
        assert data[0]['verified'] is True

    @patch('src.api.telegram_routes.telegram_app_service')
    def test_verify_telegram_user_success(self, mock_service):
        """Test successful user verification."""
        # Mock authentication
        app.dependency_overrides[require_admin] = lambda: mock_admin_user

        # Mock service response
        mock_service.verify_user.return_value = {"message": "User verified successfully"}

        # Make request
        response = client.post("/api/telegram/users/123456789/verify")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "verified successfully" in data['message']
        mock_service.verify_user.assert_called_once_with('123456789')

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_approve_telegram_user_success(self, mock_auth, mock_service):
        """Test successful user approval."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock service responses
        mock_service.approve_user.return_value = {"message": "User approved successfully"}

        # Make request
        response = client.post("/api/telegram/users/123456789/approve")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "approved successfully" in data['message']
        mock_service.approve_user.assert_called_once_with('123456789')

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_approve_telegram_user_not_verified(self, mock_auth, mock_service):
        """Test user approval when user is not verified."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock service response - user not verified (should raise HTTPException)
        from fastapi import HTTPException
        mock_service.approve_user.side_effect = HTTPException(status_code=400, detail="User must be verified first")

        # Make request
        response = client.post("/api/telegram/users/123456789/approve")

        # Assertions
        assert response.status_code == 400
        data = response.json()
        assert "must be verified" in data['detail']

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_approve_telegram_user_not_found(self, mock_auth, mock_service):
        """Test user approval when user doesn't exist."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock service response - user not found (should raise HTTPException)
        from fastapi import HTTPException
        mock_service.approve_user.side_effect = HTTPException(status_code=404, detail="User not found")

        # Make request
        response = client.post("/api/telegram/users/999999999/approve")

        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data['detail']

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_reset_telegram_user_email_success(self, mock_auth, mock_service):
        """Test successful email reset."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock database responses
        mock_service.reset_user_email.return_value = {"message": "Email reset successfully"}

        # Make request
        response = client.post("/api/telegram/users/123456789/reset-email")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "Email reset" in data['message']
        mock_service.reset_user_email.assert_called_once_with('123456789')

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.get_current_user')
    def test_get_telegram_user_stats(self, mock_auth, mock_service):
        """Test getting user statistics."""
        # Mock authentication
        mock_auth.return_value = mock_regular_user

        # Mock database response
        mock_users = [
            {'verified': True, 'approved': True, 'is_admin': False},
            {'verified': True, 'approved': False, 'is_admin': False},
            {'verified': False, 'approved': False, 'is_admin': False},
            {'verified': True, 'approved': True, 'is_admin': True}
        ]
        mock_service.get_user_stats.return_value = {
            "total_users": 3,
            "verified_users": 2,
            "approved_users": 1,
            "pending_approvals": 1,
            "admin_users": 1
        }

        # Make request
        response = client.get("/api/telegram/stats/users")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data['total_users'] == 4
        assert data['verified_users'] == 3
        assert data['approved_users'] == 2
        assert data['pending_approvals'] == 1  # verified but not approved
        assert data['admin_users'] == 1


class TestTelegramAlertManagement:
    """Test cases for Telegram alert management endpoints."""

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.get_current_user')
    def test_get_telegram_alerts_all(self, mock_auth, mock_service):
        """Test getting all Telegram alerts."""
        # Mock authentication
        mock_auth.return_value = mock_regular_user

        # Mock database response
        mock_alerts = [
            {
                'id': 1,
                'user_id': '123456789',
                'ticker': 'BTCUSDT',
                'price': 50000.0,
                'condition': 'above',
                'active': True,
                'email': False,
                'created': '2024-01-01T00:00:00Z'
            }
        ]
        mock_service.get_alerts_by_type.return_value = mock_alerts

        # Make request
        response = client.get("/api/telegram/alerts")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['id'] == 1
        assert data[0]['ticker'] == 'BTCUSDT'
        assert data[0]['active'] is True

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_toggle_telegram_alert_success(self, mock_auth, mock_service):
        """Test successful alert toggle."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock database responses
        mock_service.get_alert.return_value = {'id': 1, 'active': True}
        mock_service.update_alert.return_value = True

        # Make request
        response = client.post("/api/telegram/alerts/1/toggle")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "deactivated successfully" in data['message']
        mock_service.update_alert.assert_called_once_with(1, active=False)

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_delete_telegram_alert_success(self, mock_auth, mock_service):
        """Test successful alert deletion."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock database responses
        mock_service.get_alert.return_value = {'id': 1, 'active': True}
        mock_service.delete_alert.return_value = True

        # Make request
        response = client.delete("/api/telegram/alerts/1")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data['message']
        mock_service.delete_alert.assert_called_once_with(1)

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_delete_telegram_alert_not_found(self, mock_auth, mock_service):
        """Test alert deletion when alert doesn't exist."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock database response - alert not found
        mock_service.get_alert.return_value = None

        # Make request
        response = client.delete("/api/telegram/alerts/999")

        # Assertions
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data['detail']
        mock_service.delete_alert.assert_not_called()


class TestTelegramBroadcast:
    """Test cases for Telegram broadcast endpoints."""

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.require_admin')
    def test_send_telegram_broadcast_success(self, mock_auth, mock_service):
        """Test successful broadcast sending."""
        # Mock authentication
        mock_auth.return_value = mock_admin_user

        # Mock database response - approved users
        mock_users = [
            {'approved': True},
            {'approved': True},
            {'approved': False}  # This user won't receive the broadcast
        ]
        mock_service.list_users.return_value = mock_users

        # Make request
        response = client.post(
            "/api/telegram/broadcast",
            json={"message": "Test broadcast message"}
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data['total_recipients'] == 2  # Only approved users
        assert data['successful_deliveries'] == 2
        assert data['failed_deliveries'] == 0


class TestTelegramAudit:
    """Test cases for Telegram audit endpoints."""

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.get_current_user')
    def test_get_telegram_audit_logs(self, mock_auth, mock_service):
        """Test getting audit logs."""
        # Mock authentication
        mock_auth.return_value = mock_regular_user

        # Mock database response
        mock_logs = [
            {
                'id': 1,
                'telegram_user_id': '123456789',
                'command': '/start',
                'success': True,
                'created': '2024-01-01T00:00:00Z'
            }
        ]
        mock_service.get_all_command_audit.return_value = mock_logs

        # Make request
        response = client.get("/api/telegram/audit")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['command'] == '/start'
        assert data[0]['success'] is True

    @patch('src.api.telegram_routes.telegram_app_service')
    @patch('src.api.telegram_routes.get_current_user')
    def test_get_user_audit_logs(self, mock_auth, mock_service):
        """Test getting audit logs for specific user."""
        # Mock authentication
        mock_auth.return_value = mock_regular_user

        # Mock database response
        mock_logs = [
            {
                'id': 1,
                'command': '/start',
                'success': True,
                'created': '2024-01-01T00:00:00Z'
            }
        ]
        mock_service.get_user_command_history.return_value = mock_logs

        # Make request
        response = client.get("/api/telegram/users/123456789/audit")

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]['command'] == '/start'
        mock_service.get_user_command_history.assert_called_once_with('123456789', 50)


if __name__ == "__main__":
    pytest.main([__file__])