"""
Tests for notification routes
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Mock the notification service calls for testing
@pytest.fixture
def mock_notification_service():
    with patch('src.api.notification_routes._call_notification_service') as mock:
        yield mock


def test_notification_routes_health_endpoint(authenticated_client_viewer):
    """Test the notification routes health endpoint."""
    client = authenticated_client_viewer
    with patch('src.api.notification_routes._call_notification_service') as mock_service:
        mock_service.return_value = {
            "status": "healthy",
            "service": "notification_service",
            "version": "1.0.0"
        }

        response = client.get("/api/notifications/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "notification_service" in data
        assert data["routes"] == "operational"


def test_create_notification_endpoint_structure(authenticated_client_trader):
    """Test the create notification endpoint structure (without actual service call)."""
    client = authenticated_client_trader
    notification_data = {
        "message_type": "alert",
        "priority": "normal",
        "channels": ["telegram"],
        "recipient_id": "test_user",
        "content": {
            "title": "Test Alert",
            "message": "This is a test alert"
        }
    }

    with patch('src.api.notification_routes._call_notification_service') as mock_service:
        mock_service.return_value = {
            "message_id": 123,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "normal"
        }

        response = client.post(
            "/api/notifications/",
            json=notification_data
        )

        # Should succeed with mocked service
        assert response.status_code == 200
        data = response.json()
        assert data["message_id"] == 123
        assert data["status"] == "enqueued"


def test_list_channels_endpoint(authenticated_client_viewer):
    """Test the list channels endpoint."""
    client = authenticated_client_viewer
    with patch('src.api.notification_routes._call_notification_service') as mock_service:
        mock_service.return_value = [
            {
                "channel": "telegram",
                "enabled": True,
                "rate_limit_per_minute": 60
            },
            {
                "channel": "email",
                "enabled": True,
                "rate_limit_per_minute": 30
            }
        ]

        response = client.get("/api/notifications/channels")
        assert response.status_code == 200

        data = response.json()
        assert "channels" in data
        assert len(data["channels"]) == 2


def test_send_alert_convenience_endpoint(authenticated_client_trader):
    """Test the convenience alert endpoint."""
    client = authenticated_client_trader
    alert_data = {
        "title": "Test Alert",
        "message": "This is a test alert message",
        "severity": "high",
        "channels": ["telegram", "email"]
    }

    with patch('src.api.notification_routes._call_notification_service') as mock_service:
        mock_service.return_value = {
            "message_id": 456,
            "status": "enqueued",
            "channels": ["telegram", "email"],
            "priority": "high"
        }

        response = client.post(
            "/api/notifications/alert",
            json=alert_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message_id"] == 456


def test_send_trade_convenience_endpoint(authenticated_client_trader):
    """Test the convenience trade notification endpoint."""
    client = authenticated_client_trader
    trade_data = {
        "action": "buy",
        "symbol": "BTCUSDT",
        "quantity": 0.1,
        "price": 45000.0,
        "strategy_name": "test_strategy",
        "channels": ["telegram"]
    }

    with patch('src.api.notification_routes._call_notification_service') as mock_service:
        mock_service.return_value = {
            "message_id": 789,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "normal"
        }

        response = client.post(
            "/api/notifications/trade",
            json=trade_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["message_id"] == 789


def test_notification_service_unavailable_handling(client: TestClient):
    """Test handling when notification service is unavailable."""
    with patch('src.api.notification_routes._call_notification_service') as mock_service:
        from httpx import ConnectError
        mock_service.side_effect = ConnectError("Connection failed")

        response = client.get("/api/notifications/health")
        assert response.status_code == 200  # Health endpoint handles errors gracefully

        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["notification_service"] == "unavailable"


def test_get_notification_statistics(authenticated_client_viewer):
    """Test getting notification statistics."""
    client = authenticated_client_viewer
    with patch('src.api.notification_routes._call_notification_service') as mock_service:
        # Mock the stats call
        mock_service.return_value = {
            "delivery_statistics": {
                "delivered": 100,
                "failed": 5,
                "pending": 2
            },
            "message_statistics": {
                "delivered": 95,
                "failed": 5,
                "pending": 2,
                "processing": 3
            }
        }

        response = client.get("/api/notifications/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_messages"] == 105  # Sum of message stats
        assert data["delivered_messages"] == 100
        assert data["failed_messages"] == 5
        assert data["pending_messages"] == 2
        assert data["success_rate"] > 0.9  # Should be around 95%