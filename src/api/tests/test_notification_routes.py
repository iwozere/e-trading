"""
Tests for notification routes
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _patch_get_db(uow):
    """
    Patch ``get_database_service`` in the routes module so that
    ``with db_service.uow() as uow:`` yields the supplied mock ``uow``.

    The routes use a database-centric architecture (direct UoW access)
    rather than HTTP calls to a separate notification microservice.
    """
    service = MagicMock()
    service.uow.return_value.__enter__.return_value = uow
    service.uow.return_value.__exit__.return_value = False
    return patch("src.api.notification_routes.get_database_service", return_value=service)


def _mock_created_message(message_id, priority="NORMAL"):
    """Build a mock persisted Message returned by the repository layer."""
    created = MagicMock()
    created.id = message_id
    created.priority = priority
    created.scheduled_for = datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC)
    return created


def test_notification_routes_health_endpoint(client: TestClient):
    """Test the notification routes health endpoint (database-centric)."""
    uow = MagicMock()
    uow.s.query.return_value.count.return_value = 42

    with _patch_get_db(uow):
        response = client.get("/api/notifications/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"] == "connected"
    assert data["routes"] == "operational"
    assert data["total_messages"] == 42


def test_create_notification_endpoint_structure(authenticated_client_trader):
    """Test the create notification endpoint structure (mocked DB layer)."""
    client = authenticated_client_trader
    notification_data = {
        "message_type": "alert",
        "priority": "normal",
        "channels": ["telegram"],
        "recipient_id": "test_user",
        "content": {"title": "Test Alert", "message": "This is a test alert"},
    }

    uow = MagicMock()
    uow.notifications.messages.create_message.return_value = _mock_created_message(123)

    with _patch_get_db(uow):
        response = client.post("/api/notifications/", json=notification_data)

    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == 123
    assert data["status"] == "enqueued"
    assert data["channels"] == ["telegram"]


def test_list_channels_endpoint(authenticated_client_viewer):
    """Test the list channels endpoint (defaults added when DB has none)."""
    client = authenticated_client_viewer
    uow = MagicMock()
    uow.notifications.channel_configs.list_channel_configs.return_value = []

    with _patch_get_db(uow):
        response = client.get("/api/notifications/channels")

    assert response.status_code == 200
    data = response.json()
    assert "channels" in data
    channel_names = {ch["channel"] for ch in data["channels"]}
    # When the DB has no configs, the route backfills the default channels.
    assert {"telegram", "email", "sms"}.issubset(channel_names)


def test_send_alert_convenience_endpoint(authenticated_client_trader):
    """Test the convenience alert endpoint."""
    client = authenticated_client_trader
    alert_data = {
        "title": "Test Alert",
        "message": "This is a test alert message",
        "severity": "high",
        "channels": ["telegram", "email"],
    }

    uow = MagicMock()
    uow.notifications.messages.create_message.return_value = _mock_created_message(456, priority="HIGH")

    with _patch_get_db(uow):
        response = client.post("/api/notifications/alert", json=alert_data)

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
        "channels": ["telegram"],
    }

    uow = MagicMock()
    uow.notifications.messages.create_message.return_value = _mock_created_message(789)

    with _patch_get_db(uow):
        response = client.post("/api/notifications/trade", json=trade_data)

    assert response.status_code == 200
    data = response.json()
    assert data["message_id"] == 789


def test_search_messages_endpoint(authenticated_client_trader):
    """Test the message search endpoint returns the service result."""
    client = authenticated_client_trader
    service_result = {
        "total": 1,
        "limit": 100,
        "offset": 0,
        "items": [
            {
                "id": 1,
                "message_type": "alert",
                "priority": "NORMAL",
                "channels": ["telegram"],
                "recipient_id": "user_42",
                "template_name": None,
                "content": {"message": "hello world"},
                "status": "DELIVERED",
                "created_at": "2026-06-30T10:00:00+00:00",
                "scheduled_for": "2026-06-30T10:00:00+00:00",
                "processed_at": "2026-06-30T10:00:01+00:00",
                "retry_count": 0,
                "last_error": None,
            }
        ],
    }

    with patch(
        "src.data.db.services.notification_service.NotificationService.search_messages",
        return_value=service_result,
    ) as mock_search:
        response = client.get(
            "/api/notifications/messages/search",
            params={"recipient_id": "user_42", "search": "hello", "days": 30},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["recipient_id"] == "user_42"
        assert data["items"][0]["content"]["message"] == "hello world"

        # Filters should be forwarded to the service layer.
        _, kwargs = mock_search.call_args
        assert kwargs["recipient_id"] == "user_42"
        assert kwargs["search"] == "hello"


def test_notification_service_unavailable_handling(client: TestClient):
    """Test handling when the database is unavailable."""
    service = MagicMock()
    service.uow.return_value.__enter__.side_effect = Exception("Connection failed")

    with patch("src.api.notification_routes.get_database_service", return_value=service):
        response = client.get("/api/notifications/health")

    # Health endpoint handles errors gracefully and still returns 200.
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["database"] == "unavailable"


def test_get_notification_statistics(authenticated_client_viewer):
    """Test getting notification statistics (mocked DB + health service)."""
    client = authenticated_client_viewer

    uow = MagicMock()
    uow.notifications.delivery_status.get_delivery_statistics.return_value = {
        "delivered": 100,
        "failed": 5,
    }
    # Each MessageStatus bucket reports 2 rows -> 5 statuses == 10 total messages.
    uow.s.query.return_value.filter.return_value.count.return_value = 2

    with _patch_get_db(uow), patch("src.data.db.services.system_health_service.SystemHealthService") as mock_health:
        mock_health.return_value.get_notification_channels_health.return_value = [
            {"channel": "telegram", "status": "healthy"}
        ]

        response = client.get("/api/notifications/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["total_messages"] == 10
    assert data["delivered_messages"] == 100
    assert data["failed_messages"] == 5
    assert data["pending_messages"] == 2
    assert data["channels_health"] == {"telegram": "healthy"}
    assert "success_rate" in data
