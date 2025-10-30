"""
Test Delivery History API

Tests for the delivery history API endpoints.
"""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.main import app
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, MessageStatus, DeliveryStatus, MessagePriority
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_repo():
    """Create mock repository."""
    repo = Mock()
    repo.session = Mock()
    repo.commit = Mock()
    repo.rollback = Mock()
    repo.close = Mock()
    return repo


@pytest.fixture
def sample_message():
    """Create sample message."""
    return Message(
        id=1,
        message_type="test_alert",
        priority=MessagePriority.NORMAL.value,
        channels=["telegram", "email"],
        recipient_id="user123",
        template_name="alert_template",
        content={"title": "Test Alert", "message": "This is a test"},
        message_metadata={"source": "test"},
        created_at=datetime.now(timezone.utc),
        scheduled_for=datetime.now(timezone.utc),
        status=MessageStatus.DELIVERED.value,
        retry_count=0,
        max_retries=3,
        last_error=None,
        processed_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_delivery():
    """Create sample delivery status."""
    return MessageDeliveryStatus(
        id=1,
        message_id=1,
        channel="telegram",
        status=DeliveryStatus.DELIVERED.value,
        delivered_at=datetime.now(timezone.utc),
        response_time_ms=150,
        error_message=None,
        external_id="tg_msg_123",
        created_at=datetime.now(timezone.utc)
    )


class TestDeliveryHistoryAPI:
    """Test delivery history API endpoints."""

    @patch('src.notification.service.main.get_notification_repo')
    def test_get_message_history_basic(self, mock_get_repo, client, mock_repo, sample_message):
        """Test basic message history retrieval."""
        # Setup mock
        mock_get_repo.return_value = mock_repo
        mock_query = Mock()
        mock_repo.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [sample_message]

        # Make request
        response = client.get("/api/v1/history/messages")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert "pagination" in data
        assert "filters" in data
        assert data["pagination"]["total"] == 1
        assert len(data["messages"]) == 1

    @patch('src.notification.service.main.get_notification_repo')
    def test_get_message_history_with_filters(self, mock_get_repo, client, mock_repo, sample_message):
        """Test message history with filters."""
        # Setup mock
        mock_get_repo.return_value = mock_repo
        mock_query = Mock()
        mock_repo.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [sample_message]

        # Make request with filters
        response = client.get(
            "/api/v1/history/messages",
            params={
                "user_id": "user123",
                "channel": "telegram",
                "status": "DELIVERED",
                "message_type": "test_alert",
                "limit": 50,
                "offset": 0
            }
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["user_id"] == "user123"
        assert data["filters"]["channel"] == "telegram"
        assert data["filters"]["status"] == "DELIVERED"
        assert data["filters"]["message_type"] == "test_alert"

    @patch('src.notification.service.main.get_notification_repo')
    def test_get_delivery_history_basic(self, mock_get_repo, client, mock_repo, sample_delivery):
        """Test basic delivery history retrieval."""
        # Setup mock
        mock_get_repo.return_value = mock_repo
        mock_query = Mock()
        mock_repo.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [sample_delivery]

        # Make request
        response = client.get("/api/v1/history/deliveries")

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "deliveries" in data
        assert "pagination" in data
        assert "filters" in data
        assert data["pagination"]["total"] == 1
        assert len(data["deliveries"]) == 1

    @patch('src.notification.service.main.get_notification_repo')
    def test_get_delivery_history_with_user_filter(self, mock_get_repo, client, mock_repo, sample_delivery):
        """Test delivery history with user filter (requires join)."""
        # Setup mock
        mock_get_repo.return_value = mock_repo
        mock_query = Mock()
        mock_repo.session.query.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value = mock_query
        mock_query.offset.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = [sample_delivery]

        # Make request with user filter
        response = client.get(
            "/api/v1/history/deliveries",
            params={"user_id": "user123"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["filters"]["user_id"] == "user123"

    @patch('src.notification.service.main.get_notification_repo')
    def test_export_history_json_small(self, mock_get_repo, client, mock_repo, sample_message, sample_delivery):
        """Test small JSON export (immediate)."""
        # Setup mock
        mock_get_repo.return_value = mock_repo
        mock_query = Mock()
        mock_repo.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        # Setup message with delivery statuses
        sample_message.delivery_statuses = [sample_delivery]
        mock_query.all.return_value = [sample_message]

        # Make request
        response = client.get(
            "/api/v1/history/export",
            params={"format": "json", "limit": 100}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["export_type"] == "immediate"
        assert data["format"] == "json"
        assert "data" in data
        assert data["record_count"] == 1

    @patch('src.notification.service.main.get_notification_repo')
    def test_export_history_csv_small(self, mock_get_repo, client, mock_repo, sample_message, sample_delivery):
        """Test small CSV export (immediate)."""
        # Setup mock
        mock_get_repo.return_value = mock_repo
        mock_query = Mock()
        mock_repo.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query

        # Setup message with delivery statuses
        sample_message.delivery_statuses = [sample_delivery]
        mock_query.all.return_value = [sample_message]

        # Make request
        response = client.get(
            "/api/v1/history/export",
            params={"format": "csv", "limit": 100}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["export_type"] == "immediate"
        assert data["format"] == "csv"
        assert "data" in data
        assert isinstance(data["data"], str)  # CSV data as string

    def test_export_history_large_background(self, client):
        """Test large export (background processing)."""
        # Make request for large export
        response = client.get(
            "/api/v1/history/export",
            params={"format": "json", "limit": 5000}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["export_type"] == "background"
        assert "export_id" in data
        assert data["status"] == "processing"

    @patch('src.notification.service.main.get_notification_repo')
    def test_get_history_summary(self, mock_get_repo, client, mock_repo):
        """Test history summary endpoint."""
        # Setup mock
        mock_get_repo.return_value = mock_repo
        mock_query = Mock()
        mock_repo.session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.join.return_value = mock_query
        mock_query.count.return_value = 10
        mock_query.group_by.return_value = mock_query
        mock_query.all.return_value = []

        # Make request
        response = client.get(
            "/api/v1/history/summary",
            params={"days": 30}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert "period" in data
        assert "message_statistics" in data
        assert "delivery_statistics" in data
        assert "channel_breakdown" in data
        assert data["period"]["days"] == 30

    def test_invalid_parameters(self, client):
        """Test validation of invalid parameters."""
        # Test invalid limit
        response = client.get(
            "/api/v1/history/messages",
            params={"limit": 2000}  # Over max limit
        )
        assert response.status_code == 400

        # Test invalid offset
        response = client.get(
            "/api/v1/history/messages",
            params={"offset": -1}
        )
        assert response.status_code == 400

        # Test invalid order_by
        response = client.get(
            "/api/v1/history/messages",
            params={"order_by": "invalid_field"}
        )
        assert response.status_code == 400

        # Test invalid export format
        response = client.get(
            "/api/v1/history/export",
            params={"format": "xml"}
        )
        assert response.status_code == 400

        # Test invalid days in summary
        response = client.get(
            "/api/v1/history/summary",
            params={"days": 400}  # Over max days
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__])