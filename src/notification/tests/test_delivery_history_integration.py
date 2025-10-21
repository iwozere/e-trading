"""
Integration Tests for Delivery History API

Integration tests that use the actual PostgreSQL database.
"""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.main import app
from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, MessageStatus, DeliveryStatus, MessagePriority
)
from src.data.db.core.database import session_scope
from src.data.db.core.base import Base
from src.data.db.core.database import engine


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Setup database tables for testing."""
    # Create all tables
    Base.metadata.create_all(engine)
    yield
    # Cleanup is handled by the database


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_data():
    """Create sample data in the database."""
    with session_scope() as session:
        # Create sample message
        message = Message(
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
        session.add(message)
        session.flush()  # Get the ID

        # Create sample delivery statuses
        delivery1 = MessageDeliveryStatus(
            message_id=message.id,
            channel="telegram",
            status=DeliveryStatus.DELIVERED.value,
            delivered_at=datetime.now(timezone.utc),
            response_time_ms=150,
            error_message=None,
            external_id="tg_msg_123",
            created_at=datetime.now(timezone.utc)
        )

        delivery2 = MessageDeliveryStatus(
            message_id=message.id,
            channel="email",
            status=DeliveryStatus.DELIVERED.value,
            delivered_at=datetime.now(timezone.utc),
            response_time_ms=250,
            error_message=None,
            external_id="email_msg_456",
            created_at=datetime.now(timezone.utc)
        )

        session.add(delivery1)
        session.add(delivery2)
        session.commit()

        return {
            "message": message,
            "deliveries": [delivery1, delivery2]
        }


class TestDeliveryHistoryIntegration:
    """Integration tests for delivery history API."""

    def test_get_message_history_basic(self, client, sample_data):
        """Test basic message history retrieval."""
        response = client.get("/api/v1/history/messages")

        assert response.status_code == 200
        data = response.json()

        assert "messages" in data
        assert "pagination" in data
        assert "filters" in data

        assert data["pagination"]["total"] >= 1
        assert len(data["messages"]) >= 1

        # Check message structure
        message = data["messages"][0]
        assert "id" in message
        assert "message_type" in message
        assert "priority" in message
        assert "channels" in message
        assert "status" in message

    def test_get_message_history_with_filters(self, client, sample_data):
        """Test message history with filters."""
        response = client.get(
            "/api/v1/history/messages",
            params={
                "user_id": "user123",
                "status": "DELIVERED",
                "message_type": "test_alert",
                "limit": 50
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["user_id"] == "user123"
        assert data["filters"]["status"] == "DELIVERED"
        assert data["filters"]["message_type"] == "test_alert"

        # Should find our test message
        assert data["pagination"]["total"] >= 1

    def test_get_message_history_with_channel_filter(self, client, sample_data):
        """Test message history with channel filter."""
        response = client.get(
            "/api/v1/history/messages",
            params={
                "channel": "telegram",
                "limit": 50
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["channel"] == "telegram"

        # Should find our test message which has telegram channel
        assert data["pagination"]["total"] >= 1

        # Verify the message has telegram in channels
        if data["messages"]:
            message = data["messages"][0]
            assert "telegram" in message["channels"]

    def test_get_delivery_history_basic(self, client, sample_data):
        """Test basic delivery history retrieval."""
        response = client.get("/api/v1/history/deliveries")

        assert response.status_code == 200
        data = response.json()

        assert "deliveries" in data
        assert "pagination" in data
        assert "filters" in data

        assert data["pagination"]["total"] >= 2  # We created 2 deliveries
        assert len(data["deliveries"]) >= 2

    def test_get_delivery_history_with_user_filter(self, client, sample_data):
        """Test delivery history with user filter (requires join)."""
        response = client.get(
            "/api/v1/history/deliveries",
            params={"user_id": "user123"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["user_id"] == "user123"
        assert data["pagination"]["total"] >= 2  # Should find our deliveries

    def test_get_delivery_history_with_channel_filter(self, client, sample_data):
        """Test delivery history with channel filter."""
        response = client.get(
            "/api/v1/history/deliveries",
            params={"channel": "telegram"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["channel"] == "telegram"
        assert data["pagination"]["total"] >= 1  # Should find telegram delivery

        # Verify all deliveries are for telegram channel
        for delivery in data["deliveries"]:
            assert delivery["channel"] == "telegram"

    def test_export_history_json_small(self, client, sample_data):
        """Test small JSON export (immediate)."""
        response = client.get(
            "/api/v1/history/export",
            params={"format": "json", "limit": 100}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["export_type"] == "immediate"
        assert data["format"] == "json"
        assert "data" in data
        assert data["record_count"] >= 1

        # Check data structure
        if data["data"]:
            message_data = data["data"][0]
            assert "message_id" in message_data
            assert "deliveries" in message_data
            assert len(message_data["deliveries"]) >= 1

    def test_export_history_csv_small(self, client, sample_data):
        """Test small CSV export (immediate)."""
        response = client.get(
            "/api/v1/history/export",
            params={"format": "csv", "limit": 100}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["export_type"] == "immediate"
        assert data["format"] == "csv"
        assert "data" in data
        assert isinstance(data["data"], str)  # CSV data as string

        # Check CSV headers are present
        csv_data = data["data"]
        assert "message_id" in csv_data
        assert "delivery_id" in csv_data
        assert "channel" in csv_data

    def test_get_history_summary(self, client, sample_data):
        """Test history summary endpoint."""
        response = client.get(
            "/api/v1/history/summary",
            params={"days": 30}
        )

        assert response.status_code == 200
        data = response.json()

        assert "period" in data
        assert "message_statistics" in data
        assert "delivery_statistics" in data
        assert "channel_breakdown" in data

        assert data["period"]["days"] == 30
        assert data["message_statistics"]["total"] >= 1
        assert data["delivery_statistics"]["total"] >= 2

    def test_get_history_summary_with_filters(self, client, sample_data):
        """Test history summary with filters."""
        response = client.get(
            "/api/v1/history/summary",
            params={
                "days": 30,
                "user_id": "user123",
                "channel": "telegram"
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["user_id"] == "user123"
        assert data["filters"]["channel"] == "telegram"

        # When filtering by specific channel, channel breakdown should be empty
        # since we're already filtering by that channel
        assert data["channel_breakdown"] == {}

    def test_pagination_functionality(self, client, sample_data):
        """Test pagination works correctly."""
        # Test with small limit
        response = client.get(
            "/api/v1/history/messages",
            params={"limit": 1, "offset": 0}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["pagination"]["limit"] == 1
        assert data["pagination"]["offset"] == 0
        assert len(data["messages"]) <= 1

        if data["pagination"]["total"] > 1:
            assert data["pagination"]["has_more"] is True

    def test_date_range_filtering(self, client, sample_data):
        """Test date range filtering."""
        # Test with date range that should include our test data
        start_date = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        end_date = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

        response = client.get(
            "/api/v1/history/messages",
            params={
                "start_date": start_date,
                "end_date": end_date
            }
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["start_date"] == start_date
        assert data["filters"]["end_date"] == end_date
        assert data["pagination"]["total"] >= 1

    def test_ordering_functionality(self, client, sample_data):
        """Test ordering functionality."""
        # Test ascending order
        response = client.get(
            "/api/v1/history/messages",
            params={
                "order_by": "created_at",
                "order_desc": False
            }
        )

        assert response.status_code == 200
        data = response.json()

        # If we have multiple messages, verify ordering
        if len(data["messages"]) > 1:
            dates = [msg["created_at"] for msg in data["messages"]]
            assert dates == sorted(dates)


if __name__ == "__main__":
    pytest.main([__file__])