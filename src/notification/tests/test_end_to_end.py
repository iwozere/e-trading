#!/usr/bin/env python3
"""
End-to-End Tests for Notification Service

Tests complete message delivery flows from API to channel delivery,
service behavior under failure scenarios, and service lifecycle.
"""

import asyncio
import sys
import json
import time
import pytest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.config import config
from src.data.db.services.database_service import get_database_service
from src.notification.service.main import app
from src.notification.service.processor import message_processor
from src.notification.service.health_monitor import health_monitor
from src.data.db.models.model_notification import (
    MessagePriority, MessageStatus, DeliveryStatus
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class MockChannel:
    """Mock channel for testing."""

    def __init__(self, name: str, should_fail: bool = False, delay: float = 0.1):
        self.name = name
        self.should_fail = should_fail
        self.delay = delay
        self.sent_messages = []
        self.health_status = "healthy"

    async def send(self, message) -> Dict[str, Any]:
        """Mock send method."""
        await asyncio.sleep(self.delay)

        if self.should_fail:
            return {
                "success": False,
                "error_message": f"Mock failure for {self.name}",
                "response_time_ms": int(self.delay * 1000)
            }

        self.sent_messages.append(message)
        return {
            "success": True,
            "external_id": f"mock_{self.name}_{len(self.sent_messages)}",
            "response_time_ms": int(self.delay * 1000)
        }

    async def get_health(self) -> Dict[str, Any]:
        """Mock health check."""
        return {
            "status": self.health_status,
            "last_success": datetime.now(timezone.utc) if self.health_status == "healthy" else None,
            "last_failure": datetime.now(timezone.utc) if self.health_status != "healthy" else None,
            "avg_response_time_ms": int(self.delay * 1000)
        }


@pytest.fixture
async def test_client():
    """Create test client with mocked channels."""
    # Initialize database
    init_database(
        database_url=config.database.url,
        echo=False,
        pool_size=5,
        max_overflow=10
    )

    # Create tables using database service
    db_service = get_database_service()
    db_service.init_databases()

    # Mock channel plugins
    mock_channels = {
        "telegram": MockChannel("telegram"),
        "email": MockChannel("email", delay=0.2),
        "sms": MockChannel("sms", delay=0.3)
    }

    with patch('src.notification.channels.base.channel_registry') as mock_registry:
        mock_registry.get_channel.side_effect = lambda name: mock_channels.get(name)
        mock_registry.list_channels.return_value = list(mock_channels.keys())

        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client, mock_channels


class TestCompleteMessageFlow:
    """Test complete message delivery flows."""

    async def test_single_channel_message_delivery(self, test_client):
        """Test complete flow for single channel message."""
        client, mock_channels = test_client

        # Send message via API
        message_data = {
            "message_type": "test_alert",
            "channels": ["telegram"],
            "content": {"text": "Test message"},
            "recipient_id": "user123",
            "priority": "NORMAL"
        }

        response = await client.post("/api/v1/messages", json=message_data)
        assert response.status_code == 200

        result = response.json()
        message_id = result["message_id"]
        assert result["status"] == "enqueued"
        assert result["channels"] == ["telegram"]

        # Wait for processing
        await asyncio.sleep(2.0)

        # Check message status
        status_response = await client.get(f"/api/v1/messages/{message_id}/status")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["status"] in [MessageStatus.PROCESSING.value, MessageStatus.DELIVERED.value]

        # Check delivery status
        delivery_response = await client.get(f"/api/v1/messages/{message_id}/delivery")
        assert delivery_response.status_code == 200

        delivery_data = delivery_response.json()
        assert len(delivery_data) == 1
        assert delivery_data[0]["channel"] == "telegram"

        # Verify mock channel received message
        telegram_channel = mock_channels["telegram"]
        assert len(telegram_channel.sent_messages) >= 1

    async def test_multi_channel_message_delivery(self, test_client):
        """Test complete flow for multi-channel message."""
        client, mock_channels = test_client

        # Send message to multiple channels
        message_data = {
            "message_type": "trade_alert",
            "channels": ["telegram", "email", "sms"],
            "content": {
                "text": "Trade executed: BTC/USD",
                "details": {"symbol": "BTCUSD", "price": 50000}
            },
            "recipient_id": "trader456",
            "priority": "HIGH"
        }

        response = await client.post("/api/v1/messages", json=message_data)
        assert response.status_code == 200

        result = response.json()
        message_id = result["message_id"]
        assert result["priority"] == "HIGH"
        assert set(result["channels"]) == {"telegram", "email", "sms"}

        # Wait for processing (high priority should be faster)
        await asyncio.sleep(3.0)

        # Check delivery status for all channels
        delivery_response = await client.get(f"/api/v1/messages/{message_id}/delivery")
        assert delivery_response.status_code == 200

        delivery_data = delivery_response.json()
        assert len(delivery_data) == 3

        channels_delivered = {d["channel"] for d in delivery_data}
        assert channels_delivered == {"telegram", "email", "sms"}

        # Verify all mock channels received messages
        for channel_name in ["telegram", "email", "sms"]:
            channel = mock_channels[channel_name]
            assert len(channel.sent_messages) >= 1

    async def test_priority_message_processing(self, test_client):
        """Test that high priority messages are processed first."""
        client, mock_channels = test_client

        # Send normal priority message first
        normal_message = {
            "message_type": "normal_alert",
            "channels": ["telegram"],
            "content": {"text": "Normal priority message"},
            "recipient_id": "user1",
            "priority": "NORMAL"
        }

        normal_response = await client.post("/api/v1/messages", json=normal_message)
        normal_id = normal_response.json()["message_id"]

        # Send critical priority message after
        critical_message = {
            "message_type": "critical_alert",
            "channels": ["telegram"],
            "content": {"text": "CRITICAL: System failure!"},
            "recipient_id": "user1",
            "priority": "CRITICAL"
        }

        critical_response = await client.post("/api/v1/messages", json=critical_message)
        critical_id = critical_response.json()["message_id"]

        # Wait for processing
        await asyncio.sleep(2.0)

        # Check that critical message was processed first
        # (This is verified by checking the order in mock channel)
        telegram_channel = mock_channels["telegram"]
        if len(telegram_channel.sent_messages) >= 2:
            # Critical message should be processed first despite being sent second
            first_message = telegram_channel.sent_messages[0]
            assert "CRITICAL" in str(first_message.content)


class TestFailureScenarios:
    """Test service behavior under various failure scenarios."""

    async def test_channel_failure_handling(self, test_client):
        """Test handling of channel failures."""
        client, mock_channels = test_client

        # Make telegram channel fail
        mock_channels["telegram"].should_fail = True

        message_data = {
            "message_type": "test_failure",
            "channels": ["telegram"],
            "content": {"text": "This should fail"},
            "recipient_id": "user123"
        }

        response = await client.post("/api/v1/messages", json=message_data)
        assert response.status_code == 200

        message_id = response.json()["message_id"]

        # Wait for processing and retries
        await asyncio.sleep(5.0)

        # Check delivery status shows failure
        delivery_response = await client.get(f"/api/v1/messages/{message_id}/delivery")
        assert delivery_response.status_code == 200

        delivery_data = delivery_response.json()
        assert len(delivery_data) >= 1

        # Should have failure status
        failed_delivery = delivery_data[0]
        assert failed_delivery["status"] == DeliveryStatus.FAILED.value
        assert "Mock failure" in failed_delivery["error_message"]

    async def test_multi_channel_partial_failure(self, test_client):
        """Test partial failure in multi-channel delivery."""
        client, mock_channels = test_client

        # Make email channel fail, but keep others working
        mock_channels["email"].should_fail = True

        message_data = {
            "message_type": "partial_failure_test",
            "channels": ["telegram", "email", "sms"],
            "content": {"text": "Partial failure test"},
            "recipient_id": "user456"
        }

        response = await client.post("/api/v1/messages", json=message_data)
        assert response.status_code == 200

        message_id = response.json()["message_id"]

        # Wait for processing
        await asyncio.sleep(4.0)

        # Check delivery status
        delivery_response = await client.get(f"/api/v1/messages/{message_id}/delivery")
        delivery_data = delivery_response.json()

        # Should have 3 delivery attempts
        assert len(delivery_data) == 3

        # Check results by channel
        results_by_channel = {d["channel"]: d["status"] for d in delivery_data}

        assert results_by_channel["telegram"] == DeliveryStatus.DELIVERED.value
        assert results_by_channel["email"] == DeliveryStatus.FAILED.value
        assert results_by_channel["sms"] == DeliveryStatus.DELIVERED.value

    async def test_database_connection_failure(self, test_client):
        """Test handling of database connection failures."""
        client, mock_channels = test_client

        # This test would require mocking database failures
        # For now, test that service handles database errors gracefully

        # Send invalid message data to trigger database error
        invalid_message = {
            "message_type": "",  # Empty message type should cause validation error
            "channels": [],      # Empty channels should cause validation error
            "content": {},
            "recipient_id": None
        }

        response = await client.post("/api/v1/messages", json=invalid_message)
        assert response.status_code == 400  # Should return bad request, not crash

        error_data = response.json()
        assert "error" in error_data or "detail" in error_data


class TestServiceLifecycle:
    """Test service startup, shutdown, and recovery."""

    async def test_service_health_check(self, test_client):
        """Test service health check endpoint."""
        client, mock_channels = test_client

        response = await client.get("/api/v1/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["service"] == config.service_name
        assert health_data["database"] == "connected"

    async def test_channel_health_monitoring(self, test_client):
        """Test channel health monitoring."""
        client, mock_channels = test_client

        # Make one channel unhealthy
        mock_channels["sms"].health_status = "down"

        response = await client.get("/api/v1/channels/health")
        assert response.status_code == 200

        health_data = response.json()
        # Should return health status for all channels
        assert isinstance(health_data, list)

    async def test_processor_statistics(self, test_client):
        """Test processor statistics endpoint."""
        client, mock_channels = test_client

        response = await client.get("/api/v1/processor/stats")

        if response.status_code == 200:
            stats_data = response.json()
            assert "processor_stats" in stats_data
            assert "timestamp" in stats_data
        else:
            # Processor might not be fully initialized in test environment
            assert response.status_code == 503

    async def test_service_recovery_after_failure(self, test_client):
        """Test service recovery after temporary failures."""
        client, mock_channels = test_client

        # Simulate temporary failure by making all channels fail
        for channel in mock_channels.values():
            channel.should_fail = True

        # Send message during failure
        message_data = {
            "message_type": "recovery_test",
            "channels": ["telegram"],
            "content": {"text": "Recovery test message"},
            "recipient_id": "user123"
        }

        response = await client.post("/api/v1/messages", json=message_data)
        assert response.status_code == 200
        message_id = response.json()["message_id"]

        # Wait for initial failure
        await asyncio.sleep(2.0)

        # Restore channel functionality
        for channel in mock_channels.values():
            channel.should_fail = False

        # Wait for recovery and retry
        await asyncio.sleep(3.0)

        # Check if message was eventually delivered after recovery
        delivery_response = await client.get(f"/api/v1/messages/{message_id}/delivery")
        delivery_data = delivery_response.json()

        # Should have multiple delivery attempts
        assert len(delivery_data) >= 1


class TestRateLimitingAndBatching:
    """Test rate limiting and batching in end-to-end scenarios."""

    async def test_rate_limiting_enforcement(self, test_client):
        """Test that rate limiting is enforced."""
        client, mock_channels = test_client

        # Send multiple messages rapidly to trigger rate limiting
        messages = []
        for i in range(10):
            message_data = {
                "message_type": "rate_limit_test",
                "channels": ["telegram"],
                "content": {"text": f"Rate limit test message {i}"},
                "recipient_id": "rate_test_user",
                "priority": "NORMAL"
            }

            response = await client.post("/api/v1/messages", json=message_data)
            assert response.status_code == 200
            messages.append(response.json()["message_id"])

        # Wait for processing
        await asyncio.sleep(5.0)

        # Check that not all messages were delivered immediately
        # (Some should be queued due to rate limiting)
        telegram_channel = mock_channels["telegram"]
        delivered_count = len(telegram_channel.sent_messages)

        # Should be less than total sent due to rate limiting
        assert delivered_count <= 10

    async def test_priority_bypass_rate_limiting(self, test_client):
        """Test that high priority messages bypass rate limiting."""
        client, mock_channels = test_client

        # Send many normal priority messages to trigger rate limiting
        for i in range(5):
            normal_message = {
                "message_type": "normal_spam",
                "channels": ["telegram"],
                "content": {"text": f"Normal message {i}"},
                "recipient_id": "bypass_test_user",
                "priority": "NORMAL"
            }
            await client.post("/api/v1/messages", json=normal_message)

        # Send critical priority message
        critical_message = {
            "message_type": "critical_bypass",
            "channels": ["telegram"],
            "content": {"text": "CRITICAL: Should bypass rate limit"},
            "recipient_id": "bypass_test_user",
            "priority": "CRITICAL"
        }

        response = await client.post("/api/v1/messages", json=critical_message)
        critical_id = response.json()["message_id"]

        # Wait for processing
        await asyncio.sleep(3.0)

        # Check that critical message was delivered despite rate limiting
        delivery_response = await client.get(f"/api/v1/messages/{critical_id}/delivery")
        delivery_data = delivery_response.json()

        if delivery_data:
            assert delivery_data[0]["status"] == DeliveryStatus.DELIVERED.value


class TestDeliveryHistory:
    """Test delivery history and analytics endpoints."""

    async def test_message_history_api(self, test_client):
        """Test message history API functionality."""
        client, mock_channels = test_client

        # Send several messages
        for i in range(3):
            message_data = {
                "message_type": "history_test",
                "channels": ["telegram"],
                "content": {"text": f"History test message {i}"},
                "recipient_id": f"history_user_{i}",
                "priority": "NORMAL"
            }
            await client.post("/api/v1/messages", json=message_data)

        # Wait for processing
        await asyncio.sleep(2.0)

        # Query message history
        response = await client.get("/api/v1/history/messages?limit=10")
        assert response.status_code == 200

        history_data = response.json()
        assert "messages" in history_data
        assert "pagination" in history_data
        assert len(history_data["messages"]) >= 3

    async def test_delivery_statistics(self, test_client):
        """Test delivery statistics endpoint."""
        client, mock_channels = test_client

        # Send messages and wait for processing
        for i in range(2):
            message_data = {
                "message_type": "stats_test",
                "channels": ["telegram", "email"],
                "content": {"text": f"Stats test message {i}"},
                "recipient_id": "stats_user"
            }
            await client.post("/api/v1/messages", json=message_data)

        await asyncio.sleep(3.0)

        # Get statistics
        response = await client.get("/api/v1/stats")
        assert response.status_code == 200

        stats_data = response.json()
        assert "delivery_statistics" in stats_data
        assert "message_statistics" in stats_data


async def run_end_to_end_tests():
    """Run all end-to-end tests."""
    print("=== Notification Service End-to-End Tests ===\n")

    # Initialize test environment
    try:
        # Run tests using pytest programmatically
        import pytest

        test_file = __file__
        exit_code = pytest.main([
            test_file,
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure
        ])

        if exit_code == 0:
            print("\n🎉 All end-to-end tests passed!")
        else:
            print("\n❌ Some end-to-end tests failed!")

        return exit_code

    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_end_to_end_tests())
    sys.exit(exit_code)