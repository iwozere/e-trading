"""
Tests for NotificationServiceClient

Tests the client functionality for interacting with the notification service.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.client import (
    NotificationServiceClient, MessageType, MessagePriority,
    get_notification_client, initialize_notification_client
)


class TestNotificationServiceClient:
    """Test cases for NotificationServiceClient."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return NotificationServiceClient(
            service_url="http://localhost:8000",
            timeout=10,
            max_retries=2
        )

    @pytest.mark.asyncio
    async def test_client_initialization(self, client):
        """Test client initialization."""
        assert client.service_url == "http://localhost:8000"
        assert client.timeout.total == 10
        assert client.max_retries == 2
        assert client._session is None

    @pytest.mark.asyncio
    async def test_send_notification_success(self, client):
        """Test successful notification sending."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message_id": 123,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "normal"
        })

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            success = await client.send_notification(
                notification_type=MessageType.ALERT,
                title="Test Alert",
                message="This is a test alert",
                priority=MessagePriority.HIGH,
                channels=["telegram"],
                recipient_id="test_user"
            )

            assert success is True
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_failure(self, client):
        """Test notification sending failure."""
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            success = await client.send_notification(
                notification_type=MessageType.ERROR,
                title="Test Error",
                message="This is a test error",
                priority=MessagePriority.CRITICAL
            )

            assert success is False

    @pytest.mark.asyncio
    async def test_send_trade_notification(self, client):
        """Test trade notification sending."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message_id": 124,
            "status": "enqueued"
        })

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            success = await client.send_trade_notification(
                symbol="BTCUSDT",
                side="BUY",
                price=50000.0,
                quantity=0.1,
                recipient_id="trader_1"
            )

            assert success is True

    @pytest.mark.asyncio
    async def test_send_error_notification(self, client):
        """Test error notification sending."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "message_id": 125,
            "status": "enqueued"
        })

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            success = await client.send_error_notification(
                error_message="Database connection failed",
                source="trading_service",
                recipient_id="admin"
            )

            assert success is True

    @pytest.mark.asyncio
    async def test_get_message_status(self, client):
        """Test getting message status."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "id": 123,
            "status": "delivered",
            "created_at": "2024-01-01T00:00:00Z"
        })

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            status = await client.get_message_status(123)

            assert status is not None
            assert status["id"] == 123
            assert status["status"] == "delivered"

    @pytest.mark.asyncio
    async def test_get_message_status_not_found(self, client):
        """Test getting status for non-existent message."""
        mock_response = MagicMock()
        mock_response.status = 404

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            status = await client.get_message_status(999)

            assert status is None

    @pytest.mark.asyncio
    async def test_get_health_status(self, client):
        """Test getting service health status."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "status": "healthy",
            "service": "notification-service",
            "version": "1.0.0"
        })

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.return_value.__aenter__.return_value = mock_response

            health = await client.get_health_status()

            assert health["status"] == "healthy"
            assert health["service"] == "notification-service"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client as context manager."""
        async with NotificationServiceClient() as client:
            assert client is not None
            # Session should be created when needed
            session = await client._get_session()
            assert session is not None

        # Session should be closed after context exit
        assert client._session.closed

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, client):
        """Test retry mechanism on failures."""
        # First two calls fail, third succeeds
        mock_responses = [
            Exception("Connection error"),
            Exception("Timeout error"),
            MagicMock(status=200, json=AsyncMock(return_value={"message_id": 123}))
        ]

        with patch('aiohttp.ClientSession.request') as mock_request:
            mock_request.side_effect = mock_responses

            success = await client.send_notification(
                notification_type=MessageType.INFO,
                title="Test",
                message="Test message"
            )

            assert success is True
            assert mock_request.call_count == 3

    @pytest.mark.asyncio
    async def test_global_client_management(self):
        """Test global client initialization and access."""
        # Initially no client
        assert get_notification_client() is None

        # Initialize global client
        client = await initialize_notification_client("http://test:8000")
        assert client is not None
        assert get_notification_client() is client

        # Reinitialize should close previous client
        new_client = await initialize_notification_client("http://test2:8000")
        assert new_client is not client
        assert get_notification_client() is new_client

    def test_compatibility_methods(self, client):
        """Test AsyncNotificationManager compatibility methods."""
        # These should not raise exceptions
        asyncio.run(client.start())
        asyncio.run(client.stop())

        stats = client.get_stats()
        assert "service_url" in stats
        assert "timeout" in stats
        assert "max_retries" in stats