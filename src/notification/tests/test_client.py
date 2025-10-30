"""
Unit tests for the notification service client.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

from src.notification.service.client import (
    NotificationServiceClient,
    NotificationRequest,
    NotificationResponse,
    NotificationServiceError,
    NotificationServiceUnavailableError,
    CircuitBreaker,
    CircuitBreakerState
)
from src.notification.model import NotificationType, NotificationPriority


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.can_execute() is True

    def test_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures
        for i in range(2):
            cb.record_failure()
            assert cb.state == CircuitBreakerState.CLOSED
            assert cb.can_execute() is True

        # Third failure should open circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.can_execute() is False

    def test_recovery_timeout(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Should still be open immediately
        assert cb.can_execute() is False

        # Mock time passage
        import time
        original_time = time.time
        time.time = Mock(return_value=original_time() + 2)

        try:
            # Should transition to half-open
            assert cb.can_execute() is True
            assert cb.state == CircuitBreakerState.HALF_OPEN
        finally:
            time.time = original_time

    def test_half_open_success(self):
        """Test circuit breaker closes after successful half-open calls."""
        cb = CircuitBreaker(failure_threshold=2, half_open_max_calls=2)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        cb.state = CircuitBreakerState.HALF_OPEN
        cb.half_open_calls = 0

        # First success
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Second success should close circuit
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0


class TestNotificationRequest:
    """Test notification request data structure."""

    def test_basic_request(self):
        """Test basic notification request."""
        request = NotificationRequest(
            message_type="test",
            priority="HIGH",
            channels=["telegram"],
            content={"text": "Hello"}
        )

        data = request.to_dict()
        assert data["message_type"] == "test"
        assert data["priority"] == "HIGH"
        assert data["channels"] == ["telegram"]
        assert data["content"] == {"text": "Hello"}

    def test_optional_fields(self):
        """Test notification request with optional fields."""
        scheduled_for = datetime.now()
        request = NotificationRequest(
            message_type="test",
            recipient_id="user123",
            template_name="alert_template",
            scheduled_for=scheduled_for
        )

        data = request.to_dict()
        assert data["recipient_id"] == "user123"
        assert data["template_name"] == "alert_template"
        assert data["scheduled_for"] == scheduled_for.isoformat()


class TestNotificationServiceClient:
    """Test notification service client."""

    def setup_method(self):
        """Setup test client."""
        self.client = NotificationServiceClient(
            service_url="http://test-service:8000",
            timeout=10,
            circuit_breaker_enabled=False  # Disable for testing
        )

    def teardown_method(self):
        """Cleanup test client."""
        self.client.close()

    @patch('requests.Session.post')
    def test_send_notification_success(self, mock_post):
        """Test successful notification sending."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message_id": 123,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "NORMAL"
        }
        mock_post.return_value = mock_response

        # Send notification
        response = self.client.send_notification(
            notification_type=NotificationType.INFO,
            title="Test Alert",
            message="This is a test message",
            channels=["telegram"]
        )

        # Verify response
        assert isinstance(response, NotificationResponse)
        assert response.message_id == 123
        assert response.status == "enqueued"
        assert response.channels == ["telegram"]
        assert response.priority == "NORMAL"

        # Verify request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]["json"]["message_type"] == "info"
        assert call_args[1]["json"]["content"]["text"] == "This is a test message"
        assert call_args[1]["json"]["content"]["subject"] == "Test Alert"

    @patch('requests.Session.post')
    def test_send_notification_with_attachments(self, mock_post):
        """Test notification with attachments."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message_id": 124,
            "status": "enqueued",
            "channels": ["email"],
            "priority": "HIGH"
        }
        mock_post.return_value = mock_response

        # Send notification with attachments
        attachments = {"chart.png": b"fake_image_data"}
        response = self.client.send_notification(
            notification_type=NotificationType.TRADE_ENTRY,
            title="Trade Alert",
            message="New trade executed",
            priority=NotificationPriority.HIGH,
            channels=["email"],
            attachments=attachments
        )

        # Verify response
        assert response.message_id == 124
        assert response.priority == "HIGH"

        # Verify request includes attachments
        call_args = mock_post.call_args
        assert call_args[1]["json"]["content"]["attachments"] == attachments

    @patch('requests.Session.post')
    def test_send_notification_backward_compatibility(self, mock_post):
        """Test backward compatibility with old parameters."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message_id": 125,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "NORMAL"
        }
        mock_post.return_value = mock_response

        # Send notification with old-style parameters
        response = self.client.send_notification(
            notification_type="info",
            title="Test",
            message="Test message",
            telegram_chat_id=12345,
            reply_to_message_id=67890,
            email_receiver="test@example.com"
        )

        # Verify response
        assert response.message_id == 125

        # Verify backward compatibility parameters are handled
        call_args = mock_post.call_args
        metadata = call_args[1]["json"]["metadata"]
        assert metadata["telegram_chat_id"] == 12345
        assert metadata["reply_to_message_id"] == 67890
        assert call_args[1]["json"]["recipient_id"] == "test@example.com"

    @patch('requests.Session.post')
    def test_send_trade_notification(self, mock_post):
        """Test trade notification sending."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message_id": 126,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "HIGH"
        }
        mock_post.return_value = mock_response

        # Send trade notification
        response = self.client.send_trade_notification(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            pnl=5.5
        )

        # Verify response
        assert response.message_id == 126
        assert response.priority == "HIGH"

        # Verify request content
        call_args = mock_post.call_args
        assert call_args[1]["json"]["message_type"] == "trade_entry"
        assert "Buy 0.1 BTCUSDT at 50000.0" in call_args[1]["json"]["content"]["text"]

        metadata = call_args[1]["json"]["metadata"]
        assert metadata["symbol"] == "BTCUSDT"
        assert metadata["side"] == "BUY"
        assert metadata["pnl"] == 5.5

    @patch('requests.Session.post')
    def test_send_error_notification(self, mock_post):
        """Test error notification sending."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message_id": 127,
            "status": "enqueued",
            "channels": ["telegram", "email"],
            "priority": "CRITICAL"
        }
        mock_post.return_value = mock_response

        # Send error notification
        response = self.client.send_error_notification(
            error_message="Database connection failed",
            source="trading_engine"
        )

        # Verify response
        assert response.message_id == 127
        assert response.priority == "CRITICAL"

        # Verify request content
        call_args = mock_post.call_args
        assert call_args[1]["json"]["message_type"] == "error"
        assert call_args[1]["json"]["content"]["text"] == "Database connection failed"
        assert call_args[1]["json"]["content"]["subject"] == "Error Alert"

        metadata = call_args[1]["json"]["metadata"]
        assert metadata["source"] == "trading_engine"

    @patch('requests.Session.post')
    def test_send_notification_http_error(self, mock_post):
        """Test notification sending with HTTP error."""
        # Mock HTTP error
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")

        # Send notification should raise error
        with pytest.raises(NotificationServiceError) as exc_info:
            self.client.send_notification(
                notification_type=NotificationType.INFO,
                title="Test",
                message="Test message"
            )

        assert "Failed to send notification" in str(exc_info.value)

    @patch('requests.Session.get')
    def test_get_message_status(self, mock_get):
        """Test getting message status."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": 123,
            "status": "DELIVERED",
            "created_at": "2023-01-01T00:00:00Z"
        }
        mock_get.return_value = mock_response

        # Get message status
        status = self.client.get_message_status(123)

        # Verify response
        assert status["id"] == 123
        assert status["status"] == "DELIVERED"

        # Verify request
        mock_get.assert_called_once_with(
            "http://test-service:8000/api/notifications/123",
            timeout=10
        )

    @patch('requests.Session.get')
    def test_health_check(self, mock_get):
        """Test health check."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "status": "healthy",
            "service": "notification-service",
            "version": "1.0.0"
        }
        mock_get.return_value = mock_response

        # Perform health check
        health = self.client.health_check()

        # Verify response
        assert health["status"] == "healthy"
        assert health["service"] == "notification-service"

        # Verify request
        mock_get.assert_called_once_with(
            "http://test-service:8000/api/notifications/health",
            timeout=10
        )

    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration."""
        # Create client with circuit breaker enabled
        client = NotificationServiceClient(
            service_url="http://test-service:8000",
            circuit_breaker_enabled=True
        )

        try:
            # Open circuit breaker manually
            client.circuit_breaker.state = CircuitBreakerState.OPEN

            # Should raise unavailable error
            with pytest.raises(NotificationServiceUnavailableError):
                client.send_notification(
                    notification_type=NotificationType.INFO,
                    title="Test",
                    message="Test message"
                )
        finally:
            client.close()


class TestAsyncNotificationServiceClient:
    """Test async notification service client methods."""

    def setup_method(self):
        """Setup test client."""
        self.client = NotificationServiceClient(
            service_url="http://test-service:8000",
            timeout=10,
            circuit_breaker_enabled=False
        )

    async def teardown_method(self):
        """Cleanup test client."""
        await self.client.close_async()

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_send_notification_async_success(self, mock_post):
        """Test successful async notification sending."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message_id": 128,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "NORMAL"
        }

        # Mock context manager
        mock_post.return_value.__aenter__.return_value = mock_response
        mock_post.return_value.__aexit__.return_value = None

        # Send notification
        response = await self.client.send_notification_async(
            notification_type=NotificationType.INFO,
            title="Test Alert",
            message="This is a test message",
            channels=["telegram"]
        )

        # Verify response
        assert isinstance(response, NotificationResponse)
        assert response.message_id == 128
        assert response.status == "enqueued"

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_send_trade_notification_async(self, mock_post):
        """Test async trade notification sending."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "message_id": 129,
            "status": "enqueued",
            "channels": ["telegram"],
            "priority": "HIGH"
        }

        mock_post.return_value.__aenter__.return_value = mock_response
        mock_post.return_value.__aexit__.return_value = None

        # Send trade notification
        response = await self.client.send_trade_notification_async(
            symbol="ETHUSDT",
            side="SELL",
            price=3000.0,
            quantity=1.0,
            pnl=-2.1,
            exit_type="SL"
        )

        # Verify response
        assert response.message_id == 129
        assert response.priority == "HIGH"

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_get_message_status_async(self, mock_get):
        """Test async message status retrieval."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": 130,
            "status": "DELIVERED"
        }

        mock_get.return_value.__aenter__.return_value = mock_response
        mock_get.return_value.__aexit__.return_value = None

        # Get message status
        status = await self.client.get_message_status_async(130)

        # Verify response
        assert status["id"] == 130
        assert status["status"] == "DELIVERED"

    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """Test async context manager."""
        async with NotificationServiceClient() as client:
            assert client is not None
            # Client should be properly initialized
            assert client.service_url == "http://localhost:8000"


if __name__ == "__main__":
    pytest.main([__file__])