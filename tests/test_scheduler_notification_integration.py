"""
Tests for Scheduler Service Notification Integration

Tests the notification integration functionality including:
- Notification service client integration with mock responses
- Notification formatting with various alert types
- Error handling and retry scenarios
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.scheduler.scheduler_service import SchedulerService
from src.data.db.services.jobs_service import JobsService
from src.common.alerts.alert_evaluator import AlertEvaluator
from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority


@pytest.fixture
def mock_notification_client():
    """Create a mock notification service client."""
    client = Mock(spec=NotificationServiceClient)
    client.send_notification = AsyncMock(return_value=True)
    client.get_health_status = AsyncMock(return_value={"status": "healthy"})
    client.get_stats = Mock(return_value={
        "service_url": "http://localhost:8000",
        "timeout": 30,
        "max_retries": 3,
        "session_active": True
    })
    return client


@pytest.fixture
def mock_jobs_service():
    """Create a mock jobs service."""
    service = Mock(spec=JobsService)
    return service


@pytest.fixture
def mock_alert_evaluator():
    """Create a mock alert evaluator."""
    evaluator = Mock(spec=AlertEvaluator)
    return evaluator


@pytest.fixture
def scheduler_service_with_notification(mock_jobs_service, mock_alert_evaluator, mock_notification_client):
    """Create a scheduler service with mocked notification client."""
    return SchedulerService(
        jobs_service=mock_jobs_service,
        alert_evaluator=mock_alert_evaluator,
        notification_client=mock_notification_client,
        database_url="sqlite:///:memory:",
        max_workers=2
    )


class TestNotificationServiceIntegration:
    """Test notification service client integration."""

    @pytest.mark.asyncio
    async def test_send_notification_success(self, scheduler_service_with_notification, mock_notification_client):
        """Test successful notification sending."""
        # Setup notification data
        notification_data = {
            "ticker": "BTCUSDT",
            "timeframe": "1h",
            "price": 45000.0,
            "timestamp": "2024-01-01T12:00:00Z",
            "alert_name": "RSI Oversold Alert",
            "notify_config": {
                "channels": ["telegram"],
                "recipient_id": "user123",
                "priority": "high"
            },
            "indicators": {
                "RSI": 25.5,
                "SMA_20": 44800.0
            },
            "rule_snapshot": {
                "RSI_threshold": 30.0,
                "current_RSI": 25.5
            }
        }

        # Execute notification
        result = await scheduler_service_with_notification._send_notification(notification_data)

        # Verify success
        assert result is True

        # Verify notification client was called correctly
        mock_notification_client.send_notification.assert_called_once()
        call_args = mock_notification_client.send_notification.call_args

        # Check call arguments
        assert call_args.kwargs["notification_type"] == MessageType.ALERT
        assert "RSI Oversold Alert: BTCUSDT (1h)" in call_args.kwargs["title"]
        assert call_args.kwargs["priority"] == MessagePriority.HIGH
        assert call_args.kwargs["channels"] == ["telegram", "email"]  # Email added for high priority
        assert call_args.kwargs["recipient_id"] == "user123"
        assert call_args.kwargs["source"] == "scheduler_service"

        # Check message content
        message = call_args.kwargs["message"]
        assert "BTCUSDT" in message
        assert "45000.0" in message
        assert "RSI: 25.5" in message
        assert "RSI_threshold: 30.0" in message

    @pytest.mark.asyncio
    async def test_send_notification_with_market_data(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification with comprehensive market data."""
        notification_data = {
            "ticker": "ETHUSDT",
            "timeframe": "4h",
            "price": 3200.0,
            "timestamp": "2024-01-01T16:00:00Z",
            "alert_name": "Bollinger Band Breakout",
            "notify_config": {
                "channels": ["telegram", "email"],
                "recipient_id": "trader456",
                "priority": "critical"
            },
            "market_data": {
                "open": 3150.0,
                "high": 3220.0,
                "low": 3140.0,
                "volume": 125000
            },
            "indicators": {
                "BB_Upper": 3180.0,
                "BB_Middle": 3160.0,
                "BB_Lower": 3140.0,
                "ATR": 45.5
            },
            "rule_description": "Price breaks above upper Bollinger Band",
            "rearm_info": {
                "type": "crossing",
                "value": "above_3180"
            },
            "alert_config": {
                "description": "Bollinger Band breakout strategy",
                "priority": "critical"
            }
        }

        # Execute notification
        result = await scheduler_service_with_notification._send_notification(notification_data)

        # Verify success
        assert result is True

        # Verify enhanced formatting
        call_args = mock_notification_client.send_notification.call_args
        message = call_args.kwargs["message"]

        # Check market context
        assert "Market Context" in message
        assert "Open: $3150.0" in message
        assert "High: $3220.0" in message
        assert "📈 +1.59%" in message  # Price change calculation

        # Check indicator grouping
        assert "Technical Indicators" in message
        assert "BB_Upper: 3180.0" in message
        assert "ATR: 45.5" in message

        # Check rule details
        assert "Alert Rule Details" in message
        assert "Price breaks above upper Bollinger Band" in message

        # Check rearm status
        assert "Rearm Status" in message
        assert "Type: crossing" in message

        # Check alert configuration
        assert "Alert Configuration" in message
        assert "Bollinger Band breakout strategy" in message

        # Check priority escalation (should add email for critical alerts)
        assert call_args.kwargs["priority"] == MessagePriority.CRITICAL
        assert "email" in call_args.kwargs["channels"]

    @pytest.mark.asyncio
    async def test_send_notification_client_failure_with_retry(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification retry logic when client fails."""
        # Setup client to fail first two attempts, succeed on third
        mock_notification_client.send_notification.side_effect = [
            Exception("Connection timeout"),
            False,  # Service returns failure
            True    # Success on third attempt
        ]

        notification_data = {
            "ticker": "ADAUSDT",
            "timeframe": "15m",
            "price": 0.45,
            "alert_name": "MACD Signal",
            "notify_config": {"channels": ["telegram"], "recipient_id": "user789"}
        }

        # Execute notification
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up test
            result = await scheduler_service_with_notification._send_notification(notification_data)

        # Verify eventual success
        assert result is True

        # Verify retry attempts
        assert mock_notification_client.send_notification.call_count == 3

    @pytest.mark.asyncio
    async def test_send_notification_all_retries_fail(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification when all retry attempts fail."""
        # Setup client to always fail
        mock_notification_client.send_notification.side_effect = Exception("Service unavailable")

        notification_data = {
            "ticker": "DOGEUSDT",
            "timeframe": "1m",
            "price": 0.08,
            "alert_name": "Volume Spike",
            "notify_config": {"channels": ["telegram"], "recipient_id": "user999"}
        }

        # Execute notification
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Speed up test
            result = await scheduler_service_with_notification._send_notification(notification_data)

        # Verify failure
        assert result is False

        # Verify all retry attempts were made
        assert mock_notification_client.send_notification.call_count == 3

    @pytest.mark.asyncio
    async def test_notification_formatting_with_minimal_data(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification formatting with minimal data."""
        notification_data = {
            "ticker": "LTCUSDT",
            "timeframe": "1d",
            "price": 95.0,
            "notify_config": {}  # Empty config should use defaults
        }

        # Execute notification
        result = await scheduler_service_with_notification._send_notification(notification_data)

        # Verify success with defaults
        assert result is True

        call_args = mock_notification_client.send_notification.call_args

        # Check defaults were applied
        assert call_args.kwargs["channels"] == ["telegram"]
        assert call_args.kwargs["recipient_id"] == "default"
        assert call_args.kwargs["priority"] == MessagePriority.NORMAL

        # Check message contains basic info
        message = call_args.kwargs["message"]
        assert "LTCUSDT" in message
        assert "95.0" in message

    @pytest.mark.asyncio
    async def test_notification_channel_configuration(self, scheduler_service_with_notification, mock_notification_client):
        """Test different channel configurations."""
        test_cases = [
            # Single channel as string
            {"channels": "email", "expected": ["email"]},
            # Multiple channels as list
            {"channels": ["telegram", "email"], "expected": ["telegram", "email"]},
            # High priority should add email if not present
            {"channels": ["telegram"], "priority": "high", "expected": ["telegram", "email"]},
            # Critical priority should add email if not present
            {"channels": ["slack"], "priority": "critical", "expected": ["slack", "email"]},
        ]

        for i, case in enumerate(test_cases):
            mock_notification_client.reset_mock()

            notification_data = {
                "ticker": f"TEST{i}USDT",
                "timeframe": "1h",
                "price": 100.0,
                "alert_name": f"Test Alert {i}",
                "notify_config": {
                    "channels": case["channels"],
                    "priority": case.get("priority", "normal"),
                    "recipient_id": f"user{i}"
                }
            }

            # Execute notification
            result = await scheduler_service_with_notification._send_notification(notification_data)
            assert result is True

            # Verify channel configuration
            call_args = mock_notification_client.send_notification.call_args
            assert call_args.kwargs["channels"] == case["expected"]

    @pytest.mark.asyncio
    async def test_notification_priority_mapping(self, scheduler_service_with_notification, mock_notification_client):
        """Test priority mapping from alert config to message priority."""
        priority_mappings = [
            ("low", MessagePriority.LOW),
            ("normal", MessagePriority.NORMAL),
            ("high", MessagePriority.HIGH),
            ("critical", MessagePriority.CRITICAL),
            ("unknown", MessagePriority.NORMAL)  # Default fallback
        ]

        for alert_priority, expected_message_priority in priority_mappings:
            mock_notification_client.reset_mock()

            notification_data = {
                "ticker": "TESTUSDT",
                "timeframe": "1h",
                "price": 100.0,
                "alert_name": "Priority Test",
                "notify_config": {
                    "priority": alert_priority,
                    "recipient_id": "test_user"
                }
            }

            # Execute notification
            result = await scheduler_service_with_notification._send_notification(notification_data)
            assert result is True

            # Verify priority mapping
            call_args = mock_notification_client.send_notification.call_args
            assert call_args.kwargs["priority"] == expected_message_priority

    @pytest.mark.asyncio
    async def test_notification_data_types_handling(self, scheduler_service_with_notification, mock_notification_client):
        """Test handling of different data types in indicators and rule snapshots."""
        notification_data = {
            "ticker": "TESTUSDT",
            "timeframe": "1h",
            "price": 100.0,
            "alert_name": "Data Types Test",
            "notify_config": {"recipient_id": "test_user"},
            "indicators": {
                "float_value": 123.456,
                "int_value": 42,
                "string_value": "BULLISH",
                "bool_value": True,
                "none_value": None
            },
            "rule_snapshot": {
                "threshold": 50.0,
                "signal": "BUY",
                "confirmed": True
            }
        }

        # Execute notification
        result = await scheduler_service_with_notification._send_notification(notification_data)
        assert result is True

        # Verify message formatting handles different data types
        call_args = mock_notification_client.send_notification.call_args
        message = call_args.kwargs["message"]

        # Check numeric formatting
        assert "float_value: 123.4560" in message
        assert "int_value: 42.0000" in message

        # Check non-numeric values
        assert "string_value: BULLISH" in message
        assert "bool_value: True" in message

    def test_check_notification_health_success(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification health check when service is healthy."""
        # Execute health check
        result = asyncio.run(scheduler_service_with_notification.check_notification_health())

        # Verify health status
        assert result["status"] == "healthy"
        mock_notification_client.get_health_status.assert_called_once()

    def test_check_notification_health_no_client(self, mock_jobs_service, mock_alert_evaluator):
        """Test notification health check when no client is configured."""
        # Create scheduler without notification client
        scheduler = SchedulerService(
            jobs_service=mock_jobs_service,
            alert_evaluator=mock_alert_evaluator,
            notification_client=None,
            database_url="sqlite:///:memory:"
        )

        # Execute health check
        result = asyncio.run(scheduler.check_notification_health())

        # Verify error response
        assert result["status"] == "unavailable"
        assert "No notification client configured" in result["error"]

    def test_check_notification_health_client_error(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification health check when client throws error."""
        # Setup client to throw error
        mock_notification_client.get_health_status.side_effect = Exception("Connection failed")

        # Execute health check
        result = asyncio.run(scheduler_service_with_notification.check_notification_health())

        # Verify error handling
        assert result["status"] == "unhealthy"
        assert "Connection failed" in result["error"]

    def test_scheduler_status_includes_notification_stats(self, scheduler_service_with_notification):
        """Test that scheduler status includes notification client statistics."""
        # Get scheduler status
        status = scheduler_service_with_notification.get_scheduler_status()

        # Verify notification client stats are included
        assert "notification_client" in status
        assert status["notification_client"]["service_url"] == "http://localhost:8000"
        assert status["notification_client"]["max_retries"] == 3


class TestNotificationErrorHandling:
    """Test error handling scenarios for notifications."""

    @pytest.mark.asyncio
    async def test_notification_with_malformed_data(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification handling with malformed data."""
        # Test with missing required fields
        notification_data = {}

        # Execute notification (should not crash)
        result = await scheduler_service_with_notification._send_notification(notification_data)

        # Should still attempt to send with defaults
        assert result is True
        mock_notification_client.send_notification.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_with_invalid_price_data(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification with invalid price data."""
        notification_data = {
            "ticker": "TESTUSDT",
            "timeframe": "1h",
            "price": "invalid_price",  # Invalid price type
            "alert_name": "Invalid Price Test",
            "notify_config": {"recipient_id": "test_user"}
        }

        # Execute notification (should handle gracefully)
        result = await scheduler_service_with_notification._send_notification(notification_data)

        # Should still succeed with error handling
        assert result is True

    @pytest.mark.asyncio
    async def test_notification_unexpected_exception(self, scheduler_service_with_notification, mock_notification_client):
        """Test notification handling when unexpected exception occurs."""
        # Setup client to throw unexpected exception during message formatting
        def side_effect(*args, **kwargs):
            raise RuntimeError("Unexpected error during notification")

        mock_notification_client.send_notification.side_effect = side_effect

        notification_data = {
            "ticker": "TESTUSDT",
            "timeframe": "1h",
            "price": 100.0,
            "alert_name": "Exception Test",
            "notify_config": {"recipient_id": "test_user"}
        }

        # Execute notification (should not crash the service)
        result = await scheduler_service_with_notification._send_notification(notification_data)

        # Should return False but not raise exception
        assert result is False