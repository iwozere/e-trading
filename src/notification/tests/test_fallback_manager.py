"""
Tests for Channel Fallback and Recovery Manager

Unit tests for the fallback manager functionality including
channel fallback routing, dead letter queue, and retry mechanisms.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

from src.notification.service.fallback_manager import (
    FallbackManager, FallbackRule, FallbackStrategy, MessageFailureReason, FailedMessage
)
from src.notification.service.health_monitor import HealthMonitor, HealthStatus
from src.notification.channels.base import (
    NotificationChannel, DeliveryResult, DeliveryStatus, MessageContent, ChannelHealth, ChannelHealthStatus
)


class MockChannel(NotificationChannel):
    """Mock channel for testing."""

    def __init__(self, channel_name: str, config: dict, should_fail: bool = False):
        self.channel_name = channel_name
        self.config = config
        self.should_fail = should_fail
        self.send_calls = []

    def validate_config(self, config: dict) -> None:
        pass

    async def send_message(self, recipient: str, content: MessageContent, message_id: str = None, priority: str = "NORMAL") -> DeliveryResult:
        self.send_calls.append({
            'recipient': recipient,
            'content': content,
            'message_id': message_id,
            'priority': priority
        })

        if self.should_fail:
            # Use BOUNCED status to make it non-retryable
            return DeliveryResult(
                success=False,
                status=DeliveryStatus.BOUNCED,
                error_message=f"Mock failure for {self.channel_name}"
            )
        else:
            return DeliveryResult(
                success=True,
                status=DeliveryStatus.DELIVERED,
                external_id=f"mock_{message_id}_{self.channel_name}",
                response_time_ms=100
            )

    async def check_health(self) -> ChannelHealth:
        return ChannelHealth(
            status=ChannelHealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            response_time_ms=50
        )

    def get_rate_limit(self) -> int:
        return 60

    def supports_feature(self, feature: str) -> bool:
        return True


@pytest.fixture
def mock_health_monitor():
    """Create mock health monitor."""
    monitor = Mock(spec=HealthMonitor)

    def get_channel_status(channel_name):
        mock_status = Mock()
        mock_status.is_enabled = True
        mock_status.overall_status = HealthStatus.HEALTHY
        mock_status.uptime_percentage = 99.5
        return mock_status

    monitor.get_channel_status = get_channel_status
    return monitor


@pytest.fixture
def fallback_manager(mock_health_monitor):
    """Create fallback manager with mock health monitor."""
    return FallbackManager(mock_health_monitor)


@pytest.fixture
def sample_message_content():
    """Create sample message content."""
    return MessageContent(
        text="Test message",
        subject="Test Subject",
        metadata={"test": "data"}
    )


@pytest.fixture
def channel_instances():
    """Create mock channel instances."""
    return {
        'telegram': MockChannel('telegram', {}),
        'email': MockChannel('email', {}),
        'sms': MockChannel('sms', {})
    }


class TestFallbackRule:
    """Test FallbackRule configuration."""

    def test_valid_fallback_rule(self):
        """Test creating a valid fallback rule."""
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms'],
            strategy=FallbackStrategy.PRIORITY_ORDER,
            max_attempts=3
        )

        assert rule.primary_channel == 'telegram'
        assert rule.fallback_channels == ['email', 'sms']
        assert rule.strategy == FallbackStrategy.PRIORITY_ORDER
        assert rule.max_attempts == 3
        assert rule.enabled is True

    def test_invalid_fallback_rule_empty_fallbacks(self):
        """Test that empty fallback channels raises error."""
        with pytest.raises(ValueError, match="At least one fallback channel must be specified"):
            FallbackRule(
                primary_channel='telegram',
                fallback_channels=[],
                strategy=FallbackStrategy.PRIORITY_ORDER
            )

    def test_invalid_fallback_rule_primary_in_fallbacks(self):
        """Test that primary channel in fallbacks raises error."""
        with pytest.raises(ValueError, match="Primary channel cannot be in fallback channels list"):
            FallbackRule(
                primary_channel='telegram',
                fallback_channels=['telegram', 'email'],
                strategy=FallbackStrategy.PRIORITY_ORDER
            )

    def test_invalid_fallback_rule_max_attempts(self):
        """Test that invalid max attempts raises error."""
        with pytest.raises(ValueError, match="Max attempts must be at least 1"):
            FallbackRule(
                primary_channel='telegram',
                fallback_channels=['email'],
                strategy=FallbackStrategy.PRIORITY_ORDER,
                max_attempts=0
            )


class TestFallbackManager:
    """Test FallbackManager functionality."""

    def test_configure_fallback_rule(self, fallback_manager):
        """Test configuring a fallback rule."""
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms'],
            strategy=FallbackStrategy.PRIORITY_ORDER
        )

        fallback_manager.configure_fallback_rule(rule)

        assert 'telegram' in fallback_manager._fallback_rules
        assert fallback_manager._fallback_rules['telegram'] == rule

    def test_remove_fallback_rule(self, fallback_manager):
        """Test removing a fallback rule."""
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email'],
            strategy=FallbackStrategy.PRIORITY_ORDER
        )

        fallback_manager.configure_fallback_rule(rule)
        assert 'telegram' in fallback_manager._fallback_rules

        success = fallback_manager.remove_fallback_rule('telegram')
        assert success is True
        assert 'telegram' not in fallback_manager._fallback_rules

        # Test removing non-existent rule
        success = fallback_manager.remove_fallback_rule('nonexistent')
        assert success is False

    def test_set_global_fallback_channels(self, fallback_manager):
        """Test setting global fallback channels."""
        channels = ['email', 'sms', 'telegram']
        fallback_manager.set_global_fallback_channels(channels)

        assert fallback_manager._global_fallback_channels == channels

    @pytest.mark.asyncio
    async def test_successful_delivery_no_fallback(self, fallback_manager, sample_message_content, channel_instances):
        """Test successful delivery without needing fallback."""
        success, results, failed_msg = await fallback_manager.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success is True
        assert len(results) == 1
        assert results[0].success is True
        assert failed_msg is None

        # Verify channel was called
        telegram_channel = channel_instances['telegram']
        assert len(telegram_channel.send_calls) == 1
        assert telegram_channel.send_calls[0]['recipient'] == 'test_user'

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, fallback_manager, sample_message_content):
        """Test fallback when primary channel fails."""
        # Create channels with telegram failing
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),
            'email': MockChannel('email', {}),
            'sms': MockChannel('sms', {})
        }

        # Configure fallback rule
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email'],
            strategy=FallbackStrategy.PRIORITY_ORDER
        )
        fallback_manager.configure_fallback_rule(rule)

        success, results, failed_msg = await fallback_manager.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success is True
        assert len(results) == 2  # Both telegram (failed) and email (success)
        assert results[0].success is False  # Telegram failed
        assert results[1].success is True   # Email succeeded
        assert failed_msg is None

        # Verify both channels were called
        assert len(channel_instances['telegram'].send_calls) == 1
        assert len(channel_instances['email'].send_calls) == 1

    @pytest.mark.asyncio
    async def test_all_channels_fail_dead_letter(self, fallback_manager, sample_message_content):
        """Test that all channel failures result in dead letter queue."""
        # Create channels that all fail
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),
            'email': MockChannel('email', {}, should_fail=True),
            'sms': MockChannel('sms', {}, should_fail=True)
        }

        # Configure fallback rule
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms'],
            strategy=FallbackStrategy.PRIORITY_ORDER
        )
        fallback_manager.configure_fallback_rule(rule)

        success, results, failed_msg = await fallback_manager.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success is False
        assert len(results) == 3  # All channels attempted
        assert all(not result.success for result in results)
        assert failed_msg is not None
        assert failed_msg.failure_reason == MessageFailureReason.DELIVERY_FAILED

        # Check if message was added to dead letter queue
        assert len(fallback_manager._dead_letter_queue) == 1
        assert 1 in fallback_manager._dead_letter_queue

    @pytest.mark.asyncio
    async def test_global_fallback_channels(self, fallback_manager, sample_message_content):
        """Test using global fallback channels when no specific rule exists."""
        # Create channels with telegram failing
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),
            'email': MockChannel('email', {}),
            'sms': MockChannel('sms', {})
        }

        # Set global fallback channels (no specific rule for telegram)
        fallback_manager.set_global_fallback_channels(['email', 'sms'])

        success, results, failed_msg = await fallback_manager.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success is True
        assert len(results) == 2  # Telegram (failed) and email (success)
        assert results[0].success is False  # Telegram failed
        assert results[1].success is True   # Email succeeded
        assert failed_msg is None

    @pytest.mark.asyncio
    async def test_retry_queue_processing(self, fallback_manager, sample_message_content):
        """Test retry queue processing."""
        # Create a failed message in retry queue
        failed_msg = FailedMessage(
            message_id=1,
            original_channels=['telegram'],
            content=sample_message_content,
            recipient='test_user',
            priority='NORMAL',
            failure_reason=MessageFailureReason.DELIVERY_FAILED,
            failure_details='Test failure',
            failed_at=datetime.now(timezone.utc) - timedelta(minutes=2),  # Old enough to retry
            retry_count=0
        )

        fallback_manager._retry_queue[1] = failed_msg

        # Create channel instances with telegram now working
        channel_instances = {
            'telegram': MockChannel('telegram', {}),
            'email': MockChannel('email', {})
        }

        # Process retry queue
        results = await fallback_manager.process_retry_queue(channel_instances)

        assert results['processed'] == 1
        assert results['succeeded'] == 1
        assert results['failed'] == 0
        assert results['requeued'] == 0

        # Message should be removed from retry queue
        assert len(fallback_manager._retry_queue) == 0

    def test_dead_letter_queue_management(self, fallback_manager, sample_message_content):
        """Test dead letter queue management."""
        # Add some messages to dead letter queue
        for i in range(5):
            failed_msg = FailedMessage(
                message_id=i,
                original_channels=['telegram'],
                content=sample_message_content,
                recipient=f'user_{i}',
                priority='NORMAL',
                failure_reason=MessageFailureReason.DELIVERY_FAILED,
                failure_details=f'Test failure {i}',
                failed_at=datetime.now(timezone.utc) - timedelta(hours=i)
            )
            fallback_manager._dead_letter_queue[i] = failed_msg

        # Get dead letter messages
        messages = fallback_manager.get_dead_letter_messages(limit=3, offset=1)

        assert len(messages) == 3
        # Should be sorted by failed_at (newest first)
        assert messages[0]['message_id'] == 1  # Second newest after offset
        assert messages[1]['message_id'] == 2
        assert messages[2]['message_id'] == 3

    @pytest.mark.asyncio
    async def test_manual_reprocessing(self, fallback_manager, sample_message_content):
        """Test manual reprocessing of dead letter messages."""
        # Add message to dead letter queue
        failed_msg = FailedMessage(
            message_id=1,
            original_channels=['telegram'],
            content=sample_message_content,
            recipient='test_user',
            priority='NORMAL',
            failure_reason=MessageFailureReason.DELIVERY_FAILED,
            failure_details='Test failure',
            failed_at=datetime.now(timezone.utc)
        )
        fallback_manager._dead_letter_queue[1] = failed_msg

        # Create working channel instances
        channel_instances = {
            'telegram': MockChannel('telegram', {}),
            'email': MockChannel('email', {})
        }

        # Manually reprocess
        success, message = await fallback_manager.reprocess_dead_letter_message(
            message_id=1,
            channel_instances=channel_instances
        )

        assert success is True
        assert "reprocessed successfully" in message

        # Message should be removed from dead letter queue
        assert 1 not in fallback_manager._dead_letter_queue

    def test_fallback_statistics(self, fallback_manager):
        """Test fallback statistics collection."""
        # Add some test data
        fallback_manager._stats['total_fallback_attempts'] = 10
        fallback_manager._stats['successful_fallbacks'] = 8
        fallback_manager._stats['failed_fallbacks'] = 2

        fallback_manager._channel_usage_stats['telegram']['attempts'] = 15
        fallback_manager._channel_usage_stats['telegram']['successes'] = 12
        fallback_manager._channel_usage_stats['telegram']['failures'] = 3

        stats = fallback_manager.get_fallback_statistics()

        assert stats['statistics']['total_fallback_attempts'] == 10
        assert stats['statistics']['successful_fallbacks'] == 8
        assert stats['statistics']['failed_fallbacks'] == 2

        assert 'telegram' in stats['channel_success_rates']
        telegram_stats = stats['channel_success_rates']['telegram']
        assert telegram_stats['success_rate'] == 80.0  # 12/15 * 100
        assert telegram_stats['total_attempts'] == 15

    @pytest.mark.asyncio
    async def test_cleanup_old_dead_letters(self, fallback_manager, sample_message_content):
        """Test cleanup of old dead letter messages."""
        # Add old and new messages
        old_msg = FailedMessage(
            message_id=1,
            original_channels=['telegram'],
            content=sample_message_content,
            recipient='old_user',
            priority='NORMAL',
            failure_reason=MessageFailureReason.DELIVERY_FAILED,
            failure_details='Old failure',
            failed_at=datetime.now(timezone.utc) - timedelta(days=35)  # Older than retention
        )

        new_msg = FailedMessage(
            message_id=2,
            original_channels=['telegram'],
            content=sample_message_content,
            recipient='new_user',
            priority='NORMAL',
            failure_reason=MessageFailureReason.DELIVERY_FAILED,
            failure_details='New failure',
            failed_at=datetime.now(timezone.utc) - timedelta(days=5)  # Within retention
        )

        fallback_manager._dead_letter_queue[1] = old_msg
        fallback_manager._dead_letter_queue[2] = new_msg

        # Run cleanup
        cleaned_count = await fallback_manager.cleanup_old_dead_letters()

        assert cleaned_count == 1
        assert 1 not in fallback_manager._dead_letter_queue  # Old message removed
        assert 2 in fallback_manager._dead_letter_queue      # New message kept


if __name__ == '__main__':
    pytest.main([__file__])