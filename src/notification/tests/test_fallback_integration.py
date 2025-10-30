"""
Integration Tests for Channel Fallback System

Integration tests that verify the fallback system works correctly
with the health monitor and message processor.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.notification.service.fallback_manager import FallbackManager, FallbackRule, FallbackStrategy
from src.notification.service.health_monitor import HealthMonitor, HealthStatus, HealthCheckConfig
from src.notification.channels.base import MessageContent
from src.notification.tests.test_fallback_manager import MockChannel


@pytest.fixture
def mock_health_monitor_integration():
    """Create mock health monitor for integration tests."""
    monitor = Mock(spec=HealthMonitor)

    def get_channel_status(channel_name):
        mock_status = Mock()
        mock_status.is_enabled = True
        mock_status.overall_status = HealthStatus.HEALTHY
        mock_status.uptime_percentage = 99.5
        return mock_status

    monitor.get_channel_status = get_channel_status
    monitor.manually_disable_channel = Mock(return_value=True)
    monitor.manually_enable_channel = Mock(return_value=True)
    return monitor


@pytest.fixture
def fallback_manager_with_health(mock_health_monitor_integration):
    """Create fallback manager with mock health monitor."""
    return FallbackManager(mock_health_monitor_integration)


@pytest.fixture
def sample_message_content():
    """Create sample message content."""
    return MessageContent(
        text="Integration test message",
        subject="Test Subject",
        metadata={"test": "integration"}
    )


class TestFallbackIntegration:
    """Integration tests for fallback system."""

    @pytest.mark.asyncio
    async def test_health_based_fallback_routing(self, fallback_manager_with_health, sample_message_content):
        """Test that fallback routing considers channel health."""
        # Create channel instances
        channel_instances = {
            'telegram': MockChannel('telegram', {}),
            'email': MockChannel('email', {}),
            'sms': MockChannel('sms', {})
        }

        # Configure health-based fallback rule
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms'],
            strategy=FallbackStrategy.HEALTH_BASED,
            max_attempts=2
        )
        fallback_manager_with_health.configure_fallback_rule(rule)

        # Simulate telegram being unhealthy by making the health monitor return disabled status
        def get_channel_status_disabled(channel_name):
            mock_status = Mock()
            if channel_name == 'telegram':
                mock_status.is_enabled = False  # Telegram is disabled
                mock_status.overall_status = HealthStatus.DISABLED
            else:
                mock_status.is_enabled = True
                mock_status.overall_status = HealthStatus.HEALTHY
            mock_status.uptime_percentage = 99.5
            return mock_status

        fallback_manager_with_health.health_monitor.get_channel_status = get_channel_status_disabled

        # Attempt delivery - should skip telegram and go to fallback
        success, results, failed_msg = await fallback_manager_with_health.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        # Should succeed using fallback channel
        assert success is True
        assert failed_msg is None

        # Telegram should not have been called due to health check
        assert len(channel_instances['telegram'].send_calls) == 0

        # Email should have been called as first healthy fallback
        assert len(channel_instances['email'].send_calls) == 1

    @pytest.mark.asyncio
    async def test_fallback_with_health_recovery(self, fallback_manager_with_health, sample_message_content):
        """Test that channels can recover and be used again."""
        # Create channel instances with telegram initially failing
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),
            'email': MockChannel('email', {}),
            'sms': MockChannel('sms', {})
        }

        # Configure fallback rule
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email'],
            strategy=FallbackStrategy.PRIORITY_ORDER,
            max_attempts=2
        )
        fallback_manager_with_health.configure_fallback_rule(rule)

        # First attempt - telegram fails, email succeeds
        success1, results1, failed_msg1 = await fallback_manager_with_health.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success1 is True
        assert len(results1) == 2  # Both channels attempted
        assert results1[0].success is False  # Telegram failed
        assert results1[1].success is True   # Email succeeded

        # Now fix telegram
        channel_instances['telegram'].should_fail = False

        # Second attempt - telegram should work now
        success2, results2, failed_msg2 = await fallback_manager_with_health.attempt_delivery_with_fallback(
            message_id=2,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success2 is True
        assert len(results2) == 1  # Only telegram attempted
        assert results2[0].success is True  # Telegram succeeded

        # Verify call counts
        assert len(channel_instances['telegram'].send_calls) == 2  # Called in both attempts
        assert len(channel_instances['email'].send_calls) == 1     # Only called in first attempt

    @pytest.mark.asyncio
    async def test_retry_queue_with_health_changes(self, fallback_manager_with_health, sample_message_content):
        """Test retry queue processing when channel health changes."""
        # Create channel instances with all channels initially failing
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),
            'email': MockChannel('email', {}, should_fail=True),
            'sms': MockChannel('sms', {}, should_fail=True)
        }

        # Configure fallback rule
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms'],
            strategy=FallbackStrategy.PRIORITY_ORDER,
            max_attempts=3
        )
        fallback_manager_with_health.configure_fallback_rule(rule)

        # First attempt - all channels fail, message goes to retry queue
        success1, results1, failed_msg1 = await fallback_manager_with_health.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success1 is False
        assert failed_msg1 is not None

        # Check retry queue has the message
        retry_status = fallback_manager_with_health.get_retry_queue_status()
        assert retry_status['total_messages'] == 1

        # Now fix email channel
        channel_instances['email'].should_fail = False

        # Process retry queue - should succeed now
        results = await fallback_manager_with_health.process_retry_queue(channel_instances)

        assert results['processed'] == 1
        assert results['succeeded'] == 1
        assert results['failed'] == 0

        # Retry queue should be empty now
        retry_status_after = fallback_manager_with_health.get_retry_queue_status()
        assert retry_status_after['total_messages'] == 0

    @pytest.mark.asyncio
    async def test_dead_letter_queue_management_integration(self, fallback_manager_with_health, sample_message_content):
        """Test dead letter queue management with real scenarios."""
        # Create channel instances that will cause permanent failures
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),
            'email': MockChannel('email', {}, should_fail=True),
            'sms': MockChannel('sms', {}, should_fail=True)
        }

        # Make failures non-retryable by using BOUNCED status
        for channel in channel_instances.values():
            channel.should_fail = True

        # Configure fallback rule
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms'],
            strategy=FallbackStrategy.PRIORITY_ORDER,
            max_attempts=2
        )
        fallback_manager_with_health.configure_fallback_rule(rule)

        # Attempt delivery - should fail and go to dead letter queue
        success, results, failed_msg = await fallback_manager_with_health.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success is False
        assert failed_msg is not None

        # Check dead letter queue
        dead_letters = fallback_manager_with_health.get_dead_letter_messages(limit=10)
        assert len(dead_letters) == 1
        assert dead_letters[0]['message_id'] == 1

        # Now fix all channels
        for channel in channel_instances.values():
            channel.should_fail = False

        # Manually reprocess the dead letter message
        reprocess_success, reprocess_msg = await fallback_manager_with_health.reprocess_dead_letter_message(
            message_id=1,
            channel_instances=channel_instances
        )

        assert reprocess_success is True
        assert "reprocessed successfully" in reprocess_msg

        # Dead letter queue should be empty now
        dead_letters_after = fallback_manager_with_health.get_dead_letter_messages(limit=10)
        assert len(dead_letters_after) == 0

    @pytest.mark.asyncio
    async def test_fallback_statistics_integration(self, fallback_manager_with_health, sample_message_content):
        """Test that fallback statistics are correctly tracked."""
        # Create mixed channel instances
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),  # Will fail
            'email': MockChannel('email', {}),                          # Will succeed
            'sms': MockChannel('sms', {})                               # Backup
        }

        # Configure fallback rule
        rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms'],
            strategy=FallbackStrategy.PRIORITY_ORDER,
            max_attempts=2
        )
        fallback_manager_with_health.configure_fallback_rule(rule)

        # Perform multiple delivery attempts
        for i in range(5):
            success, results, failed_msg = await fallback_manager_with_health.attempt_delivery_with_fallback(
                message_id=i,
                channels=['telegram'],
                recipient=f'user_{i}',
                content=sample_message_content,
                priority='NORMAL',
                channel_instances=channel_instances
            )

            # All should succeed via email fallback
            assert success is True

        # Check statistics
        stats = fallback_manager_with_health.get_fallback_statistics()

        # Should have fallback attempts recorded
        assert stats['statistics']['total_fallback_attempts'] == 5
        assert stats['statistics']['successful_fallbacks'] == 5
        assert stats['statistics']['failed_fallbacks'] == 0

        # Channel success rates should be tracked
        assert 'telegram' in stats['channel_success_rates']
        assert 'email' in stats['channel_success_rates']

        telegram_stats = stats['channel_success_rates']['telegram']
        email_stats = stats['channel_success_rates']['email']

        assert telegram_stats['success_rate'] == 0.0   # All failed
        assert email_stats['success_rate'] == 100.0    # All succeeded

        # Recent attempts should be recorded
        assert len(stats['recent_fallback_attempts']) > 0

    @pytest.mark.asyncio
    async def test_multiple_fallback_strategies(self, fallback_manager_with_health, sample_message_content):
        """Test different fallback strategies work correctly."""
        # Create channel instances
        channel_instances = {
            'telegram': MockChannel('telegram', {}, should_fail=True),
            'email': MockChannel('email', {}),
            'sms': MockChannel('sms', {}),
            'slack': MockChannel('slack', {})
        }

        # Test priority order strategy
        priority_rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['email', 'sms', 'slack'],
            strategy=FallbackStrategy.PRIORITY_ORDER,
            max_attempts=2
        )
        fallback_manager_with_health.configure_fallback_rule(priority_rule)

        success, results, failed_msg = await fallback_manager_with_health.attempt_delivery_with_fallback(
            message_id=1,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success is True
        # Should use email (first in fallback list)
        assert len(channel_instances['email'].send_calls) == 1
        assert len(channel_instances['sms'].send_calls) == 0  # Should not reach SMS

        # Reset call counts
        for channel in channel_instances.values():
            channel.send_calls.clear()

        # Test round-robin strategy
        rr_rule = FallbackRule(
            primary_channel='telegram',
            fallback_channels=['sms', 'slack', 'email'],
            strategy=FallbackStrategy.ROUND_ROBIN,
            max_attempts=2
        )
        fallback_manager_with_health.configure_fallback_rule(rr_rule)

        success2, results2, failed_msg2 = await fallback_manager_with_health.attempt_delivery_with_fallback(
            message_id=2,
            channels=['telegram'],
            recipient='test_user',
            content=sample_message_content,
            priority='NORMAL',
            channel_instances=channel_instances
        )

        assert success2 is True
        # Round-robin should pick a different channel (based on time)
        total_fallback_calls = (
            len(channel_instances['email'].send_calls) +
            len(channel_instances['sms'].send_calls) +
            len(channel_instances['slack'].send_calls)
        )
        assert total_fallback_calls == 1  # One fallback channel should be called


if __name__ == '__main__':
    pytest.main([__file__])