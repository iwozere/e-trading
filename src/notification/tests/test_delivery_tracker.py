#!/usr/bin/env python3
"""
Test script for Delivery Status Tracking System.

Tests comprehensive delivery tracking, multi-channel support, and statistics.
"""

import asyncio
import sys
from pathlib import Path
import time
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.delivery_tracker import (
    delivery_tracker, DeliveryStatus, DeliveryResult,
    MessageDeliveryStatus
)
from src.notification.service.message_queue import QueuedMessage
from src.data.db.models.model_notification import MessagePriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def create_test_message(
    message_id: int,
    channels: list = None,
    priority: MessagePriority = MessagePriority.NORMAL,
    recipient_id: str = "test_user"
) -> QueuedMessage:
    """Create a test message."""
    if channels is None:
        channels = ["telegram_channel", "email_channel"]

    return QueuedMessage(
        id=message_id,
        message_type="test_message",
        priority=priority,
        channels=channels,
        recipient_id=recipient_id,
        template_name=None,
        content={"text": f"Test message {message_id}"},
        metadata=None,
        scheduled_for=datetime.now(timezone.utc),
        retry_count=0,
        max_retries=3,
        created_at=datetime.now(timezone.utc)
    )


async def test_basic_delivery_tracking():
    """Test basic delivery tracking functionality."""
    print("Testing Basic Delivery Tracking...")

    try:
        # Create test message
        message = create_test_message(1001, ["telegram_channel", "email_channel"])

        # Start tracking
        delivery_status = await delivery_tracker.start_tracking(message)
        print(f"✓ Started tracking message {message.id}")

        # Verify initial state
        assert delivery_status.message_id == message.id
        assert delivery_status.overall_status == DeliveryResult.PENDING
        assert len(delivery_status.channels) == 2
        print("✓ Initial delivery status correct")

        # Start delivery attempts
        telegram_attempt = await delivery_tracker.start_channel_attempt(message.id, "telegram_channel")
        email_attempt = await delivery_tracker.start_channel_attempt(message.id, "email_channel")

        assert telegram_attempt is not None
        assert email_attempt is not None
        assert telegram_attempt.status == DeliveryStatus.SENDING
        assert email_attempt.status == DeliveryStatus.SENDING
        print("✓ Started delivery attempts for both channels")

        # Complete telegram delivery successfully
        success = await delivery_tracker.complete_channel_attempt(
            telegram_attempt.attempt_id,
            DeliveryStatus.DELIVERED,
            response_time_ms=150,
            external_id="tg_msg_123"
        )
        assert success
        print("✓ Completed telegram delivery successfully")

        # Complete email delivery with failure
        success = await delivery_tracker.complete_channel_attempt(
            email_attempt.attempt_id,
            DeliveryStatus.FAILED,
            response_time_ms=300,
            error_message="SMTP connection failed"
        )
        assert success
        print("✓ Completed email delivery with failure")

        # Check final status
        final_status = await delivery_tracker.get_delivery_status(message.id)
        assert final_status is not None
        assert final_status.overall_status == DeliveryResult.PARTIAL_SUCCESS
        assert "telegram_channel" in final_status.get_successful_channels()
        assert "email_channel" in final_status.get_failed_channels()
        assert len(final_status.get_pending_channels()) == 0

        print("✓ Final delivery status correct (partial success)")

        return True

    except Exception as e:
        print(f"✗ Basic delivery tracking test failed: {e}")
        return False


async def test_multi_channel_success():
    """Test successful delivery to all channels."""
    print("\nTesting Multi-Channel Success...")

    try:
        # Create message with 3 channels
        message = create_test_message(1002, ["telegram_channel", "email_channel", "sms_channel"])
        delivery_status = await delivery_tracker.start_tracking(message)

        # Start and complete all deliveries successfully
        channels = ["telegram_channel", "email_channel", "sms_channel"]
        response_times = [120, 250, 180]

        for i, channel in enumerate(channels):
            attempt = await delivery_tracker.start_channel_attempt(message.id, channel)
            await delivery_tracker.complete_channel_attempt(
                attempt.attempt_id,
                DeliveryStatus.DELIVERED,
                response_time_ms=response_times[i],
                external_id=f"{channel}_msg_{message.id}"
            )

        # Check final status
        final_status = await delivery_tracker.get_delivery_status(message.id)
        assert final_status.overall_status == DeliveryResult.SUCCESS
        assert len(final_status.get_successful_channels()) == 3
        assert len(final_status.get_failed_channels()) == 0
        assert final_status.completed_at is not None

        # Check response time calculations
        total_time = final_status.get_total_response_time_ms()
        avg_time = final_status.get_average_response_time_ms()

        assert total_time == sum(response_times)
        assert abs(avg_time - (sum(response_times) / len(response_times))) < 0.1

        print("✓ All channels delivered successfully")
        print(f"✓ Total response time: {total_time}ms, Average: {avg_time:.1f}ms")

        return True

    except Exception as e:
        print(f"✗ Multi-channel success test failed: {e}")
        return False


async def test_retry_mechanism():
    """Test retry mechanism for failed deliveries."""
    print("\nTesting Retry Mechanism...")

    try:
        message = create_test_message(1003, ["telegram_channel"])
        delivery_status = await delivery_tracker.start_tracking(message)

        # First attempt - failure
        attempt1 = await delivery_tracker.start_channel_attempt(message.id, "telegram_channel")
        assert attempt1.retry_count == 0

        await delivery_tracker.complete_channel_attempt(
            attempt1.attempt_id,
            DeliveryStatus.FAILED,
            error_message="Network timeout"
        )

        # Second attempt - failure
        attempt2 = await delivery_tracker.start_channel_attempt(message.id, "telegram_channel")
        assert attempt2.retry_count == 1

        await delivery_tracker.complete_channel_attempt(
            attempt2.attempt_id,
            DeliveryStatus.FAILED,
            error_message="Rate limited"
        )

        # Third attempt - success
        attempt3 = await delivery_tracker.start_channel_attempt(message.id, "telegram_channel")
        assert attempt3.retry_count == 2

        await delivery_tracker.complete_channel_attempt(
            attempt3.attempt_id,
            DeliveryStatus.DELIVERED,
            response_time_ms=200,
            external_id="tg_msg_success"
        )

        # Check final status
        final_status = await delivery_tracker.get_delivery_status(message.id)
        assert final_status.overall_status == DeliveryResult.SUCCESS

        # Check attempt history
        telegram_attempts = final_status.channel_attempts["telegram_channel"]
        assert len(telegram_attempts) == 3
        assert telegram_attempts[0].status == DeliveryStatus.FAILED
        assert telegram_attempts[1].status == DeliveryStatus.FAILED
        assert telegram_attempts[2].status == DeliveryStatus.DELIVERED

        print("✓ Retry mechanism working correctly")
        print(f"✓ Successful delivery after {len(telegram_attempts)} attempts")

        return True

    except Exception as e:
        print(f"✗ Retry mechanism test failed: {e}")
        return False


async def test_status_callbacks():
    """Test status change callbacks."""
    print("\nTesting Status Callbacks...")

    try:
        callback_calls = []

        # Define callback function
        async def status_callback(delivery_status: MessageDeliveryStatus):
            callback_calls.append({
                "message_id": delivery_status.message_id,
                "status": delivery_status.overall_status.value,
                "timestamp": datetime.now(timezone.utc)
            })

        # Add global callback
        delivery_tracker.add_global_callback(status_callback)

        # Create and track message
        message = create_test_message(1004, ["telegram_channel"])
        delivery_status = await delivery_tracker.start_tracking(message)

        # Add message-specific callback
        message_callbacks = []
        delivery_status.add_status_callback(
            lambda ds: message_callbacks.append(ds.overall_status.value)
        )

        # Start and complete delivery
        attempt = await delivery_tracker.start_channel_attempt(message.id, "telegram_channel")
        await delivery_tracker.complete_channel_attempt(
            attempt.attempt_id,
            DeliveryStatus.DELIVERED,
            response_time_ms=100
        )

        # Wait a bit for callbacks
        await asyncio.sleep(0.1)

        # Check callbacks were called
        assert len(callback_calls) > 0
        assert callback_calls[-1]["message_id"] == message.id
        assert callback_calls[-1]["status"] == DeliveryResult.SUCCESS.value

        assert len(message_callbacks) > 0
        assert message_callbacks[-1] == DeliveryResult.SUCCESS.value

        print("✓ Status callbacks working correctly")
        print(f"✓ Global callbacks: {len(callback_calls)}, Message callbacks: {len(message_callbacks)}")

        return True

    except Exception as e:
        print(f"✗ Status callbacks test failed: {e}")
        return False


async def test_delivery_statistics():
    """Test delivery statistics calculation."""
    print("\nTesting Delivery Statistics...")

    try:
        # Create multiple messages with different outcomes
        messages_data = [
            (2001, ["telegram_channel"], DeliveryStatus.DELIVERED, 150),
            (2002, ["email_channel"], DeliveryStatus.FAILED, None),
            (2003, ["telegram_channel", "email_channel"], "partial", None),  # Special case
            (2004, ["sms_channel"], DeliveryStatus.DELIVERED, 300),
        ]

        for msg_id, channels, outcome, response_time in messages_data:
            message = create_test_message(msg_id, channels)
            delivery_status = await delivery_tracker.start_tracking(message)

            if outcome == "partial":
                # Telegram succeeds, email fails
                tg_attempt = await delivery_tracker.start_channel_attempt(msg_id, "telegram_channel")
                await delivery_tracker.complete_channel_attempt(
                    tg_attempt.attempt_id, DeliveryStatus.DELIVERED, response_time_ms=200
                )

                email_attempt = await delivery_tracker.start_channel_attempt(msg_id, "email_channel")
                await delivery_tracker.complete_channel_attempt(
                    email_attempt.attempt_id, DeliveryStatus.FAILED, error_message="SMTP error"
                )
            else:
                # Single channel outcome
                for channel in channels:
                    attempt = await delivery_tracker.start_channel_attempt(msg_id, channel)
                    await delivery_tracker.complete_channel_attempt(
                        attempt.attempt_id, outcome, response_time_ms=response_time
                    )

        # Get statistics
        stats = delivery_tracker.get_statistics()

        print("✓ Statistics calculated:")
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Successful deliveries: {stats['successful_deliveries']}")
        print(f"  Failed deliveries: {stats['failed_deliveries']}")
        print(f"  Partial deliveries: {stats['partial_deliveries']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Average response time: {stats['avg_response_time_ms']:.1f}ms")

        # Verify some basic statistics
        assert stats['total_messages'] >= 4
        assert stats['successful_deliveries'] >= 1
        assert stats['failed_deliveries'] >= 1
        assert stats['partial_deliveries'] >= 1

        # Check channel statistics
        if 'telegram_channel' in stats['channel_statistics']:
            tg_stats = stats['channel_statistics']['telegram_channel']
            print(f"  Telegram attempts: {tg_stats['total_attempts']}")
            print(f"  Telegram success rate: {tg_stats['successful_attempts']}/{tg_stats['total_attempts']}")

        return True

    except Exception as e:
        print(f"✗ Delivery statistics test failed: {e}")
        return False


async def test_delivery_history():
    """Test delivery history querying."""
    print("\nTesting Delivery History...")

    try:
        # Create messages for different recipients
        recipients = ["user1", "user2", "user1"]

        for i, recipient in enumerate(recipients):
            msg_id = 3001 + i
            message = create_test_message(msg_id, ["telegram_channel"], recipient_id=recipient)
            delivery_status = await delivery_tracker.start_tracking(message)

            # Complete delivery
            attempt = await delivery_tracker.start_channel_attempt(msg_id, "telegram_channel")
            await delivery_tracker.complete_channel_attempt(
                attempt.attempt_id, DeliveryStatus.DELIVERED, response_time_ms=100 + i * 50
            )

        # Query history for user1
        user1_history = await delivery_tracker.get_delivery_history(recipient_id="user1")
        assert len(user1_history) == 2
        assert all(d.recipient_id == "user1" for d in user1_history)

        # Query history for telegram channel
        telegram_history = await delivery_tracker.get_delivery_history(channel="telegram_channel")
        assert len(telegram_history) >= 3

        # Query successful deliveries
        success_history = await delivery_tracker.get_delivery_history(status=DeliveryResult.SUCCESS)
        assert len(success_history) >= 3

        print("✓ Delivery history queries working")
        print(f"  User1 deliveries: {len(user1_history)}")
        print(f"  Telegram deliveries: {len(telegram_history)}")
        print(f"  Successful deliveries: {len(success_history)}")

        return True

    except Exception as e:
        print(f"✗ Delivery history test failed: {e}")
        return False


async def test_performance():
    """Test performance with multiple concurrent deliveries."""
    print("\nTesting Performance...")

    try:
        num_messages = 50
        num_channels = 3

        start_time = time.time()

        # Create and track multiple messages concurrently
        tasks = []

        for i in range(num_messages):
            msg_id = 4000 + i
            channels = [f"channel_{j}" for j in range(num_channels)]
            message = create_test_message(msg_id, channels)

            task = delivery_tracker.start_tracking(message)
            tasks.append(task)

        # Wait for all tracking to start
        delivery_statuses = await asyncio.gather(*tasks)
        tracking_time = time.time() - start_time

        print(f"✓ Started tracking {num_messages} messages in {tracking_time:.3f}s")
        print(f"  Throughput: {num_messages / tracking_time:.1f} messages/sec")

        # Complete deliveries concurrently
        start_time = time.time()
        completion_tasks = []

        for delivery_status in delivery_statuses:
            for channel in delivery_status.channels:
                async def complete_delivery(msg_id, ch):
                    attempt = await delivery_tracker.start_channel_attempt(msg_id, ch)
                    await delivery_tracker.complete_channel_attempt(
                        attempt.attempt_id,
                        DeliveryStatus.DELIVERED,
                        response_time_ms=100
                    )

                task = complete_delivery(delivery_status.message_id, channel)
                completion_tasks.append(task)

        await asyncio.gather(*completion_tasks)
        completion_time = time.time() - start_time

        total_attempts = num_messages * num_channels
        print(f"✓ Completed {total_attempts} delivery attempts in {completion_time:.3f}s")
        print(f"  Throughput: {total_attempts / completion_time:.1f} attempts/sec")

        # Verify all deliveries completed successfully
        success_count = 0
        for delivery_status in delivery_statuses:
            final_status = await delivery_tracker.get_delivery_status(delivery_status.message_id)
            if final_status.overall_status == DeliveryResult.SUCCESS:
                success_count += 1

        print(f"✓ Successfully completed {success_count}/{num_messages} deliveries")

        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


async def main():
    """Run all delivery tracking tests."""
    print("=== Delivery Status Tracking Tests ===\n")

    tests = [
        ("Basic Delivery Tracking", test_basic_delivery_tracking),
        ("Multi-Channel Success", test_multi_channel_success),
        ("Retry Mechanism", test_retry_mechanism),
        ("Status Callbacks", test_status_callbacks),
        ("Delivery Statistics", test_delivery_statistics),
        ("Delivery History", test_delivery_history),
        ("Performance", test_performance),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

        print()

    # Summary
    print("=== Test Results ===")
    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("🎉 All delivery tracking tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)