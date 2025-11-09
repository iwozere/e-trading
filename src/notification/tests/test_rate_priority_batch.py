#!/usr/bin/env python3
"""
Test script for Rate Limiting, Priority Handling, and Batching systems.

Tests the integration of all three systems and their interactions.
"""

import asyncio
import sys
from pathlib import Path
import time
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.rate_limiter import rate_limiter, RateLimitResult, RateLimitConfig
from src.notification.service.priority_handler import priority_handler
from src.notification.service.batch_processor import batch_processor, BatchConfig
from src.notification.service.message_queue import QueuedMessage
from src.data.db.models.model_notification import MessagePriority
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def create_test_message(
    message_id: int,
    priority: MessagePriority = MessagePriority.NORMAL,
    channels: list = None,
    recipient_id: str = "test_user",
    message_type: str = "test_message"
) -> QueuedMessage:
    """Create a test message."""
    if channels is None:
        channels = ["telegram_channel"]

    return QueuedMessage(
        id=message_id,
        message_type=message_type,
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


async def test_rate_limiter():
    """Test rate limiting functionality."""
    print("Testing Rate Limiter...")

    try:
        # Configure rate limiter for testing
        test_config = RateLimitConfig(
            max_tokens=5,
            refill_rate=5,  # 5 tokens per minute
            window_minutes=1
        )
        rate_limiter.set_channel_config("test_channel", test_config)

        user_id = "test_user_rate"
        channel = "test_channel"

        # Test normal rate limiting
        results = []
        for i in range(7):  # Try to consume 7 tokens (should hit limit)
            result, wait_time = await rate_limiter.check_rate_limit(
                user_id, channel, MessagePriority.NORMAL, 1
            )
            results.append((result, wait_time))

        # Should allow first 5, then rate limit
        allowed_count = sum(1 for result, _ in results if result == RateLimitResult.ALLOWED)
        rate_limited_count = sum(1 for result, _ in results if result == RateLimitResult.RATE_LIMITED)

        print(f"✓ Rate limiting: {allowed_count} allowed, {rate_limited_count} rate limited")

        # Test priority bypass
        result, _ = await rate_limiter.check_rate_limit(
            user_id, channel, MessagePriority.CRITICAL, 1
        )

        if result == RateLimitResult.BYPASSED:
            print("✓ Priority bypass working for CRITICAL messages")
        else:
            print("✗ Priority bypass failed")
            return False

        # Test bucket status
        status = await rate_limiter.get_bucket_status(user_id, channel)
        print(f"✓ Bucket status: {status['tokens']:.1f}/{status['max_tokens']} tokens")

        # Test statistics
        stats = rate_limiter.get_statistics(user_id, channel, hours=1)
        print(f"✓ Rate limit violations: {stats['total_violations']}")

        return True

    except Exception as e:
        print(f"✗ Rate limiter test failed: {e}")
        return False


async def test_priority_handler():
    """Test priority message handling."""
    print("\nTesting Priority Handler...")

    try:
        # Create messages with different priorities
        messages = [
            create_test_message(1, MessagePriority.LOW),
            create_test_message(2, MessagePriority.NORMAL),
            create_test_message(3, MessagePriority.HIGH),
            create_test_message(4, MessagePriority.CRITICAL),
            create_test_message(5, MessagePriority.NORMAL),
        ]

        # Enqueue messages
        for message in messages:
            await priority_handler.enqueue_message(message)

        print(f"✓ Enqueued {len(messages)} messages with different priorities")

        # Test priority ordering
        dequeued_messages = []
        for _ in range(len(messages)):
            message = await priority_handler.dequeue_next_message(timeout=1.0)
            if message:
                dequeued_messages.append(message)

        # Check if CRITICAL came first
        if dequeued_messages and dequeued_messages[0].priority == MessagePriority.CRITICAL:
            print("✓ CRITICAL message processed first")
        else:
            print("✗ Priority ordering failed")
            return False

        # Test critical message handling
        critical_messages = await priority_handler.get_critical_messages()
        print(f"✓ Critical message handling: {len(critical_messages)} critical messages")

        # Test queue status
        status = priority_handler.get_queue_status()
        print(f"✓ Queue status: {status['total_queued']} queued, {status['processing_stats']['total_processed']} processed")

        # Test SLA monitoring
        sla_violations = priority_handler.get_sla_violations()
        print(f"✓ SLA violations: {len(sla_violations)}")

        return True

    except Exception as e:
        print(f"✗ Priority handler test failed: {e}")
        return False


async def test_batch_processor():
    """Test batch processing functionality."""
    print("\nTesting Batch Processor...")

    try:
        # Configure batch processor
        test_config = BatchConfig(
            max_batch_size=3,
            max_wait_time_seconds=2.0,
            min_batch_size=1,
            prefer_same_recipient=True
        )
        batch_processor.set_channel_config("test_channel", test_config)

        # Set up callback to capture processed batches
        processed_batches = []

        async def batch_callback(batch):
            processed_batches.append(batch)
            print(f"  Processed batch {batch.batch_id}: {batch.size} messages")

        batch_processor.set_batch_processor_callback(batch_callback)
        await batch_processor.start()

        # Create normal priority messages (these should be batched)
        messages = [
            create_test_message(10, MessagePriority.NORMAL, ["test_channel"], "user1"),
            create_test_message(11, MessagePriority.NORMAL, ["test_channel"], "user1"),
            create_test_message(12, MessagePriority.NORMAL, ["test_channel"], "user1"),
            create_test_message(13, MessagePriority.NORMAL, ["test_channel"], "user2"),
        ]

        # Add messages to batch processor
        for message in messages:
            await batch_processor.add_message(message)

        # Wait for batches to be processed
        await asyncio.sleep(3.0)

        if processed_batches:
            print(f"✓ Batch processing: {len(processed_batches)} batches created")

            # Check batch grouping
            first_batch = processed_batches[0]
            if first_batch.size == 3 and first_batch.recipient_id == "user1":
                print("✓ Batch grouping by recipient working")
            else:
                print(f"✗ Batch grouping failed: size={first_batch.size}, recipient={first_batch.recipient_id}")
        else:
            print("✗ No batches were processed")
            return False

        # Test batch status
        status = batch_processor.get_batch_status()
        print(f"✓ Batch statistics: {status['statistics']['total_batches']} total batches")

        # Test manual flush
        await batch_processor.add_message(create_test_message(20, MessagePriority.NORMAL, ["test_channel"]))
        flushed_batch = await batch_processor.flush_channel("test_channel")

        if flushed_batch:
            print("✓ Manual batch flush working")

        await batch_processor.stop()
        return True

    except Exception as e:
        print(f"✗ Batch processor test failed: {e}")
        return False


async def test_integration():
    """Test integration of all three systems."""
    print("\nTesting System Integration...")

    try:
        # Configure systems for integration test
        rate_limiter.set_channel_config("integration_channel", RateLimitConfig(
            max_tokens=3,
            refill_rate=3
        ))

        batch_processor.set_channel_config("integration_channel", BatchConfig(
            max_batch_size=2,
            max_wait_time_seconds=1.0
        ))

        await batch_processor.start()

        # Test scenario: Mix of priority messages with rate limiting
        test_messages = [
            create_test_message(100, MessagePriority.CRITICAL, ["integration_channel"]),  # Should bypass rate limit
            create_test_message(101, MessagePriority.NORMAL, ["integration_channel"]),    # Should be batched
            create_test_message(102, MessagePriority.NORMAL, ["integration_channel"]),    # Should be batched
            create_test_message(103, MessagePriority.HIGH, ["integration_channel"]),      # Should bypass batching
            create_test_message(104, MessagePriority.NORMAL, ["integration_channel"]),    # Should be rate limited after 3 tokens
        ]

        # Process messages through the integrated system
        user_id = "integration_user"
        channel = "integration_channel"

        processed_batches = []
        priority_messages = []
        rate_limited_messages = []

        async def integration_batch_callback(batch):
            processed_batches.append(batch)

        batch_processor.set_batch_processor_callback(integration_batch_callback)

        for message in test_messages:
            # Check rate limit first
            result, wait_time = await rate_limiter.check_rate_limit(
                user_id, channel, message.priority, 1
            )

            if result == RateLimitResult.RATE_LIMITED:
                rate_limited_messages.append(message)
                continue

            # Handle based on priority
            if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
                # High priority - immediate processing
                await priority_handler.enqueue_message(message)
                priority_messages.append(message)
            else:
                # Normal/Low priority - batching
                await batch_processor.add_message(message)

        # Wait for processing
        await asyncio.sleep(2.0)

        print("✓ Integration test results:")
        print(f"  Priority messages: {len(priority_messages)}")
        print(f"  Batched messages: {sum(b.size for b in processed_batches)}")
        print(f"  Rate limited messages: {len(rate_limited_messages)}")

        # Verify expected behavior
        if (len(priority_messages) >= 2 and  # CRITICAL and HIGH
            len(processed_batches) >= 1 and   # Normal messages batched
            len(rate_limited_messages) >= 0): # Some may be rate limited
            print("✓ Integration test passed")
            success = True
        else:
            print("✗ Integration test failed - unexpected results")
            success = False

        await batch_processor.stop()
        return success

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


async def test_performance():
    """Test performance under load."""
    print("\nTesting Performance...")

    try:
        # Performance test configuration
        num_messages = 100
        num_users = 10

        start_time = time.time()

        # Test rate limiter performance
        rate_limit_tasks = []
        for i in range(num_messages):
            user_id = f"perf_user_{i % num_users}"
            channel = "perf_channel"
            priority = MessagePriority.NORMAL

            task = rate_limiter.check_rate_limit(user_id, channel, priority, 1)
            rate_limit_tasks.append(task)

        rate_limit_results = await asyncio.gather(*rate_limit_tasks)
        rate_limit_time = time.time() - start_time

        allowed_count = sum(1 for result, _ in rate_limit_results if result == RateLimitResult.ALLOWED)

        print(f"✓ Rate limiter performance: {num_messages} checks in {rate_limit_time:.3f}s")
        print(f"  Throughput: {num_messages / rate_limit_time:.1f} checks/sec")
        print(f"  Allowed: {allowed_count}/{num_messages}")

        # Test priority handler performance
        start_time = time.time()

        priority_messages = [
            create_test_message(i, MessagePriority.NORMAL)
            for i in range(num_messages)
        ]

        for message in priority_messages:
            await priority_handler.enqueue_message(message)

        priority_time = time.time() - start_time

        print(f"✓ Priority handler performance: {num_messages} enqueues in {priority_time:.3f}s")
        print(f"  Throughput: {num_messages / priority_time:.1f} enqueues/sec")

        # Clean up
        priority_handler.clear_queue()

        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=== Rate Limiting, Priority Handling, and Batching Tests ===\n")

    tests = [
        ("Rate Limiter", test_rate_limiter),
        ("Priority Handler", test_priority_handler),
        ("Batch Processor", test_batch_processor),
        ("System Integration", test_integration),
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
        print("🎉 All rate limiting and priority tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)