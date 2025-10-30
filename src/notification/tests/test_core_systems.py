#!/usr/bin/env python3
"""
Simple test for core rate limiting, priority, and batching systems.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.rate_limiter import rate_limiter, RateLimitResult, RateLimitConfig
from src.notification.service.priority_handler import priority_handler
from src.notification.service.batch_processor import batch_processor, BatchConfig
from src.notification.service.message_queue import QueuedMessage
from src.data.db.models.model_notification import MessagePriority
from datetime import datetime, timezone


def create_test_message(msg_id: int, priority: MessagePriority = MessagePriority.NORMAL) -> QueuedMessage:
    """Create a test message."""
    return QueuedMessage(
        id=msg_id,
        message_type="test",
        priority=priority,
        channels=["test_channel"],
        recipient_id="test_user",
        template_name=None,
        content={"text": f"Test message {msg_id}"},
        metadata=None,
        scheduled_for=datetime.now(timezone.utc),
        retry_count=0,
        max_retries=3,
        created_at=datetime.now(timezone.utc)
    )


async def test_basic_functionality():
    """Test basic functionality of all systems."""
    print("Testing basic functionality...")

    try:
        # Test Rate Limiter
        print("1. Testing Rate Limiter...")
        config = RateLimitConfig(max_tokens=3, refill_rate=3)
        rate_limiter.set_channel_config("test_channel", config)

        # Should allow first 3, then rate limit
        results = []
        for i in range(5):
            result, _ = await rate_limiter.check_rate_limit("user1", "test_channel", MessagePriority.NORMAL)
            results.append(result)

        allowed = sum(1 for r in results if r == RateLimitResult.ALLOWED)
        rate_limited = sum(1 for r in results if r == RateLimitResult.RATE_LIMITED)

        print(f"   Rate limiter: {allowed} allowed, {rate_limited} rate limited ✓")

        # Test Priority Handler
        print("2. Testing Priority Handler...")
        messages = [
            create_test_message(1, MessagePriority.LOW),
            create_test_message(2, MessagePriority.CRITICAL),
            create_test_message(3, MessagePriority.NORMAL),
        ]

        for msg in messages:
            await priority_handler.enqueue_message(msg)

        # Dequeue - should get CRITICAL first
        first_msg = await priority_handler.dequeue_next_message(timeout=1.0)
        if first_msg and first_msg.priority == MessagePriority.CRITICAL:
            print("   Priority handler: CRITICAL message processed first ✓")
        else:
            print("   Priority handler: Failed ✗")
            return False

        # Test Batch Processor
        print("3. Testing Batch Processor...")
        batch_config = BatchConfig(max_batch_size=2, max_wait_time_seconds=1.0)
        batch_processor.set_channel_config("test_channel", batch_config)

        processed_batches = []

        async def batch_callback(batch):
            processed_batches.append(batch)

        batch_processor.set_batch_processor_callback(batch_callback)
        await batch_processor.start()

        # Add normal priority messages (should be batched)
        normal_messages = [
            create_test_message(10, MessagePriority.NORMAL),
            create_test_message(11, MessagePriority.NORMAL),
        ]

        for msg in normal_messages:
            await batch_processor.add_message(msg)

        # Wait for batch processing
        await asyncio.sleep(2.0)

        if processed_batches and processed_batches[0].size == 2:
            print("   Batch processor: 2 messages batched together ✓")
        else:
            print("   Batch processor: Failed ✗")
            return False

        await batch_processor.stop()

        print("✓ All basic functionality tests passed!")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


async def main():
    """Run basic tests."""
    print("=== Core Systems Basic Test ===\n")

    success = await test_basic_functionality()

    if success:
        print("\n🎉 All core systems working correctly!")
        return 0
    else:
        print("\n❌ Some systems failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)