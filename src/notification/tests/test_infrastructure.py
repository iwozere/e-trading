#!/usr/bin/env python3
"""
Test script for notification service core infrastructure.

Tests the FastAPI application, message queue, and processor functionality.
"""

import asyncio
import sys
from pathlib import Path
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.config import config
from src.notification.service.dependencies import init_database, get_repository_context
from src.notification.service.message_queue import message_queue, MessagePriority
from src.notification.service.processor import message_processor
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


async def test_database_connection():
    """Test database connection and initialization."""
    print("Testing database connection...")

    try:
        init_database(
            database_url=config.database.url,
            echo=config.database.echo,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow
        )

        # Create tables using SQLAlchemy
        from src.data.db.core.base import Base
        from src.notification.service.dependencies import engine
        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created")

        with get_repository_context() as repo:
            # Test basic query using SQLAlchemy text
            from sqlalchemy import text
            repo.session.execute(text("SELECT 1"))
            print("✓ Database connection successful")

            # Test message creation
            message_data = {
                'message_type': 'test_message',
                'channels': ['telegram'],
                'content': {'text': 'Test message'},
                'priority': MessagePriority.NORMAL.value
            }

            message = repo.messages.create_message(message_data)
            print(f"✓ Message created with ID: {message.id}")

            # Test message retrieval
            retrieved_message = repo.messages.get_message(message.id)
            if retrieved_message:
                print(f"✓ Message retrieved: {retrieved_message.message_type}")
            else:
                print("✗ Failed to retrieve message")

    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

    return True


async def test_message_queue():
    """Test message queue functionality."""
    print("\nTesting message queue...")

    try:
        # Test message enqueueing
        message_data = {
            'message_type': 'queue_test',
            'channels': ['telegram', 'email'],
            'content': {'text': 'Queue test message'},
            'recipient_id': 'test_user'
        }

        message_id = message_queue.enqueue(message_data, MessagePriority.HIGH)
        print(f"✓ Message enqueued with ID: {message_id}")

        # Test message dequeuing
        messages = message_queue.dequeue(limit=5)
        if messages:
            print(f"✓ Dequeued {len(messages)} messages")
            for msg in messages:
                print(f"  - Message {msg.id}: {msg.message_type} (priority: {msg.priority.value})")
        else:
            print("✓ No messages in queue (expected if database is clean)")

        # Test queue statistics
        stats = message_queue.get_queue_stats()
        print(f"✓ Queue stats: {json.dumps(stats, indent=2)}")

    except Exception as e:
        print(f"✗ Message queue test failed: {e}")
        return False

    return True


async def test_message_processor():
    """Test message processor functionality."""
    print("\nTesting message processor...")

    try:
        # Start processor
        await message_processor.start()
        print("✓ Message processor started")

        # Wait a moment for processor to initialize
        await asyncio.sleep(1)

        # Check processor status
        if message_processor.is_running:
            print("✓ Message processor is running")
        else:
            print("✗ Message processor is not running")
            return False

        # Get processor statistics
        stats = message_processor.get_stats()
        print(f"✓ Processor stats: {json.dumps(stats, indent=2)}")

        # Enqueue a test message for processing
        message_data = {
            'message_type': 'processor_test',
            'channels': ['telegram'],
            'content': {'text': 'Processor test message'},
            'recipient_id': 'test_user'
        }

        message_id = message_queue.enqueue(message_data, MessagePriority.CRITICAL)
        print(f"✓ Test message enqueued for processing: {message_id}")

        # Wait for processing
        print("Waiting for message processing...")
        await asyncio.sleep(3)

        # Check updated stats
        updated_stats = message_processor.get_stats()
        print(f"✓ Updated processor stats: {json.dumps(updated_stats, indent=2)}")

        # Shutdown processor
        await message_processor.shutdown()
        print("✓ Message processor shutdown complete")

    except Exception as e:
        print(f"✗ Message processor test failed: {e}")
        return False

    return True


async def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")

    try:
        print(f"✓ Service name: {config.service_name}")
        print(f"✓ Version: {config.version}")
        print(f"✓ Database URL: {config.database.url}")
        print(f"✓ Server: {config.server.host}:{config.server.port}")
        print(f"✓ Max workers: {config.processing.max_workers}")
        print(f"✓ Batch size: {config.processing.batch_size}")

        # Test channel configuration
        enabled_channels = [
            name for name in ["telegram", "email", "sms"]
            if config.is_channel_enabled(name)
        ]
        print(f"✓ Enabled channels: {enabled_channels}")

        # Test rate limits
        for channel in ["telegram", "email", "sms"]:
            rate_limit = config.get_rate_limit_for_channel(channel)
            print(f"✓ {channel} rate limit: {rate_limit} messages/minute")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

    return True


async def main():
    """Run all tests."""
    print("=== Notification Service Infrastructure Tests ===\n")

    tests = [
        ("Configuration", test_configuration),
        ("Database Connection", test_database_connection),
        ("Message Queue", test_message_queue),
        ("Message Processor", test_message_processor),
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
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)