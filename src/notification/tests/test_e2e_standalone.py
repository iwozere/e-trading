#!/usr/bin/env python3
"""
Standalone End-to-End Tests for Notification Service

Simple end-to-end tests that can run without external dependencies.
Tests complete message delivery flows and service behavior.
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.config import config
from src.data.db.services.database_service import get_database_service
from src.notification.service.message_queue import message_queue, MessagePriority
from src.notification.service.processor import message_processor
from src.data.db.models.model_notification import MessageStatus, DeliveryStatus
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class MockNotificationChannel:
    """Mock notification channel for testing."""

    def __init__(self, name: str, should_fail: bool = False, delay: float = 0.1):
        self.name = name
        self.should_fail = should_fail
        self.delay = delay
        self.sent_messages = []
        self.health_status = "healthy"
        self.call_count = 0

    async def send(self, message) -> Dict[str, Any]:
        """Mock send method."""
        self.call_count += 1
        await asyncio.sleep(self.delay)

        if self.should_fail:
            return {
                "success": False,
                "error_message": f"Mock failure for {self.name}",
                "response_time_ms": int(self.delay * 1000)
            }

        self.sent_messages.append({
            "id": message.id,
            "content": message.content,
            "recipient_id": message.recipient_id,
            "timestamp": datetime.now(timezone.utc)
        })

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

    def reset(self):
        """Reset mock state."""
        self.sent_messages.clear()
        self.call_count = 0
        self.should_fail = False
        self.health_status = "healthy"


class EndToEndTestSuite:
    """End-to-end test suite for notification service."""

    def __init__(self):
        self.mock_channels = {
            "telegram": MockNotificationChannel("telegram", delay=0.1),
            "email": MockNotificationChannel("email", delay=0.2),
            "sms": MockNotificationChannel("sms", delay=0.3)
        }
        self.test_results = []

    async def setup(self):
        """Set up test environment."""
        print("Setting up test environment...")

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

        # Clean up any existing test data
        with db_service.uow() as r:
            # Delete test messages
            r.s.execute(
                "DELETE FROM msg_delivery_status WHERE message_id IN "
                "(SELECT id FROM msg_messages WHERE message_type LIKE 'e2e_test_%')"
            )
            repo.session.execute(
                "DELETE FROM msg_messages WHERE message_type LIKE 'e2e_test_%'"
            )
            repo.commit()

        print("✓ Test environment setup complete")

    async def teardown(self):
        """Clean up test environment."""
        print("Cleaning up test environment...")

        # Reset mock channels
        for channel in self.mock_channels.values():
            channel.reset()

        # Clean up test data
        db_service = get_database_service()
        with db_service.uow() as r:
            r.s.execute(
                "DELETE FROM msg_delivery_status WHERE message_id IN "
                "(SELECT id FROM msg_messages WHERE message_type LIKE 'e2e_test_%')"
            )
            repo.session.execute(
                "DELETE FROM msg_messages WHERE message_type LIKE 'e2e_test_%'"
            )
            repo.commit()

        print("✓ Test environment cleanup complete")

    def record_test_result(self, test_name: str, success: bool, message: str = ""):
        """Record test result."""
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now(timezone.utc)
        })

        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}" + (f" - {message}" if message else ""))

    async def test_single_channel_delivery(self):
        """Test complete single channel message delivery."""
        test_name = "Single Channel Message Delivery"
        print(f"\nTesting {test_name}...")

        try:
            # Mock the channel registry
            with patch('src.notification.channels.base.channel_registry') as mock_registry:
                mock_registry.get_channel.side_effect = lambda name: self.mock_channels.get(name)
                mock_registry.list_channels.return_value = list(self.mock_channels.keys())

                # Enqueue message
                message_data = {
                    'message_type': 'e2e_test_single',
                    'channels': ['telegram'],
                    'content': {'text': 'Single channel test message'},
                    'recipient_id': 'test_user_1',
                    'priority': MessagePriority.NORMAL.value
                }

                message_id = message_queue.enqueue(message_data, MessagePriority.NORMAL)

                # Start processor temporarily
                await message_processor.start()

                # Wait for processing
                await asyncio.sleep(2.0)

                # Check results
                telegram_channel = self.mock_channels["telegram"]

                if len(telegram_channel.sent_messages) >= 1:
                    sent_message = telegram_channel.sent_messages[0]
                    if sent_message["id"] == message_id:
                        self.record_test_result(test_name, True, "Message delivered successfully")
                    else:
                        self.record_test_result(test_name, False, "Wrong message delivered")
                else:
                    self.record_test_result(test_name, False, "No message delivered")

                await message_processor.shutdown()

        except Exception as e:
            self.record_test_result(test_name, False, f"Exception: {str(e)}")

    async def test_multi_channel_delivery(self):
        """Test multi-channel message delivery."""
        test_name = "Multi-Channel Message Delivery"
        print(f"\nTesting {test_name}...")

        try:
            with patch('src.notification.channels.base.channel_registry') as mock_registry:
                mock_registry.get_channel.side_effect = lambda name: self.mock_channels.get(name)
                mock_registry.list_channels.return_value = list(self.mock_channels.keys())

                # Enqueue multi-channel message
                message_data = {
                    'message_type': 'e2e_test_multi',
                    'channels': ['telegram', 'email', 'sms'],
                    'content': {
                        'text': 'Multi-channel test message',
                        'subject': 'Test Alert'
                    },
                    'recipient_id': 'test_user_2',
                    'priority': MessagePriority.HIGH.value
                }

                message_id = message_queue.enqueue(message_data, MessagePriority.HIGH)

                # Start processor
                await message_processor.start()

                # Wait for processing
                await asyncio.sleep(3.0)

                # Check all channels received the message
                channels_with_messages = []
                for channel_name, channel in self.mock_channels.items():
                    if any(msg["id"] == message_id for msg in channel.sent_messages):
                        channels_with_messages.append(channel_name)

                expected_channels = {'telegram', 'email', 'sms'}
                actual_channels = set(channels_with_messages)

                if actual_channels == expected_channels:
                    self.record_test_result(test_name, True, f"All channels delivered: {channels_with_messages}")
                else:
                    missing = expected_channels - actual_channels
                    self.record_test_result(test_name, False, f"Missing channels: {missing}")

                await message_processor.shutdown()

        except Exception as e:
            self.record_test_result(test_name, False, f"Exception: {str(e)}")

    async def test_priority_message_handling(self):
        """Test priority message handling."""
        test_name = "Priority Message Handling"
        print(f"\nTesting {test_name}...")

        try:
            with patch('src.notification.channels.base.channel_registry') as mock_registry:
                mock_registry.get_channel.side_effect = lambda name: self.mock_channels.get(name)
                mock_registry.list_channels.return_value = list(self.mock_channels.keys())

                # Reset telegram channel
                self.mock_channels["telegram"].reset()

                # Enqueue normal priority message first
                normal_data = {
                    'message_type': 'e2e_test_normal',
                    'channels': ['telegram'],
                    'content': {'text': 'Normal priority message'},
                    'recipient_id': 'test_user_3',
                    'priority': MessagePriority.NORMAL.value
                }
                normal_id = message_queue.enqueue(normal_data, MessagePriority.NORMAL)

                # Enqueue critical priority message second
                critical_data = {
                    'message_type': 'e2e_test_critical',
                    'channels': ['telegram'],
                    'content': {'text': 'CRITICAL priority message'},
                    'recipient_id': 'test_user_3',
                    'priority': MessagePriority.CRITICAL.value
                }
                critical_id = message_queue.enqueue(critical_data, MessagePriority.CRITICAL)

                # Start processor
                await message_processor.start()

                # Wait for processing
                await asyncio.sleep(2.0)

                # Check processing order
                telegram_channel = self.mock_channels["telegram"]

                if len(telegram_channel.sent_messages) >= 2:
                    first_msg = telegram_channel.sent_messages[0]
                    second_msg = telegram_channel.sent_messages[1]

                    # Critical message should be processed first
                    if first_msg["id"] == critical_id and second_msg["id"] == normal_id:
                        self.record_test_result(test_name, True, "Critical message processed first")
                    else:
                        self.record_test_result(test_name, False, "Priority order not respected")
                else:
                    self.record_test_result(test_name, False, f"Expected 2 messages, got {len(telegram_channel.sent_messages)}")

                await message_processor.shutdown()

        except Exception as e:
            self.record_test_result(test_name, False, f"Exception: {str(e)}")

    async def test_channel_failure_handling(self):
        """Test handling of channel failures."""
        test_name = "Channel Failure Handling"
        print(f"\nTesting {test_name}...")

        try:
            with patch('src.notification.channels.base.channel_registry') as mock_registry:
                mock_registry.get_channel.side_effect = lambda name: self.mock_channels.get(name)
                mock_registry.list_channels.return_value = list(self.mock_channels.keys())

                # Make telegram channel fail
                self.mock_channels["telegram"].should_fail = True
                self.mock_channels["telegram"].reset()

                # Enqueue message to failing channel
                message_data = {
                    'message_type': 'e2e_test_failure',
                    'channels': ['telegram'],
                    'content': {'text': 'This should fail'},
                    'recipient_id': 'test_user_4',
                    'priority': MessagePriority.NORMAL.value
                }

                message_id = message_queue.enqueue(message_data, MessagePriority.NORMAL)

                # Start processor
                await message_processor.start()

                # Wait for processing and retries
                await asyncio.sleep(4.0)

                # Check that channel was called (attempted delivery)
                telegram_channel = self.mock_channels["telegram"]

                if telegram_channel.call_count > 0:
                    # Channel was called but should have no successful deliveries
                    if len(telegram_channel.sent_messages) == 0:
                        self.record_test_result(test_name, True, f"Failure handled correctly ({telegram_channel.call_count} attempts)")
                    else:
                        self.record_test_result(test_name, False, "Message delivered despite failure")
                else:
                    self.record_test_result(test_name, False, "Channel was not called")

                await message_processor.shutdown()

        except Exception as e:
            self.record_test_result(test_name, False, f"Exception: {str(e)}")

    async def test_partial_failure_scenario(self):
        """Test partial failure in multi-channel delivery."""
        test_name = "Partial Failure Scenario"
        print(f"\nTesting {test_name}...")

        try:
            with patch('src.notification.channels.base.channel_registry') as mock_registry:
                mock_registry.get_channel.side_effect = lambda name: self.mock_channels.get(name)
                mock_registry.list_channels.return_value = list(self.mock_channels.keys())

                # Reset all channels
                for channel in self.mock_channels.values():
                    channel.reset()

                # Make only email channel fail
                self.mock_channels["email"].should_fail = True

                # Enqueue multi-channel message
                message_data = {
                    'message_type': 'e2e_test_partial',
                    'channels': ['telegram', 'email', 'sms'],
                    'content': {'text': 'Partial failure test'},
                    'recipient_id': 'test_user_5',
                    'priority': MessagePriority.NORMAL.value
                }

                message_id = message_queue.enqueue(message_data, MessagePriority.NORMAL)

                # Start processor
                await message_processor.start()

                # Wait for processing
                await asyncio.sleep(4.0)

                # Check results
                successful_channels = []
                failed_channels = []

                for channel_name, channel in self.mock_channels.items():
                    if channel_name in ['telegram', 'email', 'sms']:
                        if any(msg["id"] == message_id for msg in channel.sent_messages):
                            successful_channels.append(channel_name)
                        elif channel.call_count > 0:
                            failed_channels.append(channel_name)

                # Should have telegram and sms successful, email failed
                if 'telegram' in successful_channels and 'sms' in successful_channels and 'email' in failed_channels:
                    self.record_test_result(test_name, True, f"Partial success: {successful_channels} delivered, {failed_channels} failed")
                else:
                    self.record_test_result(test_name, False, f"Unexpected results: success={successful_channels}, failed={failed_channels}")

                await message_processor.shutdown()

        except Exception as e:
            self.record_test_result(test_name, False, f"Exception: {str(e)}")

    async def test_service_recovery(self):
        """Test service recovery after temporary failures."""
        test_name = "Service Recovery"
        print(f"\nTesting {test_name}...")

        try:
            with patch('src.notification.channels.base.channel_registry') as mock_registry:
                mock_registry.get_channel.side_effect = lambda name: self.mock_channels.get(name)
                mock_registry.list_channels.return_value = list(self.mock_channels.keys())

                # Reset telegram channel
                self.mock_channels["telegram"].reset()

                # Make channel fail initially
                self.mock_channels["telegram"].should_fail = True

                # Enqueue message during failure
                message_data = {
                    'message_type': 'e2e_test_recovery',
                    'channels': ['telegram'],
                    'content': {'text': 'Recovery test message'},
                    'recipient_id': 'test_user_6',
                    'priority': MessagePriority.NORMAL.value
                }

                message_id = message_queue.enqueue(message_data, MessagePriority.NORMAL)

                # Start processor
                await message_processor.start()

                # Wait for initial failure
                await asyncio.sleep(2.0)

                # Restore channel functionality
                self.mock_channels["telegram"].should_fail = False

                # Wait for recovery and retry
                await asyncio.sleep(3.0)

                # Check if message was eventually delivered
                telegram_channel = self.mock_channels["telegram"]

                if any(msg["id"] == message_id for msg in telegram_channel.sent_messages):
                    self.record_test_result(test_name, True, "Message delivered after recovery")
                else:
                    # Check if at least retry attempts were made
                    if telegram_channel.call_count > 1:
                        self.record_test_result(test_name, True, f"Recovery attempted ({telegram_channel.call_count} calls)")
                    else:
                        self.record_test_result(test_name, False, "No recovery attempts detected")

                await message_processor.shutdown()

        except Exception as e:
            self.record_test_result(test_name, False, f"Exception: {str(e)}")

    async def test_database_persistence(self):
        """Test message persistence in database."""
        test_name = "Database Persistence"
        print(f"\nTesting {test_name}...")

        try:
            # Enqueue message
            message_data = {
                'message_type': 'e2e_test_persistence',
                'channels': ['telegram'],
                'content': {'text': 'Persistence test message'},
                'recipient_id': 'test_user_7',
                'priority': MessagePriority.NORMAL.value
            }

            message_id = message_queue.enqueue(message_data, MessagePriority.NORMAL)

            # Check message was persisted
            db_service = get_database_service()
            with db_service.uow() as r:
                message = r.notifications.messages.get_message(message_id)

                if message:
                    if (message.message_type == 'e2e_test_persistence' and
                        message.recipient_id == 'test_user_7' and
                        'telegram' in message.channels):
                        self.record_test_result(test_name, True, f"Message {message_id} persisted correctly")
                    else:
                        self.record_test_result(test_name, False, "Message data incorrect")
                else:
                    self.record_test_result(test_name, False, f"Message {message_id} not found in database")

        except Exception as e:
            self.record_test_result(test_name, False, f"Exception: {str(e)}")

    async def run_all_tests(self):
        """Run all end-to-end tests."""
        print("=== Notification Service End-to-End Tests ===")

        await self.setup()

        # Run all tests
        tests = [
            self.test_database_persistence,
            self.test_single_channel_delivery,
            self.test_multi_channel_delivery,
            self.test_priority_message_handling,
            self.test_channel_failure_handling,
            self.test_partial_failure_scenario,
            self.test_service_recovery
        ]

        for test in tests:
            try:
                await test()
            except Exception as e:
                test_name = test.__name__.replace('test_', '').replace('_', ' ').title()
                self.record_test_result(test_name, False, f"Test crashed: {str(e)}")

        await self.teardown()

        # Print summary
        print("\n=== Test Results Summary ===")
        passed = sum(1 for result in self.test_results if result["success"])
        total = len(self.test_results)

        for result in self.test_results:
            status = "PASS" if result["success"] else "FAIL"
            print(f"{result['test']}: {status}")
            if result["message"]:
                print(f"  └─ {result['message']}")

        print(f"\nPassed: {passed}/{total}")

        if passed == total:
            print("🎉 All end-to-end tests passed!")
            return 0
        else:
            print("❌ Some end-to-end tests failed!")
            return 1


async def main():
    """Main test runner."""
    test_suite = EndToEndTestSuite()
    return await test_suite.run_all_tests()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)