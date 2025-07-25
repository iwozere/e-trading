#!/usr/bin/env python3
"""
Async Notification System Example
=================================

This example demonstrates the comprehensive async notification system including:
- Async notification manager with queuing
- Batching and rate limiting
- Retry mechanisms with exponential backoff
- Multiple notification channels (Telegram, Email)
- Performance monitoring and metrics
- Error handling and recovery
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.notification.async_notification_manager import AsyncNotificationManager
from src.notification.telegram_notifier import TelegramNotifier
from src.notification.emailer import Emailer
from src.notification.logger import Logger


class MockTelegramNotifier:
    """Mock Telegram notifier for demonstration"""

    def __init__(self, bot_token: str = "mock_token", chat_id: str = "mock_chat"):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.sent_messages = []
        self.failures = 0

    async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Mock send message with occasional failures"""
        await asyncio.sleep(0.1)  # Simulate network delay

        # Simulate occasional failures (10% failure rate)
        if random.random() < 0.1:
            self.failures += 1
            raise Exception("Mock Telegram API error")

        self.sent_messages.append({
            "message": message,
            "parse_mode": parse_mode,
            "timestamp": datetime.now().isoformat()
        })
        return True

    async def send_photo(self, photo_path: str, caption: str = "") -> bool:
        """Mock send photo"""
        await asyncio.sleep(0.2)  # Simulate file upload delay

        if random.random() < 0.15:  # 15% failure rate for photos
            self.failures += 1
            raise Exception("Mock photo upload error")

        self.sent_messages.append({
            "photo": photo_path,
            "caption": caption,
            "timestamp": datetime.now().isoformat()
        })
        return True


class MockEmailer:
    """Mock emailer for demonstration"""

    def __init__(self, smtp_server: str = "mock.smtp.com", port: int = 587):
        self.smtp_server = smtp_server
        self.port = port
        self.sent_emails = []
        self.failures = 0

    async def send_email(self, to_email: str, subject: str, body: str,
                        html_body: str = None) -> bool:
        """Mock send email with occasional failures"""
        await asyncio.sleep(0.15)  # Simulate SMTP delay

        # Simulate occasional failures (5% failure rate)
        if random.random() < 0.05:
            self.failures += 1
            raise Exception("Mock SMTP error")

        self.sent_emails.append({
            "to": to_email,
            "subject": subject,
            "body": body,
            "html_body": html_body,
            "timestamp": datetime.now().isoformat()
        })
        return True


def create_sample_notifications() -> List[Dict[str, Any]]:
    """Create sample notifications for demonstration"""

    notifications = [
        {
            "type": "trade_entry",
            "priority": "high",
            "title": "🚀 Trade Entry Alert",
            "message": "BTCUSDT: BUY order executed at $45,000\nQuantity: 0.1 BTC\nStop Loss: $44,000\nTake Profit: $46,000",
            "data": {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "price": 45000.0,
                "quantity": 0.1,
                "stop_loss": 44000.0,
                "take_profit": 46000.0
            }
        },
        {
            "type": "trade_exit",
            "priority": "medium",
            "title": "💰 Trade Exit Alert",
            "message": "BTCUSDT: SELL order executed at $46,500\nProfit: $150.00 (+3.33%)\nDuration: 2h 15m",
            "data": {
                "symbol": "BTCUSDT",
                "side": "SELL",
                "price": 46500.0,
                "profit": 150.0,
                "profit_pct": 3.33,
                "duration": "2h 15m"
            }
        },
        {
            "type": "system_status",
            "priority": "low",
            "title": "📊 System Status Update",
            "message": "Trading bot is running normally\nActive trades: 3\nDaily P&L: +$250.00\nUptime: 99.8%",
            "data": {
                "active_trades": 3,
                "daily_pnl": 250.0,
                "uptime": 99.8
            }
        },
        {
            "type": "error",
            "priority": "high",
            "title": "⚠️ System Error Alert",
            "message": "API connection timeout\nAttempting to reconnect...\nLast successful connection: 5 minutes ago",
            "data": {
                "error_type": "api_timeout",
                "last_success": "5 minutes ago",
                "retry_count": 3
            }
        },
        {
            "type": "performance_alert",
            "priority": "medium",
            "title": "📈 Performance Alert",
            "message": "Strategy performance below threshold\nWin Rate: 45% (Target: 50%)\nSharpe Ratio: 0.8 (Target: 1.0)",
            "data": {
                "win_rate": 45.0,
                "target_win_rate": 50.0,
                "sharpe_ratio": 0.8,
                "target_sharpe": 1.0
            }
        }
    ]

    return notifications


async def demonstrate_basic_notifications():
    """Demonstrate basic notification sending"""

    print("=" * 60)
    print("BASIC NOTIFICATION SYSTEM")
    print("=" * 60)

    # Create notification manager
    notification_manager = AsyncNotificationManager(
        max_queue_size=1000,
        batch_size=10,
        batch_timeout=5.0,
        rate_limit_per_minute=60,
        max_retries=3,
        retry_delay=1.0
    )

    # Create mock notifiers
    telegram_notifier = MockTelegramNotifier()
    email_notifier = MockEmailer()

    # Register notifiers
    notification_manager.register_notifier("telegram", telegram_notifier)
    notification_manager.register_notifier("email", email_notifier)

    # Start the notification manager
    await notification_manager.start()

    print("📱 Notification Manager Started")
    print(f"   Queue Size: {notification_manager.max_queue_size}")
    print(f"   Batch Size: {notification_manager.batch_size}")
    print(f"   Rate Limit: {notification_manager.rate_limit_per_minute}/min")
    print(f"   Max Retries: {notification_manager.max_retries}")
    print()

    # Send sample notifications
    notifications = create_sample_notifications()

    print("📤 Sending Sample Notifications...")
    for i, notification in enumerate(notifications, 1):
        await notification_manager.send_notification(
            notification_type=notification["type"],
            title=notification["title"],
            message=notification["message"],
            priority=notification["priority"],
            channels=["telegram", "email"],
            data=notification["data"]
        )
        print(f"   {i}. {notification['title']}")

    # Wait for processing
    await asyncio.sleep(2)

    # Get statistics
    stats = notification_manager.get_statistics()

    print(f"\n📊 Notification Statistics:")
    print(f"   Total Sent: {stats['total_sent']}")
    print(f"   Total Failed: {stats['total_failed']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Average Processing Time: {stats['avg_processing_time']:.2f}s")
    print(f"   Queue Size: {stats['current_queue_size']}")

    # Stop the manager
    await notification_manager.stop()

    return notification_manager, telegram_notifier, email_notifier


async def demonstrate_batching_and_rate_limiting():
    """Demonstrate batching and rate limiting features"""

    print("\n" + "=" * 60)
    print("BATCHING & RATE LIMITING")
    print("=" * 60)

    # Create notification manager with aggressive batching
    notification_manager = AsyncNotificationManager(
        max_queue_size=500,
        batch_size=5,  # Small batch size
        batch_timeout=2.0,  # Short timeout
        rate_limit_per_minute=30,  # Lower rate limit
        max_retries=2,
        retry_delay=0.5
    )

    telegram_notifier = MockTelegramNotifier()
    notification_manager.register_notifier("telegram", telegram_notifier)

    await notification_manager.start()

    print("📦 Sending 20 notifications with batching...")
    start_time = time.time()

    # Send many notifications quickly
    tasks = []
    for i in range(20):
        task = notification_manager.send_notification(
            notification_type="test",
            title=f"Test Notification {i+1}",
            message=f"This is test notification number {i+1}",
            priority="low",
            channels=["telegram"]
        )
        tasks.append(task)

    # Wait for all notifications to be sent
    await asyncio.gather(*tasks)

    # Wait for processing
    await asyncio.sleep(3)

    end_time = time.time()
    total_time = end_time - start_time

    stats = notification_manager.get_statistics()

    print(f"⏱️ Performance Results:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Notifications Sent: {stats['total_sent']}")
    print(f"   Average Time per Notification: {total_time/20:.3f}s")
    print(f"   Batches Processed: {stats.get('batches_processed', 0)}")
    print(f"   Rate Limited: {stats.get('rate_limited', 0)}")

    await notification_manager.stop()


async def demonstrate_retry_mechanism():
    """Demonstrate retry mechanism with failures"""

    print("\n" + "=" * 60)
    print("RETRY MECHANISM")
    print("=" * 60)

    # Create notification manager with retry settings
    notification_manager = AsyncNotificationManager(
        max_queue_size=100,
        batch_size=1,  # Process one at a time to see retries
        batch_timeout=1.0,
        rate_limit_per_minute=100,
        max_retries=3,
        retry_delay=0.5
    )

    # Create notifier with high failure rate
    class HighFailureNotifier(MockTelegramNotifier):
        async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
            await asyncio.sleep(0.1)
            # 50% failure rate to demonstrate retries
            if random.random() < 0.5:
                self.failures += 1
                raise Exception("Simulated failure for retry demonstration")

            self.sent_messages.append({
                "message": message,
                "parse_mode": parse_mode,
                "timestamp": datetime.now().isoformat()
            })
            return True

    high_failure_notifier = HighFailureNotifier()
    notification_manager.register_notifier("telegram", high_failure_notifier)

    await notification_manager.start()

    print("🔄 Sending notifications with high failure rate...")

    # Send notifications that will likely fail and retry
    for i in range(5):
        await notification_manager.send_notification(
            notification_type="retry_test",
            title=f"Retry Test {i+1}",
            message=f"This notification may fail and retry (attempt {i+1})",
            priority="medium",
            channels=["telegram"]
        )

    # Wait for processing and retries
    await asyncio.sleep(5)

    stats = notification_manager.get_statistics()

    print(f"📊 Retry Statistics:")
    print(f"   Total Sent: {stats['total_sent']}")
    print(f"   Total Failed: {stats['total_failed']}")
    print(f"   Retries Attempted: {stats.get('retries_attempted', 0)}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Notifier Failures: {high_failure_notifier.failures}")

    await notification_manager.stop()


async def demonstrate_priority_handling():
    """Demonstrate priority-based notification handling"""

    print("\n" + "=" * 60)
    print("PRIORITY HANDLING")
    print("=" * 60)

    notification_manager = AsyncNotificationManager(
        max_queue_size=100,
        batch_size=3,
        batch_timeout=2.0,
        rate_limit_per_minute=60,
        max_retries=2,
        retry_delay=0.5
    )

    telegram_notifier = MockTelegramNotifier()
    notification_manager.register_notifier("telegram", telegram_notifier)

    await notification_manager.start()

    print("🎯 Sending notifications with different priorities...")

    # Send notifications in reverse priority order
    priorities = ["low", "medium", "high"]
    for priority in priorities:
        for i in range(2):
            await notification_manager.send_notification(
                notification_type="priority_test",
                title=f"{priority.upper()} Priority Test {i+1}",
                message=f"This is a {priority} priority notification",
                priority=priority,
                channels=["telegram"]
            )
            print(f"   Queued: {priority} priority notification {i+1}")

    # Wait for processing
    await asyncio.sleep(3)

    print(f"\n📋 Processing Order (should be high -> medium -> low):")
    for i, msg in enumerate(telegram_notifier.sent_messages, 1):
        priority = "high" if "HIGH" in msg["message"] else "medium" if "MEDIUM" in msg["message"] else "low"
        print(f"   {i}. {priority.upper()}: {msg['message'][:50]}...")

    await notification_manager.stop()


async def demonstrate_channel_selection():
    """Demonstrate selective channel notification"""

    print("\n" + "=" * 60)
    print("CHANNEL SELECTION")
    print("=" * 60)

    notification_manager = AsyncNotificationManager(
        max_queue_size=100,
        batch_size=5,
        batch_timeout=2.0,
        rate_limit_per_minute=60,
        max_retries=2,
        retry_delay=0.5
    )

    telegram_notifier = MockTelegramNotifier()
    email_notifier = MockEmailer()

    notification_manager.register_notifier("telegram", telegram_notifier)
    notification_manager.register_notifier("email", email_notifier)

    await notification_manager.start()

    print("📡 Sending notifications to different channels...")

    # Send to telegram only
    await notification_manager.send_notification(
        notification_type="telegram_only",
        title="Telegram Only",
        message="This notification goes to Telegram only",
        priority="medium",
        channels=["telegram"]
    )

    # Send to email only
    await notification_manager.send_notification(
        notification_type="email_only",
        title="Email Only",
        message="This notification goes to Email only",
        priority="medium",
        channels=["email"]
    )

    # Send to both channels
    await notification_manager.send_notification(
        notification_type="both_channels",
        title="Both Channels",
        message="This notification goes to both Telegram and Email",
        priority="high",
        channels=["telegram", "email"]
    )

    # Wait for processing
    await asyncio.sleep(2)

    print(f"\n📊 Channel Statistics:")
    print(f"   Telegram Messages: {len(telegram_notifier.sent_messages)}")
    print(f"   Email Messages: {len(email_notifier.sent_emails)}")

    print(f"\n📱 Telegram Messages:")
    for i, msg in enumerate(telegram_notifier.sent_messages, 1):
        print(f"   {i}. {msg['message'][:40]}...")

    print(f"\n📧 Email Messages:")
    for i, email in enumerate(email_notifier.sent_emails, 1):
        print(f"   {i}. To: {email['to']} - Subject: {email['subject']}")

    await notification_manager.stop()


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring and metrics"""

    print("\n" + "=" * 60)
    print("PERFORMANCE MONITORING")
    print("=" * 60)

    notification_manager = AsyncNotificationManager(
        max_queue_size=200,
        batch_size=10,
        batch_timeout=3.0,
        rate_limit_per_minute=100,
        max_retries=3,
        retry_delay=1.0
    )

    telegram_notifier = MockTelegramNotifier()
    notification_manager.register_notifier("telegram", telegram_notifier)

    await notification_manager.start()

    print("📈 Performance monitoring demonstration...")

    # Send notifications in bursts to test performance
    for burst in range(3):
        print(f"   Burst {burst + 1}: Sending 15 notifications...")

        tasks = []
        for i in range(15):
            task = notification_manager.send_notification(
                notification_type="performance_test",
                title=f"Performance Test {burst*15 + i + 1}",
                message=f"Performance monitoring test notification {burst*15 + i + 1}",
                priority="medium",
                channels=["telegram"]
            )
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Get intermediate stats
        stats = notification_manager.get_statistics()
        print(f"      Queue size: {stats['current_queue_size']}")
        print(f"      Success rate: {stats['success_rate']:.1f}%")

        await asyncio.sleep(1)

    # Final statistics
    final_stats = notification_manager.get_statistics()

    print(f"\n📊 Final Performance Statistics:")
    print(f"   Total Notifications: {final_stats['total_sent'] + final_stats['total_failed']}")
    print(f"   Successfully Sent: {final_stats['total_sent']}")
    print(f"   Failed: {final_stats['total_failed']}")
    print(f"   Success Rate: {final_stats['success_rate']:.1f}%")
    print(f"   Average Processing Time: {final_stats['avg_processing_time']:.3f}s")
    print(f"   Peak Queue Size: {final_stats.get('peak_queue_size', 0)}")
    print(f"   Batches Processed: {final_stats.get('batches_processed', 0)}")
    print(f"   Rate Limited Events: {final_stats.get('rate_limited', 0)}")

    await notification_manager.stop()


async def demonstrate_error_handling():
    """Demonstrate error handling and recovery"""

    print("\n" + "=" * 60)
    print("ERROR HANDLING & RECOVERY")
    print("=" * 60)

    notification_manager = AsyncNotificationManager(
        max_queue_size=50,
        batch_size=2,
        batch_timeout=1.0,
        rate_limit_per_minute=30,
        max_retries=2,
        retry_delay=0.5
    )

    # Create notifier that fails completely after a few messages
    class FailingNotifier(MockTelegramNotifier):
        def __init__(self):
            super().__init__()
            self.message_count = 0

        async def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
            self.message_count += 1

            # Fail completely after 3 messages
            if self.message_count > 3:
                raise Exception("Notifier has failed completely")

            await asyncio.sleep(0.1)
            self.sent_messages.append({
                "message": message,
                "parse_mode": parse_mode,
                "timestamp": datetime.now().isoformat()
            })
            return True

    failing_notifier = FailingNotifier()
    notification_manager.register_notifier("telegram", failing_notifier)

    await notification_manager.start()

    print("⚠️ Testing error handling with failing notifier...")

    # Send notifications that will eventually fail
    for i in range(10):
        await notification_manager.send_notification(
            notification_type="error_test",
            title=f"Error Test {i+1}",
            message=f"This notification may fail due to notifier issues",
            priority="medium",
            channels=["telegram"]
        )

    # Wait for processing and failures
    await asyncio.sleep(5)

    stats = notification_manager.get_statistics()

    print(f"\n📊 Error Handling Results:")
    print(f"   Total Attempted: {stats['total_sent'] + stats['total_failed']}")
    print(f"   Successfully Sent: {stats['total_sent']}")
    print(f"   Failed: {stats['total_failed']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    print(f"   Retries Attempted: {stats.get('retries_attempted', 0)}")
    print(f"   Notifier Messages Sent: {failing_notifier.message_count}")

    await notification_manager.stop()


async def main():
    """Main function to run all demonstrations"""

    print("🚀 ASYNC NOTIFICATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This example demonstrates the comprehensive async notification system")
    print("including queuing, batching, rate limiting, retry mechanisms, and more.")
    print()

    try:
        # Run all demonstrations
        await demonstrate_basic_notifications()
        await demonstrate_batching_and_rate_limiting()
        await demonstrate_retry_mechanism()
        await demonstrate_priority_handling()
        await demonstrate_channel_selection()
        await demonstrate_performance_monitoring()
        await demonstrate_error_handling()

        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The async notification system provides robust, scalable")
        print("notification capabilities with advanced features like:")
        print("• Queuing and batching for performance")
        print("• Rate limiting to prevent API abuse")
        print("• Retry mechanisms with exponential backoff")
        print("• Priority-based processing")
        print("• Multi-channel support")
        print("• Comprehensive error handling")
        print("• Performance monitoring and metrics")

    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        print("💡 Make sure all required dependencies are installed:")
        print("   pip install asyncio aiohttp")


if __name__ == "__main__":
    asyncio.run(main())
