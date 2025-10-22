"""
Example usage of the NotificationServiceClient.

This script demonstrates how to use the new notification service client
to send various types of notifications.
"""

import asyncio
from datetime import datetime, timezone

from src.notification.service.client import NotificationServiceClient, NotificationServiceError
from src.notification.model import NotificationType, NotificationPriority


async def main():
    """Main example function."""
    # Initialize the client
    client = NotificationServiceClient(
        base_url="http://localhost:8080",  # Notification service URL
        timeout=30,
        max_retries=3
    )

    try:
        # Example 1: Basic notification
        print("Sending basic notification...")
        response = await client.send_notification_async(
            notification_type=NotificationType.INFO,
            title="System Alert",
            message="Trading system is now online and ready for trading.",
            priority=NotificationPriority.NORMAL,
            channels=["telegram", "email"]
        )
        print(f"✓ Notification sent with ID: {response.message_id}")

        # Example 2: Trade notification
        print("\nSending trade notification...")
        response = await client.send_trade_notification_async(
            symbol="BTCUSDT",
            side="BUY",
            price=45000.0,
            quantity=0.1,
            pnl=None  # Entry trade, no PnL yet
        )
        print(f"✓ Trade notification sent with ID: {response.message_id}")

        # Example 3: Error notification
        print("\nSending error notification...")
        response = await client.send_error_notification_async(
            error_message="Failed to connect to exchange API. Retrying in 30 seconds.",
            source="trading_engine"
        )
        print(f"✓ Error notification sent with ID: {response.message_id}")

        # Example 4: Notification with attachments
        print("\nSending notification with attachments...")
        attachments = {
            "chart.png": b"fake_chart_data_here",  # In real usage, this would be actual image data
            "report.txt": "Trading report content"
        }
        response = await client.send_notification_async(
            notification_type=NotificationType.PERFORMANCE,
            title="Daily Trading Report",
            message="Please find attached the daily trading performance report.",
            channels=["email"],  # Email supports attachments better
            attachments=attachments
        )
        print(f"✓ Notification with attachments sent with ID: {response.message_id}")

        # Example 5: Telegram-specific notification
        print("\nSending Telegram-specific notification...")
        response = await client.send_notification_async(
            notification_type=NotificationType.TRADE_EXIT,
            title="Trade Closed",
            message="Position closed with 5.2% profit!",
            channels=["telegram"],
            metadata={
                "telegram_chat_id": "123456789",  # Specific chat ID
                "reply_to_message_id": 12345      # Reply to a specific message
            }
        )
        print(f"✓ Telegram notification sent with ID: {response.message_id}")

        # Example 6: Check message status
        print(f"\nChecking status of message {response.message_id}...")
        status = await client.get_message_status_async(response.message_id)
        print(f"✓ Message status: {status['status']}")

        # Example 7: Health check
        print("\nPerforming health check...")
        health = await client.health_check_async()
        print(f"✓ Service health: {health['status']}")

        # Example 8: Scheduled notification
        print("\nSending scheduled notification...")
        from datetime import timedelta
        scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=5)

        request = {
            "message_type": "system",
            "priority": "NORMAL",
            "channels": ["telegram"],
            "content": {
                "text": "This is a scheduled notification that will be sent in 5 minutes.",
                "subject": "Scheduled Alert"
            },
            "scheduled_for": scheduled_time.isoformat()
        }

        # For scheduled notifications, we need to use the raw API
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{client.base_url}/api/v1/messages",
                json=request
            ) as response:
                response.raise_for_status()
                data = await response.json()
                print(f"✓ Scheduled notification created with ID: {data['message_id']}")

    except NotificationServiceError as e:
        print(f"✗ Notification service error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    finally:
        # Clean up
        await client.close_async()


def sync_example():
    """Example using synchronous methods."""
    print("\n" + "="*50)
    print("SYNCHRONOUS EXAMPLES")
    print("="*50)

    # Initialize the client
    client = NotificationServiceClient(
        base_url="http://localhost:8080"
    )

    try:
        # Synchronous notification
        print("Sending synchronous notification...")
        response = client.send_notification(
            notification_type=NotificationType.WARNING,
            title="Sync Alert",
            message="This notification was sent synchronously.",
            priority=NotificationPriority.HIGH
        )
        print(f"✓ Sync notification sent with ID: {response.message_id}")

        # Synchronous trade notification
        print("\nSending synchronous trade notification...")
        response = client.send_trade_notification(
            symbol="ETHUSDT",
            side="SELL",
            price=3200.0,
            quantity=2.0,
            entry_price=3000.0,
            pnl=6.67,
            exit_type="TP"
        )
        print(f"✓ Sync trade notification sent with ID: {response.message_id}")

        # Check status synchronously
        print(f"\nChecking message status synchronously...")
        status = client.get_message_status(response.message_id)
        print(f"✓ Message status: {status['status']}")

    except NotificationServiceError as e:
        print(f"✗ Notification service error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    finally:
        # Clean up
        client.close()


def backward_compatibility_example():
    """Example showing backward compatibility with AsyncNotificationManager."""
    print("\n" + "="*50)
    print("BACKWARD COMPATIBILITY EXAMPLES")
    print("="*50)

    from src.notification.compatibility import AsyncNotificationManagerCompat

    async def compat_example():
        # Initialize compatibility wrapper
        manager = AsyncNotificationManagerCompat(
            telegram_chat_id="123456789",
            email_receiver="trader@example.com",
            notification_service_url="http://localhost:8080"
        )

        await manager.start()

        try:
            # Use old AsyncNotificationManager interface
            print("Sending notification using old interface...")
            success = await manager.send_notification(
                notification_type=NotificationType.INFO,
                title="Compatibility Test",
                message="This uses the old AsyncNotificationManager interface!",
                priority=NotificationPriority.NORMAL
            )
            print(f"✓ Old interface notification sent: {success}")

            # Trade notification with old interface
            print("\nSending trade notification using old interface...")
            success = await manager.send_trade_notification(
                symbol="ADAUSDT",
                side="BUY",
                price=0.45,
                quantity=1000.0
            )
            print(f"✓ Old interface trade notification sent: {success}")

            # Get stats (old interface)
            stats = manager.get_stats()
            print(f"\nStats: {stats}")

        finally:
            await manager.stop()

    # Run the compatibility example
    asyncio.run(compat_example())


if __name__ == "__main__":
    print("="*50)
    print("NOTIFICATION SERVICE CLIENT EXAMPLES")
    print("="*50)

    # Run async examples
    print("ASYNCHRONOUS EXAMPLES")
    print("="*50)
    asyncio.run(main())

    # Run sync examples
    sync_example()

    # Run backward compatibility examples
    backward_compatibility_example()

    print("\n" + "="*50)
    print("ALL EXAMPLES COMPLETED")
    print("="*50)