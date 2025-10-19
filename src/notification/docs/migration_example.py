"""
Migration Example: AsyncNotificationManager to NotificationServiceClient

This example shows how to migrate from AsyncNotificationManager to NotificationServiceClient.
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.client import (
    NotificationServiceClient, MessageType, MessagePriority,
    initialize_notification_client
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class LegacyService:
    """Example of a service using AsyncNotificationManager (OLD WAY)."""

    def __init__(self):
        # OLD: Import and initialize AsyncNotificationManager
        from src.notification.async_notification_manager import AsyncNotificationManager
        self.notification_manager = AsyncNotificationManager(
            telegram_token="your_token",
            telegram_chat_id="your_chat_id",
            email_api_key="your_api_key",
            email_sender="sender@example.com",
            email_receiver="receiver@example.com"
        )

    async def start(self):
        """Start the service."""
        # OLD: Start the notification manager
        await self.notification_manager.start()

    async def stop(self):
        """Stop the service."""
        # OLD: Stop the notification manager
        await self.notification_manager.stop()

    async def send_alert(self, symbol: str, price: float):
        """Send an alert notification."""
        # OLD: Use AsyncNotificationManager methods
        await self.notification_manager.send_notification(
            notification_type="alert",
            title=f"Price Alert: {symbol}",
            message=f"Price reached ${price:.2f}",
            priority="high",
            source="legacy_service"
        )

    async def send_trade_notification(self, symbol: str, side: str, price: float, quantity: float):
        """Send a trade notification."""
        # OLD: Use AsyncNotificationManager trade method
        await self.notification_manager.send_trade_notification(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity
        )

    async def send_error(self, error_message: str):
        """Send an error notification."""
        # OLD: Use AsyncNotificationManager error method
        await self.notification_manager.send_error_notification(
            error_message=error_message,
            source="legacy_service"
        )


class ModernService:
    """Example of a service using NotificationServiceClient (NEW WAY)."""

    def __init__(self, notification_client: NotificationServiceClient):
        # NEW: Accept NotificationServiceClient as dependency
        self.notification_client = notification_client

    async def start(self):
        """Start the service."""
        # NEW: No need to start client, it's managed externally
        _logger.info("Modern service started")

    async def stop(self):
        """Stop the service."""
        # NEW: No need to stop client, it's managed externally
        _logger.info("Modern service stopped")

    async def send_alert(self, symbol: str, price: float):
        """Send an alert notification."""
        # NEW: Use NotificationServiceClient with proper enums
        success = await self.notification_client.send_notification(
            notification_type=MessageType.ALERT,
            title=f"Price Alert: {symbol}",
            message=f"Price reached ${price:.2f}",
            priority=MessagePriority.HIGH,
            source="modern_service",
            channels=["telegram", "email"],
            recipient_id="trader_1"
        )

        if not success:
            _logger.warning("Failed to send alert notification")

    async def send_trade_notification(self, symbol: str, side: str, price: float, quantity: float):
        """Send a trade notification."""
        # NEW: Use NotificationServiceClient trade method (same interface)
        success = await self.notification_client.send_trade_notification(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            recipient_id="trader_1"
        )

        if not success:
            _logger.warning("Failed to send trade notification")

    async def send_error(self, error_message: str):
        """Send an error notification."""
        # NEW: Use NotificationServiceClient error method (same interface)
        success = await self.notification_client.send_error_notification(
            error_message=error_message,
            source="modern_service",
            recipient_id="admin"
        )

        if not success:
            _logger.warning("Failed to send error notification")


async def migration_example():
    """Example showing the migration process."""

    print("=== Migration Example: AsyncNotificationManager -> NotificationServiceClient ===\n")

    # Step 1: Initialize the new notification service client
    print("1. Initializing NotificationServiceClient...")
    notification_client = await initialize_notification_client(
        service_url="http://localhost:8000",
        timeout=30,
        max_retries=3
    )
    print("   ✓ NotificationServiceClient initialized\n")

    # Step 2: Create services with new client
    print("2. Creating modern service with NotificationServiceClient...")
    modern_service = ModernService(notification_client)
    await modern_service.start()
    print("   ✓ Modern service created and started\n")

    # Step 3: Test notifications
    print("3. Testing notifications...")

    # Test alert notification
    print("   Sending alert notification...")
    await modern_service.send_alert("BTCUSDT", 50000.0)
    print("   ✓ Alert sent")

    # Test trade notification
    print("   Sending trade notification...")
    await modern_service.send_trade_notification("BTCUSDT", "BUY", 50000.0, 0.1)
    print("   ✓ Trade notification sent")

    # Test error notification
    print("   Sending error notification...")
    await modern_service.send_error("Test error message")
    print("   ✓ Error notification sent\n")

    # Step 4: Check service health
    print("4. Checking notification service health...")
    health = await notification_client.get_health_status()
    print(f"   Service status: {health.get('status', 'unknown')}")
    print(f"   Service version: {health.get('version', 'unknown')}\n")

    # Step 5: Cleanup
    print("5. Cleaning up...")
    await modern_service.stop()
    await notification_client.close()
    print("   ✓ Services stopped and cleaned up\n")

    print("=== Migration Example Complete ===")


def print_migration_checklist():
    """Print a checklist for migrating services."""

    print("""
=== Migration Checklist ===

For each service that uses AsyncNotificationManager:

□ 1. Update imports:
   - Remove: from src.notification.async_notification_manager import AsyncNotificationManager
   - Add: from src.notification.service.client import NotificationServiceClient, MessageType, MessagePriority

□ 2. Update constructor:
   - Remove: AsyncNotificationManager initialization
   - Add: NotificationServiceClient as constructor parameter

□ 3. Update start/stop methods:
   - Remove: await self.notification_manager.start()
   - Remove: await self.notification_manager.stop()
   - Note: Client lifecycle is managed externally

□ 4. Update notification calls:
   - Replace string types with MessageType enums
   - Replace string priorities with MessagePriority enums
   - Add recipient_id parameter for user targeting
   - Add channels parameter for channel selection
   - Check return value (bool) for success/failure

□ 5. Update service initialization:
   - Initialize NotificationServiceClient at application startup
   - Pass client instance to services that need notifications
   - Ensure proper cleanup on application shutdown

□ 6. Test migration:
   - Verify notifications are sent correctly
   - Check delivery status through service API
   - Validate error handling and fallbacks

□ 7. Remove legacy dependencies:
   - Remove AsyncNotificationManager imports
   - Clean up unused notification code
   - Update documentation and examples

=== Key Benefits of Migration ===

✓ Decoupled notification logic from business services
✓ Centralized notification management and monitoring
✓ Better error handling and retry mechanisms
✓ Comprehensive delivery tracking and analytics
✓ Plugin-based channel architecture for extensibility
✓ Rate limiting and priority-based processing
✓ Database-backed persistence and reliability

""")


if __name__ == "__main__":
    print_migration_checklist()
    print("\n" + "="*60 + "\n")

    # Run the migration example (requires notification service to be running)
    try:
        asyncio.run(migration_example())
    except Exception as e:
        print(f"Migration example failed (service may not be running): {e}")
        print("To run this example, start the notification service first:")
        print("  python src/notification/service/main.py")