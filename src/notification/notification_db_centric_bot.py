"""
Notification Service Main Application - Database-Centric Architecture

Pure message delivery engine without REST endpoints.
Polls database for messages and delivers them through channel plugins.
All client interactions are handled through the Main API Service.
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.config import config
from src.data.db.services.database_service import get_database_service
from src.data.db.services.notification_service import NotificationService
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Global services
message_poller = None
message_processor = None
health_reporter = None


class MessagePoller:
    """
    Database message poller for continuous message processing.

    Polls the database for pending messages and coordinates with
    the message processor for delivery.
    """

    def __init__(self, poll_interval_seconds: int = 5):
        """
        Initialize the message poller.

        Args:
            poll_interval_seconds: Seconds between database polls
        """
        self.poll_interval_seconds = poll_interval_seconds
        self.running = False
        self.instance_id = f"notification_service_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._logger = setup_logger(f"{__name__}.MessagePoller")

    async def start(self):
        """Start the message poller."""
        if self.running:
            self._logger.warning("Message poller is already running")
            return

        self.running = True
        self._logger.info("Starting message poller with instance ID: %s", self.instance_id)

        # Start polling loop
        asyncio.create_task(self._polling_loop())

    async def stop(self):
        """Stop the message poller."""
        self.running = False
        self._logger.info("Message poller stopped")

    async def _polling_loop(self):
        """Main polling loop for database messages."""
        while self.running:
            try:
                # Poll for pending messages
                messages = await self._poll_pending_messages()

                if messages:
                    self._logger.info("Found %s pending messages", len(messages))

                    # Process messages through the message processor
                    if message_processor:
                        await self._process_messages(messages)

                # Wait before next poll
                await asyncio.sleep(self.poll_interval_seconds)

            except Exception as e:
                self._logger.exception("Error in polling loop:")
                # Back off on error
                await asyncio.sleep(self.poll_interval_seconds * 2)

    async def _poll_pending_messages(self):
        """
        Poll database for pending messages with distributed locking.

        Returns:
            List of messages claimed for processing
        """
        try:
            db_service = get_database_service()

            with db_service.uow() as uow:
                # Use repository method for atomic locking (this is acceptable at this level)
                messages = uow.notifications.messages.get_pending_messages_with_lock(
                    limit=10,
                    lock_instance_id=self.instance_id
                )

                # No manual commit needed - UoW auto-commits on successful exit
                return messages

        except Exception as e:
            self._logger.exception("Error polling pending messages:")
            return []

    async def _process_messages(self, messages):
        """
        Process messages through the message processor.

        Args:
            messages: List of messages to process
        """
        try:
            # Process each message using the database-centric method
            for message in messages:
                if hasattr(message_processor, 'process_database_message'):
                    result = await message_processor.process_database_message(message)
                    self._logger.debug(
                        "Processed message %s: success=%s",
                        message.id, result.success
                    )
                else:
                    self._logger.error("Message processor does not support database messages")

        except Exception as e:
            self._logger.exception("Error processing messages:")


class HealthReporter:
    """
    Database-only health reporter for notification service.

    Reports service and channel health status to database
    without exposing REST endpoints.
    """

    def __init__(self, report_interval_seconds: int = 60):
        """
        Initialize the health reporter.

        Args:
            report_interval_seconds: Seconds between health reports
        """
        self.report_interval_seconds = report_interval_seconds
        self.running = False
        self._logger = setup_logger(f"{__name__}.HealthReporter")

    async def start(self):
        """Start the health reporter."""
        if self.running:
            self._logger.warning("Health reporter is already running")
            return

        self.running = True
        self._logger.info("Starting health reporter")

        # Start reporting loop
        asyncio.create_task(self._reporting_loop())

    async def stop(self):
        """Stop the health reporter."""
        self.running = False
        self._logger.info("Health reporter stopped")

    async def _reporting_loop(self):
        """Main health reporting loop."""
        while self.running:
            try:
                # Report service health
                await self._report_service_health()

                # Report channel health
                await self._report_channel_health()

                # Wait before next report
                await asyncio.sleep(self.report_interval_seconds)

            except Exception as e:
                self._logger.exception("Error in health reporting loop:")
                await asyncio.sleep(self.report_interval_seconds)

    async def _report_service_health(self):
        """Report notification service health to database."""
        try:
            from src.data.db.services.system_health_service import SystemHealthService
            from src.data.db.models.model_system_health import SystemHealthStatus

            # Initialize health service (no arguments needed - uses @with_uow decorator)
            health_service = SystemHealthService()

            # Determine service health status
            status = SystemHealthStatus.HEALTHY
            error_message = None
            metadata = {
                'service': config.service_name,
                'version': config.version,
                'message_processor_running': message_processor.is_running if message_processor else False,
                'message_poller_running': message_poller.running if message_poller else False
            }

            if not (message_processor and message_processor.is_running):
                status = SystemHealthStatus.DOWN
                error_message = 'Message processor not running'

            # Update service health (method manages its own UoW context)
            health_service.update_system_health(
                system='notification',
                component=None,
                status=status,
                error_message=error_message,
                metadata=metadata
            )

        except Exception as e:
            self._logger.exception("Error reporting service health:")

    async def _report_channel_health(self):
        """Report channel health to database."""
        try:
            if not message_processor or not hasattr(message_processor, '_channel_instances'):
                return

            from src.data.db.services.system_health_service import SystemHealthService
            from src.data.db.models.model_system_health import SystemHealthStatus

            # Initialize health service (no arguments needed - uses @with_uow decorator)
            health_service = SystemHealthService()

            # Report health for each channel
            for channel_name, channel_instance in message_processor._channel_instances.items():
                try:
                    # Perform channel health check
                    if hasattr(channel_instance, 'health_check'):
                        health_result = await channel_instance.health_check()

                        status = SystemHealthStatus.HEALTHY if health_result.get('healthy', False) else SystemHealthStatus.DOWN
                        error_message = health_result.get('error')
                        response_time_ms = health_result.get('response_time_ms')
                    else:
                        # Default to healthy if no health check method
                        status = SystemHealthStatus.HEALTHY
                        error_message = None
                        response_time_ms = None

                    # Update channel health (method manages its own UoW context)
                    health_service.update_notification_channel_health(
                        channel=channel_name,
                        status=status,
                        response_time_ms=response_time_ms,
                        error_message=error_message,
                        metadata={'last_check': datetime.now(timezone.utc).isoformat()}
                    )

                except Exception as e:
                    self._logger.error("Error checking health for channel %s: %s", channel_name, e)

                    # Report channel as down
                    health_service.update_notification_channel_health(
                        channel=channel_name,
                        status=SystemHealthStatus.DOWN,
                        error_message=f'Health check failed: {str(e)}'
                    )

        except Exception as e:
            self._logger.exception("Error reporting channel health:")


async def _register_channel_plugins():
    """Register available channel plugins."""
    try:
        from src.notification.channels.base import channel_registry
        from src.notification.channels.telegram_channel import TelegramChannel
        from src.notification.channels.email_channel import EmailChannel
        from src.notification.channels.sms_channel import SMSChannel

        # Register channel plugins
        channel_registry.register_channel('telegram', TelegramChannel)
        channel_registry.register_channel('email', EmailChannel)
        channel_registry.register_channel('sms', SMSChannel)

        _logger.info("Registered channel plugins: %s", channel_registry.list_channels())

    except Exception as e:
        _logger.exception("Error registering channel plugins:")


async def startup():
    """Application startup."""
    global message_poller, message_processor, health_reporter

    _logger.info("Starting Notification Service (Database-Centric)...")

    # Initialize database
    from src.data.db.services.database_service import init_databases
    init_databases()
    _logger.info("Database initialized")

    # Register channel plugins
    await _register_channel_plugins()

    # Initialize message processor
    from src.notification.service.processor import message_processor as mp
    message_processor = mp
    await message_processor.start()
    _logger.info("Message processor started")

    # Start health monitor for channel monitoring
    from src.notification.service.health_monitor import health_monitor
    await health_monitor.start()
    _logger.info("Health monitor started")

    # Initialize message poller
    message_poller = MessagePoller(poll_interval_seconds=config.processing.batch_timeout_seconds or 5)
    await message_poller.start()
    _logger.info("Message poller started")

    # Initialize health reporter
    health_reporter = HealthReporter(report_interval_seconds=config.health_check_interval_seconds)
    await health_reporter.start()
    _logger.info("Health reporter started")

    _logger.info("Notification Service startup complete")


async def shutdown():
    """Application shutdown."""
    global message_poller, message_processor, health_reporter

    _logger.info("Shutting down Notification Service...")

    # Stop health monitor
    from src.notification.service.health_monitor import health_monitor
    await health_monitor.stop()

    # Stop health reporter
    if health_reporter:
        await health_reporter.stop()

    # Stop message poller
    if message_poller:
        await message_poller.stop()

    # Stop message processor
    if message_processor:
        await message_processor.shutdown()

    _logger.info("Notification Service shutdown complete")


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        _logger.info("Received signal %s, initiating shutdown...", signum)
        asyncio.create_task(shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main application entry point."""
    try:
        # Setup signal handlers
        setup_signal_handlers()

        # Start the service
        await startup()

        # Keep running until shutdown
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt")
    except Exception as e:
        _logger.exception("Unexpected error:")
    finally:
        await shutdown()


if __name__ == "__main__":
    asyncio.run(main())