"""
Scheduler Service Main Application

Main entry point for the APScheduler-based job scheduling service.
Provides centralized scheduling with database persistence and error handling.
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.scheduler.scheduler_service import SchedulerService
from src.scheduler.config import SchedulerServiceConfig
from src.common.alerts.alert_evaluator import AlertEvaluator
from src.common.alerts.schema_validator import AlertSchemaValidator
from src.data.data_manager import DataManager
from src.indicators.service import IndicatorService
from src.data.db.services.jobs_service import JobsService
from src.data.db.core.database import session_scope
from src.notification.service.client import NotificationServiceClient
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SchedulerApplication:
    """Main scheduler application with dependency injection and lifecycle management."""
    """Main scheduler application with dependency injection and lifecycle management."""

    def __init__(self, config: SchedulerServiceConfig):
        """
        Initialize the scheduler application.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.scheduler_service: Optional[SchedulerService] = None
        self._shutdown_event = asyncio.Event()

        # Service dependencies
        self.data_manager: Optional[DataManager] = None
        self.indicator_service: Optional[IndicatorService] = None
        self.jobs_service: Optional[JobsService] = None
        self.alert_evaluator: Optional[AlertEvaluator] = None
        self.notification_client: Optional[NotificationServiceClient] = None
        self.schema_validator: Optional[AlertSchemaValidator] = None

        _logger.info("Scheduler application initialized")

    async def initialize_services(self) -> None:
        """Initialize all required services with proper dependency injection."""
        try:
            _logger.info("Initializing scheduler services...")

            # Initialize schema validator
            self.schema_validator = AlertSchemaValidator(self.config.alert.schema_dir)
            _logger.debug("Schema validator initialized")

            # Initialize data manager
            self.data_manager = DataManager()
            _logger.debug("Data manager initialized")

            # Initialize indicator service
            self.indicator_service = IndicatorService()
            _logger.debug("Indicator service initialized")

            # Initialize notification client
            self.notification_client = NotificationServiceClient(
                service_url=self.config.notification.service_url,
                timeout=self.config.notification.timeout,
                max_retries=self.config.notification.max_retries
            )
            _logger.debug("Notification client initialized")

            # Initialize jobs service with database session
            with session_scope() as session:
                self.jobs_service = JobsService(session)
                _logger.debug("Jobs service initialized")

            # Initialize alert evaluator with dependencies
            self.alert_evaluator = AlertEvaluator(
                data_manager=self.data_manager,
                indicator_service=self.indicator_service,
                jobs_service=self.jobs_service,
                schema_validator=self.schema_validator
            )
            _logger.debug("Alert evaluator initialized")

            # Initialize scheduler service with all dependencies
            self.scheduler_service = SchedulerService(
                jobs_service=self.jobs_service,
                alert_evaluator=self.alert_evaluator,
                notification_client=self.notification_client,
                database_url=self.config.database.url,
                max_workers=self.config.scheduler.max_workers
            )
            _logger.debug("Scheduler service initialized")

            _logger.info("All scheduler services initialized successfully")

        except Exception as e:
            _logger.error("Failed to initialize scheduler services: %s", str(e))
            raise

    async def start(self) -> None:
        """Start the scheduler application."""
        try:
            _logger.info("Starting scheduler application...")

            # Initialize services
            await self.initialize_services()

            # Start scheduler service
            await self.scheduler_service.start()

            _logger.info("Scheduler application started successfully")

            # Log service status
            status = self.scheduler_service.get_scheduler_status()
            _logger.info("Scheduler status: %s", status)

        except Exception as e:
            _logger.error("Failed to start scheduler application: %s", str(e))
            raise

    async def stop(self) -> None:
        """Stop the scheduler application gracefully."""
        try:
            _logger.info("Stopping scheduler application...")

            # Stop scheduler service
            if self.scheduler_service:
                await self.scheduler_service.stop()

            # Close notification client
            if self.notification_client:
                await self.notification_client.close()

            # Set shutdown event
            self._shutdown_event.set()

            _logger.info("Scheduler application stopped successfully")

        except Exception as e:
            _logger.error("Error stopping scheduler application: %s", str(e))
            raise

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def reload_schedules(self) -> int:
        """Reload schedules from database."""
        if not self.scheduler_service:
            raise RuntimeError("Scheduler service not initialized")

        count = await self.scheduler_service.reload_schedules()
        _logger.info("Reloaded %d schedules", count)
        return count

    def get_status(self) -> dict:
        """Get application status."""
        status = {
            "service": self.config.service.name,
            "version": self.config.service.version,
            "environment": self.config.service.environment,
            "database_url": self.config.database.url.split('@')[1] if '@' in self.config.database.url else "local",
            "max_workers": self.config.scheduler.max_workers,
            "notification_service": self.config.notification.service_url,
            "scheduler": None
        }

        if self.scheduler_service:
            status["scheduler"] = self.scheduler_service.get_scheduler_status()

        return status


# Global application instance
app: Optional[SchedulerApplication] = None


def setup_signal_handlers(application: SchedulerApplication) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        _logger.info("Received signal %d, initiating graceful shutdown...", signum)
        asyncio.create_task(application.stop())

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)


async def main() -> None:
    """Main application entry point."""
    global app

    try:
        # Load configuration
        config = SchedulerServiceConfig()

        # Create application
        app = SchedulerApplication(config)

        # Setup signal handlers
        setup_signal_handlers(app)

        # Start application
        await app.start()

        # Log startup completion
        _logger.info("Scheduler service is running. Press Ctrl+C to stop.")

        # Wait for shutdown
        await app.wait_for_shutdown()

    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt")
    except Exception as e:
        _logger.error("Application error: %s", str(e))
        sys.exit(1)
    finally:
        if app:
            await app.stop()


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())