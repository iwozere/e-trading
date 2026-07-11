"""
Scheduler Service Main Application

Main entry point for the APScheduler-based job scheduling service.
Provides centralized scheduling with database persistence and error handling.
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Any


# Add project root to path if not already present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.alerts.alert_evaluator import AlertEvaluator
from src.common.alerts.schema_validator import AlertSchemaValidator
from src.data.data_manager import DataManager
from src.data.db.services.jobs_service import JobsService
from src.data.db.services.notification_service import NotificationService
from src.indicators.service import IndicatorService
from src.notification.logger import setup_logger
from src.scheduler.config import SchedulerServiceConfig, get_config
from src.scheduler.scheduler_service import SchedulerService

_logger = setup_logger(__name__)


class SchedulerApplication:
    """Main scheduler application with dependency injection and lifecycle management."""

    def __init__(self, config: SchedulerServiceConfig):
        """
        Initialize the scheduler application.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.scheduler_service: SchedulerService | None = None
        self._shutdown_event = asyncio.Event()

        # Service dependencies
        self.data_manager: DataManager | None = None
        self.indicator_service: IndicatorService | None = None
        self.jobs_service: JobsService | None = None
        self.alert_evaluator: AlertEvaluator | None = None
        self.notification_db_service: NotificationService | None = None
        self.schema_validator: AlertSchemaValidator | None = None

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

            # Initialize notification database service (for both alerts and data processing)
            self.notification_db_service = NotificationService()
            _logger.debug("Notification database service initialized")

            # Initialize jobs service (uses modern UoW pattern)
            self.jobs_service = JobsService()
            _logger.debug("Jobs service initialized")

            # Initialize alert evaluator with dependencies
            self.alert_evaluator = AlertEvaluator(
                data_manager=self.data_manager,
                indicator_service=self.indicator_service,
                jobs_service=self.jobs_service,
                schema_validator=self.schema_validator,
            )
            _logger.debug("Alert evaluator initialized")

            # Initialize scheduler service with all dependencies
            self.scheduler_service = SchedulerService(
                jobs_service=self.jobs_service,
                alert_evaluator=self.alert_evaluator,
                notification_db_service=self.notification_db_service,
                database_url=self.config.database.url,
                max_workers=self.config.scheduler.max_workers,
                job_timeout_seconds=self.config.scheduler.job_timeout,
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
            if self.scheduler_service is None:
                raise RuntimeError("Scheduler service is not initialized")
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

    def get_status(self) -> dict[str, Any]:
        """Get application status."""
        status: dict[str, Any] = {
            "service": self.config.service.name,
            "version": self.config.service.version,
            "environment": self.config.service.environment,
            "database_url": self.config.database.url.split("@")[1] if "@" in self.config.database.url else "local",
            "max_workers": self.config.scheduler.max_workers,
            "notification_method": "database",  # Direct database, not HTTP
            "scheduler": None,
        }

        if self.scheduler_service:
            status["scheduler"] = self.scheduler_service.get_scheduler_status()

        return status



# Global application instance
app: SchedulerApplication | None = None


def setup_signal_handlers(loop: asyncio.AbstractEventLoop, application: SchedulerApplication) -> None:
    """Set up signal handlers for graceful shutdown.

    Uses loop.call_soon_threadsafe + asyncio.Event so the shutdown sequence
    runs safely inside the event loop, not from the OS signal context.
    """

    def signal_handler(signum, frame):
        _logger.info("Received signal %d, initiating graceful shutdown...", signum)
        # Safely set the shutdown event from the signal handler (sync context).
        # wait_for_shutdown() will return and then main() will call app.stop().
        loop.call_soon_threadsafe(application._shutdown_event.set)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal_handler)  # pyright: ignore[reportAttributeAccessIssue]  # hasattr-guarded


async def main() -> None:
    """Main application entry point."""
    global app

    try:
        # Load configuration via the lazy singleton so all callers share the same instance
        config = get_config()

        # Create application
        app = SchedulerApplication(config)

        # Setup signal handlers
        loop = asyncio.get_running_loop()
        setup_signal_handlers(loop, app)

        # Start application
        await app.start()

        # Log startup completion
        _logger.info("Scheduler service is running. Press Ctrl+C to stop.")

        # Wait for shutdown event (set by signal handler)
        await app.wait_for_shutdown()

    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt")
    except Exception as e:
        _logger.error("Application error: %s", str(e))
        sys.exit(1)
    finally:
        # Always attempt a clean stop whether shutdown came from signal or exception
        if app:
            await app.stop()


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
