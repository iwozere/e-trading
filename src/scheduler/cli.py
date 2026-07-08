#!/usr/bin/env python3
"""
Scheduler Service CLI

Command-line interface for managing the scheduler service.
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path if not already present
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.scheduler.config import SchedulerServiceConfig
from src.scheduler.main import SchedulerApplication

_logger = setup_logger(__name__)


async def start_service(config: SchedulerServiceConfig) -> None:
    """Start the scheduler service."""
    app = SchedulerApplication(config)

    try:
        await app.start()
        _logger.info("Scheduler service started successfully")
        _logger.info("Environment: %s", config.service.environment)
        _logger.info("Max workers: %d", config.scheduler.max_workers)
        _logger.info(
            "Database: %s",
            config.database.url.split("@")[1] if "@" in config.database.url else "local",
        )
        _logger.info("Press Ctrl+C to stop...")

        await app.wait_for_shutdown()

    except KeyboardInterrupt:
        _logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        _logger.error("Fatal error starting scheduler service: %s", e)
        sys.exit(1)
    finally:
        await app.stop()


async def show_status(config: SchedulerServiceConfig) -> None:
    """Show scheduler service status."""
    app = SchedulerApplication(config)

    try:
        await app.initialize_services()
        status = app.get_status()

        _logger.info("Scheduler Service Status:")
        _logger.info("=" * 40)
        _logger.info("Service:              %s", status["service"])
        _logger.info("Version:              %s", status["version"])
        _logger.info("Environment:          %s", status["environment"])
        _logger.info("Database:             %s", status["database_url"])
        _logger.info("Max Workers:          %s", status["max_workers"])
        _logger.info("Notification Method:  %s", status.get("notification_method", "N/A"))

        if status["scheduler"]:
            scheduler_status = status["scheduler"]
            _logger.info("Scheduler Running:    %s", scheduler_status["is_running"])
            _logger.info("Scheduler State:      %s", scheduler_status.get("scheduler_state", "N/A"))
            _logger.info("Job Count:            %s", scheduler_status.get("job_count", 0))

    except Exception as e:
        _logger.error("Error getting status: %s", e)
        sys.exit(1)
    finally:
        await app.stop()


async def reload_schedules(config: SchedulerServiceConfig) -> None:
    """Reload schedules from database."""
    app = SchedulerApplication(config)

    try:
        await app.start()
        count = await app.reload_schedules()
        _logger.info("Successfully reloaded %d schedules", count)

    except Exception as e:
        _logger.error("Error reloading schedules: %s", e)
        sys.exit(1)
    finally:
        await app.stop()


async def validate_config(config: SchedulerServiceConfig) -> None:
    """Validate configuration."""
    try:
        _logger.info("Configuration Validation:")
        _logger.info("=" * 40)

        config_dict = config.to_dict()

        _logger.info("✓ Configuration loaded successfully")
        _logger.info("✓ Environment:     %s", config_dict["service"]["environment"])
        _logger.info("✓ Database URL:    %s", config_dict["database"]["url"])
        _logger.info("✓ Max Workers:     %s", config_dict["scheduler"]["max_workers"])
        _logger.info("✓ Alert Schema Dir: %s", config_dict["alert"]["schema_dir"])

        # Check schema directory
        schema_path = Path(config.alert.schema_dir)
        if schema_path.exists():
            _logger.info("✓ Schema directory exists: %s", schema_path)
        else:
            _logger.warning("⚠ Schema directory not found: %s", schema_path)

        _logger.info("Configuration is valid!")

    except Exception as e:
        _logger.error("Configuration validation failed: %s", e)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scheduler Service CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                    # Start the scheduler service
  %(prog)s status                   # Show service status
  %(prog)s reload                   # Reload schedules from database
  %(prog)s validate                 # Validate configuration
        """,
    )

    parser.add_argument("command", choices=["start", "status", "reload", "validate"], help="Command to execute")

    parser.add_argument("--env", default=None, help="Environment to use (development, staging, production)")

    parser.add_argument("--config-file", default=None, help="Path to configuration file (not implemented yet)")

    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")

    args = parser.parse_args()

    # Set environment if provided
    if args.env:
        import os

        os.environ["TRADING_ENV"] = args.env

    # Set log level
    import os

    os.environ["LOG_LEVEL"] = args.log_level

    try:
        # Load configuration
        config = SchedulerServiceConfig()

        # Execute command
        if args.command == "start":
            asyncio.run(start_service(config))
        elif args.command == "status":
            asyncio.run(show_status(config))
        elif args.command == "reload":
            asyncio.run(reload_schedules(config))
        elif args.command == "validate":
            asyncio.run(validate_config(config))

    except KeyboardInterrupt:
        _logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        _logger.error("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
