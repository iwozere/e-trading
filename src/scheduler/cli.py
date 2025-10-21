#!/usr/bin/env python3
"""
Scheduler Service CLI

Command-line interface for managing the scheduler service.
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.scheduler.main import SchedulerApplication
from src.scheduler.config import SchedulerServiceConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


async def start_service(config: SchedulerServiceConfig) -> None:
    """Start the scheduler service."""
    app = SchedulerApplication(config)

    try:
        await app.start()
        print(f"Scheduler service started successfully")
        print(f"Environment: {config.service.environment}")
        print(f"Max workers: {config.scheduler.max_workers}")
        print(f"Database: {config.database.url.split('@')[1] if '@' in config.database.url else 'local'}")
        print("Press Ctrl+C to stop...")

        await app.wait_for_shutdown()

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await app.stop()


async def show_status(config: SchedulerServiceConfig) -> None:
    """Show scheduler service status."""
    app = SchedulerApplication(config)

    try:
        await app.initialize_services()
        status = app.get_status()

        print("Scheduler Service Status:")
        print("=" * 40)
        print(f"Service: {status['service']}")
        print(f"Version: {status['version']}")
        print(f"Environment: {status['environment']}")
        print(f"Database: {status['database_url']}")
        print(f"Max Workers: {status['max_workers']}")
        print(f"Notification Service: {status['notification_service']}")

        if status['scheduler']:
            scheduler_status = status['scheduler']
            print(f"Scheduler Running: {scheduler_status['is_running']}")
            print(f"Scheduler State: {scheduler_status.get('scheduler_state', 'N/A')}")
            print(f"Job Count: {scheduler_status.get('job_count', 0)}")

    except Exception as e:
        print(f"Error getting status: {e}")
        sys.exit(1)
    finally:
        await app.stop()


async def reload_schedules(config: SchedulerServiceConfig) -> None:
    """Reload schedules from database."""
    app = SchedulerApplication(config)

    try:
        await app.start()
        count = await app.reload_schedules()
        print(f"Successfully reloaded {count} schedules")

    except Exception as e:
        print(f"Error reloading schedules: {e}")
        sys.exit(1)
    finally:
        await app.stop()


async def validate_config(config: SchedulerServiceConfig) -> None:
    """Validate configuration."""
    try:
        print("Configuration Validation:")
        print("=" * 40)

        config_dict = config.to_dict()

        print("✓ Configuration loaded successfully")
        print(f"✓ Environment: {config_dict['service']['environment']}")
        print(f"✓ Database URL: {config_dict['database']['url']}")
        print(f"✓ Max Workers: {config_dict['scheduler']['max_workers']}")
        print(f"✓ Notification Service: {config_dict['notification']['service_url']}")
        print(f"✓ Alert Schema Dir: {config_dict['alert']['schema_dir']}")

        # Check schema directory
        schema_path = Path(config.alert.schema_dir)
        if schema_path.exists():
            print(f"✓ Schema directory exists: {schema_path}")
        else:
            print(f"⚠ Schema directory not found: {schema_path}")

        print("\nConfiguration is valid!")

    except Exception as e:
        print(f"Configuration validation failed: {e}")
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
        """
    )

    parser.add_argument(
        "command",
        choices=["start", "status", "reload", "validate"],
        help="Command to execute"
    )

    parser.add_argument(
        "--env",
        default=None,
        help="Environment to use (development, staging, production)"
    )

    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to configuration file (not implemented yet)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )

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
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()