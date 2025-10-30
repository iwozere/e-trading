"""
Example usage of the Message Archival Service

This example demonstrates how to use the archival service for message cleanup and archiving.
"""

import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.notification.service.archival_service import (
    MessageArchivalService, ArchivalPolicy, RetentionPolicyManager,
    ScheduledCleanupService, create_archival_service, run_manual_archival
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def example_basic_archival_usage(session: Session):
    """
    Example of basic archival service usage.

    Args:
        session: Database session
    """
    _logger.info("=== Basic Archival Service Example ===")

    # Create archival service with custom policy
    policy = ArchivalPolicy(
        archive_after_days=7,  # Archive messages after 7 days (for demo)
        delete_after_days=30,  # Delete archived messages after 30 days
        failed_message_retention_days=14,  # Keep failed messages for 14 days
        archive_path="data/demo_archives",
        compress_archives=True
    )

    archival_service = MessageArchivalService(session, policy)

    # Get current archival statistics
    stats = archival_service.get_archival_statistics()
    _logger.info("Current archival statistics:")
    _logger.info("- Messages ready for archival: %d", stats['current_status']['messages_ready_for_archival'])
    _logger.info("- Failed messages ready for cleanup: %d", stats['current_status']['failed_messages_ready_for_cleanup'])
    _logger.info("- Archive files count: %d", stats['current_status']['archived_files_count'])
    _logger.info("- Is low traffic period: %s", stats['current_status']['is_low_traffic_period'])

    # Run manual archival (only during low traffic periods)
    if archival_service.is_low_traffic_period():
        _logger.info("Running manual archival cycle...")
        results = archival_service.run_full_archival_cycle()

        for operation, result in results.items():
            _logger.info("Operation '%s': %d messages processed", operation,
                        getattr(result, 'messages_archived', 0) +
                        getattr(result, 'messages_deleted', 0) +
                        getattr(result, 'failed_messages_cleaned', 0))
    else:
        _logger.info("Not in low traffic period, skipping archival")


def example_retention_policy_usage(session: Session):
    """
    Example of retention policy manager usage.

    Args:
        session: Database session
    """
    _logger.info("=== Retention Policy Manager Example ===")

    # Create retention policy manager
    policy_manager = RetentionPolicyManager(session)

    # List available policies
    policies = policy_manager.list_policies()
    _logger.info("Available retention policies:")
    for name, policy in policies.items():
        _logger.info("- %s: archive after %d days, delete after %d days",
                    name, policy.archive_after_days, policy.delete_after_days)

    # Get policy for different message types
    high_priority_policy = policy_manager.get_policy(priority='HIGH')
    _logger.info("High priority policy: archive after %d days", high_priority_policy.archive_after_days)

    system_alert_policy = policy_manager.get_policy(message_type='system_alert')
    _logger.info("System alert policy: archive after %d days", system_alert_policy.archive_after_days)

    # Create custom policy
    custom_policy = ArchivalPolicy(
        archive_after_days=60,
        delete_after_days=1095,  # 3 years
        failed_message_retention_days=180
    )
    policy_manager.update_policy('custom_long_term', custom_policy)
    _logger.info("Added custom long-term retention policy")


def example_scheduled_cleanup_usage(session: Session):
    """
    Example of scheduled cleanup service usage.

    Args:
        session: Database session
    """
    _logger.info("=== Scheduled Cleanup Service Example ===")

    # Create scheduled cleanup service
    cleanup_service = ScheduledCleanupService(session)

    # Start the service (this sets up default tasks)
    cleanup_service.start_scheduled_cleanup()

    # Get cleanup status
    status = cleanup_service.get_cleanup_status()
    _logger.info("Cleanup service status:")
    _logger.info("- Running: %s", status['running'])
    _logger.info("- Is cleanup time: %s", status['is_cleanup_time'])
    _logger.info("- Scheduled tasks: %d", len(status['tasks']))

    for task in status['tasks']:
        _logger.info("  - Task '%s': interval %d hours, next run %s",
                    task['name'], task['interval_hours'], task['next_run'])

    # Run cleanup cycle manually
    if cleanup_service.is_cleanup_time():
        _logger.info("Running cleanup cycle...")
        results = cleanup_service.run_cleanup_cycle()
        _logger.info("Cleanup cycle completed with %d tasks", len(results))
    else:
        _logger.info("Not in cleanup time window")

    # Stop the service
    cleanup_service.stop_scheduled_cleanup()


async def example_scheduled_archival_daemon(session: Session):
    """
    Example of running the archival daemon.

    Args:
        session: Database session
    """
    _logger.info("=== Scheduled Archival Daemon Example ===")

    # Create archival service
    archival_service = create_archival_service(session)

    # This would run indefinitely in a real application
    _logger.info("Starting archival daemon (demo will run for 30 seconds)...")

    try:
        # Run for a short time for demo purposes
        await asyncio.wait_for(
            archival_service.run_scheduled_archival(interval_hours=1),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        _logger.info("Demo timeout reached, stopping archival daemon")


def example_manual_archival_utility(session: Session):
    """
    Example of using the manual archival utility function.

    Args:
        session: Database session
    """
    _logger.info("=== Manual Archival Utility Example ===")

    # Create custom policy for immediate archival (for demo)
    demo_policy = ArchivalPolicy(
        archive_after_days=0,  # Archive all messages immediately
        delete_after_days=1,   # Delete archived messages after 1 day
        failed_message_retention_days=1,
        archive_path="data/manual_archives"
    )

    # Run manual archival
    results = run_manual_archival(session, demo_policy)

    _logger.info("Manual archival results:")
    for operation, stats in results.items():
        _logger.info("- %s: %d messages, %d bytes, %.2f seconds",
                    operation,
                    getattr(stats, 'messages_archived', 0) +
                    getattr(stats, 'messages_deleted', 0) +
                    getattr(stats, 'failed_messages_cleaned', 0),
                    getattr(stats, 'bytes_archived', 0) + getattr(stats, 'bytes_freed', 0),
                    stats.duration_seconds)


def main():
    """
    Main function to run all examples.

    Note: This is a demonstration. In a real application, you would:
    1. Use a proper database session from your application
    2. Run the scheduled services as background tasks
    3. Configure policies based on your business requirements
    """
    _logger.info("Message Archival Service Examples")
    _logger.info("=" * 50)

    # Mock session for demonstration
    from unittest.mock import Mock
    mock_session = Mock()

    try:
        # Run examples
        example_basic_archival_usage(mock_session)
        print()

        example_retention_policy_usage(mock_session)
        print()

        example_scheduled_cleanup_usage(mock_session)
        print()

        example_manual_archival_utility(mock_session)
        print()

        # Async example
        _logger.info("Running async archival daemon example...")
        asyncio.run(example_scheduled_archival_daemon(mock_session))

    except Exception as e:
        _logger.exception("Error in examples:")
        raise

    _logger.info("All examples completed successfully!")


if __name__ == '__main__':
    main()