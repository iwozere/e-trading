"""
Message Archival Service

Service for archiving and cleaning up old notification messages.
Implements configurable retention policies and scheduled cleanup operations.

Requirements addressed:
- 8.1: Archive delivered messages older than 30 days
- 8.2: Delete archived messages older than 1 year
- 8.3: Maintain failed message history for 90 days
- 8.4: Provide configurable retention policies
- 8.5: Perform cleanup operations during low-traffic periods
"""

import json
import gzip
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text

from src.data.db.models.model_notification import (
    Message, MessageDeliveryStatus, MessageStatus, DeliveryStatus
)
from src.data.db.repos.repo_notification import NotificationRepository
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ArchivalStatus(str, Enum):
    """Archival status enumeration."""
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


@dataclass
class ArchivalPolicy:
    """Configuration for archival policies."""
    # Days to keep active messages before archiving
    archive_after_days: int = 30

    # Days to keep archived messages before deletion
    delete_after_days: int = 365  # 1 year

    # Days to keep failed messages
    failed_message_retention_days: int = 90

    # Compression level for archived messages (1-9)
    compression_level: int = 6

    # Batch size for archival operations
    batch_size: int = 1000

    # Maximum number of messages to process per run
    max_messages_per_run: int = 10000

    # Archive storage path
    archive_path: str = "data/archives/notifications"

    # Whether to compress archived messages
    compress_archives: bool = True

    # Low traffic hours for cleanup (24-hour format)
    low_traffic_start_hour: int = 2
    low_traffic_end_hour: int = 6


@dataclass
class ArchivalStats:
    """Statistics from archival operations."""
    messages_archived: int = 0
    messages_deleted: int = 0
    failed_messages_cleaned: int = 0
    bytes_archived: int = 0
    bytes_freed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class MessageArchivalService:
    """Service for archiving and cleaning up notification messages."""

    def __init__(self, session: Session, policy: Optional[ArchivalPolicy] = None):
        """
        Initialize the archival service.

        Args:
            session: Database session
            policy: Archival policy configuration
        """
        self.session = session
        self.repo = NotificationRepository(session)
        self.policy = policy or ArchivalPolicy()

        # Ensure archive directory exists
        self.archive_path = Path(self.policy.archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)

        _logger.info("Initialized archival service with policy: archive_after=%d days, delete_after=%d days",
                    self.policy.archive_after_days, self.policy.delete_after_days)

    def is_low_traffic_period(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if current time is within low traffic period.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            True if within low traffic period
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        current_hour = current_time.hour
        start_hour = self.policy.low_traffic_start_hour
        end_hour = self.policy.low_traffic_end_hour

        if start_hour <= end_hour:
            return start_hour <= current_hour < end_hour
        else:
            # Handle overnight period (e.g., 22:00 to 06:00)
            return current_hour >= start_hour or current_hour < end_hour

    def get_messages_for_archival(self, cutoff_date: datetime, limit: int) -> List[Message]:
        """
        Get messages that are ready for archival.

        Args:
            cutoff_date: Messages older than this date will be archived
            limit: Maximum number of messages to return

        Returns:
            List of messages ready for archival
        """
        return self.session.query(Message).filter(
            and_(
                Message.created_at < cutoff_date,
                Message.status.in_([
                    MessageStatus.DELIVERED.value,
                    MessageStatus.CANCELLED.value
                ]),
                # Only archive messages that don't have a custom archival status
                # (we'll add this field to track archival status)
            )
        ).order_by(Message.created_at).limit(limit).all()

    def get_archived_messages_for_deletion(self, cutoff_date: datetime, limit: int) -> List[Dict[str, Any]]:
        """
        Get archived messages that are ready for deletion.

        Args:
            cutoff_date: Archived messages older than this date will be deleted
            limit: Maximum number of messages to return

        Returns:
            List of archived message metadata ready for deletion
        """
        # This would query a separate archival tracking table
        # For now, we'll implement a file-based approach
        archived_files = []

        try:
            for archive_file in self.archive_path.glob("*.json.gz"):
                # Parse filename to get date
                try:
                    # Expected format: messages_YYYYMMDD_HHMMSS.json.gz
                    parts = archive_file.stem.split('_')
                    if len(parts) >= 3:
                        date_str = parts[1]
                        time_str = parts[2].split('.')[0]  # Remove .json extension
                        archive_date = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")

                        if archive_date < cutoff_date:
                            archived_files.append({
                                'file_path': str(archive_file),
                                'archive_date': archive_date,
                                'size_bytes': archive_file.stat().st_size
                            })

                            if len(archived_files) >= limit:
                                break
                except (ValueError, IndexError) as e:
                    _logger.warning("Could not parse archive file date: %s - %s", archive_file, e)
                    continue

        except Exception as e:
            _logger.exception("Error scanning archive files:")

        return sorted(archived_files, key=lambda x: x['archive_date'])

    def get_failed_messages_for_cleanup(self, cutoff_date: datetime, limit: int) -> List[Message]:
        """
        Get failed messages that are ready for cleanup.

        Args:
            cutoff_date: Failed messages older than this date will be deleted
            limit: Maximum number of messages to return

        Returns:
            List of failed messages ready for cleanup
        """
        return self.session.query(Message).filter(
            and_(
                Message.created_at < cutoff_date,
                Message.status == MessageStatus.FAILED.value,
                Message.retry_count >= Message.max_retries
            )
        ).order_by(Message.created_at).limit(limit).all()

    def archive_message(self, message: Message) -> Dict[str, Any]:
        """
        Archive a single message to compressed storage.

        Args:
            message: Message to archive

        Returns:
            Dictionary with archived message data
        """
        try:
            # Create archive data structure
            archive_data = {
                'id': message.id,
                'message_type': message.message_type,
                'priority': message.priority,
                'channels': message.channels,
                'recipient_id': message.recipient_id,
                'template_name': message.template_name,
                'content': message.content,
                'metadata': message.message_metadata,
                'created_at': message.created_at.isoformat() if message.created_at else None,
                'scheduled_for': message.scheduled_for.isoformat() if message.scheduled_for else None,
                'status': message.status,
                'retry_count': message.retry_count,
                'max_retries': message.max_retries,
                'last_error': message.last_error,
                'processed_at': message.processed_at.isoformat() if message.processed_at else None,
                'archived_at': datetime.now(timezone.utc).isoformat()
            }

            # Include delivery status information
            delivery_statuses = self.repo.delivery_status.get_delivery_statuses_by_message(message.id)
            archive_data['delivery_statuses'] = []

            for ds in delivery_statuses:
                archive_data['delivery_statuses'].append({
                    'id': ds.id,
                    'channel': ds.channel,
                    'status': ds.status,
                    'delivered_at': ds.delivered_at.isoformat() if ds.delivered_at else None,
                    'response_time_ms': ds.response_time_ms,
                    'error_message': ds.error_message,
                    'external_id': ds.external_id,
                    'created_at': ds.created_at.isoformat() if ds.created_at else None
                })

            return archive_data

        except Exception as e:
            _logger.error("Error creating archive data for message %s: %s", message.id, e)
            raise

    def save_archived_messages(self, archived_messages: List[Dict[str, Any]]) -> Tuple[str, int]:
        """
        Save archived messages to compressed file.

        Args:
            archived_messages: List of archived message data

        Returns:
            Tuple of (file_path, bytes_written)
        """
        if not archived_messages:
            return "", 0

        try:
            # Create filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"messages_{timestamp}.json.gz"
            file_path = self.archive_path / filename

            # Prepare data for archival
            archive_batch = {
                'archived_at': datetime.now(timezone.utc).isoformat(),
                'message_count': len(archived_messages),
                'messages': archived_messages
            }

            # Serialize and compress
            json_data = json.dumps(archive_batch, indent=2, ensure_ascii=False)
            json_bytes = json_data.encode('utf-8')

            if self.policy.compress_archives:
                with gzip.open(file_path, 'wt', encoding='utf-8', compresslevel=self.policy.compression_level) as f:
                    f.write(json_data)
                bytes_written = file_path.stat().st_size
            else:
                # Save as uncompressed JSON
                uncompressed_path = self.archive_path / f"messages_{timestamp}.json"
                with open(uncompressed_path, 'w', encoding='utf-8') as f:
                    f.write(json_data)
                bytes_written = len(json_bytes)
                file_path = uncompressed_path

            _logger.info("Archived %d messages to %s (%d bytes)",
                        len(archived_messages), file_path, bytes_written)

            return str(file_path), bytes_written

        except Exception as e:
            _logger.exception("Error saving archived messages:")
            raise

    def delete_archived_file(self, file_path: str) -> int:
        """
        Delete an archived file.

        Args:
            file_path: Path to the archived file

        Returns:
            Number of bytes freed
        """
        try:
            path = Path(file_path)
            if path.exists():
                size = path.stat().st_size
                path.unlink()
                _logger.info("Deleted archived file %s (%d bytes)", file_path, size)
                return size
            else:
                _logger.warning("Archived file not found: %s", file_path)
                return 0
        except Exception as e:
            _logger.error("Error deleting archived file %s: %s", file_path, e)
            raise

    def archive_messages_batch(self, current_time: Optional[datetime] = None) -> ArchivalStats:
        """
        Archive a batch of old messages.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            Statistics from the archival operation
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        stats = ArchivalStats()
        start_time = current_time

        try:
            # Calculate cutoff date for archival
            cutoff_date = current_time - timedelta(days=self.policy.archive_after_days)

            # Get messages to archive
            messages_to_archive = self.get_messages_for_archival(
                cutoff_date,
                min(self.policy.batch_size, self.policy.max_messages_per_run)
            )

            if not messages_to_archive:
                _logger.info("No messages found for archival")
                return stats

            _logger.info("Found %d messages for archival (older than %s)",
                        len(messages_to_archive), cutoff_date.isoformat())

            # Archive messages in batches
            archived_data = []
            message_ids_to_delete = []

            for message in messages_to_archive:
                try:
                    archive_data = self.archive_message(message)
                    archived_data.append(archive_data)
                    message_ids_to_delete.append(message.id)

                    # Process in batches
                    if len(archived_data) >= self.policy.batch_size:
                        file_path, bytes_written = self.save_archived_messages(archived_data)
                        stats.bytes_archived += bytes_written

                        # Delete original messages from database
                        self._delete_messages_from_db(message_ids_to_delete)
                        stats.messages_archived += len(message_ids_to_delete)

                        # Reset batch
                        archived_data = []
                        message_ids_to_delete = []

                except Exception as e:
                    error_msg = f"Error archiving message {message.id}: {e}"
                    _logger.error(error_msg)
                    stats.errors.append(error_msg)

            # Process remaining messages
            if archived_data:
                file_path, bytes_written = self.save_archived_messages(archived_data)
                stats.bytes_archived += bytes_written

                # Delete original messages from database
                self._delete_messages_from_db(message_ids_to_delete)
                stats.messages_archived += len(message_ids_to_delete)

            # Commit transaction
            self.session.commit()

        except Exception as e:
            error_msg = f"Error in archive_messages_batch: {e}"
            _logger.error(error_msg)
            stats.errors.append(error_msg)
            self.session.rollback()

        finally:
            stats.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        _logger.info("Archival batch completed: %d messages archived, %d bytes, %.2f seconds",
                    stats.messages_archived, stats.bytes_archived, stats.duration_seconds)

        return stats

    def cleanup_archived_messages_batch(self, current_time: Optional[datetime] = None) -> ArchivalStats:
        """
        Clean up old archived messages.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            Statistics from the cleanup operation
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        stats = ArchivalStats()
        start_time = current_time

        try:
            # Calculate cutoff date for deletion
            cutoff_date = current_time - timedelta(days=self.policy.delete_after_days)

            # Get archived files to delete
            archived_files = self.get_archived_messages_for_deletion(
                cutoff_date,
                self.policy.max_messages_per_run
            )

            if not archived_files:
                _logger.info("No archived files found for deletion")
                return stats

            _logger.info("Found %d archived files for deletion (older than %s)",
                        len(archived_files), cutoff_date.isoformat())

            # Delete archived files
            for file_info in archived_files:
                try:
                    bytes_freed = self.delete_archived_file(file_info['file_path'])
                    stats.bytes_freed += bytes_freed
                    stats.messages_deleted += 1  # Each file represents a batch

                except Exception as e:
                    error_msg = f"Error deleting archived file {file_info['file_path']}: {e}"
                    _logger.error(error_msg)
                    stats.errors.append(error_msg)

        except Exception as e:
            error_msg = f"Error in cleanup_archived_messages_batch: {e}"
            _logger.error(error_msg)
            stats.errors.append(error_msg)

        finally:
            stats.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        _logger.info("Archived cleanup completed: %d files deleted, %d bytes freed, %.2f seconds",
                    stats.messages_deleted, stats.bytes_freed, stats.duration_seconds)

        return stats

    def cleanup_failed_messages_batch(self, current_time: Optional[datetime] = None) -> ArchivalStats:
        """
        Clean up old failed messages.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            Statistics from the cleanup operation
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        stats = ArchivalStats()
        start_time = current_time

        try:
            # Calculate cutoff date for failed message cleanup
            cutoff_date = current_time - timedelta(days=self.policy.failed_message_retention_days)

            # Get failed messages to clean up
            failed_messages = self.get_failed_messages_for_cleanup(
                cutoff_date,
                min(self.policy.batch_size, self.policy.max_messages_per_run)
            )

            if not failed_messages:
                _logger.info("No failed messages found for cleanup")
                return stats

            _logger.info("Found %d failed messages for cleanup (older than %s)",
                        len(failed_messages), cutoff_date.isoformat())

            # Delete failed messages
            message_ids = [msg.id for msg in failed_messages]
            deleted_count = self._delete_messages_from_db(message_ids)
            stats.failed_messages_cleaned = deleted_count

            # Commit transaction
            self.session.commit()

        except Exception as e:
            error_msg = f"Error in cleanup_failed_messages_batch: {e}"
            _logger.error(error_msg)
            stats.errors.append(error_msg)
            self.session.rollback()

        finally:
            stats.duration_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()

        _logger.info("Failed message cleanup completed: %d messages deleted, %.2f seconds",
                    stats.failed_messages_cleaned, stats.duration_seconds)

        return stats

    def _delete_messages_from_db(self, message_ids: List[int]) -> int:
        """
        Delete messages from database by IDs.

        Args:
            message_ids: List of message IDs to delete

        Returns:
            Number of messages deleted
        """
        if not message_ids:
            return 0

        try:
            # Delete delivery statuses first (cascade should handle this, but be explicit)
            delivery_deleted = self.session.query(MessageDeliveryStatus).filter(
                MessageDeliveryStatus.message_id.in_(message_ids)
            ).delete(synchronize_session=False)

            # Delete messages
            messages_deleted = self.session.query(Message).filter(
                Message.id.in_(message_ids)
            ).delete(synchronize_session=False)

            _logger.info("Deleted %d messages and %d delivery statuses from database",
                        messages_deleted, delivery_deleted)

            return messages_deleted

        except Exception as e:
            _logger.exception("Error deleting messages from database:")
            raise

    def run_full_archival_cycle(self, current_time: Optional[datetime] = None) -> Dict[str, ArchivalStats]:
        """
        Run a complete archival cycle including archiving, cleanup, and failed message cleanup.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            Dictionary with statistics from each operation
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        results = {}

        _logger.info("Starting full archival cycle at %s", current_time.isoformat())

        # Check if we're in low traffic period
        if not self.is_low_traffic_period(current_time):
            _logger.info("Not in low traffic period, skipping archival cycle")
            return results

        try:
            # 1. Archive old messages
            _logger.info("Step 1: Archiving old messages")
            results['archival'] = self.archive_messages_batch(current_time)

            # 2. Clean up old archived files
            _logger.info("Step 2: Cleaning up old archived files")
            results['archived_cleanup'] = self.cleanup_archived_messages_batch(current_time)

            # 3. Clean up old failed messages
            _logger.info("Step 3: Cleaning up old failed messages")
            results['failed_cleanup'] = self.cleanup_failed_messages_batch(current_time)

            # Log summary
            total_archived = results.get('archival', ArchivalStats()).messages_archived
            total_deleted = results.get('archived_cleanup', ArchivalStats()).messages_deleted
            total_failed_cleaned = results.get('failed_cleanup', ArchivalStats()).failed_messages_cleaned

            _logger.info("Full archival cycle completed: %d archived, %d deleted, %d failed cleaned",
                        total_archived, total_deleted, total_failed_cleaned)

        except Exception as e:
            _logger.exception("Error in full archival cycle:")
            raise

        return results

    def get_archival_statistics(self) -> Dict[str, Any]:
        """
        Get current archival statistics.

        Returns:
            Dictionary with archival statistics
        """
        try:
            current_time = datetime.now(timezone.utc)

            # Count messages by age and status
            archive_cutoff = current_time - timedelta(days=self.policy.archive_after_days)
            delete_cutoff = current_time - timedelta(days=self.policy.delete_after_days)
            failed_cutoff = current_time - timedelta(days=self.policy.failed_message_retention_days)

            # Messages ready for archival
            messages_for_archival = self.session.query(func.count(Message.id)).filter(
                and_(
                    Message.created_at < archive_cutoff,
                    Message.status.in_([MessageStatus.DELIVERED.value, MessageStatus.CANCELLED.value])
                )
            ).scalar() or 0

            # Failed messages ready for cleanup
            failed_for_cleanup = self.session.query(func.count(Message.id)).filter(
                and_(
                    Message.created_at < failed_cutoff,
                    Message.status == MessageStatus.FAILED.value,
                    Message.retry_count >= Message.max_retries
                )
            ).scalar() or 0

            # Count archived files
            archived_files = list(self.archive_path.glob("*.json.gz"))
            archived_files.extend(self.archive_path.glob("*.json"))

            total_archive_size = sum(f.stat().st_size for f in archived_files)

            # Count old archived files ready for deletion
            old_archived_files = self.get_archived_messages_for_deletion(delete_cutoff, 1000)

            return {
                'policy': {
                    'archive_after_days': self.policy.archive_after_days,
                    'delete_after_days': self.policy.delete_after_days,
                    'failed_message_retention_days': self.policy.failed_message_retention_days,
                    'low_traffic_hours': f"{self.policy.low_traffic_start_hour:02d}:00-{self.policy.low_traffic_end_hour:02d}:00"
                },
                'current_status': {
                    'messages_ready_for_archival': messages_for_archival,
                    'failed_messages_ready_for_cleanup': failed_for_cleanup,
                    'archived_files_count': len(archived_files),
                    'total_archive_size_bytes': total_archive_size,
                    'old_archived_files_count': len(old_archived_files),
                    'is_low_traffic_period': self.is_low_traffic_period(current_time)
                },
                'archive_path': str(self.archive_path),
                'last_checked': current_time.isoformat()
            }

        except Exception as e:
            _logger.exception("Error getting archival statistics:")
            raise

    async def run_scheduled_archival(self, interval_hours: int = 24) -> None:
        """
        Run archival operations on a schedule.

        Args:
            interval_hours: Hours between archival runs
        """
        _logger.info("Starting scheduled archival service (interval: %d hours)", interval_hours)

        while True:
            try:
                current_time = datetime.now(timezone.utc)

                # Only run during low traffic periods
                if self.is_low_traffic_period(current_time):
                    _logger.info("Running scheduled archival cycle")
                    results = self.run_full_archival_cycle(current_time)

                    # Log results
                    for operation, stats in results.items():
                        if stats.errors:
                            _logger.warning("Errors in %s: %s", operation, stats.errors)
                else:
                    _logger.debug("Skipping archival - not in low traffic period")

                # Wait for next interval
                await asyncio.sleep(interval_hours * 3600)

            except Exception as e:
                _logger.exception("Error in scheduled archival:")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes


class RetentionPolicyManager:
    """Manager for configurable retention policies."""

    def __init__(self, session: Session):
        """
        Initialize the retention policy manager.

        Args:
            session: Database session
        """
        self.session = session
        self._policies = {}
        self._load_default_policies()

    def _load_default_policies(self):
        """Load default retention policies."""
        self._policies = {
            'default': ArchivalPolicy(),
            'high_priority': ArchivalPolicy(
                archive_after_days=60,  # Keep high priority messages longer
                delete_after_days=730,  # 2 years
                failed_message_retention_days=180  # 6 months
            ),
            'system_alerts': ArchivalPolicy(
                archive_after_days=90,  # Keep system alerts longer
                delete_after_days=1095,  # 3 years
                failed_message_retention_days=365  # 1 year
            ),
            'user_notifications': ArchivalPolicy(
                archive_after_days=14,  # Archive user notifications quickly
                delete_after_days=180,  # 6 months
                failed_message_retention_days=30  # 1 month
            )
        }

    def get_policy(self, message_type: str = None, priority: str = None) -> ArchivalPolicy:
        """
        Get retention policy for a message type and priority.

        Args:
            message_type: Type of message
            priority: Message priority

        Returns:
            Appropriate archival policy
        """
        # Determine policy based on message characteristics
        if priority in ['HIGH', 'CRITICAL']:
            return self._policies.get('high_priority', self._policies['default'])
        elif message_type and 'alert' in message_type.lower():
            return self._policies.get('system_alerts', self._policies['default'])
        elif message_type and 'user' in message_type.lower():
            return self._policies.get('user_notifications', self._policies['default'])
        else:
            return self._policies['default']

    def update_policy(self, policy_name: str, policy: ArchivalPolicy):
        """
        Update a retention policy.

        Args:
            policy_name: Name of the policy
            policy: New policy configuration
        """
        self._policies[policy_name] = policy
        _logger.info("Updated retention policy '%s'", policy_name)

    def list_policies(self) -> Dict[str, ArchivalPolicy]:
        """
        List all retention policies.

        Returns:
            Dictionary of policy names to policies
        """
        return self._policies.copy()


class ScheduledCleanupService:
    """Service for scheduled cleanup operations during low-traffic periods."""

    def __init__(self, session: Session, policy_manager: Optional[RetentionPolicyManager] = None):
        """
        Initialize the scheduled cleanup service.

        Args:
            session: Database session
            policy_manager: Retention policy manager
        """
        self.session = session
        self.policy_manager = policy_manager or RetentionPolicyManager(session)
        self.archival_service = MessageArchivalService(session)
        self._running = False
        self._cleanup_tasks = []

    def is_cleanup_time(self, current_time: Optional[datetime] = None) -> bool:
        """
        Check if it's time to run cleanup operations.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            True if cleanup should run
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        # Run cleanup during low traffic hours
        return self.archival_service.is_low_traffic_period(current_time)

    def schedule_cleanup_task(self, task_name: str, task_func, interval_hours: int = 24):
        """
        Schedule a cleanup task.

        Args:
            task_name: Name of the task
            task_func: Function to execute
            interval_hours: Hours between executions
        """
        task_info = {
            'name': task_name,
            'function': task_func,
            'interval_hours': interval_hours,
            'last_run': None,
            'next_run': datetime.now(timezone.utc)
        }
        self._cleanup_tasks.append(task_info)
        _logger.info("Scheduled cleanup task '%s' with %d hour interval", task_name, interval_hours)

    def run_cleanup_cycle(self, current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Run a cleanup cycle for all scheduled tasks.

        Args:
            current_time: Current time (defaults to now)

        Returns:
            Dictionary with results from each task
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        results = {}

        if not self.is_cleanup_time(current_time):
            _logger.info("Not in cleanup time window, skipping cleanup cycle")
            return results

        _logger.info("Starting cleanup cycle at %s", current_time.isoformat())

        for task in self._cleanup_tasks:
            try:
                # Check if task is due
                if current_time >= task['next_run']:
                    _logger.info("Running cleanup task: %s", task['name'])

                    # Execute task
                    result = task['function'](current_time)
                    results[task['name']] = result

                    # Update task timing
                    task['last_run'] = current_time
                    task['next_run'] = current_time + timedelta(hours=task['interval_hours'])

                    _logger.info("Completed cleanup task '%s', next run: %s",
                                task['name'], task['next_run'].isoformat())
                else:
                    _logger.debug("Cleanup task '%s' not due yet (next run: %s)",
                                 task['name'], task['next_run'].isoformat())

            except Exception as e:
                error_msg = f"Error in cleanup task '{task['name']}': {e}"
                _logger.error(error_msg)
                results[task['name']] = {'error': error_msg}

        return results

    def start_scheduled_cleanup(self):
        """Start the scheduled cleanup service."""
        if self._running:
            _logger.warning("Scheduled cleanup service is already running")
            return

        self._running = True

        # Schedule default cleanup tasks
        self.schedule_cleanup_task(
            'message_archival',
            self.archival_service.archive_messages_batch,
            interval_hours=24
        )

        self.schedule_cleanup_task(
            'archived_cleanup',
            self.archival_service.cleanup_archived_messages_batch,
            interval_hours=168  # Weekly
        )

        self.schedule_cleanup_task(
            'failed_message_cleanup',
            self.archival_service.cleanup_failed_messages_batch,
            interval_hours=24
        )

        _logger.info("Started scheduled cleanup service with %d tasks", len(self._cleanup_tasks))

    def stop_scheduled_cleanup(self):
        """Stop the scheduled cleanup service."""
        self._running = False
        _logger.info("Stopped scheduled cleanup service")

    def get_cleanup_status(self) -> Dict[str, Any]:
        """
        Get status of scheduled cleanup tasks.

        Returns:
            Dictionary with cleanup status information
        """
        current_time = datetime.now(timezone.utc)

        task_status = []
        for task in self._cleanup_tasks:
            task_status.append({
                'name': task['name'],
                'interval_hours': task['interval_hours'],
                'last_run': task['last_run'].isoformat() if task['last_run'] else None,
                'next_run': task['next_run'].isoformat(),
                'overdue': current_time > task['next_run']
            })

        return {
            'running': self._running,
            'is_cleanup_time': self.is_cleanup_time(current_time),
            'tasks': task_status,
            'current_time': current_time.isoformat()
        }

    async def run_cleanup_daemon(self, check_interval_minutes: int = 60):
        """
        Run cleanup daemon that checks for due tasks periodically.

        Args:
            check_interval_minutes: Minutes between checks
        """
        _logger.info("Starting cleanup daemon (check interval: %d minutes)", check_interval_minutes)

        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                results = self.run_cleanup_cycle(current_time)

                if results:
                    _logger.info("Cleanup cycle completed with %d tasks", len(results))

                # Wait for next check
                await asyncio.sleep(check_interval_minutes * 60)

            except Exception as e:
                _logger.exception("Error in cleanup daemon:")
                # Wait a bit before retrying
                await asyncio.sleep(300)  # 5 minutes


# Utility functions for archival operations

def create_archival_service(session: Session, policy: Optional[ArchivalPolicy] = None) -> MessageArchivalService:
    """
    Create a message archival service instance.

    Args:
        session: Database session
        policy: Optional archival policy

    Returns:
        MessageArchivalService instance
    """
    return MessageArchivalService(session, policy)


def create_retention_policy_manager(session: Session) -> RetentionPolicyManager:
    """
    Create a retention policy manager instance.

    Args:
        session: Database session

    Returns:
        RetentionPolicyManager instance
    """
    return RetentionPolicyManager(session)


def create_scheduled_cleanup_service(session: Session) -> ScheduledCleanupService:
    """
    Create a scheduled cleanup service instance.

    Args:
        session: Database session

    Returns:
        ScheduledCleanupService instance
    """
    return ScheduledCleanupService(session)


def run_manual_archival(session: Session, policy: Optional[ArchivalPolicy] = None) -> Dict[str, ArchivalStats]:
    """
    Run manual archival operation.

    Args:
        session: Database session
        policy: Optional archival policy

    Returns:
        Dictionary with archival statistics
    """
    archival_service = MessageArchivalService(session, policy)
    return archival_service.run_full_archival_cycle()


def get_archival_statistics(session: Session) -> Dict[str, Any]:
    """
    Get current archival statistics.

    Args:
        session: Database session

    Returns:
        Dictionary with archival statistics
    """
    archival_service = MessageArchivalService(session)
    return archival_service.get_archival_statistics()