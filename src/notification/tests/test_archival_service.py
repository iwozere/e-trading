"""
Tests for Message Archival Service

Tests for archiving and cleanup functionality of the notification service.
"""

import json
import gzip
import tempfile
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.notification.service.archival_service import (
    MessageArchivalService, ArchivalPolicy, ArchivalStats, RetentionPolicyManager,
    ScheduledCleanupService, create_archival_service, create_retention_policy_manager,
    create_scheduled_cleanup_service, run_manual_archival
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestArchivalService:
    """Test cases for MessageArchivalService."""

    @pytest.fixture
    def session(self):
        """Create mock database session for testing."""
        return Mock()

    @pytest.fixture
    def temp_archive_dir(self):
        """Create temporary directory for archive testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def archival_policy(self, temp_archive_dir):
        """Create test archival policy."""
        return ArchivalPolicy(
            archive_after_days=30,
            delete_after_days=365,
            failed_message_retention_days=90,
            batch_size=10,
            max_messages_per_run=100,
            archive_path=temp_archive_dir,
            compress_archives=True,
            compression_level=6
        )

    @pytest.fixture
    def archival_service(self, session, archival_policy):
        """Create archival service for testing."""
        return MessageArchivalService(session, archival_policy)

    @pytest.fixture
    def sample_message_data(self):
        """Create sample message data for testing."""
        return {
            'id': 1,
            'message_type': 'test_message',
            'priority': 'NORMAL',
            'channels': ['telegram', 'email'],
            'recipient_id': 'user_1',
            'content': {'text': 'Test message', 'data': {'value': 1}},
            'message_metadata': {'test': True, 'index': 1},
            'created_at': datetime.now(timezone.utc) - timedelta(days=35),
            'scheduled_for': datetime.now(timezone.utc) - timedelta(days=35),
            'status': 'DELIVERED',
            'retry_count': 0,
            'max_retries': 3,
            'last_error': None,
            'processed_at': datetime.now(timezone.utc) - timedelta(days=35)
        }

    def test_archival_policy_creation(self, temp_archive_dir):
        """Test archival policy creation and defaults."""
        policy = ArchivalPolicy(archive_path=temp_archive_dir)

        assert policy.archive_after_days == 30
        assert policy.delete_after_days == 365
        assert policy.failed_message_retention_days == 90
        assert policy.batch_size == 1000
        assert policy.compress_archives is True
        assert policy.archive_path == temp_archive_dir

    def test_archival_service_initialization(self, session, archival_policy):
        """Test archival service initialization."""
        service = MessageArchivalService(session, archival_policy)

        assert service.session == session
        assert service.policy == archival_policy
        assert service.archive_path.exists()
        assert str(service.archive_path) == archival_policy.archive_path

    def test_is_low_traffic_period(self, archival_service):
        """Test low traffic period detection."""
        # Test during low traffic hours (2:00-6:00)
        low_traffic_time = datetime(2024, 1, 1, 3, 0, 0)  # 3:00 AM
        assert archival_service.is_low_traffic_period(low_traffic_time) is True

        # Test during high traffic hours
        high_traffic_time = datetime(2024, 1, 1, 14, 0, 0)  # 2:00 PM
        assert archival_service.is_low_traffic_period(high_traffic_time) is False

        # Test edge cases
        start_time = datetime(2024, 1, 1, 2, 0, 0)  # 2:00 AM (start)
        assert archival_service.is_low_traffic_period(start_time) is True

        end_time = datetime(2024, 1, 1, 6, 0, 0)  # 6:00 AM (end)
        assert archival_service.is_low_traffic_period(end_time) is False

    def test_archive_message_data(self, archival_service, sample_message_data):
        """Test archiving message data structure."""
        # Mock the repository method
        archival_service.repo.delivery_status.get_delivery_statuses_by_message = Mock(return_value=[])

        # Create a mock message object
        mock_message = Mock()
        for key, value in sample_message_data.items():
            setattr(mock_message, key, value)

        archive_data = archival_service.archive_message(mock_message)

        # Verify archive data structure
        assert archive_data['id'] == sample_message_data['id']
        assert archive_data['message_type'] == sample_message_data['message_type']
        assert archive_data['priority'] == sample_message_data['priority']
        assert archive_data['channels'] == sample_message_data['channels']
        assert archive_data['content'] == sample_message_data['content']
        assert 'archived_at' in archive_data
        assert 'delivery_statuses' in archive_data

    def test_save_archived_messages(self, archival_service, sample_message_data):
        """Test saving archived messages to file."""
        # Create mock message and archive data
        mock_message = Mock()
        for key, value in sample_message_data.items():
            setattr(mock_message, key, value)

        archival_service.repo.delivery_status.get_delivery_statuses_by_message = Mock(return_value=[])

        archived_data = [archival_service.archive_message(mock_message)]

        file_path, bytes_written = archival_service.save_archived_messages(archived_data)

        # Verify file was created
        assert file_path != ""
        assert bytes_written > 0
        assert Path(file_path).exists()

        # Verify file content
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                saved_data = json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)

        assert saved_data['message_count'] == len(archived_data)
        assert len(saved_data['messages']) == len(archived_data)
        assert 'archived_at' in saved_data

    def test_run_full_archival_cycle_outside_low_traffic(self, archival_service):
        """Test full archival cycle outside low traffic period."""
        # Run during high traffic time
        high_traffic_time = datetime(2024, 1, 1, 14, 0, 0)  # 2:00 PM

        results = archival_service.run_full_archival_cycle(high_traffic_time)

        # Should return empty results since not in low traffic period
        assert len(results) == 0

    def test_run_full_archival_cycle_during_low_traffic(self, archival_service):
        """Test full archival cycle during low traffic period."""
        # Mock the individual batch methods to return empty stats
        archival_service.archive_messages_batch = Mock(return_value=ArchivalStats())
        archival_service.cleanup_archived_messages_batch = Mock(return_value=ArchivalStats())
        archival_service.cleanup_failed_messages_batch = Mock(return_value=ArchivalStats())

        # Run during low traffic time
        current_time = datetime(2024, 1, 1, 3, 0, 0)  # 3:00 AM (low traffic)
        results = archival_service.run_full_archival_cycle(current_time)

        # Should have results from archival operations
        assert 'archival' in results
        assert 'archived_cleanup' in results
        assert 'failed_cleanup' in results

        # Verify archival stats
        archival_stats = results['archival']
        assert isinstance(archival_stats, ArchivalStats)

    def test_get_archival_statistics(self, archival_service):
        """Test getting archival statistics."""
        # Mock the session queries
        archival_service.session.query = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.scalar.return_value = 5
        archival_service.session.query.return_value = mock_query

        stats = archival_service.get_archival_statistics()

        # Verify statistics structure
        assert 'policy' in stats
        assert 'current_status' in stats
        assert 'archive_path' in stats
        assert 'last_checked' in stats

        # Verify policy information
        policy_info = stats['policy']
        assert 'archive_after_days' in policy_info
        assert 'delete_after_days' in policy_info
        assert 'failed_message_retention_days' in policy_info

        # Verify current status
        status_info = stats['current_status']
        assert 'messages_ready_for_archival' in status_info
        assert 'failed_messages_ready_for_cleanup' in status_info
        assert 'archived_files_count' in status_info
        assert 'is_low_traffic_period' in status_info


class TestRetentionPolicyManager:
    """Test cases for RetentionPolicyManager."""

    @pytest.fixture
    def session(self):
        """Mock session for testing."""
        return Mock()

    @pytest.fixture
    def policy_manager(self, session):
        """Create retention policy manager for testing."""
        return RetentionPolicyManager(session)

    def test_default_policies_loaded(self, policy_manager):
        """Test that default policies are loaded."""
        policies = policy_manager.list_policies()

        assert 'default' in policies
        assert 'high_priority' in policies
        assert 'system_alerts' in policies
        assert 'user_notifications' in policies

    def test_get_policy_by_priority(self, policy_manager):
        """Test getting policy by message priority."""
        # High priority message
        policy = policy_manager.get_policy(priority='HIGH')
        assert policy.archive_after_days == 60  # High priority policy

        # Normal priority message
        policy = policy_manager.get_policy(priority='NORMAL')
        assert policy.archive_after_days == 30  # Default policy

    def test_get_policy_by_message_type(self, policy_manager):
        """Test getting policy by message type."""
        # System alert message
        policy = policy_manager.get_policy(message_type='system_alert')
        assert policy.archive_after_days == 90  # System alerts policy

        # User notification message
        policy = policy_manager.get_policy(message_type='user_notification')
        assert policy.archive_after_days == 14  # User notifications policy

    def test_update_policy(self, policy_manager):
        """Test updating a retention policy."""
        new_policy = ArchivalPolicy(archive_after_days=45)
        policy_manager.update_policy('test_policy', new_policy)

        policies = policy_manager.list_policies()
        assert 'test_policy' in policies
        assert policies['test_policy'].archive_after_days == 45


class TestScheduledCleanupService:
    """Test cases for ScheduledCleanupService."""

    @pytest.fixture
    def session(self):
        """Mock session for testing."""
        return Mock()

    @pytest.fixture
    def cleanup_service(self, session):
        """Create scheduled cleanup service for testing."""
        return ScheduledCleanupService(session)

    def test_is_cleanup_time(self, cleanup_service):
        """Test cleanup time detection."""
        # During low traffic hours
        low_traffic_time = datetime(2024, 1, 1, 3, 0, 0)
        assert cleanup_service.is_cleanup_time(low_traffic_time) is True

        # During high traffic hours
        high_traffic_time = datetime(2024, 1, 1, 14, 0, 0)
        assert cleanup_service.is_cleanup_time(high_traffic_time) is False

    def test_schedule_cleanup_task(self, cleanup_service):
        """Test scheduling cleanup tasks."""
        def dummy_task(current_time):
            return {'result': 'success'}

        cleanup_service.schedule_cleanup_task('test_task', dummy_task, 24)

        status = cleanup_service.get_cleanup_status()
        assert len(status['tasks']) == 1
        assert status['tasks'][0]['name'] == 'test_task'
        assert status['tasks'][0]['interval_hours'] == 24

    def test_run_cleanup_cycle_outside_window(self, cleanup_service):
        """Test cleanup cycle outside cleanup window."""
        high_traffic_time = datetime(2024, 1, 1, 14, 0, 0)

        results = cleanup_service.run_cleanup_cycle(high_traffic_time)

        # Should return empty results
        assert len(results) == 0

    def test_run_cleanup_cycle_with_tasks(self, cleanup_service):
        """Test cleanup cycle with scheduled tasks."""
        def dummy_task(current_time):
            return {'messages_processed': 10}

        # Schedule a task
        cleanup_service.schedule_cleanup_task('test_task', dummy_task, 24)

        # Run during low traffic time
        low_traffic_time = datetime(2024, 1, 1, 3, 0, 0)
        results = cleanup_service.run_cleanup_cycle(low_traffic_time)

        # Should have executed the task
        assert 'test_task' in results
        assert results['test_task']['messages_processed'] == 10

    def test_start_stop_scheduled_cleanup(self, cleanup_service):
        """Test starting and stopping scheduled cleanup."""
        # Initially not running
        status = cleanup_service.get_cleanup_status()
        assert status['running'] is False

        # Start service
        cleanup_service.start_scheduled_cleanup()
        status = cleanup_service.get_cleanup_status()
        assert status['running'] is True
        assert len(status['tasks']) > 0  # Should have default tasks

        # Stop service
        cleanup_service.stop_scheduled_cleanup()
        status = cleanup_service.get_cleanup_status()
        assert status['running'] is False


class TestArchivalUtilityFunctions:
    """Test cases for archival utility functions."""

    @pytest.fixture
    def session(self):
        """Mock session for testing."""
        return Mock()

    def test_create_archival_service(self, session):
        """Test creating archival service."""
        service = create_archival_service(session)

        assert isinstance(service, MessageArchivalService)
        assert service.session == session

    def test_create_retention_policy_manager(self, session):
        """Test creating retention policy manager."""
        manager = create_retention_policy_manager(session)

        assert isinstance(manager, RetentionPolicyManager)
        assert manager.session == session

    def test_create_scheduled_cleanup_service(self, session):
        """Test creating scheduled cleanup service."""
        service = create_scheduled_cleanup_service(session)

        assert isinstance(service, ScheduledCleanupService)
        assert service.session == session

    @patch('src.notification.service.archival_service.MessageArchivalService')
    def test_run_manual_archival(self, mock_service_class, session):
        """Test running manual archival."""
        # Mock the service and its methods
        mock_service = Mock()
        mock_service.run_full_archival_cycle.return_value = {'archival': ArchivalStats()}
        mock_service_class.return_value = mock_service

        results = run_manual_archival(session)

        # Verify service was created and method was called
        mock_service_class.assert_called_once_with(session, None)
        mock_service.run_full_archival_cycle.assert_called_once()
        assert 'archival' in results


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])