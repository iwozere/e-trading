"""
Integration Tests for Scheduler Service

Tests the complete scheduler service functionality including:
- Job loading and APScheduler registration
- Job execution and callback handling
- Error recovery and retry mechanisms
"""

import pytest
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add src to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.scheduler.scheduler_service import SchedulerService
from src.data.db.services.jobs_service import JobsService
from src.data.db.models.model_jobs import Schedule, ScheduleRun, JobType, RunStatus
from src.common.alerts.alert_evaluator import AlertEvaluator, AlertEvaluationResult
from src.common.alerts.cron_parser import CronParser


class MockSchedule:
    """Mock Schedule object for testing."""

    def __init__(self, schedule_id: int, name: str, job_type: str, cron: str,
                 enabled: bool = True, user_id: int = 1, target: str = "test",
                 task_params: Dict[str, Any] = None):
        self.id = schedule_id
        self.name = name
        self.job_type = job_type
        self.cron = cron
        self.enabled = enabled
        self.user_id = user_id
        self.target = target
        self.task_params = task_params or {}
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.next_run_at = None


class MockScheduleRun:
    """Mock ScheduleRun object for testing."""

    def __init__(self, run_id: int, schedule_id: int, status: str = "PENDING"):
        self.id = run_id
        self.schedule_id = schedule_id
        self.status = status
        self.job_type = "alert"
        self.job_id = str(schedule_id)
        self.user_id = 1
        self.scheduled_for = datetime.now(timezone.utc)
        self.started_at = None
        self.finished_at = None
        self.result = None
        self.error = None
        self.job_snapshot = {}


@pytest.fixture
def mock_jobs_service():
    """Create a mock JobsService."""
    service = Mock(spec=JobsService)

    # Mock schedules
    test_schedules = [
        MockSchedule(1, "Test Alert 1", "alert", "0 9 * * *"),
        MockSchedule(2, "Test Alert 2", "alert", "*/5 * * * *"),
        MockSchedule(3, "Disabled Alert", "alert", "0 10 * * *", enabled=False),
        MockSchedule(4, "Test Screener", "screener", "0 8 * * 1-5"),
    ]

    service.list_schedules.return_value = [s for s in test_schedules if s.enabled]
    service.get_schedule.side_effect = lambda sid: next((s for s in test_schedules if s.id == sid), None)

    # Mock run creation
    service.create_run.return_value = MockScheduleRun(1, 1)
    service.update_run.return_value = MockScheduleRun(1, 1, "COMPLETED")
    service.update_schedule_next_run.return_value = True

    return service


@pytest.fixture
def mock_alert_evaluator():
    """Create a mock AlertEvaluator."""
    evaluator = Mock(spec=AlertEvaluator)

    # Default successful evaluation
    evaluator.evaluate_alert.return_value = AlertEvaluationResult(
        triggered=True,
        rearmed=False,
        state_updates={"status": "TRIGGERED", "last_triggered": datetime.now(timezone.utc).isoformat()},
        notification_data={"ticker": "AAPL", "price": 150.0},
        error=None
    )

    return evaluator


@pytest.fixture
def scheduler_service(mock_jobs_service, mock_alert_evaluator):
    """Create a SchedulerService with mocked dependencies."""
    return SchedulerService(
        jobs_service=mock_jobs_service,
        alert_evaluator=mock_alert_evaluator,
        database_url="sqlite:///:memory:",
        max_workers=2
    )


class TestSchedulerServiceInitialization:
    """Test scheduler service initialization and lifecycle."""

    def test_scheduler_service_creation(self, scheduler_service):
        """Test that scheduler service can be created with proper configuration."""
        assert scheduler_service.jobs_service is not None
        assert scheduler_service.alert_evaluator is not None
        assert scheduler_service.database_url == "sqlite:///:memory:"
        assert scheduler_service.max_workers == 2
        assert not scheduler_service.is_running
        assert scheduler_service.startup_retry_count == 0

    def test_get_scheduler_status(self, scheduler_service):
        """Test scheduler status reporting."""
        status = scheduler_service.get_scheduler_status()

        assert isinstance(status, dict)
        assert "is_running" in status
        assert "startup_retry_count" in status
        assert "max_workers" in status
        assert "scheduler_state" in status
        assert "job_count" in status
        assert "datastore_url" in status

        assert status["is_running"] == False
        assert status["max_workers"] == 2
        assert status["datastore_url"] == "sqlite:///:memory:"


class TestScheduleLoading:
    """Test schedule loading and registration functionality."""

    @pytest.mark.asyncio
    async def test_load_and_register_schedules(self, scheduler_service, mock_jobs_service):
        """Test loading schedules from database and registering with APScheduler."""
        with patch.object(scheduler_service, '_register_schedule', new_callable=AsyncMock) as mock_register:
            # Test loading schedules
            count = await scheduler_service._load_and_register_schedules()

            # Should load 3 enabled schedules (excluding disabled one)
            assert count == 3
            assert mock_register.call_count == 3

            # Verify correct schedules were registered
            registered_schedules = [call[0][0] for call in mock_register.call_args_list]
            registered_ids = [s.id for s in registered_schedules]
            assert 1 in registered_ids
            assert 2 in registered_ids
            assert 4 in registered_ids
            assert 3 not in registered_ids  # Disabled schedule should not be registered

    @pytest.mark.asyncio
    async def test_register_schedule_with_valid_cron(self, scheduler_service):
        """Test registering a schedule with valid cron expression."""
        mock_schedule = MockSchedule(1, "Test Alert", "alert", "0 9 * * *")

        # Mock the scheduler to avoid actual APScheduler initialization
        mock_scheduler_instance = Mock()
        scheduler_service.scheduler = mock_scheduler_instance

        # Test registration
        await scheduler_service._register_schedule(mock_schedule)

        # Verify job was added to scheduler
        mock_scheduler_instance.add_job.assert_called_once()
        job_call = mock_scheduler_instance.add_job.call_args
        assert job_call[1]['id'] == 'schedule_1'
        assert job_call[1]['args'] == [1]

    @pytest.mark.asyncio
    async def test_register_schedule_with_invalid_cron(self, scheduler_service):
        """Test registering a schedule with invalid cron expression."""
        mock_schedule = MockSchedule(1, "Test Alert", "alert", "invalid cron")

        with patch.object(CronParser, 'validate_cron', return_value=False):
            # Test registration should raise ValueError
            with pytest.raises(ValueError, match="Invalid cron expression"):
                await scheduler_service._register_schedule(mock_schedule)


class TestJobExecution:
    """Test job execution and callback handling."""

    @pytest.mark.asyncio
    async def test_execute_alert_job_success(self, scheduler_service, mock_jobs_service, mock_alert_evaluator):
        """Test successful alert job execution."""
        # Setup test schedule
        test_schedule = MockSchedule(1, "Test Alert", "alert", "0 9 * * *")
        mock_jobs_service.get_schedule.return_value = test_schedule

        # Setup successful alert evaluation
        mock_alert_evaluator.evaluate_alert.return_value = AlertEvaluationResult(
            triggered=True,
            rearmed=False,
            state_updates={"status": "TRIGGERED"},
            notification_data={"ticker": "AAPL", "price": 150.0},
            error=None
        )

        with patch.object(scheduler_service, '_create_run_record', return_value=MockScheduleRun(1, 1)) as mock_create, \
             patch.object(scheduler_service, '_complete_run_record', new_callable=AsyncMock) as mock_complete, \
             patch.object(scheduler_service, '_update_schedule_state', new_callable=AsyncMock) as mock_update_state, \
             patch.object(scheduler_service, '_send_notification', new_callable=AsyncMock) as mock_notify:

            # Execute job
            await scheduler_service._execute_job(1)

            # Verify job execution flow
            mock_create.assert_called_once()
            mock_alert_evaluator.evaluate_alert.assert_called_once()
            mock_update_state.assert_called_once_with(1, {"status": "TRIGGERED"})
            mock_notify.assert_called_once()
            mock_complete.assert_called_once()
            mock_jobs_service.update_schedule_next_run.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_execute_alert_job_failure(self, scheduler_service, mock_jobs_service, mock_alert_evaluator):
        """Test alert job execution with failure."""
        # Setup test schedule
        test_schedule = MockSchedule(1, "Test Alert", "alert", "0 9 * * *")
        mock_jobs_service.get_schedule.return_value = test_schedule

        # Setup alert evaluation failure
        mock_alert_evaluator.evaluate_alert.side_effect = Exception("Alert evaluation failed")

        with patch.object(scheduler_service, '_create_run_record', return_value=MockScheduleRun(1, 1)) as mock_create, \
             patch.object(scheduler_service, '_complete_run_record', new_callable=AsyncMock) as mock_complete:

            # Execute job
            await scheduler_service._execute_job(1)

            # Verify failure handling
            mock_create.assert_called_once()
            mock_alert_evaluator.evaluate_alert.assert_called_once()

            # Verify run was marked as failed
            mock_complete.assert_called_once()
            complete_call = mock_complete.call_args
            assert complete_call[0][1] == RunStatus.FAILED  # Status should be FAILED
            assert "Alert evaluation failed" in complete_call[0][3]  # Error message

    @pytest.mark.asyncio
    async def test_execute_job_schedule_not_found(self, scheduler_service, mock_jobs_service):
        """Test job execution when schedule is not found."""
        # Setup schedule not found
        mock_jobs_service.get_schedule.return_value = None

        # Execute job - should handle gracefully
        await scheduler_service._execute_job(999)

        # Verify no further processing occurred
        mock_jobs_service.create_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_job_schedule_disabled(self, scheduler_service, mock_jobs_service):
        """Test job execution when schedule is disabled."""
        # Setup disabled schedule
        disabled_schedule = MockSchedule(1, "Disabled Alert", "alert", "0 9 * * *", enabled=False)
        mock_jobs_service.get_schedule.return_value = disabled_schedule

        # Execute job - should handle gracefully
        await scheduler_service._execute_job(1)

        # Verify no further processing occurred
        mock_jobs_service.create_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_run_record(self, scheduler_service, mock_jobs_service):
        """Test creation of ScheduleRun record."""
        test_schedule = MockSchedule(1, "Test Alert", "alert", "0 9 * * *")
        test_run = MockScheduleRun(1, 1)

        mock_jobs_service.create_run.return_value = test_run
        mock_jobs_service.update_run.return_value = test_run

        # Create run record
        result = await scheduler_service._create_run_record(test_schedule)

        # Verify run creation
        assert result is not None
        mock_jobs_service.create_run.assert_called_once()
        mock_jobs_service.update_run.assert_called_once()

        # Verify job snapshot contains schedule details
        create_call = mock_jobs_service.create_run.call_args
        job_snapshot = create_call[0][1].job_snapshot
        assert job_snapshot["schedule_id"] == 1
        assert job_snapshot["schedule_name"] == "Test Alert"
        assert job_snapshot["job_type"] == "alert"

    @pytest.mark.asyncio
    async def test_complete_run_record(self, scheduler_service, mock_jobs_service):
        """Test completion of ScheduleRun record."""
        test_run = MockScheduleRun(1, 1)
        test_result = {"triggered": True, "notification_sent": True}

        # Complete run record
        await scheduler_service._complete_run_record(
            test_run, RunStatus.COMPLETED, test_result
        )

        # Verify update was called
        mock_jobs_service.update_run.assert_called_once()
        update_call = mock_jobs_service.update_run.call_args

        assert update_call[0][0] == 1  # Run ID
        update_data = update_call[0][1]
        assert update_data.status == RunStatus.COMPLETED
        assert update_data.result == test_result
        assert update_data.finished_at is not None


class TestErrorRecovery:
    """Test error handling and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_reload_schedules_not_running(self, scheduler_service):
        """Test schedule reloading when service is not running."""
        scheduler_service.is_running = False

        # Test reload should raise RuntimeError
        with pytest.raises(RuntimeError, match="Cannot reload schedules - service is not running"):
            await scheduler_service.reload_schedules()

    @pytest.mark.asyncio
    async def test_clear_all_jobs(self, scheduler_service):
        """Test clearing all jobs from scheduler."""
        mock_job1 = Mock()
        mock_job1.id = "job1"
        mock_job2 = Mock()
        mock_job2.id = "job2"

        mock_scheduler = Mock()
        mock_scheduler.get_jobs.return_value = [mock_job1, mock_job2]
        mock_scheduler.remove_job = Mock()

        scheduler_service.scheduler = mock_scheduler

        # Clear all jobs
        await scheduler_service._clear_all_jobs()

        # Verify all jobs were removed
        mock_scheduler.get_jobs.assert_called_once()
        assert mock_scheduler.remove_job.call_count == 2
        mock_scheduler.remove_job.assert_any_call("job1")
        mock_scheduler.remove_job.assert_any_call("job2")


class TestEventHandlers:
    """Test APScheduler event handlers."""

    def test_job_submitted_event_handler(self, scheduler_service):
        """Test job submission event handling."""
        # Create mock event
        event = Mock()
        event.job_id = "test_job_1"
        event.scheduled_run_time = datetime.now(timezone.utc)

        # Test event handler
        scheduler_service._on_job_submitted(event)

        # Should not raise any exceptions (just logs)

    def test_job_executed_event_handler(self, scheduler_service):
        """Test job execution completion event handling."""
        # Create mock event
        event = Mock()
        event.job_id = "test_job_1"

        # Test event handler
        scheduler_service._on_job_executed(event)

        # Should not raise any exceptions (just logs)

    def test_job_error_event_handler(self, scheduler_service):
        """Test job error event handling."""
        # Create mock event
        event = Mock()
        event.job_id = "test_job_1"
        event.exception = Exception("Test error")

        # Test event handler
        scheduler_service._on_job_error(event)

        # Should not raise any exceptions (just logs)

    def test_job_missed_event_handler(self, scheduler_service):
        """Test missed job event handling."""
        # Create mock event
        event = Mock()
        event.job_id = "test_job_1"
        event.scheduled_run_time = datetime.now(timezone.utc)

        # Test event handler
        scheduler_service._on_job_missed(event)

        # Should not raise any exceptions (just logs)


# Integration test that combines multiple components
class TestSchedulerServiceIntegration:
    """Integration tests that test multiple components working together."""

    @pytest.mark.asyncio
    async def test_full_alert_execution_flow(self, scheduler_service, mock_jobs_service, mock_alert_evaluator):
        """Test complete alert execution flow from schedule to notification."""
        # Setup test data
        test_schedule = MockSchedule(
            1, "Integration Test Alert", "alert", "0 9 * * *",
            task_params={
                "ticker": "AAPL",
                "timeframe": "1h",
                "rule": {"gt": {"lhs": {"field": "close"}, "rhs": {"value": 150.0}}}
            }
        )
        mock_jobs_service.get_schedule.return_value = test_schedule

        # Setup alert evaluation result
        mock_alert_evaluator.evaluate_alert.return_value = AlertEvaluationResult(
            triggered=True,
            rearmed=False,
            state_updates={"status": "TRIGGERED", "last_triggered": datetime.now(timezone.utc).isoformat()},
            notification_data={
                "ticker": "AAPL",
                "price": 155.0,
                "message": "AAPL crossed above $150"
            },
            error=None
        )

        with patch.object(scheduler_service, '_create_run_record', return_value=MockScheduleRun(1, 1)) as mock_create, \
             patch.object(scheduler_service, '_complete_run_record', new_callable=AsyncMock) as mock_complete, \
             patch.object(scheduler_service, '_update_schedule_state', new_callable=AsyncMock) as mock_update_state, \
             patch.object(scheduler_service, '_send_notification', new_callable=AsyncMock) as mock_notify:

            # Execute the complete flow
            await scheduler_service._execute_job(1)

            # Verify complete execution flow
            mock_create.assert_called_once()
            mock_alert_evaluator.evaluate_alert.assert_called_once()
            mock_update_state.assert_called_once()
            mock_notify.assert_called_once()
            mock_complete.assert_called_once()

            # Verify notification data
            notify_call = mock_notify.call_args
            notification_data = notify_call[0][0]
            assert notification_data["ticker"] == "AAPL"
            assert notification_data["price"] == 155.0

            # Verify run completion
            complete_call = mock_complete.call_args
            assert complete_call[0][1] == RunStatus.COMPLETED
            result = complete_call[0][2]
            assert result["triggered"] == True
            assert result["notification_sent"] == True

    @pytest.mark.asyncio
    async def test_multiple_job_types_execution(self, scheduler_service, mock_jobs_service):
        """Test execution of different job types."""
        # Test alert job
        alert_schedule = MockSchedule(1, "Alert Job", "alert", "0 9 * * *")
        mock_jobs_service.get_schedule.return_value = alert_schedule

        with patch.object(scheduler_service, '_execute_alert_job', new_callable=AsyncMock, return_value={"triggered": True}) as mock_alert, \
             patch.object(scheduler_service, '_create_run_record', return_value=MockScheduleRun(1, 1)), \
             patch.object(scheduler_service, '_complete_run_record', new_callable=AsyncMock):

            await scheduler_service._execute_job(1)
            mock_alert.assert_called_once()

        # Test screener job
        screener_schedule = MockSchedule(2, "Screener Job", "screener", "0 8 * * *")
        mock_jobs_service.get_schedule.return_value = screener_schedule

        with patch.object(scheduler_service, '_execute_screener_job', new_callable=AsyncMock, return_value={"status": "not_implemented"}) as mock_screener, \
             patch.object(scheduler_service, '_create_run_record', return_value=MockScheduleRun(2, 2)), \
             patch.object(scheduler_service, '_complete_run_record', new_callable=AsyncMock):

            await scheduler_service._execute_job(2)
            mock_screener.assert_called_once()

        # Test report job
        report_schedule = MockSchedule(3, "Report Job", "report", "0 7 * * *")
        mock_jobs_service.get_schedule.return_value = report_schedule

        with patch.object(scheduler_service, '_execute_report_job', new_callable=AsyncMock, return_value={"status": "not_implemented"}) as mock_report, \
             patch.object(scheduler_service, '_create_run_record', return_value=MockScheduleRun(3, 3)), \
             patch.object(scheduler_service, '_complete_run_record', new_callable=AsyncMock):

            await scheduler_service._execute_job(3)
            mock_report.assert_called_once()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])