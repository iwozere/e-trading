"""
Tests for Jobs Service

Tests the service layer for job scheduling and execution operations.
Uses monkeypatch to mock database_service for isolated testing.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# Models
from src.data.db.models.model_jobs import Base as JobsBase, JobType, RunStatus, ScheduleCreate, ScheduleUpdate, ScheduleRunCreate, ScheduleRunUpdate

# Service under test
from src.data.db.services.jobs_service import JobsService


# Use the shared engine and dbsess fixtures from conftest.py


@pytest.fixture()
def jobs_service(dbsess):
    """Create JobsService instance with mocked screener config."""
    with patch('src.data.db.services.jobs_service.get_screener_config') as mock_config:
        # Mock screener config
        mock_screener = Mock()
        mock_screener.validate_set_name.return_value = True
        mock_screener.get_tickers.return_value = ["AAPL", "GOOGL", "MSFT"]
        mock_config.return_value = mock_screener

        service = JobsService(dbsess)
        service.screener_config = mock_screener
        yield service


def test_create_schedule_success(jobs_service, dbsess):
    """Test creating a new schedule successfully."""
    schedule_data = ScheduleCreate(
        name="Daily Report",
        job_type=JobType.REPORT,
        target="portfolio_summary",
        task_params={"format": "pdf"},
        cron="0 9 * * *",
        enabled=True
    )

    schedule = jobs_service.create_schedule(user_id=1, schedule_data=schedule_data)

    assert schedule.id is not None
    assert schedule.name == "Daily Report"
    assert schedule.job_type == JobType.REPORT.value
    assert schedule.enabled is True
    assert schedule.next_run_at is not None


def test_create_schedule_invalid_cron(jobs_service, dbsess):
    """Test creating schedule with invalid cron expression fails."""
    schedule_data = ScheduleCreate(
        name="Invalid Cron",
        job_type=JobType.REPORT,
        target="test",
        cron="invalid cron",  # Invalid cron expression
        enabled=True
    )

    with pytest.raises(ValueError, match="Invalid cron expression"):
        jobs_service.create_schedule(user_id=1, schedule_data=schedule_data)


def test_get_schedule(jobs_service, dbsess):
    """Test getting a schedule by ID."""
    schedule_data = ScheduleCreate(
        name="Test Schedule",
        job_type=JobType.SCREENER,
        target="tech_stocks",
        cron="0 10 * * *"
    )

    created = jobs_service.create_schedule(user_id=1, schedule_data=schedule_data)
    retrieved = jobs_service.get_schedule(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Test Schedule"


def test_list_schedules_with_filters(jobs_service, dbsess):
    """Test listing schedules with filters."""
    # Create multiple schedules
    schedule1 = ScheduleCreate(name="Report1", job_type=JobType.REPORT, target="r1", cron="0 9 * * *", enabled=True)
    schedule2 = ScheduleCreate(name="Screener1", job_type=JobType.SCREENER, target="s1", cron="0 10 * * *", enabled=False)

    jobs_service.create_schedule(user_id=1, schedule_data=schedule1)
    jobs_service.create_schedule(user_id=1, schedule_data=schedule2)
    jobs_service.create_schedule(user_id=2, schedule_data=ScheduleCreate(name="Report2", job_type=JobType.REPORT, target="r2", cron="0 11 * * *"))

    # Test filtering
    user1_schedules = jobs_service.list_schedules(user_id=1)
    assert len(user1_schedules) == 2

    report_schedules = jobs_service.list_schedules(job_type=JobType.REPORT)
    assert len(report_schedules) == 2

    enabled_schedules = jobs_service.list_schedules(enabled=True)
    assert len(enabled_schedules) == 2  # schedule1 and the default enabled schedule for user 2


def test_update_schedule(jobs_service, dbsess):
    """Test updating a schedule."""
    schedule_data = ScheduleCreate(
        name="Original",
        job_type=JobType.REPORT,
        target="original",
        cron="0 9 * * *",
        enabled=True
    )

    created = jobs_service.create_schedule(user_id=1, schedule_data=schedule_data)

    # Update schedule
    update_data = ScheduleUpdate(
        name="Updated",
        enabled=False,
        cron="0 10 * * *"
    )

    updated = jobs_service.update_schedule(created.id, update_data)

    assert updated is not None
    assert updated.name == "Updated"
    assert updated.enabled is False
    assert updated.cron == "0 10 * * *"
    assert updated.next_run_at != created.next_run_at  # Should be recalculated


def test_delete_schedule(jobs_service, dbsess):
    """Test deleting a schedule."""
    schedule_data = ScheduleCreate(
        name="To Delete",
        job_type=JobType.REPORT,
        target="delete_me",
        cron="0 9 * * *"
    )

    created = jobs_service.create_schedule(user_id=1, schedule_data=schedule_data)

    success = jobs_service.delete_schedule(created.id)
    assert success is True

    # Verify it's gone
    retrieved = jobs_service.get_schedule(created.id)
    assert retrieved is None


def test_trigger_schedule(jobs_service, dbsess):
    """Test manually triggering a schedule."""
    schedule_data = ScheduleCreate(
        name="Trigger Test",
        job_type=JobType.REPORT,
        target="trigger_target",
        cron="0 9 * * *",
        enabled=True
    )

    created = jobs_service.create_schedule(user_id=1, schedule_data=schedule_data)

    # Trigger the schedule
    run = jobs_service.trigger_schedule(created.id)

    assert run is not None
    assert run.job_type == JobType.REPORT.value
    assert run.user_id == 1
    assert "manual" in run.job_snapshot["trigger_type"]
    assert run.job_snapshot["schedule_id"] == created.id


def test_trigger_disabled_schedule_fails(jobs_service, dbsess):
    """Test that triggering disabled schedule fails."""
    schedule_data = ScheduleCreate(
        name="Disabled",
        job_type=JobType.REPORT,
        target="disabled",
        cron="0 9 * * *",
        enabled=False
    )

    created = jobs_service.create_schedule(user_id=1, schedule_data=schedule_data)

    with pytest.raises(ValueError, match="Cannot trigger disabled schedule"):
        jobs_service.trigger_schedule(created.id)


def test_create_run(jobs_service, dbsess):
    """Test creating a new run."""
    run_data = ScheduleRunCreate(
        job_type=JobType.REPORT,
        job_id="test_job_123",
        scheduled_for=datetime.now(timezone.utc),
        job_snapshot={"param1": "value1"}
    )

    run = jobs_service.create_run(user_id=1, run_data=run_data)

    assert run.id is not None
    assert run.job_type == JobType.REPORT.value
    assert run.job_id == "test_job_123"
    assert run.user_id == 1
    assert run.status == RunStatus.PENDING.value


def test_list_runs_with_filters(jobs_service, dbsess):
    """Test listing runs with filters."""
    now = datetime.now(timezone.utc)

    # Create multiple runs
    run1_data = ScheduleRunCreate(job_type=JobType.REPORT, job_id="r1", scheduled_for=now, job_snapshot={})
    run2_data = ScheduleRunCreate(job_type=JobType.SCREENER, job_id="s1", scheduled_for=now, job_snapshot={})

    run1 = jobs_service.create_run(user_id=1, run_data=run1_data)
    run2 = jobs_service.create_run(user_id=1, run_data=run2_data)
    jobs_service.create_run(user_id=2, run_data=ScheduleRunCreate(job_type=JobType.REPORT, job_id="r2", scheduled_for=now, job_snapshot={}))

    # Update one run status
    jobs_service.update_run(run1.id, ScheduleRunUpdate(status=RunStatus.COMPLETED))

    # Test filtering
    user1_runs = jobs_service.list_runs(user_id=1)
    assert len(user1_runs) == 2

    report_runs = jobs_service.list_runs(job_type=JobType.REPORT)
    assert len(report_runs) == 2

    completed_runs = jobs_service.list_runs(status=RunStatus.COMPLETED)
    assert len(completed_runs) == 1


def test_update_run(jobs_service, dbsess):
    """Test updating a run."""
    run_data = ScheduleRunCreate(
        job_type=JobType.REPORT,
        job_id="update_test",
        scheduled_for=datetime.now(timezone.utc),
        job_snapshot={}
    )

    created = jobs_service.create_run(user_id=1, run_data=run_data)

    # Update run
    now = datetime.now(timezone.utc)
    update_data = ScheduleRunUpdate(
        status=RunStatus.RUNNING,
        started_at=now,
        worker_id="worker_123"
    )

    updated = jobs_service.update_run(created.id, update_data)

    assert updated is not None
    assert updated.status == RunStatus.RUNNING.value
    assert updated.started_at == now
    assert updated.worker_id == "worker_123"


def test_claim_run(jobs_service, dbsess):
    """Test claiming a run for execution."""
    run_data = ScheduleRunCreate(
        job_type=JobType.SCREENER,
        job_id="claim_test",
        scheduled_for=datetime.now(timezone.utc),
        job_snapshot={}
    )

    created = jobs_service.create_run(user_id=1, run_data=run_data)

    # Claim the run
    claimed = jobs_service.claim_run(created.id, "worker_456")

    assert claimed is not None
    assert claimed.status == RunStatus.RUNNING.value
    assert claimed.worker_id == "worker_456"
    assert claimed.started_at is not None


def test_cancel_run(jobs_service, dbsess):
    """Test cancelling a pending run."""
    run_data = ScheduleRunCreate(
        job_type=JobType.REPORT,
        job_id="cancel_test",
        scheduled_for=datetime.now(timezone.utc),
        job_snapshot={}
    )

    created = jobs_service.create_run(user_id=1, run_data=run_data)

    # Cancel the run
    success = jobs_service.cancel_run(created.id)
    assert success is True

    # Verify status
    updated = jobs_service.get_run(created.id)
    assert updated.status == RunStatus.CANCELLED.value


def test_cancel_running_run_fails(jobs_service, dbsess):
    """Test that cancelling running run fails."""
    run_data = ScheduleRunCreate(
        job_type=JobType.REPORT,
        job_id="running_test",
        scheduled_for=datetime.now(timezone.utc),
        job_snapshot={}
    )

    created = jobs_service.create_run(user_id=1, run_data=run_data)

    # Start the run
    jobs_service.claim_run(created.id, "worker_1")

    # Try to cancel - should fail
    success = jobs_service.cancel_run(created.id)
    assert success is False


def test_get_run_statistics(jobs_service, dbsess):
    """Test getting run statistics."""
    now = datetime.now(timezone.utc)

    # Create runs with different statuses
    run1_data = ScheduleRunCreate(job_type=JobType.REPORT, job_id="stats1", scheduled_for=now, job_snapshot={})
    run2_data = ScheduleRunCreate(job_type=JobType.REPORT, job_id="stats2", scheduled_for=now, job_snapshot={})

    run1 = jobs_service.create_run(user_id=1, run_data=run1_data)
    run2 = jobs_service.create_run(user_id=1, run_data=run2_data)

    # Update statuses
    jobs_service.update_run(run1.id, ScheduleRunUpdate(status=RunStatus.COMPLETED))
    jobs_service.update_run(run2.id, ScheduleRunUpdate(status=RunStatus.FAILED))

    # Get statistics
    stats = jobs_service.get_run_statistics(user_id=1)

    assert "total_runs" in stats
    assert "status_counts" in stats
    assert stats["total_runs"] == 2
    assert stats["status_counts"][RunStatus.COMPLETED.value] == 1
    assert stats["status_counts"][RunStatus.FAILED.value] == 1


def test_expand_screener_target_set_name(jobs_service):
    """Test expanding screener set name to tickers."""
    # Mock screener config is already set up in fixture
    tickers = jobs_service.expand_screener_target("tech_stocks")

    assert tickers == ["AAPL", "GOOGL", "MSFT"]
    jobs_service.screener_config.validate_set_name.assert_called_with("tech_stocks")
    jobs_service.screener_config.get_tickers.assert_called_with("tech_stocks")


def test_expand_screener_target_comma_separated(jobs_service):
    """Test expanding comma-separated tickers."""
    tickers = jobs_service.expand_screener_target("AAPL,GOOGL,TSLA")

    assert tickers == ["AAPL", "GOOGL", "TSLA"]


def test_expand_screener_target_single_ticker(jobs_service):
    """Test expanding single ticker."""
    tickers = jobs_service.expand_screener_target("AAPL")

    assert tickers == ["AAPL"]


def test_expand_screener_target_invalid(jobs_service):
    """Test expanding invalid target fails."""
    jobs_service.screener_config.validate_set_name.return_value = False

    with pytest.raises(ValueError, match="Invalid screener target"):
        jobs_service.expand_screener_target("")


def test_create_screener_run(jobs_service, dbsess):
    """Test creating a screener run with proper job snapshot."""
    run = jobs_service.create_screener_run(
        user_id=1,
        screener_set="tech_stocks",
        filter_criteria={"min_volume": 1000000},
        top_n=10
    )

    assert run.job_type == JobType.SCREENER.value
    assert run.user_id == 1
    assert "screener" in run.job_id

    snapshot = run.job_snapshot
    assert snapshot["screener_set"] == "tech_stocks"
    assert snapshot["tickers"] == ["AAPL", "GOOGL", "MSFT"]
    assert snapshot["filter_criteria"]["min_volume"] == 1000000
    assert snapshot["top_n"] == 10
    assert snapshot["ticker_count"] == 3


def test_create_screener_run_with_tickers(jobs_service, dbsess):
    """Test creating screener run with explicit tickers."""
    run = jobs_service.create_screener_run(
        user_id=1,
        tickers=["AAPL", "MSFT"],
        filter_criteria={"min_price": 100}
    )

    assert run.job_type == JobType.SCREENER.value

    snapshot = run.job_snapshot
    assert snapshot["screener_set"] is None
    assert snapshot["tickers"] == ["AAPL", "MSFT"]
    assert snapshot["ticker_count"] == 2


def test_create_screener_run_no_params_fails(jobs_service, dbsess):
    """Test creating screener run without screener_set or tickers fails."""
    with pytest.raises(ValueError, match="Either screener_set or tickers must be provided"):
        jobs_service.create_screener_run(user_id=1)


def test_create_report_run(jobs_service, dbsess):
    """Test creating a report run with proper job snapshot."""
    run = jobs_service.create_report_run(
        user_id=1,
        report_type="portfolio_summary",
        parameters={"format": "pdf", "period": "monthly"}
    )

    assert run.job_type == JobType.REPORT.value
    assert run.user_id == 1
    assert "report" in run.job_id

    snapshot = run.job_snapshot
    assert snapshot["report_type"] == "portfolio_summary"
    assert snapshot["parameters"]["format"] == "pdf"
    assert snapshot["parameters"]["period"] == "monthly"


def test_validate_cron_valid_expressions(jobs_service):
    """Test that valid cron expressions pass validation."""
    valid_crons = [
        "0 9 * * *",      # Daily at 9 AM
        "*/5 * * * *",    # Every 5 minutes
        "0 0 1 * *",      # Monthly on 1st
        "0 9 * * 1-5",    # Weekdays at 9 AM
    ]

    for cron in valid_crons:
        # Should not raise exception
        jobs_service._validate_cron(cron)


def test_validate_cron_invalid_expressions(jobs_service):
    """Test that invalid cron expressions fail validation."""
    invalid_crons = [
        "invalid",
        "60 * * * *",     # Invalid minute
        "* * * * * *",    # Too many fields
        "* * *",          # Too few fields
    ]

    for cron in invalid_crons:
        with pytest.raises(ValueError, match="Invalid cron expression"):
            jobs_service._validate_cron(cron)


def test_calculate_next_run_time(jobs_service):
    """Test calculating next run time from cron expression."""
    # Test daily at 9 AM
    next_run = jobs_service._calculate_next_run_time("0 9 * * *")

    assert isinstance(next_run, datetime)
    assert next_run > datetime.now(timezone.utc)


def test_calculate_next_run_time_invalid_cron(jobs_service):
    """Test that invalid cron returns default time."""
    # Should return default time (1 hour from now) for invalid cron
    next_run = jobs_service._calculate_next_run_time("invalid cron")

    assert isinstance(next_run, datetime)
    # Should be approximately 1 hour from now
    expected = datetime.now(timezone.utc) + timedelta(hours=1)
    assert abs((next_run - expected).total_seconds()) < 60  # Within 1 minute


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))