"""
Tests for JobsRepository

Tests the repository layer for job scheduling and execution operations.
Covers the new tables: job_schedules, job_runs.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

# Models
from src.data.db.models.model_jobs import Base as JobsBase, Schedule, ScheduleRun, JobType, RunStatus

# Repository under test
from src.data.db.repos.repo_jobs import JobsRepository


# Use the shared engine and dbsess fixtures from conftest.py


@pytest.fixture()
def repo(dbsess):
    """Create JobsRepository instance."""
    return JobsRepository(dbsess)


def test_create_schedule(repo, dbsess):
    """Test creating a new schedule."""
    schedule_data = {
        "user_id": 1,
        "name": "Daily Report",
        "job_type": JobType.REPORT.value,
        "target": "portfolio_summary",
        "task_params": {"format": "pdf"},
        "cron": "0 9 * * *",
        "enabled": True,
        "next_run_at": datetime.utcnow() + timedelta(hours=1)
    }

    schedule = repo.create_schedule(schedule_data)

    assert schedule.id is not None
    assert schedule.name == "Daily Report"
    assert schedule.job_type == JobType.REPORT.value
    assert schedule.user_id == 1
    assert schedule.enabled is True


def test_create_schedule_duplicate_name_fails(repo, dbsess):
    """Test that creating schedule with duplicate name for same user fails."""
    schedule_data = {
        "user_id": 1,
        "name": "Duplicate Name",
        "job_type": JobType.REPORT.value,
        "target": "test",
        "cron": "0 9 * * *"
    }

    # First creation should succeed
    repo.create_schedule(schedule_data)
    dbsess.commit()

    # Second creation with same name and user should fail
    with pytest.raises(IntegrityError):
        repo.create_schedule(schedule_data)


def test_get_schedule(repo, dbsess):
    """Test getting a schedule by ID."""
    schedule_data = {
        "user_id": 1,
        "name": "Test Schedule",
        "job_type": JobType.SCREENER.value,
        "target": "tech_stocks",
        "cron": "0 10 * * *"
    }

    created = repo.create_schedule(schedule_data)
    retrieved = repo.get_schedule(created.id)

    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.name == "Test Schedule"


def test_get_schedule_nonexistent(repo):
    """Test getting non-existent schedule returns None."""
    result = repo.get_schedule(99999)
    assert result is None


def test_get_schedule_by_name(repo, dbsess):
    """Test getting schedule by user ID and name."""
    schedule_data = {
        "user_id": 1,
        "name": "Named Schedule",
        "job_type": JobType.ALERT.value,
        "target": "price_alert",
        "cron": "*/5 * * * *"
    }

    created = repo.create_schedule(schedule_data)
    retrieved = repo.get_schedule_by_name(1, "Named Schedule")

    assert retrieved is not None
    assert retrieved.id == created.id


def test_list_schedules_with_filters(repo, dbsess):
    """Test listing schedules with various filters."""
    # Create test schedules
    schedules_data = [
        {"user_id": 1, "name": "User1 Report", "job_type": JobType.REPORT.value, "target": "r1", "cron": "0 9 * * *", "enabled": True},
        {"user_id": 1, "name": "User1 Screener", "job_type": JobType.SCREENER.value, "target": "s1", "cron": "0 10 * * *", "enabled": False},
        {"user_id": 2, "name": "User2 Report", "job_type": JobType.REPORT.value, "target": "r2", "cron": "0 11 * * *", "enabled": True},
    ]

    for data in schedules_data:
        repo.create_schedule(data)

    # Test filter by user_id
    user1_schedules = repo.list_schedules(user_id=1)
    assert len(user1_schedules) == 2

    # Test filter by job_type
    report_schedules = repo.list_schedules(job_type=JobType.REPORT)
    assert len(report_schedules) == 2

    # Test filter by enabled
    enabled_schedules = repo.list_schedules(enabled=True)
    assert len(enabled_schedules) == 2

    # Test combined filters
    user1_enabled = repo.list_schedules(user_id=1, enabled=True)
    assert len(user1_enabled) == 1


def test_update_schedule(repo, dbsess):
    """Test updating a schedule."""
    schedule_data = {
        "user_id": 1,
        "name": "Original Name",
        "job_type": JobType.REPORT.value,
        "target": "original_target",
        "cron": "0 9 * * *",
        "enabled": True
    }

    created = repo.create_schedule(schedule_data)

    # Update schedule
    update_data = {
        "name": "Updated Name",
        "enabled": False,
        "cron": "0 10 * * *"
    }

    updated = repo.update_schedule(created.id, update_data)

    assert updated is not None
    assert updated.name == "Updated Name"
    assert updated.enabled is False
    assert updated.cron == "0 10 * * *"
    assert updated.target == "original_target"  # Unchanged


def test_update_schedule_nonexistent(repo):
    """Test updating non-existent schedule returns None."""
    result = repo.update_schedule(99999, {"name": "New Name"})
    assert result is None


def test_delete_schedule(repo, dbsess):
    """Test deleting a schedule."""
    schedule_data = {
        "user_id": 1,
        "name": "To Delete",
        "job_type": JobType.REPORT.value,
        "target": "delete_me",
        "cron": "0 9 * * *"
    }

    created = repo.create_schedule(schedule_data)

    # Delete schedule
    success = repo.delete_schedule(created.id)
    assert success is True

    # Verify it's gone
    retrieved = repo.get_schedule(created.id)
    assert retrieved is None


def test_delete_schedule_nonexistent(repo):
    """Test deleting non-existent schedule returns False."""
    result = repo.delete_schedule(99999)
    assert result is False


def test_get_pending_schedules(repo, dbsess):
    """Test getting schedules due for execution."""
    now = datetime.utcnow()
    past = now - timedelta(minutes=5)
    future = now + timedelta(minutes=5)

    schedules_data = [
        {"user_id": 1, "name": "Past Due", "job_type": JobType.REPORT.value, "target": "r1", "cron": "0 9 * * *", "enabled": True, "next_run_at": past},
        {"user_id": 1, "name": "Future", "job_type": JobType.REPORT.value, "target": "r2", "cron": "0 10 * * *", "enabled": True, "next_run_at": future},
        {"user_id": 1, "name": "Disabled", "job_type": JobType.REPORT.value, "target": "r3", "cron": "0 11 * * *", "enabled": False, "next_run_at": past},
    ]

    for data in schedules_data:
        repo.create_schedule(data)

    pending = repo.get_pending_schedules(now)

    # Should only return enabled schedules that are due
    assert len(pending) == 1
    assert pending[0].name == "Past Due"


def test_update_schedule_next_run(repo, dbsess):
    """Test updating schedule next run time."""
    schedule_data = {
        "user_id": 1,
        "name": "Test Schedule",
        "job_type": JobType.REPORT.value,
        "target": "test",
        "cron": "0 9 * * *"
    }

    created = repo.create_schedule(schedule_data)
    new_time = datetime.utcnow() + timedelta(hours=2)

    success = repo.update_schedule_next_run(created.id, new_time)
    assert success is True

    # Verify update
    retrieved = repo.get_schedule(created.id)
    assert retrieved.next_run_at == new_time


def test_create_run(repo, dbsess):
    """Test creating a new run."""
    run_data = {
        "job_type": JobType.REPORT.value,
        "job_id": "test_job_123",
        "user_id": 1,
        "scheduled_for": datetime.utcnow(),
        "job_snapshot": {"param1": "value1"}
    }

    run = repo.create_run(run_data)

    assert run.run_id is not None
    assert run.job_type == JobType.REPORT.value
    assert run.job_id == "test_job_123"
    assert run.status == RunStatus.PENDING.value


def test_create_run_duplicate_fails(repo, dbsess):
    """Test that creating duplicate run fails."""
    scheduled_time = datetime.utcnow()
    run_data = {
        "job_type": JobType.REPORT.value,
        "job_id": "duplicate_job",
        "user_id": 1,
        "scheduled_for": scheduled_time,
        "job_snapshot": {}
    }

    # First creation should succeed
    repo.create_run(run_data)
    dbsess.commit()

    # Second creation with same job_type, job_id, scheduled_for should fail
    with pytest.raises(IntegrityError):
        repo.create_run(run_data)


def test_get_run(repo, dbsess):
    """Test getting a run by ID."""
    run_data = {
        "job_type": JobType.SCREENER.value,
        "job_id": "test_screener",
        "user_id": 1,
        "scheduled_for": datetime.utcnow(),
        "job_snapshot": {"tickers": ["AAPL", "GOOGL"]}
    }

    created = repo.create_run(run_data)
    retrieved = repo.get_run(created.run_id)

    assert retrieved is not None
    assert retrieved.run_id == created.run_id
    assert retrieved.job_id == "test_screener"


def test_list_runs_with_filters(repo, dbsess):
    """Test listing runs with various filters."""
    now = datetime.utcnow()

    runs_data = [
        {"job_type": JobType.REPORT.value, "job_id": "r1", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
        {"job_type": JobType.SCREENER.value, "job_id": "s1", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
        {"job_type": JobType.REPORT.value, "job_id": "r2", "user_id": 2, "scheduled_for": now, "job_snapshot": {}},
    ]

    created_runs = []
    for data in runs_data:
        run = repo.create_run(data)
        created_runs.append(run)

    # Update one run to completed status
    repo.update_run(created_runs[0].run_id, {"status": RunStatus.COMPLETED.value})

    # Test filter by user_id
    user1_runs = repo.list_runs(user_id=1)
    assert len(user1_runs) == 2

    # Test filter by job_type
    report_runs = repo.list_runs(job_type=JobType.REPORT)
    assert len(report_runs) == 2

    # Test filter by status
    completed_runs = repo.list_runs(status=RunStatus.COMPLETED)
    assert len(completed_runs) == 1


def test_update_run(repo, dbsess):
    """Test updating a run."""
    run_data = {
        "job_type": JobType.REPORT.value,
        "job_id": "update_test",
        "user_id": 1,
        "scheduled_for": datetime.utcnow(),
        "job_snapshot": {}
    }

    created = repo.create_run(run_data)

    # Update run
    now = datetime.utcnow()
    update_data = {
        "status": RunStatus.RUNNING.value,
        "started_at": now,
        "worker_id": "worker_123"
    }

    updated = repo.update_run(created.run_id, update_data)

    assert updated is not None
    assert updated.status == RunStatus.RUNNING.value
    assert updated.started_at == now
    assert updated.worker_id == "worker_123"


def test_claim_run(repo, dbsess):
    """Test atomically claiming a run for execution."""
    run_data = {
        "job_type": JobType.SCREENER.value,
        "job_id": "claim_test",
        "user_id": 1,
        "scheduled_for": datetime.utcnow(),
        "job_snapshot": {}
    }

    created = repo.create_run(run_data)

    # Claim the run
    claimed = repo.claim_run(created.run_id, "worker_456")

    assert claimed is not None
    assert claimed.status == RunStatus.RUNNING.value
    assert claimed.worker_id == "worker_456"
    assert claimed.started_at is not None


def test_claim_run_already_claimed(repo, dbsess):
    """Test that claiming already claimed run returns None."""
    run_data = {
        "job_type": JobType.REPORT.value,
        "job_id": "already_claimed",
        "user_id": 1,
        "scheduled_for": datetime.utcnow(),
        "job_snapshot": {}
    }

    created = repo.create_run(run_data)

    # First claim should succeed
    first_claim = repo.claim_run(created.run_id, "worker_1")
    assert first_claim is not None

    # Second claim should fail
    second_claim = repo.claim_run(created.run_id, "worker_2")
    assert second_claim is None


def test_get_pending_runs(repo, dbsess):
    """Test getting pending runs."""
    now = datetime.utcnow()

    runs_data = [
        {"job_type": JobType.REPORT.value, "job_id": "pending1", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
        {"job_type": JobType.SCREENER.value, "job_id": "pending2", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
        {"job_type": JobType.REPORT.value, "job_id": "running", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
    ]

    created_runs = []
    for data in runs_data:
        run = repo.create_run(data)
        created_runs.append(run)

    # Update one to running status
    repo.update_run(created_runs[2].run_id, {"status": RunStatus.RUNNING.value})

    # Get pending runs
    pending = repo.get_pending_runs()

    # Should only return pending runs
    assert len(pending) == 2
    for run in pending:
        assert run.status == RunStatus.PENDING.value


def test_get_runs_by_job(repo, dbsess):
    """Test getting all runs for a specific job."""
    job_id = "specific_job"
    now = datetime.utcnow()

    runs_data = [
        {"job_type": JobType.REPORT.value, "job_id": job_id, "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
        {"job_type": JobType.REPORT.value, "job_id": job_id, "user_id": 1, "scheduled_for": now + timedelta(hours=1), "job_snapshot": {}},
        {"job_type": JobType.REPORT.value, "job_id": "other_job", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
    ]

    for data in runs_data:
        repo.create_run(data)

    # Get runs for specific job
    job_runs = repo.get_runs_by_job(JobType.REPORT, job_id)

    assert len(job_runs) == 2
    for run in job_runs:
        assert run.job_id == job_id


def test_get_run_statistics(repo, dbsess):
    """Test getting run statistics."""
    now = datetime.utcnow()

    # Create runs with different statuses
    runs_data = [
        {"job_type": JobType.REPORT.value, "job_id": "stats1", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
        {"job_type": JobType.REPORT.value, "job_id": "stats2", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
        {"job_type": JobType.SCREENER.value, "job_id": "stats3", "user_id": 2, "scheduled_for": now, "job_snapshot": {}},
    ]

    created_runs = []
    for data in runs_data:
        run = repo.create_run(data)
        created_runs.append(run)

    # Update statuses
    repo.update_run(created_runs[0].run_id, {"status": RunStatus.COMPLETED.value})
    repo.update_run(created_runs[1].run_id, {"status": RunStatus.FAILED.value})

    # Get statistics
    stats = repo.get_run_statistics()

    assert "total_runs" in stats
    assert "status_counts" in stats
    assert "period_days" in stats
    assert stats["total_runs"] == 3
    assert stats["status_counts"][RunStatus.PENDING.value] == 1
    assert stats["status_counts"][RunStatus.COMPLETED.value] == 1
    assert stats["status_counts"][RunStatus.FAILED.value] == 1


def test_cleanup_old_runs(repo, dbsess):
    """Test cleaning up old completed and failed runs."""
    now = datetime.utcnow()
    old_time = now - timedelta(days=100)

    # Create old runs
    old_runs_data = [
        {"job_type": JobType.REPORT.value, "job_id": "old1", "user_id": 1, "scheduled_for": old_time, "job_snapshot": {}},
        {"job_type": JobType.REPORT.value, "job_id": "old2", "user_id": 1, "scheduled_for": old_time, "job_snapshot": {}},
        {"job_type": JobType.REPORT.value, "job_id": "recent", "user_id": 1, "scheduled_for": now, "job_snapshot": {}},
    ]

    created_runs = []
    for data in old_runs_data:
        run = repo.create_run(data)
        created_runs.append(run)

    # Update old runs to completed/failed
    repo.update_run(created_runs[0].run_id, {"status": RunStatus.COMPLETED.value})
    repo.update_run(created_runs[1].run_id, {"status": RunStatus.FAILED.value})
    # Leave recent run as pending

    # Cleanup old runs (keep 30 days)
    deleted_count = repo.cleanup_old_runs(days_to_keep=30)

    # Should delete 2 old completed/failed runs
    assert deleted_count == 2

    # Recent pending run should still exist
    remaining = repo.list_runs()
    assert len(remaining) == 1
    assert remaining[0].job_id == "recent"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))