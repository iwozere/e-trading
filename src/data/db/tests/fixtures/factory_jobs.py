"""
Test data factories for Jobs models.

Provides factory functions to create test data for Schedule and ScheduleRun models.
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from src.data.db.models.model_jobs import JobType, RunStatus


class ScheduleFactory:
    """Factory for creating Schedule test data."""

    @staticmethod
    def create_data(
        user_id: int = 1,
        name: str = "test_schedule",
        job_type: JobType = JobType.SCREENER,
        target: str = "AAPL,MSFT",
        cron: str = "0 9 * * *",
        enabled: bool = True,
        task_params: Optional[Dict[str, Any]] = None,
        next_run_at: Optional[datetime] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create schedule data dictionary."""
        return {
            "user_id": user_id,
            "name": name,
            "job_type": job_type.value if isinstance(job_type, JobType) else job_type,
            "target": target,
            "cron": cron,
            "enabled": enabled,
            "task_params": task_params or {},
            "next_run_at": next_run_at or datetime.now(timezone.utc) + timedelta(days=1),
            **kwargs
        }

    @staticmethod
    def daily_screener(user_id: int = 1, name: str = "daily_screener") -> Dict[str, Any]:
        """Create a daily screener schedule."""
        return ScheduleFactory.create_data(
            user_id=user_id,
            name=name,
            job_type=JobType.SCREENER,
            target="sp500",
            cron="0 9 * * *",  # 9 AM daily
            enabled=True
        )

    @staticmethod
    def weekly_report(user_id: int = 1, name: str = "weekly_report") -> Dict[str, Any]:
        """Create a weekly report schedule."""
        return ScheduleFactory.create_data(
            user_id=user_id,
            name=name,
            job_type=JobType.REPORT,
            target="weekly_summary",
            cron="0 10 * * 1",  # 10 AM every Monday
            enabled=True
        )

    @staticmethod
    def disabled_schedule(user_id: int = 1, name: str = "disabled_schedule") -> Dict[str, Any]:
        """Create a disabled schedule."""
        return ScheduleFactory.create_data(
            user_id=user_id,
            name=name,
            job_type=JobType.ALERT,
            target="test_alert",
            cron="0 * * * *",
            enabled=False
        )


class ScheduleRunFactory:
    """Factory for creating ScheduleRun test data."""

    @staticmethod
    def create_data(
        job_type: JobType = JobType.SCREENER,
        job_id: Optional[int] = None,  # Changed to Optional[int] to match model
        user_id: int = 1,
        status: RunStatus = RunStatus.PENDING,
        scheduled_for: Optional[datetime] = None,
        job_snapshot: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create schedule run data dictionary."""
        return {
            "job_type": job_type.value if isinstance(job_type, JobType) else job_type,
            "job_id": job_id or 1,  # Default to 1 if not provided
            "user_id": user_id,
            "status": status.value if isinstance(status, RunStatus) else status,
            "scheduled_for": scheduled_for or datetime.now(timezone.utc),
            "job_snapshot": job_snapshot or {"test": "data"},
            **kwargs
        }

    @staticmethod
    def pending_run(
        job_id: int = 1,  # Changed to int
        user_id: int = 1,
        scheduled_for: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Create a pending run."""
        return ScheduleRunFactory.create_data(
            job_type=JobType.SCREENER,
            job_id=job_id,
            user_id=user_id,
            status=RunStatus.PENDING,
            scheduled_for=scheduled_for or datetime.now(timezone.utc)
        )

    @staticmethod
    def running_run(
        job_id: int = 2,  # Changed to int
        user_id: int = 1,
        worker_id: str = "worker_1"
    ) -> Dict[str, Any]:
        """Create a running run."""
        return ScheduleRunFactory.create_data(
            job_type=JobType.SCREENER,
            job_id=job_id,
            user_id=user_id,
            status=RunStatus.RUNNING,
            worker_id=worker_id,
            started_at=datetime.now(timezone.utc)
        )

    @staticmethod
    def completed_run(
        job_id: int = 3,  # Changed to int
        user_id: int = 1,
        execution_time_ms: int = 1500
    ) -> Dict[str, Any]:
        """Create a completed run."""
        now = datetime.now(timezone.utc)
        return ScheduleRunFactory.create_data(
            job_type=JobType.SCREENER,
            job_id=job_id,
            user_id=user_id,
            status=RunStatus.COMPLETED,
            started_at=now - timedelta(seconds=2),
            ended_at=now,
            execution_time_ms=execution_time_ms,
            result={"success": True, "records_processed": 100}
        )

    @staticmethod
    def failed_run(
        job_id: int = 4,  # Changed to int
        user_id: int = 1,
        error: str = "Test error"
    ) -> Dict[str, Any]:
        """Create a failed run."""
        now = datetime.now(timezone.utc)
        return ScheduleRunFactory.create_data(
            job_type=JobType.SCREENER,
            job_id=job_id,
            user_id=user_id,
            status=RunStatus.FAILED,
            started_at=now - timedelta(seconds=1),
            ended_at=now,
            error=error
        )


# Convenient aliases
schedule_factory = ScheduleFactory()
run_factory = ScheduleRunFactory()
