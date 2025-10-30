"""
Jobs Service

Service layer for job scheduling and execution operations.
Provides business logic for managing schedules and runs.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import croniter

from src.data.db.repos.repo_jobs import JobsRepository
from src.data.db.models.model_jobs import (
    Schedule, ScheduleRun, RunStatus, JobType,
    ScheduleCreate, ScheduleUpdate, ScheduleRunCreate, ScheduleRunUpdate
)
# Removed import - using inline implementation
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SimpleScreenerConfig:
    """Simple screener configuration for basic functionality."""

    def __init__(self):
        # Define some basic screener sets
        self.screener_sets = {
            'sp500': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],  # Sample tickers
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX'],
            'finance': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
        }

    def validate_set_name(self, name: str) -> bool:
        """Check if the name is a valid screener set."""
        return name.lower() in self.screener_sets

    def get_tickers(self, name: str) -> list:
        """Get tickers for a screener set."""
        return self.screener_sets.get(name.lower(), [])


class JobsService:
    """Service for job scheduling and execution operations."""

    def __init__(self, session: Session):
        """
        Initialize the service with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session
        self.repository = JobsRepository(session)
        self.screener_config = SimpleScreenerConfig()

    # ---------- Schedule Operations ----------

    def create_schedule(self, user_id: int, schedule_data: ScheduleCreate) -> Schedule:
        """
        Create a new schedule.

        Args:
            user_id: User ID creating the schedule
            schedule_data: Schedule creation data

        Returns:
            Created Schedule object

        Raises:
            ValueError: If schedule data is invalid
            IntegrityError: If schedule with same name already exists
        """
        try:
            # Validate cron expression
            self._validate_cron(schedule_data.cron)

            # Calculate next run time
            next_run_at = self._calculate_next_run_time(schedule_data.cron)

            # Prepare schedule data
            schedule_dict = {
                "user_id": user_id,
                "name": schedule_data.name,
                "job_type": schedule_data.job_type.value,
                "target": schedule_data.target,
                "task_params": schedule_data.task_params,
                "cron": schedule_data.cron,
                "enabled": schedule_data.enabled,
                "next_run_at": next_run_at
            }

            schedule = self.repository.create_schedule(schedule_dict)
            self.session.commit()

            _logger.info("Created schedule: %s for user %s", schedule.name, user_id)
            return schedule

        except Exception as e:
            self.session.rollback()
            _logger.exception("Failed to create schedule:")
            raise

    def get_schedule(self, schedule_id: int) -> Optional[Schedule]:
        """
        Get a schedule by ID.

        Args:
            schedule_id: Schedule ID

        Returns:
            Schedule object or None if not found
        """
        return self.repository.get_schedule(schedule_id)

    def list_schedules(
        self,
        user_id: Optional[int] = None,
        job_type: Optional[JobType] = None,
        enabled: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Schedule]:
        """
        List schedules with optional filtering.

        Args:
            user_id: Filter by user ID
            job_type: Filter by job type
            enabled: Filter by enabled status
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of Schedule objects
        """
        return self.repository.list_schedules(user_id, job_type, enabled, limit, offset)

    def update_schedule(self, schedule_id: int, update_data: ScheduleUpdate) -> Optional[Schedule]:
        """
        Update a schedule.

        Args:
            schedule_id: Schedule ID
            update_data: Schedule update data

        Returns:
            Updated Schedule object or None if not found

        Raises:
            ValueError: If update data is invalid
        """
        try:
            # Prepare update dictionary (only include non-None values)
            update_dict = {}
            for field, value in update_data.dict(exclude_unset=True).items():
                if value is not None:
                    update_dict[field] = value

            # Validate cron expression if provided
            if "cron" in update_dict:
                self._validate_cron(update_dict["cron"])
                # Recalculate next run time
                update_dict["next_run_at"] = self._calculate_next_run_time(update_dict["cron"])

            schedule = self.repository.update_schedule(schedule_id, update_dict)
            if schedule:
                self.session.commit()
                _logger.info("Updated schedule: {} (ID: {})", schedule.name, schedule_id)

            return schedule

        except Exception as e:
            self.session.rollback()
            _logger.exception("Failed to update schedule %s:", schedule_id)
            raise

    def delete_schedule(self, schedule_id: int) -> bool:
        """
        Delete a schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            True if deleted, False if not found
        """
        try:
            success = self.repository.delete_schedule(schedule_id)
            if success:
                self.session.commit()
                _logger.info("Deleted schedule ID: %s", schedule_id)
            return success

        except Exception as e:
            self.session.rollback()
            _logger.exception("Failed to delete schedule %s:", schedule_id)
            raise

    def trigger_schedule(self, schedule_id: int) -> Optional[ScheduleRun]:
        """
        Manually trigger a schedule to create a run.

        Args:
            schedule_id: Schedule ID

        Returns:
            Created ScheduleRun object or None if schedule not found

        Raises:
            ValueError: If schedule is disabled or invalid
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return None

        if not schedule.enabled:
            raise ValueError("Cannot trigger disabled schedule")

        # Create a run for immediate execution
        run_data = ScheduleRunCreate(
            job_type=JobType(schedule.job_type),
            job_id=f"manual_{schedule_id}_{datetime.now(timezone.utc).timestamp()}",
            scheduled_for=datetime.now(timezone.utc),
            job_snapshot={
                "schedule_id": schedule_id,
                "schedule_name": schedule.name,
                "target": schedule.target,
                "task_params": schedule.task_params,
                "trigger_type": "manual"
            }
        )

        return self.create_run(schedule.user_id, run_data)

    def get_pending_schedules(self) -> List[Schedule]:
        """
        Get schedules that are due for execution.

        Returns:
            List of schedules that should be triggered
        """
        return self.repository.get_pending_schedules(datetime.now(timezone.utc))

    def update_schedule_next_run(self, schedule_id: int) -> bool:
        """
        Update the next run time for a schedule based on its cron expression.

        Args:
            schedule_id: Schedule ID

        Returns:
            True if updated, False if not found
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return False

        next_run_at = self._calculate_next_run_time(schedule.cron)
        return self.repository.update_schedule_next_run(schedule_id, next_run_at)

    # ---------- Run Operations ----------

    def create_run(self, user_id: int, run_data: ScheduleRunCreate) -> ScheduleRun:
        """
        Create a new run.

        Args:
            user_id: User ID creating the run
            run_data: Run creation data

        Returns:
            Created ScheduleRun object

        Raises:
            IntegrityError: If run with same job_type, job_id, scheduled_for already exists
        """
        try:
            # Prepare run data
            run_dict = {
                "job_type": run_data.job_type.value,
                "job_id": run_data.job_id,
                "user_id": user_id,
                "scheduled_for": run_data.scheduled_for,
                "job_snapshot": run_data.job_snapshot
            }

            run = self.repository.create_run(run_dict)
            self.session.commit()

            _logger.info("Created run: %s (%s:%s)", run.id, run.job_type, run.job_id)
            return run

        except Exception as e:
            self.session.rollback()
            _logger.exception("Failed to create run:")
            raise

    def get_run(self, run_id: int) -> Optional[ScheduleRun]:
        """
        Get a run by ID.

        Args:
            run_id: Run ID (integer)

        Returns:
            ScheduleRun object or None if not found
        """
        return self.repository.get_run(run_id)

    def list_runs(
        self,
        user_id: Optional[int] = None,
        job_type: Optional[JobType] = None,
        status: Optional[RunStatus] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "scheduled_for",
        order_desc: bool = True
    ) -> List[ScheduleRun]:
        """
        List runs with optional filtering.

        Args:
            user_id: Filter by user ID
            job_type: Filter by job type
            status: Filter by status
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field to order by
            order_desc: Order in descending order

        Returns:
            List of ScheduleRun objects
        """
        return self.repository.list_runs(user_id, job_type, status, limit, offset, order_by, order_desc)

    def update_run(self, run_id: int, update_data: ScheduleRunUpdate) -> Optional[ScheduleRun]:
        """
        Update a run.

        Args:
            run_id: Run ID (integer)
            update_data: Run update data

        Returns:
            Updated ScheduleRun object or None if not found
        """
        try:
            # Prepare update dictionary (only include non-None values)
            update_dict = {}
            for field, value in update_data.dict(exclude_unset=True).items():
                if value is not None:
                    if field == "status" and isinstance(value, RunStatus):
                        update_dict[field] = value.value
                    else:
                        update_dict[field] = value

            run = self.repository.update_run(run_id, update_dict)
            if run:
                self.session.commit()
                _logger.info("Updated run: %s (status: %s)", run.id, run.status)

            return run

        except Exception as e:
            self.session.rollback()
            _logger.exception("Failed to update run %s:", run_id)
            raise

    def claim_run(self, run_id: int, worker_id: str) -> Optional[ScheduleRun]:
        """
        Atomically claim a run for execution by a worker.

        Args:
            run_id: Run ID (integer)
            worker_id: Worker identifier

        Returns:
            ScheduleRun object if successfully claimed, None if already claimed or not found
        """
        try:
            run = self.repository.claim_run(run_id, worker_id)
            if run:
                self.session.commit()
                _logger.info("Claimed run: %s by worker: %s", run.id, worker_id)
            return run

        except Exception as e:
            self.session.rollback()
            _logger.exception("Failed to claim run %s:", run_id)
            raise

    def get_pending_runs(self, job_type: Optional[JobType] = None, limit: int = 10) -> List[ScheduleRun]:
        """
        Get pending runs that can be claimed by workers.

        Args:
            job_type: Filter by job type
            limit: Maximum number of results

        Returns:
            List of pending ScheduleRun objects
        """
        return self.repository.get_pending_runs(job_type, limit)

    def cancel_run(self, run_id: int) -> bool:
        """
        Cancel a pending run.

        Args:
            run_id: Run ID (integer)

        Returns:
            True if cancelled, False if not found or already running/completed
        """
        run = self.get_run(run_id)
        if not run:
            return False

        if run.status != RunStatus.PENDING:
            _logger.warning("Cannot cancel run %s with status %s", run_id, run.status)
            return False

        update_data = ScheduleRunUpdate(status=RunStatus.CANCELLED)
        updated_run = self.update_run(run_id, update_data)
        return updated_run is not None

    def get_run_statistics(
        self,
        user_id: Optional[int] = None,
        job_type: Optional[JobType] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get run statistics for a time period.

        Args:
            user_id: Filter by user ID
            job_type: Filter by job type
            days: Number of days to look back

        Returns:
            Dictionary with statistics
        """
        return self.repository.get_run_statistics(user_id, job_type, days)

    def cleanup_old_runs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old completed and failed runs.

        Args:
            days_to_keep: Number of days of runs to keep

        Returns:
            Number of runs deleted
        """
        try:
            deleted_count = self.repository.cleanup_old_runs(days_to_keep)
            self.session.commit()
            return deleted_count

        except Exception as e:
            self.session.rollback()
            _logger.exception("Failed to cleanup old runs:")
            raise

    # ---------- Helper Methods ----------

    def _validate_cron(self, cron_expression: str) -> None:
        """
        Validate a cron expression.

        Args:
            cron_expression: Cron expression to validate

        Raises:
            ValueError: If cron expression is invalid
        """
        try:
            # Use croniter to validate the expression
            croniter.croniter(cron_expression)
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{cron_expression}': {e}")

    def _calculate_next_run_time(self, cron_expression: str) -> datetime:
        """
        Calculate the next run time for a cron expression.

        Args:
            cron_expression: Cron expression

        Returns:
            Next run datetime
        """
        try:
            cron = croniter.croniter(cron_expression, datetime.now(timezone.utc))
            return cron.get_next(datetime)
        except Exception as e:
            _logger.error("Failed to calculate next run time for cron '%s':", cron_expression)
            # Return a default time (1 hour from now) if calculation fails
            return datetime.now(timezone.utc) + timedelta(hours=1)

    def expand_screener_target(self, target: str) -> List[str]:
        """
        Expand a screener target to a list of tickers.

        Args:
            target: Screener set name or comma-separated tickers

        Returns:
            List of ticker symbols

        Raises:
            ValueError: If target is invalid
        """
        # Check if it's a screener set name
        if self.screener_config.validate_set_name(target):
            return self.screener_config.get_tickers(target)

        # Check if it's comma-separated tickers
        if ',' in target:
            tickers = [ticker.strip().upper() for ticker in target.split(',')]
            if all(ticker for ticker in tickers):  # All tickers are non-empty
                return tickers

        # Check if it's a single ticker
        if target.strip():
            return [target.strip().upper()]

        raise ValueError(f"Invalid screener target: {target}")

    def create_screener_run(
        self,
        user_id: int,
        screener_set: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        filter_criteria: Optional[Dict[str, Any]] = None,
        top_n: Optional[int] = None,
        scheduled_for: Optional[datetime] = None
    ) -> ScheduleRun:
        """
        Create a screener run with proper job snapshot.

        Args:
            user_id: User ID
            screener_set: Screener set name
            tickers: List of ticker symbols
            filter_criteria: Filter criteria
            top_n: Number of top results to return
            scheduled_for: When to schedule the run

        Returns:
            Created ScheduleRun object

        Raises:
            ValueError: If parameters are invalid
        """
        if not screener_set and not tickers:
            raise ValueError("Either screener_set or tickers must be provided")

        # Expand tickers if screener set is provided
        if screener_set:
            try:
                expanded_tickers = self.expand_screener_target(screener_set)
            except ValueError as e:
                raise ValueError(f"Invalid screener set: {e}")
        else:
            expanded_tickers = [ticker.strip().upper() for ticker in tickers]

        # Create job snapshot
        job_snapshot = {
            "screener_set": screener_set,
            "tickers": expanded_tickers,
            "filter_criteria": filter_criteria or {},
            "top_n": top_n,
            "ticker_count": len(expanded_tickers)
        }

        # Create job ID
        job_id = f"screener_{screener_set or 'custom'}_{datetime.now(timezone.utc).timestamp()}"

        # Create run
        run_data = ScheduleRunCreate(
            job_type=JobType.SCREENER,
            job_id=job_id,
            scheduled_for=scheduled_for or datetime.now(timezone.utc),
            job_snapshot=job_snapshot
        )

        return self.create_run(user_id, run_data)

    def create_report_run(
        self,
        user_id: int,
        report_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        scheduled_for: Optional[datetime] = None
    ) -> ScheduleRun:
        """
        Create a report run with proper job snapshot.

        Args:
            user_id: User ID
            report_type: Type of report
            parameters: Report parameters
            scheduled_for: When to schedule the run

        Returns:
            Created Run object
        """
        # Create job snapshot
        job_snapshot = {
            "report_type": report_type,
            "parameters": parameters or {}
        }

        # Create job ID
        job_id = f"report_{report_type}_{datetime.now(timezone.utc).timestamp()}"

        # Create run
        run_data = ScheduleRunCreate(
            job_type=JobType.REPORT,
            job_id=job_id,
            scheduled_for=scheduled_for or datetime.now(timezone.utc),
            job_snapshot=job_snapshot
        )

        return self.create_run(user_id, run_data)

