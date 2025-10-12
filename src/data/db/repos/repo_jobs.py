"""
Jobs Repository

Repository layer for job scheduling and execution operations.
Provides data access methods for schedules and runs tables.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from sqlalchemy.exc import IntegrityError

from src.data.db.models.model_jobs import Schedule, Run, RunStatus, JobType
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class JobsRepository:
    """Repository for job scheduling and execution data access."""

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    # ---------- Schedule Operations ----------

    def create_schedule(self, schedule_data: Dict[str, Any]) -> Schedule:
        """
        Create a new schedule.

        Args:
            schedule_data: Dictionary with schedule data

        Returns:
            Created Schedule object

        Raises:
            IntegrityError: If schedule with same name already exists for user
        """
        try:
            schedule = Schedule(**schedule_data)
            self.session.add(schedule)
            self.session.flush()  # Get the ID without committing
            _logger.info(f"Created schedule: {schedule.name} (ID: {schedule.id})")
            return schedule
        except IntegrityError as e:
            self.session.rollback()
            _logger.error(f"Failed to create schedule: {e}")
            raise

    def get_schedule(self, schedule_id: int) -> Optional[Schedule]:
        """
        Get a schedule by ID.

        Args:
            schedule_id: Schedule ID

        Returns:
            Schedule object or None if not found
        """
        return self.session.query(Schedule).filter(Schedule.id == schedule_id).first()

    def get_schedule_by_name(self, user_id: int, name: str) -> Optional[Schedule]:
        """
        Get a schedule by user ID and name.

        Args:
            user_id: User ID
            name: Schedule name

        Returns:
            Schedule object or None if not found
        """
        return self.session.query(Schedule).filter(
            and_(Schedule.user_id == user_id, Schedule.name == name)
        ).first()

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
        query = self.session.query(Schedule)

        if user_id is not None:
            query = query.filter(Schedule.user_id == user_id)

        if job_type is not None:
            query = query.filter(Schedule.job_type == job_type.value)

        if enabled is not None:
            query = query.filter(Schedule.enabled == enabled)

        return query.order_by(desc(Schedule.created_at)).offset(offset).limit(limit).all()

    def update_schedule(self, schedule_id: int, update_data: Dict[str, Any]) -> Optional[Schedule]:
        """
        Update a schedule.

        Args:
            schedule_id: Schedule ID
            update_data: Dictionary with fields to update

        Returns:
            Updated Schedule object or None if not found
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return None

        try:
            for key, value in update_data.items():
                if hasattr(schedule, key):
                    setattr(schedule, key, value)

            self.session.flush()
            _logger.info(f"Updated schedule: {schedule.name} (ID: {schedule.id})")
            return schedule
        except Exception as e:
            self.session.rollback()
            _logger.error(f"Failed to update schedule {schedule_id}: {e}")
            raise

    def delete_schedule(self, schedule_id: int) -> bool:
        """
        Delete a schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            True if deleted, False if not found
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return False

        try:
            self.session.delete(schedule)
            _logger.info(f"Deleted schedule: {schedule.name} (ID: {schedule.id})")
            return True
        except Exception as e:
            self.session.rollback()
            _logger.error(f"Failed to delete schedule {schedule_id}: {e}")
            raise

    def get_pending_schedules(self, current_time: datetime) -> List[Schedule]:
        """
        Get schedules that are due for execution.

        Args:
            current_time: Current timestamp

        Returns:
            List of schedules that should be triggered
        """
        return self.session.query(Schedule).filter(
            and_(
                Schedule.enabled == True,
                Schedule.next_run_at <= current_time
            )
        ).all()

    def update_schedule_next_run(self, schedule_id: int, next_run_at: datetime) -> bool:
        """
        Update the next run time for a schedule.

        Args:
            schedule_id: Schedule ID
            next_run_at: Next run timestamp

        Returns:
            True if updated, False if not found
        """
        schedule = self.get_schedule(schedule_id)
        if not schedule:
            return False

        schedule.next_run_at = next_run_at
        self.session.flush()
        return True

    # ---------- Run Operations ----------

    def create_run(self, run_data: Dict[str, Any]) -> Run:
        """
        Create a new run.

        Args:
            run_data: Dictionary with run data

        Returns:
            Created Run object

        Raises:
            IntegrityError: If run with same job_type, job_id, scheduled_for already exists
        """
        try:
            run = Run(**run_data)
            self.session.add(run)
            self.session.flush()  # Get the run_id without committing
            _logger.info(f"Created run: {run.run_id} ({run.job_type}:{run.job_id})")
            return run
        except IntegrityError as e:
            self.session.rollback()
            _logger.error(f"Failed to create run: {e}")
            raise

    def get_run(self, run_id: UUID) -> Optional[Run]:
        """
        Get a run by ID.

        Args:
            run_id: Run UUID

        Returns:
            Run object or None if not found
        """
        return self.session.query(Run).filter(Run.run_id == run_id).first()

    def list_runs(
        self,
        user_id: Optional[int] = None,
        job_type: Optional[JobType] = None,
        status: Optional[RunStatus] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "scheduled_for",
        order_desc: bool = True
    ) -> List[Run]:
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
            List of Run objects
        """
        query = self.session.query(Run)

        if user_id is not None:
            query = query.filter(Run.user_id == user_id)

        if job_type is not None:
            query = query.filter(Run.job_type == job_type.value)

        if status is not None:
            query = query.filter(Run.status == status.value)

        # Apply ordering
        order_field = getattr(Run, order_by, Run.scheduled_for)
        if order_desc:
            query = query.order_by(desc(order_field))
        else:
            query = query.order_by(asc(order_field))

        return query.offset(offset).limit(limit).all()

    def update_run(self, run_id: UUID, update_data: Dict[str, Any]) -> Optional[Run]:
        """
        Update a run.

        Args:
            run_id: Run UUID
            update_data: Dictionary with fields to update

        Returns:
            Updated Run object or None if not found
        """
        run = self.get_run(run_id)
        if not run:
            return None

        try:
            for key, value in update_data.items():
                if hasattr(run, key):
                    setattr(run, key, value)

            self.session.flush()
            _logger.info(f"Updated run: {run.run_id} (status: {run.status})")
            return run
        except Exception as e:
            self.session.rollback()
            _logger.error(f"Failed to update run {run_id}: {e}")
            raise

    def claim_run(self, run_id: UUID, worker_id: str) -> Optional[Run]:
        """
        Atomically claim a run for execution by a worker.

        This prevents multiple workers from executing the same run.

        Args:
            run_id: Run UUID
            worker_id: Worker identifier

        Returns:
            Run object if successfully claimed, None if already claimed or not found
        """
        try:
            # Use a subquery to atomically update and return the run
            run = self.session.query(Run).filter(
                and_(
                    Run.run_id == run_id,
                    Run.status == RunStatus.PENDING.value
                )
            ).with_for_update().first()

            if not run:
                return None

            # Update the run to claimed state
            run.status = RunStatus.RUNNING.value
            run.started_at = datetime.utcnow()
            run.worker_id = worker_id

            self.session.flush()
            _logger.info(f"Claimed run: {run.run_id} by worker: {worker_id}")
            return run

        except Exception as e:
            self.session.rollback()
            _logger.error(f"Failed to claim run {run_id}: {e}")
            raise

    def get_pending_runs(
        self,
        job_type: Optional[JobType] = None,
        limit: int = 10
    ) -> List[Run]:
        """
        Get pending runs that can be claimed by workers.

        Args:
            job_type: Filter by job type
            limit: Maximum number of results

        Returns:
            List of pending Run objects
        """
        query = self.session.query(Run).filter(Run.status == RunStatus.PENDING.value)

        if job_type is not None:
            query = query.filter(Run.job_type == job_type.value)

        return query.order_by(asc(Run.scheduled_for)).limit(limit).all()

    def get_runs_by_job(self, job_type: JobType, job_id: str) -> List[Run]:
        """
        Get all runs for a specific job.

        Args:
            job_type: Job type
            job_id: Job identifier

        Returns:
            List of Run objects
        """
        return self.session.query(Run).filter(
            and_(Run.job_type == job_type.value, Run.job_id == job_id)
        ).order_by(desc(Run.scheduled_for)).all()

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
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        query = self.session.query(Run).filter(Run.scheduled_for >= cutoff_date)

        if user_id is not None:
            query = query.filter(Run.user_id == user_id)

        if job_type is not None:
            query = query.filter(Run.job_type == job_type.value)

        # Get counts by status
        status_counts = {}
        for status in RunStatus:
            count = query.filter(Run.status == status.value).count()
            status_counts[status.value] = count

        # Get total count
        total_count = query.count()

        # Get average execution time for completed runs
        completed_runs = query.filter(Run.status == RunStatus.COMPLETED.value).all()
        avg_execution_time = None
        if completed_runs:
            execution_times = []
            for run in completed_runs:
                if run.started_at and run.finished_at:
                    execution_time = (run.finished_at - run.started_at).total_seconds()
                    execution_times.append(execution_time)

            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)

        return {
            "total_runs": total_count,
            "status_counts": status_counts,
            "average_execution_time_seconds": avg_execution_time,
            "period_days": days
        }

    def cleanup_old_runs(self, days_to_keep: int = 90) -> int:
        """
        Clean up old completed and failed runs.

        Args:
            days_to_keep: Number of days of runs to keep

        Returns:
            Number of runs deleted
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        # Delete old completed and failed runs
        deleted_count = self.session.query(Run).filter(
            and_(
                Run.scheduled_for < cutoff_date,
                or_(
                    Run.status == RunStatus.COMPLETED.value,
                    Run.status == RunStatus.FAILED.value
                )
            )
        ).delete()

        _logger.info(f"Cleaned up {deleted_count} old runs")
        return deleted_count

