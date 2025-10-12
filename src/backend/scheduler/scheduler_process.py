"""
Scheduler Process

Main scheduler process that runs as a separate service to handle cron-based job triggering.
Uses APScheduler to check for pending schedules and enqueue jobs.
"""

import sys
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.database_service import get_database_service
from src.data.db.services.jobs_service import JobsService
from src.data.db.models.model_jobs import Schedule, JobType
from src.backend.config_loader import get_screener_config
from src.backend.workers.report_worker import run_report
from src.backend.workers.screener_worker import run_screener
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class JobScheduler:
    """Main job scheduler class."""

    def __init__(self):
        """Initialize the scheduler."""
        self.scheduler = AsyncIOScheduler()
        self.running = False
        self.screener_config = get_screener_config()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        _logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    async def start(self):
        """Start the scheduler."""
        _logger.info("Starting job scheduler...")

        try:
            # Add the main scheduling job (runs every minute)
            self.scheduler.add_job(
                self._check_pending_schedules,
                trigger=IntervalTrigger(minutes=1),
                id="check_pending_schedules",
                name="Check Pending Schedules",
                max_instances=1,
                coalesce=True
            )

            # Add cleanup job (runs daily at 2 AM)
            self.scheduler.add_job(
                self._cleanup_old_runs,
                trigger="cron",
                hour=2,
                minute=0,
                id="cleanup_old_runs",
                name="Cleanup Old Runs",
                max_instances=1
            )

            # Add event listeners
            self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)

            # Start the scheduler
            self.scheduler.start()
            self.running = True

            _logger.info("Job scheduler started successfully")

            # Keep the process running
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            _logger.error(f"Failed to start scheduler: {e}")
            raise

    def stop(self):
        """Stop the scheduler."""
        _logger.info("Stopping job scheduler...")
        self.running = False

        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)

        _logger.info("Job scheduler stopped")

    async def _check_pending_schedules(self):
        """Check for pending schedules and trigger them."""
        _logger.debug("Checking for pending schedules...")

        try:
            # Get database service
            db_service = get_database_service()

            with db_service.uow() as uow:
                jobs_service = JobsService(uow.session)

                # Get pending schedules
                pending_schedules = jobs_service.get_pending_schedules()

                if not pending_schedules:
                    _logger.debug("No pending schedules found")
                    return

                _logger.info(f"Found {len(pending_schedules)} pending schedules")

                # Process each pending schedule
                for schedule in pending_schedules:
                    try:
                        await self._process_schedule(schedule, jobs_service)
                    except Exception as e:
                        _logger.error(f"Failed to process schedule {schedule.id}: {e}")
                        continue

                uow.commit()

        except Exception as e:
            _logger.error(f"Error checking pending schedules: {e}")

    async def _process_schedule(self, schedule: Schedule, jobs_service: JobsService):
        """Process a single pending schedule."""
        _logger.info(f"Processing schedule: {schedule.name} (ID: {schedule.id})")

        try:
            # Expand the target if it's a screener set
            if schedule.job_type == JobType.SCREENER.value:
                expanded_tickers = self._expand_screener_target(schedule.target)
                if not expanded_tickers:
                    _logger.warning(f"No tickers found for screener set: {schedule.target}")
                    return

                # Create job snapshot with expanded tickers
                job_snapshot = {
                    "schedule_id": schedule.id,
                    "schedule_name": schedule.name,
                    "screener_set": schedule.target,
                    "tickers": expanded_tickers,
                    "filter_criteria": schedule.task_params.get("filter_criteria", {}),
                    "top_n": schedule.task_params.get("top_n", 50),
                    "ticker_count": len(expanded_tickers),
                    "trigger_type": "scheduled"
                }
            else:
                # For reports, use the task_params directly
                job_snapshot = {
                    "schedule_id": schedule.id,
                    "schedule_name": schedule.name,
                    "report_type": schedule.target,
                    "parameters": schedule.task_params,
                    "trigger_type": "scheduled"
                }

            # Create run
            from src.data.db.models.model_jobs import RunCreate
            run_data = RunCreate(
                job_type=JobType(schedule.job_type),
                job_id=f"scheduled_{schedule.id}_{datetime.utcnow().timestamp()}",
                scheduled_for=schedule.next_run_at,
                job_snapshot=job_snapshot
            )

            run = jobs_service.create_run(schedule.user_id, run_data)

            # Enqueue the job
            if schedule.job_type == JobType.REPORT.value:
                run_report.send(str(run.run_id))
                _logger.info(f"Enqueued report job: {run.run_id}")
            elif schedule.job_type == JobType.SCREENER.value:
                run_screener.send(str(run.run_id))
                _logger.info(f"Enqueued screener job: {run.run_id}")

            # Update schedule next run time
            next_run_at = self._calculate_next_run_time(schedule.cron)
            jobs_service.update_schedule_next_run(schedule.id, next_run_at)

            _logger.info(f"Successfully processed schedule: {schedule.name}")

        except Exception as e:
            _logger.error(f"Failed to process schedule {schedule.id}: {e}")
            raise

    def _expand_screener_target(self, target: str) -> List[str]:
        """Expand a screener target to a list of tickers."""
        try:
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

            return []

        except Exception as e:
            _logger.error(f"Failed to expand screener target '{target}': {e}")
            return []

    def _calculate_next_run_time(self, cron_expression: str) -> datetime:
        """Calculate the next run time for a cron expression."""
        try:
            import croniter
            cron = croniter.croniter(cron_expression, datetime.utcnow())
            return cron.get_next(datetime)
        except Exception as e:
            _logger.error(f"Failed to calculate next run time for cron '{cron_expression}': {e}")
            # Return a default time (1 hour from now) if calculation fails
            return datetime.utcnow() + timedelta(hours=1)

    async def _cleanup_old_runs(self):
        """Clean up old completed and failed runs."""
        _logger.info("Starting cleanup of old runs...")

        try:
            db_service = get_database_service()

            with db_service.uow() as uow:
                jobs_service = JobsService(uow.session)

                # Clean up runs older than 90 days
                deleted_count = jobs_service.cleanup_old_runs(days_to_keep=90)

                _logger.info(f"Cleaned up {deleted_count} old runs")

        except Exception as e:
            _logger.error(f"Failed to cleanup old runs: {e}")

    def _job_executed(self, event):
        """Handle job execution events."""
        _logger.debug(f"Job executed: {event.job_id}")

    def _job_error(self, event):
        """Handle job error events."""
        _logger.error(f"Job error: {event.job_id} - {event.exception}")


async def main():
    """Main entry point for the scheduler process."""
    _logger.info("Starting job scheduler process...")

    # Setup Dramatiq
    from src.backend.workers.dramatiq_config import setup_dramatiq
    setup_dramatiq()

    # Create and start scheduler
    scheduler = JobScheduler()

    try:
        await scheduler.start()
    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt")
    except Exception as e:
        _logger.error(f"Scheduler process failed: {e}")
        raise
    finally:
        scheduler.stop()


if __name__ == "__main__":
    # Run the scheduler
    asyncio.run(main())

