"""
Scheduler Service

Main APScheduler-based service for job scheduling and execution.
Provides centralized scheduling with database persistence and error handling.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
import traceback
import asyncpg
import json
import subprocess

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import (
    JobExecutionEvent, JobSubmissionEvent, EVENT_JOB_SUBMITTED, EVENT_JOB_EXECUTED, EVENT_JOB_ERROR,
    EVENT_JOB_MISSED
)

from src.data.db.services.jobs_service import JobsService
from src.data.db.services.notification_service import NotificationService
from src.data.db.models.model_jobs import Schedule, ScheduleRun, RunStatus, JobType, ScheduleResponse, ScheduleRunResponse
from src.common.alerts.cron_parser import CronParser
from src.common.alerts.alert_evaluator import AlertEvaluator
from src.notification.logger import setup_logger

# Import MessagePriority enum for compatibility
from enum import Enum

class MessagePriority(str, Enum):
    """Message priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

_logger = setup_logger(__name__)
UTC = timezone.utc

# Global service instance reference for pickle-safe job execution
_service_instance: Optional['SchedulerService'] = None


async def execute_job_wrapper(schedule_id: int) -> None:
    """
    Module-level wrapper for job execution to avoid pickling the service instance.
    APScheduler pickles the function reference, not the instance.
    """
    if _service_instance:
        await _service_instance._execute_job(schedule_id)
    else:
        _logger.error("Cannot execute job %d: SchedulerService instance not available", schedule_id)


class SchedulerService:
    """
    Main scheduler service that orchestrates job scheduling and execution.

    This service:
    - Initializes APScheduler with database persistence
    - Loads enabled schedules from database on startup
    - Registers jobs with APScheduler for execution
    - Handles job execution callbacks and error recovery
    - Manages service lifecycle
    """

    def __init__(self,
                 jobs_service: JobsService,
                 alert_evaluator: AlertEvaluator,
                 notification_db_service: NotificationService,
                 database_url: str,
                 max_workers: int = 10):
        """
        Initialize the scheduler service.

        Args:
            jobs_service: Service for database operations
            alert_evaluator: Service for alert evaluation
            notification_db_service: Database service for notifications (both alerts and data processing)
            database_url: Database connection URL for APScheduler job store only
            max_workers: Maximum number of worker threads
        """
        self.jobs_service = jobs_service
        self.alert_evaluator = alert_evaluator
        self.notification_db_service = notification_db_service
        self.database_url = database_url
        self.max_workers = max_workers

        # APScheduler components
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.jobstore: Optional[SQLAlchemyJobStore] = None
        self.notification_client = None  # Placeholder for legacy/future notification client

        # DB Listener
        self._db_listener_task: Optional[asyncio.Task] = None

        # Service state
        self.is_running = False
        self.startup_retry_count = 0
        self.max_startup_retries = 3

        _logger.info("SchedulerService initialized with max_workers=%d", max_workers)

        # Set global instance
        global _service_instance
        _service_instance = self

    async def start(self) -> None:
        """
        Start the scheduler service with retry logic.

        This method:
        - Initializes APScheduler with PostgreSQL datastore
        - Loads enabled schedules from database
        - Registers jobs with APScheduler
        - Starts the scheduler

        Raises:
            RuntimeError: If service is already running or startup fails
        """
        if self.is_running:
            raise RuntimeError("Scheduler service is already running")

        _logger.info("Starting scheduler service...")

        while self.startup_retry_count < self.max_startup_retries:
            try:
                await self._initialize_scheduler()
                await self._load_and_register_schedules()
                self.scheduler.start()

                # Start DB listener
                self._db_listener_task = asyncio.create_task(self._listen_for_db_changes())

                self.is_running = True
                self.startup_retry_count = 0

                _logger.info("Scheduler service started successfully")
                return

            except Exception as e:
                self.startup_retry_count += 1
                _logger.error("Failed to start scheduler service (attempt %d/%d): %s",
                            self.startup_retry_count, self.max_startup_retries, str(e))

                if self.startup_retry_count >= self.max_startup_retries:
                    _logger.error("Maximum startup retries exceeded, giving up")
                    raise RuntimeError(f"Failed to start scheduler after {self.max_startup_retries} attempts") from e

                # Wait before retry with exponential backoff
                wait_time = 2 ** self.startup_retry_count
                _logger.info("Retrying startup in %d seconds...", wait_time)
                await asyncio.sleep(wait_time)

                # Clean up any partial initialization
                await self._cleanup_scheduler()

        raise RuntimeError("Startup failed after all retries")

    async def stop(self) -> None:
        """
        Stop the scheduler service gracefully.

        This method:
        - Stops the APScheduler
        - Waits for running jobs to complete
        - Cleans up resources
        """
        if not self.is_running:
            _logger.warning("Scheduler service is not running")
            return

        _logger.info("Stopping scheduler service...")

        try:
            if self.scheduler:
                self.scheduler.shutdown()

            if self._db_listener_task:
                self._db_listener_task.cancel()
                try:
                    await self._db_listener_task
                except asyncio.CancelledError:
                    pass
                self._db_listener_task = None

            await self._cleanup_scheduler()

            self.is_running = False
            _logger.info("Scheduler service stopped successfully")

        except Exception:
            _logger.exception("Error stopping scheduler service:")
            raise

    async def reload_schedules(self) -> int:
        """
        Reload schedules from database and update APScheduler.

        This method:
        - Removes all existing jobs from APScheduler
        - Loads current enabled schedules from database
        - Registers updated jobs with APScheduler

        Returns:
            Number of schedules loaded and registered

        Raises:
            RuntimeError: If service is not running
        """
        if not self.is_running:
            raise RuntimeError("Cannot reload schedules - service is not running")

        _logger.info("Reloading schedules...")

        try:
            # Remove all existing jobs
            await self._clear_all_jobs()

            # Load and register current schedules
            count = await self._load_and_register_schedules()

            _logger.info("Successfully reloaded %d schedules", count)
            return count

        except Exception:
            _logger.exception("Error reloading schedules:")
            raise

    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get current scheduler status and statistics.

        Returns:
            Dictionary with scheduler status information
        """
        status = {
            "is_running": self.is_running,
            "startup_retry_count": self.startup_retry_count,
            "max_workers": self.max_workers,
            "scheduler_state": None,
            "job_count": 0,
            "datastore_url": self.database_url,
            "notification_client": self.notification_client.get_stats() if self.notification_client else None
        }

        if self.scheduler:
            status["scheduler_state"] = str(self.scheduler.state)
            # Note: APScheduler 4.x doesn't have a direct way to get job count
            # This would need to be implemented by querying the datastore directly

        return status

    async def check_notification_health(self) -> Dict[str, Any]:
        """
        Check notification service health.

        Returns:
            Dictionary with notification service health status
        """
        try:
            if not self.notification_client:
                return {"status": "unavailable", "error": "No notification client configured"}

            health_status = await self.notification_client.get_health_status()
            return health_status

        except Exception as e:
            _logger.exception("Error checking notification service health:")
            return {"status": "unhealthy", "error": str(e)}

    async def _initialize_scheduler(self) -> None:
        """
        Initialize APScheduler with PostgreSQL datastore and async executor.

        Raises:
            Exception: If initialization fails
        """
        try:
            # Create SQLAlchemy jobstore for job persistence
            self.jobstore = SQLAlchemyJobStore(url=self.database_url)

            # Create async executor
            # Note: AsyncIOExecutor doesn't support max_workers parameter
            # The asyncio event loop handles concurrency automatically
            executor = AsyncIOExecutor()

            # Configure jobstores and executors
            jobstores = {
                'default': self.jobstore
            }
            executors = {
                'default': executor
            }

            # Job defaults
            job_defaults = {
                'coalesce': False,
                'max_instances': self.max_workers  # Use max_workers here for concurrent job instances
            }

            # Create scheduler with jobstore and executor
            self.scheduler = AsyncIOScheduler(
                jobstores=jobstores,
                executors=executors,
                job_defaults=job_defaults,
                timezone=UTC
            )

            # Add event listeners for job execution tracking
            self.scheduler.add_listener(
                self._on_job_submitted,
                EVENT_JOB_SUBMITTED
            )

            self.scheduler.add_listener(
                self._on_job_executed,
                EVENT_JOB_EXECUTED
            )

            self.scheduler.add_listener(
                self._on_job_error,
                EVENT_JOB_ERROR
            )

            self.scheduler.add_listener(
                self._on_job_missed,
                EVENT_JOB_MISSED
            )

            _logger.debug("APScheduler initialized successfully")

        except Exception:
            _logger.exception("Failed to initialize APScheduler:")
            raise

    async def _load_and_register_schedules(self) -> int:
        """
        Load enabled schedules from database and register with APScheduler.

        Returns:
            Number of schedules successfully registered

        Raises:
            Exception: If loading or registration fails
        """
        try:
            # Get all enabled schedules
            schedules = self.jobs_service.list_schedules(enabled=True, limit=1000)

            registered_count = 0

            for schedule in schedules:
                try:
                    await self._register_schedule(schedule)
                    registered_count += 1

                except Exception as e:
                    _logger.error("Failed to register schedule %s (ID: %d): %s",
                                schedule.name, schedule.id, str(e))
                    # Continue with other schedules
                    continue

            _logger.info("Registered %d out of %d enabled schedules",
                        registered_count, len(schedules))

            return registered_count

        except Exception:
            _logger.exception("Failed to load schedules from database:")
            raise

    async def _register_schedule(self, schedule: ScheduleResponse) -> None:
        """
        Register a single schedule with APScheduler.

        Args:
            schedule: Schedule object to register

        Raises:
            Exception: If registration fails
        """
        try:
            # Validate cron expression
            if not CronParser.validate_cron(schedule.cron):
                raise ValueError(f"Invalid cron expression: {schedule.cron}")

            # Create cron trigger
            # APScheduler 4.x uses different syntax for cron triggers
            cron_fields = schedule.cron.split()
            if len(cron_fields) == 5:
                # 5-field cron: minute hour day month weekday
                trigger = CronTrigger(
                    minute=cron_fields[0],
                    hour=cron_fields[1],
                    day=cron_fields[2],
                    month=cron_fields[3],
                    day_of_week=cron_fields[4],
                    timezone=UTC
                )
            elif len(cron_fields) == 6:
                # 6-field cron: second minute hour day month weekday
                trigger = CronTrigger(
                    second=cron_fields[0],
                    minute=cron_fields[1],
                    hour=cron_fields[2],
                    day=cron_fields[3],
                    month=cron_fields[4],
                    day_of_week=cron_fields[5],
                    timezone=UTC
                )
            else:
                raise ValueError(f"Cron expression must have 5 or 6 fields: {schedule.cron}")

            # Create job ID
            job_id = f"schedule_{schedule.id}"

            # Register job with APScheduler
            # Use module-level wrapper instead of instance method to avoid pickling errors
            self.scheduler.add_job(
                func=execute_job_wrapper,
                trigger=trigger,
                id=job_id,
                args=[schedule.id],
                replace_existing=True
            )

            _logger.debug("Registered schedule %s (ID: %d) with job ID: %s",
                         schedule.name, schedule.id, job_id)

        except Exception as e:
            _logger.error("Failed to register schedule %s (ID: %d): %s",
                         schedule.name, schedule.id, str(e))
            raise

    async def _execute_job(self, schedule_id: int) -> None:
        """
        Execute a scheduled job.

        This method:
        - Creates a ScheduleRun record
        - Executes the job based on its type
        - Updates the run record with results
        - Handles errors and timeouts

        Args:
            schedule_id: ID of the schedule to execute
        """
        run_record = None

        try:
            # Get schedule details
            schedule = self.jobs_service.get_schedule(schedule_id)
            if not schedule:
                _logger.error("Schedule %d not found for execution", schedule_id)
                return

            if not schedule.enabled:
                _logger.warning("Schedule %d is disabled, skipping execution", schedule_id)
                return

            # Create run record
            run_record = await self._create_run_record(schedule)

            _logger.info("Executing job for schedule %s (ID: %d, Run: %d)",
                        schedule.name, schedule.id, run_record.id)

            # Execute based on job type
            result = None
            if schedule.job_type == JobType.ALERT.value:
                result = await self._execute_alert_job(schedule, run_record)
            elif schedule.job_type == JobType.SCREENER.value:
                result = await self._execute_screener_job(schedule, run_record)
            elif schedule.job_type == JobType.REPORT.value:
                result = await self._execute_report_job(schedule, run_record)
            elif schedule.job_type == JobType.DATA_PROCESSING.value:
                result = await self._execute_data_processing_job(schedule, run_record)
            else:
                raise ValueError(f"Unsupported job type: {schedule.job_type}")

            # Update run record with success
            await self._complete_run_record(run_record, RunStatus.COMPLETED, result)

            # Update schedule's next run time
            self.jobs_service.update_schedule_next_run(schedule_id)

            _logger.info("Successfully completed job for schedule %s (ID: %d)",
                        schedule.name, schedule.id)

        except Exception as e:
            error_msg = f"Job execution failed: {str(e)}"
            _logger.error("Error executing job for schedule %d: %s", schedule_id, error_msg)
            _logger.debug("Job execution error traceback: %s", traceback.format_exc())

            # Update run record with failure
            if run_record:
                await self._complete_run_record(run_record, RunStatus.FAILED, None, error_msg)

    async def _execute_alert_job(self, schedule: ScheduleResponse, run_record: ScheduleRunResponse) -> Dict[str, Any]:
        """
        Execute an alert job using AlertEvaluator.

        Args:
            schedule: Schedule object
            run_record: ScheduleRun object

        Returns:
            Dictionary with execution results
        """
        try:
            # Use AlertEvaluator to process the alert
            result = await self.alert_evaluator.evaluate_alert(run_record)

            # Update schedule state if needed
            if result.state_updates:
                await self._update_schedule_state(schedule.id, result.state_updates)

            # Send notification if triggered
            notification_sent = False
            if result.triggered and result.notification_data:
                notification_sent = await self._send_notification(result.notification_data)

            return {
                "triggered": result.triggered,
                "rearmed": result.rearmed,
                "error": result.error,
                "notification_sent": notification_sent
            }

        except Exception:
            _logger.exception("Error executing alert job:")
            raise

    async def _execute_screener_job(self, schedule: ScheduleResponse, run_record: ScheduleRunResponse) -> Dict[str, Any]:
        """
        Execute a screener job.

        Args:
            schedule: Schedule object
            run_record: ScheduleRun object

        Returns:
            Dictionary with execution results
        """
        # TODO: Implement screener job execution
        # This would integrate with the screener service
        _logger.info("Screener job execution not yet implemented for schedule %d", schedule.id)

        return {
            "status": "not_implemented",
            "message": "Screener job execution will be implemented in future version"
        }

    async def _execute_report_job(self, schedule: ScheduleResponse, run_record: ScheduleRunResponse) -> Dict[str, Any]:
        """
        Execute a report job.

        Args:
            schedule: Schedule object
            run_record: ScheduleRun object

        Returns:
            Dictionary with execution results
        """
        # TODO: Implement report job execution
        # This would integrate with the reporting service
        _logger.info("Report job execution not yet implemented for schedule %d", schedule.id)

        return {
            "status": "not_implemented",
            "message": "Report job execution will be implemented in future version"
        }

    async def _execute_data_processing_job(self, schedule: ScheduleResponse, run_record: ScheduleRunResponse) -> Dict[str, Any]:
        """
        Execute data processing job via subprocess.

        This method runs Python scripts as subprocesses and handles their output.
        """
        task_params = schedule.task_params or {}
        script_path = task_params.get("script_path")
        script_args = task_params.get("script_args", [])
        timeout_seconds = task_params.get("timeout_seconds", 600)  # Default 10 minutes

        if not script_path:
            raise ValueError("script_path is required in task_params")

        _logger.info("Executing data processing job: %s with args: %s", script_path, script_args)

        try:
            # Build command
            python_executable = sys.executable
            script_full_path = Path(PROJECT_ROOT) / script_path

            if not script_full_path.exists():
                raise FileNotFoundError(f"Script not found: {script_full_path}")

            cmd = [python_executable, str(script_full_path)] + script_args

            # Automatically inject user_id if present in the schedule
            # This ensures scripts like EMPS2 pipeline receive the user context
            if schedule.user_id:
                cmd.extend(["--user-id", str(schedule.user_id)])

            _logger.debug("Running command: %s", " ".join(cmd))
            _logger.debug("Timeout: %d seconds", timeout_seconds)

            # Run subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(PROJECT_ROOT)
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Script execution timed out after {timeout_seconds} seconds")

            exit_code = process.returncode
            stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""

            _logger.info("Script exit code: %d", exit_code)
            if stderr_str:
                _logger.warning("Script stderr: %s", stderr_str[:1000])  # Log first 1000 chars

            # Parse result from stdout
            script_result = self._parse_script_output(stdout_str)

            # Check if script execution was successful
            success = exit_code == 0

            result = {
                "success": success,
                "exit_code": exit_code,
                "script_result": script_result,
                "stdout_preview": stdout_str[-500:] if len(stdout_str) > 500 else stdout_str,  # Last 500 chars
                "stderr_preview": stderr_str[-500:] if len(stderr_str) > 500 else stderr_str,
                "notification_sent": False
            }

            # Evaluate notification rules if script succeeded
            if success:
                notification_sent = await self._evaluate_notification_rules(
                    schedule, script_result, task_params
                )
                result["notification_sent"] = notification_sent

            return result

        except Exception as e:
            _logger.exception("Error executing data processing job:")
            raise

    def _parse_script_output(self, stdout: str) -> Dict[str, Any]:
        """
        Parse script output to extract JSON result.

        Looks for a line starting with __SCHEDULER_RESULT__: followed by JSON.

        Args:
            stdout: Script stdout output

        Returns:
            Parsed result dictionary, or empty dict if not found
        """
        result = {}

        for line in stdout.splitlines():
            if line.startswith("__SCHEDULER_RESULT__:"):
                try:
                    json_str = line.split("__SCHEDULER_RESULT__:", 1)[1].strip()
                    result = json.loads(json_str)
                    _logger.debug("Parsed script result: %s", result)
                    break
                except json.JSONDecodeError as e:
                    _logger.error("Failed to parse script JSON result: %s", e)
                    result = {"parse_error": str(e), "raw": json_str[:200]}

        return result

    async def _evaluate_notification_rules(
        self,
        schedule: ScheduleResponse,
        script_result: Dict[str, Any],
        task_params: Dict[str, Any]
    ) -> bool:
        """
        Evaluate notification rules and send notifications if conditions are met.

        Args:
            schedule: Schedule object
            script_result: Parsed script result
            task_params: Task parameters with notification rules

        Returns:
            True if any notification was sent, False otherwise
        """
        from src.data.db.services.users_service import UsersService

        notification_rules = task_params.get("notification_rules", {})
        conditions = notification_rules.get("conditions", [])

        if not conditions:
            _logger.debug("No notification conditions defined")
            return False

        # Get user notification channels
        users_service = UsersService()
        user_channels = users_service.get_user_notification_channels(schedule.user_id)

        if not user_channels:
            _logger.warning("No notification channels found for user %d", schedule.user_id)
            return False

        _logger.debug("User notification channels: email=%s, telegram=%s",
                     user_channels.get("email"), user_channels.get("telegram_chat_id"))

        # Evaluate conditions and collect matching channels
        matching_channels = set()

        for condition in conditions:
            if self._check_condition(condition, script_result):
                channels = condition.get("channels", [])
                matching_channels.update(channels)
                _logger.info("Notification condition met: %s -> channels: %s",
                           condition, channels)

        if not matching_channels:
            _logger.debug("No notification conditions met")
            return False

        # Build notification message
        message_title = f"Job Alert: {schedule.name}"
        message_body_parts = [
            f"**Scheduled Job:** {schedule.name}",
            f"**Job Type:** {schedule.job_type}",
            f"**Execution Time:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ]

        # Add script results to message
        if script_result:
            message_body_parts.append("**Results:**")
            for key, value in script_result.items():
                if key not in ["parse_error", "raw"]:
                    message_body_parts.append(f"  • {key}: {value}")
            message_body_parts.append("")

        message_body = "\n".join(message_body_parts)

        # Send notification to matching channels
        notification_data = {
            "title": message_title,
            "message": message_body,
            "schedule_id": schedule.id,
            "schedule_name": schedule.name,
            "job_type": schedule.job_type,
            "script_result": script_result,
            "channels": list(matching_channels),
            "user_id": schedule.user_id
        }

        # Add channel-specific recipients
        if "email" in matching_channels and user_channels.get("email"):
            notification_data["email_receiver"] = user_channels["email"]

        if "telegram" in matching_channels and user_channels.get("telegram_chat_id"):
            notification_data["telegram_chat_id"] = user_channels["telegram_chat_id"]

        try:
            # Create notification message in database
            # The notification processor will handle delivery to channels
            message_data = {
                "message_type": "REPORT",
                "channels": list(matching_channels),
                "recipient_id": str(schedule.user_id),
                "content": {
                    "title": message_title,
                    "body": message_body,
                    "metadata": notification_data
                },
                "priority": "NORMAL",
                "source": "scheduler_data_processing"
            }

            message = self.notification_db_service.create_message(message_data)

            _logger.info("Created notification message %d for schedule %d to channels: %s",
                       message.id, schedule.id, matching_channels)

            return True

        except Exception as e:
            _logger.error("Error creating notification for schedule %d: %s", schedule.id, e)
            return False

    async def _listen_for_db_changes(self) -> None:
        """
        Listen for database notifications and reload schedules.
        """
        _logger.info("Starting database listener for scheduler updates...")

        conn = None
        current_db_url = self.database_url

        # Asyncpg requires postgresql://, fix if needed
        if "+psycopg2" in current_db_url:
            current_db_url = current_db_url.replace("+psycopg2", "")

        retry_count = 0

        while self.is_running:
            try:
                # SSL is not required for local network connection, bypassing permission issues with keys
                conn = await asyncpg.connect(current_db_url, ssl='disable')

                # Add listener
                await conn.add_listener('scheduler_updates', lambda *args: asyncio.create_task(self.reload_schedules()))

                _logger.info("Listening for 'scheduler_updates' notifications")
                retry_count = 0

                # Keep connection alive
                while self.is_running:
                    await asyncio.sleep(60)
                    if conn.is_closed():
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                retry_count += 1
                _logger.error("Database listener error (retry %d): %s", retry_count, e)
                await asyncio.sleep(min(30, 2 ** retry_count)) # Exponential backoff capped at 30s
            finally:
                if conn and not conn.is_closed():
                    try:
                        await conn.close()
                    except Exception:
                        pass

    def _check_condition(self, condition: Dict[str, Any], script_result: Dict[str, Any]) -> bool:
        """
        Check if a notification condition is met.

        Supports conditions like:
        {
            "check_field": "vix_current",
            "operator": ">=",
            "threshold": 20
        }

        Args:
            condition: Condition configuration
            script_result: Script result data

        Returns:
            True if condition is met, False otherwise
        """
        check_field = condition.get("check_field")
        operator = condition.get("operator", ">=")
        threshold = condition.get("threshold")

        if not check_field or threshold is None:
            _logger.warning("Invalid condition: %s", condition)
            return False

        field_value = script_result.get(check_field)

        if field_value is None:
            _logger.debug("Field %s not found in script result", check_field)
            return False

        try:
            field_value = float(field_value)
            threshold = float(threshold)

            if operator == ">=":
                return field_value >= threshold
            elif operator == ">":
                return field_value > threshold
            elif operator == "<=":
                return field_value <= threshold
            elif operator == "<":
                return field_value < threshold
            elif operator == "==":
                return field_value == threshold
            elif operator == "!=":
                return field_value != threshold
            else:
                _logger.warning("Unknown operator: %s", operator)
                return False

        except (ValueError, TypeError) as e:
            _logger.error("Error comparing values: %s", e)
            return False

    async def _create_run_record(self, schedule: ScheduleResponse) -> ScheduleRunResponse:
        """
        Create a ScheduleRun record for job execution tracking.

        Args:
            schedule: Schedule object

        Returns:
            Created ScheduleRun object
        """
        from src.data.db.models.model_jobs import ScheduleRunCreate

        # Create job snapshot
        job_snapshot = {
            "schedule_id": schedule.id,
            "schedule_name": schedule.name,
            "job_type": schedule.job_type,
            "target": schedule.target,
            "task_params": schedule.task_params,
            "cron": schedule.cron,
            "execution_time": datetime.now(UTC).isoformat()
        }

        # Create run data
        run_data = ScheduleRunCreate(
            job_type=JobType(schedule.job_type),
            job_id=str(schedule.id),
            scheduled_for=datetime.now(UTC),
            job_snapshot=job_snapshot
        )

        # Create run record
        run_record = self.jobs_service.create_run(schedule.user_id, run_data)

        # Update status to RUNNING
        from src.data.db.models.model_jobs import ScheduleRunUpdate
        update_data = ScheduleRunUpdate(
            status=RunStatus.RUNNING,
            started_at=datetime.now(UTC)
        )

        updated_run = self.jobs_service.update_run(run_record.id, update_data)
        return updated_run or run_record

    async def _complete_run_record(self, run_record: ScheduleRun, status: RunStatus,
                                 result: Optional[Dict[str, Any]] = None,
                                 error: Optional[str] = None) -> None:
        """
        Complete a ScheduleRun record with final status and results.

        Args:
            run_record: ScheduleRun object to update
            status: Final run status
            result: Execution results (optional)
            error: Error message if failed (optional)
        """
        from src.data.db.models.model_jobs import ScheduleRunUpdate

        update_data = ScheduleRunUpdate(
            status=status,
            finished_at=datetime.now(UTC),
            result=result,
            error=error
        )

        self.jobs_service.update_run(run_record.id, update_data)

    async def _update_schedule_state(self, schedule_id: int, state_updates: Dict[str, Any]) -> None:
        """
        Update schedule state in database.

        Args:
            schedule_id: Schedule ID
            state_updates: State updates to persist
        """
        try:
            # TODO: This requires adding state_json field to Schedule model
            # For now, we'll log the state update
            _logger.debug("Would update schedule %d state: %s", schedule_id, state_updates)

            # When state_json field is added to Schedule model, this would be:
            # state_json = json.dumps(state_updates, ensure_ascii=False, default=str)
            # self.jobs_service.update_schedule_state(schedule_id, state_json)

        except Exception:
            _logger.exception("Error updating schedule state for %d:", schedule_id)

    async def _send_notification(self, notification_data: Dict[str, Any]) -> bool:
        """
        Send notification for triggered alert with enhanced formatting and error handling.

        Args:
            notification_data: Notification data from alert evaluation

        Returns:
            True if notification was sent successfully, False otherwise
        """
        try:
            # Extract notification configuration
            notify_config = notification_data.get("notify_config", {})

            # Build enhanced notification title and message
            ticker = notification_data.get("ticker", "Unknown")
            timeframe = notification_data.get("timeframe", "Unknown")
            price = notification_data.get("price", 0.0)
            alert_name = notification_data.get("alert_name", "Alert")

            # Safely convert price to float
            try:
                price_float = float(price) if price is not None else 0.0
            except (ValueError, TypeError):
                price_float = 0.0
                _logger.warning("Invalid price value: %s, using 0.0", price)

            # Create contextual title
            title = f"🚨 {alert_name}: {ticker} ({timeframe})"

            # Build comprehensive message with enhanced formatting
            message_parts = [
                f"**Alert Triggered: {alert_name}**",
                f"📈 Symbol: {ticker}",
                f"⏱️ Timeframe: {timeframe}",
                f"💰 Current Price: ${price_float:.4f}",
                f"🕐 Triggered At: {notification_data.get('timestamp', datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'))}"
            ]

            # Add market context if available
            market_data = notification_data.get("market_data", {})
            if market_data:
                message_parts.append("\n**📊 Market Context:**")

                # Add OHLCV data if available
                if "open" in market_data:
                    try:
                        open_price = float(market_data["open"])
                        high_price = float(market_data.get("high", price_float))
                        low_price = float(market_data.get("low", price_float))
                        volume = float(market_data.get("volume", 0))

                        # Calculate price change
                        price_change = ((price_float - open_price) / open_price * 100) if open_price > 0 else 0
                        change_emoji = "📈" if price_change >= 0 else "📉"

                        message_parts.extend([
                            f"  • Open: ${open_price:.4f}",
                            f"  • High: ${high_price:.4f}",
                            f"  • Low: ${low_price:.4f}",
                            f"  • Change: {change_emoji} {price_change:+.2f}%"
                        ])

                        if volume > 0:
                            message_parts.append(f"  • Volume: {volume:,.0f}")

                    except (ValueError, TypeError):
                        _logger.warning("Invalid market data values, skipping market context")

            # Add technical indicators with enhanced formatting
            indicators = notification_data.get("indicators", {})
            if indicators:
                message_parts.append("\n**📊 Technical Indicators:**")

                # Group indicators by type for better readability
                trend_indicators = {}
                momentum_indicators = {}
                volatility_indicators = {}
                other_indicators = {}

                for name, value in indicators.items():
                    name_lower = name.lower()
                    if any(trend in name_lower for trend in ["sma", "ema", "ma", "trend", "supertrend"]):
                        trend_indicators[name] = value
                    elif any(momentum in name_lower for momentum in ["rsi", "macd", "stoch", "momentum"]):
                        momentum_indicators[name] = value
                    elif any(vol in name_lower for vol in ["bb", "bollinger", "atr", "volatility"]):
                        volatility_indicators[name] = value
                    else:
                        other_indicators[name] = value

                # Format indicators by category
                for category, indicators_dict in [
                    ("Trend", trend_indicators),
                    ("Momentum", momentum_indicators),
                    ("Volatility", volatility_indicators),
                    ("Other", other_indicators)
                ]:
                    if indicators_dict:
                        for name, value in indicators_dict.items():
                            if isinstance(value, bool):
                                message_parts.append(f"  • {name}: {value}")
                            elif isinstance(value, (int, float)):
                                message_parts.append(f"  • {name}: {value:.4f}")
                            else:
                                message_parts.append(f"  • {name}: {value}")

            # Add rule evaluation details
            rule_snapshot = notification_data.get("rule_snapshot", {})
            rule_description = notification_data.get("rule_description", "")

            if rule_snapshot or rule_description:
                message_parts.append("\n**📋 Alert Rule Details:**")

                if rule_description:
                    message_parts.append(f"  • Rule: {rule_description}")

                if rule_snapshot:
                    message_parts.append("  • Values:")
                    for name, value in rule_snapshot.items():
                        if isinstance(value, bool):
                            message_parts.append(f"    - {name}: {value}")
                        elif isinstance(value, (int, float)):
                            message_parts.append(f"    - {name}: {value:.4f}")
                        else:
                            message_parts.append(f"    - {name}: {value}")

            # Add rearm status if available
            rearm_info = notification_data.get("rearm_info", {})
            if rearm_info:
                message_parts.append("\n**🔄 Rearm Status:**")
                rearm_type = rearm_info.get("type", "unknown")
                rearm_value = rearm_info.get("value", "N/A")
                message_parts.append(f"  • Type: {rearm_type}")
                message_parts.append(f"  • Value: {rearm_value}")

            # Add alert configuration summary
            alert_config = notification_data.get("alert_config", {})
            if alert_config:
                message_parts.append("\n**⚙️ Alert Configuration:**")
                if "description" in alert_config:
                    message_parts.append(f"  • Description: {alert_config['description']}")
                if "priority" in alert_config:
                    message_parts.append(f"  • Priority: {alert_config['priority']}")

            # Add footer with action suggestions
            message_parts.extend([
                "\n" + "─" * 30,
                "💡 **Next Steps:**",
                "• Review your trading strategy",
                "• Check market conditions",
                "• Consider position sizing",
                "",
                f"🤖 Generated by Scheduler Service at {datetime.now(UTC).strftime('%H:%M:%S UTC')}"
            ])

            message = "\n".join(message_parts)

            # Determine channels from notify config with smart defaults
            channels = notify_config.get("channels", ["telegram"])
            if isinstance(channels, str):
                channels = [channels]

            # Add email for high-priority alerts
            priority = notify_config.get("priority", "normal")
            if priority in ["high", "critical"] and "email" not in channels:
                channels.append("email")

            # Determine recipient
            recipient_id = notify_config.get("recipient_id") or notify_config.get("user_id") or "default"

            # Adjust message priority based on alert configuration
            message_priority = MessagePriority.NORMAL
            if priority == "critical":
                message_priority = MessagePriority.CRITICAL
            elif priority == "high":
                message_priority = MessagePriority.HIGH
            elif priority == "low":
                message_priority = MessagePriority.LOW

            # Create notification message in database
            # The notification processor will handle delivery to channels
            try:
                # Map priority to database format
                priority_map = {
                    MessagePriority.CRITICAL: "CRITICAL",
                    MessagePriority.HIGH: "HIGH",
                    MessagePriority.NORMAL: "NORMAL",
                    MessagePriority.LOW: "LOW"
                }
                db_priority = priority_map.get(message_priority, "NORMAL")

                message_data = {
                    "message_type": "ALERT",
                    "channels": channels,
                    "recipient_id": str(recipient_id),
                    "content": {
                        "title": title,
                        "body": message,
                        "metadata": notification_data
                    },
                    "priority": db_priority,
                    "source": "scheduler_service"
                }

                message_record = self.notification_db_service.create_message(message_data)

                _logger.info("Created alert notification message %d for %s to channels: %s",
                           message_record.id, ticker, channels)
                return True

            except Exception as e:
                _logger.error("Error creating alert notification for %s: %s", ticker, str(e))
                return False

        except Exception:
            _logger.exception("Unexpected error sending enhanced notification:")
            _logger.debug("Notification error traceback: %s", traceback.format_exc())
            # Don't raise - notification failures shouldn't stop job execution
            return False

    async def _clear_all_jobs(self) -> None:
        """Remove all jobs from APScheduler."""
        if self.scheduler:
            # APScheduler 3.x way to remove all jobs
            jobs = self.scheduler.get_jobs()
            for job in jobs:
                self.scheduler.remove_job(job.id)

            _logger.debug("Cleared all jobs from scheduler")

    async def _cleanup_scheduler(self) -> None:
        """Clean up scheduler resources."""
        try:
            if self.scheduler:
                self.scheduler = None

            if self.jobstore:
                # Close jobstore connections if needed
                self.jobstore = None

            _logger.debug("Scheduler cleanup completed")

        except Exception:
            _logger.exception("Error during scheduler cleanup:")

    # Event handlers for job execution tracking

    def _on_job_submitted(self, event: JobSubmissionEvent) -> None:
        """Handle job submission events."""
        _logger.debug("Job submitted: %s (scheduled for: %s)",
                     event.job_id, event.scheduled_run_time)

    def _on_job_executed(self, event: JobExecutionEvent) -> None:
        """Handle job execution completion events."""
        _logger.debug("Job executed: %s", event.job_id)

    def _on_job_error(self, event: JobExecutionEvent) -> None:
        """Handle job execution error events."""
        _logger.error("Job error: %s - %s", event.job_id, event.exception)

    def _on_job_missed(self, event: JobExecutionEvent) -> None:
        """Handle missed job events."""
        _logger.warning("Job missed: %s (scheduled for: %s)",
                       event.job_id, event.scheduled_run_time)