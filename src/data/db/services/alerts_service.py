# src/data/db/services/alerts_service.py
"""
Alert Service

Service layer for alert operations that integrates with the centralized
alert evaluator and provides a clean API for telegram and other consumers.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from src.common.alerts.alert_evaluator import AlertEvaluator, AlertConfig, AlertEvaluationResult
from src.data.db.services.jobs_service import JobsService
from src.data.data_manager import DataManager
from src.indicators.service import IndicatorService
from src.common.alerts.schema_validator import AlertSchemaValidator
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class AlertsService:
    """
    Service layer for alert operations.

    Provides a clean API for creating, managing, and evaluating alerts
    while delegating the core evaluation logic to the centralized AlertEvaluator.
    """

    def __init__(self,
                 jobs_service: JobsService,
                 data_manager: DataManager,
                 indicator_service: IndicatorService):
        """
        Initialize the alerts service.

        Args:
            jobs_service: Service for job/schedule operations
            data_manager: Service for retrieving market data
            indicator_service: Service for calculating technical indicators
        """
        self.jobs_service = jobs_service
        self.data_manager = data_manager
        self.indicator_service = indicator_service

        # Initialize schema validator
        self.schema_validator = AlertSchemaValidator()

        # Initialize the centralized alert evaluator
        self.alert_evaluator = AlertEvaluator(
            data_manager=data_manager,
            indicator_service=indicator_service,
            jobs_service=jobs_service,
            schema_validator=self.schema_validator
        )

        _logger.info("AlertsService initialized successfully")

    async def create_alert(self, user_id: int, alert_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new alert.

        Args:
            user_id: User ID creating the alert
            alert_config: Alert configuration dictionary

        Returns:
            Dictionary with creation result
        """
        try:
            # Validate the alert configuration
            validation_result = self.schema_validator.validate_alert_config(alert_config)
            if not validation_result.is_valid:
                return {
                    "success": False,
                    "error": "Invalid alert configuration",
                    "validation_errors": validation_result.errors
                }

            # Create job schedule for the alert
            job_data = {
                "user_id": user_id,
                "name": f"Alert: {alert_config.get('ticker', 'Unknown')}",
                "job_type": "alert",
                "target": alert_config.get("ticker", ""),
                "task_params": alert_config,
                "cron": self._generate_cron_for_timeframe(alert_config.get("timeframe", "1d")),
                "enabled": True
            }

            job = self.jobs_service.create_job(job_data)

            _logger.info("Created alert job %s for user %s", job.id, user_id)

            return {
                "success": True,
                "job_id": job.id,
                "alert_config": alert_config
            }

        except Exception as e:
            _logger.exception("Error creating alert for user %s", user_id)
            return {
                "success": False,
                "error": str(e)
            }

    async def evaluate_user_alerts(self, user_id: int) -> Dict[str, Any]:
        """
        Evaluate all alerts for a specific user.

        Args:
            user_id: User ID to evaluate alerts for

        Returns:
            Dictionary with evaluation results
        """
        try:
            return await self.alert_evaluator.evaluate_all_alerts(user_id=user_id)
        except Exception as e:
            _logger.exception("Error evaluating alerts for user %s", user_id)
            return {
                "total_evaluated": 0,
                "triggered": 0,
                "rearmed": 0,
                "errors": 1,
                "results": [],
                "error": str(e)
            }

    async def evaluate_all_alerts(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate all active alerts across all users.

        Args:
            limit: Optional limit on number of alerts to evaluate

        Returns:
            Dictionary with evaluation results
        """
        try:
            return await self.alert_evaluator.evaluate_all_alerts(limit=limit)
        except Exception as e:
            _logger.exception("Error evaluating all alerts")
            return {
                "total_evaluated": 0,
                "triggered": 0,
                "rearmed": 0,
                "errors": 1,
                "results": [],
                "error": str(e)
            }

    def get_user_alerts(self, user_id: int, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get all alerts for a user.

        Args:
            user_id: User ID to get alerts for
            active_only: Whether to return only active alerts

        Returns:
            List of alert dictionaries
        """
        try:
            jobs = self.jobs_service.get_user_jobs(
                user_id=user_id,
                job_type="alert",
                enabled_only=active_only
            )

            alerts = []
            for job in jobs:
                alerts.append({
                    "id": job.id,
                    "name": job.name,
                    "ticker": job.task_params.get("ticker"),
                    "timeframe": job.task_params.get("timeframe"),
                    "rule": job.task_params.get("rule"),
                    "enabled": job.enabled,
                    "created_at": job.created_at,
                    "next_run_at": job.next_run_at
                })

            return alerts

        except Exception as e:
            _logger.exception("Error getting alerts for user %s", user_id)
            return []

    def update_alert(self, job_id: int, updates: Dict[str, Any]) -> bool:
        """
        Update an alert.

        Args:
            job_id: Job ID of the alert to update
            updates: Dictionary of fields to update

        Returns:
            True if update was successful
        """
        try:
            return self.jobs_service.update_job(job_id, updates)
        except Exception as e:
            _logger.exception("Error updating alert %s", job_id)
            return False

    def delete_alert(self, job_id: int) -> bool:
        """
        Delete an alert.

        Args:
            job_id: Job ID of the alert to delete

        Returns:
            True if deletion was successful
        """
        try:
            return self.jobs_service.delete_job(job_id)
        except Exception as e:
            _logger.exception("Error deleting alert %s", job_id)
            return False

    def _generate_cron_for_timeframe(self, timeframe: str) -> str:
        """
        Generate appropriate cron expression for a timeframe.

        Args:
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")

        Returns:
            Cron expression string
        """
        # Simple mapping - in production you might want more sophisticated logic
        timeframe_to_cron = {
            "1m": "* * * * *",      # Every minute
            "5m": "*/5 * * * *",    # Every 5 minutes
            "15m": "*/15 * * * *",  # Every 15 minutes
            "30m": "*/30 * * * *",  # Every 30 minutes
            "1h": "0 * * * *",      # Every hour
            "4h": "0 */4 * * *",    # Every 4 hours
            "1d": "0 9 * * *",      # Daily at 9 AM
            "1w": "0 9 * * 1",      # Weekly on Monday at 9 AM
        }

        return timeframe_to_cron.get(timeframe, "0 9 * * *")  # Default to daily