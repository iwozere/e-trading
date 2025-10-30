#!/usr/bin/env python3
"""
Alert Runner

Standalone script for evaluating alerts and sending notifications.
Can be run manually or scheduled via cron/scheduler.
"""

import sys
from pathlib import Path
import asyncio
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.alerts_service import AlertsService
from src.data.db.services.jobs_service import JobsService
from src.data.data_manager import DataManager
from src.indicators.service import IndicatorService
from src.notification.service.client import NotificationServiceClient
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class AlertRunner:
    """
    Alert evaluation runner that processes alerts and sends notifications.
    """

    def __init__(self):
        """Initialize the alert runner with required services."""
        self.data_manager = DataManager()
        self.indicator_service = IndicatorService()
        self.jobs_service = JobsService()
        self.alerts_service = AlertsService(
            self.jobs_service,
            self.data_manager,
            self.indicator_service
        )
        self.notification_client = NotificationServiceClient(
            service_url="database://localhost"  # Use database-only mode
        )

        _logger.info("AlertRunner initialized successfully")

    async def run_evaluation(self, user_id: int = None, limit: int = None) -> dict:
        """
        Run alert evaluation for specified user or all users.

        Args:
            user_id: Optional user ID to evaluate alerts for
            limit: Optional limit on number of alerts to evaluate

        Returns:
            Dictionary with evaluation results
        """
        try:
            _logger.info("Starting alert evaluation run (user_id=%s, limit=%s)", user_id, limit)

            # Evaluate alerts
            if user_id:
                results = await self.alerts_service.evaluate_user_alerts(user_id)
            else:
                results = await self.alerts_service.evaluate_all_alerts(limit)

            # Send notifications for triggered alerts
            notifications_sent = 0
            for result in results.get("results", []):
                if result["triggered"] and result["notification_data"]:
                    try:
                        await self._send_alert_notification(result)
                        notifications_sent += 1
                    except Exception as e:
                        _logger.error("Failed to send notification for alert %s: %s",
                                    result["job_id"], e)

            # Log summary
            _logger.info("Alert evaluation complete: %d evaluated, %d triggered, %d notifications sent, %d errors",
                        results["total_evaluated"], results["triggered"], notifications_sent, results["errors"])

            results["notifications_sent"] = notifications_sent
            return results

        except Exception as e:
            _logger.exception("Error during alert evaluation run")
            return {
                "total_evaluated": 0,
                "triggered": 0,
                "rearmed": 0,
                "errors": 1,
                "notifications_sent": 0,
                "error": str(e)
            }

    async def _send_alert_notification(self, alert_result: dict):
        """
        Send notification for a triggered alert.

        Args:
            alert_result: Alert evaluation result dictionary
        """
        try:
            notification_data = alert_result["notification_data"]

            # Prepare notification content
            title = f"🚨 Alert Triggered: {alert_result['ticker']}"
            message = notification_data.get("message", f"Alert for {alert_result['ticker']} has been triggered")

            # Get user ID from job
            job = self.jobs_service.get_job(alert_result["job_id"])
            if not job:
                _logger.error("Could not find job %s for notification", alert_result["job_id"])
                return

            user_id = str(job.user_id)

            # Send notification
            success = await self.notification_client.send_notification(
                notification_type="alert_triggered",
                title=title,
                message=message,
                priority="high",
                channels=["telegram", "email"],
                recipient_id=user_id,
                data={
                    "alert_id": alert_result["job_id"],
                    "ticker": alert_result["ticker"],
                    "trigger_time": datetime.now(timezone.utc).isoformat(),
                    **notification_data
                }
            )

            if success:
                _logger.info("Sent alert notification for job %s to user %s",
                           alert_result["job_id"], user_id)
            else:
                _logger.error("Failed to send alert notification for job %s", alert_result["job_id"])

        except Exception as e:
            _logger.exception("Error sending alert notification")
            raise


async def main():
    """Main entry point for the alert runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run alert evaluation")
    parser.add_argument("--user-id", type=int, help="Evaluate alerts for specific user ID")
    parser.add_argument("--limit", type=int, help="Limit number of alerts to evaluate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run alert runner
    runner = AlertRunner()
    results = await runner.run_evaluation(user_id=args.user_id, limit=args.limit)

    # Print summary
    print(f"Alert Evaluation Summary:")
    print(f"  Evaluated: {results['total_evaluated']}")
    print(f"  Triggered: {results['triggered']}")
    print(f"  Rearmed: {results['rearmed']}")
    print(f"  Notifications Sent: {results['notifications_sent']}")
    print(f"  Errors: {results['errors']}")

    if results.get("error"):
        print(f"  Error: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())