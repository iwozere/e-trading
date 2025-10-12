"""
Cron Handler

Utility functions for handling cron expressions and scheduling.
"""

import croniter
from datetime import datetime, timedelta
from typing import Optional, List
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class CronHandler:
    """Handler for cron expression operations."""

    @staticmethod
    def validate_cron(cron_expression: str) -> bool:
        """
        Validate a cron expression.

        Args:
            cron_expression: Cron expression to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            croniter.croniter(cron_expression)
            return True
        except Exception as e:
            logger.error(f"Invalid cron expression '{cron_expression}': {e}")
            return False

    @staticmethod
    def calculate_next_run_time(cron_expression: str, from_time: Optional[datetime] = None) -> datetime:
        """
        Calculate the next run time for a cron expression.

        Args:
            cron_expression: Cron expression
            from_time: Base time to calculate from (defaults to now)

        Returns:
            Next run datetime
        """
        if from_time is None:
            from_time = datetime.utcnow()

        try:
            cron = croniter.croniter(cron_expression, from_time)
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Failed to calculate next run time for cron '{cron_expression}': {e}")
            # Return a default time (1 hour from now) if calculation fails
            return from_time + timedelta(hours=1)

    @staticmethod
    def get_next_n_runs(cron_expression: str, n: int = 5, from_time: Optional[datetime] = None) -> List[datetime]:
        """
        Get the next N run times for a cron expression.

        Args:
            cron_expression: Cron expression
            n: Number of run times to return
            from_time: Base time to calculate from (defaults to now)

        Returns:
            List of next run datetimes
        """
        if from_time is None:
            from_time = datetime.utcnow()

        try:
            cron = croniter.croniter(cron_expression, from_time)
            runs = []

            for _ in range(n):
                next_run = cron.get_next(datetime)
                runs.append(next_run)

            return runs
        except Exception as e:
            logger.error(f"Failed to get next runs for cron '{cron_expression}': {e}")
            return []

    @staticmethod
    def get_previous_run_time(cron_expression: str, from_time: Optional[datetime] = None) -> datetime:
        """
        Get the previous run time for a cron expression.

        Args:
            cron_expression: Cron expression
            from_time: Base time to calculate from (defaults to now)

        Returns:
            Previous run datetime
        """
        if from_time is None:
            from_time = datetime.utcnow()

        try:
            cron = croniter.croniter(cron_expression, from_time)
            return cron.get_prev(datetime)
        except Exception as e:
            logger.error(f"Failed to get previous run time for cron '{cron_expression}': {e}")
            # Return a default time (1 hour ago) if calculation fails
            return from_time - timedelta(hours=1)

    @staticmethod
    def is_due(cron_expression: str, last_run: Optional[datetime] = None) -> bool:
        """
        Check if a cron expression is due to run.

        Args:
            cron_expression: Cron expression
            last_run: Last run time (defaults to None)

        Returns:
            True if due to run, False otherwise
        """
        try:
            if last_run is None:
                # If no last run, check if it should have run in the last minute
                one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
                next_run = CronHandler.calculate_next_run_time(cron_expression, one_minute_ago)
                return next_run <= datetime.utcnow()
            else:
                # Check if the next run after last_run is now or in the past
                next_run = CronHandler.calculate_next_run_time(cron_expression, last_run)
                return next_run <= datetime.utcnow()
        except Exception as e:
            logger.error(f"Failed to check if cron is due: {e}")
            return False

    @staticmethod
    def get_cron_description(cron_expression: str) -> str:
        """
        Get a human-readable description of a cron expression.

        Args:
            cron_expression: Cron expression

        Returns:
            Human-readable description
        """
        try:
            parts = cron_expression.strip().split()
            if len(parts) != 5:
                return "Invalid cron expression"

            minute, hour, day, month, weekday = parts

            # Simple descriptions for common patterns
            if minute == "0" and hour == "0" and day == "*" and month == "*" and weekday == "*":
                return "Daily at midnight"
            elif minute == "0" and hour == "9" and day == "*" and month == "*" and weekday == "*":
                return "Daily at 9:00 AM"
            elif minute == "0" and hour == "*/6" and day == "*" and month == "*" and weekday == "*":
                return "Every 6 hours"
            elif minute == "0" and hour == "*" and day == "*" and month == "*" and weekday == "*":
                return "Every hour"
            elif minute == "*/15" and hour == "*" and day == "*" and month == "*" and weekday == "*":
                return "Every 15 minutes"
            elif minute == "*/5" and hour == "*" and day == "*" and month == "*" and weekday == "*":
                return "Every 5 minutes"
            elif minute == "0" and hour == "0" and day == "1" and month == "*" and weekday == "*":
                return "Monthly on the 1st at midnight"
            elif minute == "0" and hour == "9" and day == "*" and month == "*" and weekday == "1":
                return "Every Monday at 9:00 AM"
            else:
                return f"Cron: {cron_expression}"

        except Exception as e:
            logger.error(f"Failed to get cron description: {e}")
            return f"Cron: {cron_expression}"


# Common cron expressions
COMMON_CRON_EXPRESSIONS = {
    "every_minute": "*/1 * * * *",
    "every_5_minutes": "*/5 * * * *",
    "every_15_minutes": "*/15 * * * *",
    "every_30_minutes": "*/30 * * * *",
    "every_hour": "0 * * * *",
    "every_2_hours": "0 */2 * * *",
    "every_6_hours": "0 */6 * * *",
    "every_12_hours": "0 */12 * * *",
    "daily_midnight": "0 0 * * *",
    "daily_9am": "0 9 * * *",
    "daily_6pm": "0 18 * * *",
    "weekly_monday_9am": "0 9 * * 1",
    "monthly_1st_midnight": "0 0 1 * *",
    "weekdays_9am": "0 9 * * 1-5",
    "weekends_10am": "0 10 * * 6,0"
}


def get_common_cron_expressions() -> dict:
    """
    Get a dictionary of common cron expressions with descriptions.

    Returns:
        Dictionary mapping names to cron expressions
    """
    return COMMON_CRON_EXPRESSIONS.copy()


def validate_and_describe_cron(cron_expression: str) -> dict:
    """
    Validate a cron expression and return validation result with description.

    Args:
        cron_expression: Cron expression to validate

    Returns:
        Dictionary with validation result and description
    """
    is_valid = CronHandler.validate_cron(cron_expression)

    result = {
        "cron_expression": cron_expression,
        "is_valid": is_valid,
        "description": CronHandler.get_cron_description(cron_expression) if is_valid else "Invalid cron expression"
    }

    if is_valid:
        try:
            result["next_run"] = CronHandler.calculate_next_run_time(cron_expression)
            result["next_5_runs"] = CronHandler.get_next_n_runs(cron_expression, 5)
        except Exception as e:
            logger.error(f"Failed to calculate run times: {e}")
            result["next_run"] = None
            result["next_5_runs"] = []

    return result
