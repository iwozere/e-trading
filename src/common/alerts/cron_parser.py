"""
Cron expression parser with support for 5-field and 6-field formats.

This module provides timezone-aware cron parsing and validation using croniter.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass
import pytz
from croniter import croniter

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class CronExpression:
    """Represents a parsed cron expression with metadata."""
    expression: str
    fields: list[str]
    is_six_field: bool
    next_run: datetime


class CronParser:
    """
    Parser for cron expressions supporting both 5-field and 6-field formats.

    5-field format: minute hour day month weekday
    6-field format: second minute hour day month weekday
    """

    @staticmethod
    def parse_cron(expression: str, timezone: Optional[str] = None) -> CronExpression:
        """
        Parse a cron expression and return a CronExpression object.

        Args:
            expression: The cron expression string
            timezone: Optional timezone string (e.g., 'UTC', 'America/New_York')

        Returns:
            CronExpression object with parsed metadata

        Raises:
            ValueError: If the cron expression is invalid
        """
        if not expression or not isinstance(expression, str):
            raise ValueError("Cron expression must be a non-empty string")

        expression = expression.strip()
        fields = expression.split()

        if len(fields) not in [5, 6]:
            raise ValueError(f"Cron expression must have 5 or 6 fields, got {len(fields)}")

        is_six_field = len(fields) == 6

        try:
            # Validate the expression by creating a croniter instance
            tz = pytz.timezone(timezone) if timezone else pytz.UTC
            base_time = datetime.now(tz)

            cron = croniter(expression, base_time)
            next_run = cron.get_next(datetime)

            _logger.debug("Parsed cron expression: %s (6-field: %s)", expression, is_six_field)

            return CronExpression(
                expression=expression,
                fields=fields,
                is_six_field=is_six_field,
                next_run=next_run
            )

        except Exception as e:
            _logger.exception("Failed to parse cron expression '%s':", expression)
            raise ValueError(f"Invalid cron expression '{expression}': {str(e)}")

    @staticmethod
    def validate_cron(expression: str) -> bool:
        """
        Validate a cron expression without parsing.

        Args:
            expression: The cron expression string

        Returns:
            True if valid, False otherwise
        """
        try:
            CronParser.parse_cron(expression)
            return True
        except ValueError:
            return False

    @staticmethod
    def calculate_next_run(expression: str, from_time: Optional[datetime] = None,
                          timezone: Optional[str] = None) -> datetime:
        """
        Calculate the next run time for a cron expression.

        Args:
            expression: The cron expression string
            from_time: Base time to calculate from (defaults to now)
            timezone: Optional timezone string

        Returns:
            Next run datetime (timezone-aware)

        Raises:
            ValueError: If the cron expression is invalid
        """
        if not CronParser.validate_cron(expression):
            raise ValueError(f"Invalid cron expression: {expression}")

        tz = pytz.timezone(timezone) if timezone else pytz.UTC
        base_time = from_time or datetime.now(tz)

        # Ensure base_time is timezone-aware
        if base_time.tzinfo is None:
            base_time = tz.localize(base_time)

        try:
            cron = croniter(expression, base_time)
            next_run = cron.get_next(datetime)

            _logger.debug("Next run for '%s' from %s: %s", expression, base_time, next_run)
            return next_run

        except Exception as e:
            _logger.exception("Failed to calculate next run for '%s':", expression)
            raise ValueError(f"Error calculating next run: {str(e)}")

    @staticmethod
    def is_six_field_cron(expression: str) -> bool:
        """
        Check if a cron expression uses 6-field format (includes seconds).

        Args:
            expression: The cron expression string

        Returns:
            True if 6-field format, False if 5-field format

        Raises:
            ValueError: If the expression is invalid
        """
        if not expression or not isinstance(expression, str):
            raise ValueError("Cron expression must be a non-empty string")

        fields = expression.strip().split()

        if len(fields) == 6:
            return True
        elif len(fields) == 5:
            return False
        else:
            raise ValueError(f"Invalid cron expression: must have 5 or 6 fields, got {len(fields)}")

    @staticmethod
    def get_next_n_runs(expression: str, count: int, from_time: Optional[datetime] = None,
                       timezone: Optional[str] = None) -> list[datetime]:
        """
        Get the next N run times for a cron expression.

        Args:
            expression: The cron expression string
            count: Number of future runs to calculate
            from_time: Base time to calculate from (defaults to now)
            timezone: Optional timezone string

        Returns:
            List of next N run datetimes

        Raises:
            ValueError: If the cron expression is invalid or count is invalid
        """
        if count <= 0:
            raise ValueError("Count must be positive")

        if not CronParser.validate_cron(expression):
            raise ValueError(f"Invalid cron expression: {expression}")

        tz = pytz.timezone(timezone) if timezone else pytz.UTC
        base_time = from_time or datetime.now(tz)

        # Ensure base_time is timezone-aware
        if base_time.tzinfo is None:
            base_time = tz.localize(base_time)

        try:
            cron = croniter(expression, base_time)
            runs = []

            for _ in range(count):
                next_run = cron.get_next(datetime)
                runs.append(next_run)

            _logger.debug("Generated %d runs for '%s' starting from %s", count, expression, base_time)
            return runs

        except Exception as e:
            _logger.exception("Failed to generate runs for '%s':", expression)
            raise ValueError(f"Error generating runs: {str(e)}")