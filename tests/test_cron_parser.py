"""
Unit Tests for CronParser

Tests the CronParser service functionality including:
- 5-field and 6-field cron expression parsing
- Timezone handling and next run calculations
- Validation and error handling for invalid expressions
"""

import pytest
from datetime import datetime, timezone
import pytz

# Add src to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.common.alerts.cron_parser import CronParser, CronExpression
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestCronParser:
    """Test cases for CronParser functionality."""

    def test_parse_valid_5_field_cron(self):
        """Test parsing valid 5-field cron expressions."""
        # Test basic 5-field expression
        expression = "0 12 * * *"  # Daily at noon
        result = CronParser.parse_cron(expression)

        assert isinstance(result, CronExpression)
        assert result.expression == expression
        assert result.fields == ["0", "12", "*", "*", "*"]
        assert result.is_six_field is False
        assert isinstance(result.next_run, datetime)
        assert result.next_run.tzinfo is not None

    def test_parse_valid_6_field_cron(self):
        """Test parsing valid 6-field cron expressions."""
        # Test basic 6-field expression
        expression = "30 0 12 * * *"  # Daily at 12:00:30
        result = CronParser.parse_cron(expression)

        assert isinstance(result, CronExpression)
        assert result.expression == expression
        assert result.fields == ["30", "0", "12", "*", "*", "*"]
        assert result.is_six_field is True
        assert isinstance(result.next_run, datetime)
        assert result.next_run.tzinfo is not None

    def test_parse_cron_with_timezone(self):
        """Test parsing cron expressions with specific timezone."""
        expression = "0 12 * * *"
        timezone_str = "America/New_York"

        result = CronParser.parse_cron(expression, timezone=timezone_str)

        assert isinstance(result, CronExpression)
        assert result.next_run.tzinfo.zone == timezone_str

    def test_parse_cron_invalid_field_count(self):
        """Test parsing cron expressions with invalid field count."""
        # Test too few fields
        with pytest.raises(ValueError, match="must have 5 or 6 fields"):
            CronParser.parse_cron("0 12 *")

        # Test too many fields
        with pytest.raises(ValueError, match="must have 5 or 6 fields"):
            CronParser.parse_cron("0 0 12 * * * *")

    def test_parse_cron_empty_expression(self):
        """Test parsing empty or None cron expressions."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            CronParser.parse_cron("")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            CronParser.parse_cron(None)

    def test_parse_cron_invalid_expression(self):
        """Test parsing invalid cron expressions."""
        # Invalid minute value
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.parse_cron("60 12 * * *")

        # Invalid hour value
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.parse_cron("0 25 * * *")

        # Invalid day of month
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.parse_cron("0 12 32 * *")

    def test_validate_cron_valid_expressions(self):
        """Test validation of valid cron expressions."""
        valid_expressions = [
            "0 12 * * *",      # Daily at noon
            "*/5 * * * *",     # Every 5 minutes
            "0 0 1 * *",       # First day of month
            "0 9-17 * * 1-5",  # Business hours weekdays
            "30 0 12 * * *",   # 6-field: Daily at 12:00:30
            "*/10 * * * * *"   # 6-field: Every 10 seconds
        ]

        for expr in valid_expressions:
            assert CronParser.validate_cron(expr) is True

    def test_validate_cron_invalid_expressions(self):
        """Test validation of invalid cron expressions."""
        invalid_expressions = [
            "",                # Empty
            "0 12",           # Too few fields
            "60 12 * * *",    # Invalid minute
            "0 25 * * *",     # Invalid hour
            "0 12 32 * *",    # Invalid day
            "invalid format"   # Non-numeric
        ]

        for expr in invalid_expressions:
            assert CronParser.validate_cron(expr) is False

    def test_calculate_next_run_basic(self):
        """Test calculating next run time for basic expressions."""
        # Test with current time
        expression = "0 12 * * *"  # Daily at noon
        next_run = CronParser.calculate_next_run(expression)

        assert isinstance(next_run, datetime)
        assert next_run.tzinfo is not None
        assert next_run > datetime.now(timezone.utc)

    def test_calculate_next_run_with_base_time(self):
        """Test calculating next run time with specific base time."""
        expression = "0 12 * * *"  # Daily at noon
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

        next_run = CronParser.calculate_next_run(expression, from_time=base_time)

        # Should be same day at noon since base time is 10 AM
        expected = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert next_run == expected

    def test_calculate_next_run_with_timezone(self):
        """Test calculating next run time with specific timezone."""
        expression = "0 12 * * *"
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        timezone_str = "America/New_York"

        next_run = CronParser.calculate_next_run(
            expression,
            from_time=base_time,
            timezone=timezone_str
        )

        assert next_run.tzinfo.zone == timezone_str

    def test_calculate_next_run_invalid_expression(self):
        """Test calculating next run time with invalid expression."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.calculate_next_run("invalid expression")

    def test_is_six_field_cron_detection(self):
        """Test detection of 6-field vs 5-field cron expressions."""
        # Test 5-field expressions
        five_field_expressions = [
            "0 12 * * *",
            "*/5 * * * *",
            "0 0 1 * *"
        ]

        for expr in five_field_expressions:
            assert CronParser.is_six_field_cron(expr) is False

        # Test 6-field expressions
        six_field_expressions = [
            "30 0 12 * * *",
            "*/10 * * * * *",
            "0 0 0 1 * *"
        ]

        for expr in six_field_expressions:
            assert CronParser.is_six_field_cron(expr) is True

    def test_is_six_field_cron_invalid_expressions(self):
        """Test 6-field detection with invalid expressions."""
        with pytest.raises(ValueError, match="must have 5 or 6 fields"):
            CronParser.is_six_field_cron("0 12")

        with pytest.raises(ValueError, match="must have 5 or 6 fields"):
            CronParser.is_six_field_cron("0 0 12 * * * *")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            CronParser.is_six_field_cron("")

    def test_get_next_n_runs_basic(self):
        """Test getting multiple next run times."""
        expression = "0 12 * * *"  # Daily at noon
        count = 3

        runs = CronParser.get_next_n_runs(expression, count)

        assert len(runs) == count
        assert all(isinstance(run, datetime) for run in runs)
        assert all(run.tzinfo is not None for run in runs)

        # Verify runs are in chronological order
        for i in range(1, len(runs)):
            assert runs[i] > runs[i-1]

    def test_get_next_n_runs_with_base_time(self):
        """Test getting multiple next run times with base time."""
        expression = "0 */6 * * *"  # Every 6 hours
        count = 4
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        runs = CronParser.get_next_n_runs(expression, count, from_time=base_time)

        assert len(runs) == count
        expected_times = [
            datetime(2024, 1, 1, 6, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 18, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        ]

        assert runs == expected_times

    def test_get_next_n_runs_invalid_count(self):
        """Test getting next runs with invalid count."""
        expression = "0 12 * * *"

        with pytest.raises(ValueError, match="Count must be positive"):
            CronParser.get_next_n_runs(expression, 0)

        with pytest.raises(ValueError, match="Count must be positive"):
            CronParser.get_next_n_runs(expression, -1)

    def test_get_next_n_runs_invalid_expression(self):
        """Test getting next runs with invalid expression."""
        with pytest.raises(ValueError, match="Invalid cron expression"):
            CronParser.get_next_n_runs("invalid", 3)

    def test_complex_cron_expressions(self):
        """Test parsing complex but valid cron expressions."""
        complex_expressions = [
            "0 9-17 * * 1-5",      # Business hours weekdays
            "*/15 8-18 * * 1-5",   # Every 15 min during business hours
            "0 0 1,15 * *",        # 1st and 15th of month
            "0 2 * * 0",           # Sundays at 2 AM
            "30 */2 * * *",        # Every 2 hours at 30 minutes past
            "0 0 0 1 1 *"          # 6-field: New Year's Day at midnight
        ]

        for expr in complex_expressions:
            result = CronParser.parse_cron(expr)
            assert isinstance(result, CronExpression)
            assert result.expression == expr
            assert isinstance(result.next_run, datetime)

    def test_timezone_aware_calculations(self):
        """Test timezone-aware next run calculations."""
        expression = "0 12 * * *"  # Daily at noon

        # Test different timezones
        timezones = ["UTC", "America/New_York", "Europe/London", "Asia/Tokyo"]

        for tz_str in timezones:
            result = CronParser.parse_cron(expression, timezone=tz_str)
            assert result.next_run.tzinfo.zone == tz_str

            # Calculate next run with timezone
            next_run = CronParser.calculate_next_run(expression, timezone=tz_str)
            assert next_run.tzinfo.zone == tz_str

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test leap year handling
        expression = "0 0 29 2 *"  # Feb 29th
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)  # 2024 is leap year

        next_run = CronParser.calculate_next_run(expression, from_time=base_time)
        assert next_run.month == 2
        assert next_run.day == 29

        # Test end of month
        expression = "0 0 31 * *"  # 31st of month
        result = CronParser.parse_cron(expression)
        assert isinstance(result, CronExpression)

        # Test whitespace handling
        expression = "  0   12   *   *   *  "
        result = CronParser.parse_cron(expression)
        assert result.expression == "0   12   *   *   *"  # Preserves internal spacing
        assert len(result.fields) == 5

    def test_logging_behavior(self):
        """Test that parsing works correctly and handles errors appropriately."""
        # Test successful parsing
        result = CronParser.parse_cron("0 12 * * *")
        assert result is not None
        assert result.expression == "0 12 * * *"

        # Test error handling for invalid expression
        with pytest.raises(ValueError, match="must have 5 or 6 fields"):
            CronParser.parse_cron("invalid")

    def test_performance_with_many_runs(self):
        """Test performance when calculating many future runs."""
        expression = "*/5 * * * *"  # Every 5 minutes
        count = 100

        # This should complete reasonably quickly
        import time
        start_time = time.time()
        runs = CronParser.get_next_n_runs(expression, count)
        end_time = time.time()

        assert len(runs) == count
        assert end_time - start_time < 1.0  # Should complete in under 1 second

    def test_daylight_saving_time_handling(self):
        """Test handling of daylight saving time transitions."""
        # Test spring forward (2 AM becomes 3 AM)
        expression = "0 2 * * *"  # 2 AM daily
        tz = pytz.timezone("America/New_York")

        # Date when DST starts in 2024 (March 10)
        base_time = datetime(2024, 3, 9, 0, 0, 0)
        base_time = tz.localize(base_time)

        next_run = CronParser.calculate_next_run(expression, from_time=base_time, timezone="America/New_York")

        # Should handle DST transition gracefully
        assert isinstance(next_run, datetime)
        assert next_run.tzinfo.zone == "America/New_York"

    def test_concurrent_parsing(self):
        """Test thread safety of cron parsing operations."""
        import threading

        expressions = [
            "0 12 * * *",
            "*/5 * * * *",
            "0 0 1 * *",
            "30 0 12 * * *"
        ]

        results = []
        errors = []

        def parse_expression(expr):
            try:
                result = CronParser.parse_cron(expr)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for expr in expressions * 5:  # Test with 20 concurrent operations
            thread = threading.Thread(target=parse_expression, args=(expr,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert all(isinstance(r, CronExpression) for r in results)