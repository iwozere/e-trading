"""
Unit tests for CronParser service.

Tests cron expression parsing, validation, and next run calculations
for both 5-field and 6-field formats.
"""

import unittest
from datetime import datetime
import pytz

from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.common.alerts.cron_parser import CronParser, CronExpression


class TestCronParser(unittest.TestCase):
    """Test cases for CronParser functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.utc = pytz.UTC
        self.est = pytz.timezone('America/New_York')

    def test_parse_valid_5_field_cron(self):
        """Test parsing valid 5-field cron expressions."""
        test_cases = [
            "0 12 * * *",      # Daily at noon
            "*/5 * * * *",     # Every 5 minutes
            "0 0 1 * *",       # First day of month
            "0 9-17 * * 1-5",  # Business hours weekdays
        ]

        for expression in test_cases:
            with self.subTest(expression=expression):
                result = CronParser.parse_cron(expression)

                self.assertIsInstance(result, CronExpression)
                self.assertEqual(result.expression, expression)
                self.assertEqual(len(result.fields), 5)
                self.assertFalse(result.is_six_field)
                self.assertIsInstance(result.next_run, datetime)
                self.assertIsNotNone(result.next_run.tzinfo)

    def test_parse_valid_6_field_cron(self):
        """Test parsing valid 6-field cron expressions."""
        test_cases = [
            "0 0 12 * * *",    # Daily at noon with seconds
            "*/30 */5 * * * *", # Every 30 seconds, every 5 minutes
            "0 0 0 1 * *",     # First day of month with seconds
        ]

        for expression in test_cases:
            with self.subTest(expression=expression):
                result = CronParser.parse_cron(expression)

                self.assertIsInstance(result, CronExpression)
                self.assertEqual(result.expression, expression)
                self.assertEqual(len(result.fields), 6)
                self.assertTrue(result.is_six_field)
                self.assertIsInstance(result.next_run, datetime)
                self.assertIsNotNone(result.next_run.tzinfo)

    def test_parse_invalid_cron_expressions(self):
        """Test parsing invalid cron expressions raises ValueError."""
        invalid_expressions = [
            "",                    # Empty string
            "* * *",              # Too few fields
            "* * * * * * *",      # Too many fields
            "60 * * * *",         # Invalid minute
            "* 25 * * *",         # Invalid hour
            "* * 32 * *",         # Invalid day
            "* * * 13 *",         # Invalid month
            "* * * * 8",          # Invalid weekday
        ]

        for expression in invalid_expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    CronParser.parse_cron(expression)

    def test_parse_with_timezone(self):
        """Test parsing cron expressions with different timezones."""
        expression = "0 12 * * *"  # Daily at noon

        # Test with UTC
        result_utc = CronParser.parse_cron(expression, timezone="UTC")
        self.assertEqual(str(result_utc.next_run.tzinfo), "UTC")

        # Test with EST
        result_est = CronParser.parse_cron(expression, timezone="America/New_York")
        self.assertIn("America/New_York", str(result_est.next_run.tzinfo))

        # The times should be different due to timezone
        self.assertNotEqual(result_utc.next_run, result_est.next_run)

    def test_validate_cron(self):
        """Test cron expression validation."""
        valid_expressions = [
            "0 12 * * *",
            "*/5 * * * *",
            "0 0 12 * * *",
        ]

        invalid_expressions = [
            "",
            "* * *",
            "60 * * * *",
        ]

        for expression in valid_expressions:
            with self.subTest(expression=expression):
                self.assertTrue(CronParser.validate_cron(expression))

        for expression in invalid_expressions:
            with self.subTest(expression=expression):
                self.assertFalse(CronParser.validate_cron(expression))

    def test_calculate_next_run(self):
        """Test next run calculation."""
        expression = "0 12 * * *"  # Daily at noon
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=self.utc)

        next_run = CronParser.calculate_next_run(expression, base_time)

        # Should be today at noon since base time is 10 AM
        expected = datetime(2024, 1, 1, 12, 0, 0, tzinfo=self.utc)
        self.assertEqual(next_run, expected)

    def test_calculate_next_run_with_timezone(self):
        """Test next run calculation with timezone."""
        expression = "0 12 * * *"  # Daily at noon
        base_time = datetime(2024, 1, 1, 10, 0, 0)  # Naive datetime

        next_run = CronParser.calculate_next_run(
            expression, base_time, timezone="America/New_York"
        )

        # Should be timezone-aware
        self.assertIsNotNone(next_run.tzinfo)
        self.assertIn("America/New_York", str(next_run.tzinfo))

    def test_is_six_field_cron(self):
        """Test detection of 6-field vs 5-field cron expressions."""
        five_field_expressions = [
            "0 12 * * *",
            "*/5 * * * *",
            "0 0 1 * *",
        ]

        six_field_expressions = [
            "0 0 12 * * *",
            "*/30 */5 * * * *",
            "0 0 0 1 * *",
        ]

        for expression in five_field_expressions:
            with self.subTest(expression=expression):
                self.assertFalse(CronParser.is_six_field_cron(expression))

        for expression in six_field_expressions:
            with self.subTest(expression=expression):
                self.assertTrue(CronParser.is_six_field_cron(expression))

    def test_is_six_field_cron_invalid(self):
        """Test is_six_field_cron with invalid expressions."""
        invalid_expressions = [
            "",
            "* * *",
            "* * * * * * *",
        ]

        for expression in invalid_expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    CronParser.is_six_field_cron(expression)

    def test_get_next_n_runs(self):
        """Test getting multiple next run times."""
        expression = "0 */2 * * *"  # Every 2 hours
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=self.utc)

        runs = CronParser.get_next_n_runs(expression, 3, base_time)

        self.assertEqual(len(runs), 3)

        # Should be at 12:00, 14:00, 16:00
        expected_times = [
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=self.utc),
            datetime(2024, 1, 1, 14, 0, 0, tzinfo=self.utc),
            datetime(2024, 1, 1, 16, 0, 0, tzinfo=self.utc),
        ]

        for i, expected in enumerate(expected_times):
            self.assertEqual(runs[i], expected)

    def test_get_next_n_runs_invalid_count(self):
        """Test get_next_n_runs with invalid count."""
        expression = "0 12 * * *"

        with self.assertRaises(ValueError):
            CronParser.get_next_n_runs(expression, 0)

        with self.assertRaises(ValueError):
            CronParser.get_next_n_runs(expression, -1)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with None values
        with self.assertRaises(ValueError):
            CronParser.parse_cron(None)

        # Test with non-string input
        with self.assertRaises(ValueError):
            CronParser.parse_cron(123)

        # Test whitespace handling
        result = CronParser.parse_cron("  0 12 * * *  ")
        self.assertEqual(result.expression, "0 12 * * *")

        # Test complex expressions
        complex_expr = "0 9,12,15 * * 1-5"  # 9 AM, noon, 3 PM on weekdays
        result = CronParser.parse_cron(complex_expr)
        self.assertIsInstance(result, CronExpression)


if __name__ == '__main__':
    unittest.main()