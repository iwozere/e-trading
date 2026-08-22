"""Unit tests for src.ml.pipeline.p21_momentum.calendar."""

from __future__ import annotations

import unittest
from datetime import date

from src.ml.pipeline.p21_momentum.calendar import (
    first_trading_day_of_month,
    is_first_trading_day_of_month,
    is_last_trading_day_of_month,
    is_trading_day,
    last_trading_day_of_month,
    trading_days,
)


class TestCalendar(unittest.TestCase):
    def test_weekday_is_trading_day(self):
        self.assertTrue(is_trading_day(date(2026, 8, 24)))  # Monday

    def test_weekend_is_not_trading_day(self):
        self.assertFalse(is_trading_day(date(2026, 8, 22)))  # Saturday
        self.assertFalse(is_trading_day(date(2026, 8, 23)))  # Sunday

    def test_thanksgiving_2026_is_not_trading_day(self):
        self.assertFalse(is_trading_day(date(2026, 11, 26)))

    def test_good_friday_2026_is_not_trading_day(self):
        # Good Friday 2026 = 2026-04-03
        self.assertFalse(is_trading_day(date(2026, 4, 3)))

    def test_last_trading_day_of_november_2026_skips_thanksgiving(self):
        last_day = last_trading_day_of_month(2026, 11)
        self.assertEqual(last_day, date(2026, 11, 30))

    def test_first_trading_day_of_january_2026(self):
        first_day = first_trading_day_of_month(2026, 1)
        self.assertEqual(first_day, date(2026, 1, 2))  # 2026-01-01 is a holiday

    def test_is_last_trading_day_of_month_true_and_false(self):
        self.assertTrue(is_last_trading_day_of_month(date(2026, 11, 30)))
        self.assertFalse(is_last_trading_day_of_month(date(2026, 11, 25)))

    def test_is_first_trading_day_of_month_true_and_false(self):
        self.assertTrue(is_first_trading_day_of_month(date(2026, 1, 2)))
        self.assertFalse(is_first_trading_day_of_month(date(2026, 1, 5)))

    def test_trading_days_excludes_weekends_and_holidays(self):
        days = trading_days(date(2026, 11, 25), date(2026, 11, 30))
        self.assertNotIn(date(2026, 11, 26), days)  # Thanksgiving
        self.assertNotIn(date(2026, 11, 28), days)  # Saturday
        self.assertIn(date(2026, 11, 25), days)
        self.assertIn(date(2026, 11, 30), days)

    def test_last_trading_day_of_december(self):
        last_day = last_trading_day_of_month(2026, 12)
        self.assertEqual(last_day.month, 12)
        self.assertTrue(is_trading_day(last_day))


if __name__ == "__main__":
    unittest.main()
