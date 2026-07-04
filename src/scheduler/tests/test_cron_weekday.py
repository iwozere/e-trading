"""
Tests for crontab weekday translation used when registering schedules.

These guard against the APScheduler weekday gotcha: APScheduler's native
``CronTrigger`` numbers weekdays 0=Monday..6=Sunday, whereas standard crontab
(and ``croniter``, used to compute ``next_run_at``) numbers them 0/7=Sunday,
1=Monday..6=Saturday. Without translation a ``'1-5'`` weekday field would fire
Tue-Sat instead of the intended Mon-Fri.
"""

from datetime import datetime, timedelta

import pytest
import pytz
from apscheduler.triggers.cron import CronTrigger

from src.scheduler.scheduler_service import crontab_weekday_to_apscheduler

UTC = pytz.UTC


@pytest.mark.parametrize(
    "field, expected",
    [
        ("*", "*"),
        ("1-5", "mon-fri"),
        ("0", "sun"),
        ("7", "sun"),
        ("6", "sat"),
        ("1,3,5", "mon,wed,fri"),
        ("0-4", "sun-thu"),
        ("*/2", "*/2"),
        ("mon-fri", "mon-fri"),
    ],
)
def test_crontab_weekday_to_apscheduler(field: str, expected: str) -> None:
    """Numeric crontab weekdays map to APScheduler names; operators preserved."""
    assert crontab_weekday_to_apscheduler(field) == expected


def _first_fire_dows(weekday_field: str, start: datetime, count: int) -> list[str]:
    """Return the weekday abbreviations of the next ``count`` fire times."""
    trigger = CronTrigger(
        minute="0",
        hour="13",
        day="*",
        month="*",
        day_of_week=crontab_weekday_to_apscheduler(weekday_field),
        timezone=UTC,
    )
    fires: list[str] = []
    cursor = start
    for _ in range(count):
        cursor = trigger.get_next_fire_time(None, cursor)
        fires.append(cursor.strftime("%a"))
        cursor = cursor + timedelta(minutes=1)
    return fires


def test_mon_fri_fires_on_monday() -> None:
    """'1-5' must fire on Monday and skip the weekend (crontab semantics)."""
    sunday = UTC.localize(datetime(2026, 6, 28, 0, 0, 0))  # Sun 2026-06-28
    fires = _first_fire_dows("1-5", sunday, 6)
    assert fires == ["Mon", "Tue", "Wed", "Thu", "Fri", "Mon"]


def test_mon_fri_excludes_saturday() -> None:
    """Saturday must NOT be a valid fire day for a Mon-Fri schedule."""
    saturday = UTC.localize(datetime(2026, 7, 4, 0, 0, 0))  # Sat 2026-07-04
    next_fire = CronTrigger(
        minute="0",
        hour="13",
        day="*",
        month="*",
        day_of_week=crontab_weekday_to_apscheduler("1-5"),
        timezone=UTC,
    ).get_next_fire_time(None, saturday)
    assert next_fire.strftime("%a") == "Mon"
