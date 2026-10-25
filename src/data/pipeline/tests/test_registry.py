"""Tests for `src.data.pipeline.registry` — the plugin registry itself."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest
from apscheduler.triggers.cron import CronTrigger

from src.data.pipeline.base_plugin import PluginValidationError
from src.data.pipeline.registry import PLUGIN_REGISTRY, get_by_category, get_by_name, list_categories
from src.scheduler.scheduler_service import crontab_weekday_to_apscheduler


def test_registry_not_empty():
    assert len(PLUGIN_REGISTRY) > 0


def test_no_duplicate_names():
    counts = Counter(spec.name for spec in PLUGIN_REGISTRY)
    dupes = [name for name, n in counts.items() if n > 1]
    assert dupes == []


def test_every_spec_validates():
    """Every registered script_path must resolve, exist, and be allowlisted."""
    failures = []
    for spec in PLUGIN_REGISTRY:
        try:
            spec.validate()
        except PluginValidationError as e:
            failures.append(str(e))
    assert failures == [], f"{len(failures)} invalid PluginSpec(s):\n" + "\n".join(failures)


def test_get_by_name_known_plugin():
    spec = get_by_name("P22 Daily Price Ingest")
    assert spec is not None
    assert spec.category == "p22"


def test_get_by_name_unknown_returns_none():
    assert get_by_name("does not exist") is None


def test_get_by_category():
    p20_specs = get_by_category("p20")
    assert len(p20_specs) > 0
    assert all(spec.category == "p20" for spec in p20_specs)


def test_get_by_category_unknown_returns_empty():
    assert get_by_category("no-such-category") == []


def test_list_categories_includes_known_groups():
    categories = list_categories()
    assert {"p20", "p21", "p22"}.issubset(set(categories))


@pytest.mark.parametrize("category", ["p20", "p21", "p22"])
def test_category_crons_are_nonempty_strings(category):
    for spec in get_by_category(category):
        assert isinstance(spec.cron, str) and spec.cron.strip()


def _fire_times_on(cron: str, day: datetime) -> set[datetime]:
    """All fire times of `cron` within the UTC calendar day starting at `day`."""
    # Built the way scheduler_service does (crontab weekday numbering, not from_crontab's 0=Monday).
    # start_date pinned to `day`: the default ("now") would skip a past `day` entirely.
    minute, hour, dom, month, dow = cron.split()
    trigger = CronTrigger(
        minute=minute,
        hour=hour,
        day=dom,
        month=month,
        day_of_week=crontab_weekday_to_apscheduler(dow),
        timezone=timezone.utc,
        start_date=day,
    )
    end = day + timedelta(days=1)
    times: set[datetime] = set()
    current = trigger.get_next_fire_time(None, day)
    while current is not None and current < end:
        times.add(current)
        current = trigger.get_next_fire_time(current, current + timedelta(seconds=1))
    return times


def test_dependents_never_fire_with_their_dependency():
    """A job must not share a fire time with a job it depends_on.

    Both start together, the dependent's dependency check sees the upstream run
    still 'running' and defers that slot -- e.g. P19 Shadow Poll vs Watchlist Build
    at 13:00 UTC (monitoring.txt, 2026-10-14).
    """
    monday = datetime(2026, 10, 12, tzinfo=timezone.utc)
    collisions = []
    for spec in PLUGIN_REGISTRY:
        for dep_name in spec.depends_on:
            dep = get_by_name(dep_name)
            if dep is None:
                continue
            shared = _fire_times_on(spec.cron, monday) & _fire_times_on(dep.cron, monday)
            if shared:
                first = min(shared).strftime("%H:%M")
                collisions.append(f"{spec.name} and {dep.name} both fire at {first} UTC")
    assert collisions == [], "\n".join(collisions)


def test_p18_quarterly_backfill_fires_mid_month_after_each_quarter():
    """The P18 backfill fires on the 15th of Feb/May/Aug/Nov and targets the quarter whose 13F deadline is that month."""
    from src.ml.pipeline.p18_institutional_flow_tracker.pipeline import _quarter_end_date, _resolve_current_quarter

    spec = get_by_name("P18 Quarterly Consensus Backfill")
    assert spec is not None
    assert spec.script_args == ["--auto-quarter"]
    assert spec.timeout_seconds >= 12 * 3600

    year_start = datetime(2027, 1, 1, tzinfo=timezone.utc)
    fire_times = sorted(t for d in range(365) for t in _fire_times_on(spec.cron, year_start + timedelta(days=d)))
    assert [(t.month, t.day, t.hour) for t in fire_times] == [(2, 15, 8), (5, 15, 8), (8, 15, 8), (11, 15, 8)]

    for fired in fire_times:
        _, year, quarter = _resolve_current_quarter(fired.date())
        deadline = _quarter_end_date(year, quarter) + timedelta(days=45)
        assert (deadline.year, deadline.month) == (fired.year, fired.month)
