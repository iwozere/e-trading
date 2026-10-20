"""Tests for the schedule timeline (job collision) service."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.api.services.schedule_timeline_service import (
    build_actual_timeline,
    build_planned_timeline,
    expand_cron,
    group_runs_by_schedule,
    typical_duration_s,
)
from src.data.db.models.model_jobs import Schedule

UTC = timezone.utc
ZURICH = ZoneInfo("Europe/Zurich")
# Monday 2026-10-19 12:00 UTC (CEST, UTC+2)
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _schedule(sid: int, name: str, cron: str, enabled: bool = True) -> Schedule:
    return Schedule(id=sid, user_id=1, name=name, job_type="data_processing", target="t", cron=cron, enabled=enabled)


def _run(run_id: int, sid: int, start: datetime, minutes: float, status: str = "completed") -> tuple:
    return (run_id, str(sid), status, start, start + timedelta(minutes=minutes))


def test_expand_cron_uses_crontab_weekdays() -> None:
    start = datetime(2026, 10, 19, tzinfo=UTC)  # Monday
    fires = expand_cron("0 6 * * 1-5", start, start + timedelta(days=7))

    assert [f.weekday() for f in fires] == [0, 1, 2, 3, 4]
    assert all(f.hour == 6 and f.minute == 0 for f in fires)


def test_expand_cron_six_field_and_invalid() -> None:
    start = datetime(2026, 10, 19, tzinfo=UTC)
    fires = expand_cron("30 15 6 * * *", start, start + timedelta(days=1))

    assert fires == [datetime(2026, 10, 19, 6, 15, 30, tzinfo=UTC)]
    assert expand_cron("not a cron", start, start + timedelta(days=1)) == []


def test_group_runs_ignores_unknown_job_ids_and_sorts_newest_first() -> None:
    t0 = datetime(2026, 10, 1, 6, 0)  # naive, as SQLite returns
    rows = [
        _run(1, 28, t0, 10),
        _run(2, 28, t0 + timedelta(days=1), 10),
        (3, "legacy-job", "completed", t0, t0),
        _run(4, 999, t0, 10),
    ]

    grouped = group_runs_by_schedule(rows, [28, 29])

    assert [r.id for r in grouped[28]] == [2, 1]
    assert grouped[28][0].started_at is not None and grouped[28][0].started_at.tzinfo is UTC
    assert grouped[29] == []


def test_typical_duration_is_median_of_finished_runs() -> None:
    t0 = NOW - timedelta(days=5)
    rows = [_run(i, 1, t0 + timedelta(days=i), m) for i, m in enumerate([10, 50, 30])]
    rows.append((9, "1", "running", NOW - timedelta(minutes=5), None))
    grouped = group_runs_by_schedule(rows, [1])

    assert typical_duration_s(grouped[1], NOW) == 30 * 60
    assert typical_duration_s([], NOW) is None


def test_planned_timeline_flags_p15_p17_collision_in_local_time() -> None:
    schedules = [
        _schedule(28, "P15 Options Daily", "0 6 * * 1-5"),
        _schedule(29, "P17 Penny Stock Screener Daily", "0 6 * * 1-5"),
        _schedule(30, "Moved Job", "0 9 * * 1-5"),
        _schedule(31, "Disabled", "0 6 * * 1-5", enabled=False),
    ]
    rows = [_run(1, 28, NOW - timedelta(days=1), 60), _run(2, 29, NOW - timedelta(days=1), 45)]
    grouped = group_runs_by_schedule(rows, [s.id for s in schedules])

    result = build_planned_timeline(schedules, grouped, NOW, ZURICH)

    names = {j["name"] for j in result["jobs"]}
    assert names == {"P15 Options Daily", "P17 Penny Stock Screener Daily", "Moved Job"}
    p15 = next(j for j in result["jobs"] if j["schedule_id"] == 28)
    assert p15["typical_duration_s"] == 3600
    # Week of 2026-10-19 crosses the 25 Oct DST switch; Mon-Fri are all still CEST.
    assert {s["start_minute"] for s in p15["slots"]} == {8 * 60}

    pairs = {(c["a"]["name"], c["b"]["name"]) for c in result["collisions"]}
    assert len(result["collisions"]) == 5  # Mon-Fri
    assert all("Moved Job" not in p for p in pairs)
    assert {c["weekday"] for c in result["collisions"]} == {0, 1, 2, 3, 4}


def test_planned_timeline_excludes_frequent_jobs_from_collisions() -> None:
    schedules = [_schedule(1, "Every 5 min", "*/5 * * * *"), _schedule(2, "Daily", "0 6 * * *")]

    result = build_planned_timeline(schedules, group_runs_by_schedule([], [1, 2]), NOW, UTC)

    frequent = {j["name"]: j["frequent"] for j in result["jobs"]}
    assert frequent == {"Every 5 min": True, "Daily": False}
    assert result["collisions"] == []


def test_actual_timeline_pairs_only_same_day_overlaps() -> None:
    day = datetime(2026, 10, 16, 6, 0, tzinfo=UTC)
    rows = [
        _run(1, 1, day, 60),
        _run(2, 2, day + timedelta(minutes=30), 20),
        _run(3, 2, day + timedelta(days=1, hours=3), 20),
        _run(4, 1, NOW - timedelta(days=30), 60),  # outside the 14-day window
    ]
    schedules = [_schedule(1, "A", "0 6 * * *"), _schedule(2, "B", "30 6 * * *")]

    result = build_actual_timeline(schedules, group_runs_by_schedule(rows, [1, 2]), NOW, ZURICH, days=14)

    a = next(j for j in result["jobs"] if j["name"] == "A")
    assert [s["run_id"] for s in a["slots"]] == [1]
    assert len(result["collisions"]) == 1
    collision = result["collisions"][0]
    assert collision["overlap_s"] == 20 * 60
    assert collision["date"] == "2026-10-16"
