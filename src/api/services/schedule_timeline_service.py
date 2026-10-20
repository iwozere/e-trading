"""
Schedule timeline service.

Builds a day-of-week × time-of-day view of scheduler jobs so overlapping
("colliding") jobs are easy to spot in the Web UI.

Two views are produced:

* ``planned`` — every enabled schedule's cron expanded over the next 7 days,
  each fire drawn with the job's typical (median) run duration.
* ``actual`` — the runs that really started in the last N days.

All grouping (weekday, minute of day) happens in the viewer's IANA time zone,
while collisions are detected on absolute timestamps, so jobs that run across
midnight are still caught.
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from croniter import croniter

from src.data.db.models.model_jobs import Schedule
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# A job firing more than this many times on one day (e.g. every 5 min / hourly)
# is flagged "frequent": it is still drawn, but excluded from collision pairs,
# where it would overlap with almost everything and drown the real conflicts.
FREQUENT_FIRES_PER_DAY = 6

# Shortest interval treated as "busy" for collision detection, so instant or
# unknown-duration jobs that start in the same minute still count as colliding.
MIN_SLOT_SECONDS = 60

# Upper bound on fires expanded per schedule and week (protects against "* * * * *").
MAX_FIRES_PER_WEEK = 7 * 24 * 60

# Number of recent finished runs used for the median duration.
DURATION_SAMPLE_SIZE = 10


@dataclass
class _Slot:
    """One (planned or actual) execution of a schedule on an absolute time interval."""

    schedule_id: int
    name: str
    start: datetime
    duration_s: Optional[float]
    status: Optional[str] = None
    run_id: Optional[int] = None

    @property
    def busy_end(self) -> datetime:
        """End of the interval used for collision detection (at least MIN_SLOT_SECONDS long)."""
        return self.start + timedelta(seconds=max(self.duration_s or 0.0, MIN_SLOT_SECONDS))


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    """Treat naive DB timestamps (e.g. SQLite) as UTC so they compare with aware ones."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass(frozen=True)
class RunRow:
    """The few columns of a job_schedule_runs row the timeline needs."""

    id: int
    schedule_id: int
    status: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


def group_runs_by_schedule(
    rows: Iterable[Tuple[int, Union[int, str, None], Optional[str], Optional[datetime], Optional[datetime]]],
    schedule_ids: Iterable[int],
) -> Dict[int, List[RunRow]]:
    """
    Group raw run rows by the schedule that produced them, newest start first.

    The scheduler (scheduled and manual runs alike) stores ``job_id=str(schedule.id)``;
    rows whose job_id is not a known schedule id are ignored.

    Args:
        rows: (id, job_id, status, started_at, finished_at) tuples.
        schedule_ids: Ids of the known schedules.

    Returns:
        Mapping schedule id → its runs, newest first.
    """
    grouped: Dict[int, List[RunRow]] = {sid: [] for sid in schedule_ids}
    for run_id, job_id, status, started_at, finished_at in rows:
        # The model declares job_id as String, but the prod column holds integers; accept both.
        key = str(job_id) if job_id is not None else ""
        if not key.isdigit() or int(key) not in grouped:
            continue
        grouped[int(key)].append(RunRow(run_id, int(key), status, _as_utc(started_at), _as_utc(finished_at)))
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    for runs in grouped.values():
        runs.sort(key=lambda r: r.started_at or epoch, reverse=True)
    return grouped


def run_duration_s(run: RunRow, now: datetime) -> Optional[float]:
    """
    Return a run's duration in seconds; a still-running run counts up to `now`.

    Args:
        run: The schedule run.
        now: Current time (timezone-aware).

    Returns:
        Duration in seconds, or None when the run never started.
    """
    if run.started_at is None:
        return None
    end = run.finished_at or (now if run.status == "running" else None)
    if end is None:
        return None
    return max((end - run.started_at).total_seconds(), 0.0)


def typical_duration_s(runs: Sequence[RunRow], now: datetime) -> Optional[float]:
    """
    Return the median duration of the most recent finished runs.

    Args:
        runs: Runs of one schedule, newest first.
        now: Current time (timezone-aware).

    Returns:
        Median duration in seconds, or None when no finished run is known.
    """
    durations = [d for r in runs if r.finished_at is not None and (d := run_duration_s(r, now)) is not None]
    if not durations:
        return None
    return float(statistics.median(durations[:DURATION_SAMPLE_SIZE]))


def expand_cron(cron: str, window_start: datetime, window_end: datetime) -> List[datetime]:
    """
    List the UTC fire times of a crontab expression inside [window_start, window_end).

    Uses croniter (crontab semantics: weekday 0/7 = Sunday), matching how the
    scheduler computes next_run_at. Six-field expressions carry seconds first,
    as the scheduler accepts them.

    Args:
        cron: 5- or 6-field cron expression, evaluated in UTC.
        window_start: Inclusive window start (timezone-aware).
        window_end: Exclusive window end (timezone-aware).

    Returns:
        Fire times as UTC datetimes; empty if the expression is invalid.
    """
    fields = cron.split()
    if len(fields) == 6:
        # Scheduler's 6-field form is "second minute hour day month weekday"; croniter wants seconds last.
        fields = fields[1:] + fields[:1]
    expr = " ".join(fields)
    try:
        itr = croniter(expr, window_start.astimezone(timezone.utc) - timedelta(seconds=1))
    except (ValueError, KeyError):
        _logger.warning("Skipping schedule with invalid cron expression: %s", cron)
        return []

    fires: List[datetime] = []
    while len(fires) < MAX_FIRES_PER_WEEK:
        fire = itr.get_next(datetime)
        if fire >= window_end:
            break
        fires.append(fire)
    return fires


def _find_collisions(slots: List[_Slot], same_day_only: bool, tz: tzinfo) -> List[Dict[str, Any]]:
    """Return every pair of slots of different schedules whose busy intervals overlap."""
    ordered = sorted(slots, key=lambda s: s.start)
    collisions: List[Dict[str, Any]] = []
    for i, a in enumerate(ordered):
        a_end = a.busy_end
        for b in ordered[i + 1 :]:
            if b.start >= a_end:
                break
            if a.schedule_id == b.schedule_id:
                continue
            if same_day_only and a.start.astimezone(tz).date() != b.start.astimezone(tz).date():
                continue
            overlap_end = min(a_end, b.busy_end)
            local_b = b.start.astimezone(tz)
            collisions.append(
                {
                    "weekday": local_b.weekday(),
                    "date": local_b.date().isoformat(),
                    "start_minute": local_b.hour * 60 + local_b.minute,
                    "overlap_s": round((overlap_end - b.start).total_seconds(), 1),
                    "a": {"schedule_id": a.schedule_id, "name": a.name, "start": a.start.isoformat()},
                    "b": {"schedule_id": b.schedule_id, "name": b.name, "start": b.start.isoformat()},
                }
            )
    return collisions


def _slot_to_dict(slot: _Slot, tz: tzinfo, duration_source: Optional[str] = None) -> Dict[str, Any]:
    local = slot.start.astimezone(tz)
    out: Dict[str, Any] = {
        "weekday": local.weekday(),
        "date": local.date().isoformat(),
        "start": slot.start.isoformat(),
        "start_minute": local.hour * 60 + local.minute + local.second / 60.0,
        "duration_s": round(slot.duration_s, 1) if slot.duration_s is not None else None,
    }
    if duration_source is not None:
        out["duration_source"] = duration_source
    if slot.run_id is not None:
        out["run_id"] = slot.run_id
        out["status"] = slot.status
    return out


def build_planned_timeline(
    schedules: Sequence[Schedule],
    runs_by_schedule: Dict[int, List[RunRow]],
    now: datetime,
    tz: tzinfo,
) -> Dict[str, Any]:
    """
    Expand every enabled schedule over the next 7 local days.

    Args:
        schedules: All schedules (disabled ones are skipped).
        runs_by_schedule: Recent runs per schedule id, newest first (for median duration).
        now: Current time (timezone-aware).
        tz: Viewer time zone used for weekday / minute-of-day grouping.

    Returns:
        {"window_start", "window_end", "jobs": [...], "collisions": [...]}.
    """
    local_midnight = now.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = local_midnight.astimezone(timezone.utc)
    window_end = (local_midnight + timedelta(days=7)).astimezone(timezone.utc)

    jobs: List[Dict[str, Any]] = []
    collision_slots: List[_Slot] = []
    for schedule in schedules:
        if not schedule.enabled:
            continue
        fires = expand_cron(schedule.cron, window_start, window_end)
        if not fires:
            continue
        duration = typical_duration_s(runs_by_schedule.get(schedule.id, []), now)
        source = "median" if duration is not None else "unknown"
        slots = [_Slot(schedule.id, schedule.name, f, duration) for f in fires]

        per_day: Dict[str, int] = {}
        for f in fires:
            key = f.astimezone(tz).date().isoformat()
            per_day[key] = per_day.get(key, 0) + 1
        frequent = max(per_day.values()) > FREQUENT_FIRES_PER_DAY
        if not frequent:
            collision_slots.extend(slots)

        jobs.append(
            {
                "schedule_id": schedule.id,
                "name": schedule.name,
                "cron": schedule.cron,
                "frequent": frequent,
                "typical_duration_s": round(duration, 1) if duration is not None else None,
                "slots": [_slot_to_dict(s, tz, source) for s in slots],
            }
        )

    return {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "jobs": jobs,
        "collisions": _find_collisions(collision_slots, same_day_only=False, tz=tz),
    }


def build_actual_timeline(
    schedules: Sequence[Schedule],
    runs_by_schedule: Dict[int, List[RunRow]],
    now: datetime,
    tz: tzinfo,
    days: int,
) -> Dict[str, Any]:
    """
    Collect the runs that started in the last `days` days.

    Args:
        schedules: All schedules.
        runs_by_schedule: Runs per schedule id.
        now: Current time (timezone-aware).
        tz: Viewer time zone used for weekday / minute-of-day grouping.
        days: Look-back window in days.

    Returns:
        {"window_start", "window_end", "jobs": [...], "collisions": [...]}; collisions
        only pair runs that started on the same local date.
    """
    window_start = now - timedelta(days=days)
    jobs: List[Dict[str, Any]] = []
    all_slots: List[_Slot] = []
    for schedule in schedules:
        slots = [
            _Slot(schedule.id, schedule.name, r.started_at, run_duration_s(r, now), status=r.status, run_id=r.id)
            for r in runs_by_schedule.get(schedule.id, [])
            if r.started_at is not None and r.started_at >= window_start
        ]
        if not slots:
            continue
        per_day: Dict[str, int] = {}
        for s in slots:
            key = s.start.astimezone(tz).date().isoformat()
            per_day[key] = per_day.get(key, 0) + 1
        frequent = max(per_day.values()) > FREQUENT_FIRES_PER_DAY
        if not frequent:
            all_slots.extend(slots)
        jobs.append(
            {
                "schedule_id": schedule.id,
                "name": schedule.name,
                "cron": schedule.cron,
                "frequent": frequent,
                "typical_duration_s": None,
                "slots": [_slot_to_dict(s, tz) for s in sorted(slots, key=lambda s: s.start)],
            }
        )

    return {
        "window_start": window_start.isoformat(),
        "window_end": now.isoformat(),
        "jobs": jobs,
        "collisions": _find_collisions(all_slots, same_day_only=True, tz=tz),
    }
