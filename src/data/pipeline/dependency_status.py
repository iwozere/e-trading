"""
Runtime completion gate for `PluginSpec.depends_on`.

Cron staggering (e.g. "run 30 minutes after X") is a *hope*, not a
guarantee — it has already failed once in production: P22's
`ClinicalTrials Ingest` timed out at 7200s having covered only 215/1705
companies, yet `Alias Matching` was still going to fire 60 minutes after
ingest *started*, reading whatever partial data had landed by then with no
signal that it was incomplete (see `specs/p22_specs.py`).

This module gives a consumer script a way to check, at the top of its own
`run()`, whether each of its `depends_on` plugins actually completed
successfully *today* (UTC) before doing real work — by querying the
scheduler's own `job_schedules` / `job_schedule_runs` tables, the same
source of truth `SchedulerService` itself uses.

**What this catches**: a dependency that never ran today, is still running,
crashed, or was killed for timing out (any `ScheduleRun.status` other than
`COMPLETED`).

**What this does NOT catch**: a dependency that exited 0 having only
partially finished its work — exactly the ClinicalTrials Ingest case above.
That requires a data-completeness check specific to what "done" means for
that data source (e.g. companies-landed vs. universe size), which is a
separate, per-pipeline enhancement, not something this generic gate can do.

Consumers decide what to do with a not-ready dependency — this module only
reports status, it does not raise or exit. The established idiom in this
codebase (`p19_penny_intraday/label_backfill.py`, `p21_momentum/calendar.py`)
is to safely no-op rather than run on data that isn't ready yet;
`require_dependencies_or_defer` returns that decision as a bool so a caller
can follow the same pattern:

    from src.data.pipeline.dependency_status import require_dependencies_or_defer

    def run() -> dict:
        ready, statuses = require_dependencies_or_defer("P22 Alias Matching")
        if not ready:
            return {"deferred": True, "dependency_status": [s.__dict__ for s in statuses]}
        ... normal work ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.data.db.core.database import session_scope
from src.data.db.models.model_jobs import RunStatus, Schedule, ScheduleRun
from src.data.pipeline.base_plugin import SYSTEM_USER_ID
from src.data.pipeline.registry import get_by_name
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class DependencyStatus:
    """Today's (UTC) status of one `depends_on` plugin."""

    name: str
    registered: bool  # False if `name` isn't a known PluginSpec (config error)
    ran_today: bool
    succeeded: bool
    status: Optional[str]  # raw RunStatus value of the most recent run today, if any
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


def check_dependency(name: str, *, as_of: Optional[date] = None) -> DependencyStatus:
    """
    Look up `name`'s most recent `job_schedule_runs` row scheduled for `as_of`
    (default: today, UTC).

    Args:
        name: A `PluginSpec.name` — typically one entry from another spec's
            `depends_on`.
        as_of: Date to check "today" against. Defaults to `datetime.now(UTC).date()`.

    Returns:
        `DependencyStatus`. `registered=False` means `name` isn't in
        `PLUGIN_REGISTRY` at all (a config error — `registry.py`'s
        `_check_dependencies_exist` should have already caught this at import
        time for anything actually wired into a spec's `depends_on`, but this
        function is also usable standalone).
    """
    today = as_of or datetime.now(timezone.utc).date()

    if get_by_name(name) is None:
        return DependencyStatus(name, registered=False, ran_today=False, succeeded=False,
                                 status=None, started_at=None, finished_at=None)

    with session_scope() as s:
        schedule = s.query(Schedule).filter_by(user_id=SYSTEM_USER_ID, name=name).first()
        if schedule is None:
            return DependencyStatus(name, registered=True, ran_today=False, succeeded=False,
                                     status=None, started_at=None, finished_at=None)

        run = (
            s.query(ScheduleRun)
            .filter(ScheduleRun.job_id == str(schedule.id))
            .filter(ScheduleRun.scheduled_for >= datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc))
            .filter(ScheduleRun.scheduled_for < datetime.combine(today, datetime.max.time(), tzinfo=timezone.utc))
            .order_by(ScheduleRun.scheduled_for.desc())
            .first()
        )
        if run is None:
            return DependencyStatus(name, registered=True, ran_today=False, succeeded=False,
                                     status=None, started_at=None, finished_at=None)

        return DependencyStatus(
            name,
            registered=True,
            ran_today=True,
            succeeded=run.status == RunStatus.COMPLETED.value,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )


def check_dependencies(names: List[str], *, as_of: Optional[date] = None) -> List[DependencyStatus]:
    """`check_dependency` for each name, in order."""
    return [check_dependency(name, as_of=as_of) for name in names]


def require_dependencies_or_defer(plugin_name: str, *, as_of: Optional[date] = None) -> Tuple[bool, List[DependencyStatus]]:
    """
    Check whether every dependency of the registered plugin `plugin_name`
    completed successfully today.

    Looks up `plugin_name`'s own `depends_on` list from the registry — the
    caller only needs to know its own name, not repeat the dependency list.

    Args:
        plugin_name: This script's own `PluginSpec.name`.
        as_of: Passed through to `check_dependency`.

    Returns:
        `(ready, statuses)`. `ready` is True only if every dependency
        `succeeded` today. `statuses` is always the full list, including
        successful ones, for logging. A plugin with an empty `depends_on`
        (or one not found in the registry — logs a warning) is always ready.
    """
    spec = get_by_name(plugin_name)
    if spec is None:
        _logger.warning(
            "require_dependencies_or_defer: %r is not a registered PluginSpec — "
            "treating as no dependencies. Pass the exact name from registry.py.",
            plugin_name,
        )
        return True, []

    if not spec.depends_on:
        return True, []

    statuses = check_dependencies(spec.depends_on, as_of=as_of)
    not_ready = [st for st in statuses if not st.succeeded]

    if not_ready:
        for st in not_ready:
            if not st.registered:
                _logger.warning("%s: dependency %r is not a registered plugin", plugin_name, st.name)
            elif not st.ran_today:
                _logger.warning("%s: dependency %r has not run yet today — deferring", plugin_name, st.name)
            else:
                _logger.warning(
                    "%s: dependency %r's last run today did not complete (status=%s) — deferring",
                    plugin_name, st.name, st.status,
                )
        return False, statuses

    return True, statuses


def deferred_result(statuses: List[DependencyStatus]) -> Dict[str, Any]:
    """
    Build a `__SCHEDULER_RESULT__`-safe dict for a script that deferred after
    `require_dependencies_or_defer` returned `ready=False`.

    `"success": True` is deliberate: deferring because a dependency isn't
    ready yet is a correct no-op, not a failure — same as the existing
    self-gating idiom elsewhere in this codebase (e.g.
    `LabelBackfill.run()` returning zero counts on a date that isn't old
    enough yet). Scripts should still exit 0.
    """
    return {
        "success": True,
        "deferred": True,
        "dependency_status": [
            {"name": st.name, "registered": st.registered, "ran_today": st.ran_today,
             "succeeded": st.succeeded, "status": st.status}
            for st in statuses
        ],
    }
