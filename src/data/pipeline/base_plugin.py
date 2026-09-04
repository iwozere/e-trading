"""
Plugin specification for the unified data acquisition pipeline.

A "plugin" here is deliberately data, not a class hierarchy: a ``PluginSpec``
describing one schedulable data-download/ingest job. This mirrors the
``_JOB_SPECS`` list + idempotent-upsert pattern already used by
``src/ml/pipeline/p20_kestrel/jobs/register_jobs.py``,
``p21_momentum/jobs/register_jobs.py`` and ``p22_biotech_ma/jobs/register_jobs.py``,
generalized into one project-wide registry (see ``registry.py``).

Every plugin points at an existing (or new, thin) ``run_*.py`` script rather
than reimplementing fetch logic here — see ``docs/Design.md`` for why the
runner shells out to these scripts instead of calling plugin code in-process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Scheduler owner user_id for system-registered jobs. Matches the P20/P22
# convention (bin/scheduler/insert_*.sql and their old register_jobs.py both
# hardcode/default to 2). P21's old register_jobs.py defaulted to 1, but reads
# the same env var — in any environment where SCHEDULER_SYSTEM_USER_ID is
# actually set (i.e. everywhere this matters), both resolved to the same
# value. Shared here so register_jobs.py and dependency_status.py can never
# drift apart on which user_id owns these rows.
SYSTEM_USER_ID = int(os.getenv("SCHEDULER_SYSTEM_USER_ID", "2"))

# Allowed script directories for plugin scripts — mirrors
# ``SchedulerService._execute_data_processing_job``'s ``_ALLOWED_SCRIPT_DIRS``
# (src/scheduler/scheduler_service.py). Kept as a separate copy here rather
# than importing from the scheduler module: the scheduler's copy validates
# script_path pulled from a live DB row at dispatch time and must not gain an
# import-time dependency on this new module; this copy validates PluginSpecs
# at registration/dry-run time, before anything reaches the database. Keep
# the two lists in sync when onboarding a new pipeline directory.
ALLOWED_SCRIPT_DIRS: List[str] = [
    "src/data/",
    "src/ml/pipeline/",
    "src/scheduler/scripts/",
    "src/screeners/",
    "src/strategy_pack/",
]


class PluginValidationError(ValueError):
    """Raised when a `PluginSpec` fails validation (bad path, escapes allowlist, etc.)."""


@dataclass(frozen=True)
class PluginSpec:
    """
    One registered data-download/ingest job.

    Attributes:
        name: Unique job name — matches the ``name`` column in ``job_schedules``
            (unique per ``user_id``). Keep stable across renames of the
            underlying script; the scheduler keys on this, not on script_path.
        category: Grouping tag for ``runner.py --scope <category>`` (e.g.
            ``"p22"``, ``"p20"``, ``"provider"``, ``"edgar"``). Purely
            organizational — not persisted to ``job_schedules``.
        cron: 5- or 6-field cron expression, UTC, as consumed by
            ``APScheduler``'s ``CronTrigger.from_crontab``.
        script_path: Path to the job's script, relative to the project root,
            e.g. ``"src/ml/pipeline/p22_biotech_ma/jobs/run_sec_ingest.py"``.
            Must resolve inside `PROJECT_ROOT` and under one of
            `ALLOWED_SCRIPT_DIRS` — the same constraints the scheduler itself
            enforces at dispatch time (see module docstring).
        script_args: Extra CLI args passed to the script, in order.
        timeout_seconds: Subprocess timeout. Defaults to the scheduler's own
            default (600s) when not given.
        enabled: Whether this job should be active in `job_schedules`.
        description: One-line human-readable summary (logging/diffing only).
        extra_task_params: Additional `task_params` entries beyond
            `script_path`/`script_args`/`timeout_seconds` (rare — most jobs
            don't need this).
        depends_on: Names of other `PluginSpec`s (in this registry) whose
            data this plugin's script reads. Purely declarative here — cron
            timing is NOT derived from it (each plugin keeps its own fixed
            cron; this is not a DAG scheduler). It does two things: (1)
            documents the dependency graph in one place instead of only in
            prose comments, checked by `registry.py` to catch typos/renames;
            (2) is what a consumer script's own completion gate
            (`dependency_status.require_dependencies_or_defer`) reads at
            runtime to decide whether to actually proceed or defer — see
            that module's docstring for what the gate does and does not
            catch (schedule-run success, not data completeness).
    """

    name: str
    category: str
    cron: str
    script_path: str
    script_args: List[str] = field(default_factory=list)
    timeout_seconds: int = 600
    enabled: bool = True
    description: str = ""
    extra_task_params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)

    @property
    def module_target(self) -> str:
        """Dotted module path derived from `script_path` (for `Schedule.target`)."""
        rel = self.script_path.replace("\\", "/")
        if rel.endswith(".py"):
            rel = rel[: -len(".py")]
        return rel.replace("/", ".")

    @property
    def task_params(self) -> Dict[str, Any]:
        """`task_params` payload as stored on the `Schedule` row."""
        params: Dict[str, Any] = {
            "script_path": self.script_path,
            "script_args": list(self.script_args),
            "timeout_seconds": self.timeout_seconds,
        }
        params.update(self.extra_task_params)
        return params

    def resolved_script_path(self, project_root: Optional[Path] = None) -> Path:
        """Absolute path to `script_path`, resolved against `project_root`."""
        root = project_root or PROJECT_ROOT
        return (root / self.script_path).resolve()

    def validate(self, project_root: Optional[Path] = None) -> None:
        """
        Validate this spec's `script_path` the way the scheduler would at
        dispatch time: must resolve inside the project root, under an
        allowed directory, and exist on disk.

        Raises:
            PluginValidationError: On any violation, with a message naming
                the offending spec.
        """
        root = (project_root or PROJECT_ROOT).resolve()
        script_full_path = self.resolved_script_path(root)

        try:
            script_full_path.relative_to(root)
        except ValueError:
            raise PluginValidationError(
                f"Plugin '{self.name}': script_path '{self.script_path}' resolves outside "
                f"the project root ({root})."
            ) from None

        script_rel = str(script_full_path.relative_to(root)).replace("\\", "/")
        if not any(script_rel.startswith(d) for d in ALLOWED_SCRIPT_DIRS):
            raise PluginValidationError(
                f"Plugin '{self.name}': script_path '{self.script_path}' is not under an "
                f"allowed directory. Allowed prefixes: {ALLOWED_SCRIPT_DIRS}"
            )

        if not script_full_path.exists():
            raise PluginValidationError(
                f"Plugin '{self.name}': script_path '{self.script_path}' does not exist "
                f"at {script_full_path}."
            )
