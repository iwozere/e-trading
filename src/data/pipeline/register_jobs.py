"""
Generate `job_schedules` rows from `registry.PLUGIN_REGISTRY`.

Supersedes the per-pipeline `jobs/register_jobs.py` modules (P20/P21/P22) and,
once every group in the registry is ported (see `registry.py`'s migration
status note), the hand-written `bin/scheduler/insert_*.sql` seed files —
closing the two-competing-registration-path drift documented in P20's old
`register_jobs.py` (a 2026-08-30 (user_id, name) mismatch silently created a
duplicate set of 21 jobs in production).

Usage:
    # Show what would change against the live DB without writing anything —
    # always run this first against a table you haven't already verified.
    python -m src.data.pipeline.register_jobs --dry-run

    # Apply.
    python -m src.data.pipeline.register_jobs

    # Restrict to one category (useful when porting a new group incrementally).
    python -m src.data.pipeline.register_jobs --dry-run --category p20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.data.db.core.database import session_scope
from src.data.db.models.model_jobs import Schedule
from src.data.pipeline.base_plugin import SYSTEM_USER_ID, PluginSpec, PluginValidationError
from src.data.pipeline.registry import PLUGIN_REGISTRY
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass(frozen=True)
class _RowDiff:
    """One planned change for a single `PluginSpec`."""

    name: str
    action: str  # "insert" | "update" | "unchanged"
    changes: Dict[str, Any]  # field -> (old, new), only for "update"


# task_params keys this registry owns and will set on insert/update. Any
# OTHER key already present on a row (e.g. `notification_rules`, set by an
# admin or a hand-written SQL seed script outside this registry) is preserved
# on update, never overwritten — see `apply()`. Discovered necessary via
# `--dry-run` against a DB seeded from bin/scheduler/insert_p20_schedules.sql:
# several P20 rows carry Telegram `notification_rules` that P20's old
# jobs/register_jobs.py `_JOB_SPECS` never knew about and would have silently
# dropped on the first real apply.
_MANAGED_TASK_PARAM_KEYS = ("script_path", "script_args", "timeout_seconds")


def _desired_scalar_fields(spec: PluginSpec) -> Dict[str, Any]:
    """Non-`task_params` `Schedule` columns this registry fully owns."""
    return {
        "job_type": "data_processing",
        "target": spec.module_target,
        "cron": spec.cron,
        "enabled": spec.enabled,
    }


def _managed_task_params(spec: PluginSpec) -> Dict[str, Any]:
    """
    The `task_params` subset this registry owns: `_MANAGED_TASK_PARAM_KEYS`
    plus whatever a spec opts into via `extra_task_params` (e.g. a future
    spec that wants to own its own `notification_rules`).
    """
    params = dict(spec.task_params)
    assert set(_MANAGED_TASK_PARAM_KEYS).issubset(params), "PluginSpec.task_params is missing a base managed key"
    return params


def _diff_existing(spec: PluginSpec, existing: Optional[Schedule]) -> _RowDiff:
    scalar_desired = _desired_scalar_fields(spec)
    managed_params = _managed_task_params(spec)

    if existing is None:
        return _RowDiff(name=spec.name, action="insert", changes={**scalar_desired, "task_params": managed_params})

    changes: Dict[str, Any] = {}
    for field, new_value in scalar_desired.items():
        old_value = getattr(existing, field)
        if old_value != new_value:
            changes[field] = (old_value, new_value)

    existing_params: Dict[str, Any] = existing.task_params or {}
    param_changes = {
        key: (existing_params.get(key), new_value)
        for key, new_value in managed_params.items()
        if existing_params.get(key) != new_value
    }
    if param_changes:
        changes["task_params"] = param_changes

    return _RowDiff(name=spec.name, action="update" if changes else "unchanged", changes=changes)


def plan(category: Optional[str] = None) -> List[_RowDiff]:
    """
    Validate every in-scope `PluginSpec` and compute its diff against the
    live `job_schedules` table, without writing anything.

    Raises:
        PluginValidationError: If any spec's `script_path` is invalid — this
            runs (and fails loudly) before any DB row is touched.
    """
    specs = [s for s in PLUGIN_REGISTRY if category is None or s.category == category]
    for spec in specs:
        spec.validate()

    diffs: List[_RowDiff] = []
    with session_scope() as s:
        for spec in specs:
            existing = s.query(Schedule).filter_by(user_id=SYSTEM_USER_ID, name=spec.name).first()
            diffs.append(_diff_existing(spec, existing))
    return diffs


def apply(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Idempotently upsert every in-scope `PluginSpec` into `job_schedules`.

    Returns:
        Summary dict: counts of inserted/updated/unchanged rows.
    """
    specs = [s for s in PLUGIN_REGISTRY if category is None or s.category == category]
    for spec in specs:
        spec.validate()

    counts = {"inserted": 0, "updated": 0, "unchanged": 0}
    with session_scope() as s:
        for spec in specs:
            scalar_desired = _desired_scalar_fields(spec)
            managed_params = _managed_task_params(spec)
            existing = s.query(Schedule).filter_by(user_id=SYSTEM_USER_ID, name=spec.name).first()
            if existing is None:
                s.add(
                    Schedule(
                        user_id=SYSTEM_USER_ID, name=spec.name, state_json={},
                        task_params=managed_params, **scalar_desired,
                    )
                )
                counts["inserted"] += 1
                _logger.info("Inserted schedule: %s (%s)", spec.name, spec.cron)
            else:
                diff = _diff_existing(spec, existing)
                if diff.action == "unchanged":
                    counts["unchanged"] += 1
                    continue
                for field, new_value in scalar_desired.items():
                    setattr(existing, field, new_value)
                # Merge, don't replace: preserve any task_params key this registry
                # doesn't manage (e.g. notification_rules) — see _MANAGED_TASK_PARAM_KEYS.
                existing.task_params = {**(existing.task_params or {}), **managed_params}
                counts["updated"] += 1
                _logger.info("Updated schedule: %s (%s) — changed fields: %s", spec.name, spec.cron, list(diff.changes))

    _logger.info("Job registration complete: %s", counts)
    return counts


def _print_plan(diffs: List[_RowDiff]) -> None:
    inserts = [d for d in diffs if d.action == "insert"]
    updates = [d for d in diffs if d.action == "update"]
    unchanged = [d for d in diffs if d.action == "unchanged"]

    print(f"\n{len(diffs)} plugin(s) in scope: {len(inserts)} to insert, {len(updates)} to update, "
          f"{len(unchanged)} unchanged.\n")

    for d in inserts:
        print(f"  [INSERT] {d.name}")
    for d in updates:
        print(f"  [UPDATE] {d.name}")
        for field, value in d.changes.items():
            if field == "task_params":
                for param_key, (old, new) in value.items():
                    print(f"      task_params.{param_key}: {old!r} -> {new!r}")
            else:
                old, new = value
                print(f"      {field}: {old!r} -> {new!r}")
    if not inserts and not updates:
        print("  (nothing to change)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan against the live DB; write nothing.")
    parser.add_argument("--category", default=None, help="Restrict to one category (e.g. 'p20').")
    args = parser.parse_args()

    try:
        if args.dry_run:
            diffs = plan(category=args.category)
            _print_plan(diffs)
            result = {
                "dry_run": True,
                "inserts": sum(1 for d in diffs if d.action == "insert"),
                "updates": sum(1 for d in diffs if d.action == "update"),
                "unchanged": sum(1 for d in diffs if d.action == "unchanged"),
            }
        else:
            counts = apply(category=args.category)
            result = {"dry_run": False, **counts}
    except PluginValidationError:
        _logger.exception("Plugin validation failed — no DB rows were touched")
        raise

    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")


if __name__ == "__main__":
    main()
