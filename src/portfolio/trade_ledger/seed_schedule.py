"""
Seed the `job_schedules` row that drives the monthly trade-ledger report.

Idempotent: a schedule with the same `(user_id, name)` is updated to the
current YAML cron / target / task_params instead of duplicated. Portfolio
jobs are dispatched by `target` ("portfolio.trade_ledger") under job type
ALERT, the same as `pnl_alert`.

Usage::

    python -m src.portfolio.trade_ledger.seed_schedule [--config PATH] [--user-id 1] [--name portfolio_trade_ledger] [--disabled]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.models.model_jobs import JobType, ScheduleCreate, ScheduleUpdate  # noqa: E402
from src.data.db.services.jobs_service import JobsService  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402
from src.portfolio.trade_ledger.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

_logger = setup_logger(__name__)

DEFAULT_SCHEDULE_NAME = "portfolio_trade_ledger"
DEFAULT_TARGET = "portfolio.trade_ledger"


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.portfolio.trade_ledger.seed_schedule",
        description="Insert or update the job_schedules row for the monthly trade-ledger report.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=f"Config YAML (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--user-id", type=int, default=1, help="Owner user_id of the schedule row (default: 1)")
    parser.add_argument("--name", default=DEFAULT_SCHEDULE_NAME, help=f"Schedule name (default: {DEFAULT_SCHEDULE_NAME})")
    parser.add_argument("--disabled", action="store_true", help="Seed the schedule disabled")
    return parser.parse_args(argv)


def seed(config_path: str, user_id: int, name: str, enabled: bool, jobs_service: Optional[JobsService] = None) -> int:
    """
    Insert or update the schedule row.

    Args:
        config_path: Config YAML (cron is read from it; the path is stored in task_params).
        user_id: Owner user_id of the `job_schedules` row.
        name: Schedule name (unique per user).
        enabled: Whether the schedule is enabled.
        jobs_service: Optional pre-built `JobsService` (tests).

    Returns:
        Id of the inserted / updated schedule.

    Raises:
        FileNotFoundError: If the config is missing.
        ValueError: If the config is invalid.
        RuntimeError: If an existing row could not be updated.
    """
    cfg = load_config(config_path)
    svc = jobs_service or JobsService()
    task_params = {"config_path": config_path}

    for sch in svc.list_schedules(user_id=user_id):
        if sch.name == name:
            _logger.info("Updating existing schedule %s (id=%s)", name, sch.id)
            updated = svc.update_schedule(sch.id, ScheduleUpdate(target=DEFAULT_TARGET, task_params=task_params, cron=cfg.cron, enabled=enabled))
            if updated is None:
                raise RuntimeError(f"Failed to update schedule id={sch.id}")
            return updated.id

    _logger.info("Creating schedule %s for user %s", name, user_id)
    created = svc.create_schedule(
        user_id,
        ScheduleCreate(name=name, job_type=JobType.ALERT, target=DEFAULT_TARGET, task_params=task_params, cron=cfg.cron, enabled=enabled),
    )
    return created.id


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Arguments (defaults to sys.argv).

    Returns:
        Process exit code.
    """
    args = _parse_args(argv)
    try:
        schedule_id = seed(args.config, args.user_id, args.name, enabled=not args.disabled)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _logger.error("Seeding failed: %s", exc)
        return 2
    _logger.info("Schedule ready: id=%s name=%s enabled=%s", schedule_id, args.name, not args.disabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
