"""
Seed the `job_schedules` row that drives the daily portfolio PnL alert.

Idempotent: if a schedule with the same `(user_id, name)` already exists, its
cron / target / task_params are updated to match the current YAML config.

Usage::

    python -m src.portfolio.pnl_alert.seed_schedule \
        [--config src/portfolio/pnl_alert/config/pnl_alert.yaml] \
        [--user-id 1] \
        [--name portfolio_pnl_alert] \
        [--disabled]
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.models.model_jobs import JobType, ScheduleCreate, ScheduleUpdate  # noqa: E402
from src.data.db.services.jobs_service import JobsService  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402
from src.portfolio.pnl_alert.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402

_logger = setup_logger(__name__)

DEFAULT_SCHEDULE_NAME = "portfolio_pnl_alert"
DEFAULT_TARGET = "portfolio.pnl_alert"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for seeding."""
    parser = argparse.ArgumentParser(
        prog="python -m src.portfolio.pnl_alert.seed_schedule",
        description="Insert or update the job_schedules row for the daily PnL alert.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to pipeline YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="Owner user_id for the schedule (default: 1)",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_SCHEDULE_NAME,
        help=f"Schedule name (default: {DEFAULT_SCHEDULE_NAME})",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"Dispatch target key (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Seed the schedule in disabled state",
    )
    return parser.parse_args(argv)


def seed(
    config_path: str,
    user_id: int,
    name: str,
    target: str,
    enabled: bool,
    jobs_service: JobsService | None = None,
) -> int:
    """
    Insert or update the schedule row.

    Args:
        config_path: Path to the pipeline YAML (used to read cron + params).
        user_id: Owner user_id for the `job_schedules` row.
        name: Schedule name (unique per user).
        target: Dispatch key consumed by `scheduler_service`.
        enabled: Whether the schedule is enabled.
        jobs_service: Optional pre-built `JobsService` for tests.

    Returns:
        The id of the inserted / updated schedule row.

    Raises:
        FileNotFoundError: If the pipeline config is missing.
        ValueError: If the pipeline config is invalid.
    """
    cfg = load_config(config_path)

    svc = jobs_service or JobsService()

    task_params = {"config_path": config_path}

    existing = svc.list_schedules(user_id=user_id)
    for sch in existing:
        if sch.name == name:
            _logger.info("Updating existing schedule %s (id=%s)", name, sch.id)
            update = ScheduleUpdate(
                target=target,
                task_params=task_params,
                cron=cfg.cron,
                enabled=enabled,
            )
            updated = svc.update_schedule(sch.id, update)
            if updated is None:
                raise RuntimeError(f"Failed to update schedule id={sch.id}")
            return updated.id

    _logger.info("Creating new schedule %s for user %s", name, user_id)
    create = ScheduleCreate(
        name=name,
        job_type=JobType.ALERT,
        target=target,
        task_params=task_params,
        cron=cfg.cron,
        enabled=enabled,
    )
    created = svc.create_schedule(user_id, create)
    return created.id


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    try:
        schedule_id = seed(
            config_path=args.config,
            user_id=args.user_id,
            name=args.name,
            target=args.target,
            enabled=not args.disabled,
        )
    except (FileNotFoundError, ValueError) as exc:
        _logger.error("Seeding failed: %s", exc)
        return 2

    print(f"Schedule ready: id={schedule_id} name={args.name} enabled={not args.disabled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
