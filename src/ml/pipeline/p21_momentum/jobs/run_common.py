"""
P21 Momentum — Shared bootstrap for all job scripts.

Call ``setup_run_logging()`` once at the top of every ``main()`` to create
``results/p21_momentum/YYYY-MM-DD/`` and attach a rotating file handler,
matching the pattern used by P20 Kestrel's ``jobs/run_common.py``.
"""

from __future__ import annotations

import logging
from datetime import date
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.ml.pipeline.p21_momentum.config import RESULTS_DIR


def setup_run_logging(run_date: date | None = None) -> Path:
    """
    Create the dated results directory and attach a pipeline.log file handler.

    Args:
        run_date: Date for the results folder (defaults to today).

    Returns:
        Path to results/p21_momentum/YYYY-MM-DD/ (already created).
    """
    today = run_date or date.today()
    results_dir = Path(str(RESULTS_DIR)) / today.isoformat()
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = results_dir / "pipeline.log"
    handler = RotatingFileHandler(str(log_file), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - [PID %(process)d] - %(levelname)-8s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    existing_files = {
        getattr(h, "baseFilename", None) for h in root_logger.handlers if isinstance(h, RotatingFileHandler)
    }
    if str(log_file) not in existing_files:
        root_logger.addHandler(handler)

    logging.getLogger("src.ml.pipeline.p21_momentum").setLevel(logging.DEBUG)
    return results_dir


def send_abort_alert(job_name: str, exc: Exception) -> None:
    """
    Fail-soft admin alert for a PipelineAbort. Never raises.

    Uses the database:// service_url, matching P20's notify.py: a scheduled
    backend job has no user JWT, so HTTP mode would 401 on every call.
    """
    import asyncio

    try:
        from src.notification.service.client import NotificationServiceClient

        client = NotificationServiceClient(service_url="database://")
        asyncio.run(
            client.send_to_admins(
                title=f"P21 Momentum: {job_name} ABORTED",
                message=str(exc),
            )
        )
    except Exception:  # pragma: no cover - alerting must never break the caller
        logging.getLogger(__name__).exception("Failed to send ABORT alert for %s", job_name)
