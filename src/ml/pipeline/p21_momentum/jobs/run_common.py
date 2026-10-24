"""
P21 Momentum — Shared bootstrap for all job scripts.

Call ``setup_run_logging()`` once at the top of every ``main()`` to create
``results/p21_momentum/YYYY-MM-DD/`` and attach a rotating file handler,
matching the pattern used by P20 Kestrel's ``jobs/run_common.py``.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.ml.pipeline.p21_momentum.config import REGIME_HISTORY_PATH, RESULTS_DIR
from src.notification.logger import SensitiveDataFilter

if TYPE_CHECKING:
    import pandas as pd


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
    handler.addFilter(SensitiveDataFilter())

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


def replay_live_tracks(
    today: date,
    results_dir: Path = RESULTS_DIR,
    regime_history_path: Path = REGIME_HISTORY_PATH,
) -> Optional["pd.DataFrame"]:
    """
    Replay Tracks B-E from inception up to today (strategy/live_tracks.py).

    Fail-soft: a price fetch below the coverage gate (PipelineAbort) or no targets
    yet returns None, so Track A's own daily work is never blocked by B-E.

    Args:
        today: Last session to include.
        results_dir: ``results/p21_momentum``.
        regime_history_path: ``_state/regime_history.json``.

    Returns:
        Frame indexed by session date with nav_b..nav_e, or None.
    """
    from src.ml.pipeline.p21_momentum.data.prices import fetch_price_panel
    from src.ml.pipeline.p21_momentum.quality.gates import PipelineAbort
    from src.ml.pipeline.p21_momentum.results.run_io import read_regime_history
    from src.ml.pipeline.p21_momentum.strategy.live_tracks import load_executions, replay_tracks, track_tickers, wide_frames

    executions = [e for e in load_executions(Path(str(results_dir)), read_regime_history(regime_history_path)) if e.execution_date <= today]
    if not executions:
        return None
    start = datetime.combine(executions[0].execution_date, datetime.min.time()) - timedelta(days=10)
    end = datetime.combine(today, datetime.min.time()) + timedelta(days=1)
    try:
        panel = fetch_price_panel(track_tickers(executions), start, end)
    except PipelineAbort as exc:
        logging.getLogger(__name__).warning("Tracks B-E not updated: %s", exc)
        return None
    opens, closes = wide_frames(panel)
    return replay_tracks(executions, opens, closes)
