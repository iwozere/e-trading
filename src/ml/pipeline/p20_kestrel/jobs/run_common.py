"""
Shared bootstrap for all P20 Kestrel run scripts.

Call ``setup_run_logging()`` once at the top of every ``main()`` to:
  - create results/p20_kestrel/YYYY-MM-DD/
  - attach a RotatingFileHandler to the src.ml.pipeline logger so all
    P20 modules write to that day's pipeline.log automatically
  - return the date-stamped results directory for any per-job file output
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p20_kestrel.config import RESULTS_DIR


def setup_run_logging(run_date: date | None = None) -> Path:
    """
    Create the dated results directory and attach a pipeline.log file handler.

    Args:
        run_date: Date for the results folder (defaults to today).

    Returns:
        Path to results/p20_kestrel/YYYY-MM-DD/ (already created).
    """
    today = run_date or date.today()
    results_dir = Path(str(RESULTS_DIR)) / today.strftime("%Y-%m-%d")
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = results_dir / "pipeline.log"
    handler = RotatingFileHandler(
        str(log_file),
        maxBytes=100 * 1024 * 1024,  # 100 MB
        backupCount=3,
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - [PID %(process)d] - %(levelname)-8s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    pipeline_logger = logging.getLogger("src.ml.pipeline")
    pipeline_logger.addHandler(handler)
    pipeline_logger.setLevel(logging.DEBUG)

    return results_dir
