"""
Shared bootstrap for all P22 Biotech M&A run scripts.

Call ``setup_run_logging()`` once at the top of every ``main()`` to:
  - create results/p22_biotech_ma/YYYY-MM-DD/
  - attach a RotatingFileHandler to the ROOT logger so every module
    (src.ml.pipeline.*, src.data.*, src.common.*, third-party libs)
    writes to that day's pipeline.log automatically
  - tee stdout/stderr so bare print() calls are also captured in the log
  - return the date-stamped results directory for any per-job file output

Mirrors src/ml/pipeline/p20_kestrel/jobs/run_common.py exactly — same
contract, so the existing scheduler subprocess harness needs no changes.
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import date
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import RESULTS_DIR


class _TeeStream(io.TextIOBase):
    """Write to both the original stream and a log file simultaneously."""

    def __init__(self, original: io.TextIOBase, log_path: Path) -> None:
        super().__init__()
        self._original = original
        self._log_fh = log_path.open("a", encoding="utf-8", buffering=1)

    def write(self, s: str) -> int:
        self._original.write(s)
        self._original.flush()
        if s and not s.isspace():
            self._log_fh.write(s)
            self._log_fh.flush()
        return len(s)

    def flush(self) -> None:
        self._original.flush()
        self._log_fh.flush()


def setup_run_logging(run_date: date | None = None) -> Path:
    """
    Create the dated results directory and attach a pipeline.log file handler.

    Args:
        run_date: Date for the results folder (defaults to today).

    Returns:
        Path to results/p22_biotech_ma/YYYY-MM-DD/ (already created).
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

    pipeline_logger = logging.getLogger("src.ml.pipeline")
    pipeline_logger.setLevel(logging.DEBUG)

    assert sys.__stdout__ is not None and sys.__stderr__ is not None
    sys.stdout = _TeeStream(sys.__stdout__, log_file)  # type: ignore[assignment]
    sys.stderr = _TeeStream(sys.__stderr__, log_file)  # type: ignore[assignment]

    logging.getLogger(__name__).info("Run logging active — all output going to %s", log_file)

    return results_dir
