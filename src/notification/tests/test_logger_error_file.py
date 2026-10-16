"""
Tests for the app_errors.log handler in LOG_CONFIG.

The error_file handler used to be configured but attached to no logger: dictConfig opened app_errors.log, the
handler was garbage-collected at once with its file still open (ResourceWarning) and nothing was ever written.
"""

import logging
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import src.notification.logger  # noqa: E402,F401  (applies LOG_CONFIG)


def test_root_logger_has_the_error_file_handler() -> None:
    """The root logger writes ERROR records to app_errors.log."""
    handlers = [
        h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler) and Path(h.baseFilename).name == "app_errors.log"
    ]

    assert len(handlers) == 1
    assert handlers[0].level == logging.ERROR
    assert handlers[0].stream is not None and not handlers[0].stream.closed


def test_importing_the_logger_leaks_no_file_handle() -> None:
    """A fresh interpreter that imports the logger reports no unclosed file."""
    code = "import gc, src.notification.logger; gc.collect()"
    result = subprocess.run(
        [sys.executable, "-X", "dev", "-W", "always::ResourceWarning", "-c", code],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "unclosed file" not in result.stderr
