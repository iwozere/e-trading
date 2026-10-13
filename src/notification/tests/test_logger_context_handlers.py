"""
Tests for ContextAwareLogger handler inheritance.

A f"{__name__}.ClassName" logger must not get a context file handler that its
module-level parent logger already has, or every record is written twice.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import ContextAwareLogger  # noqa: E402


def test_child_logger_does_not_duplicate_ancestor_file_handler(tmp_path: Path) -> None:
    """A record from the class logger lands once in the shared context file."""
    log_file = tmp_path / "ctx.log"
    context_name = "test_ctx_dup_context"
    module_name = "test_ctx_dup_pkg.module"

    context_logger = logging.getLogger(context_name)
    handler = RotatingFileHandler(log_file, encoding="utf-8")
    context_logger.addHandler(handler)
    try:
        module_logger = ContextAwareLogger(module_name, context_name).logger
        class_logger = ContextAwareLogger(f"{module_name}.Client", context_name).logger
        class_logger.setLevel(logging.INFO)

        class_logger.info("delivered once")
        handler.flush()

        assert handler in module_logger.handlers
        assert handler not in class_logger.handlers
        assert log_file.read_text(encoding="utf-8").count("delivered once") == 1
    finally:
        for name in (context_name, module_name, f"{module_name}.Client"):
            logging.getLogger(name).removeHandler(handler)
        handler.close()
