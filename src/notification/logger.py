"""
Provides logging utilities for the trading system, including console and file logging.

This module sets up application-wide logging configuration and exposes a logger for use throughout the project.
"""

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import logging.config
from datetime import datetime as dt

# Ensure log directory exists
log_dir = Path("logs") / "log"
log_dir.mkdir(parents=True, exist_ok=True)

# Constants for log file configuration
MAX_BYTES = 500 * 1024 * 1024  # 500MB
BACKUP_COUNT = 99  # Keep 99 backup files


####################################################################
# Simple logger, which logs to console
####################################################################
def print_log(msg: str):
    # Get current timestamp
    current_time = dt.now()
    time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Print the timestamp in a human-readable format
    print(f"{time} {msg}")


####################################################################
# This is the logger that will be used in the application.
# Example usage:
# from src.notification.logger import setup_logger
# _logger = setup_logger(__name__)
####################################################################
LOG_CONFIG = {
    "version": 1,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        },
        "standard": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",  # Less verbose for console
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/log/app.log",
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "trade_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/log/trades.log",
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "telegram_bot_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/log/telegram_bot.log",
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "telegram_screener_bot_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/log/telegram_screener_bot.log",
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/log/app_errors.log",
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "ERROR",
            "formatter": "detailed",
        },
    },
    "loggers": {
        "matplotlib": {
            "level": "WARNING",
            "handlers": ["console"],
            "propagate": False,
        },
        "live_trader": {
            "handlers": ["console", "file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "telegram_bot": {
            "handlers": ["console", "telegram_bot_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "telegram_screener_bot": {
            "handlers": ["console", "telegram_screener_bot_file"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    "root": {"handlers": ["console", "file"], "level": "DEBUG"},
}

# --- PATCH LOG_CONFIG for UTF-8 encoding on all file handlers ---
for handler_name in ["file", "trade_file", "telegram_bot_file", "error_file"]:
    if handler_name in LOG_CONFIG["handlers"]:
        LOG_CONFIG["handlers"][handler_name]["encoding"] = "utf-8"

logging.config.dictConfig(LOG_CONFIG)
_logger = logging.getLogger()


def log_exception(logger, exc_info=None):
    """Log an exception with full stack trace"""
    if exc_info is None:
        exc_info = sys.exc_info()
    logger.error("Exception occurred:", exc_info=True)
    logger.error("Full traceback:\n" + "".join(traceback.format_exception(*exc_info)))


#
# Set up the logger for the application
# Usage: setup_logger('live_trader')
def setup_logger(name: str, log_file: str = None, level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up the logger with custom configuration.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, uses default handlers.
        level (int, optional): Logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add custom handlers if logger has no handlers and log_file is specified
    if not logger.hasHandlers() and log_file:
        # Ensure the log directory exists
        log_dir = Path(log_file).resolve().parent
        if log_dir and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        # Create file handler with UTF-8 encoding
        file_handler = RotatingFileHandler(
            log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Ensure all existing StreamHandlers (console) use UTF-8 if possible
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            stream = getattr(handler, "stream", None)
            if stream and hasattr(stream, "reconfigure"):
                try:
                    stream.reconfigure(encoding="utf-8")
                except Exception:
                    pass

    return logger
