"""
Provides logging utilities for the trading system, including console and file logging.

This module sets up application-wide logging configuration and exposes a logger for use throughout the project.
"""

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging.config
from datetime import datetime as dt

# Ensure log directory exists
log_dir = os.path.join("logs", "log")
os.makedirs(log_dir, exist_ok=True)

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
# from _logging_config import _logger
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
        # Add telegram_bot_file and error_file handlers to relevant loggers as needed
    },
    "root": {"handlers": ["console", "file"], "level": "DEBUG"},
}


logging.config.dictConfig(LOG_CONFIG)
_logger = logging.getLogger()


def log_exception(logger, exc_info=None):
    """Log an exception with full stack trace"""
    if exc_info is None:
        exc_info = sys.exc_info()
    logger.error("Exception occurred:", exc_info=exc_info)
    logger.error("Full traceback:\n" + "".join(traceback.format_exception(*exc_info)))


#
# Set up the logger for the application
# Usage: setup_logger('live_trader')
def setup_logger(
    name: str, log_file: str = None, level: int = logging.DEBUG
) -> logging.Logger:
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
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
