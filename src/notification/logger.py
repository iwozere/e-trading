"""
Provides logging utilities for the trading system, including console and file logging.

This module sets up application-wide logging configuration and exposes a logger for use throughout the project.

Supports multiprocessing-safe logging through QueueHandler and QueueListener.
"""

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from pathlib import Path
from contextvars import ContextVar
from multiprocessing import Queue, Manager
import atexit
import threading

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
else:
    # Ensure it's at the front if it was already appended
    sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT))

import logging.config
from datetime import datetime as dt

# Ensure log directory exists relative to project root
log_dir = PROJECT_ROOT / "logs" / "log"
log_dir.mkdir(parents=True, exist_ok=True)

# Constants for log file configuration
MAX_BYTES = 500 * 1024 * 1024  # 500MB
BACKUP_COUNT = 99  # Keep 99 backup files

# Context variable to track the current logging context
_logging_context = ContextVar('logging_context', default=None)


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
            "format": "%(asctime)s - [PID %(process)d] - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        },
        "standard": {"format": "%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",  # Less verbose for console
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "app.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "trade_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "trades.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "order_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "orders.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },

        "telegram_screener_bot_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "telegram_screener_bot.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "telegram_background_services_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "telegram_background_services.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "telegram_alert_monitor_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "telegram_alert_monitor.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "telegram_schedule_processor_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "telegram_schedule_processor.log"),
            "maxBytes": MAX_BYTES,
            "backupCount": BACKUP_COUNT,
            "level": "DEBUG",
            "formatter": "detailed",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(PROJECT_ROOT / "logs" / "log" / "app_errors.log"),
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

        "telegram_screener_bot": {
            "handlers": ["console", "telegram_screener_bot_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "telegram_background_services": {
            "handlers": ["console", "telegram_background_services_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "telegram_alert_monitor": {
            "handlers": ["console", "telegram_alert_monitor_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        "telegram_schedule_processor": {
            "handlers": ["console", "telegram_schedule_processor_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        # Add notification manager with context awareness
        "src.notification.async_notification_manager": {
            "handlers": ["console"],  # Will be dynamically configured
            "level": "DEBUG",
            "propagate": True,  # Allow propagation to parent loggers
        },
        # Trading-specific loggers
        "orders": {
            "handlers": ["console", "order_file"],
            "level": "INFO",
            "propagate": False,
        },
        "trades": {
            "handlers": ["console", "trade_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {"handlers": ["console", "file"], "level": "DEBUG"},
}

# --- PATCH LOG_CONFIG for UTF-8 encoding and conditional file handlers ---
# Determine if we are in a main process or a fork/spawn child
import multiprocessing
import os

# Robust check for main process
# On Windows/spawn, parent_process() is None only for the true root process
# But sometimes frameworks mangle this. Let's check the name too.
try:
    _current = multiprocessing.current_process()
    _is_main_process = (_current.name == 'MainProcess')

    # Double check with parent process if available (Python 3.8+)
    if hasattr(multiprocessing, 'parent_process'):
        _parent = multiprocessing.parent_process()
        if _parent is not None:
             _is_main_process = False

    # Check if we are imported as __main__ vs module? No, logger is imported.
except Exception:
    _is_main_process = True


# These keys are not accepted by logging.NullHandler
INVALID_NULL_HANDLER_KEYS = ['filename', 'maxBytes', 'backupCount', 'encoding', 'delay']

for handler_name in LOG_CONFIG["handlers"]:
    handler_config = LOG_CONFIG["handlers"][handler_name]

    # Ensure encoding is UTF-8 for all file handlers
    if "filename" in handler_config:
        handler_config["encoding"] = "utf-8"

        # If we are in a child process, append PID to filename to avoid locking/rollover issues
        if not _is_main_process:
            import os
            base_filename = handler_config["filename"]
            # Check if it already has a PID (avoid double appending if re-imported)
            pid_suffix = f".{os.getpid()}"
            if not base_filename.endswith(pid_suffix):
                new_filename = f"{base_filename}{pid_suffix}"
                handler_config["filename"] = new_filename



logging.config.dictConfig(LOG_CONFIG)
_logger = logging.getLogger(__name__)

# Global queue listener for multiprocessing-safe logging
_queue_listener = None
_log_queue = None
_manager = None
_listener_lock = threading.Lock()


def _create_file_handlers(pid_suffix: str = ""):
    """
    Create all file handlers for multiprocessing-safe logging.

    Args:
        pid_suffix: Optional suffix to append to filenames (e.g., ".1234")

    Returns:
        list: List of configured file handlers
    """
    handlers = []

    def get_log_path(name: str) -> str:
        base_path = str(PROJECT_ROOT / "logs" / "log" / name)
        if pid_suffix:
            return f"{base_path}.{pid_suffix}"
        return base_path

    # Create file handler for main app log
    file_handler = RotatingFileHandler(
        get_log_path("app.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - [PID %(process)d] - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
    ))
    handlers.append(file_handler)

    # Create trade log handler
    trade_handler = RotatingFileHandler(
        get_log_path("trades.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    trade_handler.setLevel(logging.DEBUG)
    trade_handler.setFormatter(logging.Formatter(
        "%(asctime)s - [PID %(process)d] - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
    ))
    handlers.append(trade_handler)

    # Create order log handler
    order_handler = RotatingFileHandler(
        get_log_path("orders.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    order_handler.setLevel(logging.DEBUG)
    order_handler.setFormatter(logging.Formatter(
        "%(asctime)s - [PID %(process)d] - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
    ))
    handlers.append(order_handler)

    # Create error log handler
    error_handler = RotatingFileHandler(
        get_log_path("app_errors.log"),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s - [PID %(process)d] - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
    ))
    handlers.append(error_handler)

    return handlers


def setup_multiprocessing_logging():
    """
    Set up multiprocessing-safe logging using QueueHandler and QueueListener.

    This should be called ONCE in the main process before spawning workers.
    Worker processes will automatically use QueueHandler to send logs to the main process.

    Returns:
        Queue: The log queue that worker processes should use
    """
    global _queue_listener, _log_queue, _manager

    with _listener_lock:
        if _queue_listener is not None:
            # Already set up
            return _log_queue

        # Create a manager and a queue for log messages
        if _manager is None:
            _manager = Manager()
        _log_queue = _manager.Queue(-1)

        # Create file handlers
        file_handlers = _create_file_handlers()

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s"
        ))

        # Combine all handlers
        all_handlers = [console_handler] + file_handlers

        # Create and start the queue listener
        _queue_listener = QueueListener(_log_queue, *all_handlers, respect_handler_level=True)
        _queue_listener.start()

        # Register cleanup function
        atexit.register(shutdown_multiprocessing_logging)

        _logger.info("Multiprocessing-safe logging initialized")

        return _log_queue


def shutdown_multiprocessing_logging():
    """
    Shut down the queue listener and manager.

    This is automatically called at exit via atexit.register().
    """
    global _queue_listener, _manager

    with _listener_lock:
        if _queue_listener is not None:
            _queue_listener.stop()
            _queue_listener = None

        if _manager is not None:
            _manager.shutdown()
            _manager = None

        _logger.info("Multiprocessing-safe logging shut down")


def get_multiprocessing_logger(name: str, level: int = logging.DEBUG, external_queue: Queue = None) -> logging.Logger:
    """
    Get a logger configured for multiprocessing.

    Args:
        name: Logger name
        level: Logging level
        external_queue: Optional shared queue to use (crucial for joblib workers)

    Returns:
        logging.Logger: Configured logger
    """
    global _log_queue

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Use external queue if provided, otherwise fallback to global
    queue_to_use = external_queue or _log_queue

    # Remove existing handlers to avoid duplicates/collisions
    logger.handlers.clear()

    # Add queue handler if we have a queue
    if queue_to_use is not None:
        queue_handler = QueueHandler(queue_to_use)
        queue_handler.setLevel(level)
        logger.addHandler(queue_handler)

        if not _is_main_process:
            # CHILD PROCESS: Also add local file handlers with PID suffix to avoid rollover lock issues
            # This ensures logs go to the central queue (via main) AND to a local PID-dedicated file.
            pass

        logger.propagate = False
    else:
        # Fallback to local config
        pass

    # For child processes, automatically add PID-suffixed file handlers to the root
    if not _is_main_process:
        root = logging.getLogger()
        # Only add if not already present
        has_pid_handler = any(isinstance(h, RotatingFileHandler) and str(os.getpid()) in h.baseFilename for h in root.handlers)

        if not has_pid_handler:
            # Add PID-suffixed file handlers to root so all loggers benefit
            child_handlers = _create_file_handlers(pid_suffix=str(os.getpid()))
            for h in child_handlers:
                root.addHandler(h)

    return logger
    """Log an exception with full stack trace"""
    logger.error("Full traceback:\n" + "".join(traceback.format_exception(*exc_info)))


class ContextAwareLogger:
    """
    A logger that can inherit logging context from its parent.
    This allows child modules to log to the same file as their calling module.
    """

    def __init__(self, name: str, parent_logger_name: str = None):
        self.name = name
        self.parent_logger_name = parent_logger_name
        self.logger = logging.getLogger(name)

        # If we have a parent logger, inherit its file handlers
        if parent_logger_name:
            self._inherit_parent_handlers(parent_logger_name)

    def _inherit_parent_handlers(self, parent_name: str):
        """Inherit file handlers from parent logger"""
        parent_logger = logging.getLogger(parent_name)

        # Find file handlers in parent logger
        for handler in parent_logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                # Create a new handler with the same configuration
                new_handler = RotatingFileHandler(
                    handler.baseFilename,
                    maxBytes=handler.maxBytes,
                    backupCount=handler.backupCount,
                    encoding=handler.encoding
                )
                new_handler.setLevel(handler.level)
                new_handler.setFormatter(handler.formatter)

                # Add to this logger
                self.logger.addHandler(new_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self.logger.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)


def _apply_context_to_existing_logger(logger_name: str, context_name: str):
    """
    Apply context to an existing logger that was already created.
    This is useful for loggers created at import time.
    """
    logger = logging.getLogger(logger_name)
    context_logger = logging.getLogger(context_name)

    # Find file handlers in context logger
    for handler in context_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            # Check if this handler is already added
            handler_exists = False
            for existing_handler in logger.handlers:
                if (isinstance(existing_handler, RotatingFileHandler) and
                    existing_handler.baseFilename == handler.baseFilename):
                    handler_exists = True
                    break

            if not handler_exists:
                # Create a new handler with the same configuration
                new_handler = RotatingFileHandler(
                    handler.baseFilename,
                    maxBytes=handler.maxBytes,
                    backupCount=handler.backupCount,
                    encoding=handler.encoding
                )
                new_handler.setLevel(handler.level)
                new_handler.setFormatter(handler.formatter)

                # Add to this logger
                logger.addHandler(new_handler)


def set_logging_context(context_name: str):
    """
    Set the current logging context.
    This allows child modules to inherit the logging context.

    Args:
        context_name: The name of the logging context (e.g., 'telegram_screener_bot')
    """
    _logging_context.set(context_name)

    # Apply context to existing notification loggers
    _apply_context_to_existing_logger("src.notification.async_notification_manager", context_name)


def get_logging_context() -> str:
    """Get the current logging context"""
    return _logging_context.get()


def setup_logger(name: str, log_file: str = None, level: int = logging.DEBUG,
                inherit_from: str = None, use_multiprocessing: bool = False, **kwargs) -> logging.Logger:
    """
    Set up the logger with custom configuration.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file. If None, uses default handlers.
        level (int, optional): Logging level. Defaults to logging.DEBUG.
        inherit_from (str, optional): Parent logger name to inherit handlers from.
        use_multiprocessing (bool, optional): If True, use multiprocessing-safe logging.
                                             This should be True for code that runs in
                                             worker processes (e.g., Optuna trials).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # If multiprocessing mode is requested, use queue-based logging
    if use_multiprocessing:
        # Check if we have an external queue in kwargs (hacky but works for joblib passing)
        external_queue = kwargs.get('external_queue')
        return get_multiprocessing_logger(name, level, external_queue=external_queue)
    # Check if we should inherit from a parent logger
    if inherit_from:
        return ContextAwareLogger(name, inherit_from).logger

    # Check if we have a logging context set
    context = get_logging_context()
    if context:
        # Inherit from the context logger for notification modules
        if name.startswith('src.notification.'):
            return ContextAwareLogger(name, context).logger

    # Standard logger setup
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
            "%(asctime)s - [PID %(process)d] - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
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


def setup_context_logger(name: str, context_name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up a logger that inherits from a specific context.

    Args:
        name (str): Name of the logger.
        context_name (str): Name of the context logger to inherit from.
        level (int, optional): Logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return setup_logger(name, inherit_from=context_name, level=level)
