"""
Logging configuration and integration for the Short Squeeze Detection Pipeline.

This module extends the existing notification system logging to provide
pipeline-specific logging with structured context and performance metrics.
"""

import logging
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional
from datetime import datetime

from src.notification.logger import setup_logger, set_logging_context

# Pipeline-specific logger names
PIPELINE_LOGGER = "short_squeeze_pipeline"
SCREENER_LOGGER = "short_squeeze_screener"
DEEP_SCAN_LOGGER = "short_squeeze_deep_scan"
ALERT_LOGGER = "short_squeeze_alerts"
REPORTING_LOGGER = "short_squeeze_reporting"


class PipelineLogger:
    """
    Enhanced logger for pipeline operations with structured context and metrics.

    Provides context-aware logging with performance tracking and integration
    with the existing notification system.
    """

    def __init__(self, name: str, run_id: Optional[str] = None):
        """
        Initialize pipeline logger.

        Args:
            name: Logger name (should be one of the predefined logger constants).
            run_id: Optional run identifier for context tracking.
        """
        self.name = name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger(name)
        self._context: Dict[str, Any] = {}
        self._metrics: Dict[str, Any] = {}

        # Set logging context for notification system integration
        set_logging_context(name)

    def set_context(self, **kwargs) -> None:
        """
        Set logging context for structured logging.

        Args:
            **kwargs: Context key-value pairs to include in log messages.
        """
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all logging context."""
        self._context.clear()

    def add_metric(self, key: str, value: Any) -> None:
        """
        Add a performance metric.

        Args:
            key: Metric name.
            value: Metric value.
        """
        self._metrics[key] = value

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self._metrics.copy()

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()

    def _format_message(self, message: str) -> str:
        """Format message with context and run_id."""
        context_str = ""
        if self._context:
            context_parts = [f"{k}={v}" for k, v in self._context.items()]
            context_str = f" [{', '.join(context_parts)}]"

        return f"[{self.run_id}]{context_str} {message}"

    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message with context."""
        self.logger.debug(self._format_message(message), *args, **kwargs)

    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message with context."""
        self.logger.info(self._format_message(message), *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message with context."""
        self.logger.warning(self._format_message(message), *args, **kwargs)

    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message with context."""
        self.logger.error(self._format_message(message), *args, **kwargs)

    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with context and stack trace."""
        self.logger.exception(self._format_message(message), *args, **kwargs)

    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message with context."""
        self.logger.critical(self._format_message(message), *args, **kwargs)

    @contextmanager
    def operation_context(self, operation: str, **context):
        """
        Context manager for operation-specific logging.

        Args:
            operation: Operation name.
            **context: Additional context for the operation.
        """
        original_context = self._context.copy()

        try:
            self.set_context(operation=operation, **context)
            self.info("Starting operation: %s", operation)
            start_time = time.time()

            yield self

            duration = time.time() - start_time
            self.add_metric(f"{operation}_duration", duration)
            self.info("Completed operation: %s (%.2fs)", operation, duration)

        except Exception as e:
            duration = time.time() - start_time
            self.add_metric(f"{operation}_duration", duration)
            self.add_metric(f"{operation}_error", str(e))
            self.error("Failed operation: %s (%.2fs) - %s", operation, duration, str(e))
            raise
        finally:
            self._context = original_context

    @contextmanager
    def timed_operation(self, operation: str):
        """
        Context manager for timing operations.

        Args:
            operation: Operation name for metrics.
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.add_metric(f"{operation}_duration", duration)
            self.debug("Operation %s completed in %.2fs", operation, duration)

    def log_performance_summary(self) -> None:
        """Log a summary of collected performance metrics."""
        if not self._metrics:
            self.info("No performance metrics collected")
            return

        self.info("Performance Summary:")
        for key, value in self._metrics.items():
            if isinstance(value, float) and key.endswith('_duration'):
                self.info("  %s: %.2fs", key, value)
            else:
                self.info("  %s: %s", key, value)


class LoggerFactory:
    """Factory for creating pipeline-specific loggers."""

    @staticmethod
    def get_pipeline_logger(run_id: Optional[str] = None) -> PipelineLogger:
        """Get main pipeline logger."""
        return PipelineLogger(PIPELINE_LOGGER, run_id)

    @staticmethod
    def get_screener_logger(run_id: Optional[str] = None) -> PipelineLogger:
        """Get screener module logger."""
        return PipelineLogger(SCREENER_LOGGER, run_id)

    @staticmethod
    def get_deep_scan_logger(run_id: Optional[str] = None) -> PipelineLogger:
        """Get deep scan module logger."""
        return PipelineLogger(DEEP_SCAN_LOGGER, run_id)

    @staticmethod
    def get_alert_logger(run_id: Optional[str] = None) -> PipelineLogger:
        """Get alert engine logger."""
        return PipelineLogger(ALERT_LOGGER, run_id)

    @staticmethod
    def get_reporting_logger(run_id: Optional[str] = None) -> PipelineLogger:
        """Get reporting engine logger."""
        return PipelineLogger(REPORTING_LOGGER, run_id)


def setup_pipeline_logging(run_id: Optional[str] = None) -> Dict[str, PipelineLogger]:
    """
    Set up all pipeline loggers with consistent run_id.

    Args:
        run_id: Optional run identifier. Generated if not provided.

    Returns:
        Dictionary of logger name to PipelineLogger instance.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    loggers = {
        'pipeline': LoggerFactory.get_pipeline_logger(run_id),
        'screener': LoggerFactory.get_screener_logger(run_id),
        'deep_scan': LoggerFactory.get_deep_scan_logger(run_id),
        'alert': LoggerFactory.get_alert_logger(run_id),
        'reporting': LoggerFactory.get_reporting_logger(run_id),
    }

    # Log initialization
    loggers['pipeline'].info("Pipeline logging initialized with run_id: %s", run_id)

    return loggers