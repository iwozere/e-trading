"""
Unit tests for the logging configuration system.

Tests pipeline-specific logging, context management, and performance metrics.
"""

import unittest
from unittest.mock import patch, MagicMock
import time

from src.ml.pipeline.p04_short_squeeze.config.logging_config import (
    PipelineLogger, LoggerFactory, setup_pipeline_logging,
    PIPELINE_LOGGER, SCREENER_LOGGER
)


class TestPipelineLogger(unittest.TestCase):
    """Test cases for PipelineLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        self.run_id = "test_20241025_120000"
        self.logger = PipelineLogger(PIPELINE_LOGGER, self.run_id)

    def test_initialization(self):
        """Test logger initialization."""
        self.assertEqual(self.logger.name, PIPELINE_LOGGER)
        self.assertEqual(self.logger.run_id, self.run_id)
        self.assertIsNotNone(self.logger.logger)
        self.assertEqual(self.logger._context, {})
        self.assertEqual(self.logger._metrics, {})

    def test_set_and_clear_context(self):
        """Test setting and clearing logging context."""
        self.logger.set_context(ticker="AAPL", operation="screener")

        self.assertEqual(self.logger._context["ticker"], "AAPL")
        self.assertEqual(self.logger._context["operation"], "screener")

        self.logger.clear_context()
        self.assertEqual(self.logger._context, {})

    def test_add_and_get_metrics(self):
        """Test adding and retrieving metrics."""
        self.logger.add_metric("api_calls", 10)
        self.logger.add_metric("duration", 5.5)

        metrics = self.logger.get_metrics()
        self.assertEqual(metrics["api_calls"], 10)
        self.assertEqual(metrics["duration"], 5.5)

        self.logger.clear_metrics()
        self.assertEqual(self.logger.get_metrics(), {})

    def test_format_message_with_context(self):
        """Test message formatting with context."""
        self.logger.set_context(ticker="AAPL", operation="test")

        formatted = self.logger._format_message("Test message")

        self.assertIn(self.run_id, formatted)
        self.assertIn("ticker=AAPL", formatted)
        self.assertIn("operation=test", formatted)
        self.assertIn("Test message", formatted)

    def test_format_message_without_context(self):
        """Test message formatting without context."""
        formatted = self.logger._format_message("Test message")

        self.assertIn(self.run_id, formatted)
        self.assertIn("Test message", formatted)
        self.assertNotIn("[", formatted.split("]")[1])  # No context brackets after run_id

    @patch('src.ml.pipeline.p04_short_squeeze.config.logging_config.setup_logger')
    def test_logging_methods(self, mock_setup_logger):
        """Test all logging level methods."""
        mock_logger = MagicMock()
        mock_setup_logger.return_value = mock_logger

        logger = PipelineLogger("test_logger", self.run_id)
        logger.set_context(test="value")

        # Test all logging methods
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.exception("Exception message")
        logger.critical("Critical message")

        # Verify all methods were called
        mock_logger.debug.assert_called_once()
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_called_once()
        mock_logger.exception.assert_called_once()
        mock_logger.critical.assert_called_once()

        # Check that messages were formatted with context
        debug_call_args = mock_logger.debug.call_args[0][0]
        self.assertIn(self.run_id, debug_call_args)
        self.assertIn("test=value", debug_call_args)

    def test_operation_context_success(self):
        """Test operation context manager for successful operations."""
        with patch.object(self.logger, 'info') as mock_info:
            with self.logger.operation_context("test_operation", param="value"):
                time.sleep(0.01)  # Small delay to test timing

        # Check that start and completion messages were logged
        self.assertEqual(mock_info.call_count, 2)

        # Check that context was set
        start_call_args = mock_info.call_args_list[0]
        self.assertIn("Starting operation: %s", start_call_args[0][0])
        self.assertEqual(start_call_args[0][1], "test_operation")

        completion_call_args = mock_info.call_args_list[1]
        self.assertIn("Completed operation: %s", completion_call_args[0][0])
        self.assertEqual(completion_call_args[0][1], "test_operation")

        # Check that metrics were collected
        metrics = self.logger.get_metrics()
        self.assertIn("test_operation_duration", metrics)
        self.assertGreater(metrics["test_operation_duration"], 0)

    def test_operation_context_failure(self):
        """Test operation context manager for failed operations."""
        with patch.object(self.logger, 'info') as mock_info, \
             patch.object(self.logger, 'error') as mock_error:

            with self.assertRaises(ValueError):
                with self.logger.operation_context("failing_operation"):
                    raise ValueError("Test error")

        # Check that start message was logged
        mock_info.assert_called_once()
        start_call_args = mock_info.call_args
        self.assertIn("Starting operation: %s", start_call_args[0][0])
        self.assertEqual(start_call_args[0][1], "failing_operation")

        # Check that error message was logged
        mock_error.assert_called_once()
        error_call_args = mock_error.call_args
        self.assertIn("Failed operation: %s", error_call_args[0][0])
        self.assertEqual(error_call_args[0][1], "failing_operation")

        # Check that error metrics were collected
        metrics = self.logger.get_metrics()
        self.assertIn("failing_operation_duration", metrics)
        self.assertIn("failing_operation_error", metrics)
        self.assertEqual(metrics["failing_operation_error"], "Test error")

    def test_timed_operation(self):
        """Test timed operation context manager."""
        with patch.object(self.logger, 'debug') as mock_debug:
            with self.logger.timed_operation("timed_test"):
                time.sleep(0.01)

        # Check that completion message was logged
        mock_debug.assert_called_once()
        debug_call_args = mock_debug.call_args
        self.assertIn("Operation %s completed in %.2fs", debug_call_args[0][0])
        self.assertEqual(debug_call_args[0][1], "timed_test")

        # Check that timing metric was collected
        metrics = self.logger.get_metrics()
        self.assertIn("timed_test_duration", metrics)
        self.assertGreater(metrics["timed_test_duration"], 0)

    def test_log_performance_summary_with_metrics(self):
        """Test performance summary logging with metrics."""
        self.logger.add_metric("api_calls", 25)
        self.logger.add_metric("operation_duration", 12.5)
        self.logger.add_metric("success_rate", 0.95)

        with patch.object(self.logger, 'info') as mock_info:
            self.logger.log_performance_summary()

        # Should log summary header plus one line per metric
        self.assertEqual(mock_info.call_count, 4)

        # Check summary header
        header_call = mock_info.call_args_list[0][0][0]
        self.assertIn("Performance Summary", header_call)

        # Check that duration metrics are formatted properly
        duration_logged = False
        for call in mock_info.call_args_list[1:]:
            call_args = call[0]
            if len(call_args) >= 3 and "operation_duration" in str(call_args[1]):
                # Check format string contains %.2fs for duration formatting
                self.assertIn("%.2fs", call_args[0])
                self.assertEqual(call_args[2], 12.5)
                duration_logged = True

        self.assertTrue(duration_logged, "Duration metric should be formatted with 's' suffix")

    def test_log_performance_summary_no_metrics(self):
        """Test performance summary logging without metrics."""
        with patch.object(self.logger, 'info') as mock_info:
            self.logger.log_performance_summary()

        mock_info.assert_called_once()
        call_msg = mock_info.call_args[0][0]
        self.assertIn("No performance metrics collected", call_msg)


class TestLoggerFactory(unittest.TestCase):
    """Test cases for LoggerFactory class."""

    def test_get_pipeline_logger(self):
        """Test getting pipeline logger."""
        run_id = "test_run_123"
        logger = LoggerFactory.get_pipeline_logger(run_id)

        self.assertIsInstance(logger, PipelineLogger)
        self.assertEqual(logger.name, PIPELINE_LOGGER)
        self.assertEqual(logger.run_id, run_id)

    def test_get_screener_logger(self):
        """Test getting screener logger."""
        run_id = "test_run_456"
        logger = LoggerFactory.get_screener_logger(run_id)

        self.assertIsInstance(logger, PipelineLogger)
        self.assertEqual(logger.name, SCREENER_LOGGER)
        self.assertEqual(logger.run_id, run_id)

    def test_logger_factory_auto_run_id(self):
        """Test logger factory with automatic run_id generation."""
        logger = LoggerFactory.get_pipeline_logger()

        self.assertIsInstance(logger, PipelineLogger)
        self.assertIsNotNone(logger.run_id)
        self.assertTrue(len(logger.run_id) > 0)


class TestSetupPipelineLogging(unittest.TestCase):
    """Test cases for setup_pipeline_logging function."""

    def test_setup_pipeline_logging_with_run_id(self):
        """Test setting up pipeline logging with specific run_id."""
        run_id = "test_setup_123"
        loggers = setup_pipeline_logging(run_id)

        self.assertIsInstance(loggers, dict)
        self.assertIn('pipeline', loggers)
        self.assertIn('screener', loggers)
        self.assertIn('deep_scan', loggers)
        self.assertIn('alert', loggers)
        self.assertIn('reporting', loggers)

        # Check that all loggers have the same run_id
        for logger in loggers.values():
            self.assertEqual(logger.run_id, run_id)

    def test_setup_pipeline_logging_auto_run_id(self):
        """Test setting up pipeline logging with automatic run_id."""
        loggers = setup_pipeline_logging()

        self.assertIsInstance(loggers, dict)
        self.assertEqual(len(loggers), 5)

        # Check that all loggers have the same auto-generated run_id
        run_ids = [logger.run_id for logger in loggers.values()]
        self.assertEqual(len(set(run_ids)), 1)  # All should be the same


if __name__ == '__main__':
    unittest.main()