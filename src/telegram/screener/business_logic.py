import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import asyncio
import time
from typing import Any, Dict, List, Optional
from src.telegram.command_parser import ParsedCommand
from src.common.common import determine_provider
# Removed calculate_technicals_unified - now using IndicatorService directly
from src.model.telegram_bot import TickerAnalysis
from src.common.ticker_analyzer import format_ticker_report, analyze_ticker
from src.telegram.screener.report_config_parser import ReportConfigParser
from src.indicators.service import IndicatorService
from src.indicators.models import TickerIndicatorsRequest, IndicatorResultSet
# Service layer imports - direct db import removed, now using service instances

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class TelegramBusinessLogic:
    """
    Service-aware business logic class for telegram bot operations.

    This class handles all telegram bot business logic while delegating
    database operations to telegram_service and indicator calculations
    to IndicatorService, following clean architecture principles.
    """

    def __init__(self, telegram_service, indicator_service: Optional[IndicatorService] = None):
        """
        Initialize business logic with service dependencies and error handling.

        Args:
            telegram_service: Service for telegram-related database operations
            indicator_service: Service for technical and fundamental indicator calculations
        """
        self.telegram_service = telegram_service

        # Initialize indicator service with error handling
        try:
            self.indicator_service = indicator_service or IndicatorService()
        except Exception:
            _logger.exception("Failed to initialize IndicatorService:")
            self.indicator_service = None

        # Validate service instances
        self._validate_services()

    def _validate_services(self) -> None:
        """
        Validate that required services are properly initialized.

        Logs warnings for missing services but doesn't fail initialization
        to allow graceful degradation.
        """
        if not self.telegram_service:
            _logger.error("TelegramService not available - database operations will fail")
        else:
            _logger.debug("TelegramService initialized successfully")

        if not self.indicator_service:
            _logger.warning("IndicatorService not available - indicator calculations will fail")
        else:
            _logger.debug("IndicatorService initialized successfully")

    def get_service_health_status(self) -> Dict[str, Any]:
        """
        Get the health status of all service dependencies.

        Returns:
            Dict with service health information
        """
        return {
            "telegram_service": {
                "available": self.telegram_service is not None,
                "type": type(self.telegram_service).__name__ if self.telegram_service else None
            },
            "indicator_service": {
                "available": self.indicator_service is not None,
                "type": type(self.indicator_service).__name__ if self.indicator_service else None,
                "adapters": list(self.indicator_service.adapters.keys()) if self.indicator_service and hasattr(self.indicator_service, 'adapters') else []
            }
        }

    async def handle_command(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Main business logic handler with comprehensive error handling.

        Dispatches based on command and parameters with service layer error handling,
        fallback behavior, and user-friendly error messages.

        Args:
            parsed: ParsedCommand object containing command and arguments

        Returns:
            Dict with result/status/data for notification manager
        """
        # Validate service availability before processing commands
        if not self.telegram_service:
            _logger.error("Cannot process command %s - telegram service not available", parsed.command)
            return {
                "status": "error",
                "message": "Service temporarily unavailable. Please try again later.",
                "error_type": "ServiceUnavailable"
            }

        # Log command processing start
        telegram_user_id = parsed.args.get("telegram_user_id", "unknown")
        _logger.info("Processing command %s for user %s", parsed.command, telegram_user_id)

        try:
            # Dispatch to appropriate handler with error handling
            if parsed.command == "report":
                return await self._handle_with_error_wrapper(self.handle_report, parsed, requires_indicator_service=True)
            elif parsed.command == "help":
                return await self._handle_with_error_wrapper(self.handle_help, parsed)
            elif parsed.command == "info":
                return await self._handle_with_error_wrapper(self.handle_info, parsed)
            elif parsed.command == "register":
                return await self._handle_with_error_wrapper(self.handle_register, parsed)
            elif parsed.command == "verify":
                return await self._handle_with_error_wrapper(self.handle_verify, parsed)
            elif parsed.command == "request_approval":
                return await self._handle_with_error_wrapper(self.handle_request_approval, parsed)
            elif parsed.command == "language":
                return await self._handle_with_error_wrapper(self.handle_language, parsed)
            elif parsed.command == "admin":
                return await self._handle_with_error_wrapper(self.handle_admin, parsed)
            elif parsed.command == "alerts":
                return await self._handle_with_error_wrapper(self.handle_alerts, parsed)
            elif parsed.command == "schedules":
                return await self._handle_with_error_wrapper(self.handle_schedules, parsed)
            elif parsed.command == "screener":
                return await self._handle_with_error_wrapper(self.handle_screener, parsed, requires_indicator_service=True)
            elif parsed.command == "feedback":
                return await self._handle_with_error_wrapper(self.handle_feedback, parsed)
            elif parsed.command == "feature":
                return await self._handle_with_error_wrapper(self.handle_feature, parsed)
            else:
                return {"status": "error", "message": f"Unknown command: {parsed.command}"}

        except Exception:
            _logger.exception("Unexpected error in handle_command for %s", parsed.command)
            return {
                "status": "error",
                "message": "An unexpected error occurred. Please try again later.",
                "error_type": "UnexpectedError"
            }

    async def _handle_with_error_wrapper(self, handler_func, parsed: ParsedCommand, requires_indicator_service: bool = False):
        """
        Wrapper for command handlers that provides consistent error handling.

        Args:
            handler_func: The handler function to call
            parsed: ParsedCommand object
            requires_indicator_service: Whether this command requires indicator service

        Returns:
            Dict with result or error information
        """
        try:
            # Check if indicator service is required and available
            if requires_indicator_service and not self.indicator_service:
                _logger.warning("Command %s requires indicator service but it's not available", parsed.command)
                return {
                    "status": "error",
                    "message": "Indicator calculation service temporarily unavailable. Please try again later.",
                    "error_type": "IndicatorServiceUnavailable"
                }

            # Call the handler function
            if asyncio.iscoroutinefunction(handler_func):
                result = await handler_func(parsed)
            else:
                result = handler_func(parsed)

            return result

        except Exception as e:
            _logger.exception("Error in handler %s", handler_func.__name__)

            # Provide specific error messages based on exception type
            error_msg = str(e).lower()
            if "timeout" in error_msg or "connection" in error_msg:
                user_message = "Connection timeout. Please try again in a moment."
            elif "permission" in error_msg or "access" in error_msg:
                user_message = "Access denied. Please check your permissions."
            elif "limit" in error_msg or "quota" in error_msg:
                user_message = "Usage limit reached. Please try again later or contact admin."
            else:
                user_message = "An error occurred processing your request. Please try again."

            return {
                "status": "error",
                "message": user_message,
                "error_type": type(e).__name__
            }

    def is_admin_user(self, telegram_user_id: str) -> bool:
        """Check if user is an admin using service layer with error handling."""
        try:
            status = self.safe_telegram_service_call(
                self.telegram_service.get_user_status,
                "get_user_status",
                "is_admin_user",
                telegram_user_id
            )
            return status and status.get("is_admin", False)
        except Exception as e:
            _logger.warning("Error checking admin status for user %s: %s", telegram_user_id, e)
            return False

    def is_approved_user(self, telegram_user_id: str) -> bool:
        """Check if user is approved for restricted features using service layer with error handling."""
        try:
            status = self.safe_telegram_service_call(
                self.telegram_service.get_user_status,
                "get_user_status",
                "is_approved_user",
                telegram_user_id
            )
            return status and status.get("approved", False)
        except Exception as e:
            _logger.warning("Error checking approval status for user %s: %s", telegram_user_id, e)
            return False

    def check_admin_access(self, telegram_user_id: str) -> Dict[str, Any]:
        """Check if user has admin access. Returns error dict if not."""
        if not self.is_admin_user(telegram_user_id):
            return {"status": "error", "message": "Access denied. Admin privileges required."}
        return {"status": "ok"}

    def check_approved_access(self, telegram_user_id: str) -> Dict[str, Any]:
        """Check if user has approved access for restricted features. Returns error dict if not."""
        if not self.is_approved_user(telegram_user_id):
            return {"status": "error", "message": "Access denied, please contact chat's admin or send request for approval using command /request_approval"}
        return {"status": "ok"}

    # Indicator Service Integration Methods
    #
    # These methods provide integration between telegram bot commands and the centralized
    # IndicatorService, replacing direct talib usage and manual indicator calculations.
    #
    # Key features:
    # - Async indicator calculation using IndicatorService
    # - Conversion of telegram parameters to service requests
    # - Comprehensive error handling with user-friendly messages
    # - Value extraction and formatting for telegram display

    async def calculate_indicators_for_ticker(
        self,
        ticker: str,
        indicators: List[str],
        timeframe: str = "1d",
        period: str = "1y",
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate indicators for a ticker using IndicatorService.

        Args:
            ticker: Stock/crypto ticker symbol
            indicators: List of indicator names (e.g., ["RSI", "MACD", "SMA"])
            timeframe: Data timeframe (e.g., "1d", "1h", "15m")
            period: Data period (e.g., "1y", "6mo", "3mo")
            provider: Data provider override

        Returns:
            Dict with status and indicator results or error information
        """
        try:
            # Create request for IndicatorService
            request = TickerIndicatorsRequest(
                ticker=ticker,
                indicators=indicators,
                timeframe=timeframe,
                period=period,
                provider=provider
            )

            # Calculate indicators using service with comprehensive error handling
            result_set = await self.safe_indicator_service_call(
                self.indicator_service.compute_for_ticker,
                "compute_for_ticker",
                "calculate_indicators_for_ticker",
                request
            )

            # Check if we got an error result instead of a proper result set
            if isinstance(result_set, dict) and result_set.get("status") == "error":
                return result_set

            return {
                "status": "ok",
                "ticker": ticker,
                "technical": result_set.technical,
                "fundamental": result_set.fundamental,
                "timeframe": timeframe,
                "period": period
            }

        except ValueError as e:
            # Handle validation errors (invalid ticker, missing columns, etc.)
            _logger.warning("Validation error calculating indicators for ticker %s: %s", ticker, e)
            return self.handle_indicator_service_error(e, f"calculate_indicators_for_ticker({ticker})")

        except RuntimeError as e:
            # Handle adapter/provider errors
            _logger.error("Runtime error calculating indicators for ticker %s: %s", ticker, e)
            return self.handle_indicator_service_error(e, f"calculate_indicators_for_ticker({ticker})")

        except ConnectionError as e:
            # Handle network/connection errors
            _logger.error("Connection error calculating indicators for ticker %s: %s", ticker, e)
            return self.handle_indicator_service_error(e, f"calculate_indicators_for_ticker({ticker})")

        except TimeoutError as e:
            # Handle timeout errors
            _logger.error("Timeout error calculating indicators for ticker %s: %s", ticker, e)
            return self.handle_indicator_service_error(e, f"calculate_indicators_for_ticker({ticker})")

        except Exception as e:
            # Handle any other unexpected errors
            _logger.exception("Unexpected error calculating indicators for ticker %s: %s", ticker, e)
            return self.handle_indicator_service_error(e, f"calculate_indicators_for_ticker({ticker})")

    def convert_telegram_indicator_request(
        self,
        telegram_params: Dict[str, Any]
    ) -> TickerIndicatorsRequest:
        """
        Convert telegram command parameters to IndicatorService request format.

        Args:
            telegram_params: Parameters from telegram command parsing

        Returns:
            TickerIndicatorsRequest object for IndicatorService
        """
        try:
            # Extract parameters with defaults
            ticker = telegram_params.get("ticker", "").upper()
            if not ticker:
                raise ValueError("Ticker is required")

            # Parse indicators from various formats
            indicators_raw = telegram_params.get("indicators", [])
            if isinstance(indicators_raw, str) and indicators_raw.strip():
                # Handle comma-separated string
                indicators = [ind.strip().upper() for ind in indicators_raw.split(",")]
            elif isinstance(indicators_raw, list) and indicators_raw:
                indicators = [str(ind).upper() for ind in indicators_raw]
            else:
                # Default indicators for reports
                indicators = ["RSI", "MACD", "SMA", "BollingerBands"]

            # Clean up indicator names to match registry
            indicator_mapping = {
                "BB": "BollingerBands",
                "BOLLINGER": "BollingerBands",
                "BOLLINGERBANDS": "BollingerBands",
                "MA": "SMA",
                "MOVINGAVERAGE": "SMA"
            }

            normalized_indicators = []
            for ind in indicators:
                normalized = indicator_mapping.get(ind, ind)
                if normalized not in normalized_indicators:
                    normalized_indicators.append(normalized)

            return TickerIndicatorsRequest(
                ticker=ticker,
                indicators=normalized_indicators,
                timeframe=telegram_params.get("interval", "1d"),
                period=telegram_params.get("period", "1y"),
                provider=telegram_params.get("provider")
            )

        except Exception as e:
            _logger.exception("Error converting telegram parameters to indicator request:")
            raise ValueError(f"Invalid indicator request parameters: {str(e)}")

    def handle_indicator_service_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle IndicatorService errors with appropriate user-friendly messages and fallback behavior.

        Args:
            error: Exception from IndicatorService
            context: Additional context for error handling

        Returns:
            Dict with error status and user-friendly message
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__

        # Handle specific error types with appropriate user messages
        if "api key" in error_msg or "authentication" in error_msg:
            user_message = "Unable to fetch market data. API configuration issue detected."
            fallback_available = False
        elif "rate limit" in error_msg or "too many requests" in error_msg:
            user_message = "Rate limit exceeded. Please try again in a few minutes."
            fallback_available = True
        elif "ticker" in error_msg and ("not found" in error_msg or "invalid" in error_msg):
            user_message = "Ticker symbol not found. Please check the symbol and try again."
            fallback_available = False
        elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            user_message = "Network error occurred. Please try again later."
            fallback_available = True
        elif "insufficient data" in error_msg or "no data" in error_msg:
            user_message = "Insufficient historical data available for this ticker."
            fallback_available = False
        elif "unknown indicator" in error_msg or "missing input column" in error_msg:
            user_message = "Invalid indicator configuration. Please check your request."
            fallback_available = False
        elif "no adapter supports" in error_msg:
            user_message = "Indicator calculation not supported. Please try a different indicator."
            fallback_available = False
        elif isinstance(error, ValueError):
            user_message = "Invalid input parameters. Please check your request and try again."
            fallback_available = False
        elif isinstance(error, RuntimeError):
            user_message = "Service temporarily unavailable. Please try again later."
            fallback_available = True
        elif isinstance(error, (ConnectionError, TimeoutError)):
            user_message = "Connection issue occurred. Please try again in a moment."
            fallback_available = True
        else:
            user_message = "Unable to calculate indicators. Please try again later."
            fallback_available = True

        # Log error with appropriate level based on severity
        if fallback_available:
            _logger.warning("IndicatorService error in %s (recoverable): %s", context, error)
        else:
            _logger.error("IndicatorService error in %s (non-recoverable): %s", context, error)

        result = {
            "status": "error",
            "message": user_message,
            "error_type": error_type,
            "fallback_available": fallback_available
        }

        # Include technical details in debug mode
        if _logger.isEnabledFor(10):  # DEBUG level
            result["technical_error"] = str(error)
            result["context"] = context

        return result

    def handle_telegram_service_error(self, error: Exception, operation: str, context: str = "") -> Dict[str, Any]:
        """
        Handle telegram_service errors with appropriate user-friendly messages and fallback behavior.

        Args:
            error: Exception from telegram_service
            operation: The operation that failed (e.g., "get_user_status", "add_alert")
            context: Additional context for error handling

        Returns:
            Dict with error status and user-friendly message
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__

        # Determine if error is recoverable and what fallback behavior to use
        fallback_available = True
        retry_suggested = True

        # Handle specific error types based on operation with enhanced categorization
        if operation in ["get_user_status", "get_user_limit", "list_users", "get_all_users"]:
            if "not found" in error_msg or "does not exist" in error_msg:
                user_message = "User not found. Please register first using /register."
                fallback_available = False
                retry_suggested = False
            elif "database" in error_msg or "connection" in error_msg:
                user_message = "Unable to retrieve user information due to database issue. Please try again in a moment."
            else:
                user_message = "Unable to retrieve user information. Please try again."

        elif operation in ["add_alert", "update_alert", "delete_alert", "list_alerts", "get_alert"]:
            if "limit" in error_msg or "maximum" in error_msg:
                user_message = "Alert limit reached. Please delete some alerts first or contact admin."
                fallback_available = False
                retry_suggested = False
            elif "not found" in error_msg:
                user_message = "Alert not found or access denied."
                fallback_available = False
                retry_suggested = False
            elif "constraint" in error_msg or "duplicate" in error_msg:
                user_message = "Similar alert already exists. Please check your existing alerts."
                fallback_available = False
                retry_suggested = False
            else:
                user_message = "Unable to manage alerts. Please try again later."

        elif operation in ["add_schedule", "update_schedule", "delete_schedule", "list_schedules", "get_schedule"]:
            if "limit" in error_msg or "maximum" in error_msg:
                user_message = "Schedule limit reached. Please delete some schedules first or contact admin."
                fallback_available = False
                retry_suggested = False
            elif "not found" in error_msg:
                user_message = "Schedule not found or access denied."
                fallback_available = False
                retry_suggested = False
            elif "constraint" in error_msg or "duplicate" in error_msg:
                user_message = "Similar schedule already exists. Please check your existing schedules."
                fallback_available = False
                retry_suggested = False
            else:
                user_message = "Unable to manage schedules. Please try again later."

        elif operation in ["verify_user_email", "set_user_limit", "approve_user", "reject_user", "update_user_language"]:
            if "not found" in error_msg:
                user_message = "User not found. Please register first."
                fallback_available = False
                retry_suggested = False
            elif "permission" in error_msg or "access" in error_msg:
                user_message = "Access denied. Admin privileges required."
                fallback_available = False
                retry_suggested = False
            else:
                user_message = "Unable to update user settings. Please try again."

        elif operation in ["log_command_audit", "add_feedback"]:
            user_message = "Unable to log activity. The operation may have completed successfully."
            # These are non-critical operations, so we don't need to fail the main operation
            fallback_available = True
            retry_suggested = False

        elif operation in ["set_verification_code", "count_codes_last_hour"]:
            if "rate limit" in error_msg or "too many" in error_msg:
                user_message = "Too many verification attempts. Please wait before requesting another code."
                fallback_available = False
                retry_suggested = False
            else:
                user_message = "Unable to process verification. Please try again."

        # Handle database-specific errors
        elif "database" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            user_message = "Database connection issue. Please try again in a moment."
            fallback_available = True
            retry_suggested = True
        elif "constraint" in error_msg or "unique" in error_msg or "duplicate" in error_msg:
            user_message = "Data conflict detected. Please check your input and try again."
            fallback_available = False
            retry_suggested = False
        elif "permission" in error_msg or "access" in error_msg or "denied" in error_msg:
            user_message = "Access denied. Please check your permissions."
            fallback_available = False
            retry_suggested = False
        elif "lock" in error_msg or "busy" in error_msg:
            user_message = "System is busy. Please try again in a moment."
            fallback_available = True
            retry_suggested = True
        else:
            user_message = "Service temporarily unavailable. Please try again later."

        # Log error with appropriate level based on severity and recoverability
        if fallback_available and retry_suggested:
            _logger.warning("TelegramService error in %s.%s (recoverable): %s", context or "unknown", operation, error)
        elif fallback_available:
            _logger.info("TelegramService error in %s.%s (non-critical): %s", context or "unknown", operation, error)
        else:
            _logger.error("TelegramService error in %s.%s (non-recoverable): %s", context or "unknown", operation, error)

        result = {
            "status": "error",
            "message": user_message,
            "error_type": error_type,
            "operation": operation,
            "fallback_available": fallback_available,
            "retry_suggested": retry_suggested
        }

        # Include technical details in debug mode
        if _logger.isEnabledFor(10):  # DEBUG level
            result["technical_error"] = str(error)
            result["context"] = context

        return result

    def safe_telegram_service_call(self, operation_func, operation_name: str, context: str = "", *args, **kwargs):
        """
        Safely call a telegram_service method with comprehensive error handling and fallback behavior.

        Args:
            operation_func: The telegram_service method to call
            operation_name: Name of the operation for error reporting
            context: Additional context for error handling
            *args, **kwargs: Arguments to pass to the operation function

        Returns:
            Result from the operation, appropriate default value, or raises exception for critical operations
        """
        # Log service operation start with context information
        operation_context = f"{context}.{operation_name}" if context else operation_name
        _logger.debug("Starting telegram service operation: %s with args=%s, kwargs=%s",
                     operation_context, args, kwargs)

        # Check if service is available
        if not self.telegram_service:
            _logger.error("Telegram service not available for operation %s in %s", operation_name, context)

            # Return appropriate default values based on operation type
            if operation_name in ["get_user_status", "get_user_limit", "get_alert", "get_schedule", "get_schedule_by_id"]:
                return None
            elif operation_name in ["verify_user_email", "approve_user", "reject_user", "update_alert", "update_schedule", "delete_alert", "delete_schedule"]:
                return False
            elif operation_name in ["list_alerts", "list_schedules", "get_all_users", "list_users"]:
                return []
            elif operation_name in ["count_codes_last_hour"]:
                return 0
            elif operation_name in ["log_command_audit", "add_feedback"]:
                # Non-critical operations - return success to not block main operation
                return True
            else:
                # For operations that create resources (add_alert, add_schedule), we need to fail
                raise RuntimeError(f"Service not available for {operation_name}")

        # Implement retry logic for recoverable operations
        max_retries = 2 if operation_name in ["get_user_status", "list_alerts", "list_schedules"] else 1
        last_exception = None
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                result = operation_func(*args, **kwargs)

                # Log successful operation with timing and result summary
                elapsed_ms = int((time.time() - start_time) * 1000)
                result_summary = self._get_result_summary(result, operation_name)

                if attempt > 0:
                    _logger.info("Successfully completed %s after %d retries (took %dms): %s",
                               operation_context, attempt, elapsed_ms, result_summary)
                else:
                    _logger.debug("Completed %s successfully (took %dms): %s",
                                operation_context, elapsed_ms, result_summary)

                return result

            except Exception as e:
                last_exception = e
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Check if this is a retryable error
                error_msg = str(e).lower()
                is_retryable = (
                    "timeout" in error_msg or
                    "connection" in error_msg or
                    "busy" in error_msg or
                    "lock" in error_msg or
                    "temporary" in error_msg
                )

                # If this is the last attempt or error is not retryable, handle the error
                if attempt == max_retries - 1 or not is_retryable:
                    _logger.error("Failed %s after %d attempts (took %dms): %s",
                                operation_context, attempt + 1, elapsed_ms, e)
                    break

                # Log retry attempt
                _logger.warning("Retrying %s (attempt %d/%d) after error (took %dms): %s",
                              operation_context, attempt + 1, max_retries, elapsed_ms, e)

                # Brief delay before retry for connection/timeout issues
                if "connection" in error_msg or "timeout" in error_msg:
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff

        # Handle the final error
        error_result = self.handle_telegram_service_error(last_exception, operation_name, context)

        # Determine return value based on operation type and error recoverability
        if error_result.get("fallback_available", False):
            # For recoverable errors, return appropriate default values
            if operation_name in ["get_user_status", "get_user_limit", "get_alert", "get_schedule", "get_schedule_by_id"]:
                _logger.info("Returning None for %s due to recoverable error", operation_context)
                return None
            elif operation_name in ["verify_user_email", "approve_user", "reject_user", "update_alert", "update_schedule", "delete_alert", "delete_schedule"]:
                _logger.info("Returning False for %s due to recoverable error", operation_context)
                return False
            elif operation_name in ["list_alerts", "list_schedules", "get_all_users", "list_users"]:
                _logger.info("Returning empty list for %s due to recoverable error", operation_context)
                return []
            elif operation_name in ["count_codes_last_hour"]:
                _logger.info("Returning 0 for %s due to recoverable error", operation_context)
                return 0
            elif operation_name in ["log_command_audit", "add_feedback"]:
                # Non-critical operations - return success to not block main operation
                _logger.info("Returning True for non-critical operation %s", operation_context)
                return True

        # For non-recoverable errors or critical operations that create resources, raise the exception
        # For read operations, return appropriate defaults even for non-recoverable errors
        if operation_name in ["add_alert", "add_schedule", "set_verification_code", "verify_user_email", "approve_user"]:
            _logger.error("Raising exception for non-recoverable error in critical operation %s", operation_context)
            raise last_exception
        else:
            # For read operations, return safe defaults
            _logger.warning("Returning safe default for non-recoverable error in read operation %s", operation_context)
            if operation_name in ["get_user_status", "get_user_limit", "get_alert", "get_schedule", "get_schedule_by_id"]:
                return None
            elif operation_name in ["list_alerts", "list_schedules", "get_all_users", "list_users"]:
                return []
            elif operation_name in ["count_codes_last_hour"]:
                return 0
            else:
                return False

    async def safe_indicator_service_call(self, operation_func, operation_name: str, context: str = "", *args, **kwargs):
        """
        Safely call an IndicatorService method with comprehensive error handling and retry logic.

        Args:
            operation_func: The IndicatorService method to call
            operation_name: Name of the operation for error reporting
            context: Additional context for error handling
            *args, **kwargs: Arguments to pass to the operation function

        Returns:
            Result from the operation or error dict with fallback behavior
        """
        # Log service operation start with context information
        operation_context = f"{context}.{operation_name}" if context else operation_name
        _logger.debug("Starting indicator service operation: %s with args=%s, kwargs=%s",
                     operation_context, args, kwargs)

        # Check if service is available
        if not self.indicator_service:
            _logger.error("Indicator service not available for operation %s in %s", operation_name, context)
            return self.handle_indicator_service_error(
                RuntimeError("Indicator service not available"),
                f"{context}.{operation_name}"
            )

        # Implement retry logic for network-related operations
        max_retries = 3 if operation_name in ["compute_for_ticker", "compute"] else 1
        last_exception = None
        start_time = time.time()

        for attempt in range(max_retries):
            try:
                # Handle both async and sync operations
                if asyncio.iscoroutinefunction(operation_func):
                    result = await operation_func(*args, **kwargs)
                else:
                    result = operation_func(*args, **kwargs)

                # Log successful operation with timing and result summary
                elapsed_ms = int((time.time() - start_time) * 1000)
                result_summary = self._get_indicator_result_summary(result, operation_name)

                if attempt > 0:
                    _logger.info("Successfully completed %s after %d retries (took %dms): %s",
                               operation_context, attempt, elapsed_ms, result_summary)
                else:
                    _logger.debug("Completed %s successfully (took %dms): %s",
                                operation_context, elapsed_ms, result_summary)

                return result

            except Exception as e:
                last_exception = e
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Check if this is a retryable error
                error_msg = str(e).lower()
                is_retryable = (
                    "timeout" in error_msg or
                    "connection" in error_msg or
                    "network" in error_msg or
                    "rate limit" in error_msg or
                    "temporary" in error_msg or
                    "503" in error_msg or  # Service unavailable
                    "502" in error_msg or  # Bad gateway
                    "504" in error_msg     # Gateway timeout
                )

                # If this is the last attempt or error is not retryable, handle the error
                if attempt == max_retries - 1 or not is_retryable:
                    _logger.error("Failed %s after %d attempts (took %dms): %s",
                                operation_context, attempt + 1, elapsed_ms, e)
                    break

                # Log retry attempt
                _logger.warning("Retrying %s (attempt %d/%d) after error (took %dms): %s",
                              operation_context, attempt + 1, max_retries, elapsed_ms, e)

                # Implement exponential backoff for rate limits and network issues
                if "rate limit" in error_msg:
                    await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s delays
                elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
                    await asyncio.sleep(0.5 * (attempt + 1))  # 0.5s, 1s, 1.5s delays

        # Handle the final error with enhanced error information
        error_result = self.handle_indicator_service_error(last_exception, f"{context}.{operation_name}")

        # Add retry information to error result
        error_result["retry_attempts"] = max_retries
        error_result["final_attempt"] = True

        return error_result

    def _get_result_summary(self, result: Any, operation_name: str) -> str:
        """
        Generate a concise summary of service operation results for logging.

        Args:
            result: The result from the service operation
            operation_name: Name of the operation

        Returns:
            String summary of the result
        """
        try:
            if result is None:
                return "None"
            elif isinstance(result, bool):
                return str(result)
            elif isinstance(result, (int, float)):
                return str(result)
            elif isinstance(result, str):
                return f"'{result[:50]}...'" if len(result) > 50 else f"'{result}'"
            elif isinstance(result, list):
                return f"list[{len(result)} items]"
            elif isinstance(result, dict):
                if operation_name == "get_user_status" and "approved" in result:
                    return f"user(approved={result.get('approved')}, verified={result.get('verified')})"
                else:
                    return f"dict[{len(result)} keys]"
            else:
                return f"{type(result).__name__}"
        except Exception:
            return "unknown"

    def _get_indicator_result_summary(self, result: Any, operation_name: str) -> str:
        """
        Generate a concise summary of indicator service results for logging.

        Args:
            result: The result from the indicator service operation
            operation_name: Name of the operation

        Returns:
            String summary of the result
        """
        try:
            if hasattr(result, 'technical') and hasattr(result, 'fundamental'):
                # IndicatorResultSet
                tech_count = len(result.technical) if result.technical else 0
                fund_count = len(result.fundamental) if result.fundamental else 0
                return f"IndicatorResultSet(technical={tech_count}, fundamental={fund_count})"
            elif isinstance(result, dict):
                if "status" in result and result.get("status") == "error":
                    return f"error: {result.get('message', 'unknown')}"
                else:
                    return f"dict[{len(result)} keys]"
            elif isinstance(result, list):
                return f"list[{len(result)} items]"
            else:
                return f"{type(result).__name__}"
        except Exception:
            return "unknown"

    def extract_indicator_values(self, result_set: IndicatorResultSet) -> Dict[str, Any]:
        """
        Extract indicator values from IndicatorResultSet for telegram display.

        Args:
            result_set: Result from IndicatorService

        Returns:
            Dict with formatted indicator values for display
        """
        try:
            formatted_values = {}

            # Process technical indicators
            for name, indicator_value in result_set.technical.items():
                if indicator_value.value is not None:
                    formatted_values[name] = {
                        "value": indicator_value.value,
                        "type": "technical",
                        "formatted": self._format_indicator_value(name, indicator_value.value)
                    }

            # Process fundamental indicators
            for name, indicator_value in result_set.fundamental.items():
                if indicator_value.value is not None:
                    formatted_values[name] = {
                        "value": indicator_value.value,
                        "type": "fundamental",
                        "formatted": self._format_indicator_value(name, indicator_value.value)
                    }

            return formatted_values

        except Exception:
            _logger.exception("Error extracting indicator values:")
            return {}

    def _format_indicator_value(self, indicator_name: str, value: Any) -> str:
        """Format indicator value for display."""
        try:
            if isinstance(value, (int, float)):
                if indicator_name.upper() in ["RSI"]:
                    return f"{value:.2f}"
                elif indicator_name.upper() in ["PRICE", "SMA", "EMA"]:
                    return f"${value:.2f}"
                elif "PERCENT" in indicator_name.upper() or "%" in str(value):
                    return f"{value:.2f}%"
                else:
                    return f"{value:.4f}"
            else:
                return str(value)
        except Exception:
            return str(value)

    def handle_help(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /help and /start commands.
        Returns appropriate help text based on user admin status.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            is_admin = self.is_admin_user(telegram_user_id)

            # Import help texts here to avoid circular imports
            from src.telegram.telegram_bot import HELP_TEXT, ADMIN_HELP_TEXT

            # Show regular help text
            help_text = HELP_TEXT

            # Add admin commands if user is admin
            if is_admin:
                help_text += "\n\n" + ADMIN_HELP_TEXT

            return {
                "status": "ok",
                "help_text": help_text,
                "is_admin": is_admin
            }
        except Exception as e:
            _logger.exception("Error generating help")
            return {"status": "error", "message": f"Error generating help: {str(e)}"}

    def handle_register(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /register command.
        Register or update user email and send verification code.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            email = parsed.args.get("email")
            language = parsed.args.get("language", "en")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not email:
                return {"status": "error", "message": "Please provide an email address. Usage: /register email@example.com [language]"}

            # Validate email format
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return {"status": "error", "message": "Please provide a valid email address."}

            # Check rate limiting using service layer with error handling
            try:
                codes_sent = self.safe_telegram_service_call(
                    self.telegram_service.count_codes_last_hour,
                    "count_codes_last_hour",
                    "handle_register",
                    telegram_user_id
                )
                if codes_sent is None:
                    return {"status": "error", "message": "Unable to check rate limits. Please try again later."}

                if codes_sent >= 5:
                    return {"status": "error", "message": "Too many verification codes sent. Please wait an hour before requesting another."}
            except Exception as e:
                _logger.warning("Rate limit check failed for user %s: %s", telegram_user_id, e)
                # Continue with registration but log the issue
                codes_sent = 0

            # Generate verification code
            import random
            code = f"{random.randint(100000, 999999):06d}"
            sent_time = int(time.time())

            # Store user and code using service layer with comprehensive error handling
            try:
                self.safe_telegram_service_call(
                    self.telegram_service.set_verification_code,
                    "set_verification_code",
                    "handle_register",
                    telegram_user_id,
                    code=code,
                    sent_time=sent_time
                )
            except Exception as e:
                error_result = self.handle_telegram_service_error(e, "set_verification_code", "handle_register")
                return error_result

            # Send verification code via email
            # This will be handled by the notification system
            return {
                "status": "ok",
                "title": "Email Registration",
                "message": f"A 6-digit verification code has been sent to {email}. Use /verify CODE to verify your email.",
                "email_verification": {
                    "email": email,
                    "code": code,
                    "user_id": telegram_user_id
                }
            }

        except Exception:
            _logger.exception("Error in register command: ")
            return {"status": "error", "message": "Unable to process registration. Please try again later."}

    def handle_verify(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /verify command.
        Verify user email with the provided code.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            code = parsed.args.get("code")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not code:
                return {"status": "error", "message": "Please provide the verification code. Usage: /verify CODE"}

            # Validate code format
            if not code.isdigit() or len(code) != 6:
                return {"status": "error", "message": "Verification code must be a 6-digit number."}

            # Get user status using service layer with comprehensive error handling
            try:
                user_status = self.safe_telegram_service_call(
                    self.telegram_service.get_user_status,
                    "get_user_status",
                    "handle_verify",
                    telegram_user_id
                )
                if not user_status:
                    return {"status": "error", "message": "User not found. Please register first using /register."}
            except Exception as e:
                error_result = self.handle_telegram_service_error(e, "get_user_status", "handle_verify")
                return error_result

            # Check if code matches and is not expired
            stored_code = user_status.get("verification_code")
            code_sent_time = user_status.get("code_sent_time", 0)
            current_time = int(time.time())

            if stored_code == code and (current_time - code_sent_time) <= 3600:  # 1 hour expiry
                # Mark user as verified using service layer with comprehensive error handling
                try:
                    success = self.safe_telegram_service_call(
                        self.telegram_service.verify_user_email,
                        "verify_user_email",
                        "handle_verify",
                        telegram_user_id
                    )
                    if success:
                        return {
                            "status": "ok",
                            "title": "Email Verified",
                            "message": "Your email has been successfully verified! You can now use all bot features including email reports."
                        }
                    else:
                        return {"status": "error", "message": "Unable to verify email. Please try again later."}
                except Exception as e:
                    error_result = self.handle_telegram_service_error(e, "verify_user_email", "handle_verify")
                    return error_result
            else:
                return {
                    "status": "error",
                    "message": "Invalid or expired verification code. Please check the code or request a new one with /register."
                }

        except Exception:
            _logger.exception("Error in verify command: ")
            return {"status": "error", "message": "Unable to process verification. Please try again later."}

    def handle_info(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /info command.
        Display user information and status.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            # Get user status using service layer with error handling
            status = self.safe_telegram_service_call(
                self.telegram_service.get_user_status,
                "get_user_status",
                "handle_info",
                telegram_user_id
            )

            if status:
                email = status["email"] or "(not set)"
                verified = "Yes" if status["verified"] else "No"
                approved = "Yes" if status["approved"] else "No"
                admin = "Yes" if status["is_admin"] else "No"
                language = status["language"] or "(not set)"
                return {
                    "status": "ok",
                    "title": "Your Info",
                    "message": f"Email: {email}\nVerified: {verified}\nApproved: {approved}\nAdmin: {admin}\nLanguage: {language}"
                }
            else:
                return {
                    "status": "ok",
                    "title": "Your Info",
                    "message": "Email: (not set)\nVerified: No\nApproved: No\nAdmin: No\nLanguage: (not set)"
                }
        except Exception:
            _logger.exception("Error in info command: ")
            return {"status": "error", "message": "Unable to retrieve user information. Please try again later."}

    def handle_language(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /language command.
        Update user's language preference using service layer.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            language = parsed.args.get("language")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not language:
                return {"status": "error", "message": "Please provide a language code. Usage: /language en (supported: en, ru)"}

            # Validate language
            supported_languages = ["en", "ru"]
            if language.lower() not in supported_languages:
                return {"status": "error", "message": f"Language '{language}' not supported. Supported languages: {', '.join(supported_languages)}"}

            # Check if user has approved access
            access_check = self.check_approved_access(telegram_user_id)
            if access_check["status"] != "ok":
                return access_check

            # Get user status using service layer with error handling
            user_status = self.safe_telegram_service_call(
                self.telegram_service.get_user_status,
                "get_user_status",
                "handle_language",
                telegram_user_id
            )
            if not user_status:
                return {"status": "error", "message": "Please register first using /register email@example.com"}

            # Update language preference through service layer with error handling
            success = self.safe_telegram_service_call(
                self.telegram_service.update_user_language,
                "update_user_language",
                "handle_language",
                telegram_user_id,
                language.lower()
            )
            if not success:
                return {"status": "error", "message": "Unable to update language preference. Please try again later."}

            return {
                "status": "ok",
                "title": "Language Updated",
                "message": f"Your language preference has been updated to {language.upper()}."
            }

        except Exception:
            _logger.exception("Error in language command: ")
            return {"status": "error", "message": "Unable to update language preference. Please try again later."}

    def handle_feedback(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /feedback command with service layer error handling.
        Collects user feedback and forwards to administrators.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            feedback = parsed.args.get("feedback")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not feedback:
                return {"status": "error", "message": "Please provide feedback message. Usage: /feedback Your message here"}

            # Log feedback for admin review
            _logger.info("User feedback", extra={
                "user_id": telegram_user_id,
                "feedback": feedback,
                "type": "feedback"
            })

            # Store feedback in database for admin panel using service layer with error handling
            try:
                feedback_id = self.safe_telegram_service_call(
                    self.telegram_service.add_feedback,
                    "add_feedback",
                    "handle_feedback",
                    telegram_user_id,
                    "feedback",
                    feedback
                )

                if not feedback_id:
                    _logger.warning("Failed to store feedback in database for user %s", telegram_user_id)
                    # Continue anyway - feedback was logged
            except Exception as e:
                _logger.warning("Error storing feedback in database: %s", e)
                # Continue anyway - feedback was logged

            return {
                "status": "ok",
                "title": "Feedback Received",
                "message": "Thank you for your feedback! It has been forwarded to the development team.",
                "admin_notification": {
                    "type": "feedback",
                    "user_id": telegram_user_id,
                    "message": feedback
                }
            }

        except Exception:
            _logger.exception("Error processing feedback: ")
            return {"status": "error", "message": "Unable to process feedback. Please try again later."}

    def handle_feature(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /feature command with service layer error handling.
        Collects feature requests and forwards to administrators.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            feature_request = parsed.args.get("feature")

            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            if not feature_request:
                return {"status": "error", "message": "Please provide feature request. Usage: /feature Your feature idea here"}

            # Log feature request for admin review
            _logger.info("Feature request", extra={
                "user_id": telegram_user_id,
                "feature_request": feature_request,
                "type": "feature_request"
            })

            # Store feature request in database for admin panel using service layer with error handling
            try:
                feature_id = self.safe_telegram_service_call(
                    self.telegram_service.add_feedback,
                    "add_feedback",
                    "handle_feature",
                    telegram_user_id,
                    "feature_request",
                    feature_request
                )

                if not feature_id:
                    _logger.warning("Failed to store feature request in database for user %s", telegram_user_id)
                    # Continue anyway - request was logged
            except Exception as e:
                _logger.warning("Error storing feature request in database: %s", e)
                # Continue anyway - request was logged

            return {
                "status": "ok",
                "title": "Feature Request Received",
                "message": "Thank you for your feature request! It has been added to our development backlog.",
                "admin_notification": {
                    "type": "feature_request",
                    "user_id": telegram_user_id,
                    "message": feature_request
                }
            }

        except Exception:
            _logger.exception("Error processing feature request: ")
            return {"status": "error", "message": "Unable to process feature request. Please try again later."}

    async def handle_screener(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /screener command with comprehensive error handling.
        Handles immediate screener execution with indicator service integration.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            # Check if user has approved access
            access_check = self.check_approved_access(telegram_user_id)
            if access_check["status"] != "ok":
                return access_check

            # Delegate to standalone function for now, but with error handling
            # This maintains compatibility while adding service layer error handling
            try:
                # Import the standalone function
                from src.telegram.screener.business_logic import handle_screener as standalone_handle_screener
                result = await standalone_handle_screener(parsed)
                return result
            except Exception as e:
                _logger.exception("Error in screener command: ")
                error_msg = str(e).lower()
                if "indicator" in error_msg or "calculation" in error_msg:
                    return {"status": "error", "message": "Unable to calculate screening indicators. Please try again later."}
                elif "data" in error_msg or "provider" in error_msg:
                    return {"status": "error", "message": "Unable to fetch market data for screening. Please try again later."}
                else:
                    return {"status": "error", "message": "Screener temporarily unavailable. Please try again later."}

        except Exception:
            _logger.exception("Error in handle_screener: ")
            return {"status": "error", "message": "Unable to process screener request. Please try again later."}

    def handle_alerts(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /alerts commands with service layer error handling.
        Handles creating, listing, editing, and deleting price alerts.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            # Check if user has approved access
            access_check = self.check_approved_access(telegram_user_id)
            if access_check["status"] != "ok":
                return access_check

            # Delegate to standalone function for now, but with error handling
            # This maintains compatibility while adding service layer error handling
            try:
                # Import the standalone function
                from src.telegram.screener.business_logic import handle_alerts as standalone_handle_alerts
                result = standalone_handle_alerts(parsed)
                return result
            except Exception as e:
                _logger.exception("Error in alerts command: ")
                error_msg = str(e).lower()
                if "limit" in error_msg or "maximum" in error_msg:
                    return {"status": "error", "message": "Alert limit reached. Please delete some alerts first or contact admin."}
                elif "not found" in error_msg:
                    return {"status": "error", "message": "Alert not found or access denied."}
                else:
                    return {"status": "error", "message": "Unable to manage alerts. Please try again later."}

        except Exception:
            _logger.exception("Error in handle_alerts: ")
            return {"status": "error", "message": "Unable to process alerts request. Please try again later."}

    def handle_schedules(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /schedules commands with service layer error handling.
        Handles creating, listing, editing, and deleting scheduled reports.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            # Check if user has approved access
            access_check = self.check_approved_access(telegram_user_id)
            if access_check["status"] != "ok":
                return access_check

            # Delegate to standalone function for now, but with error handling
            # This maintains compatibility while adding service layer error handling
            try:
                # Import the standalone function
                from src.telegram.screener.business_logic import handle_schedules as standalone_handle_schedules
                result = standalone_handle_schedules(parsed)
                return result
            except Exception as e:
                _logger.exception("Error in schedules command: ")
                error_msg = str(e).lower()
                if "limit" in error_msg or "maximum" in error_msg:
                    return {"status": "error", "message": "Schedule limit reached. Please delete some schedules first or contact admin."}
                elif "not found" in error_msg:
                    return {"status": "error", "message": "Schedule not found or access denied."}
                else:
                    return {"status": "error", "message": "Unable to manage schedules. Please try again later."}

        except Exception:
            _logger.exception("Error in handle_schedules: ")
            return {"status": "error", "message": "Unable to process schedules request. Please try again later."}

    def handle_request_approval(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /request_approval command.
        User requests admin approval after email verification.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            status = self.telegram_service.get_user_status(telegram_user_id)

            if not status:
                return {"status": "error", "message": "Please register first using /register your@email.com"}

            if not status.get("verified", False):
                return {"status": "error", "message": "Please verify your email first using /verify CODE"}

            if status.get("approved", False):
                return {"status": "error", "message": "You are already approved for restricted features"}

            # Check if user already has a pending request (optional - could add a separate table for requests)
            # For now, we'll just notify admins about the request

            return {
                "status": "ok",
                "message": "Your approval request has been submitted. Admins will review your request and notify you of the decision.",
                "user_id": telegram_user_id,
                "email": status.get("email"),
                "notify_admins": True
            }
        except Exception as e:
            _logger.exception("Error processing approval request")
            return {"status": "error", "message": f"Error processing approval request: {str(e)}"}

    async def handle_report(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /report command.
        For each ticker:
          - Use analyze_ticker_business for unified analysis logic
          - Use format_ticker_report to generate message and chart
        """
        args = parsed.args

        # Check if user has approved access
        telegram_user_id = args.get("telegram_user_id")
        access_check = self.check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Check if JSON configuration is provided
        config_json = args.get("config")
        if config_json:
            # Validate and parse JSON configuration
            try:
                is_valid, errors = ReportConfigParser.validate_report_config(config_json)
                if not is_valid:
                    return {"status": "error", "title": "Report Error", "message": f"Invalid report configuration: {'; '.join(errors)}"}

                report_config = ReportConfigParser.parse_report_config(config_json)
                if not report_config:
                    return {"status": "error", "title": "Report Error", "message": "Failed to parse report configuration"}

                # Use configuration from JSON
                tickers = [t.upper() for t in report_config.tickers]
                period = report_config.period
                interval = report_config.interval
                provider = report_config.provider
                indicators = ",".join(report_config.indicators) if report_config.indicators else None
                email = report_config.email

            except Exception as e:
                _logger.exception("Error processing JSON configuration: %s", e)
                return {"status": "error", "title": "Report Error", "message": f"Error processing JSON configuration: {str(e)}"}
        else:
            # Use traditional command-line parameters
            tickers_raw = args.get("tickers")
            if isinstance(tickers_raw, str):
                tickers = [tickers_raw.upper()]
            elif isinstance(tickers_raw, list):
                tickers = [t.upper() for t in tickers_raw]
            else:
                tickers = [t.upper() for t in parsed.positionals]

            if not tickers:
                return {"status": "error", "title": "Report Error", "message": "No tickers specified"}

            period = args.get("period") or "2y"
            interval = args.get("interval") or "1d"
            provider = args.get("provider")
            indicators = args.get("indicators")
            email = args.get("email", False)

        reports = []

        # Fetch the registered email for the current user with error handling
        telegram_user_id = args.get("telegram_user_id")
        user_email = None
        if telegram_user_id:
            try:
                status = self.safe_telegram_service_call(
                    self.telegram_service.get_user_status,
                    "get_user_status",
                    "handle_report",
                    telegram_user_id
                )
                if status and status.get("email"):
                    user_email = status["email"]
            except Exception as e:
                _logger.warning("Failed to get user email for report in handle_report: %s", e)
                # Continue without email - this is not critical for report generation

        all_failed = True
        for ticker in tickers:
            analysis = await analyze_ticker_business(
                ticker=ticker,
                provider=provider,
                period=period,
                interval=interval
            )
            report = format_ticker_report(analysis)
            report['ticker'] = ticker
            report['error'] = analysis.error if analysis.error else None
            reports.append(report)
            if not analysis.error:
                all_failed = False

        # If all analyses failed due to missing keys
        if all_failed and any(report['error'] and any(key in report['error'] for key in ["Alpha Vantage API key", "Finnhub API key", "Twelve Data API key", "Polygon.io API key"]) for report in reports):
            return {
                "status": "error",
                "title": "Report Error",
                "message": f"No data could be retrieved for {', '.join(tickers)}. Missing or invalid API keys for 1 or more providers. Please check your API keys in donotshare.py."
            }
        # If all analyses failed for any reason
        if all_failed:
            return {
                "status": "error",
                "title": "Report Error",
                "message": f"No data could be retrieved for {', '.join(tickers)}. Please check your API keys or try a different provider/ticker."
            }
        # Otherwise, return reports for Telegram/email delivery
        return {
            "status": "ok",
            "reports": reports,
            "email": email,
            "user_email": user_email,
            "title": f"Report for {', '.join(tickers)}",
            "message": "Report generated successfully."
        }

    def handle_admin(self, parsed: ParsedCommand) -> Dict[str, Any]:
        """
        Business logic for /admin commands using service layer.
        Handles user management, system settings, and administrative functions.
        """
        try:
            telegram_user_id = parsed.args.get("telegram_user_id")
            if not telegram_user_id:
                return {"status": "error", "message": "No telegram_user_id provided"}

            # Check if user has admin access
            access_check = self.check_admin_access(telegram_user_id)
            if access_check["status"] != "ok":
                return access_check

            # Get action and parameters from positionals
            action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
            params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

            if not action:
                return {
                    "status": "error",
                    "title": "Admin Help",
                    "message": ("Available admin commands:\n"
                               "/admin users - List all registered users\n"
                               "/admin listusers - List users as telegram_user_id - email pairs\n"
                               "/admin pending - List users waiting for approval\n"
                               "/admin approve USER_ID - Approve user for restricted features\n"
                               "/admin reject USER_ID - Reject user's approval request\n"
                               "/admin verify USER_ID - Manually verify user's email\n"
                               "/admin resetemail USER_ID - Reset user's email\n"
                               "/admin setlimit alerts N [USER_ID] - Set max alerts (global or per-user)\n"
                               "/admin setlimit schedules N [USER_ID] - Set max schedules (global or per-user)\n"
                               "/admin broadcast MESSAGE - Send broadcast message to all users")
                }

            if action == "users":
                return self._handle_admin_list_users(telegram_user_id)
            elif action == "listusers":
                return self._handle_admin_list_users(telegram_user_id)
            elif action == "resetemail" and len(params) >= 1:
                return self._handle_admin_reset_email(telegram_user_id, params[0])
            elif action == "verify" and len(params) >= 1:
                return self._handle_admin_verify_user(telegram_user_id, params[0])
            elif action == "approve" and len(params) >= 1:
                return self._handle_admin_approve_user(telegram_user_id, params[0])
            elif action == "reject" and len(params) >= 1:
                return self._handle_admin_reject_user(telegram_user_id, params[0])
            elif action == "pending":
                return self._handle_admin_list_pending_approvals(telegram_user_id)
            elif action == "setlimit" and len(params) >= 2:
                limit_type = params[0]  # "alerts" or "schedules"
                limit_value = params[1]
                target_user_id = params[2] if len(params) > 2 else None
                return self._handle_admin_set_limit(telegram_user_id, limit_type, limit_value, target_user_id)
            elif action == "broadcast" and len(params) >= 1:
                message = " ".join(params)
                return self._handle_admin_schedule_broadcast(telegram_user_id, message, "now")
            else:
                return {"status": "error", "message": f"Unknown admin command: {action}"}

        except Exception as e:
            _logger.exception("Error in admin command")
            return {"status": "error", "message": f"Error in admin command: {str(e)}"}

    def _handle_admin_list_users(self, admin_telegram_user_id: str) -> Dict[str, Any]:
        """List all users for admin review using service layer with comprehensive error handling."""
        try:
            # Get all users using service layer with error handling
            users = self.safe_telegram_service_call(
                self.telegram_service.get_all_users,
                "get_all_users",
                "_handle_admin_list_users"
            )

            if users is None:
                return {"status": "error", "message": "Unable to retrieve user list due to service error. Please try again later."}

            if not users:
                return {"status": "ok", "message": "No users found", "is_admin": True}

            # Format user list with error handling for individual user data
            user_list = []
            for user in users:
                try:
                    email = user.get('email', 'N/A')
                    verified = user.get("verified", False)
                    approved = user.get("approved", False)

                    status_text = "✅ Verified & Approved" if verified and approved else \
                                 "✅ Verified" if verified else "❌ Not Verified"
                    user_list.append(f"• {email} - {status_text}")
                except Exception as user_error:
                    _logger.warning("Error formatting user data: %s", user_error)
                    user_list.append("• [Error displaying user data]")

            return {
                "status": "ok",
                "message": f"**User List ({len(users)} users)**\n\n" + "\n".join(user_list),
                "is_admin": True
            }

        except Exception as e:
            error_result = self.handle_telegram_service_error(e, "get_all_users", "_handle_admin_list_users")
            return error_result

    def _handle_admin_list_pending_approvals(self, admin_telegram_user_id: str) -> Dict[str, Any]:
        """List users pending approval using service layer."""
        try:
            # Get all users using service layer with error handling
            users = self.safe_telegram_service_call(
                self.telegram_service.get_all_users,
                "get_all_users",
                "_handle_admin_list_pending_approvals"
            )

            if not users:
                return {"status": "ok", "message": "No users found"}

            # Filter for verified but not approved users
            pending_users = [user for user in users if user.get("verified") and not user.get("approved")]

            if not pending_users:
                return {"status": "ok", "message": "No users pending approval"}

            # Format pending user list
            user_list = []
            for user in pending_users:
                user_list.append(f"• {user.get('email', 'N/A')} (ID: {user.get('telegram_user_id')})")

            return {
                "status": "ok",
                "message": "**Users Pending Approval**\n\n" + "\n".join(user_list),
                "is_admin": True
            }

        except Exception:
            _logger.exception("Error listing pending approvals")
            return {"status": "error", "message": "Unable to retrieve pending approvals. Please try again later."}

    def _handle_admin_reset_email(self, admin_telegram_user_id: str, user_id: str) -> Dict[str, Any]:
        """Reset user's email verification status using service layer."""
        try:
            success = self.telegram_service.reset_user_email_verification(user_id)

            if success:
                return {
                    "status": "ok",
                    "message": f"Email verification reset for user {user_id}",
                    "is_admin": True
                }
            else:
                return {"status": "error", "message": f"Failed to reset email for user {user_id}"}

        except Exception as e:
            _logger.exception("Error resetting email")
            return {"status": "error", "message": f"Error resetting email: {str(e)}"}

    def _handle_admin_verify_user(self, admin_telegram_user_id: str, user_id: str) -> Dict[str, Any]:
        """Verify a user's email using service layer."""
        try:
            success = self.telegram_service.verify_user_email(user_id)

            if success:
                return {
                    "status": "ok",
                    "message": f"User {user_id} verified successfully",
                    "is_admin": True
                }
            else:
                return {"status": "error", "message": f"Failed to verify user {user_id}"}

        except Exception as e:
            _logger.exception("Error verifying user")
            return {"status": "error", "message": f"Error verifying user: {str(e)}"}

    def _handle_admin_approve_user(self, admin_telegram_user_id: str, user_id: str) -> Dict[str, Any]:
        """Approve a user for restricted features using service layer."""
        try:
            success = self.telegram_service.approve_user(user_id)

            if success:
                return {
                    "status": "ok",
                    "message": f"User {user_id} approved successfully",
                    "is_admin": True
                }
            else:
                return {"status": "error", "message": f"Failed to approve user {user_id}"}

        except Exception as e:
            _logger.exception("Error approving user")
            return {"status": "error", "message": f"Error approving user: {str(e)}"}

    def _handle_admin_reject_user(self, admin_telegram_user_id: str, user_id: str) -> Dict[str, Any]:
        """Reject a user's approval request using service layer."""
        try:
            success = self.telegram_service.reject_user(user_id)

            if success:
                return {
                    "status": "ok",
                    "message": f"User {user_id} rejected",
                    "is_admin": True
                }
            else:
                return {"status": "error", "message": f"Failed to reject user {user_id}"}

        except Exception as e:
            _logger.exception("Error rejecting user")
            return {"status": "error", "message": f"Error rejecting user: {str(e)}"}

    def _handle_admin_set_limit(self, admin_telegram_user_id: str, limit_type: str, limit_value: str, target_user_id: str = None) -> Dict[str, Any]:
        """Set user limits using service layer."""
        try:
            if limit_type not in ["alerts", "schedules"]:
                return {"status": "error", "message": "Limit type must be 'alerts' or 'schedules'"}

            try:
                limit = int(limit_value)
            except ValueError:
                return {"status": "error", "message": "Limit must be a number"}

            # Map limit type to service method parameter
            limit_key = f"max_{limit_type}"

            if target_user_id:
                # Set limit for specific user
                self.telegram_service.set_user_limit(target_user_id, limit_key, limit)
                return {
                    "status": "ok",
                    "message": f"{limit_type.capitalize()} limit set to {limit} for user {target_user_id}",
                    "is_admin": True
                }
            else:
                # Set global default limit (this would need to be implemented in service layer)
                # For now, return an error message
                return {"status": "error", "message": "Global limit setting not yet implemented"}

        except Exception as e:
            _logger.exception("Error setting limit")
            return {"status": "error", "message": f"Error setting limit: {str(e)}"}

    def _handle_admin_schedule_broadcast(self, admin_telegram_user_id: str, message: str, scheduled_time: str) -> Dict[str, Any]:
        """Schedule a broadcast message using service layer."""
        try:
            success = self.telegram_service.schedule_broadcast(message, scheduled_time, admin_telegram_user_id)

            if success:
                return {
                    "status": "ok",
                    "message": f"Broadcast scheduled for {scheduled_time}",
                    "is_admin": True
                }
            else:
                return {"status": "error", "message": "Failed to schedule broadcast"}

        except Exception as e:
            _logger.exception("Error scheduling broadcast")
            return {"status": "error", "message": f"Error scheduling broadcast: {str(e)}"}


def is_admin_user(telegram_user_id: str) -> bool:
    """Check if user is an admin using service layer."""
    telegram_svc, _ = get_service_instances()
    if not telegram_svc:
        return False
    status = telegram_svc.get_user_status(telegram_user_id)
    return status and status.get("is_admin", False)

def is_approved_user(telegram_user_id: str) -> bool:
    """Check if user is approved for restricted features using service layer."""
    telegram_svc, _ = get_service_instances()
    if not telegram_svc:
        return False
    status = telegram_svc.get_user_status(telegram_user_id)
    return status and status.get("approved", False)

def check_admin_access(telegram_user_id: str) -> Dict[str, Any]:
    """Check if user has admin access. Returns error dict if not."""
    if not is_admin_user(telegram_user_id):
        return {"status": "error", "message": "Access denied. Admin privileges required."}
    return {"status": "ok"}

def check_approved_access(telegram_user_id: str) -> Dict[str, Any]:
    """Check if user has approved access for restricted features. Returns error dict if not."""
    if not is_approved_user(telegram_user_id):
        return {"status": "error", "message": "Access denied, please contact chat's admin or send request for approval using command /request_approval"}
    return {"status": "ok"}


def handle_request_approval(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /request_approval command.
    User requests admin approval after email verification.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        status = telegram_svc.get_user_status(telegram_user_id)

        if not status:
            return {"status": "error", "message": "Please register first using /register your@email.com"}

        if not status.get("verified", False):
            return {"status": "error", "message": "Please verify your email first using /verify CODE"}

        if status.get("approved", False):
            return {"status": "error", "message": "You are already approved for restricted features"}

        # Check if user already has a pending request (optional - could add a separate table for requests)
        # For now, we'll just notify admins about the request

        return {
            "status": "ok",
            "message": "Your approval request has been submitted. Admins will review your request and notify you of the decision.",
            "user_id": telegram_user_id,
            "email": status.get("email"),
            "notify_admins": True
        }
    except Exception as e:
        _logger.exception("Error processing approval request")
        return {"status": "error", "message": f"Error processing approval request: {str(e)}"}

async def handle_report(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /report command.
    For each ticker:
      - Use analyze_ticker_business for unified analysis logic
      - Use format_ticker_report to generate message and chart
    """
    args = parsed.args

    # Check if user has approved access
    telegram_user_id = args.get("telegram_user_id")
    access_check = check_approved_access(telegram_user_id)
    if access_check["status"] != "ok":
        return access_check

    # Check if JSON configuration is provided
    config_json = args.get("config")
    if config_json:
        # Validate and parse JSON configuration
        try:
            is_valid, errors = ReportConfigParser.validate_report_config(config_json)
            if not is_valid:
                return {"status": "error", "title": "Report Error", "message": f"Invalid report configuration: {'; '.join(errors)}"}

            report_config = ReportConfigParser.parse_report_config(config_json)
            if not report_config:
                return {"status": "error", "title": "Report Error", "message": "Failed to parse report configuration"}

            # Use configuration from JSON
            tickers = [t.upper() for t in report_config.tickers]
            period = report_config.period
            interval = report_config.interval
            provider = report_config.provider
            indicators = ",".join(report_config.indicators) if report_config.indicators else None
            email = report_config.email

        except Exception as e:
            _logger.exception("Error processing JSON configuration: %s", e)
            return {"status": "error", "title": "Report Error", "message": f"Error processing JSON configuration: {str(e)}"}
    else:
        # Use traditional command-line parameters
        tickers_raw = args.get("tickers")
        if isinstance(tickers_raw, str):
            tickers = [tickers_raw.upper()]
        elif isinstance(tickers_raw, list):
            tickers = [t.upper() for t in tickers_raw]
        else:
            tickers = [t.upper() for t in parsed.positionals]

        if not tickers:
            return {"status": "error", "title": "Report Error", "message": "No tickers specified"}

        period = args.get("period") or "2y"
        interval = args.get("interval") or "1d"
        provider = args.get("provider")
        indicators = args.get("indicators")
        email = args.get("email", False)

    reports = []

    # Fetch the registered email for the current user
    telegram_user_id = args.get("telegram_user_id")
    user_email = None
    if telegram_user_id:
        telegram_svc, _ = get_service_instances()
        if telegram_svc:
            status = telegram_svc.get_user_status(telegram_user_id)
            if status and status.get("email"):
                user_email = status["email"]

    all_failed = True
    for ticker in tickers:
        analysis = await analyze_ticker_business(
            ticker=ticker,
            provider=provider,
            period=period,
            interval=interval
        )
        report = format_ticker_report(analysis)
        report['ticker'] = ticker
        report['error'] = analysis.error if analysis.error else None
        reports.append(report)
        if not analysis.error:
            all_failed = False
    # If all analyses failed due to missing keys
    if all_failed and any(report['error'] and any(key in report['error'] for key in ["Alpha Vantage API key", "Finnhub API key", "Twelve Data API key", "Polygon.io API key"]) for report in reports):
        return {
            "status": "error",
            "title": "Report Error",
            "message": f"No data could be retrieved for {', '.join(tickers)}. Missing or invalid API keys for 1 or more providers. Please check your API keys in donotshare.py."
        }
    # If all analyses failed for any reason
    if all_failed:
        return {
            "status": "error",
            "title": "Report Error",
            "message": f"No data could be retrieved for {', '.join(tickers)}. Please check your API keys or try a different provider/ticker."
        }
    # Otherwise, return reports for Telegram/email delivery
    return {
        "status": "ok",
        "reports": reports,
        "email": email,
        "user_email": user_email,
        "title": f"Report for {', '.join(tickers)}",
        "message": "Report generated successfully."
    }


async def analyze_ticker_business(
    ticker: str,
    provider: str = None,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = True,
    force_refresh_fundamentals: bool = False
) -> TickerAnalysis:
    """
    Business logic: fetch OHLCV for ticker/provider/period/interval, return TickerAnalysis.
    Uses common functions from src/common for data retrieval and analysis.

    Args:
        ticker: Stock or crypto ticker symbol
        provider: Data provider code (optional, auto-selected if None)
        period: Time period for historical data (default: "2y")
        interval: Data interval (default: "1d")
        force_refresh: If True, bypass cache for OHLCV (default: True for current prices)
        force_refresh_fundamentals: If True, bypass cache for fundamentals (default: False)
            Fundamentals use TTL-based caching and don't need forced refresh

    Returns:
        TickerAnalysis object with complete analysis
    """
    try:
        # Use the analyze_ticker function from src.common.ticker_analyzer
        # This ensures that indicators are properly calculated and added to the DataFrame
        # Force refresh OHLCV by default for current prices, but respect fundamentals TTL
        analysis = await analyze_ticker(
            ticker=ticker,
            provider=provider,
            period=period,
            interval=interval,
            force_refresh=force_refresh,
            force_refresh_fundamentals=force_refresh_fundamentals
        )

        return analysis

    except Exception as e:
        _logger.exception("Error in analyze_ticker_business for %s: %s", ticker, e)
        return TickerAnalysis(
            ticker=ticker.upper(),
            provider=provider or determine_provider(ticker),
            period=period,
            interval=interval,
            ohlcv=None,
            fundamentals=None,
            technicals=None,
            current_price=None,
            change_percentage=None,
            error=str(e),
            chart_image=None
        )


def handle_info(parsed: ParsedCommand) -> Dict[str, Any]:
    telegram_user_id = parsed.args.get("telegram_user_id")
    if not telegram_user_id:
        return {"status": "error", "message": "No telegram_user_id provided"}

    telegram_svc, _ = get_service_instances()
    if not telegram_svc:
        return {"status": "error", "message": "Service temporarily unavailable"}

    status = telegram_svc.get_user_status(telegram_user_id)
    if status:
        email = status["email"] or "(not set)"
        verified = "Yes" if status["verified"] else "No"
        approved = "Yes" if status["approved"] else "No"
        admin = "Yes" if status["is_admin"] else "No"
        language = status["language"] or "(not set)"
        return {
            "status": "ok",
            "title": "Your Info",
            "message": f"Email: {email}\nVerified: {verified}\nApproved: {approved}\nAdmin: {admin}\nLanguage: {language}"
        }
    else:
        return {
            "status": "ok",
            "title": "Your Info",
            "message": "Email: (not set)\nVerified: No\nApproved: No\nAdmin: No\nLanguage: (not set)"
        }


def handle_admin(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /admin commands using service-aware business logic.
    This function delegates to the TelegramBusinessLogic class for proper service layer usage.
    """
    try:
        # Get service instances
        telegram_svc, indicator_svc = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        # Create business logic instance with service dependencies
        business_logic = TelegramBusinessLogic(telegram_svc, indicator_svc)

        # Delegate to the service-aware business logic class
        return business_logic.handle_admin(parsed)

    except Exception as e:
        _logger.exception("Error in admin command")
        return {"status": "error", "message": f"Error in admin command: {str(e)}"}


def handle_alerts(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /alerts commands.
    Handles creating, listing, editing, and deleting price alerts.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            # List all alerts for user
            return handle_alerts_list(telegram_user_id)

        if action == "add" and len(params) >= 3:
            ticker, price, condition = params[0], params[1], params[2]
            # Get email flag from parsed args
            email = parsed.args.get("email", False)
            return handle_alerts_add(telegram_user_id, ticker, price, condition, email)
        elif action == "add_indicator" and len(params) >= 2:
            ticker, config_json = params[0], params[1]
            # Get additional parameters from parsed args
            email = parsed.args.get("email", False)
            timeframe = parsed.args.get("timeframe", "15m")
            alert_action = parsed.args.get("action_type", "notify")
            return handle_alerts_add_indicator(telegram_user_id, ticker, config_json, timeframe, alert_action, email)
        elif action == "edit" and len(params) >= 1:
            alert_id = params[0]
            new_price = params[1] if len(params) > 1 else None
            new_condition = params[2] if len(params) > 2 else None
            # Get email flag from parsed args
            email = parsed.args.get("email")
            return handle_alerts_edit(telegram_user_id, alert_id, new_price, new_condition, email)
        elif action == "delete" and len(params) >= 1:
            alert_id = params[0]
            return handle_alerts_delete(telegram_user_id, alert_id)
        elif action == "pause" and len(params) >= 1:
            alert_id = params[0]
            return handle_alerts_pause(telegram_user_id, alert_id)
        elif action == "resume" and len(params) >= 1:
            alert_id = params[0]
            return handle_alerts_resume(telegram_user_id, alert_id)
        else:
            return {
                "status": "error",
                "title": "Alerts Help",
                "message": ("Available alert commands:\n"
                           "/alerts - List all alerts\n"
                           "/alerts add TICKER PRICE CONDITION [flags] - Add price alert\n"
                           "  CONDITION: above or below\n"
                           "  Example: /alerts add BTCUSDT 65000 above -email\n"
                           "/alerts add_indicator TICKER CONFIG_JSON [flags] - Add indicator alert\n"
                           "  Example: /alerts add_indicator AAPL '{\"type\":\"indicator\",\"indicator\":\"RSI\",\"parameters\":{\"period\":14},\"condition\":{\"operator\":\"<\",\"value\":30},\"alert_action\":\"BUY\",\"timeframe\":\"15m\"}' -email\n"
                           "Flags:\n"
                           "  -email: Send alert notification to email\n"
                           "  -timeframe=15m: Set timeframe (5m, 15m, 1h, 4h, 1d)\n"
                           "  -action_type=notify: Set action (BUY, SELL, HOLD, notify)\n"
                           "/alerts edit ALERT_ID [PRICE] [CONDITION] [flags] - Edit alert\n"
                           "  Example: /alerts edit 1 70000 below -email\n"
                           "/alerts delete ALERT_ID - Delete alert\n"
                           "/alerts pause ALERT_ID - Pause alert\n"
                           "/alerts resume ALERT_ID - Resume alert\n\n"
                           "Indicator Alert Examples:\n"
                           "• RSI oversold: {\"type\":\"indicator\",\"indicator\":\"RSI\",\"parameters\":{\"period\":14},\"condition\":{\"operator\":\"<\",\"value\":30}}\n"
                           "• Bollinger Bands: {\"type\":\"indicator\",\"indicator\":\"BollingerBands\",\"parameters\":{\"period\":20},\"condition\":{\"operator\":\"below_lower_band\"}}\n"
                           "• MACD crossover: {\"type\":\"indicator\",\"indicator\":\"MACD\",\"condition\":{\"operator\":\"crossover\"}}")
            }

    except Exception as e:
        _logger.exception("Error in alerts command: ")
        return {"status": "error", "message": f"Error processing alerts command: {str(e)}"}


def handle_alerts_list(telegram_user_id: str) -> Dict[str, Any]:
    """List all alerts for a user."""
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        alerts = telegram_svc.list_alerts(telegram_user_id)
        if not alerts:
            return {"status": "ok", "title": "Your Alerts", "message": "You have no active alerts."}

        alert_list = []
        for alert in alerts:
            status = "🟢 Active" if alert.get("active") else "🔴 Paused"
            email_flag = "📧" if alert.get("email") else "💬"

            # Handle different alert types
            alert_type = alert.get("alert_type", "price")
            if alert_type == "price":
                alert_list.append(
                    f"#{alert['id']}: {alert['ticker']} {alert['condition']} ${alert['price']:.2f} {email_flag} - {status}"
                )
            else:
                # Indicator alert - simplified since alert_logic_evaluator was removed
                summary = {"indicators": [alert.get('condition', 'Unknown condition')]}
                alert_type_icon = "📊" if alert_type == "indicator" else "❓"
                timeframe = alert.get("timeframe", "15m")
                action = alert.get("alert_action", "notify")

                if "indicators" in summary:
                    indicators_text = ", ".join(summary["indicators"])
                    alert_list.append(
                        f"#{alert['id']}: {alert['ticker']} {alert_type_icon} {indicators_text} ({timeframe}, {action}) {email_flag} - {status}"
                    )
                else:
                    alert_list.append(
                        f"#{alert['id']}: {alert['ticker']} {alert_type_icon} Indicator Alert ({timeframe}, {action}) {email_flag} - {status}"
                    )

        message = f"Your alerts ({len(alerts)}):\n\n" + "\n".join(alert_list)
        return {"status": "ok", "title": "Your Alerts", "message": message}

    except Exception as e:
        _logger.exception("Error listing alerts: ")
        return {"status": "error", "message": f"Error listing alerts: {str(e)}"}


def handle_alerts_add(telegram_user_id: str, ticker: str, price_str: str, condition: str, email: bool = False) -> Dict[str, Any]:
    """Add a new price alert with re-arm functionality."""
    try:
        # Validate condition
        if condition.lower() not in ["above", "below"]:
            return {"status": "error", "message": "Condition must be 'above' or 'below'"}

        # Validate price
        try:
            price = float(price_str)
            if price <= 0:
                raise ValueError("Price must be positive")
        except ValueError:
            return {"status": "error", "message": "Price must be a positive number"}

        # Check user limits using service layer with comprehensive error handling
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        try:
            user_status = telegram_svc.get_user_status(telegram_user_id)
            if not user_status:
                return {"status": "error", "message": "User not found. Please register first using /register."}

            max_alerts = user_status.get("max_alerts", 5)

            current_alerts = telegram_svc.list_alerts(telegram_user_id)
            if current_alerts is None:
                return {"status": "error", "message": "Unable to check current alerts. Please try again later."}

            if len(current_alerts) >= max_alerts:
                return {
                    "status": "error",
                    "message": f"Alert limit reached ({max_alerts}). Delete some alerts first or contact admin."
                }
        except Exception:
            _logger.exception("Error checking user limits for alerts:")
            return {"status": "error", "message": "Unable to verify alert limits. Please try again later."}

        # Create simple alert configuration (enhanced alert system moved to scheduler)
        try:
            enhanced_config = {
                "ticker": ticker.upper(),
                "threshold": price,
                "direction": condition.lower(),
                "email": email,
                "rearm_enabled": True
            }
        except Exception:
            _logger.exception("Error creating enhanced alert configuration:")
            return {"status": "error", "message": "Unable to create alert configuration. Please try again."}

        # Add the alert with enhanced configuration and comprehensive error handling
        try:
            alert_id = telegram_svc.add_alert(telegram_user_id, ticker.upper(), price, condition.lower(), email)
            if not alert_id:
                return {"status": "error", "message": "Failed to create alert. Please try again later."}
        except Exception as e:
            _logger.exception("Error adding alert to database:")
            error_msg = str(e).lower()
            if "limit" in error_msg or "maximum" in error_msg:
                return {"status": "error", "message": "Alert limit reached. Please delete some alerts first."}
            elif "duplicate" in error_msg or "exists" in error_msg:
                return {"status": "error", "message": "Similar alert already exists. Please check your existing alerts."}
            else:
                return {"status": "error", "message": "Unable to create alert. Please try again later."}

        # Update with re-arm configuration with error handling
        try:
            telegram_svc.update_alert(
                alert_id,
                re_arm_config=enhanced_config.to_json(),
                is_armed=True,
                last_price=None,
                last_triggered_at=None
            )
        except Exception as e:
            _logger.warning("Error updating alert with re-arm configuration: %s", e)
            # Alert was created successfully, just log the re-arm config error
            # Don't fail the entire operation

        email_text = " and email" if email else ""
        rearm_level = enhanced_config.re_arm_config.hysteresis
        rearm_type = enhanced_config.re_arm_config.hysteresis_type

        return {
            "status": "ok",
            "title": "Re-Arm Alert Added",
            "message": (f"Alert #{alert_id} created: {ticker.upper()} {condition.lower()} ${price:.2f}{email_text}\n"
                       f"🔄 Re-arm enabled with {rearm_level}{'' if rearm_type == 'fixed' else '%'} hysteresis\n"
                       f"Alert will re-trigger when price crosses back and forth across threshold.")
        }

    except Exception as e:
        _logger.exception("Error adding alert: ")
        return {"status": "error", "message": f"Error adding alert: {str(e)}"}


def handle_alerts_edit(telegram_user_id: str, alert_id_str: str, new_price_str: str = None, new_condition: str = None, email: bool = None) -> Dict[str, Any]:
    """Edit an existing alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        alert = telegram_svc.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        updates = {}

        # Validate and set new price
        if new_price_str:
            try:
                new_price = float(new_price_str)
                if new_price <= 0:
                    raise ValueError("Price must be positive")
                updates["price"] = new_price
            except ValueError:
                return {"status": "error", "message": "Price must be a positive number"}

        # Validate and set new condition
        if new_condition:
            if new_condition.lower() not in ["above", "below"]:
                return {"status": "error", "message": "Condition must be 'above' or 'below'"}
            updates["condition"] = new_condition.lower()

        # Validate and set email flag
        if email is not None:
            updates["email"] = 1 if email else 0

        if not updates:
            return {"status": "error", "message": "No updates provided. Specify new price, condition, and/or email flag."}

        # Update the alert using service layer
        telegram_svc.update_alert(alert_id, **updates)

        # Get updated alert for confirmation
        updated_alert = telegram_svc.get_alert(alert_id)

        return {
            "status": "ok",
            "title": "Alert Updated",
            "message": f"Alert #{alert_id} updated: {updated_alert['ticker']} {updated_alert['condition']} ${updated_alert['price']:.2f}"
        }

    except Exception as e:
        _logger.exception("Error editing alert: ")
        return {"status": "error", "message": f"Error editing alert: {str(e)}"}


def handle_alerts_add_indicator(telegram_user_id: str, ticker: str, config_json: str, timeframe: str = "15m",
                               alert_action: str = "notify", email: bool = False) -> Dict[str, Any]:
    """Add a new indicator-based alert."""
    try:
        # Validate ticker
        if not ticker or len(ticker.strip()) == 0:
            return {"status": "error", "message": "Ticker is required"}

        # Validate timeframe
        valid_timeframes = ["5m", "15m", "1h", "4h", "1d"]
        if timeframe not in valid_timeframes:
            return {"status": "error", "message": f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"}

        # Validate alert action
        valid_actions = ["BUY", "SELL", "HOLD", "notify"]
        if alert_action not in valid_actions:
            return {"status": "error", "message": f"Invalid action. Must be one of: {', '.join(valid_actions)}"}

        # Simple validation (alert_config_parser moved to scheduler)
        try:
            # Basic validation - ensure required fields exist
            required_fields = ["ticker", "condition", "threshold"]
            is_valid = all(field in config_json for field in required_fields)
            errors = [f"Missing field: {field}" for field in required_fields if field not in config_json]
            if not is_valid:
                return {"status": "error", "message": f"Invalid alert configuration: {'; '.join(errors)}"}
        except Exception as e:
            return {"status": "error", "message": f"Error validating alert configuration: {str(e)}"}

        # Check user limits using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        user_status = telegram_svc.get_user_status(telegram_user_id)
        max_alerts = user_status.get("max_alerts", 5)
        current_alerts = len(telegram_svc.list_alerts(telegram_user_id))

        if current_alerts >= max_alerts:
            return {
                "status": "error",
                "message": f"Alert limit reached ({max_alerts}). Delete some alerts first or contact admin."
            }

        # Add the indicator alert using service layer
        alert_id = telegram_svc.add_indicator_alert(
            telegram_user_id=telegram_user_id,
            ticker=ticker.upper(),
            indicator="custom",  # Will be parsed from config_json
            condition=config_json,  # Full config as condition
            value=0.0,  # Placeholder
            timeframe=timeframe,
            alert_action=alert_action,
            email=email
        )

        # Simple alert summary (alert_logic_evaluator moved to scheduler)
        alert_data = {
            "id": alert_id,
            "ticker": ticker.upper(),
            "alert_type": "indicator",
            "config_json": config_json,
            "timeframe": timeframe,
            "alert_action": alert_action
        }
        summary = {"indicators": [f"{ticker.upper()} {config_json.get('condition', 'alert')}"]}  # Simplified summary

        email_text = " and email" if email else ""
        return {
            "status": "ok",
            "title": "Indicator Alert Added",
            "message": f"Alert #{alert_id} created: {ticker.upper()} - {summary.get('type', 'Indicator Alert')} ({timeframe}){email_text}"
        }

    except Exception as e:
        _logger.exception("Error adding indicator alert: ")
        return {"status": "error", "message": f"Error adding indicator alert: {str(e)}"}


def handle_alerts_delete(telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
    """Delete an alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        alert = telegram_svc.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        # Delete the alert using service layer
        telegram_svc.delete_alert(alert_id)

        return {
            "status": "ok",
            "title": "Alert Deleted",
            "message": f"Alert #{alert_id} for {alert['ticker']} has been deleted."
        }

    except Exception as e:
        _logger.exception("Error deleting alert: ")
        return {"status": "error", "message": f"Error deleting alert: {str(e)}"}


def handle_alerts_pause(telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
    """Pause an alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        alert = telegram_svc.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        # Pause the alert using service layer
        telegram_svc.update_alert(alert_id, active=False)

        return {
            "status": "ok",
            "title": "Alert Paused",
            "message": f"Alert #{alert_id} for {alert['ticker']} has been paused."
        }

    except Exception as e:
        _logger.exception("Error pausing alert: ")
        return {"status": "error", "message": f"Error pausing alert: {str(e)}"}


def handle_alerts_resume(telegram_user_id: str, alert_id_str: str) -> Dict[str, Any]:
    """Resume a paused alert."""
    try:
        # Validate alert ID
        try:
            alert_id = int(alert_id_str)
        except ValueError:
            return {"status": "error", "message": "Alert ID must be a number"}

        # Check if alert exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        alert = telegram_svc.get_alert(alert_id)
        if not alert or alert.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Alert #{alert_id} not found or access denied."}

        # Resume the alert using service layer
        telegram_svc.update_alert(alert_id, active=True)

        return {
            "status": "ok",
            "title": "Alert Resumed",
            "message": f"Alert #{alert_id} for {alert['ticker']} has been resumed."
        }

    except Exception as e:
        _logger.exception("Error resuming alert: ")
        return {"status": "error", "message": f"Error resuming alert: {str(e)}"}


def handle_schedules(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /schedules commands.
    Handles creating, listing, editing, and deleting scheduled reports.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Get action and parameters from positionals
        action = parsed.positionals[0] if len(parsed.positionals) > 0 else None
        params = parsed.positionals[1:] if len(parsed.positionals) > 1 else []

        if not action:
            # List all schedules for user
            return handle_schedules_list(telegram_user_id)

        if action == "screener" and len(params) >= 1:
            list_type = params[0]
            time = params[1] if len(params) > 1 else "09:00"  # Default time
            # Get flags from parsed args
            email = parsed.args.get("email", False)
            indicators = parsed.args.get("indicators")
            return handle_schedules_screener(telegram_user_id, list_type, time, email, indicators)
        elif action == "enhanced_screener" and len(params) >= 1:
            config_json = params[0]
            return handle_schedules_enhanced_screener(telegram_user_id, config_json)
        elif action == "add" and len(params) >= 2:
            ticker, time = params[0], params[1]
            # Get flags from parsed args
            email = parsed.args.get("email", False)
            indicators = parsed.args.get("indicators")
            period = parsed.args.get("period", "2y")
            interval = parsed.args.get("interval", "1d")
            provider = parsed.args.get("provider")
            return handle_schedules_add(telegram_user_id, ticker, time, email, indicators, period, interval, provider)
        elif action == "edit" and len(params) >= 1:
            schedule_id = params[0]
            new_time = params[1] if len(params) > 1 else None
            return handle_schedules_edit(telegram_user_id, schedule_id, new_time, parsed.args)
        elif action == "delete" and len(params) >= 1:
            schedule_id = params[0]
            return handle_schedules_delete(telegram_user_id, schedule_id)
        elif action == "pause" and len(params) >= 1:
            schedule_id = params[0]
            return handle_schedules_pause(telegram_user_id, schedule_id)
        elif action == "resume" and len(params) >= 1:
            schedule_id = params[0]
            return handle_schedules_resume(telegram_user_id, schedule_id)
        elif action == "add_json" and len(params) >= 1:
            config_json = params[0]
            return handle_schedules_add_json(telegram_user_id, config_json)
        else:
            return {
                "status": "error",
                "title": "Schedules Help",
                "message": ("Available schedule commands:\n"
                           "/schedules - List all schedules\n"
                           "/schedules add TICKER TIME [flags] - Schedule daily report\n"
                           "  TIME: HH:MM format (24h UTC)\n"
                           "  Example: /schedules add AAPL 09:00 -email\n"
                           "Flags:\n"
                           "  -email: Send report to email\n"
                           "  -indicators=RSI,MACD: Specify indicators\n"
                           "  -period=1y: Data period\n"
                           "  -interval=1d: Data interval\n"
                           "  -provider=yf: Data provider\n"
                           "/schedules add_json CONFIG_JSON - Add advanced schedule with JSON config\n"
                           "  Example: /schedules add_json '{\"type\":\"report\",\"ticker\":\"AAPL\",\"scheduled_time\":\"09:00\",\"period\":\"1y\",\"interval\":\"1d\",\"email\":true}'\n"
                           "/schedules screener LIST_TYPE [TIME] [flags] - Schedule fundamental screener\n"
                           "  LIST_TYPE: us_small_cap, us_medium_cap, us_large_cap, swiss_shares, custom_list\n"
                           "  TIME: HH:MM format (24h UTC)\n"
                           "  Example: /schedules screener us_small_cap 09:00 -email\n"
                           "  Example: /schedules screener us_large_cap -indicators=PE,PB,ROE\n"
                           "/schedules enhanced_screener CONFIG_JSON - Schedule enhanced screener with JSON config\n"
                           "  Example: /schedules enhanced_screener '{\"screener_type\":\"hybrid\",\"list_type\":\"us_medium_cap\",\"fmp_criteria\":{\"marketCapMoreThan\":2000000000,\"peRatioLessThan\":20,\"returnOnEquityMoreThan\":0.12,\"limit\":50},\"fundamental_criteria\":[{\"indicator\":\"PE\",\"operator\":\"max\",\"value\":15,\"weight\":1.0,\"required\":true}],\"technical_criteria\":[{\"indicator\":\"RSI\",\"parameters\":{\"period\":14},\"condition\":{\"operator\":\"<\",\"value\":70},\"weight\":0.6,\"required\":false}],\"max_results\":10,\"min_score\":7.0,\"email\":true}'\n"
                           "/schedules edit SCHEDULE_ID [TIME] [flags] - Edit schedule\n"
                           "/schedules delete SCHEDULE_ID - Delete schedule\n"
                           "/schedules pause SCHEDULE_ID - Pause schedule\n"
                           "/schedules resume SCHEDULE_ID - Resume schedule\n\n"
                           "JSON Schedule Examples:\n"
                           "• Single Report: {\"type\":\"report\",\"ticker\":\"AAPL\",\"scheduled_time\":\"09:00\",\"period\":\"1y\",\"interval\":\"1d\",\"email\":true}\n"
                           "• Multiple Reports: {\"type\":\"report\",\"tickers\":[\"AAPL\",\"MSFT\",\"GOOGL\"],\"scheduled_time\":\"09:00\",\"period\":\"1y\",\"interval\":\"1d\",\"indicators\":\"RSI,MACD\",\"email\":true}\n"
                           "• Advanced Report: {\"type\":\"report\",\"ticker\":\"TSLA\",\"scheduled_time\":\"16:30\",\"period\":\"6mo\",\"interval\":\"1h\",\"indicators\":\"RSI,MACD,BollingerBands\",\"email\":true}\n"
                           "• Screener: {\"type\":\"screener\",\"list_type\":\"us_small_cap\",\"scheduled_time\":\"08:00\",\"period\":\"1y\",\"interval\":\"1d\",\"indicators\":\"PE,PB,ROE\",\"email\":true}\n"
                           "• FMP Enhanced Screener: {\"screener_type\":\"hybrid\",\"list_type\":\"us_medium_cap\",\"fmp_criteria\":{\"marketCapMoreThan\":2000000000,\"peRatioLessThan\":20,\"returnOnEquityMoreThan\":0.12,\"limit\":50},\"fundamental_criteria\":[{\"indicator\":\"PE\",\"operator\":\"max\",\"value\":15,\"weight\":1.0,\"required\":true}],\"max_results\":10,\"min_score\":7.0,\"email\":true}\n"
                           "• FMP Strategy Screener: {\"screener_type\":\"hybrid\",\"list_type\":\"us_large_cap\",\"fmp_strategy\":\"conservative_value\",\"fundamental_criteria\":[{\"indicator\":\"ROE\",\"operator\":\"min\",\"value\":15,\"weight\":1.0,\"required\":true}],\"max_results\":15,\"min_score\":7.5,\"email\":true}")
            }

    except Exception as e:
        _logger.exception("Error in schedules command: ")
        return {"status": "error", "message": f"Error processing schedules command: {str(e)}"}


def handle_schedules_add_json(telegram_user_id: str, config_json: str) -> Dict[str, Any]:
    """Add a new JSON-based schedule with support for multiple tickers and report configurations."""
    try:
        # Parse JSON to determine schedule type
        import json
        config = json.loads(config_json)
        schedule_type = config.get("type", "report")

        # Validate JSON configuration based on type
        if schedule_type == "report":
            # Validate report-specific fields
            required_fields = ["scheduled_time"]
            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                return {"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}

            # Check for either 'ticker' (single) or 'tickers' (multiple)
            ticker = config.get("ticker")
            tickers = config.get("tickers", [])

            if not ticker and not tickers:
                return {"status": "error", "message": "Either 'ticker' (single) or 'tickers' (multiple) must be specified"}

            if ticker and tickers:
                return {"status": "error", "message": "Cannot specify both 'ticker' and 'tickers' - use one or the other"}

            # Validate scheduled_time format (HH:MM)
            scheduled_time = config.get("scheduled_time", "")
            if not scheduled_time or not isinstance(scheduled_time, str):
                return {"status": "error", "message": "scheduled_time must be a string in HH:MM format"}

            # Basic time format validation
            try:
                hour, minute = scheduled_time.split(":")
                if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
                    raise ValueError("Invalid time")
            except:
                return {"status": "error", "message": "scheduled_time must be in HH:MM format (24h)"}

            # Validate tickers list if multiple
            if tickers and not isinstance(tickers, list):
                return {"status": "error", "message": "tickers must be a list"}

        else:
            # Use existing validation for other schedule types
            try:
                from src.common.alerts.schema_validator import validate_schedule_config
                is_valid, errors = validate_schedule_config(config_json)
                if not is_valid:
                    return {"status": "error", "message": f"Invalid schedule configuration: {'; '.join(errors)}"}
            except Exception as e:
                return {"status": "error", "message": f"Error validating schedule configuration: {str(e)}"}

        # Check user limits using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        user_status = telegram_svc.get_user_status(telegram_user_id)
        max_schedules = user_status.get("max_schedules", 5)
        current_schedules = len(telegram_svc.list_schedules(telegram_user_id))

        if current_schedules >= max_schedules:
            return {
                "status": "error",
                "message": f"Schedule limit reached ({max_schedules}). Delete some schedules first or contact admin."
            }

        # Determine schedule_config based on type
        if schedule_type == "report":
            schedule_config = "report"
        elif schedule_type == "enhanced_screener":
            schedule_config = "enhanced_screener"
        else:
            schedule_config = "advanced"

        # Add the JSON schedule using service layer
        schedule_id = telegram_svc.add_json_schedule(
            telegram_user_id=telegram_user_id,
            config_json=config_json,
            schedule_config=schedule_config
        )

        # Create success message based on type
        if schedule_type == "report":
            # Handle report schedule
            ticker = config.get("ticker")
            tickers = config.get("tickers", [])

            if ticker:
                tickers_str = ticker
                ticker_count = 1
            else:
                tickers_str = ", ".join(tickers)
                ticker_count = len(tickers)

            period = config.get("period", "2y")
            interval = config.get("interval", "1d")
            indicators = config.get("indicators", "")
            email_flag = " (with email)" if config.get("email", False) else ""

            message = f"Report schedule #{schedule_id} created for {tickers_str} at {scheduled_time} UTC{email_flag}"
            if indicators:
                message += f"\nIndicators: {indicators}"
            message += f"\nPeriod: {period}, Interval: {interval}"

            if ticker_count > 1:
                message += f"\n📊 Multiple tickers: {ticker_count} reports will be generated"

        else:
            # Use existing summary for other types
            from src.common.alerts.schema_validator import get_schedule_summary
            summary = get_schedule_summary(config_json)

            if "error" in summary:
                return {"status": "error", "message": f"Error creating schedule: {summary['error']}"}

            message = f"Schedule #{schedule_id} created: {summary.get('type', 'Unknown')} at {summary.get('scheduled_time', 'Unknown')}"

        return {
            "status": "ok",
            "title": "Schedule Added",
            "message": message
        }

    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid JSON format"}
    except Exception as e:
        _logger.exception("Error adding JSON schedule: ")
        return {"status": "error", "message": f"Error adding JSON schedule: {str(e)}"}



def handle_schedules_enhanced_screener(telegram_user_id: str, config_json: str) -> Dict[str, Any]:
    """Handle enhanced screener schedule creation with JSON configuration."""
    try:
        # Validate JSON configuration
        try:
            from src.telegram.screener.screener_config_parser import validate_screener_config
            is_valid, errors = validate_screener_config(config_json)
            if not is_valid:
                return {"status": "error", "message": f"Invalid screener configuration: {'; '.join(errors)}"}
        except Exception as e:
            return {"status": "error", "message": f"Error validating screener configuration: {str(e)}"}

        # Check user limits using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        user_status = telegram_svc.get_user_status(telegram_user_id)
        max_schedules = user_status.get("max_schedules", 5)
        current_schedules = len(telegram_svc.list_schedules(telegram_user_id))

        if current_schedules >= max_schedules:
            return {
                "status": "error",
                "message": f"Schedule limit reached ({max_schedules}). Delete some schedules first or contact admin."
            }

        # Parse the configuration to get summary
        from src.telegram.screener.screener_config_parser import get_screener_summary
        summary = get_screener_summary(config_json)

        if "error" in summary:
            return {"status": "error", "message": f"Error parsing screener configuration: {summary['error']}"}

        # Add the enhanced screener schedule using service layer
        schedule_id = telegram_svc.add_json_schedule(
            telegram_user_id=telegram_user_id,
            config_json=config_json,
            schedule_config="enhanced_screener"
        )

        # Create success message
        screener_type = summary.get("screener_type", "Unknown")
        list_type = summary.get("list_type", "Unknown")
        fundamental_count = summary.get("fundamental_criteria_count", 0)
        technical_count = summary.get("technical_criteria_count", 0)
        max_results = summary.get("max_results", 10)
        min_score = summary.get("min_score", 7.0)

        message = "✅ Enhanced screener scheduled successfully!\n\n"
        message += f"📊 **Screener Type**: {screener_type.title()}\n"
        message += f"🔍 **List Type**: {list_type.replace('_', ' ').title()}\n"

        # Add FMP information if available
        fmp_criteria_count = summary.get("fmp_criteria_count", 0)
        fmp_strategy = summary.get("fmp_strategy")

        if fmp_criteria_count > 0:
            message += f"🚀 **FMP Pre-filtering**: {fmp_criteria_count} criteria\n"
        if fmp_strategy:
            message += f"📋 **FMP Strategy**: {fmp_strategy}\n"

        message += f"📈 **Fundamental Criteria**: {fundamental_count} indicators\n"
        message += f"📊 **Technical Criteria**: {technical_count} indicators\n"
        message += f"🎯 **Max Results**: {max_results}\n"
        message += f"📊 **Min Score**: {min_score}/10\n"
        message += f"🆔 **Schedule ID**: {schedule_id}\n\n"

        if screener_type == "fundamental":
            message += "This screener will analyze stocks based on fundamental metrics only."
        elif screener_type == "technical":
            message += "This screener will analyze stocks based on technical indicators only."
        elif screener_type == "hybrid":
            message += "This screener will combine fundamental and technical analysis for comprehensive screening."

        return {
            "status": "ok",
            "title": "Enhanced Screener Scheduled",
            "message": message
        }

    except Exception as e:
        _logger.exception("Error adding enhanced screener schedule: ")
        return {"status": "error", "message": f"Error adding enhanced screener schedule: {str(e)}"}


def handle_schedules_list(telegram_user_id: str) -> Dict[str, Any]:
    """List all schedules for a user."""
    try:
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        schedules = telegram_svc.list_schedules(telegram_user_id)
        if not schedules:
            return {"status": "ok", "title": "Your Schedules", "message": "You have no scheduled reports."}

        schedule_list = []
        for schedule in schedules:
            status = "🟢 Active" if schedule.get("active") else "🔴 Paused"
            email_flag = "📧" if schedule.get("email") else "💬"

            # Handle different schedule types
            schedule_config = schedule.get("schedule_config", "simple")
            if schedule_config == "simple":
                period = schedule.get("period", "daily")
                schedule_list.append(
                    f"#{schedule['id']}: {schedule['ticker']} at {schedule['scheduled_time']} ({period}) {email_flag} - {status}"
                )
            else:
                # JSON-based schedule
                from src.common.alerts.schema_validator import get_schedule_summary
                config_json = schedule.get("config_json")
                if config_json:
                    summary = get_schedule_summary(config_json)
                    if "error" not in summary:
                        schedule_type = summary.get("type", "Unknown")
                        scheduled_time = summary.get("scheduled_time", "Unknown")
                        ticker = summary.get("ticker", "")
                        list_type = summary.get("list_type", "")

                        if schedule_type == "report":
                            schedule_list.append(
                                f"#{schedule['id']}: 📊 {ticker} Report at {scheduled_time} {email_flag} - {status}"
                            )
                        elif schedule_type == "screener":
                            schedule_list.append(
                                f"#{schedule['id']}: 🔍 {list_type} Screener at {scheduled_time} {email_flag} - {status}"
                            )
                        else:
                            schedule_list.append(
                                f"#{schedule['id']}: ⚙️ {schedule_type} at {scheduled_time} {email_flag} - {status}"
                            )
                    else:
                        schedule_list.append(
                            f"#{schedule['id']}: ⚙️ JSON Schedule at {schedule.get('scheduled_time', 'Unknown')} {email_flag} - {status}"
                        )
                else:
                    schedule_list.append(
                        f"#{schedule['id']}: ⚙️ JSON Schedule at {schedule.get('scheduled_time', 'Unknown')} {email_flag} - {status}"
                    )

        message = f"Your scheduled reports ({len(schedules)}):\n\n" + "\n".join(schedule_list)
        return {"status": "ok", "title": "Your Schedules", "message": message}

    except Exception as e:
        _logger.exception("Error listing schedules: ")
        return {"status": "error", "message": f"Error listing schedules: {str(e)}"}


def handle_schedules_add(telegram_user_id: str, ticker: str, time: str, email: bool = False,
                        indicators: str = None, period: str = "2y", interval: str = "1d",
                        provider: str = None) -> Dict[str, Any]:
    """Add a new scheduled report."""
    try:
        # Validate time format (HH:MM)
        import re
        if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', time):
            return {"status": "error", "message": "Time must be in HH:MM format (24-hour, e.g., 09:00, 15:30)"}

        # Check user limits using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        user_status = telegram_svc.get_user_status(telegram_user_id)
        max_schedules = user_status.get("max_schedules", 5)
        current_schedules = len(telegram_svc.list_schedules(telegram_user_id))

        if current_schedules >= max_schedules:
            return {
                "status": "error",
                "message": f"Schedule limit reached ({max_schedules}). Delete some schedules first or contact admin."
            }

        # Add the schedule (convert ticker to uppercase)
        schedule_id = telegram_svc.add_schedule(
            telegram_user_id,
            ticker.upper(),
            time,
            period="daily",  # Default to daily for now
            email=email,
            indicators=indicators,
            interval=interval,
            provider=provider
        )

        email_text = " and email" if email else ""
        indicators_text = f" with indicators: {indicators}" if indicators else ""

        return {
            "status": "ok",
            "title": "Schedule Added",
            "message": f"Schedule #{schedule_id} created: {ticker.upper()} daily at {time} (UTC) via Telegram{email_text}{indicators_text}"
        }

    except Exception as e:
        _logger.exception("Error adding schedule: ")
        return {"status": "error", "message": f"Error adding schedule: {str(e)}"}


def handle_schedules_edit(telegram_user_id: str, schedule_id_str: str, new_time: str = None, args: dict = None) -> Dict[str, Any]:
    """Edit an existing schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        schedule = telegram_svc.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        updates = {}

        # Validate and set new time
        if new_time:
            import re
            if not re.match(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$', new_time):
                return {"status": "error", "message": "Time must be in HH:MM format (24-hour, e.g., 09:00, 15:30)"}
            updates["scheduled_time"] = new_time

        # Update other flags if provided
        if args:
            if "email" in args:
                updates["email"] = args["email"]
            if "indicators" in args and args["indicators"]:
                updates["indicators"] = args["indicators"]
            if "period" in args and args["period"]:
                updates["period"] = args["period"]
            if "interval" in args and args["interval"]:
                updates["interval"] = args["interval"]
            if "provider" in args and args["provider"]:
                updates["provider"] = args["provider"]

        if not updates:
            return {"status": "error", "message": "No updates provided. Specify new time and/or flags."}

        # Update the schedule using service layer
        telegram_svc.update_schedule(schedule_id, **updates)

        # Get updated schedule for confirmation
        updated_schedule = telegram_svc.get_schedule_by_id(schedule_id)

        return {
            "status": "ok",
            "title": "Schedule Updated",
            "message": f"Schedule #{schedule_id} updated: {updated_schedule['ticker']} at {updated_schedule['scheduled_time']}"
        }

    except Exception as e:
        _logger.exception("Error editing schedule: ")
        return {"status": "error", "message": f"Error editing schedule: {str(e)}"}


def handle_schedules_delete(telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
    """Delete a schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        schedule = telegram_svc.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        # Delete the schedule using service layer
        telegram_svc.delete_schedule(schedule_id)

        return {
            "status": "ok",
            "title": "Schedule Deleted",
            "message": f"Schedule #{schedule_id} for {schedule['ticker']} has been deleted."
        }

    except Exception as e:
        _logger.exception("Error deleting schedule: ")
        return {"status": "error", "message": f"Error deleting schedule: {str(e)}"}


def handle_schedules_pause(telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
    """Pause a schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        schedule = telegram_svc.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        # Pause the schedule using service layer
        telegram_svc.update_schedule(schedule_id, active=False)

        return {
            "status": "ok",
            "title": "Schedule Paused",
            "message": f"Schedule #{schedule_id} for {schedule['ticker']} has been paused."
        }

    except Exception as e:
        _logger.exception("Error pausing schedule: ")
        return {"status": "error", "message": f"Error pausing schedule: {str(e)}"}


def handle_schedules_resume(telegram_user_id: str, schedule_id_str: str) -> Dict[str, Any]:
    """Resume a paused schedule."""
    try:
        # Validate schedule ID
        try:
            schedule_id = int(schedule_id_str)
        except ValueError:
            return {"status": "error", "message": "Schedule ID must be a number"}

        # Check if schedule exists and belongs to user using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        schedule = telegram_svc.get_schedule_by_id(schedule_id)
        if not schedule or schedule.get("user_id") != telegram_user_id:
            return {"status": "error", "message": f"Schedule #{schedule_id} not found or access denied."}

        # Resume the schedule using service layer
        telegram_svc.update_schedule(schedule_id, active=True)

        return {
            "status": "ok",
            "title": "Schedule Resumed",
            "message": f"Schedule #{schedule_id} for {schedule['ticker']} has been resumed."
        }

    except Exception as e:
        _logger.exception("Error resuming schedule: ")
        return {"status": "error", "message": f"Error resuming schedule: {str(e)}"}


def handle_schedules_screener(telegram_user_id: str, list_type: str, time: str,
                            email: bool = False, indicators: str = None) -> Dict[str, Any]:
    """Handle screener schedule creation."""
    try:
        # Validate list type (case-insensitive)
        valid_list_types = ['us_small_cap', 'us_medium_cap', 'us_large_cap', 'swiss_shares', 'custom_list']
        if list_type.lower() not in [lt.lower() for lt in valid_list_types]:
            return {
                'status': 'error',
                'message': f"Invalid list type. Valid types: {', '.join(valid_list_types)}"
            }

        # Convert list_type to lowercase for consistency
        list_type = list_type.lower()

        # Validate time format
        try:
            # Simple time validation (HH:MM format)
            hour, minute = map(int, time.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Invalid time")
        except (ValueError, AttributeError):
            return {
                'status': 'error',
                'message': "Invalid time format. Use HH:MM (24-hour format, UTC)"
            }

        # Check user limits using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        current_schedules = telegram_svc.list_schedules(telegram_user_id)
        user_limit = telegram_svc.get_user_limit(telegram_user_id, 'max_schedules')

        # Default to 5 if no limit is set
        if user_limit is None:
            user_limit = 5

        if len(current_schedules) >= user_limit:
            return {
                'status': 'error',
                'message': f"You have reached your limit of {user_limit} scheduled reports. Delete some schedules first."
            }

        # Create screener schedule
        schedule_data = {
            'telegram_user_id': telegram_user_id,
            'ticker': f"SCREENER_{list_type.upper()}",  # Special ticker format for screeners
            'scheduled_time': time,
            'email': email,
            'indicators': indicators,
            'period': 'daily',  # Screeners run daily
            'interval': '1d',
            'provider': 'yf',
            'active': True,
            'schedule_type': 'screener',  # New field to distinguish screeners
            'list_type': list_type  # Store the list type
        }

        schedule_id = telegram_svc.add_json_schedule(telegram_user_id, str(schedule_data))

        if schedule_id:
            message = "✅ Fundamental screener scheduled successfully!\n"
            message += f"📊 **List Type**: {list_type.replace('_', ' ').title()}\n"
            message += f"⏰ **Time**: {time} UTC (daily)\n"
            message += f"📧 **Email**: {'Yes' if email else 'No'}\n"
            if indicators:
                message += f"📈 **Indicators**: {indicators}\n"
            message += f"🆔 **Schedule ID**: {schedule_id}\n\n"
            message += "The screener will analyze stocks for undervaluation based on:\n"
            message += "• P/E < 15, P/B < 1.5, P/S < 1\n"
            message += "• ROE > 15%, Debt/Equity < 0.5\n"
            message += "• Positive Free Cash Flow\n"
            message += "• Composite scoring (0-10 scale)\n"
            message += "• DCF valuation analysis"

            return {
                'status': 'ok',
                'title': "Screener Scheduled",
                'message': message
            }
        else:
            return {
                'status': 'error',
                'message': "Failed to create screener schedule. Please try again."
            }

    except Exception as e:
        _logger.exception("Error creating screener schedule: ")
        return {
            'status': 'error',
            'message': f"Error creating screener schedule: {str(e)}"
        }


def handle_feedback(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /feedback command.
    Collects user feedback and forwards to administrators.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        feedback = parsed.args.get("feedback")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not feedback:
            return {"status": "error", "message": "Please provide feedback message. Usage: /feedback Your message here"}

        # Log feedback for admin review
        _logger.info("User feedback", extra={
            "user_id": telegram_user_id,
            "feedback": feedback,
            "type": "feedback"
        })

        # Store feedback in database for admin panel using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        feedback_id = telegram_svc.add_feedback(telegram_user_id, "feedback", feedback)

        return {
            "status": "ok",
            "title": "Feedback Received",
            "message": "Thank you for your feedback! It has been forwarded to the development team.",
            "admin_notification": {
                "type": "feedback",
                "user_id": telegram_user_id,
                "message": feedback
            }
        }

    except Exception as e:
        _logger.exception("Error processing feedback: ")
        return {"status": "error", "message": f"Error processing feedback: {str(e)}"}


def handle_feature(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /feature command.
    Collects feature requests and forwards to administrators.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        feature_request = parsed.args.get("feature")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not feature_request:
            return {"status": "error", "message": "Please provide feature request. Usage: /feature Your feature idea here"}

        # Log feature request for admin review
        _logger.info("Feature request", extra={
            "user_id": telegram_user_id,
            "feature_request": feature_request,
            "type": "feature_request"
        })

        # Store feature request in database for admin panel using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        feature_id = telegram_svc.add_feedback(telegram_user_id, "feature_request", feature_request)

        return {
            "status": "ok",
            "title": "Feature Request Received",
            "message": "Thank you for your feature request! It has been added to our development backlog.",
            "admin_notification": {
                "type": "feature_request",
                "user_id": telegram_user_id,
                "message": feature_request
            }
        }

    except Exception as e:
        _logger.exception("Error processing feature request: ")
        return {"status": "error", "message": f"Error processing feature request: {str(e)}"}


def handle_register(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /register command.
    Register or update user email and send verification code.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        email = parsed.args.get("email")
        language = parsed.args.get("language", "en")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not email:
            return {"status": "error", "message": "Please provide an email address. Usage: /register email@example.com [language]"}

        # Validate email format
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return {"status": "error", "message": "Please provide a valid email address."}

        # Check rate limiting using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        codes_sent = telegram_svc.count_codes_last_hour(telegram_user_id)
        if codes_sent >= 5:
            return {"status": "error", "message": "Too many verification codes sent. Please wait an hour before requesting another."}

        # Generate verification code
        import random
        code = f"{random.randint(100000, 999999):06d}"
        sent_time = int(time.time())

        # Store user and code using service layer
        telegram_svc.set_user_email(telegram_user_id, email, code, sent_time, language)

        # Send verification code via email
        # This will be handled by the notification system
        return {
            "status": "ok",
            "title": "Email Registration",
            "message": f"A 6-digit verification code has been sent to {email}. Use /verify CODE to verify your email.",
            "email_verification": {
                "email": email,
                "code": code,
                "user_id": telegram_user_id
            }
        }

    except Exception as e:
        _logger.exception("Error in register command: ")
        return {"status": "error", "message": f"Error registering email: {str(e)}"}


def handle_verify(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /verify command.
    Verify user email with the provided code.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        code = parsed.args.get("code")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not code:
            return {"status": "error", "message": "Please provide the verification code. Usage: /verify CODE"}

        # Validate code format
        if not code.isdigit() or len(code) != 6:
            return {"status": "error", "message": "Verification code must be a 6-digit number."}

        # Verify the code using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        if telegram_svc.verify_code(telegram_user_id, code, expiry_seconds=3600):
            return {
                "status": "ok",
                "title": "Email Verified",
                "message": "Your email has been successfully verified! You can now use all bot features including email reports."
            }
        else:
            return {
                "status": "error",
                "message": "Invalid or expired verification code. Please check the code or request a new one with /register."
            }

    except Exception as e:
        _logger.exception("Error in verify command: ")
        return {"status": "error", "message": f"Error verifying code: {str(e)}"}


def handle_language(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /language command.
    Update user's language preference.
    """
    try:
        telegram_user_id = parsed.args.get("telegram_user_id")
        language = parsed.args.get("language")

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not language:
            return {"status": "error", "message": "Please provide a language code. Usage: /language en (supported: en, ru)"}

        # Validate language
        supported_languages = ["en", "ru"]
        if language.lower() not in supported_languages:
            return {"status": "error", "message": f"Language '{language}' not supported. Supported languages: {', '.join(supported_languages)}"}

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Update user language using service layer
        telegram_svc, _ = get_service_instances()
        if not telegram_svc:
            return {"status": "error", "message": "Service temporarily unavailable"}

        user_status = telegram_svc.get_user_status(telegram_user_id)
        if not user_status:
            return {"status": "error", "message": "Please register first using /register email@example.com"}

        # Update language using service layer
        success = telegram_svc.update_user_language(telegram_user_id, language.lower())
        if not success:
            return {"status": "error", "message": "Failed to update language preference"}

        return {
            "status": "ok",
            "title": "Language Updated",
            "message": f"Your language preference has been updated to {language.upper()}."
        }

    except Exception as e:
        _logger.exception("Error in language command: ")
        return {"status": "error", "message": f"Error updating language: {str(e)}"}


async def handle_screener(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Business logic for /screener command for immediate screener execution.
    Supports both predefined screeners and custom JSON configuration.
    """
    try:
        # Extract parameters
        telegram_user_id = parsed.args.get("telegram_user_id")
        config_json = parsed.args.get("screener_name_or_config")
        send_email = parsed.args.get("email", False)

        if not telegram_user_id:
            return {"status": "error", "message": "No telegram_user_id provided"}

        if not config_json:
            return {"status": "error", "message": "Please provide screener name or configuration. Usage: /screener <SCREENER_NAME> [-email] or /screener <JSON_CONFIG> [-email]"}

        # Check if user has approved access
        access_check = check_approved_access(telegram_user_id)
        if access_check["status"] != "ok":
            return access_check

        # Import screener modules
        from src.telegram.screener.enhanced_screener import EnhancedScreener
        from src.telegram.screener.screener_config_parser import (
            parse_screener_config,
            validate_screener_config
        )

        # Check if config_json is a predefined screener name
        screener_config = None
        if config_json.startswith('{'):
            # It's a JSON configuration
            is_valid, errors = validate_screener_config(config_json)
            if not is_valid:
                return {"status": "error", "message": f"Invalid screener configuration: {errors}"}
            screener_config = parse_screener_config(config_json)
        else:
            # It's a predefined screener name
            screener_config = _get_predefined_screener_config(config_json)
            if not screener_config:
                return {"status": "error", "message": f"Unknown screener: {config_json}. Available screeners: {', '.join(_get_available_screeners())}"}

        # Acquire service instances (module-level globals set via set_service_instances)
        telegram_svc, indicator_svc = get_service_instances()

        if not indicator_svc:
            # Fallback: attempt lazy import/creation of indicator service
            try:
                from src.indicators.service import get_unified_indicator_service
                indicator_svc = get_unified_indicator_service()
            except Exception:
                _logger.exception("Indicator service unavailable for screener execution")
                return {"status": "error", "message": "Indicator service unavailable. Try again later."}

        if not telegram_svc:
            return {"status": "error", "message": "Telegram service unavailable. Try again later."}

        # Run enhanced screener immediately using resolved indicator service
        enhanced_screener = EnhancedScreener(indicator_service=indicator_svc)
        report = await enhanced_screener.run_enhanced_screener(screener_config)

        if report.error:
            return {"status": "error", "message": report.error}

        # Format results
        message = enhanced_screener.format_enhanced_telegram_message(report, screener_config)

        # Send results
        if send_email:
            # Get user email using service layer
            user_status = telegram_svc.get_user_status(telegram_user_id)
            if not user_status or not user_status.get("email"):
                return {"status": "error", "message": "Email not registered. Please use /register email@example.com first"}

            # Send via email
            from src.telegram.screener.notifications import send_screener_email
            send_screener_email(user_status["email"], report, screener_config)
            return {"status": "success", "message": "Screener results sent to your email"}
        else:
            # Return for Telegram display
            return {"status": "success", "message": message, "report": report}

    except Exception as e:
        _logger.exception("Error in screener command")
        return {"status": "error", "message": f"Screener error: {str(e)}"}


def _get_predefined_screener_config(screener_name: str):
    """
    Get predefined screener configuration by name.
    """
    try:
        import json
        from pathlib import Path
        from src.telegram.screener.screener_config_parser import ScreenerConfigParser

        # Load FMP screener criteria
        config_path = Path(__file__).resolve().parents[4] / "config" / "screener" / "fmp_screener_criteria.json"

        with open(config_path, 'r') as f:
            fmp_config = json.load(f)

        # Check if screener exists in predefined strategies
        if screener_name in fmp_config.get("predefined_strategies", {}):
            strategy = fmp_config["predefined_strategies"][screener_name]

            # Create screener configuration dictionary
            config_dict = {
                "screener_name": screener_name,  # Add screener name for email titles
                "screener_type": "hybrid",
                "list_type": _get_list_type_for_screener(screener_name),
                "fmp_criteria": strategy["criteria"],
                "fundamental_criteria": _get_fundamental_criteria_for_screener(screener_name),
                "technical_criteria": _get_technical_criteria_for_screener(screener_name),
                "max_results": strategy["criteria"].get("limit", 50),
                "min_score": 0.5,
                "period": "1y",
                "interval": "1d"
            }

            # Convert dictionary to ScreenerConfig object
            parser = ScreenerConfigParser()
            return parser._parse_config_dict(config_dict)

        return None

    except Exception as e:
        _logger.error("Error loading predefined screener config for %s: %s", screener_name, e)
        return None


def _get_list_type_for_screener(screener_name: str) -> str:
    """
    Determine the appropriate list type for a given screener.
    """
    if screener_name == "six_stocks":
        return "swiss_shares"
    elif screener_name == "mid_cap_stocks":
        return "us_medium_cap"  # Use medium cap list, filtered by FMP criteria
    elif screener_name in ["large_cap_stocks", "extra_large_cap_stocks"]:
        return "us_large_cap"  # Will be filtered by FMP criteria
    else:
        return "us_medium_cap"  # Default fallback


def _get_fundamental_criteria_for_screener(screener_name: str):
    """
    Get fundamental criteria for a predefined screener.
    """
    # Base fundamental criteria for all screeners
    base_criteria = [
        {
            "indicator": "PE",
            "operator": "max",
            "value": 30,
            "weight": 1.0,
            "required": False
        },
        {
            "indicator": "PB",
            "operator": "max",
            "value": 3.0,
            "weight": 1.0,
            "required": False
        },
        {
            "indicator": "ROE",
            "operator": "min",
            "value": 0.10,
            "weight": 1.0,
            "required": False
        }
    ]

    # Adjust criteria based on screener type
    if screener_name == "conservative_value":
        base_criteria[0]["value"] = 12  # PE < 12
        base_criteria[1]["value"] = 1.2  # PB < 1.2
        base_criteria[2]["value"] = 0.15  # ROE > 15%
    elif screener_name == "deep_value":
        base_criteria[0]["value"] = 8   # PE < 8
        base_criteria[1]["value"] = 0.8  # PB < 0.8
        base_criteria[2]["value"] = 0.08  # ROE > 8%
    elif screener_name == "quality_growth":
        base_criteria[0]["value"] = 25  # PE < 25
        base_criteria[1]["value"] = 5.0  # PB < 5.0
        base_criteria[2]["value"] = 0.18  # ROE > 18%
    elif screener_name == "large_cap_stocks":
        base_criteria[0]["value"] = 35  # PE < 35
        base_criteria[1]["value"] = 5.0  # PB < 5.0
        base_criteria[2]["value"] = 0.15  # ROE > 15%
    elif screener_name == "extra_large_cap_stocks":
        base_criteria[0]["value"] = 40  # PE < 40
        base_criteria[1]["value"] = 6.0  # PB < 6.0
        base_criteria[2]["value"] = 0.15  # ROE > 15%
    elif screener_name == "six_stocks":
        base_criteria[0]["value"] = 20  # PE < 20
        base_criteria[1]["value"] = 2.5  # PB < 2.5
        base_criteria[2]["value"] = 0.08  # ROE > 8%

    return base_criteria


def _get_technical_criteria_for_screener(screener_name: str):
    """
    Get technical criteria for a predefined screener.
    """
    # Base technical criteria for all screeners
    return [
        {
            "indicator": "RSI",
            "parameters": {"period": 14},
            "condition": {"operator": "<", "value": 75},
            "weight": 0.5,
            "required": False
        }
    ]


def _get_available_screeners():
    """
    Get list of available predefined screeners.
    """
    try:
        import json
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[4] / "config" / "screener" / "fmp_screener_criteria.json"

        with open(config_path, 'r') as f:
            fmp_config = json.load(f)

        return list(fmp_config.get("predefined_strategies", {}).keys())

    except Exception:
        _logger.exception("Error loading available screeners:")
        return []


# Global service instances - will be set by bot.py during initialization
_telegram_service_instance = None
_indicator_service_instance = None


def set_service_instances(telegram_service, indicator_service):
    """
    Set global service instances for use by standalone functions with enhanced error handling.

    This function should be called by bot.py during initialization to provide
    service instances to the business logic layer.

    Args:
        telegram_service: Telegram service instance for database operations
        indicator_service: Indicator service instance for calculations

    Raises:
        ValueError: If required service instances are None or invalid
        RuntimeError: If service validation fails
    """
    global _telegram_service_instance, _indicator_service_instance

    try:
        # Validate service instances before setting
        if telegram_service is None:
            raise ValueError("Telegram service instance cannot be None")

        if indicator_service is None:
            _logger.warning("Indicator service instance is None - creating default instance")
            try:
                from src.indicators.service import IndicatorService
                indicator_service = IndicatorService()
                _logger.info("Created default IndicatorService instance")
            except Exception as e:
                _logger.exception("Failed to create default IndicatorService:")
                raise RuntimeError(f"Failed to create default IndicatorService: {str(e)}")

        # Validate telegram service has required methods
        required_telegram_methods = [
            'get_user_status', 'set_user_limit', 'add_alert', 'list_alerts',
            'add_schedule', 'list_schedules', 'log_command_audit', 'add_feedback'
        ]

        for method_name in required_telegram_methods:
            if not hasattr(telegram_service, method_name):
                raise ValueError(f"Telegram service missing required method: {method_name}")

        # Validate indicator service has required methods and attributes
        if not hasattr(indicator_service, 'compute_for_ticker'):
            raise ValueError("Indicator service missing required method: compute_for_ticker")

        # Set the validated instances
        _telegram_service_instance = telegram_service
        _indicator_service_instance = indicator_service

        _logger.info("Service instances set for business logic layer successfully")
        _logger.debug("Telegram service type: %s", type(telegram_service).__name__)
        _logger.debug("Indicator service type: %s", type(indicator_service).__name__)

    except Exception:
        _logger.exception("Failed to set service instances:")
        # Reset instances to None on failure to prevent partial initialization
        _telegram_service_instance = None
        _indicator_service_instance = None
        raise


def get_service_instances():
    """
    Get the current service instances with validation.

    Returns:
        tuple: (telegram_service, indicator_service) or (None, None) if not set

    Logs warnings if services are not properly initialized.
    """
    try:
        if _telegram_service_instance is None:
            _logger.warning("Telegram service instance not initialized - call set_service_instances() first")

        if _indicator_service_instance is None:
            _logger.warning("Indicator service instance not initialized - call set_service_instances() first")

        # Log service status for debugging
        _logger.debug("Service instances status: telegram=%s, indicator=%s",
                     "available" if _telegram_service_instance else "None",
                     "available" if _indicator_service_instance else "None")

        return _telegram_service_instance, _indicator_service_instance

    except Exception:
        _logger.exception("Error retrieving service instances:")
        return None, None


async def handle_command(parsed: ParsedCommand) -> Dict[str, Any]:
    """
    Standalone handle_command function that uses service instances with enhanced error handling.

    This function creates a TelegramBusinessLogic instance with the global
    service instances and delegates to its handle_command method.

    Args:
        parsed: ParsedCommand object containing command and arguments

    Returns:
        Dict with result/status/data for notification manager
    """
    try:
        # Validate input parameters
        if not parsed:
            _logger.error("ParsedCommand object is None")
            return {
                "status": "error",
                "message": "Invalid command format. Please try again."
            }

        if not hasattr(parsed, 'command') or not parsed.command:
            _logger.error("ParsedCommand missing command attribute")
            return {
                "status": "error",
                "message": "Invalid command format. Please try again."
            }

        # Get service instances with enhanced error handling
        telegram_svc, indicator_svc = get_service_instances()

        if not telegram_svc:
            _logger.error("Telegram service not available for command processing: %s", parsed.command)
            return {
                "status": "error",
                "message": "Service temporarily unavailable. Please try again later.",
                "error_type": "ServiceUnavailable"
            }

        # Handle missing indicator service with fallback
        if not indicator_svc:
            _logger.warning("Indicator service not available for command %s, attempting to create default instance", parsed.command)
            try:
                indicator_svc = IndicatorService()
                _logger.info("Successfully created fallback IndicatorService instance")
            except Exception as indicator_error:
                _logger.error("Failed to create fallback IndicatorService: %s", indicator_error)
                # For commands that don't require indicators, continue without it
                if parsed.command in ["help", "info", "register", "verify", "language", "feedback", "feature"]:
                    indicator_svc = None
                    _logger.info("Continuing without IndicatorService for command: %s", parsed.command)
                else:
                    return {
                        "status": "error",
                        "message": "Indicator calculation service unavailable. Please try again later.",
                        "error_type": "IndicatorServiceUnavailable"
                    }

        # Create business logic instance with services and enhanced error handling
        try:
            business_logic = TelegramBusinessLogic(telegram_svc, indicator_svc)
            _logger.debug("Created TelegramBusinessLogic instance for command: %s", parsed.command)
        except Exception as init_error:
            _logger.error("Failed to create TelegramBusinessLogic instance: %s", init_error)
            return {
                "status": "error",
                "message": "Service initialization error. Please try again later.",
                "error_type": "ServiceInitializationError"
            }

        # Delegate to class method with timeout protection
        try:
            result = await business_logic.handle_command(parsed)

            # Validate result format
            if not isinstance(result, dict):
                _logger.error("Invalid result format from business logic: %s", type(result))
                return {
                    "status": "error",
                    "message": "Internal processing error. Please try again later.",
                    "error_type": "InvalidResultFormat"
                }

            return result

        except asyncio.TimeoutError:
            _logger.error("Command processing timeout for: %s", parsed.command)
            return {
                "status": "error",
                "message": "Command processing timeout. Please try again with a simpler request.",
                "error_type": "ProcessingTimeout"
            }

    except Exception as e:
        _logger.exception("Unexpected error in standalone handle_command for command %s: %s",
                         getattr(parsed, 'command', 'unknown'), e)

        # Provide user-friendly error messages based on exception type
        error_msg = str(e).lower()
        if "timeout" in error_msg:
            user_message = "Request timeout. Please try again."
        elif "memory" in error_msg or "resource" in error_msg:
            user_message = "System resources temporarily unavailable. Please try again later."
        elif "connection" in error_msg or "network" in error_msg:
            user_message = "Connection issue. Please check your network and try again."
        else:
            user_message = "An unexpected error occurred. Please try again later."

        return {
            "status": "error",
            "message": user_message,
            "error_type": "UnexpectedError"
        }
