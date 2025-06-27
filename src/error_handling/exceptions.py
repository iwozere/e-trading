"""
Custom Exception Classes
=======================

This module defines custom exception classes for the trading platform.
These exceptions provide standardized error handling with detailed context
and categorization for different types of errors.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import traceback


class TradingException(Exception):
    """
    Base exception class for all trading-related errors.
    
    Provides standardized error handling with:
    - Error categorization
    - Context information
    - Timestamp tracking
    - Stack trace preservation
    - Recovery suggestions
    """
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 severity: str = "ERROR",
                 recoverable: bool = True,
                 retry_after: Optional[int] = None):
        """
        Initialize the trading exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for categorization
            context: Additional context information
            severity: Error severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            recoverable: Whether the error is recoverable
            retry_after: Seconds to wait before retry (if applicable)
        """
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.severity = severity.upper()
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.timestamp = datetime.utcnow()
        self.stack_trace = traceback.format_exc()
        
        # Add default context
        if 'timestamp' not in self.context:
            self.context['timestamp'] = self.timestamp.isoformat()
        if 'error_code' not in self.context:
            self.context['error_code'] = self.error_code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'severity': self.severity,
            'recoverable': self.recoverable,
            'retry_after': self.retry_after,
            'timestamp': self.timestamp.isoformat(),
            'stack_trace': self.stack_trace
        }
    
    def get_recovery_suggestion(self) -> Optional[str]:
        """Get recovery suggestion for this error."""
        return self.context.get('recovery_suggestion')
    
    def should_retry(self) -> bool:
        """Determine if this error should be retried."""
        return self.recoverable and self.retry_after is not None
    
    def __str__(self) -> str:
        """String representation of the exception."""
        parts = [f"{self.__class__.__name__}: {self.message}"]
        
        if self.error_code:
            parts.append(f"Code: {self.error_code}")
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items() 
                                   if k not in ['timestamp', 'error_code']])
            if context_str:
                parts.append(f"Context: {context_str}")
        
        return " | ".join(parts)


class DataFeedException(TradingException):
    """Exception raised for data feed related errors."""
    
    def __init__(self, 
                 message: str,
                 data_source: Optional[str] = None,
                 symbol: Optional[str] = None,
                 interval: Optional[str] = None,
                 **kwargs):
        """
        Initialize data feed exception.
        
        Args:
            message: Error message
            data_source: Data source (e.g., 'binance', 'yahoo')
            symbol: Trading symbol
            interval: Data interval
            **kwargs: Additional arguments for TradingException
        """
        context = kwargs.get('context', {})
        context.update({
            'data_source': data_source,
            'symbol': symbol,
            'interval': interval,
            'component': 'data_feed'
        })
        
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, error_code="DATA_FEED_ERROR", context=context, **kwargs)


class BrokerException(TradingException):
    """Exception raised for broker related errors."""
    
    def __init__(self, 
                 message: str,
                 broker_type: Optional[str] = None,
                 symbol: Optional[str] = None,
                 order_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize broker exception.
        
        Args:
            message: Error message
            broker_type: Type of broker (e.g., 'binance', 'ibkr')
            symbol: Trading symbol
            order_type: Type of order that failed
            **kwargs: Additional arguments for TradingException
        """
        context = kwargs.get('context', {})
        context.update({
            'broker_type': broker_type,
            'symbol': symbol,
            'order_type': order_type,
            'component': 'broker'
        })
        
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, error_code="BROKER_ERROR", context=context, **kwargs)


class StrategyException(TradingException):
    """Exception raised for strategy related errors."""
    
    def __init__(self, 
                 message: str,
                 strategy_name: Optional[str] = None,
                 strategy_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize strategy exception.
        
        Args:
            message: Error message
            strategy_name: Name of the strategy
            strategy_type: Type of strategy (entry/exit)
            **kwargs: Additional arguments for TradingException
        """
        context = kwargs.get('context', {})
        context.update({
            'strategy_name': strategy_name,
            'strategy_type': strategy_type,
            'component': 'strategy'
        })
        
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, error_code="STRATEGY_ERROR", context=context, **kwargs)


class ConfigurationException(TradingException):
    """Exception raised for configuration related errors."""
    
    def __init__(self, 
                 message: str,
                 config_file: Optional[str] = None,
                 config_section: Optional[str] = None,
                 **kwargs):
        """
        Initialize configuration exception.
        
        Args:
            message: Error message
            config_file: Configuration file path
            config_section: Configuration section that failed
            **kwargs: Additional arguments for TradingException
        """
        context = kwargs.get('context', {})
        context.update({
            'config_file': config_file,
            'config_section': config_section,
            'component': 'configuration'
        })
        
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, error_code="CONFIG_ERROR", context=context, **kwargs)


class NetworkException(TradingException):
    """Exception raised for network related errors."""
    
    def __init__(self, 
                 message: str,
                 url: Optional[str] = None,
                 status_code: Optional[int] = None,
                 timeout: Optional[float] = None,
                 **kwargs):
        """
        Initialize network exception.
        
        Args:
            message: Error message
            url: URL that failed
            status_code: HTTP status code
            timeout: Timeout value
            **kwargs: Additional arguments for TradingException
        """
        context = kwargs.get('context', {})
        context.update({
            'url': url,
            'status_code': status_code,
            'timeout': timeout,
            'component': 'network'
        })
        
        # Set retry_after for network errors
        if 'retry_after' not in kwargs:
            kwargs['retry_after'] = 30  # Default 30 seconds
        
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, error_code="NETWORK_ERROR", context=context, **kwargs)


class ValidationException(TradingException):
    """Exception raised for validation errors."""
    
    def __init__(self, 
                 message: str,
                 field: Optional[str] = None,
                 value: Optional[Any] = None,
                 expected_type: Optional[str] = None,
                 **kwargs):
        """
        Initialize validation exception.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            expected_type: Expected data type
            **kwargs: Additional arguments for TradingException
        """
        context = kwargs.get('context', {})
        context.update({
            'field': field,
            'value': str(value) if value is not None else None,
            'expected_type': expected_type,
            'component': 'validation'
        })
        
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, error_code="VALIDATION_ERROR", context=context, **kwargs)


class RecoveryException(TradingException):
    """Exception raised when recovery strategies fail."""
    
    def __init__(self, 
                 message: str,
                 original_error: Optional[Exception] = None,
                 recovery_strategy: Optional[str] = None,
                 **kwargs):
        """
        Initialize recovery exception.
        
        Args:
            message: Error message
            original_error: Original error that triggered recovery
            recovery_strategy: Recovery strategy that failed
            **kwargs: Additional arguments for TradingException
        """
        context = kwargs.get('context', {})
        context.update({
            'original_error': str(original_error) if original_error else None,
            'recovery_strategy': recovery_strategy,
            'component': 'recovery'
        })
        
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, error_code="RECOVERY_ERROR", context=context, **kwargs)


# Specific error types for common scenarios
class InsufficientFundsException(BrokerException):
    """Exception raised when there are insufficient funds for a trade."""
    
    def __init__(self, symbol: str, required_amount: float, available_amount: float, **kwargs):
        message = f"Insufficient funds for {symbol}: required {required_amount}, available {available_amount}"
        context = kwargs.get('context', {})
        context.update({
            'required_amount': required_amount,
            'available_amount': available_amount
        })
        # Remove context from kwargs to avoid duplicate argument
        kwargs.pop('context', None)
        super().__init__(message, symbol=symbol, context=context, **kwargs)


class RateLimitException(NetworkException):
    """Exception raised when API rate limits are exceeded."""
    
    def __init__(self, url: str, retry_after: int = 60, **kwargs):
        message = f"Rate limit exceeded for {url}, retry after {retry_after} seconds"
        super().__init__(message, url=url, retry_after=retry_after, **kwargs)


class ConnectionTimeoutException(NetworkException):
    """Exception raised when network connections timeout."""
    
    def __init__(self, url: str, timeout: float, **kwargs):
        message = f"Connection timeout for {url} after {timeout} seconds"
        super().__init__(message, url=url, timeout=timeout, **kwargs)


class DataUnavailableException(DataFeedException):
    """Exception raised when data is unavailable."""
    
    def __init__(self, symbol: str, interval: str, data_source: str, **kwargs):
        message = f"Data unavailable for {symbol} {interval} from {data_source}"
        super().__init__(message, symbol=symbol, interval=interval, data_source=data_source, **kwargs)


class InvalidOrderException(BrokerException):
    """Exception raised when an order is invalid."""
    
    def __init__(self, symbol: str, order_type: str, reason: str, **kwargs):
        message = f"Invalid {order_type} order for {symbol}: {reason}"
        super().__init__(message, symbol=symbol, order_type=order_type, **kwargs)


# Error code constants for easy reference
class ErrorCodes:
    """Constants for error codes."""
    
    # Data Feed Errors
    DATA_FEED_CONNECTION_FAILED = "DATA_FEED_CONNECTION_FAILED"
    DATA_FEED_AUTHENTICATION_FAILED = "DATA_FEED_AUTHENTICATION_FAILED"
    DATA_FEED_RATE_LIMIT_EXCEEDED = "DATA_FEED_RATE_LIMIT_EXCEEDED"
    DATA_FEED_INVALID_SYMBOL = "DATA_FEED_INVALID_SYMBOL"
    
    # Broker Errors
    BROKER_INSUFFICIENT_FUNDS = "BROKER_INSUFFICIENT_FUNDS"
    BROKER_INVALID_ORDER = "BROKER_INVALID_ORDER"
    BROKER_ORDER_NOT_FOUND = "BROKER_ORDER_NOT_FOUND"
    BROKER_ACCOUNT_SUSPENDED = "BROKER_ACCOUNT_SUSPENDED"
    
    # Strategy Errors
    STRATEGY_INVALID_PARAMETERS = "STRATEGY_INVALID_PARAMETERS"
    STRATEGY_INDICATOR_FAILURE = "STRATEGY_INDICATOR_FAILURE"
    STRATEGY_SIGNAL_GENERATION_FAILED = "STRATEGY_SIGNAL_GENERATION_FAILED"
    
    # Configuration Errors
    CONFIG_MISSING_REQUIRED_FIELD = "CONFIG_MISSING_REQUIRED_FIELD"
    CONFIG_INVALID_VALUE = "CONFIG_INVALID_VALUE"
    CONFIG_FILE_NOT_FOUND = "CONFIG_FILE_NOT_FOUND"
    
    # Network Errors
    NETWORK_CONNECTION_FAILED = "NETWORK_CONNECTION_FAILED"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    NETWORK_RATE_LIMIT = "NETWORK_RATE_LIMIT"
    NETWORK_SERVER_ERROR = "NETWORK_SERVER_ERROR"
    
    # Validation Errors
    VALIDATION_INVALID_TYPE = "VALIDATION_INVALID_TYPE"
    VALIDATION_OUT_OF_RANGE = "VALIDATION_OUT_OF_RANGE"
    VALIDATION_MISSING_REQUIRED = "VALIDATION_MISSING_REQUIRED"
    
    # Recovery Errors
    RECOVERY_STRATEGY_FAILED = "RECOVERY_STRATEGY_FAILED"
    RECOVERY_FALLBACK_FAILED = "RECOVERY_FALLBACK_FAILED"
    RECOVERY_MAX_ATTEMPTS_EXCEEDED = "RECOVERY_MAX_ATTEMPTS_EXCEEDED"


class CircuitBreakerOpenException(TradingException):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, circuit_name: str, last_failure_time: Optional[float] = None):
        context = {
            'circuit_name': circuit_name,
            'last_failure_time': last_failure_time,
            'component': 'circuit_breaker'
        }
        super().__init__(message, error_code="CIRCUIT_BREAKER_OPEN", context=context) 