"""
Error Handling & Resilience System
==================================

This module provides a comprehensive error handling and resilience system for the trading platform.

Features:
- Standardized error handling with custom exception classes
- Retry mechanisms with exponential backoff and circuit breaker patterns
- Error recovery strategies with fallback mechanisms
- Error monitoring and alerting with detailed tracking
- Graceful degradation and fault tolerance

Classes:
- TradingException: Base exception for trading-related errors
- RetryManager: Manages retry logic with exponential backoff
- CircuitBreaker: Implements circuit breaker pattern for API calls
- ErrorRecoveryManager: Handles error recovery strategies
- ErrorMonitor: Monitors and tracks errors for alerting
- ResilienceDecorator: Decorators for adding resilience to functions
"""

from .exceptions import (
    TradingException,
    DataFeedException,
    BrokerException,
    StrategyException,
    ConfigurationException,
    NetworkException,
    ValidationException,
    RecoveryException,
    # Specific exception types
    InsufficientFundsException,
    RateLimitException,
    ConnectionTimeoutException,
    DataUnavailableException,
    InvalidOrderException,
    CircuitBreakerOpenException,
    ErrorCodes
)

from .retry_manager import RetryManager, RetryConfig, RetryStrategy
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from .recovery_manager import ErrorRecoveryManager, RecoveryStrategy, RecoveryConfig
from .error_monitor import ErrorMonitor, ErrorSeverity, ErrorEvent, AlertConfig
from .resilience_decorator import (
    resilient,
    retry_on_failure,
    circuit_breaker,
    fallback,
    timeout,
    resilient_api_call,
    resilient_database_call,
    resilient_strategy_call
)

__all__ = [
    # Exceptions
    'TradingException',
    'DataFeedException', 
    'BrokerException',
    'StrategyException',
    'ConfigurationException',
    'NetworkException',
    'ValidationException',
    'RecoveryException',
    
    # Specific Exception Types
    'InsufficientFundsException',
    'RateLimitException',
    'ConnectionTimeoutException',
    'DataUnavailableException',
    'InvalidOrderException',
    'CircuitBreakerOpenException',
    'ErrorCodes',
    
    # Retry Management
    'RetryManager',
    'RetryConfig',
    'RetryStrategy',
    
    # Circuit Breaker
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitState',
    
    # Recovery Management
    'ErrorRecoveryManager',
    'RecoveryStrategy',
    'RecoveryConfig',
    
    # Error Monitoring
    'ErrorMonitor',
    'ErrorSeverity',
    'ErrorEvent',
    'AlertConfig',
    
    # Decorators
    'resilient',
    'retry_on_failure',
    'circuit_breaker',
    'fallback',
    'timeout',
    'resilient_api_call',
    'resilient_database_call',
    'resilient_strategy_call'
] 