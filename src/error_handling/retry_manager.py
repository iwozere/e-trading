"""
Retry Manager
============

This module provides a comprehensive retry mechanism with:
- Exponential backoff with jitter
- Configurable retry strategies
- Retry condition evaluation
- Retry statistics tracking
- Circuit breaker integration
"""

import time
import random
import logging
from typing import Callable, Optional, Dict, Any, List, Union
from functools import wraps
from dataclasses import dataclass
from enum import Enum

from .exceptions import TradingException

_logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    jitter_factor: float = 0.1  # 10% jitter
    backoff_factor: float = 2.0  # for exponential backoff
    
    # Retry conditions
    retry_on_exceptions: tuple = (Exception,)
    retry_on_result: Optional[Callable] = None  # Function to evaluate result
    
    # Logging
    log_retries: bool = True
    log_level: str = "WARNING"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        if self.backoff_factor < 1:
            raise ValueError("backoff_factor must be at least 1")


class RetryManager:
    """
    Manages retry logic with configurable strategies and exponential backoff.
    
    Features:
    - Multiple retry strategies (fixed, exponential, linear, fibonacci)
    - Jitter to prevent thundering herd
    - Configurable retry conditions
    - Retry statistics tracking
    - Integration with circuit breaker
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry manager.
        
        Args:
            config: Retry configuration, uses defaults if None
        """
        self.config = config or RetryConfig()
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retry_attempts': 0,
            'total_retry_time': 0.0
        }
    
    def execute(self, 
                func: Callable, 
                *args, 
                context: Optional[Dict[str, Any]] = None,
                **kwargs) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            context: Additional context for logging
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        context = context or {}
        last_exception = None
        start_time = time.time()
        
        self.stats['total_calls'] += 1
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Check if result indicates failure
                if self.config.retry_on_result and self.config.retry_on_result(result):
                    raise ValueError(f"Function returned failure result: {result}")
                
                # Success
                self.stats['successful_calls'] += 1
                if attempt > 1:
                    self._log_retry_success(attempt, context)
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception
                if not self._should_retry(e):
                    self.stats['failed_calls'] += 1
                    raise e
                
                # Only increment retry attempts if we're actually going to retry
                self.stats['retry_attempts'] += 1
                
                # Check if we've exhausted retries
                if attempt >= self.config.max_attempts:
                    self.stats['failed_calls'] += 1
                    self._log_final_failure(attempt, e, context)
                    raise e
                
                # Calculate delay
                delay = self._calculate_delay(attempt)
                
                # Log retry attempt
                self._log_retry_attempt(attempt, e, delay, context)
                
                # Wait before retry
                time.sleep(delay)
        
        # This should never be reached, but just in case
        self.stats['failed_calls'] += 1
        raise last_exception
    
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if exception should trigger a retry."""
        # Check if exception type is in retry list
        if not isinstance(exception, self.config.retry_on_exceptions):
            return False
        
        # Check if it's a TradingException with retry_after
        if isinstance(exception, TradingException):
            return exception.should_retry()
        
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.backoff_factor ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            delay = self.config.base_delay * self._fibonacci(attempt)
        else:
            delay = self.config.base_delay
        
        # Apply max delay limit
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _log_retry_attempt(self, attempt: int, exception: Exception, delay: float, context: Dict[str, Any]):
        """Log retry attempt."""
        if not self.config.log_retries:
            return
        
        log_level = getattr(logging, self.config.log_level.upper(), logging.WARNING)
        _logger.log(log_level, 
                   f"Retry attempt {attempt}/{self.config.max_attempts} failed: {type(exception).__name__}: {str(exception)}. "
                   f"Retrying in {delay:.2f}s. Context: {context}")
    
    def _log_retry_success(self, attempt: int, context: Dict[str, Any]):
        """Log successful retry."""
        if not self.config.log_retries:
            return
        
        log_level = getattr(logging, self.config.log_level.upper(), logging.WARNING)
        _logger.log(log_level, 
                   f"Retry succeeded after {attempt} attempts. Context: {context}")
    
    def _log_final_failure(self, attempt: int, exception: Exception, context: Dict[str, Any]):
        """Log final failure after all retries."""
        if not self.config.log_retries:
            return
        
        _logger.error(f"All {attempt} retry attempts failed. Final error: {type(exception).__name__}: {str(exception)}. "
                     f"Context: {context}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        stats = self.stats.copy()
        if stats['total_calls'] > 0:
            stats['success_rate'] = stats['successful_calls'] / stats['total_calls']
            stats['retry_rate'] = stats['retry_attempts'] / stats['total_calls']
        else:
            stats['success_rate'] = 0.0
            stats['retry_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset retry statistics."""
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retry_attempts': 0,
            'total_retry_time': 0.0
        }


def retry(config: Optional[RetryConfig] = None):
    """
    Decorator for adding retry functionality to functions.
    
    Args:
        config: Retry configuration
        
    Example:
        @retry(RetryConfig(max_attempts=3, base_delay=1.0))
        def my_function():
            # Function that may fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        retry_manager = RetryManager(config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return retry_manager.execute(func, *args, **kwargs)
        
        # Add retry manager to wrapper for access to stats
        wrapper.retry_manager = retry_manager
        return wrapper
    
    return decorator


# Pre-configured retry decorators for common use cases
def retry_on_network_error(max_attempts: int = 3, base_delay: float = 1.0):
    """Retry decorator specifically for network errors."""
    from .exceptions import NetworkException
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        retry_on_exceptions=(NetworkException, ConnectionError, TimeoutError)
    )
    return retry(config)


def retry_on_api_error(max_attempts: int = 3, base_delay: float = 2.0):
    """Retry decorator specifically for API errors."""
    from .exceptions import BrokerException, DataFeedException
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        retry_on_exceptions=(BrokerException, DataFeedException)
    )
    return retry(config)


def retry_on_validation_error(max_attempts: int = 2, base_delay: float = 0.5):
    """Retry decorator for validation errors (usually don't retry much)."""
    from .exceptions import ValidationException
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=RetryStrategy.FIXED,
        retry_on_exceptions=(ValidationException,)
    )
    return retry(config) 