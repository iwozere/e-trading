"""
Resilience Decorators
====================

This module provides decorators that combine multiple resilience features
for easy application to functions and methods.

Features:
- Combined retry, circuit breaker, and fallback
- Timeout protection
- Graceful degradation
- Error monitoring integration
- Easy-to-use decorators
"""

import time
import logging
import functools
from typing import Callable, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from .retry_manager import RetryManager, RetryConfig
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .recovery_manager import RecoveryStrategy
from .error_monitor import error_monitor, ErrorSeverity

_logger = logging.getLogger(__name__)


def resilient(retry_config: Optional[RetryConfig] = None,
              circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
              fallback_func: Optional[Callable] = None,
              timeout: Optional[float] = None,
              monitor_errors: bool = True,
              component: Optional[str] = None):
    """
    Comprehensive resilience decorator combining multiple features.
    
    Args:
        retry_config: Retry configuration
        circuit_breaker_config: Circuit breaker configuration
        fallback_func: Fallback function
        timeout: Timeout in seconds
        monitor_errors: Whether to monitor errors
        component: Component name for monitoring
        
    Example:
        @resilient(
            retry_config=RetryConfig(max_attempts=3),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
            fallback_func=backup_api,
            timeout=30.0
        )
        def api_call():
            # Function with full resilience
            pass
    """
    def decorator(func: Callable) -> Callable:
        # Create retry manager if configured
        retry_manager = None
        if retry_config:
            retry_manager = RetryManager(retry_config)
        
        # Create circuit breaker if configured
        circuit_breaker = None
        if circuit_breaker_config:
            circuit_breaker = CircuitBreaker(
                name=f"{func.__module__}.{func.__name__}",
                config=circuit_breaker_config
            )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Auto-detect component if not provided
            comp = component or func.__module__.split('.')[-1]
            
            # Define the function to execute with timeout
            def execute_with_timeout():
                if timeout:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func, *args, **kwargs)
                        try:
                            return future.result(timeout=timeout)
                        except FutureTimeoutError:
                            raise TimeoutError(f"Function {func.__name__} timed out after {timeout}s")
                else:
                    return func(*args, **kwargs)
            
            # Define the function to execute with circuit breaker
            def execute_with_circuit_breaker():
                if circuit_breaker:
                    return circuit_breaker.call(execute_with_timeout)
                else:
                    return execute_with_timeout()
            
            # Define the function to execute with retry
            def execute_with_retry():
                if retry_manager:
                    return retry_manager.execute(execute_with_circuit_breaker)
                else:
                    return execute_with_circuit_breaker()
            
            try:
                result = execute_with_retry()
                
                # Log success if monitoring
                if monitor_errors:
                    _logger.debug(f"Function {func.__name__} executed successfully")
                
                return result
                
            except Exception as e:
                # Monitor error if enabled
                if monitor_errors:
                    error_monitor.record_error(
                        error=e,
                        severity=ErrorSeverity.ERROR,
                        component=comp,
                        context={
                            'function': func.__name__,
                            'args': str(args),
                            'kwargs': str(kwargs)
                        }
                    )
                
                # Try fallback if available
                if fallback_func:
                    try:
                        _logger.warning(f"Using fallback for {func.__name__}: {str(e)}")
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        _logger.error(f"Fallback also failed for {func.__name__}: {str(fallback_error)}")
                        raise fallback_error
                else:
                    raise e
        
        # Add resilience info to wrapper
        wrapper.resilience_info = {
            'retry_manager': retry_manager,
            'circuit_breaker': circuit_breaker,
            'fallback_func': fallback_func,
            'timeout': timeout,
            'monitor_errors': monitor_errors,
            'component': component
        }
        
        return wrapper
    
    return decorator


def retry_on_failure(max_attempts: int = 3,
                    base_delay: float = 1.0,
                    strategy: str = "exponential",
                    retry_on_exceptions: tuple = (Exception,)):
    """
    Decorator for adding retry functionality.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries
        strategy: Retry strategy ("fixed", "exponential", "linear")
        retry_on_exceptions: Exceptions to retry on
        
    Example:
        @retry_on_failure(max_attempts=3, base_delay=2.0)
        def api_call():
            # Function that may fail
            pass
    """
    from .retry_manager import RetryStrategy
    
    strategy_map = {
        "fixed": RetryStrategy.FIXED,
        "exponential": RetryStrategy.EXPONENTIAL,
        "linear": RetryStrategy.LINEAR
    }
    
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        strategy=strategy_map.get(strategy, RetryStrategy.EXPONENTIAL),
        retry_on_exceptions=retry_on_exceptions
    )
    
    return resilient(retry_config=retry_config)


def circuit_breaker(name: Optional[str] = None,
                   failure_threshold: int = 5,
                   recovery_timeout: int = 60,
                   failure_exceptions: tuple = (Exception,)):
    """
    Decorator for adding circuit breaker functionality.
    
    Args:
        name: Circuit breaker name (auto-generated if None)
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before attempting recovery
        failure_exceptions: Exceptions that trigger circuit breaker
        
    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def api_call():
            # Function with circuit breaker protection
            pass
    """
    def decorator(func: Callable) -> Callable:
        cb_name = name or f"{func.__module__}.{func.__name__}"
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            failure_exceptions=failure_exceptions
        )
        
        return resilient(circuit_breaker_config=circuit_breaker_config)(func)
    
    return decorator


def fallback(fallback_func: Callable):
    """
    Decorator for adding fallback functionality.
    
    Args:
        fallback_func: Function to call if main function fails
        
    Example:
        @fallback(backup_api)
        def api_call():
            # Function with fallback
            pass
    """
    return resilient(fallback_func=fallback_func)


def timeout(timeout_seconds: float):
    """
    Decorator for adding timeout functionality.
    
    Args:
        timeout_seconds: Timeout in seconds
        
    Example:
        @timeout(30.0)
        def slow_function():
            # Function with timeout protection
            pass
    """
    return resilient(timeout=timeout_seconds)


# Pre-configured decorators for common use cases
def resilient_api_call(max_attempts: int = 3,
                      failure_threshold: int = 5,
                      timeout_seconds: float = 30.0,
                      fallback_func: Optional[Callable] = None):
    """
    Pre-configured resilience decorator for API calls.
    
    Args:
        max_attempts: Maximum retry attempts
        failure_threshold: Circuit breaker failure threshold
        timeout_seconds: Timeout in seconds
        fallback_func: Fallback function
        
    Example:
        @resilient_api_call(max_attempts=3, timeout_seconds=30)
        def api_call():
            # API call with comprehensive resilience
            pass
    """
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=2.0,
        strategy="exponential"
    )
    
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=60,
        failure_exceptions=(Exception,)
    )
    
    return resilient(
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        fallback_func=fallback_func,
        timeout=timeout_seconds,
        component="api"
    )


def resilient_database_call(max_attempts: int = 2,
                           failure_threshold: int = 3,
                           timeout_seconds: float = 10.0):
    """
    Pre-configured resilience decorator for database calls.
    
    Args:
        max_attempts: Maximum retry attempts
        failure_threshold: Circuit breaker failure threshold
        timeout_seconds: Timeout in seconds
        
    Example:
        @resilient_database_call()
        def db_query():
            # Database query with resilience
            pass
    """
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=1.0,
        strategy="fixed"
    )
    
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=30,
        failure_exceptions=(Exception,)
    )
    
    return resilient(
        retry_config=retry_config,
        circuit_breaker_config=circuit_breaker_config,
        timeout=timeout_seconds,
        component="database"
    )


def resilient_strategy_call(max_attempts: int = 2,
                           timeout_seconds: float = 5.0):
    """
    Pre-configured resilience decorator for strategy calls.
    
    Args:
        max_attempts: Maximum retry attempts
        timeout_seconds: Timeout in seconds
        
    Example:
        @resilient_strategy_call()
        def strategy_signal():
            # Strategy signal generation with resilience
            pass
    """
    retry_config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=0.5,
        strategy="fixed"
    )
    
    return resilient(
        retry_config=retry_config,
        timeout=timeout_seconds,
        component="strategy"
    )


# Utility decorators for specific scenarios
def with_caching(cache_func: Callable, ttl: int = 300):
    """
    Decorator for adding caching with resilience.
    
    Args:
        cache_func: Cache function (get/set)
        ttl: Time to live in seconds
        
    Example:
        @with_caching(redis_cache, ttl=300)
        def expensive_call():
            # Expensive call with caching
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache first
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            try:
                cached_result = cache_func.get(cache_key)
                if cached_result is not None:
                    return cached_result
            except Exception as e:
                _logger.warning(f"Cache get failed: {e}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            try:
                cache_func.set(cache_key, result, ttl)
            except Exception as e:
                _logger.warning(f"Cache set failed: {e}")
            
            return result
        
        return wrapper
    
    return decorator


def with_rate_limiting(max_calls: int, time_window: int):
    """
    Decorator for adding rate limiting.
    
    Args:
        max_calls: Maximum calls allowed
        time_window: Time window in seconds
        
    Example:
        @with_rate_limiting(max_calls=100, time_window=60)
        def api_call():
            # API call with rate limiting
            pass
    """
    def decorator(func: Callable) -> Callable:
        call_times = []
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove old calls outside window
            call_times[:] = [t for t in call_times if current_time - t < time_window]
            
            # Check rate limit
            if len(call_times) >= max_calls:
                raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_window}s")
            
            # Record call
            call_times.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def with_metrics(metric_func: Callable):
    """
    Decorator for adding metrics collection.
    
    Args:
        metric_func: Function to record metrics
        
    Example:
        @with_metrics(prometheus_metrics)
        def api_call():
            # API call with metrics
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metric
                metric_func.record_success(func.__name__, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failure metric
                metric_func.record_failure(func.__name__, duration, type(e).__name__)
                
                raise e
        
        return wrapper
    
    return decorator 