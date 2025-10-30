"""
Retry utilities for data requests.

This module provides retry mechanisms with exponential backoff for
handling transient failures in data requests.
"""

import time
import random
from typing import Callable, Any, Optional, TypeVar, Union
from functools import wraps
import logging

_logger = logging.getLogger(__name__)

T = TypeVar('T')


def exponential_backoff(
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    factor: float = 2.0,
    jitter: bool = True
) -> Callable[[int], float]:
    """
    Create an exponential backoff function.

    Args:
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        factor: Multiplicative factor for each retry
        jitter: Whether to add random jitter to delays

    Returns:
        Function that takes retry attempt number and returns delay
    """
    def backoff(attempt: int) -> float:
        delay = min(base_delay * (factor ** attempt), max_delay)
        if jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay

    return backoff


def request_with_backoff(
    func: Callable[..., T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> T:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplicative factor for each retry
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback function called before each retry

    Returns:
        Result of the function execution

    Raises:
        Last exception encountered if all retries fail
    """
    backoff = exponential_backoff(base_delay, max_delay, backoff_factor, jitter)
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt == max_attempts - 1:
                _logger.exception("Final retry attempt failed:")
                raise

            delay = backoff(attempt)
            _logger.warning(
                "Attempt %d failed: %s. Retrying in %.2f seconds...",
                attempt + 1, e, delay
            )

            if on_retry:
                on_retry(attempt + 1, e, delay)

            time.sleep(delay)

    # This should never be reached, but just in case
    raise last_exception


def retry_on_exception(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
):
    """
    Decorator for retrying functions on exceptions.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplicative factor for each retry
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exceptions to catch and retry on
        on_retry: Optional callback function called before each retry

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            def wrapped_func() -> T:
                return func(*args, **kwargs)

            return request_with_backoff(
                wrapped_func,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                exceptions=exceptions,
                on_retry=on_retry
            )

        return wrapper
    return decorator


def retry_on_rate_limit(
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 300.0,  # 5 minutes for rate limits
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """
    Decorator specifically for retrying on rate limit errors.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds (longer for rate limits)
        backoff_factor: Multiplicative factor for each retry
        jitter: Whether to add random jitter to delays

    Returns:
        Decorated function with rate limit retry logic
    """
    def on_rate_limit_retry(attempt: int, exception: Exception, delay: float):
        _logger.info(
            "Rate limit hit (attempt %d). Waiting %.2f seconds before retry...",
            attempt, delay
        )

    return retry_on_exception(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_factor=backoff_factor,
        jitter=jitter,
        exceptions=(Exception,),  # Catch all exceptions for rate limits
        on_retry=on_rate_limit_retry
    )
