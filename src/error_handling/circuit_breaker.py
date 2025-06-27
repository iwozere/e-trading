"""
Circuit Breaker
==============

This module implements the Circuit Breaker pattern to prevent cascading failures
and provide automatic recovery for external service calls.

Features:
- Three states: CLOSED, OPEN, HALF_OPEN
- Configurable failure thresholds
- Automatic state transitions
- Timeout-based recovery
- Failure rate monitoring
"""

import time
import logging
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from threading import Lock

from .exceptions import TradingException, NetworkException, CircuitBreakerOpenException

_logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation, calls pass through
    OPEN = "OPEN"          # Calls fail fast, no external calls
    HALF_OPEN = "HALF_OPEN"  # Limited calls allowed to test recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    # Failure threshold
    failure_threshold: int = 5  # Number of failures before opening circuit
    failure_window: int = 60    # Time window for failure counting (seconds)
    
    # Recovery settings
    recovery_timeout: int = 60  # Time to wait before attempting recovery (seconds)
    success_threshold: int = 2  # Number of successes needed to close circuit
    
    # Monitoring
    monitor_interval: int = 10  # Interval for monitoring calls (seconds)
    
    # Exceptions that should trigger circuit breaker
    failure_exceptions: tuple = (Exception,)
    
    # Logging
    log_state_changes: bool = True
    log_level: str = "WARNING"


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Calls fail fast, no external calls made
    - HALF_OPEN: Limited calls allowed to test recovery
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker for identification
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self.state = CircuitState.CLOSED
        self.last_failure_time = None
        self.last_state_change = time.time()
        
        # Failure tracking
        self.failure_count = 0
        self.failure_times: List[float] = []
        self.success_count = 0
        
        # Call tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        # Thread safety
        self._lock = Lock()
        
        _logger.info(f"Circuit breaker '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Original exception: If function fails
        """
        with self._lock:
            self._update_state()
            
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN",
                    circuit_name=self.name,
                    last_failure_time=self.last_failure_time
                )
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                self._on_success()
                # Update state after success to check for transitions
                self._update_state()
                return result
                
            except Exception as e:
                self._on_failure(e)
                raise e
    
    def _update_state(self):
        """Update circuit breaker state based on current conditions."""
        current_time = time.time()
        
        # Clean up old failures outside the window
        self._cleanup_old_failures(current_time)
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                current_time - self.last_failure_time >= self.config.recovery_timeout):
                self._transition_to_half_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Check if we have enough successes to close
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
    
    def _cleanup_old_failures(self, current_time: float):
        """Remove failures outside the failure window."""
        cutoff_time = current_time - self.config.failure_window
        self.failure_times = [t for t in self.failure_times if t > cutoff_time]
        self.failure_count = len(self.failure_times)
    
    def _on_success(self):
        """Handle successful call."""
        current_time = time.time()
        
        self.total_calls += 1
        self.successful_calls += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            _logger.info(f"Circuit breaker '{self.name}' success in HALF_OPEN state: {self.success_count}/{self.config.success_threshold}")
        
        # Reset failure count in CLOSED state
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
            self.failure_times.clear()
    
    def _on_failure(self, exception: Exception):
        """Handle failed call."""
        current_time = time.time()
        
        self.total_calls += 1
        self.failed_calls += 1
        
        # Check if this exception should trigger circuit breaker
        if not isinstance(exception, self.config.failure_exceptions):
            return
        
        self.failure_count += 1
        self.failure_times.append(current_time)
        self.last_failure_time = current_time
        
        _logger.warning(f"Circuit breaker '{self.name}' failure: {self.failure_count}/{self.config.failure_threshold}")
        
        # Check if we should open the circuit
        if (self.state == CircuitState.CLOSED and 
            self.failure_count >= self.config.failure_threshold):
            self._transition_to_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # Reset to OPEN state on failure in HALF_OPEN
            self._transition_to_open()
    
    def _transition_to_open(self):
        """Transition circuit to OPEN state."""
        if self.state != CircuitState.OPEN:
            self.state = CircuitState.OPEN
            self.last_state_change = time.time()
            self.success_count = 0
            
            if self.config.log_state_changes:
                _logger.warning(f"Circuit breaker '{self.name}' transitioned to OPEN state")
    
    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        if self.state != CircuitState.HALF_OPEN:
            self.state = CircuitState.HALF_OPEN
            self.last_state_change = time.time()
            self.success_count = 0
            
            if self.config.log_state_changes:
                _logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN state")
    
    def _transition_to_closed(self):
        """Transition circuit to CLOSED state."""
        if self.state != CircuitState.CLOSED:
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()
            self.failure_count = 0
            self.failure_times.clear()
            self.success_count = 0
            
            if self.config.log_state_changes:
                _logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED state")
    
    def force_open(self):
        """Force circuit to OPEN state."""
        with self._lock:
            self._transition_to_open()
    
    def force_close(self):
        """Force circuit to CLOSED state."""
        with self._lock:
            self._transition_to_closed()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'total_calls': self.total_calls,
                'successful_calls': self.successful_calls,
                'failed_calls': self.failed_calls,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'last_state_change': self.last_state_change,
                'failure_rate': self.failed_calls / self.total_calls if self.total_calls > 0 else 0.0
            }
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        with self._lock:
            self._update_state()
            return self.state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        with self._lock:
            self._update_state()
            return self.state == CircuitState.CLOSED
    
    def is_half_open(self) -> bool:
        """Check if circuit is half open."""
        with self._lock:
            self._update_state()
            return self.state == CircuitState.HALF_OPEN


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for adding circuit breaker functionality to functions.
    
    Args:
        name: Name of the circuit breaker
        config: Circuit breaker configuration
        
    Example:
        @circuit_breaker("api_calls", CircuitBreakerConfig(failure_threshold=5))
        def api_call():
            # Function that may fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        cb = CircuitBreaker(name, config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        # Add circuit breaker to wrapper for access
        wrapper.circuit_breaker = cb
        return wrapper
    
    return decorator


# Pre-configured circuit breaker decorators
def circuit_breaker_for_api(name: str, failure_threshold: int = 5):
    """Circuit breaker decorator specifically for API calls."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        failure_window=60,
        recovery_timeout=60,
        failure_exceptions=(NetworkException, ConnectionError, TimeoutError)
    )
    return circuit_breaker(name, config)


def circuit_breaker_for_database(name: str, failure_threshold: int = 3):
    """Circuit breaker decorator specifically for database operations."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        failure_window=30,
        recovery_timeout=30,
        failure_exceptions=(Exception,)  # Most database exceptions
    )
    return circuit_breaker(name, config)


# Global circuit breaker registry
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._lock = Lock()
    
    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker(name, config)
            return self._circuit_breakers[name]
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._circuit_breakers.get(name)
    
    def list_all(self) -> List[str]:
        """List all circuit breaker names."""
        with self._lock:
            return list(self._circuit_breakers.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        with self._lock:
            return {name: cb.get_stats() for name, cb in self._circuit_breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers to CLOSED state."""
        with self._lock:
            for cb in self._circuit_breakers.values():
                cb.force_close()


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry() 