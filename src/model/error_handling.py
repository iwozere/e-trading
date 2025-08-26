"""
Error handling and resilience models.

Includes:
- Circuit breaker, error event, alert, recovery, and retry config dataclasses
- Circuit state, error severity, recovery and retry strategy enums
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation, calls pass through
    OPEN = "OPEN"          # Calls fail fast, no external calls
    HALF_OPEN = "HALF_OPEN"  # Limited calls allowed to test recovery

class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    RESTART = "restart"
    IGNORE = "ignore"
    ALERT = "alert"

class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"

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

@dataclass
class ErrorEvent:
    """Represents an error event."""

    timestamp: datetime
    error: Exception
    severity: ErrorSeverity
    component: str
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.now(timezone.utc).isoformat(),
            'error_type': type(self.error).__name__,
            'error_message': str(self.error),
            'severity': self.severity.value,
            'component': self.component,
            'context': self.context,
            'stack_trace': self.stack_trace,
            'user_id': self.user_id,
            'session_id': self.session_id
        }


@dataclass
class AlertConfig:
    """Configuration for error alerting."""

    severity_threshold: ErrorSeverity = ErrorSeverity.ERROR
    error_rate_threshold: float = 0.1  # 10% error rate
    time_window: int = 300  # 5 minutes
    max_alerts_per_window: int = 10
    alert_functions: List[Callable] = field(default_factory=list)

    # Rate limiting
    alert_cooldown: int = 60  # seconds between alerts for same error type

@dataclass
class RecoveryConfig:
    """Configuration for recovery behavior."""

    strategy: RecoveryStrategy
    max_attempts: int = 3
    timeout: float = 30.0  # seconds
    fallback_function: Optional[Callable] = None
    degrade_function: Optional[Callable] = None
    alert_function: Optional[Callable] = None

    # Strategy-specific settings
    retry_delay: float = 1.0
    restart_delay: float = 5.0

    # Monitoring
    log_recovery: bool = True
    track_metrics: bool = True

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
