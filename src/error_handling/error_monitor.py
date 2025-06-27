"""
Error Monitor
============

This module provides comprehensive error monitoring and alerting capabilities
for the trading platform.

Features:
- Error tracking and categorization
- Severity-based alerting
- Error rate monitoring
- Error reporting and analytics
- Integration with notification systems
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import threading

from .exceptions import TradingException

_logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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
            'timestamp': self.timestamp.isoformat(),
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


class ErrorMonitor:
    """
    Monitors and tracks errors for alerting and reporting.
    
    Features:
    - Error event tracking
    - Severity-based filtering
    - Error rate monitoring
    - Alert generation
    - Error reporting
    """
    
    def __init__(self, alert_config: Optional[AlertConfig] = None):
        """
        Initialize error monitor.
        
        Args:
            alert_config: Alert configuration
        """
        self.alert_config = alert_config or AlertConfig()
        
        # Error tracking
        self.error_events: deque = deque(maxlen=10000)  # Keep last 10k errors
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.severity_counts: Dict[ErrorSeverity, int] = defaultdict(int)
        self.component_counts: Dict[str, int] = defaultdict(int)
        
        # Alert tracking
        self.last_alert_time: Dict[str, float] = {}
        self.alert_count = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        _logger.info("Error monitor initialized")
    
    def record_error(self, 
                    error: Exception, 
                    severity: ErrorSeverity = ErrorSeverity.ERROR,
                    component: str = "unknown",
                    context: Optional[Dict[str, Any]] = None,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None) -> None:
        """
        Record an error event.
        
        Args:
            error: The exception that occurred
            severity: Error severity level
            component: Component where error occurred
            context: Additional context information
            user_id: User ID if applicable
            session_id: Session ID if applicable
        """
        with self._lock:
            # Create error event
            event = ErrorEvent(
                timestamp=datetime.utcnow(),
                error=error,
                severity=severity,
                component=component,
                context=context or {},
                stack_trace=self._get_stack_trace(error),
                user_id=user_id,
                session_id=session_id
            )
            
            # Store error event
            self.error_events.append(event)
            
            # Update counters
            error_type = type(error).__name__
            self.error_counts[error_type] += 1
            self.severity_counts[severity] += 1
            self.component_counts[component] += 1
            
            # Check if we should generate alert
            self._check_alert_conditions(event)
            
            # Log error
            log_level = getattr(logging, severity.value)
            _logger.log(log_level, 
                       f"Error in {component}: {error_type}: {str(error)}")
    
    def _get_stack_trace(self, error: Exception) -> Optional[str]:
        """Get stack trace for error."""
        try:
            import traceback
            return ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        except:
            return None
    
    def _check_alert_conditions(self, event: ErrorEvent) -> None:
        """Check if alert conditions are met."""
        current_time = time.time()
        
        # Check severity threshold - compare enum members directly
        severity_order = {
            ErrorSeverity.DEBUG: 0,
            ErrorSeverity.INFO: 1,
            ErrorSeverity.WARNING: 2,
            ErrorSeverity.ERROR: 3,
            ErrorSeverity.CRITICAL: 4
        }
        
        if severity_order[event.severity] < severity_order[self.alert_config.severity_threshold]:
            return
        
        # Check error rate
        if self._calculate_error_rate() > self.alert_config.error_rate_threshold:
            self._generate_alert("High error rate detected", event)
            return
        
        # Check alert rate limiting
        error_type = type(event.error).__name__
        last_alert = self.last_alert_time.get(error_type, 0)
        
        if current_time - last_alert > self.alert_config.alert_cooldown:
            self._generate_alert(f"Error occurred: {error_type}", event)
            self.last_alert_time[error_type] = current_time
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        window_start = datetime.utcnow() - timedelta(seconds=self.alert_config.time_window)
        
        # Count errors in window
        error_count = sum(1 for event in self.error_events 
                         if event.timestamp >= window_start)
        
        # Estimate total requests (this is a simplified approach)
        total_requests = max(error_count * 10, 1)  # Assume 10% error rate max
        
        return error_count / total_requests
    
    def _generate_alert(self, message: str, event: ErrorEvent) -> None:
        """Generate and send alert."""
        if self.alert_count >= self.alert_config.max_alerts_per_window:
            return
        
        alert_data = {
            'message': message,
            'error_event': event.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Send alerts via configured functions
        for alert_func in self.alert_config.alert_functions:
            try:
                alert_func(alert_data)
            except Exception as e:
                _logger.error(f"Failed to send alert: {e}")
        
        self.alert_count += 1
        _logger.warning(f"Alert generated: {message}")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                # Reset alert count periodically
                time.sleep(self.alert_config.time_window)
                self.alert_count = 0
                
            except Exception as e:
                _logger.error(f"Error in monitor loop: {e}")
    
    def get_error_stats(self, 
                       time_window: Optional[int] = None,
                       component: Optional[str] = None) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Args:
            time_window: Time window in seconds (None for all time)
            component: Filter by component (None for all components)
            
        Returns:
            Dictionary with error statistics
        """
        with self._lock:
            if time_window:
                cutoff_time = datetime.utcnow() - timedelta(seconds=time_window)
                events = [e for e in self.error_events if e.timestamp >= cutoff_time]
            else:
                events = list(self.error_events)
            
            if component:
                events = [e for e in events if e.component == component]
            
            # Calculate statistics
            total_errors = len(events)
            if total_errors == 0:
                return {
                    'total_errors': 0,
                    'error_rate': 0.0,
                    'severity_distribution': {},
                    'component_distribution': {},
                    'top_errors': []
                }
            
            # Severity distribution
            severity_dist = defaultdict(int)
            for event in events:
                severity_dist[event.severity.value] += 1
            
            # Component distribution
            component_dist = defaultdict(int)
            for event in events:
                component_dist[event.component] += 1
            
            # Top errors
            error_type_counts = defaultdict(int)
            for event in events:
                error_type_counts[type(event.error).__name__] += 1
            
            top_errors = sorted(error_type_counts.items(), 
                              key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_errors': total_errors,
                'error_rate': total_errors / max(len(self.error_events), 1),
                'severity_distribution': dict(severity_dist),
                'component_distribution': dict(component_dist),
                'top_errors': top_errors
            }
    
    def get_recent_errors(self, 
                         limit: int = 100,
                         severity: Optional[ErrorSeverity] = None,
                         component: Optional[str] = None) -> List[ErrorEvent]:
        """
        Get recent error events.
        
        Args:
            limit: Maximum number of errors to return
            severity: Filter by severity
            component: Filter by component
            
        Returns:
            List of recent error events
        """
        with self._lock:
            events = list(self.error_events)
            
            if severity:
                events = [e for e in events if e.severity == severity]
            
            if component:
                events = [e for e in events if e.component == component]
            
            return events[-limit:]
    
    def generate_error_report(self, 
                            time_window: Optional[int] = None,
                            format: str = "json") -> str:
        """
        Generate error report.
        
        Args:
            time_window: Time window in seconds
            format: Report format ("json" or "text")
            
        Returns:
            Formatted error report
        """
        stats = self.get_error_stats(time_window)
        recent_errors = self.get_recent_errors(50)
        
        if format == "json":
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'time_window': time_window,
                'statistics': stats,
                'recent_errors': [e.to_dict() for e in recent_errors]
            }
            return json.dumps(report, indent=2)
        
        elif format == "text":
            lines = [
                f"Error Report - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Time Window: {time_window}s" if time_window else "Time Window: All time",
                "",
                "Statistics:",
                f"  Total Errors: {stats['total_errors']}",
                f"  Error Rate: {stats['error_rate']:.2%}",
                "",
                "Severity Distribution:",
            ]
            
            for severity, count in stats['severity_distribution'].items():
                lines.append(f"  {severity}: {count}")
            
            lines.extend([
                "",
                "Component Distribution:",
            ])
            
            for component, count in stats['component_distribution'].items():
                lines.append(f"  {component}: {count}")
            
            lines.extend([
                "",
                "Top Errors:",
            ])
            
            for error_type, count in stats['top_errors']:
                lines.append(f"  {error_type}: {count}")
            
            lines.extend([
                "",
                "Recent Errors:",
            ])
            
            for event in recent_errors[-10:]:  # Last 10 errors
                lines.append(f"  {event.timestamp.strftime('%H:%M:%S')} - "
                           f"{event.component} - {type(event.error).__name__}: {str(event.error)}")
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def add_alert_function(self, alert_func: Callable) -> None:
        """Add alert function for sending notifications."""
        self.alert_config.alert_functions.append(alert_func)
        _logger.info(f"Added alert function: {alert_func.__name__}")
    
    def clear_errors(self) -> None:
        """Clear all error events."""
        with self._lock:
            self.error_events.clear()
            self.error_counts.clear()
            self.severity_counts.clear()
            self.component_counts.clear()
            _logger.info("All error events cleared")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread."""
        self._monitoring = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        _logger.info("Error monitoring stopped")


# Global error monitor instance
error_monitor = ErrorMonitor()


def monitor_errors(severity: ErrorSeverity = ErrorSeverity.ERROR,
                  component: Optional[str] = None):
    """
    Decorator for monitoring errors in functions.
    
    Args:
        severity: Minimum severity to monitor
        component: Component name (auto-detected if None)
        
    Example:
        @monitor_errors(ErrorSeverity.WARNING, "api")
        def api_call():
            # Function that may fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Auto-detect component if not provided
                comp = component or func.__module__.split('.')[-1]
                
                # Determine severity based on exception type
                if isinstance(e, TradingException):
                    sev = ErrorSeverity.ERROR
                elif isinstance(e, (ValueError, TypeError)):
                    sev = ErrorSeverity.WARNING
                else:
                    sev = ErrorSeverity.ERROR
                
                # Record error if it meets severity threshold
                if sev.value >= severity.value:
                    error_monitor.record_error(
                        error=e,
                        severity=sev,
                        component=comp,
                        context={
                            'function': func.__name__,
                            'args': str(args),
                            'kwargs': str(kwargs)
                        }
                    )
                
                raise e
        
        return wrapper
    
    return decorator


# Pre-configured monitoring decorators
def monitor_api_errors(func: Callable) -> Callable:
    """Monitor API-related errors."""
    return monitor_errors(ErrorSeverity.WARNING, "api")(func)


def monitor_database_errors(func: Callable) -> Callable:
    """Monitor database-related errors."""
    return monitor_errors(ErrorSeverity.ERROR, "database")(func)


def monitor_strategy_errors(func: Callable) -> Callable:
    """Monitor strategy-related errors."""
    return monitor_errors(ErrorSeverity.ERROR, "strategy")(func) 