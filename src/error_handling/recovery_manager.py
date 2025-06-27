"""
Error Recovery Manager
=====================

This module provides error recovery strategies and fallback mechanisms
for handling failures gracefully and maintaining system resilience.

Features:
- Multiple recovery strategies
- Fallback mechanisms
- Graceful degradation
- Recovery monitoring
- Automatic recovery execution
"""

import time
import logging
from typing import Callable, Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
from functools import wraps

from .exceptions import TradingException, RecoveryException

_logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    RESTART = "restart"
    IGNORE = "ignore"
    ALERT = "alert"


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


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and fallback mechanisms.
    
    Provides:
    - Multiple recovery strategies
    - Fallback mechanisms
    - Graceful degradation
    - Recovery monitoring
    """
    
    def __init__(self):
        """Initialize recovery manager."""
        self.recovery_configs: Dict[str, RecoveryConfig] = {}
        self.metrics = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'recovery_time': 0.0
        }
    
    def register_recovery(self, 
                         error_type: str, 
                         config: RecoveryConfig) -> None:
        """
        Register recovery configuration for an error type.
        
        Args:
            error_type: Type of error (e.g., 'network', 'database', 'api')
            config: Recovery configuration
        """
        self.recovery_configs[error_type] = config
        _logger.info(f"Registered recovery strategy for {error_type}: {config.strategy.value}")
    
    def execute_recovery(self, 
                        error: Exception, 
                        context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute recovery strategy for an error.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            Recovery result or None if recovery failed
            
        Raises:
            RecoveryException: If recovery fails
        """
        context = context or {}
        
        # Use context component if provided, otherwise classify error
        error_type = context.get('component') or self._classify_error(error)
        
        config = self.recovery_configs.get(error_type)
        
        if not config:
            _logger.warning(f"No recovery strategy registered for {error_type}")
            return None
        
        start_time = time.time()
        self.metrics['total_recoveries'] += 1
        
        try:
            result = self._execute_strategy(config, error, context)
            self.metrics['successful_recoveries'] += 1
            self.metrics['recovery_time'] += time.time() - start_time
            
            if config.log_recovery:
                _logger.info(f"Recovery successful for {error_type}: {config.strategy.value}")
            
            return result
            
        except Exception as recovery_error:
            self.metrics['failed_recoveries'] += 1
            self.metrics['recovery_time'] += time.time() - start_time
            
            if config.log_recovery:
                _logger.error(f"Recovery failed for {error_type}: {str(recovery_error)}")
            
            raise RecoveryException(
                f"Recovery strategy {config.strategy.value} failed for {error_type}",
                original_error=error,
                recovery_strategy=config.strategy.value
            )
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for recovery strategy selection."""
        if isinstance(error, TradingException):
            # Use component from TradingException
            return error.context.get('component', 'unknown')
        
        # Classify based on exception type
        error_type = type(error).__name__.lower()
        
        if 'network' in error_type or 'connection' in error_type:
            return 'network'
        elif 'database' in error_type or 'sql' in error_type:
            return 'database'
        elif 'api' in error_type or 'http' in error_type:
            return 'api'
        elif 'validation' in error_type:
            return 'validation'
        elif 'timeout' in error_type:
            return 'timeout'
        else:
            return 'general'
    
    def _execute_strategy(self, 
                         config: RecoveryConfig, 
                         error: Exception, 
                         context: Dict[str, Any]) -> Any:
        """Execute specific recovery strategy."""
        if config.strategy == RecoveryStrategy.RETRY:
            return self._execute_retry(config, error, context)
        elif config.strategy == RecoveryStrategy.FALLBACK:
            return self._execute_fallback(config, error, context)
        elif config.strategy == RecoveryStrategy.DEGRADE:
            return self._execute_degrade(config, error, context)
        elif config.strategy == RecoveryStrategy.RESTART:
            return self._execute_restart(config, error, context)
        elif config.strategy == RecoveryStrategy.IGNORE:
            return self._execute_ignore(config, error, context)
        elif config.strategy == RecoveryStrategy.ALERT:
            return self._execute_alert(config, error, context)
        else:
            raise ValueError(f"Unknown recovery strategy: {config.strategy}")
    
    def _execute_retry(self, 
                      config: RecoveryConfig, 
                      error: Exception, 
                      context: Dict[str, Any]) -> Any:
        """Execute retry strategy."""
        for attempt in range(config.max_attempts):
            try:
                # Get the original function from context
                func = context.get('function')
                args = context.get('args', [])
                kwargs = context.get('kwargs', {})
                
                if func:
                    return func(*args, **kwargs)
                else:
                    _logger.warning("No function provided in context for retry")
                    return None
                    
            except Exception as retry_error:
                if attempt < config.max_attempts - 1:
                    time.sleep(config.retry_delay)
                    continue
                else:
                    raise retry_error
    
    def _execute_fallback(self, 
                         config: RecoveryConfig, 
                         error: Exception, 
                         context: Dict[str, Any]) -> Any:
        """Execute fallback strategy."""
        if not config.fallback_function:
            raise ValueError("Fallback function not configured")
        
        args = context.get('fallback_args', [])
        kwargs = context.get('fallback_kwargs', {})
        
        return config.fallback_function(*args, **kwargs)
    
    def _execute_degrade(self, 
                        config: RecoveryConfig, 
                        error: Exception, 
                        context: Dict[str, Any]) -> Any:
        """Execute degradation strategy."""
        if not config.degrade_function:
            raise ValueError("Degrade function not configured")
        
        args = context.get('degrade_args', [])
        kwargs = context.get('degrade_kwargs', {})
        
        return config.degrade_function(*args, **kwargs)
    
    def _execute_restart(self, 
                        config: RecoveryConfig, 
                        error: Exception, 
                        context: Dict[str, Any]) -> Any:
        """Execute restart strategy."""
        # Get restart function from context
        restart_func = context.get('restart_function')
        if not restart_func:
            raise ValueError("Restart function not provided in context")
        
        time.sleep(config.restart_delay)
        return restart_func()
    
    def _execute_ignore(self, 
                       config: RecoveryConfig, 
                       error: Exception, 
                       context: Dict[str, Any]) -> Any:
        """Execute ignore strategy."""
        _logger.info(f"Ignoring error: {str(error)}")
        return context.get('default_value')
    
    def _execute_alert(self, 
                      config: RecoveryConfig, 
                      error: Exception, 
                      context: Dict[str, Any]) -> Any:
        """Execute alert strategy."""
        if config.alert_function:
            config.alert_function(error, context)
        
        # Return default value or re-raise
        default_value = context.get('default_value')
        if default_value is not None:
            return default_value
        else:
            raise error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get recovery metrics."""
        metrics = self.metrics.copy()
        if metrics['total_recoveries'] > 0:
            metrics['success_rate'] = metrics['successful_recoveries'] / metrics['total_recoveries']
            metrics['average_recovery_time'] = metrics['recovery_time'] / metrics['total_recoveries']
        else:
            metrics['success_rate'] = 0.0
            metrics['average_recovery_time'] = 0.0
        return metrics
    
    def reset_metrics(self):
        """Reset recovery metrics."""
        self.metrics = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'recovery_time': 0.0
        }


# Global recovery manager instance
recovery_manager = ErrorRecoveryManager()


def with_recovery(error_type: str, 
                 strategy: RecoveryStrategy,
                 **strategy_kwargs):
    """
    Decorator for adding recovery functionality to functions.
    
    Args:
        error_type: Type of error to recover from
        strategy: Recovery strategy to use
        **strategy_kwargs: Additional strategy configuration
        
    Example:
        @with_recovery('network', RecoveryStrategy.FALLBACK, fallback_function=backup_api)
        def api_call():
            # Function that may fail
            pass
    """
    def decorator(func: Callable) -> Callable:
        config = RecoveryConfig(strategy=strategy, **strategy_kwargs)
        recovery_manager.register_recovery(error_type, config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func,
                    'args': args,
                    'kwargs': kwargs
                }
                return recovery_manager.execute_recovery(e, context)
        
        return wrapper
    
    return decorator


# Pre-configured recovery decorators
def with_fallback(fallback_func: Callable, error_type: str = 'general'):
    """Decorator for adding fallback functionality."""
    return with_recovery(
        error_type=error_type,
        strategy=RecoveryStrategy.FALLBACK,
        fallback_function=fallback_func
    )


def with_degradation(degrade_func: Callable, error_type: str = 'general'):
    """Decorator for adding degradation functionality."""
    return with_recovery(
        error_type=error_type,
        strategy=RecoveryStrategy.DEGRADE,
        degrade_function=degrade_func
    )


def with_retry(max_attempts: int = 3, error_type: str = 'general'):
    """Decorator for adding retry functionality."""
    return with_recovery(
        error_type=error_type,
        strategy=RecoveryStrategy.RETRY,
        max_attempts=max_attempts
    )


def with_alert(alert_func: Callable, error_type: str = 'general'):
    """Decorator for adding alert functionality."""
    return with_recovery(
        error_type=error_type,
        strategy=RecoveryStrategy.ALERT,
        alert_function=alert_func
    )


# Common recovery strategies
class CommonRecoveryStrategies:
    """Pre-defined recovery strategies for common scenarios."""
    
    @staticmethod
    def setup_default_strategies():
        """Setup default recovery strategies."""
        
        # Network errors - retry with fallback
        recovery_manager.register_recovery('network', RecoveryConfig(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=3,
            timeout=30.0
        ))
        
        # Database errors - retry with alert
        recovery_manager.register_recovery('database', RecoveryConfig(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=2,
            timeout=10.0
        ))
        
        # API errors - fallback to cached data
        recovery_manager.register_recovery('api', RecoveryConfig(
            strategy=RecoveryStrategy.FALLBACK,
            timeout=15.0
        ))
        
        # Validation errors - ignore with default
        recovery_manager.register_recovery('validation', RecoveryConfig(
            strategy=RecoveryStrategy.IGNORE,
            timeout=5.0
        ))
        
        # Timeout errors - retry with longer timeout
        recovery_manager.register_recovery('timeout', RecoveryConfig(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=2,
            timeout=60.0
        ))
        
        _logger.info("Default recovery strategies configured")


# Initialize default strategies
CommonRecoveryStrategies.setup_default_strategies() 