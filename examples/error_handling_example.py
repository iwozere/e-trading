#!/usr/bin/env python3
"""
Error Handling & Resilience System Examples
===========================================

This example demonstrates how to use the new comprehensive error handling
and resilience system for the crypto trading platform.

Features demonstrated:
- Custom exception classes with context
- Retry mechanisms with exponential backoff
- Circuit breaker pattern
- Error recovery strategies
- Error monitoring and alerting
- Resilience decorators
"""

import time
import random
import logging
from typing import Dict, Any
import sys
from pathlib import Path

from src.error_handling import RecoveryConfig


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.error_handling import (
    # Exceptions
    TradingException, DataFeedException, BrokerException, NetworkException,
    InsufficientFundsException, RateLimitException,
    
    # Retry Management
    RetryManager, RetryConfig, retry_on_failure,
    
    # Circuit Breaker
    CircuitBreaker, CircuitBreakerConfig, circuit_breaker,
    
    # Recovery Management
    ErrorRecoveryManager, RecoveryStrategy, with_fallback,
    
    # Error Monitoring
    ErrorMonitor, ErrorSeverity, monitor_errors,
    
    # Resilience Decorators
    resilient, retry_on_failure, circuit_breaker, fallback, timeout,
    resilient_api_call, resilient_database_call
)


def setup_logging():
    """Setup logging for the example"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_custom_exceptions():
    """Demonstrate custom exception classes with context"""
    print("\n=== Custom Exception Classes ===")
    
    try:
        # Simulate insufficient funds error
        raise InsufficientFundsException(
            symbol="BTCUSDT",
            required_amount=1000.0,
            available_amount=500.0,
            context={
                'order_type': 'market_buy',
                'recovery_suggestion': 'Reduce order size or add funds'
            }
        )
    except InsufficientFundsException as e:
        print(f"Caught insufficient funds error:")
        print(f"  Message: {e.message}")
        print(f"  Error Code: {e.error_code}")
        print(f"  Context: {e.context}")
        print(f"  Recoverable: {e.recoverable}")
        print(f"  Should Retry: {e.should_retry()}")
        print(f"  Recovery Suggestion: {e.get_recovery_suggestion()}")
    
    try:
        # Simulate rate limit error
        raise RateLimitException(
            url="https://api.binance.com/v1/ticker/price",
            retry_after=60
        )
    except RateLimitException as e:
        print(f"\nCaught rate limit error:")
        print(f"  Message: {e.message}")
        print(f"  Retry After: {e.retry_after} seconds")
        print(f"  Should Retry: {e.should_retry()}")


def example_retry_mechanism():
    """Demonstrate retry mechanisms with exponential backoff"""
    print("\n=== Retry Mechanisms ===")
    
    # Create retry manager
    retry_config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        strategy="exponential",
        jitter=True,
        retry_on_exceptions=(NetworkException,)
    )
    
    retry_manager = RetryManager(retry_config)
    
    # Simulate unreliable API call
    def unreliable_api_call():
        if random.random() < 0.7:  # 70% failure rate
            raise NetworkException(
                message="API temporarily unavailable",
                url="https://api.example.com",
                status_code=503
            )
        return "API call successful"
    
    # Execute with retry
    try:
        result = retry_manager.execute(unreliable_api_call)
        print(f"API call result: {result}")
    except Exception as e:
        print(f"API call failed after retries: {e}")
    
    # Show retry statistics
    stats = retry_manager.get_stats()
    print(f"Retry statistics: {stats}")


def example_circuit_breaker():
    """Demonstrate circuit breaker pattern"""
    print("\n=== Circuit Breaker Pattern ===")
    
    # Create circuit breaker
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold=3,
        failure_window=60,
        recovery_timeout=30
    )
    
    cb = CircuitBreaker("api_circuit", circuit_breaker_config)
    
    # Simulate failing API calls
    def failing_api_call():
        raise NetworkException("API failure", url="https://api.example.com")
    
    # Make calls until circuit opens
    for i in range(5):
        try:
            result = cb.call(failing_api_call)
            print(f"Call {i+1}: Success")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")
            print(f"Circuit state: {cb.state.value}")
    
    # Show circuit breaker statistics
    stats = cb.get_stats()
    print(f"Circuit breaker statistics: {stats}")


def example_error_recovery():
    """Demonstrate error recovery strategies"""
    print("\n=== Error Recovery Strategies ===")
    
    # Create recovery manager
    recovery_manager = ErrorRecoveryManager()
    
    # Register recovery strategies
    def backup_api_call():
        return "Backup API response"
    
    def degraded_service():
        return "Degraded service response"
    
    # Register recovery configurations
    recovery_manager.register_recovery('network', RecoveryConfig(
        strategy=RecoveryStrategy.FALLBACK,
        fallback_function=backup_api_call
    ))
    
    recovery_manager.register_recovery('api', RecoveryConfig(
        strategy=RecoveryStrategy.DEGRADE,
        degrade_function=degraded_service
    ))
    
    # Simulate network error
    try:
        raise NetworkException("Network connection failed", url="https://api.example.com")
    except Exception as e:
        try:
            result = recovery_manager.execute_recovery(e, {'component': 'network'})
            print(f"Recovery result: {result}")
        except Exception as recovery_error:
            print(f"Recovery failed: {recovery_error}")
    
    # Show recovery metrics
    metrics = recovery_manager.get_metrics()
    print(f"Recovery metrics: {metrics}")


def example_error_monitoring():
    """Demonstrate error monitoring and alerting"""
    print("\n=== Error Monitoring ===")
    
    # Create error monitor
    monitor = ErrorMonitor()
    
    # Add alert function
    def alert_function(alert_data):
        print(f"ALERT: {alert_data['message']}")
        print(f"Error: {alert_data['error_event']['error_type']}")
    
    monitor.add_alert_function(alert_function)
    
    # Simulate various errors
    errors = [
        (NetworkException("Connection timeout", url="https://api.example.com"), "api"),
        (BrokerException("Order failed", broker_type="binance", symbol="BTCUSDT"), "broker"),
        (DataFeedException("Data unavailable", symbol="BTCUSDT", interval="1h"), "data_feed"),
        (NetworkException("Rate limit exceeded", url="https://api.example.com"), "api"),
    ]
    
    for error, component in errors:
        monitor.record_error(
            error=error,
            severity=ErrorSeverity.ERROR,
            component=component,
            context={'user_id': 'example_user'}
        )
    
    # Get error statistics
    stats = monitor.get_error_stats(time_window=300)  # Last 5 minutes
    print(f"Error statistics: {stats}")
    
    # Get recent errors
    recent_errors = monitor.get_recent_errors(limit=5)
    print(f"Recent errors: {len(recent_errors)}")
    
    # Generate error report
    report = monitor.generate_error_report(time_window=300, format="text")
    print(f"\nError Report:\n{report}")


def example_resilience_decorators():
    """Demonstrate resilience decorators"""
    print("\n=== Resilience Decorators ===")
    
    # Simulate unreliable functions
    def unreliable_api_call():
        if random.random() < 0.6:  # 60% failure rate
            raise NetworkException("API call failed", url="https://api.example.com")
        return "API call successful"
    
    def backup_api_call():
        return "Backup API response"
    
    def slow_function():
        time.sleep(2)  # Simulate slow operation
        return "Slow function completed"
    
    # Apply resilience decorators
    @resilient_api_call(max_attempts=3, timeout_seconds=5.0, fallback_func=backup_api_call)
    def resilient_api():
        return unreliable_api_call()
    
    @timeout(1.0)
    def timeout_protected():
        return slow_function()
    
    @retry_on_failure(max_attempts=2, base_delay=0.5)
    def retry_protected():
        return unreliable_api_call()
    
    # Test resilient functions
    print("Testing resilient API call:")
    try:
        result = resilient_api()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    print("\nTesting timeout protection:")
    try:
        result = timeout_protected()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")
    
    print("\nTesting retry protection:")
    try:
        result = retry_protected()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")


def example_integration():
    """Demonstrate integration of all error handling features"""
    print("\n=== Integration Example ===")
    
    # Simulate a trading bot component with comprehensive error handling
    class TradingBotComponent:
        def __init__(self):
            self.retry_manager = RetryManager(RetryConfig(max_attempts=3))
            self.circuit_breaker = CircuitBreaker("trading_api", CircuitBreakerConfig(failure_threshold=3))
            self.monitor = ErrorMonitor()
        
        @resilient_api_call(max_attempts=3, timeout_seconds=10.0)
        def get_market_data(self, symbol: str) -> Dict[str, Any]:
            """Get market data with resilience"""
            # Simulate API call
            if random.random() < 0.3:  # 30% failure rate
                raise NetworkException(f"Failed to get market data for {symbol}")
            
            return {
                'symbol': symbol,
                'price': 50000.0,
                'volume': 1000.0,
                'timestamp': time.time()
            }
        
        @circuit_breaker(failure_threshold=2, recovery_timeout=30)
        def place_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
            """Place order with circuit breaker protection"""
            # Simulate order placement
            if random.random() < 0.4:  # 40% failure rate
                raise BrokerException(f"Order placement failed for {symbol}")
            
            return {
                'order_id': f"order_{int(time.time())}",
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'status': 'filled'
            }
        
        @monitor_errors(ErrorSeverity.WARNING, "strategy")
        def calculate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate trading signals with error monitoring"""
            # Simulate signal calculation
            if random.random() < 0.1:  # 10% failure rate
                raise ValueError("Invalid data for signal calculation")
            
            return {
                'signal': 'BUY' if data['price'] > 45000 else 'SELL',
                'confidence': 0.8,
                'timestamp': time.time()
            }
    
    # Test the component
    component = TradingBotComponent()
    
    print("Testing market data retrieval:")
    for i in range(3):
        try:
            data = component.get_market_data("BTCUSDT")
            print(f"  Market data: {data}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\nTesting order placement:")
    for i in range(3):
        try:
            order = component.place_order("BTCUSDT", "BUY", 0.1)
            print(f"  Order: {order}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\nTesting signal calculation:")
    for i in range(3):
        try:
            signals = component.calculate_signals({'price': 50000})
            print(f"  Signals: {signals}")
        except Exception as e:
            print(f"  Failed: {e}")


def main():
    """Run all error handling examples"""
    print("Error Handling & Resilience System Examples")
    print("=" * 50)
    
    setup_logging()
    
    try:
        # Run all examples
        example_custom_exceptions()
        example_retry_mechanism()
        example_circuit_breaker()
        example_error_recovery()
        example_error_monitoring()
        example_resilience_decorators()
        example_integration()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nKey benefits demonstrated:")
        print("1. Standardized error handling with rich context")
        print("2. Automatic retry with exponential backoff")
        print("3. Circuit breaker pattern for fault tolerance")
        print("4. Error recovery strategies with fallbacks")
        print("5. Comprehensive error monitoring and alerting")
        print("6. Easy-to-use decorators for resilience")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 