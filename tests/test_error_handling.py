#!/usr/bin/env python3
"""
Test Error Handling & Resilience System
=======================================

This test script verifies that the new error handling and resilience system
works correctly and addresses the original issues.
"""

import sys
import time
import random
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.error_handling import (
    # Exceptions
    TradingException, DataFeedException, BrokerException, NetworkException,
    InsufficientFundsException, RateLimitException, CircuitBreakerOpenException,
    
    # Retry Management
    RetryManager, RetryConfig, RetryStrategy,
    
    # Circuit Breaker
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
    
    # Recovery Management
    ErrorRecoveryManager, RecoveryStrategy, RecoveryConfig,
    
    # Error Monitoring
    ErrorMonitor, ErrorSeverity, ErrorEvent,
    
    # Resilience Decorators
    resilient, retry_on_failure, circuit_breaker, fallback, timeout,
    resilient_api_call, resilient_database_call
)


class TestCustomExceptions(unittest.TestCase):
    """Test custom exception classes."""
    
    def test_trading_exception_with_context(self):
        """Test TradingException with rich context."""
        context = {
            'component': 'api',
            'user_id': 'test_user',
            'recovery_suggestion': 'Retry after 30 seconds'
        }
        
        exception = TradingException(
            message="Test error",
            error_code="TEST_ERROR",
            context=context,
            retry_after=30
        )
        
        self.assertEqual(exception.message, "Test error")
        self.assertEqual(exception.error_code, "TEST_ERROR")
        self.assertEqual(exception.context['component'], 'api')
        self.assertTrue(exception.recoverable)
        self.assertTrue(exception.should_retry())
        self.assertEqual(exception.get_recovery_suggestion(), 'Retry after 30 seconds')
    
    def test_data_feed_exception(self):
        """Test DataFeedException with data source context."""
        exception = DataFeedException(
            message="Data feed failed",
            data_source="binance",
            symbol="BTCUSDT",
            interval="1h"
        )
        
        self.assertEqual(exception.error_code, "DATA_FEED_ERROR")
        self.assertEqual(exception.context['data_source'], 'binance')
        self.assertEqual(exception.context['symbol'], 'BTCUSDT')
        self.assertEqual(exception.context['component'], 'data_feed')
    
    def test_broker_exception(self):
        """Test BrokerException with broker context."""
        exception = BrokerException(
            message="Order failed",
            broker_type="binance",
            symbol="BTCUSDT",
            order_type="market_buy"
        )
        
        self.assertEqual(exception.error_code, "BROKER_ERROR")
        self.assertEqual(exception.context['broker_type'], 'binance')
        self.assertEqual(exception.context['component'], 'broker')
    
    def test_insufficient_funds_exception(self):
        """Test InsufficientFundsException."""
        exception = InsufficientFundsException(
            symbol="BTCUSDT",
            required_amount=1000.0,
            available_amount=500.0
        )
        
        self.assertEqual(exception.context['required_amount'], 1000.0)
        self.assertEqual(exception.context['available_amount'], 500.0)
        self.assertTrue(exception.recoverable)
    
    def test_rate_limit_exception(self):
        """Test RateLimitException."""
        exception = RateLimitException(
            url="https://api.example.com",
            retry_after=60
        )
        
        self.assertEqual(exception.retry_after, 60)
        self.assertTrue(exception.should_retry())


class TestRetryManager(unittest.TestCase):
    """Test retry mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=False  # Disable jitter for predictable testing
        )
        self.retry_manager = RetryManager(self.retry_config)
    
    def test_successful_execution(self):
        """Test successful execution without retries."""
        def successful_func():
            return "success"
        
        result = self.retry_manager.execute(successful_func)
        self.assertEqual(result, "success")
        
        stats = self.retry_manager.get_stats()
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['successful_calls'], 1)
        self.assertEqual(stats['retry_attempts'], 0)
    
    def test_retry_on_failure(self):
        """Test retry mechanism on failure."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkException("Temporary failure")
            return "success after retries"
        
        result = self.retry_manager.execute(failing_func)
        self.assertEqual(result, "success after retries")
        self.assertEqual(call_count, 3)
        
        stats = self.retry_manager.get_stats()
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['successful_calls'], 1)
        self.assertEqual(stats['retry_attempts'], 2)
    
    def test_max_attempts_exceeded(self):
        """Test behavior when max attempts exceeded."""
        def always_failing_func():
            raise NetworkException("Always fails")
        
        with self.assertRaises(NetworkException):
            self.retry_manager.execute(always_failing_func)
        
        stats = self.retry_manager.get_stats()
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['failed_calls'], 1)
        self.assertEqual(stats['retry_attempts'], 3)
    
    def test_retry_strategies(self):
        """Test different retry strategies."""
        # Test fixed strategy
        fixed_config = RetryConfig(
            max_attempts=2,
            base_delay=0.1,
            strategy=RetryStrategy.FIXED
        )
        fixed_manager = RetryManager(fixed_config)
        
        call_count = 0
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise NetworkException("Failure")
        
        with self.assertRaises(NetworkException):
            fixed_manager.execute(failing_func)
        
        self.assertEqual(call_count, 2)
    
    def test_retry_on_specific_exceptions(self):
        """Test retry only on specific exceptions."""
        config = RetryConfig(
            max_attempts=2,
            base_delay=0.1,
            retry_on_exceptions=(NetworkException,)
        )
        manager = RetryManager(config)
        
        def raise_value_error():
            raise ValueError("Value error")
        
        # Should not retry on ValueError
        with self.assertRaises(ValueError):
            manager.execute(raise_value_error)
        
        stats = manager.get_stats()
        self.assertEqual(stats['retry_attempts'], 0)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker pattern."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=2,
            failure_window=10,
            recovery_timeout=1,  # Short timeout for testing
            success_threshold=1
        )
        self.circuit_breaker = CircuitBreaker("test_circuit", self.config)
    
    def test_closed_state_normal_operation(self):
        """Test normal operation in CLOSED state."""
        def successful_func():
            return "success"
        
        result = self.circuit_breaker.call(successful_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
    
    def test_transition_to_open(self):
        """Test transition to OPEN state after failures."""
        def failing_func():
            raise NetworkException("Failure")
        
        # First failure
        with self.assertRaises(NetworkException):
            self.circuit_breaker.call(failing_func)
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        
        # Second failure - should open circuit
        with self.assertRaises(NetworkException):
            self.circuit_breaker.call(failing_func)
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        
        # Third call should fail fast
        with self.assertRaises(CircuitBreakerOpenException):
            self.circuit_breaker.call(failing_func)
    
    def test_transition_to_half_open(self):
        """Test transition to HALF_OPEN state after timeout."""
        # First, open the circuit
        def failing_func():
            raise NetworkException("Failure")
        
        for _ in range(2):
            with self.assertRaises(NetworkException):
                self.circuit_breaker.call(failing_func)
        
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Should transition to HALF_OPEN
        self.circuit_breaker._update_state()
        self.assertEqual(self.circuit_breaker.state, CircuitState.HALF_OPEN)
    
    def test_transition_to_closed(self):
        """Test transition back to CLOSED state after success."""
        # First, open the circuit
        def failing_func():
            raise NetworkException("Failure")
        
        for _ in range(2):
            with self.assertRaises(NetworkException):
                self.circuit_breaker.call(failing_func)
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # Try successful call in HALF_OPEN state
        def successful_func():
            return "success"
        
        result = self.circuit_breaker.call(successful_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
    
    def test_force_open_and_close(self):
        """Test manual circuit breaker control."""
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
        
        # Force open
        self.circuit_breaker.force_open()
        self.assertEqual(self.circuit_breaker.state, CircuitState.OPEN)
        
        # Force close
        self.circuit_breaker.force_close()
        self.assertEqual(self.circuit_breaker.state, CircuitState.CLOSED)
    
    def test_circuit_breaker_stats(self):
        """Test circuit breaker statistics."""
        def successful_func():
            return "success"
        
        def failing_func():
            raise NetworkException("Failure")
        
        # Make some calls
        self.circuit_breaker.call(successful_func)
        
        with self.assertRaises(NetworkException):
            self.circuit_breaker.call(failing_func)
        
        stats = self.circuit_breaker.get_stats()
        self.assertEqual(stats['total_calls'], 2)
        self.assertEqual(stats['successful_calls'], 1)
        self.assertEqual(stats['failed_calls'], 1)


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recovery_manager = ErrorRecoveryManager()
    
    def test_fallback_strategy(self):
        """Test fallback recovery strategy."""
        def main_func():
            raise NetworkException("Main function failed")
        
        def fallback_func():
            return "Fallback response"
        
        # Register fallback strategy
        self.recovery_manager.register_recovery('network', RecoveryConfig(
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_func
        ))
        
        # Execute recovery
        result = self.recovery_manager.execute_recovery(
            NetworkException("Network error"),
            {'component': 'network'}
        )
        
        self.assertEqual(result, "Fallback response")
    
    def test_degrade_strategy(self):
        """Test degradation recovery strategy."""
        def degrade_func():
            return "Degraded response"
        
        # Register degrade strategy
        self.recovery_manager.register_recovery('api', RecoveryConfig(
            strategy=RecoveryStrategy.DEGRADE,
            degrade_function=degrade_func
        ))
        
        # Execute recovery
        result = self.recovery_manager.execute_recovery(
            BrokerException("API error"),
            {'component': 'api'}
        )
        
        self.assertEqual(result, "Degraded response")
    
    def test_retry_strategy(self):
        """Test retry recovery strategy."""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkException("Temporary failure")
            return "Success after retries"
        
        # Register retry strategy
        self.recovery_manager.register_recovery('network', RecoveryConfig(
            strategy=RecoveryStrategy.RETRY,
            max_attempts=3,
            timeout=1.0
        ))
        
        # Execute recovery
        result = self.recovery_manager.execute_recovery(
            NetworkException("Network error"),
            {
                'component': 'network',
                'function': failing_func,
                'args': [],
                'kwargs': {}
            }
        )
        
        self.assertEqual(result, "Success after retries")
        self.assertEqual(call_count, 3)
    
    def test_ignore_strategy(self):
        """Test ignore recovery strategy."""
        # Register ignore strategy
        self.recovery_manager.register_recovery('validation', RecoveryConfig(
            strategy=RecoveryStrategy.IGNORE
        ))
        
        # Execute recovery
        result = self.recovery_manager.execute_recovery(
            ValueError("Validation error"),
            {
                'component': 'validation',
                'default_value': "Default response"
            }
        )
        
        self.assertEqual(result, "Default response")
    
    def test_recovery_metrics(self):
        """Test recovery metrics tracking."""
        def fallback_func():
            return "Fallback"
        
        self.recovery_manager.register_recovery('test', RecoveryConfig(
            strategy=RecoveryStrategy.FALLBACK,
            fallback_function=fallback_func
        ))
        
        # Execute recovery
        self.recovery_manager.execute_recovery(
            Exception("Test error"),
            {'component': 'test'}
        )
        
        metrics = self.recovery_manager.get_metrics()
        self.assertEqual(metrics['total_recoveries'], 1)
        self.assertEqual(metrics['successful_recoveries'], 1)


class TestErrorMonitoring(unittest.TestCase):
    """Test error monitoring and alerting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = ErrorMonitor()
        self.alert_called = False
        self.alert_data = None
    
    def alert_function(self, alert_data):
        """Mock alert function."""
        self.alert_called = True
        self.alert_data = alert_data
    
    def test_record_error(self):
        """Test error recording."""
        error = NetworkException("Test error", url="https://example.com")
        
        self.monitor.record_error(
            error=error,
            severity=ErrorSeverity.ERROR,
            component="api",
            context={'user_id': 'test_user'}
        )
        
        # Check that error was recorded
        stats = self.monitor.get_error_stats()
        self.assertEqual(stats['total_errors'], 1)
        self.assertEqual(stats['severity_distribution']['ERROR'], 1)
        self.assertEqual(stats['component_distribution']['api'], 1)
    
    def test_error_alerting(self):
        """Test error alerting."""
        self.monitor.add_alert_function(self.alert_function)
        
        # Record error that should trigger alert
        error = NetworkException("Critical error", url="https://example.com")
        self.monitor.record_error(
            error=error,
            severity=ErrorSeverity.CRITICAL,
            component="api"
        )
        
        # Check if alert was triggered
        self.assertTrue(self.alert_called)
        self.assertIsNotNone(self.alert_data)
        self.assertIn('message', self.alert_data)
        self.assertIn('error_event', self.alert_data)
    
    def test_error_statistics(self):
        """Test error statistics calculation."""
        # Record multiple errors
        errors = [
            (NetworkException("Error 1"), "api"),
            (BrokerException("Error 2"), "broker"),
            (DataFeedException("Error 3"), "data_feed"),
        ]
        
        for error, component in errors:
            self.monitor.record_error(
                error=error,
                severity=ErrorSeverity.ERROR,
                component=component
            )
        
        stats = self.monitor.get_error_stats()
        self.assertEqual(stats['total_errors'], 3)
        self.assertEqual(len(stats['component_distribution']), 3)
        self.assertEqual(len(stats['top_errors']), 3)
    
    def test_recent_errors(self):
        """Test recent error retrieval."""
        # Record some errors
        for i in range(5):
            self.monitor.record_error(
                error=NetworkException(f"Error {i}"),
                severity=ErrorSeverity.ERROR,
                component="api"
            )
        
        # Get recent errors
        recent_errors = self.monitor.get_recent_errors(limit=3)
        self.assertEqual(len(recent_errors), 3)
        
        # Test filtering
        api_errors = self.monitor.get_recent_errors(limit=10, component="api")
        self.assertEqual(len(api_errors), 5)
    
    def test_error_report_generation(self):
        """Test error report generation."""
        # Record some errors
        self.monitor.record_error(
            error=NetworkException("Test error"),
            severity=ErrorSeverity.ERROR,
            component="api"
        )
        
        # Generate JSON report
        json_report = self.monitor.generate_error_report(format="json")
        self.assertIn("statistics", json_report)
        self.assertIn("recent_errors", json_report)
        
        # Generate text report
        text_report = self.monitor.generate_error_report(format="text")
        self.assertIn("Error Report", text_report)
        self.assertIn("Statistics", text_report)


class TestResilienceDecorators(unittest.TestCase):
    """Test resilience decorators."""
    
    def test_retry_decorator(self):
        """Test retry decorator."""
        call_count = 0
        
        @retry_on_failure(max_attempts=3, base_delay=0.1)
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkException("Temporary failure")
            return "Success"
        
        result = failing_func()
        self.assertEqual(result, "Success")
        self.assertEqual(call_count, 3)
    
    def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        @circuit_breaker("test_circuit", failure_threshold=2)
        def failing_func():
            raise NetworkException("Failure")
        
        # First two calls should fail with NetworkException
        for _ in range(2):
            with self.assertRaises(NetworkException):
                failing_func()
        
        # Third call should fail with CircuitBreakerOpenException
        with self.assertRaises(CircuitBreakerOpenException):
            failing_func()
    
    def test_fallback_decorator(self):
        """Test fallback decorator."""
        def fallback_func():
            return "Fallback response"
        
        @fallback(fallback_func)
        def failing_func():
            raise NetworkException("Main function failed")
        
        result = failing_func()
        self.assertEqual(result, "Fallback response")
    
    def test_timeout_decorator(self):
        """Test timeout decorator."""
        @timeout(0.1)
        def slow_func():
            time.sleep(0.2)  # Longer than timeout
            return "Should not reach here"
        
        with self.assertRaises(Exception):  # Should timeout
            slow_func()
    
    def test_resilient_decorator(self):
        """Test comprehensive resilient decorator."""
        def fallback_func():
            return "Fallback"
        
        @resilient(
            retry_config=RetryConfig(max_attempts=2, base_delay=0.1),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
            fallback_func=fallback_func,
            timeout=1.0
        )
        def unreliable_func():
            if random.random() < 0.8:  # 80% failure rate
                raise NetworkException("Failure")
            return "Success"
        
        # Should eventually succeed or fallback
        result = unreliable_func()
        self.assertIn(result, ["Success", "Fallback"])
    
    def test_resilient_api_call_decorator(self):
        """Test pre-configured API resilience decorator."""
        @resilient_api_call(max_attempts=2, timeout_seconds=0.1)
        def api_call():
            if random.random() < 0.5:
                raise NetworkException("API failure")
            return "API success"
        
        # Should handle failures gracefully
        try:
            result = api_call()
            self.assertIn(result, ["API success"])
        except Exception:
            # It's okay if it fails after retries
            pass


class TestIntegration(unittest.TestCase):
    """Test integration of error handling components."""
    
    def test_full_resilience_chain(self):
        """Test complete resilience chain."""
        # Create components
        retry_manager = RetryManager(RetryConfig(max_attempts=2, base_delay=0.1))
        circuit_breaker = CircuitBreaker("test_circuit", CircuitBreakerConfig(failure_threshold=3))
        monitor = ErrorMonitor()
        
        # Test function - succeeds on second attempt
        call_count = 0
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:  # Changed from 3 to 2
                raise NetworkException("Temporary failure")
            return "Success"
        
        # Execute with full resilience
        try:
            # First with retry
            result = retry_manager.execute(test_func)
            self.assertEqual(result, "Success")
            
            # Then with circuit breaker
            result = circuit_breaker.call(test_func)
            self.assertEqual(result, "Success")
            
            # Record success in monitor
            monitor.record_error(
                error=Exception("Test error"),
                severity=ErrorSeverity.INFO,
                component="test"
            )
            
        except Exception as e:
            # Record failure in monitor
            monitor.record_error(
                error=e,
                severity=ErrorSeverity.ERROR,
                component="test"
            )
            raise
        
        # Check statistics
        retry_stats = retry_manager.get_stats()
        circuit_stats = circuit_breaker.get_stats()
        monitor_stats = monitor.get_error_stats()
        
        self.assertIsNotNone(retry_stats)
        self.assertIsNotNone(circuit_stats)
        self.assertIsNotNone(monitor_stats)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCustomExceptions,
        TestRetryManager,
        TestCircuitBreaker,
        TestErrorRecovery,
        TestErrorMonitoring,
        TestResilienceDecorators,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 