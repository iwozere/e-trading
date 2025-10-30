"""
Adapter manager with factory and registration system.

This module provides centralized management of sentiment adapters including
registration, health monitoring, and circuit breaker functionality.
"""
import asyncio
from typing import Dict, List, Optional, Type, Any, Callable
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import time
from pathlib import Path
import sys

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.common.sentiments.adapters.base_adapter import BaseSentimentAdapter, AdapterStatus, AdapterHealthInfo
from src.common.sentiments.rate_limiting.global_coordinator import GlobalRateLimitCoordinator, GlobalLimitConfig
from src.common.sentiments.rate_limiting.adaptive_limiter import AdaptiveConfig
from src.common.sentiments.rate_limiting.priority_queue import RequestPriority

_logger = setup_logger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker functionality."""
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker implementation for adapter fault tolerance.

    Prevents cascading failures by temporarily disabling failed adapters
    and allowing gradual recovery.
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = AdapterStatus.HEALTHY
        self.half_open_calls = 0

    def can_execute(self) -> bool:
        """Check if operation can be executed based on circuit breaker state."""
        if self.state == AdapterStatus.HEALTHY:
            return True

        if self.state == AdapterStatus.FAILED:
            # Check if we should transition to half-open
            if (self.last_failure_time and
                datetime.now(timezone.utc) - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout_seconds)):
                self.state = AdapterStatus.CIRCUIT_OPEN
                self.half_open_calls = 0
                return True
            return False

        if self.state == AdapterStatus.CIRCUIT_OPEN:
            # Allow limited calls in half-open state
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def record_success(self) -> None:
        """Record successful operation."""
        if self.state == AdapterStatus.CIRCUIT_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                # Transition back to healthy
                self.state = AdapterStatus.HEALTHY
                self.failure_count = 0
                self.half_open_calls = 0
        else:
            self.failure_count = 0
            self.state = AdapterStatus.HEALTHY

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.state == AdapterStatus.CIRCUIT_OPEN:
            # Failed during half-open, go back to failed
            self.state = AdapterStatus.FAILED
        elif self.failure_count >= self.config.failure_threshold:
            self.state = AdapterStatus.FAILED


class AdapterRegistry:
    """Registry for managing sentiment adapter types and instances."""

    def __init__(self):
        self._adapter_types: Dict[str, Type[BaseSentimentAdapter]] = {}
        self._adapter_configs: Dict[str, Dict[str, Any]] = {}

    def register_adapter(self, name: str, adapter_class: Type[BaseSentimentAdapter],
                        default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register an adapter type.

        Args:
            name: Unique name for the adapter
            adapter_class: Adapter class that implements BaseSentimentAdapter
            default_config: Default configuration for the adapter
        """
        if not issubclass(adapter_class, BaseSentimentAdapter):
            raise ValueError(f"Adapter class {adapter_class} must inherit from BaseSentimentAdapter")

        self._adapter_types[name] = adapter_class
        self._adapter_configs[name] = default_config or {}
        _logger.info("Registered adapter: %s", name)

    def create_adapter(self, name: str, config: Optional[Dict[str, Any]] = None) -> BaseSentimentAdapter:
        """
        Create an adapter instance.

        Args:
            name: Name of the registered adapter
            config: Configuration overrides

        Returns:
            Configured adapter instance
        """
        if name not in self._adapter_types:
            raise ValueError(f"Unknown adapter: {name}. Available: {list(self._adapter_types.keys())}")

        adapter_class = self._adapter_types[name]
        final_config = dict(self._adapter_configs[name])
        if config:
            final_config.update(config)

        return adapter_class(name=name, **final_config)

    def list_adapters(self) -> List[str]:
        """Get list of registered adapter names."""
        return list(self._adapter_types.keys())


class AdapterManager:
    """
    Central manager for sentiment adapters with health monitoring and circuit breaker.

    Provides unified interface for managing multiple sentiment data adapters
    with fault tolerance and health monitoring capabilities.
    """

    def __init__(self, circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 enable_global_rate_limiting: bool = True):
        """
        Initialize adapter manager.

        Args:
            circuit_breaker_config: Circuit breaker configuration
            enable_global_rate_limiting: Enable global rate limiting coordination
        """
        self.registry = AdapterRegistry()
        self._adapters: Dict[str, BaseSentimentAdapter] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._circuit_config = circuit_breaker_config or CircuitBreakerConfig()
        self._health_check_interval = 300  # 5 minutes
        self._last_health_check: Optional[datetime] = None

        # Initialize global rate limiting if enabled
        self._global_coordinator: Optional[GlobalRateLimitCoordinator] = None
        if enable_global_rate_limiting:
            try:
                from src.common.sentiments.rate_limiting.global_coordinator import initialize_global_coordinator
                global_config = GlobalLimitConfig(
                    max_global_requests_per_second=8.0,
                    max_concurrent_requests=15,
                    enable_adaptive_global_limit=True
                )
                self._global_coordinator = initialize_global_coordinator(global_config)
                _logger.info("Global rate limiting enabled")
            except Exception as e:
                _logger.warning("Failed to initialize global rate limiting: %s", e)

    def register_adapter_type(self, name: str, adapter_class: Type[BaseSentimentAdapter],
                             default_config: Optional[Dict[str, Any]] = None) -> None:
        """Register an adapter type with the manager."""
        self.registry.register_adapter(name, adapter_class, default_config)

    def add_adapter(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an adapter instance to the manager.

        Args:
            name: Name of the registered adapter type
            config: Configuration overrides for this instance
        """
        if name in self._adapters:
            _logger.warning("Adapter %s already exists, replacing", name)

        adapter = self.registry.create_adapter(name, config)
        self._adapters[name] = adapter
        self._circuit_breakers[name] = CircuitBreaker(self._circuit_config)

        # Register with global rate limiting coordinator
        if self._global_coordinator:
            try:
                adaptive_config = AdaptiveConfig(
                    base_requests_per_second=config.get('rate_limit', 1.0) if config else 1.0,
                    min_requests_per_second=0.1,
                    max_requests_per_second=5.0
                )
                priority_weight = config.get('priority_weight', 1.0) if config else 1.0
                self._global_coordinator.register_adapter(name, adaptive_config, priority_weight)
            except Exception as e:
                _logger.warning("Failed to register adapter %s with global coordinator: %s", name, e)

        _logger.info("Added adapter: %s", name)

    def remove_adapter(self, name: str) -> None:
        """Remove an adapter from the manager."""
        if name in self._adapters:
            asyncio.create_task(self._adapters[name].close())
            del self._adapters[name]
            del self._circuit_breakers[name]
            _logger.info("Removed adapter: %s", name)

    async def fetch_messages_from_adapter(self, adapter_name: str, ticker: str,
                                        since_ts: Optional[int] = None,
                                        limit: int = 200) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch messages from a specific adapter with circuit breaker protection.

        Args:
            adapter_name: Name of the adapter to use
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch messages since
            limit: Maximum number of messages to fetch

        Returns:
            List of messages or None if adapter is unavailable
        """
        if adapter_name not in self._adapters:
            _logger.warning("Adapter %s not found", adapter_name)
            return None

        circuit_breaker = self._circuit_breakers[adapter_name]
        if not circuit_breaker.can_execute():
            _logger.debug("Circuit breaker open for adapter %s", adapter_name)
            return None

        adapter = self._adapters[adapter_name]
        start_time = time.time()

        # Acquire global rate limiting permission if available
        global_permission_acquired = False
        if self._global_coordinator:
            try:
                global_permission_acquired = await self._global_coordinator.acquire_global_permission(
                    adapter_name, timeout=30.0
                )
                if not global_permission_acquired:
                    _logger.debug("Global rate limit exceeded for adapter %s", adapter_name)
                    return None
            except Exception as e:
                _logger.debug("Global rate limiting failed for %s: %s", adapter_name, e)

        try:
            messages = await adapter.fetch_messages(ticker, since_ts, limit)

            # Record success
            response_time_ms = (time.time() - start_time) * 1000
            adapter._update_health_success(response_time_ms)
            circuit_breaker.record_success()

            # Record with global coordinator
            if self._global_coordinator:
                self._global_coordinator.record_adapter_response(
                    adapter_name, response_time_ms, True
                )

            return messages

        except Exception as e:
            _logger.warning("Adapter %s failed for ticker %s: %s", adapter_name, ticker, e)

            # Record failure
            adapter._update_health_failure(e)
            circuit_breaker.record_failure()

            # Record with global coordinator
            if self._global_coordinator:
                response_time_ms = (time.time() - start_time) * 1000
                self._global_coordinator.record_adapter_response(
                    adapter_name, response_time_ms, False, error_type=str(type(e).__name__)
                )

            return None

        finally:
            # Release global permission
            if global_permission_acquired and self._global_coordinator:
                self._global_coordinator.release_global_permission(adapter_name)

    async def fetch_summary_from_adapter(self, adapter_name: str, ticker: str,
                                       since_ts: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch summary from a specific adapter with circuit breaker protection.

        Args:
            adapter_name: Name of the adapter to use
            ticker: Stock ticker symbol
            since_ts: Unix timestamp to fetch data since

        Returns:
            Summary dictionary or None if adapter is unavailable
        """
        if adapter_name not in self._adapters:
            _logger.warning("Adapter %s not found", adapter_name)
            return None

        circuit_breaker = self._circuit_breakers[adapter_name]
        if not circuit_breaker.can_execute():
            _logger.debug("Circuit breaker open for adapter %s", adapter_name)
            return None

        adapter = self._adapters[adapter_name]
        start_time = time.time()

        # Acquire global rate limiting permission if available
        global_permission_acquired = False
        if self._global_coordinator:
            try:
                global_permission_acquired = await self._global_coordinator.acquire_global_permission(
                    adapter_name, timeout=30.0
                )
                if not global_permission_acquired:
                    _logger.debug("Global rate limit exceeded for adapter %s", adapter_name)
                    return None
            except Exception as e:
                _logger.debug("Global rate limiting failed for %s: %s", adapter_name, e)

        try:
            summary = await adapter.fetch_summary(ticker, since_ts)

            # Record success
            response_time_ms = (time.time() - start_time) * 1000
            adapter._update_health_success(response_time_ms)
            circuit_breaker.record_success()

            # Record with global coordinator
            if self._global_coordinator:
                self._global_coordinator.record_adapter_response(
                    adapter_name, response_time_ms, True
                )

            return summary

        except Exception as e:
            _logger.warning("Adapter %s failed for ticker %s: %s", adapter_name, ticker, e)

            # Record failure
            adapter._update_health_failure(e)
            circuit_breaker.record_failure()

            # Record with global coordinator
            if self._global_coordinator:
                response_time_ms = (time.time() - start_time) * 1000
                self._global_coordinator.record_adapter_response(
                    adapter_name, response_time_ms, False, error_type=str(type(e).__name__)
                )

            return None

        finally:
            # Release global permission
            if global_permission_acquired and self._global_coordinator:
                self._global_coordinator.release_global_permission(adapter_name)

    async def get_health_status(self) -> Dict[str, AdapterHealthInfo]:
        """Get health status for all adapters."""
        health_status = {}

        for name, adapter in self._adapters.items():
            health_info = await adapter.health_check()
            circuit_breaker = self._circuit_breakers[name]

            # Override status if circuit breaker is open
            if circuit_breaker.state == AdapterStatus.FAILED:
                health_info.status = AdapterStatus.CIRCUIT_OPEN

            health_status[name] = health_info

        return health_status

    async def perform_health_checks(self) -> None:
        """Perform periodic health checks on all adapters."""
        current_time = datetime.now(timezone.utc)

        if (self._last_health_check is None or
            current_time - self._last_health_check > timedelta(seconds=self._health_check_interval)):

            _logger.debug("Performing health checks on %d adapters", len(self._adapters))
            health_status = await self.get_health_status()

            for name, health in health_status.items():
                if health.status != AdapterStatus.HEALTHY:
                    _logger.warning("Adapter %s health: %s - %s", name, health.status.value, health.error_message)

            self._last_health_check = current_time

    def get_available_adapters(self) -> List[str]:
        """Get list of adapters that are currently available (not circuit broken)."""
        available = []
        for name, circuit_breaker in self._circuit_breakers.items():
            if circuit_breaker.can_execute():
                available.append(name)
        return available

    async def close_all(self) -> None:
        """Close all adapters and clean up resources."""
        _logger.info("Closing %d adapters", len(self._adapters))

        close_tasks = []
        for adapter in self._adapters.values():
            close_tasks.append(adapter.close())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self._adapters.clear()
        self._circuit_breakers.clear()

    def get_rate_limiting_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics from global coordinator."""
        if not self._global_coordinator:
            return {}

        return self._global_coordinator.get_global_statistics()


# Global adapter manager instance
_global_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = AdapterManager()
    return _global_manager


def register_default_adapters() -> None:
    """Register the default sentiment adapters."""
    manager = get_adapter_manager()

    try:
        from src.common.sentiments.adapters.async_stocktwits import AsyncStocktwitsAdapter
        manager.register_adapter_type("stocktwits", AsyncStocktwitsAdapter, {
            "concurrency": 5,
            "rate_limit_delay": 0.5
        })
    except ImportError as e:
        _logger.warning("Could not register StockTwits adapter: %s", e)

    try:
        from src.common.sentiments.adapters.async_pushshift import AsyncPushshiftAdapter
        manager.register_adapter_type("reddit", AsyncPushshiftAdapter, {
            "concurrency": 5,
            "rate_limit_delay": 0.5
        })
    except ImportError as e:
        _logger.warning("Could not register Reddit adapter: %s", e)

    try:
        from src.common.sentiments.adapters.async_hf_sentiment import AsyncHFSentiment
        manager.register_adapter_type("huggingface", AsyncHFSentiment, {
            "device": -1,
            "max_workers": 1
        })
    except ImportError as e:
        _logger.warning("Could not register HuggingFace adapter: %s", e)

    try:
        from src.common.sentiments.adapters.async_twitter import AsyncTwitterAdapter
        manager.register_adapter_type("twitter", AsyncTwitterAdapter, {
            "concurrency": 3,
            "rate_limit_delay": 1.0
        })
    except ImportError as e:
        _logger.warning("Could not register Twitter adapter: %s", e)

    try:
        from src.common.sentiments.adapters.async_discord import AsyncDiscordAdapter
        manager.register_adapter_type("discord", AsyncDiscordAdapter, {
            "concurrency": 2,
            "rate_limit_delay": 1.0
        })
    except ImportError as e:
        _logger.warning("Could not register Discord adapter: %s", e)

    try:
        from src.common.sentiments.adapters.async_news import AsyncNewsAdapter
        manager.register_adapter_type("news", AsyncNewsAdapter, {
            "concurrency": 3,
            "rate_limit_delay": 1.0
        })
    except ImportError as e:
        _logger.warning("Could not register News adapter: %s", e)

    try:
        from src.common.sentiments.adapters.async_trends import AsyncTrendsAdapter
        manager.register_adapter_type("trends", AsyncTrendsAdapter, {
            "concurrency": 1,
            "rate_limit_delay": 2.0
        })
    except ImportError as e:
        _logger.warning("Could not register Google Trends adapter: %s", e)