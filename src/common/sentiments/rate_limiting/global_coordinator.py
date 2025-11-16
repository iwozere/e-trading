"""
Global rate limit coordination across all adapters.

Provides centralized coordination of rate limits across multiple adapters
to prevent system overload and ensure fair resource allocation.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import threading

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from .adaptive_limiter import AdaptiveRateLimiter, AdaptiveConfig

_logger = setup_logger(__name__)


@dataclass
class GlobalLimitConfig:
    """Configuration for global rate limiting coordination."""
    max_global_requests_per_second: float = 10.0
    max_concurrent_requests: int = 20
    adapter_priority_weights: Dict[str, float] = None
    enable_fair_sharing: bool = True
    enable_adaptive_global_limit: bool = True
    system_load_threshold: float = 0.8
    memory_threshold_mb: int = 1024

    def __post_init__(self):
        if self.adapter_priority_weights is None:
            self.adapter_priority_weights = {}


@dataclass
class AdapterAllocation:
    """Resource allocation for an adapter."""
    adapter_name: str
    allocated_rate: float
    max_concurrent: int
    priority_weight: float
    current_usage: float = 0.0
    active_requests: int = 0


class GlobalRateLimitCoordinator:
    """
    Global coordinator for rate limiting across all sentiment adapters.

    Manages system-wide rate limits, coordinates between adapters,
    and ensures fair resource allocation while preventing overload.
    """

    def __init__(self, config: GlobalLimitConfig):
        """
        Initialize global rate limit coordinator.

        Args:
            config: Global coordination configuration
        """
        self.config = config
        self._adapters: Dict[str, AdaptiveRateLimiter] = {}
        self._allocations: Dict[str, AdapterAllocation] = {}
        self._global_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._active_requests: Set[str] = set()
        self._request_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

        # Global statistics
        self.total_global_requests = 0
        self.total_global_rejections = 0
        self.created_at = datetime.now(timezone.utc)

        # System monitoring
        self._last_reallocation = datetime.now(timezone.utc)
        self._reallocation_interval = timedelta(minutes=5)

    def register_adapter(self, name: str,
                        adaptive_config: Optional[AdaptiveConfig] = None,
                        priority_weight: float = 1.0) -> AdaptiveRateLimiter:
        """
        Register an adapter with the global coordinator.

        Args:
            name: Unique adapter name
            adaptive_config: Adaptive rate limiting configuration
            priority_weight: Priority weight for resource allocation

        Returns:
            Configured adaptive rate limiter for the adapter
        """
        with self._lock:
            if name in self._adapters:
                _logger.warning("Adapter %s already registered, replacing", name)

            # Create adaptive rate limiter
            if adaptive_config is None:
                adaptive_config = AdaptiveConfig()

            limiter = AdaptiveRateLimiter(adaptive_config, name)
            self._adapters[name] = limiter

            # Create allocation
            allocation = AdapterAllocation(
                adapter_name=name,
                allocated_rate=adaptive_config.base_requests_per_second,
                max_concurrent=max(2, int(adaptive_config.base_requests_per_second)),
                priority_weight=priority_weight
            )
            self._allocations[name] = allocation

            # Reallocate resources
            self._reallocate_resources()

            _logger.info("Registered adapter %s with global coordinator (weight: %.2f)",
                        name, priority_weight)
            return limiter

    def unregister_adapter(self, name: str) -> bool:
        """
        Unregister an adapter from the coordinator.

        Args:
            name: Adapter name to unregister

        Returns:
            True if adapter was unregistered
        """
        with self._lock:
            if name not in self._adapters:
                return False

            del self._adapters[name]
            del self._allocations[name]

            # Reallocate resources among remaining adapters
            self._reallocate_resources()

            _logger.info("Unregistered adapter %s from global coordinator", name)
            return True

    async def acquire_global_permission(self, adapter_name: str,
                                      timeout: Optional[float] = None) -> bool:
        """
        Acquire global permission for a request.

        Args:
            adapter_name: Name of the requesting adapter
            timeout: Maximum time to wait

        Returns:
            True if permission granted
        """
        if adapter_name not in self._adapters:
            _logger.warning("Unknown adapter requesting permission: %s", adapter_name)
            return False

        # Check system load
        if self._is_system_overloaded():
            _logger.debug("System overloaded, rejecting request from %s", adapter_name)
            with self._lock:
                self.total_global_rejections += 1
            return False

        # Acquire global semaphore
        try:
            acquired = await asyncio.wait_for(
                self._global_semaphore.acquire(),
                timeout=timeout
            )
            if not acquired:
                with self._lock:
                    self.total_global_rejections += 1
                return False
        except asyncio.TimeoutError:
            with self._lock:
                self.total_global_rejections += 1
            return False

        # Acquire adapter-specific permission
        adapter_limiter = self._adapters[adapter_name]
        adapter_acquired = await adapter_limiter.acquire(timeout)

        if not adapter_acquired:
            # Release global semaphore if adapter limit exceeded
            self._global_semaphore.release()
            with self._lock:
                self.total_global_rejections += 1
            return False

        # Track active request
        request_id = f"{adapter_name}_{time.time()}_{id(self)}"
        with self._lock:
            self._active_requests.add(request_id)
            self._allocations[adapter_name].active_requests += 1
            self.total_global_requests += 1

        return True

    def release_global_permission(self, adapter_name: str) -> None:
        """
        Release global permission after request completion.

        Args:
            adapter_name: Name of the adapter releasing permission
        """
        # Release global semaphore
        self._global_semaphore.release()

        # Update tracking
        with self._lock:
            if adapter_name in self._allocations:
                self._allocations[adapter_name].active_requests = max(
                    0, self._allocations[adapter_name].active_requests - 1
                )

    def record_adapter_response(self, adapter_name: str,
                              response_time_ms: float,
                              success: bool,
                              status_code: Optional[int] = None,
                              error_type: Optional[str] = None) -> None:
        """
        Record response metrics for an adapter.

        Args:
            adapter_name: Name of the adapter
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
            status_code: HTTP status code (if applicable)
            error_type: Type of error (if any)
        """
        if adapter_name not in self._adapters:
            return

        # Record with adapter's limiter
        adapter_limiter = self._adapters[adapter_name]
        adapter_limiter.record_response(response_time_ms, success, status_code, error_type)

        # Record global metrics
        self._request_history.append({
            'adapter': adapter_name,
            'timestamp': datetime.now(timezone.utc),
            'response_time_ms': response_time_ms,
            'success': success,
            'status_code': status_code,
            'error_type': error_type
        })

        # Keep history manageable
        if len(self._request_history) > 1000:
            self._request_history = self._request_history[-500:]

        # Check if reallocation is needed
        self._check_reallocation_schedule()

    def _reallocate_resources(self) -> None:
        """Reallocate rate limits among adapters based on priority and performance."""
        if not self._adapters:
            return

        with self._lock:
            total_weight = sum(alloc.priority_weight for alloc in self._allocations.values())
            if total_weight == 0:
                return

            # Calculate base allocations
            for name, allocation in self._allocations.items():
                # Base allocation based on priority weight
                base_rate = (allocation.priority_weight / total_weight) * self.config.max_global_requests_per_second

                # Adjust based on recent performance if adaptive is enabled
                if self.config.enable_adaptive_global_limit:
                    adapter_limiter = self._adapters[name]
                    perf_metrics = adapter_limiter.get_performance_metrics()

                    if perf_metrics:
                        error_rate = perf_metrics.get('error_rate', 0)

                        # Reduce allocation for adapters with high error rates
                        if error_rate > 0.2:
                            base_rate *= 0.7
                        elif error_rate > 0.1:
                            base_rate *= 0.85
                        # Increase allocation for well-performing adapters
                        elif error_rate < 0.05:
                            base_rate *= 1.1

                # Update allocation
                allocation.allocated_rate = max(0.1, min(base_rate, self.config.max_global_requests_per_second))
                allocation.max_concurrent = max(1, int(allocation.allocated_rate * 2))

                # Update adapter's rate limiter
                adapter_limiter = self._adapters[name]
                adapter_limiter.force_rate_adjustment(allocation.allocated_rate, "global_reallocation")

            self._last_reallocation = datetime.now(timezone.utc)

            _logger.debug("Reallocated resources: %s",
                         {name: f"{alloc.allocated_rate:.2f} req/s"
                          for name, alloc in self._allocations.items()})

    def _check_reallocation_schedule(self) -> None:
        """Check if it's time for resource reallocation."""
        if datetime.now(timezone.utc) - self._last_reallocation > self._reallocation_interval:
            self._reallocate_resources()

    def _is_system_overloaded(self) -> bool:
        """Check if the system is currently overloaded."""
        try:
            import psutil

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.config.system_load_threshold * 100:
                return True

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.used / (1024 * 1024) > self.config.memory_threshold_mb:
                return True

            # Check active request count
            with self._lock:
                total_active = sum(alloc.active_requests for alloc in self._allocations.values())
                if total_active >= self.config.max_concurrent_requests * 0.9:
                    return True

            return False

        except Exception as e:
            _logger.debug("Failed to check system load: %s", e)
            return False

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get comprehensive global coordination statistics."""
        with self._lock:
            uptime = (datetime.now(timezone.utc) - self.created_at).total_seconds()
            total_active = sum(alloc.active_requests for alloc in self._allocations.values())

            stats = {
                'global_metrics': {
                    'total_requests': self.total_global_requests,
                    'total_rejections': self.total_global_rejections,
                    'success_rate': (self.total_global_requests - self.total_global_rejections) / max(1, self.total_global_requests),
                    'active_requests': total_active,
                    'max_concurrent': self.config.max_concurrent_requests,
                    'utilization': total_active / self.config.max_concurrent_requests,
                    'uptime_seconds': uptime
                },
                'adapter_allocations': {
                    name: {
                        'allocated_rate': alloc.allocated_rate,
                        'max_concurrent': alloc.max_concurrent,
                        'priority_weight': alloc.priority_weight,
                        'active_requests': alloc.active_requests,
                        'utilization': alloc.active_requests / max(1, alloc.max_concurrent)
                    }
                    for name, alloc in self._allocations.items()
                },
                'system_status': {
                    'overloaded': self._is_system_overloaded(),
                    'last_reallocation': self._last_reallocation,
                    'registered_adapters': len(self._adapters)
                }
            }

        return stats

    def get_adapter_statistics(self, adapter_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific adapter.

        Args:
            adapter_name: Name of the adapter

        Returns:
            Adapter statistics or None if not found
        """
        if adapter_name not in self._adapters:
            return None

        adapter_limiter = self._adapters[adapter_name]
        allocation = self._allocations[adapter_name]

        return {
            'allocation': {
                'allocated_rate': allocation.allocated_rate,
                'max_concurrent': allocation.max_concurrent,
                'priority_weight': allocation.priority_weight,
                'active_requests': allocation.active_requests
            },
            'adaptive_stats': adapter_limiter.get_comprehensive_stats()
        }

    def update_adapter_priority(self, adapter_name: str, new_weight: float) -> bool:
        """
        Update priority weight for an adapter.

        Args:
            adapter_name: Name of the adapter
            new_weight: New priority weight

        Returns:
            True if updated successfully
        """
        with self._lock:
            if adapter_name not in self._allocations:
                return False

            old_weight = self._allocations[adapter_name].priority_weight
            self._allocations[adapter_name].priority_weight = new_weight

            # Trigger reallocation
            self._reallocate_resources()

            _logger.info("Updated priority for %s: %.2f -> %.2f",
                        adapter_name, old_weight, new_weight)
            return True

    def force_global_rate_limit(self, new_limit: float) -> None:
        """
        Force a new global rate limit.

        Args:
            new_limit: New global requests per second limit
        """
        old_limit = self.config.max_global_requests_per_second
        self.config.max_global_requests_per_second = new_limit

        # Reallocate with new limit
        self._reallocate_resources()

        _logger.info("Updated global rate limit: %.2f -> %.2f req/s",
                    old_limit, new_limit)

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health indicators."""
        return {
            'system_overloaded': self._is_system_overloaded(),
            'global_utilization': len(self._active_requests) / self.config.max_concurrent_requests,
            'adapter_count': len(self._adapters),
            'total_active_requests': len(self._active_requests),
            'recent_error_rate': self._calculate_recent_error_rate(),
            'avg_response_time_ms': self._calculate_avg_response_time()
        }

    def _calculate_recent_error_rate(self) -> float:
        """Calculate error rate from recent requests."""
        if not self._request_history:
            return 0.0

        recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        recent_requests = [r for r in self._request_history if r['timestamp'] > recent_cutoff]

        if not recent_requests:
            return 0.0

        error_count = sum(1 for r in recent_requests if not r['success'])
        return error_count / len(recent_requests)

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent requests."""
        if not self._request_history:
            return 0.0

        recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
        recent_requests = [r for r in self._request_history
                          if r['timestamp'] > recent_cutoff and r['success']]

        if not recent_requests:
            return 0.0

        total_time = sum(r['response_time_ms'] for r in recent_requests)
        return total_time / len(recent_requests)


# Global coordinator instance
_global_coordinator: Optional[GlobalRateLimitCoordinator] = None


def get_global_coordinator() -> Optional[GlobalRateLimitCoordinator]:
    """Get the global rate limit coordinator instance."""
    return _global_coordinator


def initialize_global_coordinator(config: GlobalLimitConfig) -> GlobalRateLimitCoordinator:
    """Initialize the global rate limit coordinator."""
    global _global_coordinator
    _global_coordinator = GlobalRateLimitCoordinator(config)
    _logger.info("Initialized global rate limit coordinator")
    return _global_coordinator