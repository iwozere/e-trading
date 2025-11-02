"""
Comprehensive system health monitoring service.

This module provides centralized health monitoring for all subsystems
in the e-trading platform.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

import aiohttp
import psutil
from sqlalchemy.exc import SQLAlchemyError

from src.data.db.services.database_service import DatabaseService
from src.data.db.models.model_system_health import SystemHealth, SystemType, SystemHealthStatus
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    system: str
    component: Optional[str]
    status: SystemHealthStatus
    response_time_ms: Optional[int]
    error_message: Optional[str]
    metadata: Optional[Dict[str, Any]] = None


class HealthMonitor:
    """
    Centralized health monitoring service for all subsystems.
    """

    def __init__(self, db_service: Optional[DatabaseService] = None):
        """
        Initialize the health monitor.

        Args:
            db_service: Database service instance
        """
        self.db_service = db_service or DatabaseService()
        self._health_checkers: Dict[str, Callable] = {}
        self._running = False
        self._check_interval = 60  # seconds

        # Register default health checkers
        self._register_default_checkers()

    def _register_default_checkers(self):
        """Register default health check functions for core systems."""
        self._health_checkers.update({
            "database": self._check_database_health,
            "notification": self._check_notification_health,
            "telegram_bot": self._check_telegram_bot_health,
            "api_service": self._check_api_service_health,
            "web_ui": self._check_web_ui_health,
            "trading_bot": self._check_trading_bot_health,
            "system_resources": self._check_system_resources,
        })

    def register_health_checker(self, system: str, checker_func: Callable):
        """
        Register a custom health checker function.

        Args:
            system: System name
            checker_func: Async function that returns HealthCheckResult
        """
        self._health_checkers[system] = checker_func
        _logger.info("Registered health checker for system: %s", system)

    async def check_system_health(self, system: str, component: Optional[str] = None) -> HealthCheckResult:
        """
        Check health of a specific system.

        Args:
            system: System name
            component: Optional component name

        Returns:
            HealthCheckResult instance
        """
        start_time = time.time()

        try:
            if system in self._health_checkers:
                result = await self._health_checkers[system](component)
                result.response_time_ms = int((time.time() - start_time) * 1000)
                return result
            else:
                return HealthCheckResult(
                    system=system,
                    component=component,
                    status=SystemHealthStatus.UNKNOWN,
                    response_time_ms=None,
                    error_message=f"No health checker registered for system: {system}"
                )
        except Exception as e:
            _logger.exception("Health check failed for system %s:", system)
            return HealthCheckResult(
                system=system,
                component=component,
                status=SystemHealthStatus.DOWN,
                response_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e)
            )

    async def check_all_systems_health(self) -> List[HealthCheckResult]:
        """
        Check health of all registered systems.

        Returns:
            List of HealthCheckResult instances
        """
        results = []

        # Run all health checks concurrently
        tasks = [
            self.check_system_health(system)
            for system in self._health_checkers.keys()
        ]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert exceptions to error results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    system = list(self._health_checkers.keys())[i]
                    results[i] = HealthCheckResult(
                        system=system,
                        component=None,
                        status=SystemHealthStatus.DOWN,
                        response_time_ms=None,
                        error_message=str(result)
                    )
        except Exception as e:
            _logger.exception("Failed to check all systems health:")

        return results

    async def update_health_status(self, result: HealthCheckResult):
        """
        Update health status in the database.

        Args:
            result: HealthCheckResult to store
        """
        try:
            with self.db_service.uow() as uow:
                # Get or create system health record
                health_record = SystemHealth.get_system_status(
                    uow.s, result.system, result.component
                )

                if not health_record:
                    health_record = SystemHealth(
                        system=result.system,
                        component=result.component,
                        status=result.status.value,
                        failure_count=0
                    )
                    uow.s.add(health_record)

                # Update health status
                metadata_json = None
                if result.metadata:
                    metadata_json = json.dumps(result.metadata)

                health_record.update_health_status(
                    status=result.status,
                    response_time_ms=result.response_time_ms,
                    error_message=result.error_message,
                    metadata=metadata_json
                )

                uow.commit()

        except Exception as e:
            _logger.exception("Failed to update health status for %s:", result.system)

    async def get_system_health_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive health summary for all systems.

        Returns:
            Dictionary with health summary data
        """
        try:
            with self.db_service.uow() as uow:
                # Get all system health records
                health_records = SystemHealth.get_all_systems_status(uow.s)

                summary = {
                    "overall_status": "HEALTHY",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "systems": {},
                    "statistics": {
                        "total_systems": 0,
                        "healthy_systems": 0,
                        "degraded_systems": 0,
                        "down_systems": 0,
                        "unknown_systems": 0
                    }
                }

                # Process each health record
                for record in health_records:
                    system_key = record.system_identifier

                    system_data = {
                        "status": record.status,
                        "last_success": record.last_success.isoformat() if record.last_success else None,
                        "last_failure": record.last_failure.isoformat() if record.last_failure else None,
                        "failure_count": record.failure_count,
                        "avg_response_time_ms": record.avg_response_time_ms,
                        "error_message": record.error_message,
                        "checked_at": record.checked_at.isoformat(),
                        "metadata": json.loads(record.metadata) if record.metadata else None
                    }

                    summary["systems"][system_key] = system_data

                    # Update statistics
                    summary["statistics"]["total_systems"] += 1
                    if record.status == SystemHealthStatus.HEALTHY.value:
                        summary["statistics"]["healthy_systems"] += 1
                    elif record.status == SystemHealthStatus.DEGRADED.value:
                        summary["statistics"]["degraded_systems"] += 1
                        if summary["overall_status"] == "HEALTHY":
                            summary["overall_status"] = "DEGRADED"
                    elif record.status == SystemHealthStatus.DOWN.value:
                        summary["statistics"]["down_systems"] += 1
                        summary["overall_status"] = "DOWN"
                    else:
                        summary["statistics"]["unknown_systems"] += 1

                return summary

        except Exception as e:
            _logger.exception("Failed to get health summary:")
            return {
                "overall_status": "UNKNOWN",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "systems": {},
                "statistics": {}
            }

    # Health checker implementations

    async def _check_database_health(self, component: Optional[str] = None) -> HealthCheckResult:
        """Check database connectivity and performance."""
        try:
            with self.db_service.uow() as uow:
                # Test basic connectivity
                start_time = time.time()
                result = uow.s.execute("SELECT 1").scalar()
                query_time = int((time.time() - start_time) * 1000)

                if result == 1:
                    # Additional checks for database health
                    metadata = {
                        "query_time_ms": query_time,
                        "connection_pool_size": getattr(self.db_service.engine.pool, 'size', 'unknown'),
                        "checked_out_connections": getattr(self.db_service.engine.pool, 'checkedout', 'unknown')
                    }

                    status = SystemHealthStatus.HEALTHY
                    if query_time > 1000:  # > 1 second is degraded
                        status = SystemHealthStatus.DEGRADED

                    return HealthCheckResult(
                        system="database",
                        component=component,
                        status=status,
                        response_time_ms=query_time,
                        error_message=None,
                        metadata=metadata
                    )
                else:
                    return HealthCheckResult(
                        system="database",
                        component=component,
                        status=SystemHealthStatus.DOWN,
                        response_time_ms=query_time,
                        error_message="Database query returned unexpected result"
                    )

        except SQLAlchemyError as e:
            return HealthCheckResult(
                system="database",
                component=component,
                status=SystemHealthStatus.DOWN,
                response_time_ms=None,
                error_message=f"Database error: {str(e)}"
            )

    async def _check_notification_health(self, component: Optional[str] = None) -> HealthCheckResult:
        """Check notification service health via database (database-centric architecture)."""
        try:
            # Check notification service health from database instead of HTTP
            from src.data.db.services.system_health_service import SystemHealthService

            # Initialize health service (no arguments needed - uses @with_uow decorator)
            health_service = SystemHealthService()

            # Get notification system health from database (method manages its own UoW context)
            system_health = health_service.get_system_health("notification", component)

            if system_health:
                # Convert database health status to HealthCheckResult
                return HealthCheckResult(
                    system="notification",
                    component=component,
                    status=system_health.get('status', SystemHealthStatus.UNKNOWN),
                    response_time_ms=system_health.get('avg_response_time_ms'),
                    error_message=system_health.get('error_message'),
                    metadata=system_health.get('metadata', {})
                )
            else:
                # No health record found - service might not be running
                return HealthCheckResult(
                    system="notification",
                    component=component,
                    status=SystemHealthStatus.UNKNOWN,
                    response_time_ms=None,
                    error_message="No health record found in database"
                )

        except Exception as e:
            return HealthCheckResult(
                system="notification",
                component=component,
                status=SystemHealthStatus.DOWN,
                response_time_ms=None,
                error_message=f"Database health check failed: {str(e)}"
            )

    async def _check_telegram_bot_health(self, component: Optional[str] = None) -> HealthCheckResult:
        """Check Telegram bot health."""
        # This would need to be implemented based on your telegram bot architecture
        # For now, return a placeholder
        return HealthCheckResult(
            system="telegram_bot",
            component=component,
            status=SystemHealthStatus.UNKNOWN,
            response_time_ms=None,
            error_message="Health check not implemented yet"
        )

    async def _check_api_service_health(self, component: Optional[str] = None) -> HealthCheckResult:
        """Check API service health."""
        # This would check your main API service
        return HealthCheckResult(
            system="api_service",
            component=component,
            status=SystemHealthStatus.UNKNOWN,
            response_time_ms=None,
            error_message="Health check not implemented yet"
        )

    async def _check_web_ui_health(self, component: Optional[str] = None) -> HealthCheckResult:
        """Check Web UI health."""
        # This would check your web UI service
        return HealthCheckResult(
            system="web_ui",
            component=component,
            status=SystemHealthStatus.UNKNOWN,
            response_time_ms=None,
            error_message="Health check not implemented yet"
        )

    async def _check_trading_bot_health(self, component: Optional[str] = None) -> HealthCheckResult:
        """Check trading bot health."""
        # This would check your trading bot service
        return HealthCheckResult(
            system="trading_bot",
            component=component,
            status=SystemHealthStatus.UNKNOWN,
            response_time_ms=None,
            error_message="Health check not implemented yet"
        )

    async def _check_system_resources(self, component: Optional[str] = None) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            metadata = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }

            # Determine status based on resource usage
            status = SystemHealthStatus.HEALTHY
            error_messages = []

            if cpu_percent > 90:
                status = SystemHealthStatus.DEGRADED
                error_messages.append(f"High CPU usage: {cpu_percent}%")

            if memory.percent > 90:
                status = SystemHealthStatus.DEGRADED
                error_messages.append(f"High memory usage: {memory.percent}%")

            if disk.percent > 90:
                status = SystemHealthStatus.DEGRADED
                error_messages.append(f"High disk usage: {disk.percent}%")

            if cpu_percent > 95 or memory.percent > 95 or disk.percent > 95:
                status = SystemHealthStatus.DOWN

            return HealthCheckResult(
                system="system_resources",
                component=component,
                status=status,
                response_time_ms=None,
                error_message="; ".join(error_messages) if error_messages else None,
                metadata=metadata
            )

        except Exception as e:
            return HealthCheckResult(
                system="system_resources",
                component=component,
                status=SystemHealthStatus.DOWN,
                response_time_ms=None,
                error_message=str(e)
            )

    async def start_monitoring(self, interval: int = 60):
        """
        Start continuous health monitoring.

        Args:
            interval: Check interval in seconds
        """
        self._check_interval = interval
        self._running = True

        _logger.info("Starting health monitoring with %d second interval", interval)

        while self._running:
            try:
                # Check all systems
                results = await self.check_all_systems_health()

                # Update database with results
                for result in results:
                    await self.update_health_status(result)

                _logger.debug("Completed health check cycle for %d systems", len(results))

            except Exception as e:
                _logger.exception("Error during health monitoring cycle:")

            # Wait for next cycle
            await asyncio.sleep(self._check_interval)

    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._running = False
        _logger.info("Stopped health monitoring")


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor