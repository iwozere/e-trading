"""
System Health Service

Service layer for system health monitoring operations.
Provides high-level business logic for monitoring all subsystems including notification channels,
telegram bot, API services, web UI, trading components, and system resources.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import sys
import json
from functools import wraps

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.base_service import BaseDBService, with_uow, handle_db_error
from src.data.db.repos.repo_system_health import SystemHealthRepository
from src.data.db.models.model_system_health import SystemHealth, SystemType, SystemHealthStatus
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SystemHealthService(BaseDBService):
    """
    Service layer for system health monitoring operations.

    Provides high-level business logic for monitoring and managing system health
    across all subsystems in the e-trading platform.
    """

    def __init__(self):
        """Initialize the system health service."""
        super().__init__()

    @with_uow
    @handle_db_error
    def update_system_health(
        self,
        system: str,
        component: Optional[str] = None,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemHealth:
        """
        Update system health status.

        Args:
            system: System name (e.g., 'notification', 'telegram_bot', 'api_service')
            component: Component name (e.g., 'email', 'slack' for notification system)
            status: Health status
            response_time_ms: Response time in milliseconds
            error_message: Error message if status is not healthy
            metadata: Additional metadata dictionary

        Returns:
            Updated SystemHealth object
        """
        metadata_json = None
        if metadata:
            metadata_json = json.dumps(metadata)

        health_record = self.uow.system_health.update_system_status(
            system=system,
            component=component,
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata_json
        )

        _logger.debug(
            "Updated health for %s.%s: %s",
            system,
            component or 'main',
            status.value
        )
        return health_record

    @with_uow
    @handle_db_error
    def get_system_health(self, system: str, component: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get health status for a specific system/component.

        Args:
            system: System name
            component: Component name (optional)

        Returns:
            Dictionary with health data or None if not found
        """
        health_record = self.uow.system_health.get_system_health(system, component)

        if not health_record:
            return None

        return self._format_health_record(health_record)

    @with_uow
    @handle_db_error
    def get_all_systems_health(self, include_stale: bool = True) -> List[Dict[str, Any]]:
        """
        Get health status for all systems.

        Args:
            include_stale: Whether to include stale records

        Returns:
            List of dictionaries with health data
        """
        health_records = self.uow.system_health.list_system_health(include_stale=include_stale)
        return [self._format_health_record(record) for record in health_records]

    @with_uow
    @handle_db_error
    def get_systems_overview(self) -> Dict[str, Any]:
        """
        Get overview of all systems with aggregated statistics.

        Returns:
            Dictionary with systems overview data
        """
        overview_data = self.uow.system_health.get_systems_overview()
        statistics = self.uow.system_health.get_health_statistics()

        # Determine overall platform status
        overall_status = "HEALTHY"
        if statistics.get('down_systems', 0) > 0:
            overall_status = "DOWN"
        elif statistics.get('degraded_systems', 0) > 0:
            overall_status = "DEGRADED"
        elif statistics.get('unknown_systems', 0) > 0:
            overall_status = "UNKNOWN"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "systems_overview": overview_data,
            "statistics": statistics
        }

    @with_uow
    @handle_db_error
    def get_unhealthy_systems(self) -> List[Dict[str, Any]]:
        """
        Get all systems that are not healthy.

        Returns:
            List of dictionaries with unhealthy systems data
        """
        unhealthy_records = self.uow.system_health.get_unhealthy_systems()
        return [self._format_health_record(record) for record in unhealthy_records]

    # Backward compatibility methods for notification channels

    @with_uow
    @handle_db_error
    def get_notification_channels_health(self) -> List[Dict[str, Any]]:
        """
        Get health status for all notification channels (backward compatibility).

        Returns:
            List of dictionaries with channel health data
        """
        channel_records = self.uow.system_health.get_notification_channels_health()
        return [self._format_channel_health_record(record) for record in channel_records]

    @with_uow
    @handle_db_error
    def get_notification_channel_health(self, channel: str) -> Optional[Dict[str, Any]]:
        """
        Get health status for a specific notification channel (backward compatibility).

        Args:
            channel: Channel name (e.g., 'email', 'slack', 'discord')

        Returns:
            Dictionary with channel health data or None if not found
        """
        channel_record = self.uow.system_health.get_notification_channel_health(channel)

        if not channel_record:
            return None

        return self._format_channel_health_record(channel_record)

    def update_notification_channel_health(
        self,
        channel: str,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemHealth:
        """
        Update notification channel health (backward compatibility).

        Args:
            channel: Channel name (e.g., 'email', 'slack', 'discord')
            status: Health status
            response_time_ms: Response time in milliseconds
            error_message: Error message if status is not healthy
            metadata: Additional metadata dictionary

        Returns:
            Updated SystemHealth object
        """
        return self.update_system_health(
            system='notification',
            component=channel,
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

    # System-specific health update methods

    def update_telegram_bot_health(
        self,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemHealth:
        """Update Telegram bot health status."""
        return self.update_system_health(
            system='telegram_bot',
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

    def update_api_service_health(
        self,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemHealth:
        """Update API service health status."""
        return self.update_system_health(
            system='api_service',
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

    def update_web_ui_health(
        self,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemHealth:
        """Update Web UI health status."""
        return self.update_system_health(
            system='web_ui',
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

    def update_trading_bot_health(
        self,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemHealth:
        """Update trading bot health status."""
        return self.update_system_health(
            system='trading_bot',
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

    def update_database_health(
        self,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemHealth:
        """Update database health status."""
        return self.update_system_health(
            system='database',
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

    # Utility methods

    def cleanup_stale_records(self, stale_threshold_hours: int = 24) -> int:
        """
        Clean up stale health records.

        Args:
            stale_threshold_hours: Hours after which a record is considered stale

        Returns:
            Number of records deleted
        """
        try:
            deleted_count = self.repo.cleanup_stale_records(stale_threshold_hours)
            _logger.info("Cleaned up %d stale health records", deleted_count)
            return deleted_count

        except Exception as e:
            _logger.exception("Failed to cleanup stale records:")
            raise

    def delete_system_health(self, system: str, component: Optional[str] = None) -> bool:
        """
        Delete a system health record.

        Args:
            system: System name
            component: Component name (optional)

        Returns:
            True if record was deleted, False if not found
        """
        try:
            return self.repo.delete_system_health(system, component)

        except Exception as e:
            _logger.error("Failed to delete system health for %s.%s: %s", system, component, e)
            raise

    # Private helper methods

    def _format_health_record(self, record: SystemHealth) -> Dict[str, Any]:
        """Format a SystemHealth record as a dictionary."""
        metadata = None
        if record.metadata:
            try:
                metadata = json.loads(record.metadata)
            except json.JSONDecodeError:
                metadata = {"raw": record.metadata}

        return {
            "system": record.system,
            "component": record.component,
            "status": record.status,
            "last_success": record.last_success.isoformat() if record.last_success else None,
            "last_failure": record.last_failure.isoformat() if record.last_failure else None,
            "failure_count": record.failure_count,
            "avg_response_time_ms": record.avg_response_time_ms,
            "error_message": record.error_message,
            "metadata": metadata,
            "checked_at": record.checked_at.isoformat(),
            "system_identifier": record.system_identifier,
            "is_healthy": record.is_healthy,
            "is_degraded": record.is_degraded,
            "is_down": record.is_down
        }

    def _format_channel_health_record(self, record: SystemHealth) -> Dict[str, Any]:
        """Format a SystemHealth record as a channel health dictionary (backward compatibility)."""
        metadata = None
        if record.metadata:
            try:
                metadata = json.loads(record.metadata)
            except json.JSONDecodeError:
                metadata = {"raw": record.metadata}

        return {
            "channel": record.component,  # Map component to channel for backward compatibility
            "status": record.status,
            "last_success": record.last_success.isoformat() if record.last_success else None,
            "last_failure": record.last_failure.isoformat() if record.last_failure else None,
            "failure_count": record.failure_count,
            "avg_response_time_ms": record.avg_response_time_ms,
            "error_message": record.error_message,
            "metadata": metadata,
            "checked_at": record.checked_at.isoformat(),
            "is_healthy": record.is_healthy
        }