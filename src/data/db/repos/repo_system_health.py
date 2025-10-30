"""
System Health Repository

Repository layer for system health monitoring operations.
Provides data access methods for monitoring all subsystems including notification channels,
telegram bot, API services, web UI, trading components, and system resources.
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.exc import IntegrityError

from src.data.db.models.model_system_health import SystemHealth, SystemType, SystemHealthStatus
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SystemHealthRepository:
    """Repository for system health operations."""

    def __init__(self, session: Session):
        """
        Initialize the repository with a database session.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def create_or_update_system_health(self, health_data: Dict[str, Any]) -> SystemHealth:
        """
        Create or update system health record.

        Args:
            health_data: Dictionary with health data containing:
                - system: System name (required)
                - component: Component name (optional)
                - status: Health status (required)
                - last_success: Last successful check timestamp (optional)
                - last_failure: Last failure timestamp (optional)
                - failure_count: Number of consecutive failures (optional)
                - avg_response_time_ms: Average response time (optional)
                - error_message: Error message (optional)
                - metadata: Additional metadata as JSON string (optional)

        Returns:
            Created or updated SystemHealth object

        Raises:
            IntegrityError: If there's a database constraint violation
            Exception: For other database errors
        """
        try:
            system = health_data.get('system')
            component = health_data.get('component')

            if not system:
                raise ValueError("System name is required")

            # Try to find existing record
            existing = self.get_system_health(system, component)

            if existing:
                # Update existing record
                existing.status = health_data.get('status', existing.status)
                existing.last_success = health_data.get('last_success', existing.last_success)
                existing.last_failure = health_data.get('last_failure', existing.last_failure)
                existing.failure_count = health_data.get('failure_count', existing.failure_count)
                existing.avg_response_time_ms = health_data.get('avg_response_time_ms', existing.avg_response_time_ms)
                existing.error_message = health_data.get('error_message', existing.error_message)
                existing.metadata = health_data.get('metadata', existing.metadata)
                existing.checked_at = health_data.get('checked_at', datetime.now(timezone.utc))

                self.session.flush()
                _logger.debug("Updated system health for %s.%s", system, component or 'main')
                return existing
            else:
                # Create new record
                health_record = SystemHealth(
                    system=system,
                    component=component,
                    status=health_data.get('status', SystemHealthStatus.UNKNOWN.value),
                    last_success=health_data.get('last_success'),
                    last_failure=health_data.get('last_failure'),
                    failure_count=health_data.get('failure_count', 0),
                    avg_response_time_ms=health_data.get('avg_response_time_ms'),
                    error_message=health_data.get('error_message'),
                    metadata=health_data.get('metadata'),
                    checked_at=health_data.get('checked_at', datetime.now(timezone.utc))
                )

                self.session.add(health_record)
                self.session.flush()
                _logger.debug("Created system health record for %s.%s", system, component or 'main')
                return health_record

        except IntegrityError as e:
            _logger.exception("Integrity error creating/updating system health:")
            self.session.rollback()
            raise
        except Exception as e:
            _logger.exception("Error creating/updating system health:")
            self.session.rollback()
            raise

    def get_system_health(self, system: str, component: Optional[str] = None) -> Optional[SystemHealth]:
        """
        Get system health by system and component name.

        Args:
            system: System name
            component: Component name (optional)

        Returns:
            SystemHealth object if found, None otherwise
        """
        query = self.session.query(SystemHealth).filter(SystemHealth.system == system)

        if component:
            query = query.filter(SystemHealth.component == component)
        else:
            query = query.filter(SystemHealth.component.is_(None))

        return query.first()

    def list_system_health(
        self,
        system: Optional[str] = None,
        status: Optional[SystemHealthStatus] = None,
        include_stale: bool = True,
        stale_threshold_minutes: int = 10
    ) -> List[SystemHealth]:
        """
        List system health records with optional filtering.

        Args:
            system: Filter by system name (optional)
            status: Filter by health status (optional)
            include_stale: Whether to include stale records (default: True)
            stale_threshold_minutes: Minutes after which a record is considered stale

        Returns:
            List of SystemHealth objects
        """
        query = self.session.query(SystemHealth)

        if system:
            query = query.filter(SystemHealth.system == system)

        if status:
            query = query.filter(SystemHealth.status == status.value)

        if not include_stale:
            stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_threshold_minutes)
            query = query.filter(SystemHealth.checked_at >= stale_cutoff)

        return query.order_by(SystemHealth.system, SystemHealth.component).all()

    def get_systems_overview(self) -> List[Dict[str, Any]]:
        """
        Get an overview of all systems with aggregated health statistics.

        Returns:
            List of dictionaries with system overview data
        """
        query = text("""
            SELECT
                system,
                COUNT(*) as total_components,
                COUNT(CASE WHEN status = 'HEALTHY' THEN 1 END) as healthy_components,
                COUNT(CASE WHEN status = 'DEGRADED' THEN 1 END) as degraded_components,
                COUNT(CASE WHEN status = 'DOWN' THEN 1 END) as down_components,
                COUNT(CASE WHEN status = 'UNKNOWN' THEN 1 END) as unknown_components,
                CASE
                    WHEN COUNT(CASE WHEN status = 'DOWN' THEN 1 END) > 0 THEN 'DOWN'
                    WHEN COUNT(CASE WHEN status = 'DEGRADED' THEN 1 END) > 0 THEN 'DEGRADED'
                    WHEN COUNT(CASE WHEN status = 'HEALTHY' THEN 1 END) = COUNT(*) THEN 'HEALTHY'
                    ELSE 'UNKNOWN'
                END as overall_status,
                AVG(avg_response_time_ms) as avg_response_time_ms,
                MAX(checked_at) as last_checked
            FROM msg_system_health
            GROUP BY system
            ORDER BY
                CASE
                    WHEN COUNT(CASE WHEN status = 'DOWN' THEN 1 END) > 0 THEN 1
                    WHEN COUNT(CASE WHEN status = 'DEGRADED' THEN 1 END) > 0 THEN 2
                    ELSE 3
                END,
                system
        """)

        result = self.session.execute(query)
        return [dict(row._mapping) for row in result]

    def get_unhealthy_systems(self) -> List[SystemHealth]:
        """
        Get all systems that are not healthy.

        Returns:
            List of SystemHealth objects with non-healthy status
        """
        return self.session.query(SystemHealth).filter(
            SystemHealth.status != SystemHealthStatus.HEALTHY.value
        ).order_by(
            SystemHealth.system,
            SystemHealth.component
        ).all()

    def get_notification_channels_health(self) -> List[SystemHealth]:
        """
        Get health status for all notification channels (backward compatibility).

        Returns:
            List of SystemHealth objects for notification system components
        """
        return self.session.query(SystemHealth).filter(
            SystemHealth.system == 'notification'
        ).order_by(SystemHealth.component).all()

    def get_notification_channel_health(self, channel: str) -> Optional[SystemHealth]:
        """
        Get health status for a specific notification channel (backward compatibility).

        Args:
            channel: Channel name (e.g., 'email', 'slack', 'discord')

        Returns:
            SystemHealth object if found, None otherwise
        """
        return self.session.query(SystemHealth).filter(
            and_(
                SystemHealth.system == 'notification',
                SystemHealth.component == channel
            )
        ).first()

    def update_system_status(
        self,
        system: str,
        component: Optional[str],
        status: SystemHealthStatus,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[str] = None
    ) -> SystemHealth:
        """
        Update system health status with automatic failure counting and timestamps.

        Args:
            system: System name
            component: Component name (optional)
            status: New health status
            response_time_ms: Response time in milliseconds (optional)
            error_message: Error message if status is not healthy (optional)
            metadata: Additional metadata as JSON string (optional)

        Returns:
            Updated SystemHealth object
        """
        health_record = self.get_system_health(system, component)

        if not health_record:
            # Create new record if it doesn't exist
            health_data = {
                'system': system,
                'component': component,
                'status': status.value,
                'failure_count': 1 if status != SystemHealthStatus.HEALTHY else 0,
                'avg_response_time_ms': response_time_ms,
                'error_message': error_message,
                'metadata': metadata
            }

            if status == SystemHealthStatus.HEALTHY:
                health_data['last_success'] = datetime.now(timezone.utc)
            else:
                health_data['last_failure'] = datetime.now(timezone.utc)

            return self.create_or_update_system_health(health_data)

        # Update existing record using the model's method
        health_record.update_health_status(
            status=status,
            response_time_ms=response_time_ms,
            error_message=error_message,
            metadata=metadata
        )

        self.session.flush()
        return health_record

    def delete_system_health(self, system: str, component: Optional[str] = None) -> bool:
        """
        Delete a system health record.

        Args:
            system: System name
            component: Component name (optional)

        Returns:
            True if record was deleted, False if not found
        """
        health_record = self.get_system_health(system, component)

        if health_record:
            self.session.delete(health_record)
            self.session.flush()
            _logger.info("Deleted system health record for %s.%s", system, component or 'main')
            return True

        return False

    def cleanup_stale_records(self, stale_threshold_hours: int = 24) -> int:
        """
        Clean up stale health records that haven't been updated recently.

        Args:
            stale_threshold_hours: Hours after which a record is considered stale

        Returns:
            Number of records deleted
        """
        stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=stale_threshold_hours)

        deleted_count = self.session.query(SystemHealth).filter(
            SystemHealth.checked_at < stale_cutoff
        ).delete()

        self.session.flush()
        _logger.info("Cleaned up %d stale system health records", deleted_count)
        return deleted_count

    def get_health_statistics(self) -> Dict[str, Any]:
        """
        Get overall health statistics across all systems.

        Returns:
            Dictionary with health statistics
        """
        query = text("""
            SELECT
                COUNT(*) as total_systems,
                COUNT(CASE WHEN status = 'HEALTHY' THEN 1 END) as healthy_systems,
                COUNT(CASE WHEN status = 'DEGRADED' THEN 1 END) as degraded_systems,
                COUNT(CASE WHEN status = 'DOWN' THEN 1 END) as down_systems,
                COUNT(CASE WHEN status = 'UNKNOWN' THEN 1 END) as unknown_systems,
                AVG(avg_response_time_ms) as overall_avg_response_time_ms,
                COUNT(DISTINCT system) as unique_systems
            FROM msg_system_health
        """)

        result = self.session.execute(query).first()
        return dict(result._mapping) if result else {}