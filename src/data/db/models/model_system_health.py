"""
System health monitoring models.

This module provides models for monitoring the health of all subsystems
in the e-trading platform, including notification channels, telegram bot,
API services, web UI, and trading components.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import (
    BigInteger, CheckConstraint, Column, DateTime, Integer, String, Text, func
)

from src.data.db.core.base import Base
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SystemType(Enum):
    """Enumeration of system types that can be monitored."""

    NOTIFICATION = "notification"
    TELEGRAM_BOT = "telegram_bot"
    API_SERVICE = "api_service"
    WEB_UI = "web_ui"
    TRADING_BOT = "trading_bot"
    DATABASE = "database"
    SCHEDULER = "scheduler"
    ANALYTICS = "analytics"


class SystemHealthStatus(Enum):
    """Enumeration of system health statuses."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class SystemHealth(Base):
    """System health model for monitoring all subsystem statuses."""

    __tablename__ = "msg_system_health"

    id = Column(BigInteger, primary_key=True, index=True)
    system = Column(String(50), nullable=False, index=True)
    component = Column(String(100), nullable=True)  # For specific components within a system
    status = Column(String(20), nullable=False, index=True)
    last_success = Column(DateTime(timezone=True), nullable=True)
    last_failure = Column(DateTime(timezone=True), nullable=True)
    failure_count = Column(Integer, nullable=False, default=0)
    avg_response_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    system_metadata = Column("metadata", Text, nullable=True)  # JSON string for system-specific data
    checked_at = Column(DateTime(timezone=True), nullable=False, default=func.now(), index=True)

    # Constraints
    __table_args__ = (
        CheckConstraint(
            f"status IN ('{SystemHealthStatus.HEALTHY.value}', "
            f"'{SystemHealthStatus.DEGRADED.value}', "
            f"'{SystemHealthStatus.DOWN.value}', "
            f"'{SystemHealthStatus.UNKNOWN.value}')",
            name="check_system_health_status"
        ),
        CheckConstraint("failure_count >= 0", name="check_failure_count_positive"),
        CheckConstraint("avg_response_time_ms >= 0", name="check_avg_response_time_positive"),
        # Unique constraint on system + component combination
        CheckConstraint(
            "(system, COALESCE(component, '')) IS NOT NULL",
            name="check_system_component_unique"
        ),
    )

    def __repr__(self):
        component_str = f", component='{self.component}'" if self.component else ""
        return (f"<SystemHealth(id={self.id}, system='{self.system}'"
                f"{component_str}, status='{self.status}')>")

    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == SystemHealthStatus.HEALTHY.value

    @property
    def is_degraded(self) -> bool:
        """Check if system is degraded."""
        return self.status == SystemHealthStatus.DEGRADED.value

    @property
    def is_down(self) -> bool:
        """Check if system is down."""
        return self.status == SystemHealthStatus.DOWN.value

    @property
    def system_identifier(self) -> str:
        """Get unique system identifier."""
        if self.component:
            return f"{self.system}.{self.component}"
        return self.system

    def update_health_status(
        self,
        status: SystemHealthStatus,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[str] = None
    ) -> None:
        """
        Update the health status of the system.

        Args:
            status: New health status
            response_time_ms: Response time in milliseconds
            error_message: Error message if status is not healthy
            metadata: Additional system-specific metadata as JSON string
        """
        self.status = status.value
        self.checked_at = datetime.now(timezone.utc)

        if status == SystemHealthStatus.HEALTHY:
            self.last_success = self.checked_at
            self.failure_count = 0
            self.error_message = None
        else:
            self.last_failure = self.checked_at
            self.failure_count += 1
            if error_message:
                self.error_message = error_message

        if response_time_ms is not None:
            # Update average response time (simple moving average)
            if self.avg_response_time_ms is None:
                self.avg_response_time_ms = response_time_ms
            else:
                # Weighted average favoring recent measurements
                self.avg_response_time_ms = int(
                    (self.avg_response_time_ms * 0.7) + (response_time_ms * 0.3)
                )

        if metadata:
            self.metadata = metadata

    @classmethod
    def get_system_status(cls, session, system: str, component: Optional[str] = None):
        """
        Get the current health status for a specific system/component.

        Args:
            session: Database session
            system: System name
            component: Optional component name

        Returns:
            SystemHealth instance or None if not found
        """
        query = session.query(cls).filter(cls.system == system)
        if component:
            query = query.filter(cls.component == component)
        else:
            query = query.filter(cls.component.is_(None))

        return query.first()

    @classmethod
    def get_all_systems_status(cls, session):
        """
        Get health status for all monitored systems.

        Args:
            session: Database session

        Returns:
            List of SystemHealth instances
        """
        return session.query(cls).order_by(cls.system, cls.component).all()

    @classmethod
    def get_unhealthy_systems(cls, session):
        """
        Get all systems that are not healthy.

        Args:
            session: Database session

        Returns:
            List of SystemHealth instances with non-healthy status
        """
        return session.query(cls).filter(
            cls.status != SystemHealthStatus.HEALTHY.value
        ).order_by(cls.system, cls.component).all()