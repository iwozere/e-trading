"""
Heartbeat Manager

Centralized heartbeat management for all subsystems.
Each process can use this to regularly update their health status.
"""

import asyncio
import time
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.db.services.database_service import DatabaseService
from src.data.db.repos.repo_system_health import SystemHealthRepository
from src.data.db.services.system_health_service import SystemHealthService
from src.data.db.models.model_system_health import SystemHealthStatus
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class HeartbeatManager:
    """
    Manages heartbeat updates for a specific system/component.

    Each process should create one instance and call start_heartbeat()
    to begin regular health status updates.
    """

    def __init__(
        self,
        system: str,
        component: Optional[str] = None,
        interval_seconds: int = 30,
        db_service: Optional[DatabaseService] = None
    ):
        """
        Initialize heartbeat manager.

        Args:
            system: System name (e.g., 'telegram_bot', 'api_service', 'web_ui')
            component: Component name (optional, e.g., 'email' for notification system)
            interval_seconds: Heartbeat interval in seconds
            db_service: Database service instance (optional)
        """
        self.system = system
        self.component = component
        self.interval_seconds = interval_seconds
        self.db_service = db_service or DatabaseService()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._health_check_func: Optional[Callable] = None
        self._last_heartbeat = None

        _logger.info(
            "Initialized heartbeat manager for %s.%s (interval: %ds)",
            system, component or 'main', interval_seconds
        )

    def set_health_check_function(self, health_check_func: Callable[[], Dict[str, Any]]):
        """
        Set a custom health check function.

        The function should return a dictionary with:
        - status: 'HEALTHY', 'DEGRADED', 'DOWN', or 'UNKNOWN'
        - response_time_ms: Optional response time in milliseconds
        - error_message: Optional error message
        - metadata: Optional metadata dictionary

        Args:
            health_check_func: Function that returns health status data
        """
        self._health_check_func = health_check_func
        _logger.debug("Set custom health check function for %s.%s", self.system, self.component or 'main')

    def start_heartbeat(self):
        """Start the heartbeat thread."""
        if self._running:
            _logger.warning("Heartbeat already running for %s.%s", self.system, self.component or 'main')
            return

        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

        _logger.info("Started heartbeat for %s.%s", self.system, self.component or 'main')

    def stop_heartbeat(self):
        """Stop the heartbeat thread."""
        if not self._running:
            return

        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        _logger.info("Stopped heartbeat for %s.%s", self.system, self.component or 'main')

    def send_immediate_heartbeat(
        self,
        status: SystemHealthStatus = SystemHealthStatus.HEALTHY,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Send an immediate heartbeat update.

        Args:
            status: Health status
            response_time_ms: Response time in milliseconds
            error_message: Error message if status is not healthy
            metadata: Additional metadata
        """
        try:
            self._update_health_status(status, response_time_ms, error_message, metadata)
            _logger.debug("Sent immediate heartbeat for %s.%s: %s",
                         self.system, self.component or 'main', status.value)
        except Exception as e:
            _logger.exception("Failed to send immediate heartbeat for %s.%s:",
                         self.system, self.component or 'main')

    def _heartbeat_loop(self):
        """Main heartbeat loop running in a separate thread."""
        while self._running:
            try:
                # Perform health check
                if self._health_check_func:
                    # Use custom health check function
                    health_data = self._health_check_func()
                    status_str = health_data.get('status', 'HEALTHY').upper()

                    try:
                        status = SystemHealthStatus(status_str)
                    except ValueError:
                        _logger.warning("Invalid status '%s' from health check function, using UNKNOWN", status_str)
                        status = SystemHealthStatus.UNKNOWN

                    self._update_health_status(
                        status=status,
                        response_time_ms=health_data.get('response_time_ms'),
                        error_message=health_data.get('error_message'),
                        metadata=health_data.get('metadata')
                    )
                else:
                    # Default: just send HEALTHY status
                    self._update_health_status(SystemHealthStatus.HEALTHY)

                self._last_heartbeat = datetime.now(timezone.utc)

            except Exception as e:
                _logger.exception("Error in heartbeat loop for %s.%s:",
                             self.system, self.component or 'main')

                # Send DOWN status on error
                try:
                    self._update_health_status(
                        SystemHealthStatus.DOWN,
                        error_message=f"Heartbeat error: {str(e)}"
                    )
                except Exception as update_error:
                    _logger.exception("Failed to update DOWN status:")

            # Wait for next heartbeat
            time.sleep(self.interval_seconds)

    def _update_health_status(
        self,
        status: SystemHealthStatus,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update health status in the database."""
        try:
            # SystemHealthService handles its own UoW via @with_uow decorator
            health_service = SystemHealthService()

            health_service.update_system_health(
                system=self.system,
                component=self.component,
                status=status,
                response_time_ms=response_time_ms,
                error_message=error_message,
                metadata=metadata
            )

        except Exception as e:
            _logger.exception("Failed to update health status for %s.%s:",
                         self.system, self.component or 'main')
            # Don't re-raise - heartbeat should continue even if health update fails

    @property
    def is_running(self) -> bool:
        """Check if heartbeat is currently running."""
        return self._running

    @property
    def last_heartbeat(self) -> Optional[datetime]:
        """Get timestamp of last successful heartbeat."""
        return self._last_heartbeat


class ProcessHeartbeatManager:
    """
    Manages heartbeats for an entire process that may have multiple components.

    For example, the notification service might have components for each channel.
    """

    def __init__(self, system: str, db_service: Optional[DatabaseService] = None):
        """
        Initialize process heartbeat manager.

        Args:
            system: System name
            db_service: Database service instance (optional)
        """
        self.system = system
        self.db_service = db_service or DatabaseService()
        self._heartbeat_managers: Dict[str, HeartbeatManager] = {}

        _logger.info("Initialized process heartbeat manager for %s", system)

    def add_component_heartbeat(
        self,
        component: str,
        interval_seconds: int = 30,
        health_check_func: Optional[Callable] = None
    ) -> HeartbeatManager:
        """
        Add a heartbeat for a specific component.

        Args:
            component: Component name
            interval_seconds: Heartbeat interval
            health_check_func: Optional health check function

        Returns:
            HeartbeatManager instance for the component
        """
        key = f"{self.system}.{component}"

        if key in self._heartbeat_managers:
            _logger.warning("Component heartbeat already exists for %s", key)
            return self._heartbeat_managers[key]

        manager = HeartbeatManager(
            system=self.system,
            component=component,
            interval_seconds=interval_seconds,
            db_service=self.db_service
        )

        if health_check_func:
            manager.set_health_check_function(health_check_func)

        self._heartbeat_managers[key] = manager
        return manager

    def add_main_heartbeat(
        self,
        interval_seconds: int = 30,
        health_check_func: Optional[Callable] = None
    ) -> HeartbeatManager:
        """
        Add a heartbeat for the main system (no component).

        Args:
            interval_seconds: Heartbeat interval
            health_check_func: Optional health check function

        Returns:
            HeartbeatManager instance for the main system
        """
        key = self.system

        if key in self._heartbeat_managers:
            _logger.warning("Main heartbeat already exists for %s", key)
            return self._heartbeat_managers[key]

        manager = HeartbeatManager(
            system=self.system,
            component=None,
            interval_seconds=interval_seconds,
            db_service=self.db_service
        )

        if health_check_func:
            manager.set_health_check_function(health_check_func)

        self._heartbeat_managers[key] = manager
        return manager

    def start_all_heartbeats(self):
        """Start all registered heartbeats."""
        for manager in self._heartbeat_managers.values():
            manager.start_heartbeat()

        _logger.info("Started %d heartbeats for %s", len(self._heartbeat_managers), self.system)

    def stop_all_heartbeats(self):
        """Stop all registered heartbeats."""
        for manager in self._heartbeat_managers.values():
            manager.stop_heartbeat()

        _logger.info("Stopped %d heartbeats for %s", len(self._heartbeat_managers), self.system)

    def get_heartbeat_manager(self, component: Optional[str] = None) -> Optional[HeartbeatManager]:
        """
        Get heartbeat manager for a specific component or main system.

        Args:
            component: Component name (None for main system)

        Returns:
            HeartbeatManager instance or None if not found
        """
        key = f"{self.system}.{component}" if component else self.system
        return self._heartbeat_managers.get(key)


# Global registry for process heartbeat managers
_process_managers: Dict[str, ProcessHeartbeatManager] = {}


def get_process_heartbeat_manager(system: str) -> ProcessHeartbeatManager:
    """
    Get or create a process heartbeat manager for a system.

    Args:
        system: System name

    Returns:
        ProcessHeartbeatManager instance
    """
    if system not in _process_managers:
        _process_managers[system] = ProcessHeartbeatManager(system)

    return _process_managers[system]


def cleanup_all_heartbeats():
    """Stop all heartbeats (useful for shutdown)."""
    for manager in _process_managers.values():
        manager.stop_all_heartbeats()

    _process_managers.clear()
    _logger.info("Cleaned up all heartbeat managers")