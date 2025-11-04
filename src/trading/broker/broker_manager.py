#!/usr/bin/env python3
"""
Broker Management System
-----------------------

This module provides comprehensive broker management capabilities including
broker health monitoring, connection pooling, and lifecycle management.

Features:
- Broker health monitoring and status reporting
- Connection pooling and management
- Broker lifecycle management (start, stop, restart)
- Performance monitoring and analytics
- Error handling and recovery
- Configuration hot-reloading

Classes:
- BrokerManager: Main broker management class
- BrokerHealthMonitor: Health monitoring and status reporting
- BrokerConnectionPool: Connection pooling and management
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

from src.trading.broker.broker_factory import get_broker, BrokerConfigurationError
from src.trading.broker.base_broker import BaseBroker

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class BrokerStatus(Enum):
    """Broker status enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class BrokerHealthMetrics:
    """Broker health metrics."""
    broker_id: str
    broker_type: str
    trading_mode: str
    status: BrokerStatus
    uptime_seconds: float
    connection_status: bool
    last_heartbeat: datetime
    error_count: int = 0
    reconnect_count: int = 0
    orders_processed: int = 0
    positions_count: int = 0
    portfolio_value: float = 0.0
    last_error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'broker_id': self.broker_id,
            'broker_type': self.broker_type,
            'trading_mode': self.trading_mode,
            'status': self.status.value,
            'uptime_seconds': self.uptime_seconds,
            'connection_status': self.connection_status,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'error_count': self.error_count,
            'reconnect_count': self.reconnect_count,
            'orders_processed': self.orders_processed,
            'positions_count': self.positions_count,
            'portfolio_value': self.portfolio_value,
            'last_error': self.last_error,
            'performance_metrics': self.performance_metrics
        }


class BrokerHealthMonitor:
    """Health monitoring system for brokers."""

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_metrics: Dict[str, BrokerHealthMetrics] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.health_callbacks: List[Callable[[str, BrokerHealthMetrics], None]] = []

    def add_broker(self, broker_id: str, broker: BaseBroker):
        """Add a broker to health monitoring."""
        self.health_metrics[broker_id] = BrokerHealthMetrics(
            broker_id=broker_id,
            broker_type=broker.config.get('type', 'unknown'),
            trading_mode=broker.trading_mode.value,
            status=BrokerStatus.STOPPED,
            uptime_seconds=0.0,
            connection_status=False,
            last_heartbeat=datetime.now(timezone.utc)
        )
        _logger.info("Added broker %s to health monitoring", broker_id)

    def remove_broker(self, broker_id: str):
        """Remove a broker from health monitoring."""
        if broker_id in self.health_metrics:
            del self.health_metrics[broker_id]
            _logger.info("Removed broker %s from health monitoring", broker_id)

    def update_broker_status(self, broker_id: str, status: BrokerStatus,
                           error_message: Optional[str] = None):
        """Update broker status."""
        if broker_id in self.health_metrics:
            metrics = self.health_metrics[broker_id]
            metrics.status = status
            metrics.last_heartbeat = datetime.now(timezone.utc)

            if error_message:
                metrics.last_error = error_message
                metrics.error_count += 1

            # Notify callbacks
            for callback in self.health_callbacks:
                try:
                    callback(broker_id, metrics)
                except Exception as e:
                    _logger.exception("Error in health callback:")

    def add_health_callback(self, callback: Callable[[str, BrokerHealthMetrics], None]):
        """Add a health status callback."""
        self.health_callbacks.append(callback)

    def start_monitoring(self):
        """Start health monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        _logger.info("Started broker health monitoring")

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        _logger.info("Stopped broker health monitoring")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                for broker_id, metrics in self.health_metrics.items():
                    # Check if broker is responsive
                    time_since_heartbeat = (datetime.now(timezone.utc) - metrics.last_heartbeat).total_seconds()

                    if time_since_heartbeat > self.check_interval * 2:
                        # Broker appears unresponsive
                        if metrics.status == BrokerStatus.RUNNING:
                            self.update_broker_status(broker_id, BrokerStatus.ERROR,
                                                    "Broker unresponsive - no heartbeat")

                time.sleep(self.check_interval)

            except Exception as e:
                _logger.exception("Error in health monitoring loop:")
                time.sleep(5)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        total_brokers = len(self.health_metrics)
        running_brokers = sum(1 for m in self.health_metrics.values() if m.status == BrokerStatus.RUNNING)
        error_brokers = sum(1 for m in self.health_metrics.values() if m.status == BrokerStatus.ERROR)

        return {
            'total_brokers': total_brokers,
            'running_brokers': running_brokers,
            'error_brokers': error_brokers,
            'health_percentage': (running_brokers / max(total_brokers, 1)) * 100,
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'broker_details': {broker_id: metrics.to_dict() for broker_id, metrics in self.health_metrics.items()}
        }


class BrokerConnectionPool:
    """Connection pool manager for brokers."""

    def __init__(self, max_connections_per_type: int = 5):
        self.max_connections_per_type = max_connections_per_type
        self.connections: Dict[str, List[BaseBroker]] = {}
        self.connection_usage: Dict[str, int] = {}
        self.pool_lock = threading.Lock()

    def get_connection(self, broker_config: Dict[str, Any]) -> BaseBroker:
        """Get a broker connection from the pool or create a new one."""
        broker_type = broker_config.get('type', 'unknown')
        trading_mode = broker_config.get('trading_mode', 'paper')
        pool_key = f"{broker_type}_{trading_mode}"

        with self.pool_lock:
            # Check if we have available connections
            if pool_key in self.connections and self.connections[pool_key]:
                broker = self.connections[pool_key].pop()
                self.connection_usage[pool_key] = self.connection_usage.get(pool_key, 0) + 1
                _logger.debug("Reused broker connection from pool: %s", pool_key)
                return broker

            # Create new connection
            try:
                broker = get_broker(broker_config)
                self.connection_usage[pool_key] = self.connection_usage.get(pool_key, 0) + 1
                _logger.debug("Created new broker connection: %s", pool_key)
                return broker

            except Exception as e:
                _logger.exception("Failed to create broker connection:")
                raise

    def return_connection(self, broker: BaseBroker):
        """Return a broker connection to the pool."""
        broker_type = broker.config.get('type', 'unknown')
        trading_mode = broker.trading_mode.value
        pool_key = f"{broker_type}_{trading_mode}"

        with self.pool_lock:
            if pool_key not in self.connections:
                self.connections[pool_key] = []

            # Only keep connection if pool isn't full
            if len(self.connections[pool_key]) < self.max_connections_per_type:
                self.connections[pool_key].append(broker)
                _logger.debug("Returned broker connection to pool: %s", pool_key)
            else:
                # Pool is full, disconnect the broker
                asyncio.create_task(broker.disconnect())
                _logger.debug("Pool full, disconnected broker: %s", pool_key)

    async def cleanup_pool(self):
        """Clean up all connections in the pool."""
        with self.pool_lock:
            for pool_key, brokers in self.connections.items():
                for broker in brokers:
                    try:
                        await broker.disconnect()
                    except Exception as e:
                        _logger.exception("Error disconnecting broker in cleanup:")
                brokers.clear()

            self.connections.clear()
            self.connection_usage.clear()
            _logger.info("Cleaned up broker connection pool")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self.pool_lock:
            return {
                'pool_sizes': {key: len(brokers) for key, brokers in self.connections.items()},
                'connection_usage': self.connection_usage.copy(),
                'max_connections_per_type': self.max_connections_per_type,
                'total_active_connections': sum(len(brokers) for brokers in self.connections.values())
            }


class BrokerManager:
    """
    Comprehensive broker management system.

    Features:
    - Broker lifecycle management
    - Health monitoring and status reporting
    - Connection pooling and management
    - Configuration hot-reloading
    - Performance monitoring
    - Error handling and recovery
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.brokers: Dict[str, BaseBroker] = {}
        self.broker_configs: Dict[str, Dict[str, Any]] = {}
        self.broker_start_times: Dict[str, datetime] = {}

        # Initialize components
        self.health_monitor = BrokerHealthMonitor(
            check_interval=self.config.get('health_check_interval', 30)
        )
        self.connection_pool = BrokerConnectionPool(
            max_connections_per_type=self.config.get('max_connections_per_type', 5)
        )

        # Management state
        self.manager_active = False
        self.auto_restart_enabled = self.config.get('auto_restart_enabled', True)

        # Setup health monitoring callback
        self.health_monitor.add_health_callback(self._on_broker_health_change)

        _logger.info("Broker manager initialized")

    async def add_broker(self, broker_id: str, broker_config: Dict[str, Any]) -> bool:
        """
        Add a broker to management.

        Args:
            broker_id: Unique identifier for the broker
            broker_config: Broker configuration dictionary

        Returns:
            True if broker was added successfully
        """
        try:
            # Validate configuration
            from src.trading.broker.config_validator import validate_and_create_broker_config
            validated_config = validate_and_create_broker_config(broker_config)

            # Create broker
            broker = get_broker(validated_config)

            # Store broker and config
            self.brokers[broker_id] = broker
            self.broker_configs[broker_id] = validated_config
            self.broker_start_times[broker_id] = datetime.now(timezone.utc)

            # Add to health monitoring
            self.health_monitor.add_broker(broker_id, broker)

            _logger.info("Added broker %s (%s)", broker_id, broker_config.get('type', 'unknown'))
            return True

        except Exception as e:
            _logger.exception("Failed to add broker %s:", broker_id)
            return False

    async def remove_broker(self, broker_id: str) -> bool:
        """
        Remove a broker from management.

        Args:
            broker_id: Broker identifier

        Returns:
            True if broker was removed successfully
        """
        try:
            if broker_id in self.brokers:
                # Stop broker if running
                await self.stop_broker(broker_id)

                # Remove from management
                del self.brokers[broker_id]
                del self.broker_configs[broker_id]
                if broker_id in self.broker_start_times:
                    del self.broker_start_times[broker_id]

                # Remove from health monitoring
                self.health_monitor.remove_broker(broker_id)

                _logger.info("Removed broker %s", broker_id)
                return True

            return False

        except Exception as e:
            _logger.exception("Failed to remove broker %s:", broker_id)
            return False

    async def shutdown(self) -> None:
        """
        Shutdown all managed brokers gracefully.

        This method stops all brokers and cleans up resources.
        Called during service shutdown.
        """
        try:
            _logger.info("Shutting down Broker Manager...")

            # Stop all brokers
            broker_ids = list(self.brokers.keys())
            for broker_id in broker_ids:
                try:
                    await self.stop_broker(broker_id)
                except Exception as e:
                    _logger.warning("Error stopping broker %s during shutdown: %s", broker_id, e)

            # Stop health monitoring
            self.health_monitor.stop_monitoring()

            # Clear broker registry
            self.brokers.clear()
            self.broker_configs.clear()
            self.broker_start_times.clear()

            self.manager_active = False

            _logger.info("Broker Manager shutdown complete")

        except Exception as e:
            _logger.exception("Error during Broker Manager shutdown:")

    async def start_broker(self, broker_id: str) -> bool:
        """
        Start a broker.

        Args:
            broker_id: Broker identifier

        Returns:
            True if broker started successfully
        """
        if broker_id not in self.brokers:
            _logger.error("Broker %s not found", broker_id)
            return False

        try:
            broker = self.brokers[broker_id]

            # Update status
            self.health_monitor.update_broker_status(broker_id, BrokerStatus.STARTING)

            # Connect broker
            connected = await broker.connect()

            if connected:
                self.health_monitor.update_broker_status(broker_id, BrokerStatus.RUNNING)
                _logger.info("Started broker %s", broker_id)
                return True
            else:
                self.health_monitor.update_broker_status(broker_id, BrokerStatus.ERROR, "Failed to connect")
                return False

        except Exception as e:
            error_msg = f"Error starting broker {broker_id}: {str(e)}"
            _logger.exception(error_msg)
            self.health_monitor.update_broker_status(broker_id, BrokerStatus.ERROR, error_msg)
            return False

    async def stop_broker(self, broker_id: str) -> bool:
        """
        Stop a broker.

        Args:
            broker_id: Broker identifier

        Returns:
            True if broker stopped successfully
        """
        if broker_id not in self.brokers:
            _logger.error("Broker %s not found", broker_id)
            return False

        try:
            broker = self.brokers[broker_id]

            # Update status
            self.health_monitor.update_broker_status(broker_id, BrokerStatus.STOPPING)

            # Disconnect broker
            disconnected = await broker.disconnect()

            if disconnected:
                self.health_monitor.update_broker_status(broker_id, BrokerStatus.STOPPED)
                _logger.info("Stopped broker %s", broker_id)
                return True
            else:
                self.health_monitor.update_broker_status(broker_id, BrokerStatus.ERROR,
                                                       "Failed to disconnect")
                return False

        except Exception as e:
            error_msg = f"Error stopping broker {broker_id}: {str(e)}"
            _logger.exception(error_msg)
            self.health_monitor.update_broker_status(broker_id, BrokerStatus.ERROR, error_msg)
            return False

    async def restart_broker(self, broker_id: str) -> bool:
        """
        Restart a broker.

        Args:
            broker_id: Broker identifier

        Returns:
            True if broker restarted successfully
        """
        _logger.info("Restarting broker %s", broker_id)

        # Stop broker
        stopped = await self.stop_broker(broker_id)
        if not stopped:
            return False

        # Wait a moment
        await asyncio.sleep(2)

        # Start broker
        return await self.start_broker(broker_id)

    async def start_all_brokers(self) -> Dict[str, bool]:
        """
        Start all managed brokers.

        Returns:
            Dictionary with broker_id -> success status
        """
        results = {}

        for broker_id in self.brokers.keys():
            results[broker_id] = await self.start_broker(broker_id)

        return results

    async def stop_all_brokers(self) -> Dict[str, bool]:
        """
        Stop all managed brokers.

        Returns:
            Dictionary with broker_id -> success status
        """
        results = {}

        for broker_id in self.brokers.keys():
            results[broker_id] = await self.stop_broker(broker_id)

        return results

    def get_broker(self, broker_id: str) -> Optional[BaseBroker]:
        """Get a managed broker by ID."""
        return self.brokers.get(broker_id)

    def list_brokers(self) -> List[Dict[str, Any]]:
        """List all managed brokers with their status."""
        brokers = []

        for broker_id, broker in self.brokers.items():
            config = self.broker_configs.get(broker_id, {})
            metrics = self.health_monitor.health_metrics.get(broker_id)

            broker_info = {
                'broker_id': broker_id,
                'broker_type': config.get('type', 'unknown'),
                'trading_mode': config.get('trading_mode', 'unknown'),
                'status': metrics.status.value if metrics else 'unknown',
                'uptime_seconds': metrics.uptime_seconds if metrics else 0,
                'connection_status': metrics.connection_status if metrics else False,
                'error_count': metrics.error_count if metrics else 0
            }

            brokers.append(broker_info)

        return brokers

    def start_management(self):
        """Start the broker management system."""
        if self.manager_active:
            return

        self.manager_active = True
        self.health_monitor.start_monitoring()
        _logger.info("Started broker management system")

    def stop_management(self):
        """Stop the broker management system."""
        if not self.manager_active:
            return

        self.manager_active = False
        self.health_monitor.stop_monitoring()
        _logger.info("Stopped broker management system")

    async def cleanup(self):
        """Clean up all resources."""
        # Stop all brokers
        await self.stop_all_brokers()

        # Stop management
        self.stop_management()

        # Clean up connection pool
        await self.connection_pool.cleanup_pool()

        _logger.info("Broker manager cleanup completed")

    def _on_broker_health_change(self, broker_id: str, metrics: BrokerHealthMetrics):
        """Handle broker health status changes."""
        if not self.auto_restart_enabled:
            return

        # Auto-restart on error (with limits)
        if metrics.status == BrokerStatus.ERROR and metrics.error_count <= 3:
            _logger.warning("Broker %s in error state, attempting restart", broker_id)
            asyncio.create_task(self.restart_broker(broker_id))

    def get_management_status(self) -> Dict[str, Any]:
        """Get comprehensive management status."""
        health_summary = self.health_monitor.get_health_summary()
        pool_stats = self.connection_pool.get_pool_stats()

        return {
            'manager_active': self.manager_active,
            'auto_restart_enabled': self.auto_restart_enabled,
            'total_managed_brokers': len(self.brokers),
            'health_summary': health_summary,
            'connection_pool_stats': pool_stats,
            'uptime': (datetime.now(timezone.utc) - min(self.broker_start_times.values())).total_seconds() if self.broker_start_times else 0
        }