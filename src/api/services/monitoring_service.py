"""
System Monitoring Service
------------------------

Service for monitoring system resources, trading service health,
and generating alerts for the web UI.
"""

import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class SystemAlert:
    """Represents a system alert."""

    def __init__(self, alert_type: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.alert_type = alert_type
        self.severity = severity  # info, warning, error, critical
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)
        self.acknowledged = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


class SystemMonitoringService:
    """
    Service for monitoring system resources and health.

    Provides:
    - CPU, memory, disk, and temperature monitoring
    - Trading service health checks
    - Alert generation and management
    - Performance data collection
    """

    def __init__(self):
        """Initialize the monitoring service."""
        self.start_time = datetime.now(timezone.utc)
        self.alerts: List[SystemAlert] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.max_alerts = 100

        # Alert thresholds
        self.thresholds = {
            'cpu_warning': 80.0,
            'cpu_critical': 95.0,
            'memory_warning': 85.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'temperature_warning': 70.0,
            'temperature_critical': 80.0
        }

        _logger.info("System monitoring service initialized")

    def get_cpu_metrics(self) -> Dict[str, Any]:
        """
        Get CPU usage metrics.

        Returns:
            Dict: CPU metrics including usage percentage and core count
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Get per-core usage
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

            return {
                'usage_percent': round(cpu_percent, 2),
                'core_count': cpu_count,
                'per_core_usage': [round(usage, 2) for usage in cpu_per_core],
                'frequency_mhz': round(cpu_freq.current, 2) if cpu_freq else None,
                'max_frequency_mhz': round(cpu_freq.max, 2) if cpu_freq else None
            }

        except Exception:
            _logger.exception("Failed to get CPU metrics:")
            return {
                'usage_percent': 0.0,
                'core_count': 0,
                'per_core_usage': [],
                'frequency_mhz': None,
                'max_frequency_mhz': None
            }

    def get_memory_metrics(self) -> Dict[str, Any]:
        """
        Get memory usage metrics.

        Returns:
            Dict: Memory metrics including usage, available, and swap
        """
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'usage_percent': round(memory.percent, 2),
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2),
                'swap_usage_percent': round(swap.percent, 2)
            }

        except Exception:
            _logger.exception("Failed to get memory metrics:")
            return {
                'total_gb': 0.0,
                'available_gb': 0.0,
                'used_gb': 0.0,
                'usage_percent': 0.0,
                'swap_total_gb': 0.0,
                'swap_used_gb': 0.0,
                'swap_usage_percent': 0.0
            }

    def get_disk_metrics(self) -> Dict[str, Any]:
        """
        Get disk usage metrics.

        Returns:
            Dict: Disk metrics for all mounted drives
        """
        try:
            disk_usage = {}

            # Get disk usage for all mounted drives
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.device] = {
                        'mountpoint': partition.mountpoint,
                        'filesystem': partition.fstype,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'usage_percent': round((usage.used / usage.total) * 100, 2)
                    }
                except (PermissionError, OSError):
                    # Skip drives that can't be accessed
                    continue

            # Get overall disk I/O stats
            disk_io = psutil.disk_io_counters()
            io_stats = {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0,
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0
            }

            return {
                'partitions': disk_usage,
                'io_stats': io_stats
            }

        except Exception:
            _logger.exception("Failed to get disk metrics:")
            return {
                'partitions': {},
                'io_stats': {
                    'read_bytes': 0,
                    'write_bytes': 0,
                    'read_count': 0,
                    'write_count': 0
                }
            }

    def get_temperature_metrics(self) -> Dict[str, Any]:
        """
        Get system temperature metrics.

        Returns:
            Dict: Temperature metrics for available sensors
        """
        try:
            temperatures = {}

            # Try to get temperature sensors
            if hasattr(psutil, 'sensors_temperatures'):
                temp_sensors = psutil.sensors_temperatures()

                for sensor_name, sensor_list in temp_sensors.items():
                    temperatures[sensor_name] = []
                    for sensor in sensor_list:
                        temperatures[sensor_name].append({
                            'label': sensor.label or 'Unknown',
                            'current': round(sensor.current, 1),
                            'high': round(sensor.high, 1) if sensor.high else None,
                            'critical': round(sensor.critical, 1) if sensor.critical else None
                        })

            # Calculate average temperature if available
            avg_temp = None
            if temperatures:
                all_temps = []
                for sensor_list in temperatures.values():
                    for sensor in sensor_list:
                        all_temps.append(sensor['current'])

                if all_temps:
                    avg_temp = round(sum(all_temps) / len(all_temps), 1)

            return {
                'sensors': temperatures,
                'average_celsius': avg_temp
            }

        except Exception:
            _logger.exception("Failed to get temperature metrics:")
            return {
                'sensors': {},
                'average_celsius': None
            }

    def get_network_metrics(self) -> Dict[str, Any]:
        """
        Get network usage metrics.

        Returns:
            Dict: Network metrics including bytes sent/received
        """
        try:
            net_io = psutil.net_io_counters()

            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errors_in': net_io.errin,
                'errors_out': net_io.errout,
                'drops_in': net_io.dropin,
                'drops_out': net_io.dropout
            }

        except Exception:
            _logger.exception("Failed to get network metrics:")
            return {
                'bytes_sent': 0,
                'bytes_recv': 0,
                'packets_sent': 0,
                'packets_recv': 0,
                'errors_in': 0,
                'errors_out': 0,
                'drops_in': 0,
                'drops_out': 0
            }

    def get_process_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the current process and related processes.

        Returns:
            Dict: Process metrics including CPU and memory usage
        """
        try:
            current_process = psutil.Process()

            # Get current process metrics
            process_info = {
                'pid': current_process.pid,
                'name': current_process.name(),
                'cpu_percent': round(current_process.cpu_percent(), 2),
                'memory_percent': round(current_process.memory_percent(), 2),
                'memory_mb': round(current_process.memory_info().rss / (1024**2), 2),
                'num_threads': current_process.num_threads(),
                'create_time': datetime.fromtimestamp(current_process.create_time()).isoformat()
            }

            # Look for related trading processes
            trading_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if any(keyword in proc.info['name'].lower() for keyword in ['python', 'trading', 'strategy']):
                        if proc.pid != current_process.pid:
                            trading_processes.append({
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cpu_percent': round(proc.info['cpu_percent'] or 0, 2),
                                'memory_percent': round(proc.info['memory_percent'] or 0, 2)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return {
                'current_process': process_info,
                'related_processes': trading_processes[:10]  # Limit to 10 processes
            }

        except Exception:
            _logger.exception("Failed to get process metrics:")
            return {
                'current_process': {},
                'related_processes': []
            }

    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system metrics.

        Returns:
            Dict: All system metrics combined
        """
        timestamp = datetime.now(timezone.utc)

        metrics = {
            'timestamp': timestamp.isoformat(),
            'uptime_seconds': (timestamp - self.start_time).total_seconds(),
            'cpu': self.get_cpu_metrics(),
            'memory': self.get_memory_metrics(),
            'disk': self.get_disk_metrics(),
            'temperature': self.get_temperature_metrics(),
            'network': self.get_network_metrics(),
            'processes': self.get_process_metrics()
        }

        # Store in history
        self.performance_history.append(metrics)

        # Limit history size
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]

        # Check for alerts
        self._check_thresholds(metrics)

        return metrics

    def _check_thresholds(self, metrics: Dict[str, Any]):
        """
        Check metrics against thresholds and generate alerts.

        Args:
            metrics: Current system metrics
        """
        try:
            # Check CPU usage
            cpu_usage = metrics['cpu']['usage_percent']
            if cpu_usage >= self.thresholds['cpu_critical']:
                self._add_alert('cpu', 'critical', f'CPU usage critical: {cpu_usage}%')
            elif cpu_usage >= self.thresholds['cpu_warning']:
                self._add_alert('cpu', 'warning', f'CPU usage high: {cpu_usage}%')

            # Check memory usage
            memory_usage = metrics['memory']['usage_percent']
            if memory_usage >= self.thresholds['memory_critical']:
                self._add_alert('memory', 'critical', f'Memory usage critical: {memory_usage}%')
            elif memory_usage >= self.thresholds['memory_warning']:
                self._add_alert('memory', 'warning', f'Memory usage high: {memory_usage}%')

            # Check disk usage
            for device, disk_info in metrics['disk']['partitions'].items():
                disk_usage = disk_info['usage_percent']
                if disk_usage >= self.thresholds['disk_critical']:
                    self._add_alert('disk', 'critical', f'Disk usage critical on {device}: {disk_usage}%')
                elif disk_usage >= self.thresholds['disk_warning']:
                    self._add_alert('disk', 'warning', f'Disk usage high on {device}: {disk_usage}%')

            # Check temperature
            avg_temp = metrics['temperature']['average_celsius']
            if avg_temp is not None:
                if avg_temp >= self.thresholds['temperature_critical']:
                    self._add_alert('temperature', 'critical', f'Temperature critical: {avg_temp}°C')
                elif avg_temp >= self.thresholds['temperature_warning']:
                    self._add_alert('temperature', 'warning', f'Temperature high: {avg_temp}°C')

        except Exception:
            _logger.exception("Failed to check thresholds:")

    def _add_alert(self, alert_type: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Add a new alert if it doesn't already exist.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            message: Alert message
            details: Additional alert details
        """
        # Check if similar alert already exists (within last 5 minutes)
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        for existing_alert in self.alerts:
            if (existing_alert.alert_type == alert_type and
                existing_alert.severity == severity and
                existing_alert.timestamp > cutoff_time and
                not existing_alert.acknowledged):
                return  # Don't add duplicate alert

        # Add new alert
        alert = SystemAlert(alert_type, severity, message, details)
        self.alerts.append(alert)

        # Limit alerts list size
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        _logger.warning("System alert generated: %s - %s", severity.upper(), message)

    def get_alerts(self, unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get system alerts.

        Args:
            unacknowledged_only: Return only unacknowledged alerts

        Returns:
            List: List of alert dictionaries
        """
        alerts = self.alerts

        if unacknowledged_only:
            alerts = [alert for alert in alerts if not alert.acknowledged]

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)

        return [alert.to_dict() for alert in alerts]

    def acknowledge_alert(self, alert_index: int) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_index: Index of alert to acknowledge

        Returns:
            bool: True if alert was acknowledged
        """
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].acknowledged = True
                return True
            return False
        except Exception:
            _logger.exception("Failed to acknowledge alert:")
            return False

    def clear_acknowledged_alerts(self):
        """Clear all acknowledged alerts."""
        self.alerts = [alert for alert in self.alerts if not alert.acknowledged]

    def get_performance_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """
        Get performance history for the specified time period.

        Args:
            hours: Number of hours of history to return

        Returns:
            List: Performance history data
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            metrics for metrics in self.performance_history
            if datetime.fromisoformat(metrics['timestamp']) > cutoff_time
        ]

    def get_service_status(self) -> Dict[str, Any]:
        """
        Get monitoring service status.

        Returns:
            Dict: Service status information
        """
        return {
            'service_name': 'System Monitoring Service',
            'status': 'running',
            'uptime_seconds': (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            'total_alerts': len(self.alerts),
            'unacknowledged_alerts': len([a for a in self.alerts if not a.acknowledged]),
            'performance_history_size': len(self.performance_history),
            'thresholds': self.thresholds
        }