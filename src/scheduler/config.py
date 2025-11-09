"""
Scheduler Service Configuration

Configuration management for the scheduler service with environment variable support.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from src.data.db.core.database import get_database_url
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration."""
    url: str = field(default_factory=get_database_url)
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class APSchedulerConfig:
    """APScheduler configuration."""
    max_workers: int = 10
    job_timeout: int = 300  # 5 minutes
    coalesce: bool = False
    max_instances: int = 1
    timezone: str = "UTC"


@dataclass
class NotificationConfig:
    """Notification service configuration."""
    service_url: str = "http://localhost:8000"
    timeout: int = 30
    max_retries: int = 3
    enabled: bool = True


@dataclass
class DataConfig:
    """Data service configuration."""
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    default_lookback: int = 200
    max_retries: int = 3
    timeout: int = 60


@dataclass
class AlertConfig:
    """Alert evaluation configuration."""
    schema_dir: str = "src/common/alerts/schemas"
    default_lookback: int = 200
    max_evaluation_time: int = 120  # 2 minutes
    enable_once_per_bar: bool = True


@dataclass
class ServiceConfig:
    """General service configuration."""
    name: str = "scheduler-service"
    version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    health_check_interval: int = 60  # 1 minute


@dataclass
class SchedulerServiceConfig:
    """Complete scheduler service configuration."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    scheduler: APSchedulerConfig = field(default_factory=APSchedulerConfig)
    notification: NotificationConfig = field(default_factory=NotificationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    def __post_init__(self):
        """Load configuration from environment variables."""
        self._load_from_environment()
        self._validate_configuration()
        _logger.info("Scheduler service configuration loaded for environment: %s",
                    self.service.environment)

    def _load_from_environment(self) -> None:
        """Load configuration values from environment variables."""
        # Database configuration
        if os.getenv("DATABASE_URL"):
            self.database.url = os.getenv("DATABASE_URL")
        self.database.pool_size = int(os.getenv("DB_POOL_SIZE", self.database.pool_size))
        self.database.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", self.database.max_overflow))
        self.database.echo = os.getenv("SQL_ECHO", "false").lower() == "true"

        # APScheduler configuration
        self.scheduler.max_workers = int(os.getenv("SCHEDULER_MAX_WORKERS", self.scheduler.max_workers))
        self.scheduler.job_timeout = int(os.getenv("SCHEDULER_JOB_TIMEOUT", self.scheduler.job_timeout))
        self.scheduler.timezone = os.getenv("SCHEDULER_TIMEZONE", self.scheduler.timezone)

        # Notification configuration
        self.notification.service_url = os.getenv("NOTIFICATION_SERVICE_URL", self.notification.service_url)
        self.notification.timeout = int(os.getenv("NOTIFICATION_TIMEOUT", self.notification.timeout))
        self.notification.max_retries = int(os.getenv("NOTIFICATION_RETRIES", self.notification.max_retries))
        self.notification.enabled = os.getenv("NOTIFICATION_ENABLED", "true").lower() == "true"

        # Data configuration
        self.data.cache_enabled = os.getenv("DATA_CACHE_ENABLED", "true").lower() == "true"
        self.data.cache_ttl = int(os.getenv("DATA_CACHE_TTL", self.data.cache_ttl))
        self.data.default_lookback = int(os.getenv("DATA_DEFAULT_LOOKBACK", self.data.default_lookback))
        self.data.max_retries = int(os.getenv("DATA_MAX_RETRIES", self.data.max_retries))
        self.data.timeout = int(os.getenv("DATA_TIMEOUT", self.data.timeout))

        # Alert configuration
        self.alert.schema_dir = os.getenv("ALERT_SCHEMA_DIR", self.alert.schema_dir)
        self.alert.default_lookback = int(os.getenv("ALERT_DEFAULT_LOOKBACK", self.alert.default_lookback))
        self.alert.max_evaluation_time = int(os.getenv("ALERT_MAX_EVAL_TIME", self.alert.max_evaluation_time))
        self.alert.enable_once_per_bar = os.getenv("ALERT_ONCE_PER_BAR", "true").lower() == "true"

        # Service configuration
        self.service.environment = os.getenv("TRADING_ENV", self.service.environment)
        self.service.log_level = os.getenv("LOG_LEVEL", self.service.log_level)
        self.service.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", self.service.health_check_interval))

    def _validate_configuration(self) -> None:
        """Validate configuration values."""
        # Validate database URL
        if not self.database.url:
            raise ValueError("Database URL is required")

        # Validate positive integers
        if self.scheduler.max_workers <= 0:
            raise ValueError("Scheduler max_workers must be positive")

        if self.scheduler.job_timeout <= 0:
            raise ValueError("Scheduler job_timeout must be positive")

        if self.notification.timeout <= 0:
            raise ValueError("Notification timeout must be positive")

        if self.data.cache_ttl <= 0:
            raise ValueError("Data cache TTL must be positive")

        # Validate schema directory exists
        schema_path = Path(self.alert.schema_dir)
        if not schema_path.exists():
            _logger.warning("Alert schema directory does not exist: %s", schema_path)

        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if self.service.environment not in valid_environments:
            _logger.warning("Unknown environment: %s. Valid options: %s",
                          self.service.environment, valid_environments)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "database": {
                "url": self.database.url.split('@')[1] if '@' in self.database.url else "local",
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "echo": self.database.echo
            },
            "scheduler": {
                "max_workers": self.scheduler.max_workers,
                "job_timeout": self.scheduler.job_timeout,
                "coalesce": self.scheduler.coalesce,
                "max_instances": self.scheduler.max_instances,
                "timezone": self.scheduler.timezone
            },
            "notification": {
                "service_url": self.notification.service_url,
                "timeout": self.notification.timeout,
                "max_retries": self.notification.max_retries,
                "enabled": self.notification.enabled
            },
            "data": {
                "cache_enabled": self.data.cache_enabled,
                "cache_ttl": self.data.cache_ttl,
                "default_lookback": self.data.default_lookback,
                "max_retries": self.data.max_retries,
                "timeout": self.data.timeout
            },
            "alert": {
                "schema_dir": self.alert.schema_dir,
                "default_lookback": self.alert.default_lookback,
                "max_evaluation_time": self.alert.max_evaluation_time,
                "enable_once_per_bar": self.alert.enable_once_per_bar
            },
            "service": {
                "name": self.service.name,
                "version": self.service.version,
                "environment": self.service.environment,
                "log_level": self.service.log_level,
                "health_check_interval": self.service.health_check_interval
            }
        }


# Global configuration instance
config = SchedulerServiceConfig()