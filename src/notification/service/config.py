"""
Notification Service Configuration

Pydantic-based configuration management for the notification service.
Handles environment variables, database settings, and service configuration.
"""

from typing import Dict, Any, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
import os

from config.donotshare.donotshare import SMTP_PASSWORD, SMTP_PORT, SMTP_SERVER, SMTP_USER, TELEGRAM_BOT_TOKEN
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = Field(
        default="postgresql://localhost/trading",  # Default PostgreSQL URL
        env="DATABASE_URL",
        description="Database connection URL"
    )
    pool_size: int = Field(
        default=10,
        env="DATABASE_POOL_SIZE",
        description="Database connection pool size"
    )
    max_overflow: int = Field(
        default=20,
        env="DATABASE_MAX_OVERFLOW",
        description="Maximum database connection overflow"
    )
    echo: bool = Field(
        default=False,
        env="DATABASE_ECHO",
        description="Enable SQLAlchemy query logging"
    )
    model_config = ConfigDict(env_prefix = "DB_")



class ServerConfig(BaseSettings):
    """Server configuration."""

    host: str = Field(
        default="0.0.0.0",
        env="NOTIFICATION_HOST",
        description="Server host address"
    )
    port: int = Field(
        default=8080,
        env="NOTIFICATION_PORT",
        description="Server port"
    )
    workers: int = Field(
        default=4,
        env="NOTIFICATION_WORKERS",
        description="Number of worker processes"
    )
    reload: bool = Field(
        default=False,
        env="NOTIFICATION_RELOAD",
        description="Enable auto-reload for development"
    )
    log_level: str = Field(
        default="info",
        env="NOTIFICATION_LOG_LEVEL",
        description="Logging level"
    )
    model_config = ConfigDict(env_prefix = "SERVER_")


class ProcessingConfig(BaseSettings):
    """Message processing configuration."""

    batch_size: int = Field(
        default=10,
        env="PROCESSING_BATCH_SIZE",
        description="Maximum messages per batch"
    )
    batch_timeout_seconds: int = Field(
        default=30,
        env="PROCESSING_BATCH_TIMEOUT",
        description="Maximum time to wait for batch completion"
    )
    max_workers: int = Field(
        default=20,
        env="PROCESSING_MAX_WORKERS",
        description="Maximum number of concurrent workers"
    )
    cleanup_interval_hours: int = Field(
        default=24,
        env="PROCESSING_CLEANUP_INTERVAL",
        description="Hours between cleanup operations"
    )
    retry_delay_minutes: int = Field(
        default=5,
        env="PROCESSING_RETRY_DELAY",
        description="Minutes to wait before retrying failed messages"
    )
    max_retries: int = Field(
        default=3,
        env="PROCESSING_MAX_RETRIES",
        description="Maximum retry attempts per message"
    )
    model_config = ConfigDict(env_prefix = "PROCESSING_")


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""

    default_limits: Dict[str, int] = Field(
        default_factory=lambda: {
            "telegram": 30,  # messages per minute
            "email": 10,
            "sms": 5
        },
        description="Default rate limits per channel"
    )
    token_refill_rate: int = Field(
        default=60,
        env="RATE_LIMIT_REFILL_RATE",
        description="Tokens refilled per minute"
    )
    bypass_high_priority: bool = Field(
        default=True,
        env="RATE_LIMIT_BYPASS_HIGH_PRIORITY",
        description="Allow high priority messages to bypass rate limits"
    )
    model_config = ConfigDict(env_prefix = "RATE_LIMIT_")


class ChannelConfig(BaseSettings):
    """Channel configuration."""

    telegram: Dict[str, Any] = Field(
        default_factory=lambda: {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
            "rate_limit_per_minute": 30,
            "max_retries": 3
        },
        description="Telegram channel configuration"
    )
    email: Dict[str, Any] = Field(
        default_factory=lambda: {
            # SMTP configuration from environment variables
            "smtp_host": os.getenv("SMTP_SERVER", SMTP_SERVER),
            "smtp_port": int(os.getenv("SMTP_PORT", SMTP_PORT)),
            "smtp_username": os.getenv("SMTP_USER", SMTP_USER),
            "smtp_password": os.getenv("SMTP_PASSWORD", SMTP_PASSWORD),
            "from_email": os.getenv("SMTP_USER", SMTP_USER),
            "from_name": "Alkotrader Bot",
            "use_tls": True,
            "use_ssl": False,
            "rate_limit_per_minute": 10,
            "max_retries": 3
        },
        description="Email channel configuration"
    )
    sms: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enabled": False,
            "plugin": "sms_plugin",
            "timeout": 30,
            "rate_limit_per_minute": 5,
            "max_retries": 3
        },
        description="SMS channel configuration"
    )
    model_config = ConfigDict(env_prefix = "CHANNEL_")


class NotificationServiceConfig(BaseSettings):
    """Main notification service configuration."""

    # Service information
    service_name: str = Field(
        default="Notification Service",
        description="Service name"
    )
    version: str = Field(
        default="1.0.0",
        description="Service version"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    rate_limiting: RateLimitConfig = Field(default_factory=RateLimitConfig)
    channels: ChannelConfig = Field(default_factory=ChannelConfig)

    # Security
    api_key: Optional[str] = Field(
        default=None,
        env="NOTIFICATION_API_KEY",
        description="API key for service authentication"
    )
    allowed_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        env="NOTIFICATION_ALLOWED_ORIGINS",
        description="CORS allowed origins"
    )

    # Health check
    health_check_interval_seconds: int = Field(
        default=60,
        env="HEALTH_CHECK_INTERVAL",
        description="Seconds between health checks"
    )

    @field_validator('database', mode='before')
    @classmethod
    def validate_database_config(cls, v):
        """Validate database configuration."""
        if isinstance(v, dict):
            return DatabaseConfig(**v)
        return v

    @field_validator('server', mode='before')
    @classmethod
    def validate_server_config(cls, v):
        """Validate server configuration."""
        if isinstance(v, dict):
            return ServerConfig(**v)
        return v

    @field_validator('processing', mode='before')
    @classmethod
    def validate_processing_config(cls, v):
        """Validate processing configuration."""
        if isinstance(v, dict):
            return ProcessingConfig(**v)
        return v

    @field_validator('rate_limiting', mode='before')
    @classmethod
    def validate_rate_limiting_config(cls, v):
        """Validate rate limiting configuration."""
        if isinstance(v, dict):
            return RateLimitConfig(**v)
        return v

    @field_validator('channels', mode='before')
    @classmethod
    def validate_channels_config(cls, v):
        """Validate channels configuration."""
        if isinstance(v, dict):
            return ChannelConfig(**v)
        return v

    model_config = ConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False
    )

    def get_channel_config(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific channel.

        Args:
            channel_name: Name of the channel

        Returns:
            Channel configuration dictionary or None if not found
        """
        return getattr(self.channels, channel_name, None)

    def is_channel_enabled(self, channel_name: str) -> bool:
        """
        Check if a channel is enabled.

        Args:
            channel_name: Name of the channel

        Returns:
            True if channel is enabled, False otherwise
        """
        channel_config = self.get_channel_config(channel_name)
        if not channel_config:
            return False
        return channel_config.get("enabled", False)

    def get_rate_limit_for_channel(self, channel_name: str) -> int:
        """
        Get rate limit for a specific channel.

        Args:
            channel_name: Name of the channel

        Returns:
            Rate limit per minute for the channel
        """
        # First check channel-specific config
        channel_config = self.get_channel_config(channel_name)
        if channel_config and "rate_limit_per_minute" in channel_config:
            return channel_config["rate_limit_per_minute"]

        # Fall back to default limits
        return self.rate_limiting.default_limits.get(channel_name, 60)


# Global configuration instance
config = NotificationServiceConfig()

# Override database URL from main config
try:
    from config.donotshare.donotshare import DB_URL
    config.database.url = DB_URL
except ImportError:
    # Keep the default PostgreSQL URL
    pass

_logger.info("Notification service configuration loaded")
_logger.debug("Database URL: %s", config.database.url)
_logger.debug("Server: %s:%s", config.server.host, config.server.port)
_logger.debug("Enabled channels: %s", [
    name for name in ["telegram", "email", "sms"]
    if config.is_channel_enabled(name)
])