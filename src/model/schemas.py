"""
Configuration Schemas
====================

Pydantic-based schemas for all configuration types with validation rules,
documentation, and environment-specific defaults.
"""

import os
from datetime import time
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class BrokerType(str, Enum):
    """Supported broker types"""
    BINANCE_PAPER = "binance_paper"
    BINANCE_LIVE = "binance_live"
    IBKR = "ibkr"
    MOCK = "mock"

class DataSourceType(str, Enum):
    """Supported data source types"""
    BINANCE = "binance"
    YAHOO = "yahoo"
    IBKR = "ibkr"
    CSV = "csv"

class StrategyType(str, Enum):
    """Supported strategy types"""
    CUSTOM = "custom"
    RSI_BB = "rsi_bb"
    RSI_BB_VOLUME = "rsi_bb_volume"
    RSI_ICHIMOKU = "rsi_ichimoku"
    RSI_VOLUME_SUPERTREND = "rsi_volume_supertrend"
    BB_VOLUME_SUPERTREND = "bb_volume_supertrend"

class NotificationType(str, Enum):
    """Notification event types"""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    ERROR = "error"
    STATUS = "status"
    DAILY_SUMMARY = "daily_summary"
    PERFORMANCE_ALERT = "performance_alert"

class ConfigSchema(BaseModel):
    """Base configuration schema with common fields"""
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Environment (development/staging/production/testing)"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )
    description: Optional[str] = Field(
        default=None,
        description="Configuration description"
    )
    created_at: Optional[str] = Field(
        default=None,
        description="Configuration creation timestamp"
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="Configuration last update timestamp"
    )

class RiskManagementConfig(ConfigSchema):
    """Risk management configuration"""
    stop_loss_pct: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Stop loss percentage"
    )
    take_profit_pct: float = Field(
        default=10.0,
        ge=0.1,
        le=100.0,
        description="Take profit percentage"
    )
    max_daily_trades: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum trades per day"
    )
    max_daily_loss: float = Field(
        default=50.0,
        ge=0.1,
        le=100.0,
        description="Maximum daily loss percentage"
    )
    max_drawdown_pct: float = Field(
        default=20.0,
        ge=1.0,
        le=100.0,
        description="Maximum drawdown percentage"
    )
    max_exposure: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Maximum portfolio exposure"
    )
    trailing_stop: Dict[str, Any] = Field(
        default={
            "enabled": False,
            "activation_pct": 3.0,
            "trailing_pct": 2.0
        },
        description="Trailing stop configuration"
    )
    @field_validator('take_profit_pct')
    @classmethod
    def validate_take_profit(cls, v):
        """Ensure take profit is greater than stop loss"""
        if v <= 0:
            raise ValueError("Take profit must be positive")
        return v
    @model_validator(mode='after')
    def validate_risk_parameters(self):
        """Validate risk management parameters"""
        if self.take_profit_pct <= self.stop_loss_pct:
            raise ValueError("Take profit must be greater than stop loss")
        return self

class LoggingConfig(ConfigSchema):
    """Logging configuration"""
    level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level"
    )
    save_trades: bool = Field(
        default=True,
        description="Save trade logs to database"
    )
    save_equity_curve: bool = Field(
        default=True,
        description="Save equity curve data"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum log file size in MB"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        le=50,
        description="Number of backup log files"
    )
    @field_validator('log_file')
    @classmethod
    def validate_log_file(cls, v):
        """Ensure log directory exists"""
        if v:
            log_dir = os.path.dirname(v)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
        return v

class SchedulingConfig(ConfigSchema):
    """Trading schedule configuration"""
    enabled: bool = Field(
        default=False,
        description="Enable scheduled trading"
    )
    start_time: time = Field(
        default=time(9, 0),
        description="Trading start time"
    )
    end_time: time = Field(
        default=time(17, 0),
        description="Trading end time"
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for scheduling"
    )
    trading_days: List[str] = Field(
        default=["monday", "tuesday", "wednesday", "thursday", "friday"],
        description="Trading days of the week"
    )
    @field_validator('trading_days')
    @classmethod
    def validate_trading_days(cls, v):
        """Validate trading days"""
        valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for day in v:
            if day.lower() not in valid_days:
                raise ValueError(f"Invalid trading day: {day}")
        return [day.lower() for day in v]

# ...rest of the file (other config models, validators, etc.) should be restored here as well...
