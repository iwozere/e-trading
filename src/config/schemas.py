"""
Configuration Schemas
====================

Pydantic-based schemas for all configuration types with validation rules,
documentation, and environment-specific defaults.
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import time
import os


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


class PerformanceConfig(ConfigSchema):
    """Performance monitoring configuration"""
    
    target_sharpe_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Target Sharpe ratio"
    )
    
    target_win_rate: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="Target win rate percentage"
    )
    
    target_profit_factor: float = Field(
        default=1.5,
        ge=0.0,
        le=10.0,
        description="Target profit factor"
    )
    
    max_consecutive_losses: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum consecutive losses before alert"
    )
    
    performance_check_interval: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Performance check interval in hours"
    )


class DataConfig(ConfigSchema):
    """Data feed configuration"""
    
    data_source: DataSourceType = Field(
        description="Data source type"
    )
    
    symbol: str = Field(
        description="Trading symbol"
    )
    
    interval: str = Field(
        default="1h",
        description="Data interval"
    )
    
    lookback_bars: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of historical bars to load"
    )
    
    retry_interval: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Retry interval in seconds"
    )
    
    # Binance-specific fields
    testnet: bool = Field(
        default=False,
        description="Use Binance testnet"
    )
    
    # Yahoo-specific fields
    polling_interval: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Polling interval in seconds"
    )
    
    # IBKR-specific fields
    host: str = Field(
        default="127.0.0.1",
        description="IBKR host"
    )
    
    port: int = Field(
        default=7497,
        ge=1,
        le=65535,
        description="IBKR port"
    )
    
    client_id: int = Field(
        default=1,
        ge=1,
        le=999999,
        description="IBKR client ID"
    )
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v):
        """Validate symbol format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()


class NotificationConfig(ConfigSchema):
    """Notification configuration"""
    
    enabled: bool = Field(
        default=True,
        description="Enable notifications"
    )
    
    telegram: Dict[str, Any] = Field(
        default={
            "enabled": False,
            "notify_on": [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT, NotificationType.ERROR]
        },
        description="Telegram notification settings"
    )
    
    email: Dict[str, Any] = Field(
        default={
            "enabled": False,
            "notify_on": [NotificationType.TRADE_ENTRY, NotificationType.TRADE_EXIT, NotificationType.ERROR]
        },
        description="Email notification settings"
    )
    
    @field_validator('telegram', 'email')
    @classmethod
    def validate_notification_settings(cls, v):
        """Validate notification settings"""
        if v.get('enabled', False):
            required_fields = ['bot_token', 'chat_id'] if 'bot_token' in v else ['smtp_server', 'email']
            for field in required_fields:
                if not v.get(field):
                    raise ValueError(f"Missing required field for notification: {field}")
        return v


class OptimizerConfig(ConfigSchema):
    """Optimization configuration"""
    
    optimizer_type: str = Field(
        default="optuna",
        description="Optimization algorithm"
    )
    
    initial_capital: float = Field(
        default=1000.0,
        ge=100.0,
        le=1000000.0,
        description="Initial capital for backtests"
    )
    
    commission: float = Field(
        default=0.001,
        ge=0.0,
        le=0.1,
        description="Trading commission rate"
    )
    
    risk_free_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Risk-free rate for metrics"
    )
    
    n_trials: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of optimization trials"
    )
    
    n_jobs: int = Field(
        default=1,
        ge=-1,
        le=32,
        description="Number of parallel jobs (-1 for all cores)"
    )
    
    position_size: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Position size as fraction of capital"
    )
    
    # Entry/Exit strategy configurations
    entry_strategies: List[Dict[str, Any]] = Field(
        default=[],
        description="Entry strategy configurations"
    )
    
    exit_strategies: List[Dict[str, Any]] = Field(
        default=[],
        description="Exit strategy configurations"
    )
    
    # Visualization settings
    plot: bool = Field(
        default=True,
        description="Generate plots"
    )
    
    save_trades: bool = Field(
        default=True,
        description="Save trade logs"
    )
    
    output_dir: str = Field(
        default="results",
        description="Output directory for results"
    )


class TradingConfig(ConfigSchema):
    """Trading bot configuration"""
    
    bot_id: str = Field(
        description="Unique bot identifier"
    )
    
    broker: Dict[str, Any] = Field(
        description="Broker configuration"
    )
    
    trading: Dict[str, Any] = Field(
        description="Trading parameters"
    )
    
    data: DataConfig = Field(
        description="Data feed configuration"
    )
    
    strategy: Dict[str, Any] = Field(
        description="Strategy configuration"
    )
    
    risk_management: RiskManagementConfig = Field(
        default_factory=RiskManagementConfig,
        description="Risk management settings"
    )
    
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging settings"
    )
    
    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig,
        description="Scheduling settings"
    )
    
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance monitoring settings"
    )
    
    notifications: NotificationConfig = Field(
        default_factory=NotificationConfig,
        description="Notification settings"
    )
    
    @field_validator('bot_id')
    @classmethod
    def validate_bot_id(cls, v):
        """Validate bot ID format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Bot ID cannot be empty")
        return v.strip()
    
    @model_validator(mode='after')
    def validate_broker_config(self):
        """Validate broker configuration"""
        broker_config = self.broker
        if not broker_config:
            raise ValueError("Broker configuration is required")
        
        broker_type = broker_config.get('type')
        if not broker_type:
            raise ValueError("Broker type is required")
        
        # Validate broker-specific requirements
        if broker_type in [BrokerType.BINANCE_LIVE, BrokerType.BINANCE_PAPER]:
            if not broker_config.get('api_key') and broker_type == BrokerType.BINANCE_LIVE:
                raise ValueError("API key required for live Binance trading")
        
        return self
    
    @model_validator(mode='after')
    def validate_strategy_config(self):
        """Validate strategy configuration"""
        strategy_config = self.strategy
        if not strategy_config:
            raise ValueError("Strategy configuration is required")
        
        strategy_type = strategy_config.get('type')
        if not strategy_type:
            raise ValueError("Strategy type is required")
        
        # Validate strategy-specific requirements
        if strategy_type == StrategyType.CUSTOM:
            if not strategy_config.get('entry_logic') or not strategy_config.get('exit_logic'):
                raise ValueError("Custom strategy requires entry and exit logic")
        
        return self 