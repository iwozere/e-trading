"""
Simplified Configuration Models
==============================

Pydantic-based configuration models for the trading platform.
These models provide strict typing, automatic validation, and centralized
management of defaults and validation rules.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator, field_validator

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class BrokerType(str, Enum):
    """Supported broker types"""
    BINANCE = "binance"
    BINANCE_PAPER = "binance_paper"
    IBKR = "ibkr"
    MOCK = "mock"

class DataSourceType(str, Enum):
    """Supported data source types"""
    BINANCE = "binance"
    YAHOO = "yahoo"
    IBKR = "ibkr"

class StrategyType(str, Enum):
    """Supported strategy types"""
    CUSTOM = "custom"
    RSI_BB = "rsi_bb"
    RSI_BB_VOLUME = "rsi_bb_volume"
    RSI_ICHIMOKU = "rsi_ichimoku"
    RSI_VOLUME_SUPERTREND = "rsi_volume_supertrend"
    BB_VOLUME_SUPERTREND = "bb_volume_supertrend"

class TradingBotConfig(BaseModel):
    """Configuration for a trading bot."""
    bot_id: str = Field(..., description="Unique bot identifier")
    name: str = Field(..., description="Bot name")
    description: Optional[str] = Field(None, description="Bot description")
    # Trading parameters
    symbol: str = Field(..., description="Trading symbol (e.g., 'BTCUSDT')")
    broker_type: BrokerType = Field(..., description="Broker type")
    data_source: DataSourceType = Field(..., description="Data source type")
    # Risk management
    initial_balance: float = Field(10000.0, description="Initial account balance")
    risk_per_trade: float = Field(2.0, description="Risk per trade as percentage")
    max_positions: int = Field(5, description="Maximum concurrent positions")
    # Strategy parameters
    strategy_name: str = Field(..., description="Strategy name")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    # Data parameters
    interval: str = Field("1h", description="Data interval (1m, 5m, 15m, 1h, 1d)")
    lookback: int = Field(1000, description="Number of historical bars to load")
    # Logging and notifications
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[str] = Field(None, description="Log file path")
    enable_notifications: bool = Field(True, description="Enable notifications")
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    @field_validator('bot_id')
    def validate_bot_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Bot ID cannot be empty')
        return v.strip()
    @field_validator('risk_per_trade')
    def validate_risk_per_trade(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('Risk per trade must be between 0 and 100')
        return v
    @field_validator('initial_balance')
    def validate_initial_balance(cls, v):
        if v <= 0:
            raise ValueError('Initial balance must be positive')
        return v

class OptimizerConfig(BaseModel):
    """Configuration for strategy optimization."""
    optimizer_id: str = Field(..., description="Unique optimizer identifier")
    name: str = Field(..., description="Optimizer name")
    description: Optional[str] = Field(None, description="Optimizer description")
    # Optimization parameters
    optimizer_type: str = Field("optuna", description="Optimizer type (optuna, hyperopt, etc.)")
    n_trials: int = Field(100, description="Number of optimization trials")
    timeout: Optional[int] = Field(None, description="Optimization timeout in seconds")
    # Strategy parameters
    strategy_name: str = Field(..., description="Strategy to optimize")
    param_ranges: Dict[str, Any] = Field(..., description="Parameter ranges for optimization")
    # Data parameters
    symbol: str = Field(..., description="Trading symbol")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    interval: str = Field("1h", description="Data interval")
    # Capital and risk
    initial_capital: float = Field(10000.0, description="Initial capital for backtesting")
    commission: float = Field(0.001, description="Commission rate")
    # Optimization metrics
    optimization_metric: str = Field("sharpe_ratio", description="Metric to optimize")
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

class StrategyConfig(BaseModel):
    """Configuration for trading strategies."""
    id: str = Field(..., description="Unique strategy identifier")
    name: str = Field(..., description="Strategy name")
    description: Optional[str] = Field(None, description="Strategy description")
    enabled: bool = Field(True, description="Whether strategy is enabled")

    # Trading parameters
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    broker: Dict[str, Any] = Field(..., description="Broker configuration")
    strategy: Dict[str, Any] = Field(..., description="Strategy parameters")

    # Optional configurations
    data: Dict[str, Any] = Field(default_factory=dict, description="Data configuration")
    trading: Dict[str, Any] = Field(default_factory=dict, description="Trading settings")
    risk_management: Dict[str, Any] = Field(default_factory=dict, description="Risk management settings")
    notifications: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    @field_validator('id')
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError('Strategy ID cannot be empty')
        return v.strip()

class DataConfig(BaseModel):
    """Configuration for data sources."""
    data_id: str = Field(..., description="Unique data configuration identifier")
    name: str = Field(..., description="Data configuration name")
    description: Optional[str] = Field(None, description="Data configuration description")
    # Data source
    data_source: DataSourceType = Field(..., description="Data source type")
    # Data parameters
    symbols: List[str] = Field(..., description="List of symbols to fetch")
    interval: str = Field("1h", description="Data interval")
    lookback: int = Field(1000, description="Number of historical bars")
    # Storage
    save_to_csv: bool = Field(False, description="Save data to CSV files")
    csv_directory: Optional[str] = Field(None, description="CSV output directory")
    # Real-time data
    enable_live_feed: bool = Field(False, description="Enable real-time data feed")
    live_feed_interval: Optional[str] = Field(None, description="Live feed update interval")
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
