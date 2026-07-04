"""
Simplified Configuration Models
==============================

Pydantic-based configuration models for the trading platform.
These models provide strict typing, automatic validation, and centralized
management of defaults and validation rules.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


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


class BrokerConfig(BaseModel):
    """Configuration for a broker."""

    type: str = Field(..., description="Broker type (binance, ibkr, mock)")
    trading_mode: str = Field("paper", description="Trading mode (paper, live)")
    cash: float = Field(10000.0, description="Initial balance")
    live_trading_confirmed: bool = Field(False, description="Explicit confirmation for live trading")
    paper_trading_config: Dict[str, Any] = Field(default_factory=dict, description="Paper trading specific settings")
    risk_management: Dict[str, Any] = Field(default_factory=dict, description="Broker-level risk settings")


class StrategyParamsConfig(BaseModel):
    """Configuration for strategy parameters."""

    type: str = Field(..., description="Strategy type name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class TradingBotConfig(BaseModel):
    """
    Modernized configuration for a trading bot.
    Aligns with StrategyManager Nested structure.
    """

    bot_id: str = Field(..., description="Unique bot identifier")
    name: str = Field("TradingBot", description="Bot name")
    symbol: str = Field(..., description="Trading symbol")

    broker: BrokerConfig = Field(..., description="Broker configuration")
    strategy: StrategyParamsConfig = Field(..., description="Strategy configuration")

    data: Dict[str, Any] = Field(default_factory=dict, description="Data feed configuration")
    risk: Dict[str, Any] = Field(default_factory=dict, description="Risk management configuration")
    notifications: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")
    logging: Dict[str, Any] = Field(default_factory=dict, description="Logging settings")
    trading: Dict[str, Any] = Field(default_factory=dict, description="General trading settings")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @field_validator("bot_id")
    def validate_bot_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Bot ID cannot be empty")
        return v.strip()


class OptimizerConfig(BaseModel):
    """Configuration for strategy optimization."""

    optimizer_id: str = Field(..., description="Unique optimizer identifier")
    name: str = Field(..., description="Optimizer name")
    description: str | None = Field(None, description="Optimizer description")
    # Optimization parameters
    optimizer_type: str = Field("optuna", description="Optimizer type (optuna, hyperopt, etc.)")
    n_trials: int = Field(100, description="Number of optimization trials")
    timeout: int | None = Field(None, description="Optimization timeout in seconds")
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
    description: str | None = Field(None, description="Strategy description")
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

    @field_validator("id")
    def validate_id(cls, v):
        if not v or not v.strip():
            raise ValueError("Strategy ID cannot be empty")
        return v.strip()


class DataConfig(BaseModel):
    """Configuration for data sources."""

    data_id: str = Field(..., description="Unique data configuration identifier")
    name: str = Field(..., description="Data configuration name")
    description: str | None = Field(None, description="Data configuration description")
    # Data source
    data_source: DataSourceType = Field(..., description="Data source type")
    # Data parameters
    symbols: List[str] = Field(..., description="List of symbols to fetch")
    interval: str = Field("1h", description="Data interval")
    lookback: int = Field(1000, description="Number of historical bars")
    # Storage
    save_to_csv: bool = Field(False, description="Save data to CSV files")
    csv_directory: str | None = Field(None, description="CSV output directory")
    # Real-time data
    enable_live_feed: bool = Field(False, description="Enable real-time data feed")
    live_feed_interval: str | None = Field(None, description="Live feed update interval")
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
