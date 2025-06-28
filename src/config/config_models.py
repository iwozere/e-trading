"""
Simplified Configuration Models
==============================

Pydantic-based configuration models for the trading platform.
These models provide strict typing, automatic validation, and centralized
management of defaults and validation rules.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
from enum import Enum


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


class TradingBotConfig(BaseModel):
    """Main trading bot configuration with validation"""
    
    # Basic identification
    bot_id: str = Field(description="Unique bot identifier")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    version: str = Field(default="1.0.0", description="Configuration version")
    description: Optional[str] = Field(default=None, description="Configuration description")
    
    # Trading parameters
    symbol: str = Field(description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(default="1h", description="Trading timeframe")
    risk_per_trade: float = Field(default=0.01, ge=0.001, le=0.1, description="Risk per trade as fraction of balance")
    max_open_trades: int = Field(default=5, gt=0, le=50, description="Maximum number of open trades")
    position_size: float = Field(default=0.1, ge=0.01, le=1.0, description="Position size as fraction of capital")
    
    # Broker configuration (without sensitive data)
    broker_type: BrokerType = Field(description="Broker type")
    initial_balance: float = Field(default=1000.0, gt=0, description="Initial account balance")
    commission: float = Field(default=0.001, ge=0, le=0.01, description="Trading commission rate")
    
    # Data configuration
    data_source: DataSourceType = Field(description="Data source type")
    lookback_bars: int = Field(default=1000, ge=100, le=10000, description="Number of historical bars to load")
    retry_interval: int = Field(default=60, ge=1, le=3600, description="Retry interval in seconds")
    
    # Strategy configuration
    strategy_type: StrategyType = Field(default=StrategyType.CUSTOM, description="Strategy type")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    
    # Risk management
    stop_loss_pct: float = Field(default=5.0, ge=0.1, le=50.0, description="Stop loss percentage")
    take_profit_pct: float = Field(default=10.0, ge=0.1, le=100.0, description="Take profit percentage")
    max_daily_trades: int = Field(default=10, ge=1, le=1000, description="Maximum trades per day")
    max_daily_loss: float = Field(default=50.0, ge=0.1, le=100.0, description="Maximum daily loss percentage")
    max_drawdown_pct: float = Field(default=20.0, ge=1.0, le=100.0, description="Maximum drawdown percentage")
    max_exposure: float = Field(default=1.0, ge=0.1, le=10.0, description="Maximum portfolio exposure")
    
    # Logging configuration
    log_level: str = Field(default="INFO", description="Logging level")
    save_trades: bool = Field(default=True, description="Save trade logs to database")
    save_equity_curve: bool = Field(default=True, description="Save equity curve data")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Notifications (without sensitive data)
    notifications_enabled: bool = Field(default=True, description="Enable notifications")
    telegram_enabled: bool = Field(default=False, description="Enable Telegram notifications")
    email_enabled: bool = Field(default=False, description="Enable email notifications")
    
    # Additional settings
    paper_trading: bool = Field(default=True, description="Use paper trading mode")
    
    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe format"""
        valid_timeframes = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        if v not in valid_timeframes:
            raise ValueError(f"Unsupported timeframe: {v}. Valid options: {valid_timeframes}")
        return v
    
    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator('bot_id')
    @classmethod
    def validate_bot_id(cls, v: str) -> str:
        """Validate bot ID format"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Bot ID cannot be empty")
        return v.strip()
    
    @field_validator('take_profit_pct')
    @classmethod
    def validate_take_profit(cls, v: float) -> float:
        """Ensure take profit is greater than stop loss"""
        if v <= 0:
            raise ValueError("Take profit must be positive")
        return v
    
    def get_broker_config(self) -> Dict[str, Any]:
        """Get broker configuration dictionary"""
        return {
            "type": self.broker_type,
            "initial_balance": self.initial_balance,
            "commission": self.commission,
            "paper_trading": self.paper_trading
        }
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration dictionary"""
        return {
            "symbol": self.symbol,
            "position_size": self.position_size,
            "max_positions": self.max_open_trades,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_exposure": self.max_exposure
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration dictionary"""
        return {
            "data_source": self.data_source,
            "symbol": self.symbol,
            "interval": self.timeframe,
            "lookback_bars": self.lookback_bars,
            "retry_interval": self.retry_interval
        }
    
    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration dictionary"""
        return {
            "type": self.strategy_type,
            "params": self.strategy_params
        }
    
    def get_risk_management_config(self) -> Dict[str, Any]:
        """Get risk management configuration dictionary"""
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_daily_trades": self.max_daily_trades,
            "max_daily_loss": self.max_daily_loss,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_exposure": self.max_exposure,
            "trailing_stop": {
                "enabled": False,
                "activation_pct": 3.0,
                "trailing_pct": 2.0
            }
        }
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration dictionary"""
        return {
            "level": self.log_level,
            "save_trades": self.save_trades,
            "save_equity_curve": self.save_equity_curve,
            "log_file": self.log_file
        }
    
    def get_notifications_config(self) -> Dict[str, Any]:
        """Get notifications configuration dictionary"""
        return {
            "enabled": self.notifications_enabled,
            "telegram": {
                "enabled": self.telegram_enabled,
                "notify_on": ["trade_entry", "trade_exit", "error", "status"]
            },
            "email": {
                "enabled": self.email_enabled,
                "notify_on": ["trade_entry", "trade_exit", "error"]
            }
        }


class OptimizerConfig(BaseModel):
    """Optimization configuration"""
    
    optimizer_type: str = Field(default="optuna", description="Optimization algorithm")
    initial_capital: float = Field(default=1000.0, ge=100.0, le=1000000.0, description="Initial capital for backtests")
    commission: float = Field(default=0.001, ge=0.0, le=0.1, description="Trading commission rate")
    n_trials: int = Field(default=100, ge=10, le=10000, description="Number of optimization trials")
    n_jobs: int = Field(default=1, ge=-1, le=32, description="Number of parallel jobs (-1 for all cores)")
    position_size: float = Field(default=0.1, ge=0.01, le=1.0, description="Position size as fraction of capital")
    plot: bool = Field(default=True, description="Generate plots")
    save_trades: bool = Field(default=True, description="Save trade logs")
    output_dir: str = Field(default="results", description="Output directory for results")


class DataConfig(BaseModel):
    """Data feed configuration"""
    
    data_source: DataSourceType = Field(description="Data source type")
    symbol: str = Field(description="Trading symbol")
    interval: str = Field(default="1h", description="Data interval")
    lookback_bars: int = Field(default=1000, ge=100, le=10000, description="Number of historical bars to load")
    retry_interval: int = Field(default=60, ge=1, le=3600, description="Retry interval in seconds")
    testnet: bool = Field(default=False, description="Use testnet (for Binance)")
    host: str = Field(default="127.0.0.1", description="Host (for IBKR)")
    port: int = Field(default=7497, ge=1, le=65535, description="Port (for IBKR)")
    client_id: int = Field(default=1, ge=1, le=999999, description="Client ID (for IBKR)")
    
    @field_validator('interval')
    @classmethod
    def validate_interval(cls, v: str) -> str:
        """Validate interval format"""
        valid_intervals = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
        if v not in valid_intervals:
            raise ValueError(f"Unsupported interval: {v}. Valid options: {valid_intervals}")
        return v 