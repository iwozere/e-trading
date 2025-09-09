"""
Data schemas and type definitions for the e-trading system.

This module provides type definitions, protocols, and data structures used across
the data management system, ensuring consistency and type safety.
"""

from typing import Protocol, runtime_checkable, Optional, Union, Dict, Any, List
from datetime import datetime
from enum import Enum
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field

# Import cache directory setting
try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"


class DataInterval(Enum):
    """Standard data intervals supported by the system."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


class DataProvider(Enum):
    """Supported data providers."""
    BINANCE = "binance"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    FMP = "fmp"
    COINGECKO = "coingecko"
    IBKR = "ibkr"


class Environment(Enum):
    """Environment types for the trading system."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class BrokerType(Enum):
    """Supported broker types."""
    BINANCE_PAPER = "binance_paper"
    BINANCE_LIVE = "binance_live"
    IBKR_PAPER = "ibkr_paper"
    IBKR_LIVE = "ibkr_live"


class DataSourceType(Enum):
    """Data source types."""
    BINANCE = "binance"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    POLYGON = "polygon"
    TWELVE_DATA = "twelve_data"
    FMP = "fmp"
    COINGECKO = "coingecko"
    IBKR = "ibkr"


class StrategyType(Enum):
    """Strategy types."""
    CUSTOM = "custom"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    SCALPING = "scalping"


class NotificationType(Enum):
    """Notification types."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    ERROR = "error"
    STATUS = "status"
    ALERT = "alert"


@runtime_checkable
class Fundamentals(Protocol):
    """
    Protocol for fundamentals data that can be returned by data providers.

    This allows for optional fundamentals support without forcing every
    downloader to implement it.
    """

    def get_pe_ratio(self) -> Optional[float]:
        """Get Price-to-Earnings ratio."""
        ...

    def get_market_cap(self) -> Optional[float]:
        """Get market capitalization."""
        ...

    def get_dividend_yield(self) -> Optional[float]:
        """Get dividend yield."""
        ...

    def get_beta(self) -> Optional[float]:
        """Get beta value."""
        ...

    def get_52_week_high(self) -> Optional[float]:
        """Get 52-week high price."""
        ...

    def get_52_week_low(self) -> Optional[float]:
        """Get 52-week low price."""
        ...


class OHLCVData:
    """
    Standardized OHLCV (Open, High, Low, Close, Volume) data structure.

    This class provides a consistent interface for market data across
    different providers and timeframes.
    """

    def __init__(
        self,
        open_prices: Union[pd.Series, list, None] = None,
        high_prices: Union[pd.Series, list, None] = None,
        low_prices: Union[pd.Series, list, None] = None,
        close_prices: Union[pd.Series, list, None] = None,
        volumes: Union[pd.Series, list, None] = None,
        timestamps: Union[pd.Series, list, None] = None,
        symbol: Optional[str] = None,
        interval: Optional[DataInterval] = None,
        provider: Optional[DataProvider] = None
    ):
        self.open_prices = pd.Series(open_prices) if open_prices is not None else pd.Series()
        self.high_prices = pd.Series(high_prices) if high_prices is not None else pd.Series()
        self.low_prices = pd.Series(low_prices) if low_prices is not None else pd.Series()
        self.close_prices = pd.Series(close_prices) if close_prices is not None else pd.Series()
        self.volumes = pd.Series(volumes) if volumes is not None else pd.Series()
        self.timestamps = pd.Series(timestamps) if timestamps is not None else pd.Series()
        self.symbol = symbol
        self.interval = interval
        self.provider = provider

        # Ensure all series have the same length
        self._validate_data()

    def _validate_data(self):
        """Validate that all data series have the same length."""
        lengths = [
            len(self.open_prices),
            len(self.high_prices),
            len(self.low_prices),
            len(self.close_prices),
            len(self.volumes),
            len(self.timestamps)
        ]

        if len(set(lengths)) > 1:
            raise ValueError(f"All data series must have the same length. Got: {lengths}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with standard column names."""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'open': self.open_prices,
            'high': self.high_prices,
            'low': self.low_prices,
            'close': self.close_prices,
            'volume': self.volumes
        }).set_index('timestamp')

    def to_csv(self, filepath: Union[str, Path], **kwargs) -> None:
        """Save data to CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, **kwargs)

    def to_parquet(self, filepath: Union[str, Path], **kwargs) -> None:
        """Save data to Parquet file."""
        df = self.to_dataframe()
        df.to_parquet(filepath, **kwargs)

    def __len__(self) -> int:
        """Return the number of data points."""
        return len(self.timestamps)

    def __repr__(self) -> str:
        return f"OHLCVData(symbol={self.symbol}, interval={self.interval}, provider={self.provider}, length={len(self)})"


class DataCacheConfig:
    """Configuration for data caching system."""

    def __init__(
        self,
        cache_dir: Union[str, Path] = DATA_CACHE_DIR,
        max_cache_size_gb: float = 10.0,
        compression: str = "snappy",
        partition_by: list = None
    ):
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_gb = max_cache_size_gb
        self.compression = compression
        self.partition_by = partition_by or ["provider", "symbol", "interval"]

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, provider: str, symbol: str, interval: str) -> Path:
        """Get cache path for specific provider, symbol, and interval."""
        return self.cache_dir / provider / symbol / interval

    def get_cache_file_path(self, provider: str, symbol: str, interval: str,
                           start_date: datetime, end_date: datetime,
                           file_format: str = "parquet") -> Path:
        """Get full cache file path with date range."""
        cache_dir = self.get_cache_path(provider, symbol, interval)
        cache_dir.mkdir(parents=True, exist_ok=True)

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        filename = f"{symbol}_{interval}_{start_str}_{end_str}.{file_format}"

        return cache_dir / filename


# Type aliases for common use cases
DataFrameOrOHLCV = Union[pd.DataFrame, OHLCVData]
OptionalFundamentals = Optional[Fundamentals]
CachePath = Union[str, Path]

__all__ = [
    'DataInterval',
    'DataProvider',
    'Fundamentals',
    'Fundamentals',
    'OHLCVData',
    'DataCacheConfig',
    'DataFrameOrOHLCV',
    'OptionalFundamentals',
    'CachePath'
]


@dataclass
class Fundamentals:
    """
    Concrete implementation of Fundamentals Protocol.

    This class provides all the fundamental data fields that can be returned
    by data providers, maintaining compatibility with existing code.
    """
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    current_price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    dividend_yield: Optional[float] = None
    earnings_per_share: Optional[float] = None
    # Additional fields for comprehensive fundamental analysis
    price_to_book: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None
    net_income: Optional[float] = None
    net_income_growth: Optional[float] = None
    free_cash_flow: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    beta: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    exchange: Optional[str] = None
    currency: Optional[str] = None
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    short_ratio: Optional[float] = None
    payout_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value: Optional[float] = None
    enterprise_value_to_ebitda: Optional[float] = None
    # Data source information
    data_source: Optional[str] = None
    last_updated: Optional[str] = None
    # Track which provider supplied each value
    sources: Optional[Dict[str, str]] = field(default_factory=dict)

    def get_pe_ratio(self) -> Optional[float]:
        """Get Price-to-Earnings ratio."""
        return self.pe_ratio

    def get_market_cap(self) -> Optional[float]:
        """Get market capitalization."""
        return self.market_cap

    def get_dividend_yield(self) -> Optional[float]:
        """Get dividend yield."""
        return self.dividend_yield

    def get_beta(self) -> Optional[float]:
        """Get beta value."""
        return self.beta

    def get_52_week_high(self) -> Optional[float]:
        """Get 52-week high price."""
        # This field doesn't exist in the original, but we can add it
        return getattr(self, 'fifty_two_week_high', None)

    def get_52_week_low(self) -> Optional[float]:
        """Get 52-week low price."""
        # This field doesn't exist in the original, but we can add it
        return getattr(self, 'fifty_two_week_low', None)


# Configuration Schema Classes
@dataclass
class ConfigSchema:
    """Base configuration schema."""
    version: str = "1.0.0"
    description: Optional[str] = None
    environment: str = "development"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TradingConfig(ConfigSchema):
    """Trading bot configuration schema."""
    bot_id: str = ""
    broker: Dict[str, Any] = field(default_factory=dict)
    trading: Dict[str, Any] = field(default_factory=dict)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    notifications: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizerConfig(ConfigSchema):
    """Optimizer configuration schema."""
    optimizer_id: str = ""
    strategy: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    backtest: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig(ConfigSchema):
    """Data configuration schema."""
    data_source: str = "binance"
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    cache: Dict[str, Any] = field(default_factory=dict)
    providers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationConfig(ConfigSchema):
    """Notification configuration schema."""
    notification_id: str = ""
    type: str = "email"
    enabled: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)
    recipients: List[str] = field(default_factory=list)


@dataclass
class RiskManagementConfig(ConfigSchema):
    """Risk management configuration schema."""
    risk_id: str = ""
    max_position_size: float = 0.1
    stop_loss: float = 0.02
    take_profit: float = 0.04
    max_drawdown: float = 0.15
    risk_per_trade: float = 0.02


@dataclass
class LoggingConfig(ConfigSchema):
    """Logging configuration schema."""
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class SchedulingConfig(ConfigSchema):
    """Scheduling configuration schema."""
    schedule_id: str = ""
    cron_expression: str = "0 0 * * *"
    timezone: str = "UTC"
    enabled: bool = True
    max_retries: int = 3


@dataclass
class PerformanceConfig(ConfigSchema):
    """Performance configuration schema."""
    performance_id: str = ""
    benchmark: str = "SPY"
    risk_free_rate: float = 0.02
    max_leverage: float = 1.0
    rebalance_frequency: str = "monthly"
