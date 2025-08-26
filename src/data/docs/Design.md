# Design

## Purpose

The data module provides a comprehensive, extensible framework for acquiring, processing, and streaming financial market data from multiple sources. It serves as the foundation for the e-trading platform's data infrastructure, supporting both historical analysis and real-time trading operations.

**Core Objectives:**
- Unified interface for multiple data providers
- Support for both historical and real-time data
- Consistent data formats across all sources
- Robust error handling and failover capabilities
- Scalable architecture for high-frequency operations

## Architecture

### High-Level Overview

The data module follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (Trading Strategies, Analytics, Telegram Bot)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Data Access Layer                       │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Data Downloader │    │  Live Data Feed │                │
│  │    Factory      │    │     Factory     │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────┬─────────────────────┬───────────────────────────┘
              │                     │
┌─────────────▼───────────┐ ┌───────▼─────────────────────────┐
│   Historical Data       │ │      Real-time Data             │
│     Downloaders         │ │       Feeds                     │
│ ┌─────────────────────┐ │ │ ┌─────────────────────────────┐ │
│ │ Yahoo Finance       │ │ │ │ Binance WebSocket          │ │
│ │ Alpha Vantage       │ │ │ │ Yahoo Finance Polling      │ │
│ │ Finnhub             │ │ │ │ IBKR Native API           │ │
│ │ Polygon.io          │ │ │ │ CoinGecko Polling         │ │
│ │ Twelve Data         │ │ │ └─────────────────────────────┘ │
│ │ Binance             │ │ └─────────────────────────────────┘
│ │ CoinGecko           │ │
│ └─────────────────────┘ │
└─────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│                    Storage Layer                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   CSV Files     │  │   SQLite DB     │  │  PostgreSQL │ │
│  │  (Historical)   │  │    (Trades)     │  │ (Production)│ │
│  │  ┌─────────────┐ │  └─────────────────┘  └─────────────┘ │
│  │  │ VIX Data    │ │                                         │
│  │  │ (Volatility)│ │                                         │
│  │  └─────────────┘ │                                         │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. Data Downloaders (Historical Data)

**Base Architecture Pattern:**
- Abstract base class (`BaseDataDownloader`) defines common interface
- Concrete implementations for each data provider
- Factory pattern for provider selection and instantiation
- Consistent data format using pandas DataFrames and Fundamentals dataclass

**Key Components:**
```python
BaseDataDownloader (ABC)
├── get_fundamentals() -> Fundamentals
├── get_ohlcv() -> pd.DataFrame
├── save_data() / load_data()
└── download_multiple_symbols()

Concrete Implementations:
├── YahooDataDownloader
├── AlphaVantageDataDownloader  
├── FinnhubDataDownloader
├── PolygonDataDownloader
├── TwelveDataDataDownloader
├── BinanceDataDownloader
├── CoinGeckoDataDownloader
└── VIXDataManager (Specialized)
```

#### 2. Live Data Feeds (Real-time Data)

**Base Architecture Pattern:**
- Abstract base class (`BaseLiveDataFeed`) extends Backtrader's PandasData
- WebSocket/polling implementations for real-time updates
- Automatic reconnection and error recovery
- Historical data preloading for strategy initialization

**Key Components:**
```python
BaseLiveDataFeed (extends bt.feeds.PandasData)
├── _load_historical_data() (abstract)
├── _connect_realtime() (abstract)
├── _get_latest_data() (abstract)
└── _update_loop() (background thread)

Concrete Implementations:
├── BinanceLiveDataFeed (WebSocket)
├── YahooLiveDataFeed (Polling)
├── IBKRLiveDataFeed (Native API)
└── CoinGeckoLiveDataFeed (Polling)
```

#### 3. Factory Pattern Implementation

**DataDownloaderFactory:**
- Provider code mapping (e.g., "yf" → Yahoo Finance)
- Environment variable integration for API keys
- Parameter validation and error handling
- Provider information and capabilities discovery

**DataFeedFactory:**
- Configuration-based feed creation
- Support for different data source types
- Real-time parameter optimization
- Source capability validation

### Data Flow

#### Historical Data Flow

```
1. Request → DataDownloaderFactory.create_downloader(provider_code)
2. Factory → Instantiate specific downloader with API credentials
3. Downloader → Fetch data from external API
4. Processing → Convert to standardized DataFrame/Fundamentals format
5. Storage → Save to CSV files with standardized naming
6. Return → Processed data to application layer
```

#### Real-time Data Flow

```
1. Configuration → DataFeedFactory.create_data_feed(config)
2. Factory → Instantiate specific live feed implementation
3. Historical Load → Fetch initial data for strategy context
4. Real-time Start → Establish WebSocket/polling connection
5. Data Updates → Continuous data processing in background thread
6. Backtrader Integration → Update strategy with new bars
7. Callbacks → Optional user-defined event handlers
```

### Data Models

#### Fundamentals Data Model

Standardized fundamental data structure supporting comprehensive financial analysis:

```python
@dataclass
class Fundamentals:
    # Core identification
    ticker: str
    company_name: str
    data_source: str
    last_updated: str
    
    # Market data
    current_price: float
    market_cap: float
    shares_outstanding: Optional[float]
    
    # Valuation metrics
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    peg_ratio: Optional[float]
    enterprise_value: Optional[float]
    enterprise_value_to_ebitda: Optional[float]
    
    # Financial health
    return_on_equity: Optional[float]
    return_on_assets: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    quick_ratio: Optional[float]
    
    # Profitability
    earnings_per_share: Optional[float]
    revenue: Optional[float]
    net_income: Optional[float]
    free_cash_flow: Optional[float]
    operating_margin: Optional[float]
    profit_margin: Optional[float]
    
    # Growth metrics
    revenue_growth: Optional[float]
    net_income_growth: Optional[float]
    
    # Dividend and risk
    dividend_yield: Optional[float]
    payout_ratio: Optional[float]
    beta: Optional[float]
    
    # Company details
    sector: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    exchange: Optional[str]
    currency: Optional[str]
```

#### OHLCV Data Model

Standardized time series data format:

```python
DataFrame columns:
- timestamp: pd.Timestamp (timezone-aware)
- open: float
- high: float  
- low: float
- close: float
- volume: float
- (optional) adj_close: float
```

#### VIX Data Model

Specialized data structure for CBOE VIX (Volatility Index) data:

```python
# File-based storage in data/vix/vix.csv
DataFrame columns:
- Date: pd.Timestamp (index)
- Open: float
- High: float
- Low: float
- Close: float
- Adj Close: float
- Volume: float

# Key Features:
- Automatic directory creation (data/vix/)
- Incremental updates (only new data)
- Deduplication of overlapping data
- Local CSV storage for offline access
- Historical data from 1990 to present
```

### Database Schema

#### Trade Tracking

```sql
-- Trade lifecycle management
CREATE TABLE trades (
    id VARCHAR(36) PRIMARY KEY,
    bot_id VARCHAR(255) NOT NULL,
    trade_type VARCHAR(10) NOT NULL, -- 'paper', 'live', 'optimization'
    symbol VARCHAR(50) NOT NULL,
    strategy_name VARCHAR(255),
    
    -- Entry details
    entry_timestamp TIMESTAMP,
    entry_price NUMERIC(20, 8),
    entry_size NUMERIC(20, 8),
    entry_order_id VARCHAR(255),
    
    -- Exit details
    exit_timestamp TIMESTAMP,
    exit_price NUMERIC(20, 8),
    exit_size NUMERIC(20, 8),
    exit_order_id VARCHAR(255),
    
    -- Performance
    pnl NUMERIC(20, 8),
    pnl_percent NUMERIC(10, 4),
    commission NUMERIC(20, 8),
    
    -- Status and metadata
    trade_status VARCHAR(20) DEFAULT 'OPEN',
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_trades_bot_id ON trades(bot_id);
CREATE INDEX idx_trades_symbol_timestamp ON trades(symbol, entry_timestamp);
CREATE INDEX idx_trades_status ON trades(trade_status);
```

#### Performance Metrics

```sql
-- Strategy performance tracking
CREATE TABLE performance_metrics (
    id VARCHAR(36) PRIMARY KEY,
    bot_id VARCHAR(255) NOT NULL,
    measurement_timestamp TIMESTAMP NOT NULL,
    
    -- Core metrics
    total_return NUMERIC(10, 4),
    sharpe_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    win_rate NUMERIC(10, 4),
    profit_factor NUMERIC(10, 4),
    
    -- Trade statistics
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    
    -- Additional metrics (JSON for flexibility)
    metrics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Design Decisions

### 1. Multiple Data Provider Support

**Decision:** Support multiple financial data providers with unified interface

**Rationale:**
- **Redundancy**: Failover capabilities when providers have outages
- **Data Quality**: Cross-validation between multiple sources
- **Cost Optimization**: Use free tiers strategically before paid upgrades
- **Coverage**: Different providers excel in different asset classes

**Implementation:**
- Factory pattern for provider selection
- Abstract base classes for consistent interface
- Provider-specific optimizations while maintaining compatibility

### 2. Factory Pattern for Object Creation

**Decision:** Use factory pattern for both data downloaders and live feeds

**Rationale:**
- **Simplicity**: Single entry point for creating data sources
- **Configuration**: Support for environment variables and explicit parameters
- **Validation**: Centralized parameter validation and error handling
- **Extensibility**: Easy addition of new providers without changing client code

**Trade-offs:**
- Additional abstraction layer
- Centralized configuration management complexity

### 3. Pandas DataFrame as Standard Data Format

**Decision:** Use pandas DataFrames for all time series data

**Rationale:**
- **Ecosystem Integration**: Compatible with most Python financial libraries
- **Performance**: Optimized for numerical operations
- **Functionality**: Rich time series and data manipulation capabilities
- **Backtrader Compatibility**: Native support in backtesting framework

**Considerations:**
- Memory usage for large datasets
- Potential need for alternative formats (e.g., Arrow) in future

### 4. Backtrader Integration for Live Feeds

**Decision:** Extend Backtrader's PandasData for live feed base class

**Rationale:**
- **Unified Interface**: Same data feed interface for backtesting and live trading
- **Strategy Compatibility**: Existing strategies work with both historical and live data
- **Framework Benefits**: Leverage Backtrader's data handling and event system

**Implementation Details:**
- Historical data preloading for strategy initialization
- Background thread for real-time updates
- Thread-safe data sharing between update loop and Backtrader

### 5. Database Schema Design

**Decision:** Use both SQLite (development) and PostgreSQL (production) support

**Rationale:**
- **Development Ease**: SQLite for local development and testing
- **Production Scale**: PostgreSQL for production deployments
- **Data Integrity**: Proper constraints and foreign keys
- **Performance**: Optimized indexes for common query patterns

**Schema Decisions:**
- UUID primary keys for distributed system compatibility
- JSONB fields for flexible metadata storage
- Audit trails with created_at/updated_at timestamps
- Proper numeric types for financial calculations

### 6. Error Handling Strategy

**Decision:** Multi-layered error handling with graceful degradation

**Rationale:**
- **Reliability**: Trading systems must handle failures gracefully
- **Debugging**: Comprehensive logging for troubleshooting
- **User Experience**: Meaningful error messages and fallback options

**Implementation:**
- Provider-specific error handling
- Rate limiting and retry logic
- Fallback to alternative data sources
- Comprehensive logging at all levels

### 7. Configuration Management

**Decision:** Environment variables + programmatic configuration

**Rationale:**
- **Security**: API keys not stored in code
- **Flexibility**: Support for both deployment and development scenarios
- **Twelve-Factor App**: Follow modern application deployment practices

**Implementation:**
- Environment variable detection in factories
- Override capability through explicit parameters
- Validation and meaningful error messages for missing credentials

## Performance Considerations

### Data Caching Strategy

1. **File-based Caching**: Historical data saved as CSV files with standardized naming
2. **Memory Caching**: In-memory DataFrame storage for frequently accessed data
3. **Database Caching**: SQLite/PostgreSQL for structured trade and performance data

### Rate Limiting

1. **Provider-Specific Limits**: Respect individual API rate limits
2. **Exponential Backoff**: Implement retry logic with increasing delays
3. **Request Queuing**: Queue requests to avoid overwhelming APIs
4. **Load Balancing**: Distribute requests across multiple providers when possible

### Memory Management

1. **Data Chunking**: Process large datasets in chunks to manage memory
2. **Lazy Loading**: Load data on-demand rather than preloading everything
3. **Garbage Collection**: Explicit cleanup of large DataFrames when no longer needed
4. **Streaming**: Use generators for large data processing operations

### Network Optimization

1. **Connection Pooling**: Reuse HTTP connections for multiple requests
2. **Compression**: Use gzip compression for large data transfers
3. **Parallel Requests**: Concurrent API calls where rate limits allow
4. **Local Proxy**: Optional caching proxy for repeated requests

## Security Considerations

### API Key Management

1. **Environment Variables**: Store sensitive credentials in environment
2. **Key Rotation**: Support for regular API key rotation
3. **Scope Limitation**: Use read-only keys where possible
4. **Audit Logging**: Log API key usage for security monitoring

### Data Privacy

1. **PII Handling**: Minimize personally identifiable information storage
2. **Data Retention**: Implement configurable data retention policies
3. **Encryption**: Encrypt sensitive data at rest and in transit
4. **Access Control**: Role-based access to different data sources

### Network Security

1. **HTTPS Only**: All API communications use encrypted connections
2. **Certificate Validation**: Verify SSL certificates for all connections
3. **IP Whitelisting**: Support for IP-based access restrictions where available
4. **VPN Support**: Compatible with VPN deployments for additional security

## Scalability Design

### Horizontal Scaling

1. **Stateless Design**: Data downloaders are stateless and can be distributed
2. **Database Scaling**: Support for database read replicas and sharding
3. **Load Distribution**: Factory pattern supports multiple instance creation
4. **Microservice Ready**: Components can be deployed as separate services

### Vertical Scaling

1. **Multi-threading**: Background threads for real-time data processing
2. **Async Support**: Async/await patterns for high-concurrency operations
3. **Memory Optimization**: Efficient data structures and memory management
4. **CPU Optimization**: Vectorized operations using pandas and numpy

### Future Extensibility

1. **Plugin Architecture**: Easy addition of new data providers
2. **Protocol Abstraction**: Support for different communication protocols
3. **Data Format Evolution**: Ability to migrate to new data formats
4. **Configuration Evolution**: Backward-compatible configuration updates
