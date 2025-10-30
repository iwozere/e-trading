# Design

## Purpose

The data module provides a comprehensive, extensible framework for acquiring, processing, and streaming financial market data from multiple sources. It serves as the foundation for the e-trading platform's data infrastructure, supporting both historical analysis and real-time trading operations with an intelligent unified cache system.

**Core Objectives:**
- Unified interface through DataManager facade for all data operations
- Intelligent provider selection with configuration-driven rules
- Support for both historical and real-time data with caching
- Consistent data formats across all sources with validation
- Robust error handling and automatic failover capabilities
- Scalable architecture for high-frequency operations
- Efficient file-based caching with gzip compression and metadata
- Multi-step pipeline system for data processing and transformation

## Architecture

### High-Level Overview

The data module follows a layered architecture with clear separation of concerns, intelligent provider selection, and unified database management:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (Trading Strategies, Analytics, Telegram Bot)             │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    DataManager Facade                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ get_ohlcv()     │  │ get_fundamentals│  │ get_live_   │ │
│  │ (Historical)    │  │ (Cached JSON)   │  │ feed()      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────┬─────────────────────┬───────────────────────────┘
              │                     │
┌─────────────▼───────────┐ ┌───────▼─────────────────────────┐
│   ProviderSelector      │ │      Cache Pipeline             │
│   (Intelligent)         │ │      System                     │
│ ┌─────────────────────┐ │ │ ┌─────────────────────────────┐ │
│ │ Symbol Classifier   │ │ │ │ Step 1: Download Alpaca 1m │ │
│ │ Config-Driven Rules │ │ │ │ Step 2: Calculate Timeframes│ │
│ │ Provider Failover   │ │ │ │ Gap Filling & Validation   │ │
│ └─────────────────────┘ │ │ └─────────────────────────────┘ │
└─────────────────────────┘ └─────────────────────────────────┘
              │                     │
┌─────────────▼───────────────────────────────────────────────┐
│                    Data Providers                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────┐ │
│  │ Binance     │ │ FMP         │ │ Yahoo       │ │ Alpaca│ │
│  │ (Crypto)    │ │ (Stocks     │ │ (Stocks     │ │ (US   │ │
│  │             │ │ Intraday)   │ │ Daily)      │ │ Pro)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────┘ │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────┐ │
│  │ Alpha       │ │ Tiingo      │ │ Polygon     │ │ Others│ │
│  │ Vantage     │ │ (Weekly/    │ │ (Pro Data)  │ │       │ │
│  │             │ │ Monthly)    │ │             │ │       │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────┘ │
└─────────────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│                    Unified Cache Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ OHLCV Cache     │  │ Fundamentals    │  │ Metadata &  │ │
│  │ (ohlcv/symbol/  │  │ Cache (JSON)    │  │ Quality     │ │
│  │ timeframe/)     │  │ (TTL-based)     │  │ Tracking    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│                 Unified Database Layer                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Database Service│  │ Repository      │  │ Models &    │ │
│  │ (Orchestration) │  │ Pattern         │  │ Connections │ │
│  │                 │  │ (Data Access)   │  │ (Core DB)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. Unified Database Layer

**New Database Architecture:**
- **Single Database**: Unified SQLite database for all operations (trading + telegram)
- **Service Layer**: `DatabaseService` provides orchestration and session management
- **Repository Pattern**: Clean data access with automatic session cleanup
- **Model Separation**: Trading models (Trade, BotInstance, PerformanceMetrics) and Telegram models (User, Alert, Schedule)

**Key Components:**
```python
DatabaseService
├── get_trading_repo() -> TradeRepository with session management
├── get_telegram_repo() -> TelegramRepository with session management
├── trading_manager -> DatabaseManager for trading operations
└── telegram_manager -> DatabaseManager for telegram operations (same DB)

TelegramService (Clean Interface)
├── User Management -> set_user_email(), verify_code(), approve_user()
├── Alert Management -> add_alert(), get_active_alerts(), update_alert()
├── Schedule Management -> add_schedule(), get_active_schedules()
├── Settings Management -> set_setting(), get_setting()
└── Audit Logging -> log_command_audit(), get_command_audit_stats()
```

**Database Models:**
- **Trading**: Trade, BotInstance, PerformanceMetrics (financial data with UUID keys)
- **Telegram**: TelegramUser, Alert, Schedule, Setting, Feedback, CommandAudit (user management)
- **Shared Base**: Single SQLAlchemy Base for all models with unified metadata

#### 2. Unified Cache System

**New Architecture Pattern:**
- Simplified directory structure: `symbol/timeframe/` instead of `provider/symbol/timeframe/year/`
- Gzip-compressed CSV files for efficient storage
- Embedded provider metadata in JSON files
- Intelligent provider selection based on symbol type and timeframe

**Key Components:**
```python
UnifiedCache
├── put() -> Store data with provider metadata
├── get() -> Retrieve data across multiple years
├── _get_available_years() -> List available data years
├── _save_compressed_csv() -> Save with gzip compression
├── _load_year_data() -> Load specific year data
├── get_stats() -> Cache statistics and health
├── list_symbols() -> List available symbols
├── list_timeframes() -> List timeframes for symbol
├── list_years() -> List years for symbol/timeframe
├── get_data_info() -> Get detailed data information
└── cleanup_old_data() -> Remove old data
```

#### 3. Intelligent Provider Selection

**Ticker Classification System:**
- Automatic symbol type detection (crypto vs stock)
- Provider selection based on symbol type and timeframe
- Fallback mechanisms for provider failures
- Provider capability mapping

**Key Components:**
```python
ProviderSelector
├── get_best_provider() -> Select best provider
├── get_data_provider_config() -> Get provider configuration
├── get_ticker_info() -> Get comprehensive ticker information
├── _classify_symbol() -> Determine symbol type
└── _get_provider_capabilities() -> Check provider features

Provider Selection Logic:
├── Crypto Symbols → Binance (all timeframes)
├── Stock Symbols (Daily) → Yahoo Finance
└── Stock Symbols (Intraday) → Alpha Vantage
```

#### 4. Data Downloaders (Historical Data)

**Base Architecture Pattern:**
- Abstract base class (`BaseDataDownloader`) defines common interface
- Concrete implementations for each data provider
- Intelligent provider selection via ticker classifier
- Consistent data format using pandas DataFrames

**Key Components:**
```python
BaseDataDownloader (ABC)
├── get_fundamentals() -> Fundamentals
├── get_ohlcv() -> pd.DataFrame
├── save_data() / load_data()
└── download_multiple_symbols()

Concrete Implementations:
├── BinanceDataDownloader (Crypto - All timeframes)
├── YahooDataDownloader (Stocks - Daily data)
├── AlphaVantageDataDownloader (Stocks - Intraday data)
└── MockDataSource (Testing fallback)
```

#### 5. Cache Population System

**Automated Cache Management:**
- Intelligent provider selection for each symbol/timeframe
- Incremental updates (only missing data)
- Data validation before caching
- Rate limiting and error handling

**Key Components:**
```python
populate_cache()
├── Configure unified cache
├── Initialize data downloaders
├── Initialize ticker classifier
├── For each symbol/timeframe:
│   ├── Check existing cache data
│   ├── Select best provider
│   ├── Download data if needed
│   ├── Validate data quality
│   └── Cache with metadata
└── Return statistics and results
```

### Data Flow

#### Database Operations Flow

```
1. Application Request → telegram_service.py (Clean Interface)
2. Service Layer → database_service.py (Session Management)
3. Repository Pattern → telegram_repository.py (Data Access)
4. Database Models → telegram_models.py (SQLAlchemy ORM)
5. Database Engine → database.py (Core Connection)
6. SQLite Database → Single unified database file
```

**Example Flow:**
```python
# Application Layer
from src.data.db.services import telegram_service as db
user_status = db.get_user_status("123456")

# Service Layer (automatic)
service = get_database_service()
with service.get_telegram_repo() as repo:
    # Repository Layer (automatic)
    return repo.get_user_status("123456")
    # Session automatically closed
```

#### Historical Data Flow with Intelligent Selection

```
1. Request → populate_cache.py with symbols and timeframes
2. Classification → ProviderSelector determines symbol type
3. Provider Selection → Select best provider based on symbol/timeframe
4. Cache Check → Check existing data in unified cache
5. Download → Fetch data from selected provider (if needed)
6. Validation → Validate OHLCV data quality
7. Storage → Save to unified cache with gzip compression
8. Metadata → Store provider and quality information
9. Return → Processed data to application layer
```

#### Real-time Data Flow

```
1. Configuration → DataFeedFactory.create_data_feed(config)
2. Factory → Instantiate specific live feed implementation
3. Historical Load → Fetch initial data from unified cache
4. Real-time Start → Establish WebSocket/polling connection
5. Data Updates → Continuous data processing in background thread
6. Backtrader Integration → Update strategy with new bars
7. Callbacks → Optional user-defined event handlers
```

### Data Models

#### Fundamentals Cache System

**Cache-First Architecture:**
- **7-Day Cache Rule**: All fundamentals data cached for 7 days before refresh
- **Multi-Provider Support**: Combine data from multiple providers (FMP, Yahoo, Alpha Vantage, IBKR)
- **Provider Priority**: FMP > Yahoo Finance > Alpha Vantage > IBKR > others
- **Stale Data Cleanup**: Automatic removal of outdated data when new data is downloaded

**Data Combination Strategy:**
```python
class FundamentalsCombiner:
    def combine_snapshots(self, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        # Priority-based field selection
        # Fill missing fields from lower-priority providers
        # Validate data consistency across providers
```

**Cache Operations:**
- `find_latest_json(symbol, provider)` - Find most recent cached data
- `write_json(symbol, provider, data, timestamp)` - Cache new fundamentals
- `is_cache_valid(timestamp, max_age_days=7)` - Validate cache age
- `cleanup_stale_data(symbol, provider, new_timestamp)` - Remove old data

#### Unified Cache Data Model

**File Structure:**
```
data-cache/
├── BTCUSDT/
│   ├── 5m/
│   │   ├── 2025.csv.gz          # Data for 2025 only
│   │   ├── 2025.metadata.json   # Metadata for 2025
│   │   ├── 2024.csv.gz          # Data for 2024 only
│   │   └── 2024.metadata.json   # Metadata for 2024
│   └── 1h/
├── AAPL/
│   ├── 5m/
│   ├── 1d/
│   └── fundamentals/
│       ├── yfinance_20250106_143022.json
│       ├── fmp_20250106_143045.json
│       └── alpha_vantage_20250106_143067.json
└── _metadata/
    ├── symbols.json             # Global symbol metadata
    ├── providers.json           # Provider information
    └── quality_scores.json      # Data quality tracking
```

**CSV File Format (Gzip Compressed):**
```python
DataFrame columns:
- timestamp: pd.Timestamp (timezone-naive for compatibility)
- open: float
- high: float  
- low: float
- close: float
- volume: float
```

**Metadata File Format:**
```python
{
    "symbol": "BTCUSDT",
    "timeframe": "5m",
    "year": 2025,
    "data_source": "binance",
    "created_at": "2025-01-15T10:30:00",
    "last_updated": "2025-01-15T10:30:00",
    "start_date": "2025-01-01T00:00:00",
    "end_date": "2025-01-15T10:30:00",
    "data_quality": {
        "score": 0.95,
        "validation_errors": [],
        "gaps": 0,
        "duplicates": 0
    },
    "file_info": {
        "format": "csv.gz",
        "size_bytes": 1024000,
        "rows": 50000,
        "columns": ["open", "high", "low", "close", "volume"]
    },
    "provider_info": {
        "name": "binance",
        "reliability": 0.95,
        "rate_limit": "1200/minute"
    }
}
```

#### Provider Selection Model

**Ticker Classification:**
```python
@dataclass
class TickerInfo:
    symbol: str
    symbol_type: str  # "crypto" or "stock"
    exchange: Optional[str]
    base_asset: Optional[str]  # For crypto pairs
    quote_asset: Optional[str]  # For crypto pairs
    original_provider: str

@dataclass
class ProviderConfig:
    symbol: str
    interval: str
    original_provider: str
    best_provider: str
    reason: str
    exchange: Optional[str]
    base_asset: Optional[str]
    quote_asset: Optional[str]
```

**Provider Selection Logic:**
```python
def _select_best_provider(ticker_info: TickerInfo, interval: str) -> str:
    if ticker_info.symbol_type == "crypto":
        return "binance"  # Always use Binance for crypto
    elif ticker_info.symbol_type == "stock":
        if interval in ["1d", "1w", "1M"]:
            return "yfinance"  # Yahoo Finance for daily data
        else:
            return "alpha_vantage"  # Alpha Vantage for intraday
    else:
        return "yfinance"  # Default fallback
```

#### OHLCV Data Model

**Standardized time series data format:**
```python
DataFrame columns:
- timestamp: pd.Timestamp (timezone-naive for cache compatibility)
- open: float
- high: float  
- low: float
- close: float
- volume: float
- (optional) adj_close: float
```

#### Data Validation Model

**Validation Results:**
```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    quality_score: float
    gaps: int
    duplicates: int
    missing_columns: List[str]
    data_types_valid: bool
    logical_consistency: bool
```

## Design Decisions

### 1. Unified Cache Architecture

**Decision:** Implement simplified cache structure without provider-based directories

**Rationale:**
- **Simplicity**: Easier to manage and navigate cache structure
- **Efficiency**: Reduced directory depth and file system overhead
- **Provider Agnostic**: Data source information embedded in metadata
- **Scalability**: Better performance with large numbers of symbols

**Implementation:**
- Directory structure: `symbol/timeframe/` instead of `provider/symbol/timeframe/year/`
- Provider information stored in metadata files
- Gzip compression for efficient storage
- Direct file access without provider lookup

### 2. Intelligent Provider Selection

**Decision:** Automatic provider selection based on symbol type and timeframe

**Rationale:**
- **Optimization**: Use best provider for each use case
- **Reliability**: Fallback mechanisms for provider failures
- **User Experience**: No need to manually select providers
- **Data Quality**: Ensure best data source for each symbol/timeframe combination

**Implementation:**
- Ticker classification system for symbol type detection
- Provider capability mapping
- Automatic selection logic based on symbol type and timeframe
- Fallback to mock data for testing

### 3. Gzip Compression for Cache Files

**Decision:** Use gzip compression for all cached CSV files

**Rationale:**
- **Storage Efficiency**: 60-80% reduction in storage space
- **Performance**: Faster I/O operations with compressed files
- **Standardization**: Built-in Python support, no external dependencies
- **Compatibility**: Works across all platforms

**Implementation:**
- All CSV files saved as `.csv.gz`
- Transparent compression/decompression in cache operations
- Metadata files remain uncompressed for quick access

### 4. Provider-Specific Optimization

**Decision:** Optimize provider selection for specific use cases

**Rationale:**
- **Data Quality**: Each provider excels in different areas
- **Rate Limits**: Distribute load across multiple providers
- **Coverage**: Ensure comprehensive data coverage
- **Cost Efficiency**: Use free tiers strategically

**Provider Selection Strategy:**
- **Binance**: All cryptocurrency data (best rate limits and coverage)
- **Yahoo Finance**: Stock daily data (no API key required, comprehensive)
- **Alpha Vantage**: Stock intraday data (full historical data, no 60-day limit)

### 5. Incremental Cache Updates

**Decision:** Only download missing data, skip existing recent data

**Rationale:**
- **Efficiency**: Avoid redundant downloads
- **Rate Limiting**: Respect API limits by minimizing requests
- **Speed**: Faster cache population for existing data
- **Cost**: Reduce API usage costs

**Implementation:**
- Check existing cache data before downloading
- Skip recent data (configurable age threshold)
- Update only when data is old or missing
- Maintain data freshness for current year

### 6. Comprehensive Data Validation

**Decision:** Validate all data before caching

**Rationale:**
- **Data Quality**: Ensure only valid data is cached
- **Error Prevention**: Catch data issues early
- **Reliability**: Prevent downstream errors from bad data
- **Monitoring**: Track data quality over time

**Validation Checks:**
- Required columns (open, high, low, close, volume)
- Data type validation
- Logical consistency (high >= max(open, close), low <= min(open, close))
- Timestamp validation and ordering
- Gap detection
- Duplicate detection
- Quality scoring

### 7. Rate Limiting and Error Handling

**Decision:** Built-in rate limiting and comprehensive error handling

**Rationale:**
- **API Compliance**: Respect provider rate limits
- **Reliability**: Handle API failures gracefully
- **User Experience**: Provide meaningful error messages
- **Monitoring**: Track provider performance

**Implementation:**
- Provider-specific rate limiting
- Exponential backoff for failed requests
- Fallback to alternative providers
- Comprehensive logging and error reporting

## Performance Considerations

### Cache Performance Optimization

1. **Gzip Compression**: 60-80% storage reduction
2. **Local File System**: Fast local access
3. **Incremental Updates**: Only download missing data
4. **Parallel Processing**: Support for concurrent operations
5. **Memory Efficiency**: Load data on-demand

### Provider Selection Performance

1. **Caching**: Cache provider selection results
2. **Fast Classification**: Efficient symbol type detection
3. **Minimal Overhead**: Lightweight selection logic
4. **Fallback Speed**: Quick fallback to alternative providers

### Data Processing Performance

1. **Vectorized Operations**: Use pandas/numpy for efficiency
2. **Chunked Processing**: Process large datasets in chunks
3. **Memory Management**: Efficient DataFrame operations
4. **Lazy Loading**: Load data only when needed

### Network Optimization

1. **Connection Pooling**: Reuse HTTP connections
2. **Compression**: Gzip compression for data transfer
3. **Rate Limiting**: Respect API limits with built-in throttling
4. **Parallel Downloads**: Concurrent downloads where possible

## Security Considerations

### API Key Management

1. **Environment Variables**: Store sensitive credentials in environment
2. **Key Rotation**: Support for regular API key rotation
3. **Scope Limitation**: Use read-only keys where possible
4. **Audit Logging**: Log API key usage for security monitoring

### Cache Security

1. **Local Storage**: Cache data stored locally on user's system
2. **No Sensitive Data**: Cache contains only market data
3. **Access Control**: Proper file system permissions
4. **Data Integrity**: Validation and checksum verification

### Network Security

1. **HTTPS Only**: All API communications use encrypted connections
2. **Certificate Validation**: Verify SSL certificates for all connections
3. **Rate Limiting**: Prevent abuse of external APIs
4. **Error Handling**: Don't expose sensitive information in errors

## Scalability Design

### Horizontal Scaling

1. **Stateless Design**: Cache operations are stateless
2. **Distributed Cache**: Support for distributed cache systems
3. **Load Distribution**: Provider selection distributes load
4. **Microservice Ready**: Components can be deployed separately

### Vertical Scaling

1. **Multi-threading**: Background threads for data processing
2. **Memory Optimization**: Efficient data structures
3. **CPU Optimization**: Vectorized operations using pandas/numpy
4. **Storage Optimization**: Gzip compression and efficient file formats

### Future Extensibility

1. **Plugin Architecture**: Easy addition of new providers
2. **Protocol Abstraction**: Support for different communication protocols
3. **Data Format Evolution**: Ability to migrate to new formats
4. **Configuration Evolution**: Backward-compatible configuration updates

## Monitoring and Observability

### Cache Monitoring

1. **Size Tracking**: Monitor cache growth over time
2. **Quality Metrics**: Track data quality scores
3. **Provider Performance**: Monitor provider success rates
4. **Storage Efficiency**: Track compression ratios

### Performance Monitoring

1. **Download Times**: Track data download performance
2. **Cache Hit Rates**: Monitor cache effectiveness
3. **Provider Response Times**: Track API performance
4. **Error Rates**: Monitor system reliability

### Health Checks

1. **Cache Validation**: Regular cache integrity checks
2. **Provider Health**: Monitor provider availability
3. **Data Freshness**: Track data age and updates
4. **System Resources**: Monitor disk space and memory usage