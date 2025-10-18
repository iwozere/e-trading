# Data Management Module

## Purpose & Responsibilities

The Data Management module serves as the comprehensive data infrastructure foundation for the Advanced Trading Framework. It provides unified access to historical and real-time market data from multiple providers, intelligent caching, and seamless integration with trading strategies and analytics components.

## 🔗 Quick Navigation
- **[📖 Documentation Index](../INDEX.md)** - Complete documentation guide
- **[🏗️ System Architecture](../README.md)** - Overall system overview
- **[📊 Database Schema](../database-architecture.md)** - Data models and relationships
- **[🔌 Data Providers](../data-providers-sources.md)** - External data source integration
- **[📈 Trading Engine](trading-engine.md)** - Strategy execution and data consumption
- **[🧠 ML & Analytics](ml-analytics.md)** - Advanced analytics and feature engineering

## 🔄 Related Modules
| Module | Relationship | Integration Points |
|--------|--------------|-------------------|
| **[Trading Engine](trading-engine.md)** | Data Consumer | Market data feeds, strategy signals |
| **[ML & Analytics](ml-analytics.md)** | Data Consumer | Feature engineering, model training data |
| **[Infrastructure](infrastructure.md)** | Service Provider | Database access, caching, error handling |
| **[Configuration](configuration.md)** | Configuration | Provider settings, cache configuration |

**Core Responsibilities:**
- **Unified Data Access**: Single entry point for all data operations through DataManager facade
- **Multi-Provider Support**: Integration with 10+ data providers (Binance, Yahoo Finance, FMP, Alpha Vantage, etc.)
- **Intelligent Provider Selection**: Automatic selection of optimal data provider based on symbol type and timeframe
- **Unified Caching System**: Efficient file-based caching with gzip compression and metadata tracking
- **Real-time Data Feeds**: WebSocket-based live data streaming for multiple exchanges
- **Data Validation**: Comprehensive OHLCV data quality checks and scoring
- **Fundamentals Management**: Multi-provider fundamentals data with TTL-based caching

## Key Components

### 1. DataManager (Main Facade)

The `DataManager` class serves as the unified entry point for all data operations, implementing the facade pattern to simplify complex data retrieval workflows.

```python
from src.data import DataManager, get_data_manager

# Get singleton instance
data_manager = get_data_manager()

# Historical data with intelligent provider selection
df = data_manager.get_ohlcv("BTCUSDT", "5m", start_date, end_date)

# Fundamentals data with caching
fundamentals = data_manager.get_fundamentals("AAPL")

# Live data feed creation
live_feed = data_manager.get_live_feed("BTCUSDT", "5m")
```

**Key Features:**
- Automatic provider selection based on symbol type and timeframe
- Transparent caching with cache-first strategy
- Comprehensive error handling with provider failover
- Rate limiting and retry logic for all providers
- Data validation and quality scoring

### 2. ProviderSelector (Intelligent Selection)

The `ProviderSelector` implements configuration-driven provider selection with sophisticated symbol classification and compatibility checking.

```python
selector = ProviderSelector()

# Automatic provider selection
provider = selector.get_best_provider("BTCUSDT", "5m")  # Returns "binance"
provider = selector.get_best_provider("AAPL", "1d")     # Returns "yahoo"

# Provider sequence with failover
providers = selector.get_provider_with_failover("AAPL", "5m")
# Returns ["fmp", "alpaca", "alpha_vantage", "polygon"]

# Symbol classification
symbol_info = selector.classify_symbol("BTCUSDT")  # Returns "crypto"
```

**Selection Logic:**
- **Cryptocurrency Symbols**: Primary: Binance, Backup: CoinGecko, Alpha Vantage
- **Stock Symbols (Intraday)**: Primary: FMP, Backup: Alpaca, Alpha Vantage, Polygon
- **Stock Symbols (Daily)**: Primary: Yahoo Finance, Backup: Alpaca, Tiingo, FMP
- **International Symbols**: Enhanced support with provider compatibility checking

### 3. UnifiedCache (Simplified Architecture)

The `UnifiedCache` implements a streamlined caching system with year-based file splitting and gzip compression.

```
data-cache/
├── ohlcv/
│   ├── BTCUSDT/
│   │   ├── 5m/
│   │   │   ├── 2025.csv.gz          # Compressed data
│   │   │   ├── 2025.metadata.json   # Provider & quality info
│   │   │   └── 2024.csv.gz
│   │   └── 1h/
│   └── AAPL/
│       ├── 5m/
│       └── 1d/
└── fundamentals/
    └── AAPL/
        ├── yfinance_20250106_143022.json
        └── fmp_20250106_143045.json
```

**Key Features:**
- **Simplified Structure**: `symbol/timeframe/` instead of provider-based directories
- **Gzip Compression**: 60-80% storage reduction for CSV files
- **Year Splitting**: Automatic data splitting by year for efficient access
- **Provider Metadata**: Data source information embedded in metadata files
- **Quality Tracking**: Data quality scores and validation results

### 4. Data Downloaders (Provider Implementations)

All data downloaders inherit from `BaseDataDownloader` and implement provider-specific data retrieval logic.

**Available Providers:**

#### Binance Data Downloader
- **Best for**: Cryptocurrency data with comprehensive historical coverage
- **Capabilities**: All major crypto pairs, multiple timeframes (1m-1M), high rate limits (1200/min)
- **Data Quality**: Excellent - direct from exchange

#### Yahoo Finance Data Downloader  
- **Best for**: Stock fundamental data and daily historical data
- **Capabilities**: Global stocks/ETFs, comprehensive fundamentals, no API key required
- **Limitation**: Only 60 days of intraday data

#### Alpha Vantage Data Downloader
- **Best for**: Full historical intraday stock data (no 60-day limit)
- **Capabilities**: Complete intraday history, multiple intervals, professional-grade data
- **Rate Limits**: 5 calls/minute, 500/day (free tier)

#### FMP Data Downloader
- **Best for**: Professional-grade financial data with comprehensive coverage
- **Capabilities**: Global stocks/ETFs, high rate limits (3000/min), real-time data
- **Data Quality**: Excellent - professional-grade financial data

#### Alpaca Data Downloader
- **Best for**: Professional-grade US market data with trading integration
- **Capabilities**: US stocks/ETFs, exchange-sourced data, 10,000 bars per request
- **Rate Limits**: 200 requests/minute (free tier)

### 5. Live Data Feeds (Real-time Streaming)

Live data feeds provide real-time market data through WebSocket connections with automatic historical data backfilling.

```python
# Create live feed
feed = data_manager.get_live_feed("BTCUSDT", "5m", lookback_bars=1000)

# Feed automatically:
# 1. Loads 1000 historical bars via DataManager
# 2. Establishes WebSocket connection
# 3. Streams real-time updates
# 4. Handles reconnection on failures
```

**Available Live Feeds:**
- **BinanceLiveDataFeed**: Real-time crypto data via WebSocket
- **YahooLiveDataFeed**: Real-time stock quotes
- **IBKRLiveDataFeed**: Professional trading data
- **CoinGeckoLiveDataFeed**: Alternative crypto data source

### 6. Fundamentals Cache System

The fundamentals cache provides TTL-based caching for fundamental data with multi-provider support and intelligent data combination.

```python
# Automatic provider selection and caching
fundamentals = data_manager.get_fundamentals("AAPL")

# Multi-provider data combination
fundamentals = data_manager.get_fundamentals(
    "AAPL", 
    providers=["fmp", "yahoo", "alpha_vantage"],
    combination_strategy="priority_based"
)
```

**Features:**
- **TTL-based Caching**: 7-day default TTL with data-type specific rules
- **Multi-provider Combination**: Priority-based field selection from multiple sources
- **Automatic Cleanup**: Stale data removal when new data is cached
- **Provider Priority**: FMP > Yahoo Finance > Alpha Vantage > others

## Architecture Patterns

### 1. Facade Pattern (DataManager)
The DataManager implements the facade pattern to provide a simplified interface to the complex data retrieval subsystem, hiding the complexity of provider selection, caching, and error handling.

### 2. Strategy Pattern (Provider Selection)
Provider selection uses the strategy pattern with configuration-driven rules, allowing dynamic selection of optimal data providers based on symbol characteristics and requirements.

### 3. Repository Pattern (Cache Access)
The cache system implements the repository pattern with consistent data access methods across different storage mechanisms (OHLCV cache, fundamentals cache).

### 4. Observer Pattern (Live Feeds)
Live data feeds implement the observer pattern with callback mechanisms for real-time data updates and integration with trading strategies.

### 5. Factory Pattern (Provider Creation)
Data downloader and live feed creation uses the factory pattern with automatic initialization and configuration based on provider requirements.

## Integration Points

### With Trading Engine
- **Historical Data**: Provides OHLCV data for backtesting and strategy analysis
- **Real-time Feeds**: Streams live market data to trading bots and strategies
- **Data Validation**: Ensures data quality for trading decisions

### With ML & Analytics
- **Feature Engineering**: Provides clean, validated data for ML model training
- **Fundamentals Data**: Supplies fundamental metrics for quantitative analysis
- **Data Pipeline**: Supports batch processing and model training workflows

### With Configuration System
- **Provider Rules**: Loads provider selection rules from YAML configuration
- **Cache Settings**: Configures cache directories and size limits
- **API Keys**: Manages provider API credentials and authentication

### With Notification System
- **Error Reporting**: Reports data retrieval failures and quality issues
- **Cache Statistics**: Provides cache usage and performance metrics
- **Provider Status**: Monitors provider availability and performance

## Data Models

### OHLCV Data Model
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

### Fundamentals Data Model
```python
{
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "market_cap": 3000000000000,
    "pe_ratio": 25.5,
    "revenue_growth": 0.08,
    "provider_info": {
        "primary_source": "fmp",
        "combined_from": ["fmp", "yahoo"],
        "last_updated": "2025-01-15T10:30:00"
    }
}
```

### Cache Metadata Model
```python
{
    "symbol": "BTCUSDT",
    "timeframe": "5m",
    "year": 2025,
    "data_source": "binance",
    "created_at": "2025-01-15T10:30:00",
    "data_quality": {
        "score": 0.95,
        "validation_errors": [],
        "gaps": 0,
        "duplicates": 0
    },
    "provider_info": {
        "name": "binance",
        "reliability": 0.95,
        "rate_limit": "1200/minute"
    }
}
```

## Roadmap & Feature Status

### ✅ Implemented Features (Q3-Q4 2024)
- **DataManager Facade**: Complete unified interface for all data operations
- **Multi-Provider Support**: 10+ data providers with automatic initialization
- **Intelligent Provider Selection**: Configuration-driven selection with symbol classification
- **Unified Cache System**: Simplified structure with gzip compression and metadata
- **Live Data Feeds**: WebSocket-based real-time streaming for major exchanges
- **Data Validation**: Comprehensive OHLCV validation and quality scoring
- **Fundamentals Caching**: TTL-based caching with multi-provider combination
- **Error Handling**: Comprehensive retry logic with exponential backoff
- **Rate Limiting**: Provider-specific rate limiting and throttling

### 🔄 In Progress (Q1 2025)
- **International Symbol Support**: Enhanced support for non-US markets (Target: Jan 2025)
- **Data Pipeline System**: Multi-step processing pipeline for data transformation (Target: Feb 2025)
- **Cache Optimization**: Performance improvements for large datasets (Target: Jan 2025)
- **Provider Monitoring**: Real-time provider health and performance tracking (Target: Mar 2025)

### 📋 Planned Enhancements

#### Q2 2025 - Advanced Data Infrastructure
- **Database Integration**: Optional PostgreSQL storage for high-frequency data
  - Timeline: April-May 2025
  - Benefits: Better query performance, ACID compliance, advanced indexing
  - Dependencies: Infrastructure module database optimization

- **Data Streaming**: Apache Kafka integration for real-time data distribution
  - Timeline: May-June 2025
  - Benefits: Scalable real-time data distribution, event sourcing
  - Dependencies: Infrastructure upgrade, containerization

#### Q3 2025 - Analytics & Intelligence
- **Advanced Analytics**: Built-in technical indicators and feature engineering
  - Timeline: July-August 2025
  - Benefits: Reduced ML pipeline complexity, standardized features
  - Dependencies: ML & Analytics module integration

- **Data Lineage**: Complete data provenance and audit trail tracking
  - Timeline: August-September 2025
  - Benefits: Regulatory compliance, debugging capabilities
  - Dependencies: Database schema updates, audit logging

#### Q4 2025 - Cloud & Scale
- **Cloud Storage**: S3/Azure Blob storage support for large-scale caching
  - Timeline: October-November 2025
  - Benefits: Unlimited storage, cost optimization, disaster recovery
  - Dependencies: Cloud infrastructure setup

- **Multi-Region Support**: Geographic data distribution and failover
  - Timeline: November-December 2025
  - Benefits: Reduced latency, improved reliability
  - Dependencies: Cloud storage, infrastructure scaling

### Migration & Evolution Strategy

#### Phase 1: Performance Optimization (Q1 2025)
- **Current State**: File-based caching with good performance
- **Target State**: Optimized caching with database integration option
- **Migration Path**: 
  - Implement database storage as optional feature
  - Gradual migration of high-frequency symbols to database
  - Maintain file-based cache for backward compatibility
- **Backward Compatibility**: Full compatibility maintained

#### Phase 2: Streaming Architecture (Q2 2025)
- **Current State**: Pull-based data retrieval with caching
- **Target State**: Hybrid pull/push architecture with streaming
- **Migration Path**:
  - Implement Kafka infrastructure alongside existing system
  - Migrate real-time feeds to streaming architecture
  - Maintain REST API compatibility for historical data
- **Backward Compatibility**: Existing APIs remain functional

#### Phase 3: Cloud-Native (Q3-Q4 2025)
- **Current State**: Single-server deployment with local storage
- **Target State**: Cloud-native with distributed storage and processing
- **Migration Path**:
  - Implement cloud storage as additional cache tier
  - Gradual migration of data to cloud storage
  - Implement multi-region failover capabilities
- **Backward Compatibility**: Local deployment option maintained

### Version History & Updates

| Version | Release Date | Key Features | Breaking Changes |
|---------|--------------|--------------|------------------|
| **1.0.0** | Oct 2024 | Initial release with basic data management | N/A |
| **1.1.0** | Nov 2024 | Added live data feeds, improved caching | None |
| **1.2.0** | Dec 2024 | Multi-provider support, intelligent selection | Configuration format changes |
| **1.3.0** | Jan 2025 | International symbols, cache optimization | None (planned) |
| **2.0.0** | Q2 2025 | Database integration, streaming support | Cache API changes (planned) |
| **3.0.0** | Q4 2025 | Cloud-native, multi-region support | Deployment changes (planned) |

### Deprecation Timeline

#### Deprecated Features
- **Legacy Cache Format** (Deprecated: Dec 2024, Removed: Jun 2025)
  - Reason: Performance improvements and metadata enhancements
  - Migration: Automatic conversion on first access
  - Impact: Minimal - transparent to users

#### Future Deprecations
- **File-Only Caching** (Deprecation: Q3 2025, Removal: Q1 2026)
  - Reason: Database integration provides better performance and features
  - Migration: Gradual migration tools provided
  - Impact: Configuration changes required for optimal performance

## Configuration

### Provider Selection Rules
```yaml
# config/data/provider_rules.yaml
crypto:
  primary: binance
  backup: [coingecko, alpha_vantage]
  timeframes: [1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w, 1M]

stock_intraday:
  primary: fmp
  backup: [alpaca, alpha_vantage, polygon]
  timeframes: [1m, 5m, 15m, 30m, 1h, 4h]

stock_daily:
  primary: yahoo
  backup: [alpaca, tiingo, fmp]
  timeframes: [1d, 1w, 1M]
```

### Environment Variables
```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_api_key
FMP_API_KEY=your_api_key
POLYGON_KEY=your_api_key
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key

# Cache Configuration
DATA_CACHE_DIR=c:/data-cache
CACHE_MAX_SIZE_GB=20.0
```

### Cache Configuration
```python
# Configure cache with custom settings
cache = UnifiedCache(
    cache_dir="./my-cache",
    max_size_gb=20.0
)

data_manager = DataManager(
    cache_dir="./my-cache",
    config_path="./my-provider-rules.yaml"
)
```

## Performance Characteristics

### Cache Performance
- **Storage Efficiency**: 60-80% reduction with gzip compression
- **Access Speed**: Local file system access with sub-second retrieval
- **Scalability**: Handles millions of data points with year-based splitting
- **Memory Usage**: Efficient DataFrame operations with lazy loading

### Provider Performance
- **Binance**: 1200 requests/minute, excellent for crypto data
- **Yahoo Finance**: No rate limits, good for stock fundamentals
- **Alpha Vantage**: 5 requests/minute, comprehensive historical data
- **FMP**: 3000 requests/minute (paid), professional-grade data

### Network Optimization
- **Connection Pooling**: Reuses HTTP connections for efficiency
- **Compression**: Gzip compression for data transfer
- **Parallel Downloads**: Concurrent requests where supported
- **Rate Limiting**: Built-in throttling to respect API limits

## Error Handling & Resilience

### Provider Failover
- **Automatic Failover**: Seamless switching to backup providers on failure
- **Error Classification**: Intelligent error type detection for appropriate handling
- **Retry Logic**: Exponential backoff with jitter for transient failures
- **Circuit Breaker**: Temporary provider disabling on repeated failures

### Data Quality Assurance
- **Validation Pipeline**: Multi-stage data validation before caching
- **Quality Scoring**: Numerical quality assessment for all data
- **Gap Detection**: Identification and handling of missing data periods
- **Duplicate Removal**: Automatic deduplication of overlapping data

### Cache Resilience
- **Corruption Recovery**: Automatic detection and recovery from corrupted cache files
- **Partial Data Handling**: Graceful handling of incomplete data sets
- **Metadata Consistency**: Automatic metadata repair and validation
- **Storage Monitoring**: Disk space monitoring with automatic cleanup

## Testing Strategy

### Unit Tests
- **Provider Tests**: Individual provider functionality and error handling
- **Cache Tests**: Cache operations, compression, and metadata management
- **Validation Tests**: Data quality checks and scoring algorithms
- **Selection Tests**: Provider selection logic and symbol classification

### Integration Tests
- **End-to-End Tests**: Complete data retrieval workflows with real providers
- **Cache Integration**: Cache consistency across multiple operations
- **Live Feed Tests**: Real-time data streaming and reconnection logic
- **Multi-Provider Tests**: Provider failover and data combination

### Performance Tests
- **Load Tests**: High-volume data retrieval and caching performance
- **Stress Tests**: System behavior under resource constraints
- **Benchmark Tests**: Provider performance comparison and optimization
- **Memory Tests**: Memory usage patterns and leak detection

## Monitoring & Observability

### Metrics Collection
- **Provider Metrics**: Success rates, response times, error rates
- **Cache Metrics**: Hit rates, storage usage, compression ratios
- **Data Quality Metrics**: Validation scores, gap frequencies, error patterns
- **Performance Metrics**: Throughput, latency, resource utilization

### Logging Strategy
- **Structured Logging**: JSON-formatted logs with consistent fields
- **Log Levels**: Appropriate use of DEBUG, INFO, WARNING, ERROR levels
- **Context Preservation**: Request tracing across provider calls
- **Error Aggregation**: Centralized error collection and analysis

### Health Checks
- **Provider Health**: Regular connectivity and response time checks
- **Cache Health**: Storage space, file integrity, metadata consistency
- **Data Freshness**: Monitoring of data age and update frequencies
- **System Resources**: Memory usage, disk space, network connectivity

---

**Module Version**: 1.3.0  
**Last Updated**: January 15, 2025  
**Next Review**: February 15, 2025  
**Owner**: Data Team  
**Dependencies**: [Infrastructure](infrastructure.md), [Configuration](configuration.md)  
**Used By**: [Trading Engine](trading-engine.md), [ML & Analytics](ml-analytics.md), [Communication](communication.md)