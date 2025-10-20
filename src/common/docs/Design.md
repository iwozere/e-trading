# Design Documentation

## Overview
This document describes the architecture, design patterns, and technical implementation of the `src/common` module, which provides unified access to data providers, technical analysis, and fundamental analysis capabilities.

## Architecture Overview

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    src/common Module                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Public    │  │  Internal   │  │   Utility   │         │
│  │  Interface  │  │  Services   │  │  Functions  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Data      │  │ Technical   │  │ Fundamental │         │
│  │ Providers   │  │  Analysis   │  │  Analysis   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Ticker      │  │ Chart       │  │  Testing    │         │
│  │Classifier   │  │ Generation  │  │  Framework  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Module Dependencies
```
src/common/
├── __init__.py              # Public interface
├── (indicator_service.py removed - moved to src/indicators/)
├── recommendation_engine.py # Recommendation system
├── (ticker_classifier.py removed - functionality moved to ProviderSelector)
├── fundamentals.py          # Fundamental analysis
├── technicals.py           # Technical analysis (legacy)
├── ticker_analyzer.py      # Complete ticker analysis
├── ticker_chart.py         # Chart generation
├── docs/                   # Documentation
└── tests/                  # Test suite
```

## Design Patterns

### 1. Factory Pattern
**Location**: `src/common/__init__.py`
**Purpose**: Create appropriate data providers based on ticker characteristics

```python
def determine_provider(ticker: str) -> str:
    from src.data.data_manager import ProviderSelector
    selector = ProviderSelector()
    ticker_info = selector.get_ticker_info(ticker)
    if ticker_info['symbol_type'] == 'crypto':
        return "bnc"
    elif ticker_info['symbol_type'] == 'stock':
        return "yf"
    return "yf"  # Default fallback
```

### 2. Strategy Pattern
**Location**: `src/data/data_manager.py` (ProviderSelector)
**Purpose**: Different classification strategies for different asset types

```python
class ProviderSelector:
    def __init__(self):
        self.rules = self._load_provider_rules()
        self.symbol_patterns = self._load_symbol_patterns()
```

### 3. Singleton Pattern
**Location**: `src/indicators/service.py` (moved from src/common/)
**Purpose**: Ensure single instance of indicator service with shared cache

```python
_indicator_service = None

def get_indicator_service() -> IndicatorService:
    global _indicator_service
    if _indicator_service is None:
        _indicator_service = IndicatorService()
    return _indicator_service
```

### 4. Template Method Pattern
**Location**: `src/common/recommendation_engine.py`
**Purpose**: Standardized recommendation generation process

```python
class RecommendationEngine:
    def get_recommendation(self, indicator: str, value: float, context: Dict = None):
        # Template method for recommendation generation
        if indicator in self.technical_functions:
            return self.technical_functions[indicator](value, context)
        elif indicator in self.fundamental_functions:
            return self.fundamental_functions[indicator](value, context)
```

### 5. Decorator Pattern
**Location**: `src/indicators/service.py` (moved from src/common/)
**Purpose**: Add caching functionality to indicator calculations

```python
class SimpleMemoryCache:
    def get(self, ticker: str, indicators: List[str], timeframe: str, period: str, **kwargs):
        # Cache decorator functionality
        key = self._generate_key(ticker, indicators, timeframe, period, **kwargs)
        if key in self.cache:
            return self.cache[key]
        return None
```

## Component Design

### 1. Ticker Classifier

#### Purpose
Intelligently classify tickers and determine appropriate data providers based on ticker characteristics.

#### Design Decisions
- **Pattern Matching**: Use regex patterns for efficient classification
- **Extensibility**: Easy to add new patterns and exchanges
- **Fallback Strategy**: Default to Yahoo Finance for unknown tickers

#### Key Components
```python
@dataclass
class TickerInfo:
    original_ticker: str
    provider: DataProvider
    formatted_ticker: str
    exchange: Optional[str] = None
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None

class ProviderSelector:
    def get_ticker_info(self, ticker: str) -> Dict
    def get_data_provider_config(self, ticker: str) -> Dict
    def get_best_provider(self, ticker: str, interval: str) -> str
```

#### Supported Patterns
- **Crypto**: BTCUSDT, ETHUSD, ADABTC, etc.
- **US Stocks**: AAPL, GOOGL, MSFT, etc.
- **International**: VUSD.L, NESN.SW, BMW.DE, etc.
- **Special Cases**: BRK.A, BRK.B, etc.

### 2. Unified Indicator Service

#### Purpose
Provide a single, unified interface for calculating technical and fundamental indicators with intelligent caching.

#### Design Decisions
- **TA-Lib Integration**: Direct use of TA-Lib for performance
- **Memory Caching**: Simple in-memory cache with TTL
- **Parameter Awareness**: Cache keys include all calculation parameters
- **Batch Processing**: Support for concurrent multi-ticker analysis

#### Architecture
```python
class IndicatorService:
    def __init__(self):
        self.cache = SimpleMemoryCache()
        self.recommendation_engine = RecommendationEngine()
    
    async def calculate_indicators(self, request: IndicatorCalculationRequest) -> IndicatorSet
    async def calculate_batch_indicators(self, request: BatchIndicatorRequest) -> List[IndicatorSet]
```

#### Cache Strategy
- **Key Generation**: Include ticker, indicators, timeframe, period, and parameters
- **TTL**: 5 minutes default, configurable
- **LRU Eviction**: When cache reaches maximum size
- **Parameter Awareness**: Different parameters = different cache entries

### 3. Recommendation Engine

#### Purpose
Generate consistent recommendations for technical and fundamental indicators with confidence scores and reasoning.

#### Design Decisions
- **Unified Interface**: Single engine for all indicator types
- **Confidence Scoring**: 0.0-1.0 scale for recommendation strength
- **Reasoning**: Human-readable explanations for recommendations
- **Extensibility**: Easy to add new indicators and rules

#### Architecture
```python
class RecommendationEngine:
    def __init__(self):
        self.technical_rules = TechnicalRecommendationRules()
        self.fundamental_rules = FundamentalRecommendationRules()
        self.technical_functions = {...}  # Mapping of indicator names to functions
        self.fundamental_functions = {...}

class TechnicalRecommendationRules:
    @staticmethod
    def get_rsi_recommendation(value: float) -> Tuple[RecommendationType, float, str]
    @staticmethod
    def get_macd_recommendation(macd: float, signal: float, macd_hist: float) -> Tuple[RecommendationType, float, str]
```

#### Recommendation Types
- **STRONG_BUY**: High confidence buy signal
- **BUY**: Moderate confidence buy signal
- **HOLD**: Neutral position
- **SELL**: Moderate confidence sell signal
- **STRONG_SELL**: High confidence sell signal

### 4. Fundamental Analysis

#### Purpose
Retrieve and normalize fundamental data from multiple providers with intelligent fallback mechanisms.

#### Design Decisions
- **Multi-Provider Support**: Yahoo Finance, Alpha Vantage, Finnhub, etc.
- **Priority-Based Selection**: YF > AV > FH > TD > PG
- **Data Normalization**: Consistent schema across providers
- **Source Attribution**: Track which provider provided each field

#### Architecture
```python
def get_fundamentals(ticker: str, provider: str = None, **kwargs) -> Fundamentals:
    if provider:
        # Single provider path
        downloader = DataDownloaderFactory.create_downloader(provider, **kwargs)
        result = downloader.get_fundamentals(ticker)
        return normalize_fundamentals({provider: result})
    else:
        # Multi-provider path with fallback
        provider_results = {}
        for code in PROVIDER_CODES:
            try:
                downloader = DataDownloaderFactory.create_downloader(code, **kwargs)
                if downloader:
                    result = downloader.get_fundamentals(ticker)
                    provider_results[code] = result
            except Exception:
                continue
        return normalize_fundamentals(provider_results)
```

#### Data Normalization
- **Field Mapping**: Map provider-specific fields to standard schema
- **Type Conversion**: Ensure consistent data types
- **Unit Standardization**: Normalize currencies, percentages, etc.
- **Missing Data Handling**: Graceful handling of unavailable fields

### 5. Technical Analysis (Legacy)

#### Purpose
Calculate technical indicators using TA-Lib with legacy interface for backward compatibility.

#### Design Decisions
- **TA-Lib Integration**: Direct use of TA-Lib functions
- **DataFrame Integration**: Add indicator columns to existing DataFrames
- **Error Handling**: Graceful handling of insufficient data
- **Legacy Support**: Maintain compatibility with existing code

#### Architecture
```python
def calculate_technicals_from_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Technicals]:
    # Calculate all indicators
    df['rsi'] = talib.RSI(df['close'].values)
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
    # ... more indicators
    
    # Create Technicals object
    technicals = Technicals(
        rsi=df['rsi'].iloc[-1],
        macd=df['macd'].iloc[-1],
        # ... more fields
    )
    
    return df, technicals
```

### 6. Chart Generation

#### Purpose
Generate comprehensive technical analysis charts with multiple indicators and professional styling.

#### Design Decisions
- **Matplotlib Integration**: Use matplotlib for chart generation
- **Multi-Panel Layout**: Separate panels for different indicator types
- **Professional Styling**: Consistent color scheme and formatting
- **Error Handling**: Graceful handling of missing data

#### Architecture
```python
def generate_chart(analysis: TickerAnalysis) -> bytes:
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(6, 1, height_ratios=[3, 1, 1, 1, 1, 1], hspace=0.4)
    
    # 1. Main price chart with Bollinger Bands, SMA 50, SMA 200
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, close, label='Close Price', color='black', linewidth=1)
    ax1.plot(dates, bb_upper, label='BB Upper', color='gray', alpha=0.7)
    ax1.plot(dates, bb_middle, label='BB Middle', color='blue', alpha=0.7)
    ax1.plot(dates, bb_lower, label='BB Lower', color='gray', alpha=0.7)
    ax1.plot(dates, sma_fast, label='SMA Fast', color='orange', alpha=0.8)
    ax1.plot(dates, sma_slow, label='SMA Slow', color='darkblue', alpha=0.8)
    
    # 2. RSI with 30/70 lines
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, rsi, label='RSI', color='purple', linewidth=1)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    
    # 3. MACD (MACD, Signal, Histogram)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(dates, macd, label='MACD', color='red', linewidth=1)
    ax3.plot(dates, macd_signal, label='Signal', color='blue', linewidth=1)
    ax3.bar(dates, macd_hist, label='Histogram', color='gray', alpha=0.5)
    
    # 4. Stochastic
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(dates, stoch_k, label='%K', color='purple', linewidth=1)
    ax4.plot(dates, stoch_d, label='%D', color='orange', linewidth=1)
    ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought')
    ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold')
    
    # 5. ADX (ADX, +DI, -DI, Trend Threshold)
    ax5 = fig.add_subplot(gs[4])
    ax5.plot(dates, adx, label='ADX', color='teal', linewidth=1)
    ax5.plot(dates, plus_di, label='+DI', color='green', linewidth=1)
    ax5.plot(dates, minus_di, label='-DI', color='red', linewidth=1)
    ax5.axhline(y=25, color='gray', linestyle='--', alpha=0.7, label='Trend Threshold')
    
    # 6. Volume (OBV and volume histogram)
    ax6 = fig.add_subplot(gs[5])
    ax6.plot(dates, obv, label='OBV', color='yellow', linewidth=1)
    ax6.bar(dates, volume, label='Volume', color='blue', alpha=0.3)
    
    # Convert to bytes
    buffer = io.BytesIO()
    # Chart generation returns bytes directly, no file creation
img_buffer = io.BytesIO()
plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
img_buffer.seek(0)
chart_bytes = img_buffer.getvalue()
    return buffer.getvalue()
```

## Data Flow

### 1. Single Ticker Analysis Flow
```
User Request → Ticker Classification → Provider Selection → Data Retrieval → 
Indicator Calculation → Recommendation Generation → Chart Generation → Response
```

### 2. Batch Analysis Flow
```
Batch Request → Ticker Classification (Batch) → Provider Grouping → 
Concurrent Data Retrieval → Concurrent Indicator Calculation → 
Recommendation Generation → Response Aggregation
```

### 3. Caching Flow
```
Request → Cache Check → Cache Hit? → Yes: Return Cached Data
                                    No: Calculate → Cache Result → Return Data
```

## Error Handling Strategy

### 1. Provider Failures
- **Automatic Fallback**: Try alternative providers
- **Graceful Degradation**: Return partial data if possible
- **Error Logging**: Comprehensive logging for debugging

### 2. Data Quality Issues
- **Validation**: Check data integrity before processing
- **Default Values**: Use sensible defaults for missing data
- **Quality Indicators**: Flag low-quality data

### 3. Calculation Errors
- **Parameter Validation**: Validate input parameters
- **Data Sufficiency**: Check minimum data requirements
- **Exception Handling**: Catch and handle calculation errors

### 4. Debug Output Management
- **Clean Logging**: Streamlined logging without verbose debug output
- **Production Focus**: Optimized for production environments
- **No File Pollution**: Chart generation doesn't create files in project root

## Performance Considerations

### 1. Caching Strategy
- **Memory Cache**: Fast access for frequently requested data
- **Parameter Awareness**: Different parameters = different cache entries
- **TTL Management**: Automatic expiration of stale data
- **LRU Eviction**: Efficient memory management

### 2. Batch Processing
- **Concurrent Execution**: Process multiple tickers simultaneously
- **Resource Pooling**: Reuse connections and resources
- **Load Balancing**: Distribute load across providers

### 3. Memory Management
- **DataFrame Optimization**: Efficient data structures
- **Garbage Collection**: Proper cleanup of temporary objects
- **Memory Monitoring**: Track memory usage

## Security Considerations

### 1. Input Validation
- **Ticker Validation**: Validate ticker symbols
- **Parameter Sanitization**: Sanitize all input parameters
- **Type Checking**: Ensure correct data types

### 2. API Key Management
- **Secure Storage**: Store API keys securely
- **Access Control**: Limit access to sensitive operations
- **Audit Logging**: Log all API key usage

### 3. Data Protection
- **Encryption**: Encrypt sensitive data in transit
- **Access Control**: Control access to financial data
- **Compliance**: Ensure regulatory compliance

## Testing Strategy

### 1. Unit Testing
- **Component Testing**: Test individual components
- **Mock Dependencies**: Mock external dependencies
- **Edge Cases**: Test boundary conditions

### 2. Integration Testing
- **Provider Integration**: Test with real providers
- **End-to-End Testing**: Test complete workflows
- **Performance Testing**: Test under load

### 3. Test Coverage
- **Code Coverage**: >90% coverage target
- **Scenario Coverage**: Test all major scenarios
- **Error Coverage**: Test error conditions

## Recent Improvements

### Debug Output Cleanup
- **Streamlined Logging**: Removed verbose debug output for cleaner production logs
- **No File Pollution**: Chart generation no longer creates files in project root directory
- **Memory Efficient**: Charts returned as bytes for direct use in applications
- **Production Ready**: Optimized for production environments with minimal logging overhead

### Chart Generation Updates
- **Updated Function Signature**: `generate_chart(ticker, df)` instead of `generate_chart(analysis)`
- **Direct Byte Return**: Charts returned as bytes for immediate use
- **No Side Effects**: Chart generation is pure function with no file system side effects
- **Better Error Handling**: Improved error handling for chart generation failures

## Future Enhancements

### 1. Performance Improvements
- **Redis Caching**: Add Redis for distributed caching
- **Async Processing**: Improve async capabilities
- **Parallel Processing**: Add parallel indicator calculations

### 2. Feature Additions
- **More Indicators**: Add additional technical indicators
- **Custom Indicators**: Support user-defined indicators
- **Real-time Updates**: Add real-time data streaming

### 3. Scalability Improvements
- **Microservices**: Split into microservices
- **Load Balancing**: Add load balancing capabilities
- **Horizontal Scaling**: Support horizontal scaling
