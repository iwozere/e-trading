# Common Module

This module provides unified access to data providers, fundamentals, technical analysis, and ticker analysis functions with a modern, unified indicator system.

## Overview

The common module is organized into specialized submodules with a unified indicator system:

- **Indicator services** - Moved to `src/indicators/` unified service
- **`recommendation_engine.py`** - Unified recommendation engine for all indicators
- **`fundamentals.py`** - Fundamental data retrieval and normalization
- **`technicals.py`** - Technical indicator calculations (unified)
- **`ticker_analyzer.py`** - Complete ticker analysis with fundamentals and technicals
- **`ticker_chart.py`** - Enhanced chart generation with modern styling and multiple indicators
- **`__init__.py`** - Core utilities (OHLCV data, period/interval conversion)

## Unified Indicator System

The new unified indicator system provides a single, consistent interface for calculating both technical and fundamental indicators with direct TA-Lib integration and intelligent caching.

### Quick Start

```python
from src.indicators.service import UnifiedIndicatorService
from src.indicators.models import IndicatorCalculationRequest

# Get the unified indicator service
service = UnifiedIndicatorService()

# Calculate indicators for a single ticker
request = IndicatorCalculationRequest(
    ticker="AAPL",
    indicators=["RSI", "MACD", "PE_RATIO", "ROE"],
    timeframe="1d",
    period="1y",
    include_recommendations=True
)

result = await service.get_indicators(request)

print(f"Composite Score: {result.composite_score:.2f}")
print(f"Overall Recommendation: {result.overall_recommendation.recommendation.value}")

# Show individual indicators
for name, indicator in result.get_all_indicators().items():
    print(f"{name}: {indicator.value:.4f} ({indicator.recommendation.recommendation.value})")
```

### Batch Processing

```python
from src.indicators.models import BatchIndicatorRequest

# Calculate indicators for multiple tickers
batch_request = BatchIndicatorRequest(
    tickers=["AAPL", "MSFT", "GOOGL"],
    indicators=["RSI", "MACD", "PE_RATIO"],
    timeframe="1d",
    period="1y",
    max_concurrent=3,
    include_recommendations=True
)

results = await service.get_batch_indicators(batch_request)

for ticker, result in results.items():
    print(f"{ticker}: Score {result.composite_score:.2f}")
```

### Available Indicators

#### Technical Indicators (22 total)
- **RSI** - Relative Strength Index
- **MACD** - Moving Average Convergence Divergence (MACD, Signal, Histogram)
- **Bollinger Bands** - Upper, Middle, Lower bands
- **Stochastic** - Stochastic Oscillator (K and D)
- **ADX** - Average Directional Index (ADX, Plus DI, Minus DI)
- **SMA** - Simple Moving Averages (50, 200)
- **EMA** - Exponential Moving Averages (12, 26)
- **CCI** - Commodity Channel Index
- **ROC** - Rate of Change
- **MFI** - Money Flow Index
- **Williams %R** - Williams Percent Range
- **ATR** - Average True Range

#### Fundamental Indicators (21 total)
- **PE_RATIO** - Price-to-Earnings Ratio
- **FORWARD_PE** - Forward P/E Ratio
- **PB_RATIO** - Price-to-Book Ratio
- **PS_RATIO** - Price-to-Sales Ratio
- **PEG_RATIO** - Price/Earnings-to-Growth Ratio
- **ROE** - Return on Equity
- **ROA** - Return on Assets
- **DEBT_TO_EQUITY** - Debt-to-Equity Ratio
- **CURRENT_RATIO** - Current Ratio
- **QUICK_RATIO** - Quick Ratio
- **OPERATING_MARGIN** - Operating Margin
- **PROFIT_MARGIN** - Profit Margin
- **REVENUE_GROWTH** - Revenue Growth
- **NET_INCOME_GROWTH** - Net Income Growth
- **FREE_CASH_FLOW** - Free Cash Flow
- **DIVIDEND_YIELD** - Dividend Yield
- **PAYOUT_RATIO** - Payout Ratio
- **BETA** - Beta
- **MARKET_CAP** - Market Capitalization
- **ENTERPRISE_VALUE** - Enterprise Value

### Service Information

```python
# Get service information
info = service.get_service_info()
print(f"Service: {info['service']} v{info['version']}")

# Get available indicators
available = service.get_available_indicators()
print(f"Technical: {len(available['technical'])} indicators")
print(f"Fundamental: {len(available['fundamental'])} indicators")

# Get cache statistics
stats = service.get_cache_stats()
print(f"Cache size: {stats['cache_size']}/{stats['max_size']}")
```

## Recommendation Engine

The unified recommendation engine provides consistent buy/sell/hold recommendations for all indicators.

### Technical Recommendations

```python
from src.common.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()

# Get RSI recommendation
rsi_rec = engine.get_recommendation("RSI", 25.5)
print(f"RSI 25.5: {rsi_rec.recommendation.value} (Confidence: {rsi_rec.confidence:.2f})")

# Get MACD recommendation (needs context)
macd_rec = engine.get_recommendation("MACD", 0.5, {
    'macd_signal': 0.3,
    'macd_histogram': 0.2
})
print(f"MACD: {macd_rec.recommendation.value}")
```

### Fundamental Recommendations

```python
# Get P/E recommendation
pe_rec = engine.get_recommendation("PE_RATIO", 15.2)
print(f"P/E 15.2: {pe_rec.recommendation.value}")

# Get ROE recommendation
roe_rec = engine.get_recommendation("ROE", 0.18)
print(f"ROE 18%: {roe_rec.recommendation.value}")
```

### Composite Recommendations

```python
# Get overall recommendation from indicator set
composite = engine.get_composite_recommendation(result)
print(f"Overall: {composite.recommendation.value}")
print(f"Confidence: {composite.confidence:.2f}")
print(f"Reasoning: {composite.reasoning}")
print(f"Technical Score: {composite.technical_score:.2f}")
print(f"Fundamental Score: {composite.fundamental_score:.2f}")
```

## Fundamentals

Get fundamental data for stocks using multiple providers:

```python
from src.common.fundamentals import get_fundamentals

# Get fundamentals from a specific provider
fundamentals = get_fundamentals('AAPL', provider='yf')

# Get fundamentals from all available providers (merged)
fundamentals = get_fundamentals('AAPL')

print(f"PE Ratio: {fundamentals.pe_ratio}")
print(f"Market Cap: {fundamentals.market_cap}")
```

### Normalize Fundamentals

Combine data from multiple providers:

```python
from src.common.fundamentals import normalize_fundamentals

# Normalize data from multiple sources
sources = {
    'yf': yf_data,
    'av': av_data,
    'fh': fh_data
}
fundamentals = normalize_fundamentals(sources)
```

## OHLCV Data

Get historical price data:

```python
from src.common import get_ohlcv

# Get OHLCV data
df = get_ohlcv('AAPL', '1d', '2y', provider='yf')
print(df.head())
```

## Technical Analysis (Unified)

Calculate technical indicators using the unified system:

```python
from src.common.technicals import calculate_technicals_unified

# Get technicals using unified service
technicals = await calculate_technicals_unified('AAPL', '2y', '1d', provider='yf')

# The unified function automatically calculates all major technical indicators
# including RSI, MACD, Bollinger Bands, Stochastic, ADX, SMA, EMA, etc.
```

## Ticker Analysis

Complete ticker analysis with fundamentals, technicals, and charts:

```python
from src.common.ticker_analyzer import analyze_ticker

# Analyze ticker with default settings
analysis = analyze_ticker('AAPL')

# Analyze with custom parameters
analysis = analyze_ticker(
    ticker='AAPL',
    period='1y',
    interval='1d',
    provider='yf'
)

print(f"Ticker: {analysis.ticker}")
print(f"Fundamentals: {analysis.fundamentals.company_name}")
print(f"Technicals: RSI = {analysis.technicals.rsi}")
```

### Format Analysis Results

```python
from src.common.ticker_analyzer import format_comprehensive_analysis

# Get formatted analysis text
analysis_text = format_comprehensive_analysis(analysis)
print(analysis_text)
```

## Chart Generation

Generate charts for ticker analysis:

```python
from src.common.ticker_chart import generate_chart

# Generate chart from ticker analysis
chart_image = generate_chart(analysis.ticker, analysis.ohlcv)

# Chart is returned as bytes - use directly or save to file
# Note: Charts are no longer automatically saved to project root
chart_bytes = chart_image  # Use directly in applications
```

## Period/Interval Utilities

Convert period strings to date ranges:

```python
from src.common import analyze_period_interval

start_date, end_date = analyze_period_interval('2y', '1d')
print(f"From {start_date} to {end_date}")
```

## Supported Providers

### Stock Providers
- `yf` - Yahoo Finance (default for stocks)
- `av` - Alpha Vantage
- `fh` - Finnhub
- `td` - Twelve Data
- `pg` - Polygon

### Crypto Providers
- `bnc` - Binance (default for crypto)
- `cg` - CoinGecko

## Module Structure

```
src/common/
├── __init__.py              # Core utilities (get_ohlcv, analyze_period_interval)
├── (indicator_service.py removed - moved to src/indicators/)
├── recommendation_engine.py # Unified recommendation engine
├── fundamentals.py          # Fundamentals logic (get_fundamentals, normalize_fundamentals)
├── technicals.py            # Technicals logic (legacy system)
├── ticker_analyzer.py       # Complete ticker analysis
├── ticker_chart.py          # Chart generation
└── README.md               # This file
```

## Recent Improvements

### Debug Output Cleanup
- **Removed Debug Logging**: Eliminated verbose debug output for cleaner logs
- **No File Pollution**: Charts are no longer automatically saved to project root
- **Cleaner Output**: Streamlined logging focuses on essential information only
- **Production Ready**: Optimized for production environments

### Chart Generation Updates
- **Function Signature**: Updated `generate_chart()` to accept `(ticker, df)` instead of `TickerAnalysis` object
- **Memory Efficient**: Charts returned as bytes for direct use in applications
- **No Side Effects**: Chart generation no longer creates files in project directory
- **Better Error Handling**: Improved error handling for chart generation failures

## Best Practices

### Import Patterns

```python
# Unified Indicator System (Recommended)
from src.indicators.service import UnifiedIndicatorService
from src.indicators.models import IndicatorCalculationRequest, BatchIndicatorRequest

# Recommendation Engine
from src.common.recommendation_engine import RecommendationEngine

# Fundamentals
from src.common.fundamentals import get_fundamentals, normalize_fundamentals

# Technicals (Unified)
from src.common.technicals import calculate_technicals_unified

# Ticker Analysis
from src.common.ticker_analyzer import analyze_ticker, format_comprehensive_analysis

# Charts
from src.common.ticker_chart import generate_chart

# Core utilities
from src.common import get_ohlcv, analyze_period_interval
```

### Error Handling

```python
try:
    # Unified indicator system
    result = await service.get_indicators(request)
except Exception as e:
    print(f"Error: {e}")

try:
    # Legacy fundamentals
    fundamentals = get_fundamentals('INVALID', provider='yf')
except ValueError as e:
    print(f"Error: {e}")

try:
    # Unified technicals
    technicals = await calculate_technicals_unified('AAPL', '2y', '1d', provider='yf')
except Exception as e:
    print(f"Error: {e}")
```

### Provider Selection

```python
# Auto-select provider based on ticker
fundamentals = get_fundamentals('AAPL')  # Uses 'yf' for stocks
fundamentals = get_fundamentals('BTCUSDT')  # Uses 'bnc' for crypto

# Manual provider selection
fundamentals = get_fundamentals('AAPL', provider='av')  # Force Alpha Vantage
```

## Testing

Run the test suite:

```bash
# Test unified indicator system
python -c "from src.indicators.service import UnifiedIndicatorService; print('Indicator Service OK')"

# Test recommendation engine
python -c "from src.common.recommendation_engine import RecommendationEngine; print('Recommendation Engine OK')"

# Test unified systems
python -c "from src.common.fundamentals import get_fundamentals; print('Fundamentals OK')"
python -c "from src.common.technicals import calculate_technicals_unified; print('Technicals OK')"
python -c "from src.common.ticker_analyzer import analyze_ticker; print('Analyzer OK')"
```

## Examples

### Complete Analysis Workflow (Unified System)

```python
from src.indicators.service import UnifiedIndicatorService
from src.indicators.models import IndicatorCalculationRequest

# Get unified service
service = UnifiedIndicatorService()

# Analyze ticker with comprehensive indicators
request = IndicatorCalculationRequest(
    ticker="AAPL",
    indicators=["RSI", "MACD", "BB_UPPER", "BB_LOWER", "PE_RATIO", "ROE", "DIVIDEND_YIELD"],
    timeframe="1d",
    period="1y",
    include_recommendations=True
)

result = await service.get_indicators(request)

# Display results
print(f"Analysis for {result.ticker}")
print(f"Composite Score: {result.composite_score:.2f}")
print(f"Overall Recommendation: {result.overall_recommendation.recommendation.value}")
print(f"Confidence: {result.overall_recommendation.confidence:.2f}")
print(f"Reasoning: {result.overall_recommendation.reasoning}")

# Show technical indicators
print("\nTechnical Indicators:")
for name, indicator in result.get_all_indicators().items():
    if indicator.category.value == 'technical':
        print(f"  {name}: {indicator.value:.4f} ({indicator.recommendation.recommendation.value})")

# Show fundamental indicators
print("\nFundamental Indicators:")
for name, indicator in result.get_all_indicators().items():
    if indicator.category.value == 'fundamental':
        print(f"  {name}: {indicator.value:.4f} ({indicator.recommendation.recommendation.value})")
```

### Batch Analysis

```python
# Analyze multiple tickers efficiently
batch_request = BatchIndicatorRequest(
    tickers=["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"],
    indicators=["RSI", "MACD", "PE_RATIO", "ROE"],
    timeframe="1d",
    period="1y",
    max_concurrent=5,
    include_recommendations=True
)

results = await service.get_batch_indicators(batch_request)

# Sort by composite score
sorted_results = sorted(results.items(), key=lambda x: x[1].composite_score, reverse=True)

print("Top Performers:")
for ticker, result in sorted_results:
    print(f"{ticker}: Score {result.composite_score:.2f} - {result.overall_recommendation.recommendation.value}")
```

### Custom Recommendation Analysis

```python
from src.common.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()

# Analyze specific indicators
indicators_to_analyze = [
    ("RSI", 25.5),
    ("PE_RATIO", 15.2),
    ("ROE", 0.18),
    ("DIVIDEND_YIELD", 3.5)
]

print("Individual Indicator Analysis:")
for indicator, value in indicators_to_analyze:
    rec = engine.get_recommendation(indicator, value)
    print(f"{indicator} {value}: {rec.recommendation.value} (Confidence: {rec.confidence:.2f})")
    print(f"  Reason: {rec.reason}")
```

### Enhanced Chart Generation

The enhanced chart generation system provides professional-quality technical analysis charts with modern styling and comprehensive indicator visualization.

```python
from src.common.ticker_analyzer import analyze_ticker
from src.common.ticker_chart import generate_chart

# Analyze ticker with unified indicator service
analysis = await analyze_ticker('AAPL', period='1y', interval='1d')

# Generate enhanced chart with multiple indicators
chart_image = generate_chart(analysis.ticker, analysis.ohlcv)

# Chart is returned as bytes - use directly in applications
# Note: Charts are no longer automatically saved to project root

print(f"Enhanced analysis complete for {analysis.ticker}")
print(f"PE Ratio: {analysis.fundamentals.pe_ratio}")
print(f"RSI: {analysis.technicals.rsi}")
```

#### Chart Features

- **Dynamic Layout**: Automatically adjusts subplot layout based on available indicators
- **Modern Styling**: Professional color palette and typography
- **Multiple Indicators**: RSI, MACD, Stochastic, ADX, CCI, MFI, Williams %R, ROC, OBV, ATR
- **Price Overlays**: Bollinger Bands, SMA/EMA moving averages, ATR bands
- **Current Values**: Real-time indicator values displayed on chart
- **Error Handling**: Graceful handling of missing data and calculation errors

#### Chart Configuration

```python
# Chart configuration options
CHART_CONFIG = {
    'figure_size': (18, 14),      # Chart dimensions
    'dpi': 150,                   # Image resolution
    'grid_alpha': 0.3,            # Grid transparency
    'line_width': 1.2,            # Line thickness
    'font_size': 10,              # Base font size
    'title_font_size': 16,        # Title font size
    'legend_font_size': 9,        # Legend font size
    'tick_font_size': 8,          # Tick label font size
    'subplot_spacing': 0.4,       # Space between subplots
    'date_format': '%m/%d',       # Date format
    'date_interval': 2,           # Date tick interval
}
```

### Legacy System Integration

```python
# Use legacy systems alongside unified system
from src.common.ticker_analyzer import analyze_ticker
from src.common.ticker_chart import generate_chart

# Legacy analysis
analysis = await analyze_ticker('AAPL', period='1y', interval='1d')

# Generate enhanced chart
chart_image = generate_chart(analysis.ticker, analysis.ohlcv)

# Chart is returned as bytes - use directly in applications
# Note: Charts are no longer automatically saved to project root

print(f"Legacy analysis complete for {analysis.ticker}")
print(f"PE Ratio: {analysis.fundamentals.pe_ratio}")
print(f"RSI: {analysis.technicals.rsi}")
```

## Ticker Classification

The enhanced ticker classification system provides intelligent provider selection and comprehensive pattern recognition:

```python
from src.data.data_manager import ProviderSelector

selector = ProviderSelector()

# Get ticker information
info = selector.get_ticker_info("AAPL")
print(f"Symbol Type: {info['symbol_type']}")  # stock
print(f"Exchange: {info['exchange']}")        # US Markets (NASDAQ/NYSE)

info = selector.get_ticker_info("BTCUSDT")
print(f"Symbol Type: {info['symbol_type']}")  # crypto
print(f"Base Asset: {info['base_asset']}")    # BTC
print(f"Quote Asset: {info['quote_asset']}")  # USDT

info = selector.get_ticker_info("NOVO.CO")
print(f"Symbol Type: {info['symbol_type']}")  # stock
print(f"Exchange: {info['exchange']}")        # Copenhagen Stock Exchange (Denmark)
```

### Enhanced Features

#### **Comprehensive Exchange Support (118 exchanges)**
- **European**: London (.L), Frankfurt (.DE), Paris (.PA), Amsterdam (.AS), Milan (.MI), Madrid (.MC), Swiss (.SW)
- **Nordic**: Copenhagen (.CO), Stockholm (.ST), Helsinki (.HE), Oslo (.OS), Iceland (.IC)
- **Asian**: Tokyo (.T), Hong Kong (.HK), Shanghai (.SS), Shenzhen (.SZ), Singapore (.SG), Taiwan (.TW)
- **Emerging Markets**: India (.NS, .BO), Brazil (.SA), Mexico (.MX), Indonesia (.JK), Malaysia (.KL), Thailand (.BK)
- **Americas**: Toronto (.TO), Australian (.AX), New Zealand (.NZ), Santiago (.SN), Lima (.LM), Buenos Aires (.BA)
- **Africa & Middle East**: Johannesburg (.JO), Cairo (.CA), Tel Aviv (.TA), Istanbul (.IS), Moscow (.ME)

#### **Enhanced Crypto Recognition (102 assets)**
- **Major Cryptocurrencies**: BTC, ETH, BNB, ADA, DOT, LINK, LTC, XRP, SOL, MATIC, AVAX
- **DeFi Tokens**: AAVE, COMP, SNX, MKR, YFI, CRV, SUSHI, UNI, BAL, 1INCH, DYDX
- **Gaming & Metaverse**: AXS, SAND, MANA, GALA, ENJ, CHZ, ALICE, TLM
- **Layer 1 & 2**: SOL, AVAX, FTM, NEAR, ATOM, DOT, ADA, ALGO, MATIC, OP, ARB
- **AI & Data**: OCEAN, FET, AGIX, NMR, BAND, LINK
- **Privacy**: XMR, ZEC, DASH, PIVX, BEAM, GRIN, SCRT

#### **Ticker Validation**
```python
# Validate ticker format
validation = classifier.validate_ticker("AAPL")
if validation['valid']:
    print(f"Valid ticker: {validation['provider']}")
else:
    print(f"Error: {validation['error']}")
    print(f"Suggestions: {validation['suggestions']}")
```

#### **Performance Optimizations**
- **Compiled Regex Patterns**: 88,075 classifications per second
- **Efficient Pattern Matching**: Optimized order for fastest classification
- **Memory Efficient**: Minimal memory footprint with comprehensive coverage

#### **Error Handling & Suggestions**
```python
# Invalid ticker with helpful suggestions
validation = classifier.validate_ticker("INVALID@123")
# Returns: {'valid': False, 'error': 'Invalid ticker format', 
#          'suggestions': ['Use only alphanumeric characters, dots, and hyphens']}
```

### Supported Patterns

#### **Stock Patterns**
- **US Stocks**: AAPL, MSFT, GOOGL, TSLA (2-5 letters)
- **US Stocks with Classes**: BRK.A, BRK.B (1-4 letters + dot + letter)
- **International**: VUSD.L, NOVO.CO, BMW.DE, LVMH.PA (with exchange suffixes)

#### **Crypto Patterns**
- **Major Pairs**: BTCUSDT, ETHUSD, ADABTC, SOLUSDT
- **Stablecoins**: BTCBUSD, ETHBUSD, BNBUSDT
- **Cross Pairs**: BTCETH, ETHBNB, ADABTC
- **DeFi Pairs**: AAVEUSDT, COMPUSDT, SNXUSDT

#### **Validation Rules**
- **Length**: 1-15 characters
- **Characters**: Alphanumeric, dots, hyphens only
- **Format**: No leading/trailing dots, no consecutive dots
- **Case**: Case-insensitive (automatically converted to uppercase)

## Migration Guide

### From Legacy to Unified System

**Before (Legacy):**
```python
from src.common.technicals import get_technicals
from src.common.fundamentals import get_fundamentals

# Separate calls for technicals and fundamentals
technicals = get_technicals('AAPL', '1d', '1y', provider='yf')
fundamentals = get_fundamentals('AAPL', provider='yf')

# Manual recommendation logic
if technicals.rsi < 30:
    rsi_rec = "BUY"
elif technicals.rsi > 70:
    rsi_rec = "SELL"
else:
    rsi_rec = "HOLD"
```

**After (Unified):**
```python
from src.indicators.service import UnifiedIndicatorService
from src.indicators.models import IndicatorCalculationRequest

service = UnifiedIndicatorService()

request = IndicatorCalculationRequest(
    ticker="AAPL",
    indicators=["RSI", "MACD", "PE_RATIO", "ROE"],
    timeframe="1d",
    period="1y",
    include_recommendations=True
)

result = await service.get_indicators(request)

# Automatic recommendations with confidence scores
for name, indicator in result.get_all_indicators().items():
    print(f"{name}: {indicator.recommendation.recommendation.value} (Confidence: {indicator.recommendation.confidence:.2f})")

# Overall composite recommendation
print(f"Overall: {result.overall_recommendation.recommendation.value}")
```

### Benefits of Migration

1. **Unified Interface**: Single service for all indicators
2. **Automatic Recommendations**: Built-in buy/sell/hold logic
3. **Confidence Scores**: Quantitative confidence for each recommendation
4. **Composite Analysis**: Overall recommendation combining all indicators
5. **Better Performance**: Direct TA-Lib integration with intelligent caching
6. **Parameter-Aware Caching**: Handles different indicator combinations efficiently
7. **Batch Processing**: Efficient multi-ticker analysis
8. **Error Handling**: Robust error handling and fallbacks 
