# Common Module

This module provides unified access to data providers, fundamentals, technical analysis, and ticker analysis functions.

## Overview

The common module is organized into specialized submodules:

- **`fundamentals.py`** - Fundamental data retrieval and normalization
- **`technicals.py`** - Technical indicator calculations
- **`ticker_analyzer.py`** - Complete ticker analysis with fundamentals and technicals
- **`ticker_chart.py`** - Chart generation for ticker analysis
- **`__init__.py`** - Core utilities (OHLCV data, period/interval conversion)

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

## Technical Analysis

Calculate technical indicators:

```python
from src.common.technicals import get_technicals, calculate_technicals_from_df

# Get technicals directly
technicals = get_technicals('AAPL', '1d', '2y', provider='yf')

# Or calculate from existing DataFrame
df, technicals = calculate_technicals_from_df(df, indicators=['rsi', 'macd'])

# Get technicals with custom parameters
technicals = get_technicals(
    'AAPL', 
    '1d', 
    '2y', 
    provider='yf',
    indicators=['rsi', 'macd', 'bollinger'],
    indicator_params={'rsi': {'timeperiod': 10}}
)
```

### Available Technical Indicators

- **RSI** - Relative Strength Index
- **MACD** - Moving Average Convergence Divergence
- **Bollinger Bands** - Upper, Middle, Lower bands
- **Stochastic** - Stochastic Oscillator (K and D)
- **ADX** - Average Directional Index
- **OBV** - On Balance Volume
- **ADR** - Average Daily Range
- **SMA** - Simple Moving Averages (50, 200)

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
chart_image = generate_chart(analysis)

# Save chart to file
with open('chart.png', 'wb') as f:
    f.write(chart_image)
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
├── fundamentals.py          # Fundamentals logic (get_fundamentals, normalize_fundamentals)
├── technicals.py            # Technicals logic (get_technicals, calculate_technicals_from_df)
├── ticker_analyzer.py       # Complete ticker analysis
├── ticker_chart.py          # Chart generation
└── README.md               # This file
```

## Best Practices

### Import Patterns

```python
# Fundamentals
from src.common.fundamentals import get_fundamentals, normalize_fundamentals

# Technicals  
from src.common.technicals import get_technicals, calculate_technicals_from_df

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
    fundamentals = get_fundamentals('INVALID', provider='yf')
except ValueError as e:
    print(f"Error: {e}")

try:
    technicals = get_technicals('AAPL', '1d', '2y', provider='yf')
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
# Run all common module tests
python -m pytest tests/test_common.py -v

# Test specific functionality
python -c "from src.common.fundamentals import get_fundamentals; print('Fundamentals OK')"
python -c "from src.common.technicals import get_technicals; print('Technicals OK')"
python -c "from src.common.ticker_analyzer import analyze_ticker; print('Analyzer OK')"
```

## Examples

### Complete Analysis Workflow

```python
from src.common.ticker_analyzer import analyze_ticker
from src.common.ticker_chart import generate_chart

# Analyze ticker
analysis = analyze_ticker('AAPL', period='1y', interval='1d')

# Generate chart
chart_image = generate_chart(analysis)

# Save results
with open('aapl_analysis.png', 'wb') as f:
    f.write(chart_image)

print(f"Analysis complete for {analysis.ticker}")
print(f"PE Ratio: {analysis.fundamentals.pe_ratio}")
print(f"RSI: {analysis.technicals.rsi}")
```

### Multi-Provider Fundamentals

```python
from src.common.fundamentals import get_fundamentals

# Get comprehensive fundamentals from all providers
fundamentals = get_fundamentals('AAPL')

print(f"Data sources: {fundamentals.sources}")
print(f"PE Ratio from: {fundamentals.sources.get('pe_ratio', 'Unknown')}")
```

### Custom Technical Analysis

```python
from src.common.technicals import get_technicals

# Get specific indicators with custom parameters
technicals = get_technicals(
    'AAPL',
    '1d',
    '6mo',
    provider='yf',
    indicators=['rsi', 'macd', 'bollinger'],
    indicator_params={
        'rsi': {'timeperiod': 10},
        'macd': {'fastperiod': 8, 'slowperiod': 21}
    }
)

print(f"RSI (10): {technicals.rsi}")
print(f"MACD: {technicals.macd}")
``` 