# Unified Indicator Service API Documentation

## Overview

The Unified Indicator Service provides a comprehensive API for calculating technical and fundamental indicators with intelligent recommendations. This service consolidates all indicator functionality into a single, consistent interface that supports multiple calculation backends and provides advanced features like batch processing, error handling, and performance monitoring.

## Quick Start

```python
from src.indicators.service import get_unified_indicator_service
from src.indicators.models import IndicatorCalculationRequest

# Get service instance
service = get_unified_indicator_service()

# Create request
request = IndicatorCalculationRequest(
    ticker="AAPL",
    indicators=["rsi", "macd", "bbands"],
    timeframe="1d",
    period="1y"
)

# Calculate indicators
result = await service.get_indicators(request)

# Access results
rsi_value = result.get_indicator("rsi").value
macd_recommendation = result.get_indicator("macd").recommendation
```

## Core API Methods

### Single Ticker Calculation

#### `get_indicators(request: IndicatorCalculationRequest) -> IndicatorSet`

Calculate indicators for a single ticker with recommendations.

**Parameters:**
- `request`: IndicatorCalculationRequest object containing:
  - `ticker`: Stock ticker symbol (e.g., "AAPL", "GOOGL")
  - `indicators`: List of indicator names (canonical or legacy)
  - `timeframe`: Data timeframe ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M")
  - `period`: Data period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
  - `provider`: Optional data provider ("yfinance", "alpha_vantage", "polygon")
  - `force_refresh`: Force cache refresh (default: False)
  - `include_recommendations`: Include trading recommendations (default: True)

**Returns:**
- `IndicatorSet` object containing:
  - `ticker`: Ticker symbol
  - `technical_indicators`: Dict of technical indicator results
  - `fundamental_indicators`: Dict of fundamental indicator results
  - `overall_recommendation`: Composite recommendation
  - `composite_score`: Overall score (-1.0 to 1.0)
  - `last_updated`: Calculation timestamp

**Example:**
```python
from src.indicators.models import IndicatorCalculationRequest

request = IndicatorCalculationRequest(
    ticker="AAPL",
    indicators=["rsi", "macd", "pe_ratio", "roe"],
    timeframe="1d",
    period="2y",
    include_recommendations=True
)

result = await service.get_indicators(request)

# Access technical indicators
rsi_result = result.get_indicator("rsi")
print(f"RSI: {rsi_result.value}")
print(f"Recommendation: {rsi_result.recommendation.recommendation}")
print(f"Confidence: {rsi_result.recommendation.confidence}")

# Access fundamental indicators
pe_result = result.get_indicator("pe_ratio")
print(f"P/E Ratio: {pe_result.value}")

# Overall recommendation
if result.overall_recommendation:
    print(f"Overall: {result.overall_recommendation.recommendation}")
    print(f"Reasoning: {result.overall_recommendation.reasoning}")
```

### Batch Processing

#### `get_batch_indicators(request: BatchIndicatorRequest) -> Dict[str, IndicatorSet]`

Calculate indicators for multiple tickers efficiently.

**Parameters:**
- `request`: BatchIndicatorRequest object containing:
  - `tickers`: List of ticker symbols
  - `indicators`: List of indicator names
  - `timeframe`: Data timeframe (default: "1d")
  - `period`: Data period (default: "2y")
  - `provider`: Optional data provider
  - `max_concurrent`: Maximum concurrent requests (default: 10)
  - `force_refresh`: Force cache refresh (default: False)
  - `include_recommendations`: Include recommendations (default: True)

**Returns:**
- Dictionary mapping ticker symbols to IndicatorSet objects

**Example:**
```python
from src.indicators.models import BatchIndicatorRequest

request = BatchIndicatorRequest(
    tickers=["AAPL", "GOOGL", "MSFT", "TSLA"],
    indicators=["rsi", "macd", "pe_ratio"],
    timeframe="1d",
    period="1y",
    max_concurrent=5
)

results = await service.get_batch_indicators(request)

for ticker, indicator_set in results.items():
    rsi = indicator_set.get_indicator("rsi")
    if rsi:
        print(f"{ticker} RSI: {rsi.value} ({rsi.recommendation.recommendation})")
```

#### `get_batch_indicators_enhanced(request: BatchIndicatorRequest, config: BatchProcessingConfig) -> BatchResult`

Enhanced batch processing with detailed error handling and partial results.

**Parameters:**
- `request`: BatchIndicatorRequest object
- `config`: Optional BatchProcessingConfig for advanced settings:
  - `max_concurrent`: Maximum concurrent operations (default: 10)
  - `batch_size`: Batch size for processing (default: 50)
  - `timeout_per_ticker`: Timeout per ticker in seconds (default: 30.0)
  - `retry_attempts`: Number of retry attempts (default: 2)
  - `partial_results`: Return partial results for failed tickers (default: True)
  - `fail_fast`: Stop on first failure (default: False)

**Returns:**
- `BatchResult` object containing:
  - `successful`: Dict of successful calculations
  - `failed`: Dict of failed calculations with error details
  - `partial`: Dict of partial results
  - `performance_metrics`: Performance statistics
  - `success_rate`: Overall success rate

**Example:**
```python
from src.indicators.service import BatchProcessingConfig

config = BatchProcessingConfig(
    max_concurrent=15,
    timeout_per_ticker=45.0,
    retry_attempts=3,
    partial_results=True
)

result = await service.get_batch_indicators_enhanced(request, config)

print(f"Success rate: {result.success_rate:.2%}")
print(f"Successful: {len(result.successful)}")
print(f"Failed: {len(result.failed)}")

# Process successful results
for ticker, indicator_set in result.successful.items():
    # Process indicators...
    pass

# Handle failures
for ticker, error in result.failed.items():
    print(f"Failed to process {ticker}: {error}")
```

## Configuration Management

### `config_manager` Property

Access the unified configuration manager for parameter customization.

#### Key Methods:

**`get_parameters(indicator: str, preset: str = None) -> Dict[str, Any]`**

Get parameters for a specific indicator with preset support.

```python
# Get default parameters
params = service.config_manager.get_parameters("rsi")
print(params)  # {'timeperiod': 14}

# Get conservative preset parameters
params = service.config_manager.get_parameters("rsi", "conservative")
print(params)  # {'timeperiod': 21}
```

**`set_preset(preset_name: str) -> bool`**

Set the current parameter preset.

```python
# Available presets: "default", "conservative", "aggressive", "day_trading"
service.config_manager.set_preset("aggressive")

# Now all indicators will use aggressive parameters
result = await service.get_indicators(request)
```

**`set_parameter_override(indicator: str, parameter: str, value: Any)`**

Override specific parameters at runtime.

```python
# Override RSI period to 21
service.config_manager.set_parameter_override("rsi", "timeperiod", 21)

# Override MACD parameters
service.config_manager.set_parameter_override("macd", "fastperiod", 8)
service.config_manager.set_parameter_override("macd", "slowperiod", 17)
```

**`get_available_presets() -> List[str]`**

Get list of available parameter presets.

```python
presets = service.config_manager.get_available_presets()
print(presets)  # ['default', 'conservative', 'aggressive', 'day_trading']
```

## Recommendation Engine

### `recommendation_engine` Property

Access the unified recommendation engine for custom recommendations.

#### Key Methods:

**`get_recommendation(indicator: str, value: float, context: Dict = None) -> Recommendation`**

Get recommendation for a specific indicator value.

```python
from src.indicators.models import Recommendation

# Simple recommendation
rec = service.recommendation_engine.get_recommendation("rsi", 75.0)
print(f"RSI 75: {rec.recommendation} (confidence: {rec.confidence})")

# Contextual recommendation with additional data
context = {
    "current_price": 150.0,
    "bb_upper": 155.0,
    "bb_middle": 150.0,
    "bb_lower": 145.0
}
rec = service.recommendation_engine.get_recommendation("bbands", 150.0, context)
```

**`get_composite_recommendation(indicator_set: IndicatorSet) -> CompositeRecommendation`**

Generate overall recommendation from multiple indicators.

```python
composite = service.recommendation_engine.get_composite_recommendation(indicator_set)
print(f"Overall: {composite.recommendation}")
print(f"Confidence: {composite.confidence}")
print(f"Contributing indicators: {composite.contributing_indicators}")
print(f"Technical score: {composite.technical_score}")
print(f"Fundamental score: {composite.fundamental_score}")
```

## Available Indicators

### Technical Indicators (23 indicators)

| Canonical Name | Legacy Names | Description |
|----------------|--------------|-------------|
| `rsi` | `RSI` | Relative Strength Index |
| `macd` | `MACD`, `MACD_SIGNAL`, `MACD_HISTOGRAM` | Moving Average Convergence Divergence |
| `bbands` | `BB_UPPER`, `BB_MIDDLE`, `BB_LOWER` | Bollinger Bands |
| `stoch` | `STOCH_K`, `STOCH_D` | Stochastic Oscillator |
| `adx` | `ADX` | Average Directional Index |
| `plus_di` | `PLUS_DI` | Plus Directional Indicator |
| `minus_di` | `MINUS_DI` | Minus Directional Indicator |
| `sma` | `SMA_FAST`, `SMA_SLOW`, `SMA_50`, `SMA_200` | Simple Moving Average |
| `ema` | `EMA_FAST`, `EMA_SLOW`, `EMA_12`, `EMA_26` | Exponential Moving Average |
| `cci` | `CCI` | Commodity Channel Index |
| `roc` | `ROC` | Rate of Change |
| `mfi` | `MFI` | Money Flow Index |
| `williams_r` | `WILLIAMS_R` | Williams %R |
| `atr` | `ATR` | Average True Range |
| `obv` | `OBV` | On-Balance Volume |
| `adr` | `ADR` | Average Daily Range |
| `aroon` | `AROON_UP`, `AROON_DOWN` | Aroon Oscillator |
| `ichimoku` | `ICHIMOKU` | Ichimoku Cloud |
| `sar` | `SAR` | Parabolic SAR |
| `super_trend` | `SUPER_TREND` | Super Trend |
| `ad` | `AD` | Accumulation/Distribution Line |
| `adosc` | `ADOSC` | Chaikin A/D Oscillator |
| `bop` | `BOP` | Balance of Power |

### Fundamental Indicators (21 indicators)

| Canonical Name | Legacy Names | Description |
|----------------|--------------|-------------|
| `pe_ratio` | `PE_RATIO` | Price-to-Earnings Ratio |
| `forward_pe` | `FORWARD_PE` | Forward P/E Ratio |
| `pb_ratio` | `PB_RATIO` | Price-to-Book Ratio |
| `ps_ratio` | `PS_RATIO` | Price-to-Sales Ratio |
| `peg_ratio` | `PEG_RATIO` | Price/Earnings-to-Growth Ratio |
| `roe` | `ROE` | Return on Equity |
| `roa` | `ROA` | Return on Assets |
| `debt_to_equity` | `DEBT_TO_EQUITY` | Debt-to-Equity Ratio |
| `current_ratio` | `CURRENT_RATIO` | Current Ratio |
| `quick_ratio` | `QUICK_RATIO` | Quick Ratio |
| `operating_margin` | `OPERATING_MARGIN` | Operating Margin |
| `profit_margin` | `PROFIT_MARGIN` | Profit Margin |
| `revenue_growth` | `REVENUE_GROWTH` | Revenue Growth |
| `net_income_growth` | `NET_INCOME_GROWTH` | Net Income Growth |
| `free_cash_flow` | `FREE_CASH_FLOW` | Free Cash Flow |
| `dividend_yield` | `DIVIDEND_YIELD` | Dividend Yield |
| `payout_ratio` | `PAYOUT_RATIO` | Payout Ratio |
| `beta` | `BETA` | Beta |
| `market_cap` | `MARKET_CAP` | Market Capitalization |
| `enterprise_value` | `ENTERPRISE_VALUE` | Enterprise Value |
| `ev_to_ebitda` | `EV_TO_EBITDA` | Enterprise Value to EBITDA |

### Multi-Output Indicators

Some indicators return multiple values:

**MACD (`macd`)**
- `macd`: MACD line
- `signal`: Signal line  
- `hist`: Histogram

**Bollinger Bands (`bbands`)**
- `upper`: Upper band
- `middle`: Middle band (SMA)
- `lower`: Lower band

**Stochastic (`stoch`)**
- `k`: %K line
- `d`: %D line

**Aroon (`aroon`)**
- `up`: Aroon Up
- `down`: Aroon Down

## Error Handling

The service provides comprehensive error handling with specific exception types:

### Exception Types

```python
from src.indicators.service import (
    IndicatorServiceError,      # Base exception
    ConfigurationError,         # Configuration issues
    DataError,                  # Data availability/quality
    CalculationError,          # Calculation failures
    TimeoutError,              # Operation timeouts
    CircuitBreakerError        # Circuit breaker open
)

try:
    result = await service.get_indicators(request)
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Context: {e.context}")
except DataError as e:
    print(f"Data error: {e}")
except CalculationError as e:
    print(f"Calculation error: {e}")
except TimeoutError as e:
    print(f"Timeout error: {e}")
```

### Circuit Breakers

The service includes circuit breakers for external dependencies:

```python
# Check circuit breaker status
status = service.get_circuit_breaker_status()
for name, info in status.items():
    print(f"{name}: {'OPEN' if info['is_open'] else 'CLOSED'}")
    print(f"  Failures: {info['failure_count']}/{info['failure_threshold']}")

# Manually reset a circuit breaker
service.reset_circuit_breaker("data_provider")
```

## Performance Monitoring

### Performance Metrics

```python
# Get performance report
metrics = service.get_performance_metrics()
print(f"Total operations: {metrics['statistics']['_overall']['total_operations']}")
print(f"Error rate: {metrics['statistics']['_overall']['overall_error_rate']:.2%}")
print(f"Cache hit rate: {metrics['statistics']['_overall']['overall_cache_hit_rate']:.2%}")

# Reset metrics
service.reset_performance_metrics()
```

### Benchmarking

```python
# Single ticker benchmark
benchmark = await service.run_benchmark(
    ticker="AAPL",
    indicators=["rsi", "macd", "bbands"],
    iterations=10
)
print(f"Average time: {benchmark['unified_service']['avg_time']:.3f}s")

# Batch processing benchmark
batch_benchmark = await service.run_batch_benchmark(
    tickers=["AAPL", "GOOGL", "MSFT"],
    indicators=["rsi", "macd"],
    batch_sizes=[1, 5, 10]
)
for batch_size, results in batch_benchmark['batch_results'].items():
    print(f"Batch size {batch_size}: {results['throughput']:.1f} tickers/sec")
```

## Health Checks

```python
# Comprehensive health check
health = await service.health_check()
print(f"Status: {health['status']}")

for component, status in health['components'].items():
    print(f"{component}: {status['status']}")

# Check for open circuit breakers
if health.get('open_circuit_breakers'):
    print(f"Open circuit breakers: {health['open_circuit_breakers']}")
```

## Service Information

```python
# Get service information
info = service.get_service_info()
print(f"Service: {info['service']} v{info['version']}")
print(f"Available adapters: {info['adapters']}")
print(f"Total indicators: {info['available_indicators']['total_count']}")
print(f"Technical: {info['available_indicators']['technical_count']}")
print(f"Fundamental: {info['available_indicators']['fundamental_count']}")

# Get available indicators
indicators = service.get_available_indicators()
print(f"Technical indicators: {indicators['technical']}")
print(f"Fundamental indicators: {indicators['fundamental']}")
```

## Migration from Legacy Services

### Parameter Changes

The unified service uses simplified parameter names:

| Legacy Parameter | Unified Parameter | Notes |
|------------------|-------------------|-------|
| `use_cache` | Removed | Caching is always enabled |
| `cache_ttl` | Removed | Uses global cache settings |
| `backend` | Removed | Automatic backend selection |
| `fallback` | Removed | Built-in fallback logic |
| `timeout` | Removed | Uses global timeout settings |

### Method Mapping

| Legacy Method | Unified Method | Notes |
|---------------|----------------|-------|
| `IndicatorService.get_indicators()` | `get_indicators()` | Same interface |
| `IndicatorService.get_batch_indicators()` | `get_batch_indicators()` | Same interface |
| `IndicatorFactory.create_*()` | `get_indicators()` | Use indicator names instead |
| `get_recommendation()` | `recommendation_engine.get_recommendation()` | Enhanced context support |

### Import Changes

```python
# Old imports
from src.common.indicator_service import IndicatorService  # Legacy - removed
from src.common.indicator_config import get_config  # Legacy - removed

# New imports
from src.indicators.service import get_unified_indicator_service
from src.indicators.config_manager import get_config_manager
```

## Best Practices

### 1. Use Canonical Names

Prefer canonical (lowercase) indicator names for consistency:

```python
# Preferred
indicators = ["rsi", "macd", "bbands"]

# Legacy (still supported)
indicators = ["RSI", "MACD", "BB_UPPER"]
```

### 2. Handle Errors Gracefully

Always handle potential errors in production code:

```python
try:
    result = await service.get_indicators(request)
    if result.get_indicator("rsi"):
        # Process RSI...
        pass
except IndicatorServiceError as e:
    logger.exception("Indicator calculation failed:")
    # Handle error appropriately
```

### 3. Use Batch Processing for Multiple Tickers

For multiple tickers, use batch processing for better performance:

```python
# Efficient
request = BatchIndicatorRequest(tickers=["AAPL", "GOOGL", "MSFT"], ...)
results = await service.get_batch_indicators(request)

# Inefficient
for ticker in ["AAPL", "GOOGL", "MSFT"]:
    request = IndicatorCalculationRequest(ticker=ticker, ...)
    result = await service.get_indicators(request)
```

### 4. Configure Appropriate Timeouts

For batch processing, configure appropriate timeouts:

```python
config = BatchProcessingConfig(
    max_concurrent=10,
    timeout_per_ticker=30.0,
    retry_attempts=2
)
```

### 5. Monitor Performance

Regularly check performance metrics in production:

```python
metrics = service.get_performance_metrics()
if metrics['statistics']['_overall']['overall_error_rate'] > 0.05:
    # Alert on high error rate
    pass
```

## Common Use Cases

### 1. Technical Analysis Dashboard

```python
async def get_technical_analysis(ticker: str):
    request = IndicatorCalculationRequest(
        ticker=ticker,
        indicators=["rsi", "macd", "bbands", "stoch", "adx"],
        timeframe="1d",
        period="6mo"
    )
    
    result = await service.get_indicators(request)
    
    analysis = {
        "ticker": ticker,
        "rsi": {
            "value": result.get_indicator("rsi").value,
            "signal": result.get_indicator("rsi").recommendation.recommendation
        },
        "macd": {
            "macd": result.get_indicator("macd").value.get("macd"),
            "signal": result.get_indicator("macd").value.get("signal"),
            "recommendation": result.get_indicator("macd").recommendation.recommendation
        },
        "overall": result.overall_recommendation.recommendation if result.overall_recommendation else "HOLD"
    }
    
    return analysis
```

### 2. Stock Screener

```python
async def screen_stocks(tickers: List[str], criteria: Dict[str, Any]):
    request = BatchIndicatorRequest(
        tickers=tickers,
        indicators=["rsi", "pe_ratio", "roe", "debt_to_equity"],
        timeframe="1d",
        period="1y"
    )
    
    results = await service.get_batch_indicators(request)
    
    screened = []
    for ticker, indicator_set in results.items():
        rsi = indicator_set.get_indicator("rsi")
        pe = indicator_set.get_indicator("pe_ratio")
        roe = indicator_set.get_indicator("roe")
        
        if (rsi and rsi.value < criteria.get("max_rsi", 70) and
            pe and pe.value < criteria.get("max_pe", 25) and
            roe and roe.value > criteria.get("min_roe", 0.15)):
            screened.append({
                "ticker": ticker,
                "rsi": rsi.value,
                "pe_ratio": pe.value,
                "roe": roe.value,
                "overall_score": indicator_set.composite_score
            })
    
    return sorted(screened, key=lambda x: x["overall_score"], reverse=True)
```

### 3. Risk Assessment

```python
async def assess_risk(ticker: str):
    request = IndicatorCalculationRequest(
        ticker=ticker,
        indicators=["atr", "beta", "debt_to_equity", "current_ratio"],
        timeframe="1d",
        period="1y"
    )
    
    result = await service.get_indicators(request)
    
    atr = result.get_indicator("atr")
    beta = result.get_indicator("beta")
    debt_ratio = result.get_indicator("debt_to_equity")
    liquidity = result.get_indicator("current_ratio")
    
    risk_score = 0
    if atr and atr.value > 5.0: risk_score += 1  # High volatility
    if beta and beta.value > 1.5: risk_score += 1  # High market risk
    if debt_ratio and debt_ratio.value > 1.0: risk_score += 1  # High leverage
    if liquidity and liquidity.value < 1.5: risk_score += 1  # Low liquidity
    
    risk_levels = ["LOW", "MODERATE", "HIGH", "VERY HIGH"]
    return {
        "ticker": ticker,
        "risk_level": risk_levels[min(risk_score, 3)],
        "risk_score": risk_score,
        "factors": {
            "volatility": atr.value if atr else None,
            "beta": beta.value if beta else None,
            "leverage": debt_ratio.value if debt_ratio else None,
            "liquidity": liquidity.value if liquidity else None
        }
    }
```

## Troubleshooting

### Common Issues

1. **Unknown Indicator Error**
   ```python
   # Check available indicators
   indicators = service.get_available_indicators()
   print("Available:", indicators["all"])
   ```

2. **Data Not Available**
   ```python
   # Try different provider or period
   request.provider = "alpha_vantage"  # Instead of yfinance
   request.period = "1y"  # Instead of "5y"
   ```

3. **Timeout Errors**
   ```python
   # Increase timeout for batch processing
   config = BatchProcessingConfig(timeout_per_ticker=60.0)
   ```

4. **High Error Rate**
   ```python
   # Check circuit breaker status
   status = service.get_circuit_breaker_status()
   # Reset if needed
   service.reset_circuit_breaker("data_provider")
   ```

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.getLogger("src.indicators").setLevel(logging.DEBUG)
```

## Support

For additional support:
- Check the [Configuration Guide](CONFIGURATION.md)
- Review [Migration Guide](MIGRATION_GUIDE.md)
- See [Developer Guide](DEVELOPER_GUIDE.md)
- Report issues in the project repository