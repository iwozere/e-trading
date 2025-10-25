# Unified Indicator Service Developer Guide

## Overview

This guide provides comprehensive information for developers working with or extending the Unified Indicator Service. It covers architecture, design decisions, extension points, testing procedures, and contribution guidelines.

## Architecture Overview

The Unified Indicator Service follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                             │
│  Trading Strategies │ Analytics │ Backtester │ API Endpoints │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Service Layer                             │
│  UnifiedIndicatorService │ ConfigManager │ RecommendationEngine │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Adapter Layer                             │
│  TA-Lib │ Pandas-TA │ Backtrader │ Fundamentals │ Custom    │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  Market Data │ Fundamentals │ Configuration │ Cache         │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. UnifiedIndicatorService
- **Location**: `src/indicators/service.py`
- **Purpose**: Main orchestrator for all indicator operations
- **Responsibilities**:
  - Request routing and validation
  - Adapter coordination
  - Error handling and recovery
  - Performance monitoring
  - Batch processing

#### 2. Adapter Pattern
- **Location**: `src/indicators/adapters/`
- **Purpose**: Abstract different calculation backends
- **Key Adapters**:
  - `TaLibAdapter`: TA-Lib calculations
  - `PandasTaAdapter`: pandas-ta calculations
  - `BacktraderAdapter`: Backtrader integration
  - `FundamentalsAdapter`: Fundamental data

#### 3. Configuration Manager
- **Location**: `src/indicators/config_manager.py`
- **Purpose**: Centralized parameter management
- **Features**:
  - Preset management
  - Runtime overrides
  - Parameter validation
  - Legacy name mapping

#### 4. Recommendation Engine
- **Location**: `src/indicators/recommendation_engine.py`
- **Purpose**: Generate trading recommendations
- **Features**:
  - Individual indicator recommendations
  - Composite recommendations
  - Context-aware analysis
  - Confidence scoring

#### 5. Registry System
- **Location**: `src/indicators/registry.py`
- **Purpose**: Indicator metadata and discovery
- **Features**:
  - Indicator definitions
  - Parameter schemas
  - Backend mappings
  - Legacy name resolution

## Design Decisions

### 1. Async-First Architecture

**Decision**: Implement async/await throughout the service
**Rationale**: 
- Better concurrency for batch operations
- Non-blocking I/O for external API calls
- Improved scalability

**Implementation**:
```python
async def get_indicators(self, request: IndicatorCalculationRequest) -> IndicatorSet:
    """All public methods are async."""
    # Async data fetching
    df = await self.circuit_breakers["data_provider"].call(
        asyncio.to_thread, get_ohlcv, req.ticker, req.timeframe, req.period, provider
    )
    
    # Async computation
    result_set = await self.compute_for_ticker(ticker_request)
    return result_set
```

### 2. Adapter Pattern for Backends

**Decision**: Use adapter pattern instead of direct backend integration
**Rationale**:
- Flexibility to add new backends
- Isolation of backend-specific logic
- Consistent interface across backends

**Implementation**:
```python
class BaseAdapter(ABC):
    @abstractmethod
    async def compute(self, name: str, df: pd.DataFrame, 
                     inputs: Dict[str, pd.Series], params: Dict[str, Any]) -> Dict[str, pd.Series]:
        pass
    
    @abstractmethod
    def supports(self, indicator_name: str) -> bool:
        pass
```

### 3. Registry-Based Indicator Discovery

**Decision**: Use centralized registry instead of dynamic discovery
**Rationale**:
- Better performance
- Explicit control over available indicators
- Clear metadata management

**Implementation**:
```python
INDICATOR_META = {
    "rsi": IndicatorMeta(
        name="rsi",
        kind="tech",
        inputs=["close"],
        outputs=["value"],
        providers=["ta-lib", "pandas-ta"],
        defaults={"timeperiod": 14}
    )
}
```

### 4. Pydantic for Data Validation

**Decision**: Use Pydantic models for request/response validation
**Rationale**:
- Automatic validation and serialization
- Better error messages
- Type safety

**Implementation**:
```python
class IndicatorCalculationRequest(BaseModel):
    ticker: str = Field(min_length=1)
    indicators: List[str] = Field(min_length=1)
    timeframe: str = Field(default="1d")
    
    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        if not v.isupper():
            v = v.upper()
        return v
```

### 5. Circuit Breaker Pattern

**Decision**: Implement circuit breakers for external dependencies
**Rationale**:
- Prevent cascade failures
- Graceful degradation
- Automatic recovery

**Implementation**:
```python
class CircuitBreaker:
    async def call(self, func, *args, **kwargs):
        if self.state.is_open and not self._should_attempt_reset():
            raise CircuitBreakerError(f"Circuit breaker {self.state.name} is open")
        
        try:
            result = await func(*args, **kwargs)
            self.state.success_count += 1
            return result
        except Exception as e:
            self.state.failure_count += 1
            raise
```

## Adding New Indicators

### 1. Define Indicator Metadata

Add the indicator to the registry in `src/indicators/registry.py`:

```python
INDICATOR_META["my_indicator"] = IndicatorMeta(
    name="my_indicator",
    kind="tech",  # or "fund"
    inputs=["close", "volume"],  # Required OHLCV columns
    outputs=["value"],  # or ["upper", "lower"] for multi-output
    providers=["ta-lib", "pandas-ta"],  # Supported backends
    defaults={"timeperiod": 14, "multiplier": 2.0},  # Default parameters
    description="My custom indicator"
)
```

### 2. Implement Adapter Support

Add support in relevant adapters:

#### TA-Lib Adapter (`src/indicators/adapters/ta_lib_adapter.py`)

```python
class TaLibAdapter(BaseAdapter):
    def __init__(self):
        self._indicators = {
            # ... existing indicators
            "my_indicator": self._compute_my_indicator,
        }
    
    async def _compute_my_indicator(self, df: pd.DataFrame, inputs: Dict[str, pd.Series], 
                                   params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Compute my custom indicator using TA-Lib."""
        import talib
        
        close = inputs["close"].values
        volume = inputs["volume"].values
        timeperiod = params.get("timeperiod", 14)
        multiplier = params.get("multiplier", 2.0)
        
        # Custom calculation
        result = talib.SMA(close, timeperiod) * multiplier
        
        return {"value": pd.Series(result, index=df.index)}
```

#### Pandas-TA Adapter (`src/indicators/adapters/pandas_ta_adapter.py`)

```python
class PandasTaAdapter(BaseAdapter):
    def __init__(self):
        self._indicators = {
            # ... existing indicators
            "my_indicator": self._compute_my_indicator,
        }
    
    async def _compute_my_indicator(self, df: pd.DataFrame, inputs: Dict[str, pd.Series], 
                                   params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Compute my custom indicator using pandas-ta."""
        import pandas_ta as ta
        
        timeperiod = params.get("timeperiod", 14)
        multiplier = params.get("multiplier", 2.0)
        
        # Use pandas-ta or custom pandas operations
        sma = ta.sma(inputs["close"], length=timeperiod)
        result = sma * multiplier
        
        return {"value": result}
```

### 3. Add Recommendation Logic

Add recommendation rules in `src/indicators/recommendation_engine.py`:

```python
class RecommendationEngine:
    def __init__(self):
        self._tech_map = {
            # ... existing indicators
            "my_indicator": self._rule_my_indicator,
        }
    
    def _rule_my_indicator(self, value: float, ctx: Dict[str, Any]) -> Tuple[RecommendationType, float, str]:
        """Recommendation logic for my indicator."""
        if value is None:
            return RecommendationType.HOLD, 0.3, "No data"
        
        current_price = ctx.get("current_price", 0)
        if current_price == 0:
            return RecommendationType.HOLD, 0.5, "No price context"
        
        # Custom recommendation logic
        if value > current_price * 1.1:
            return RecommendationType.BUY, 0.8, "Indicator above price threshold"
        elif value < current_price * 0.9:
            return RecommendationType.SELL, 0.8, "Indicator below price threshold"
        else:
            return RecommendationType.HOLD, 0.5, "Indicator neutral"
```

### 4. Add Tests

Create comprehensive tests in `src/indicators/tests/`:

```python
# test_my_indicator.py
import pytest
import pandas as pd
import numpy as np
from src.indicators.service import get_unified_indicator_service
from src.indicators.models import IndicatorCalculationRequest

class TestMyIndicator:
    @pytest.fixture
    def service(self):
        return get_unified_indicator_service()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_prices = close + np.random.randn(100) * 0.5
        volume = np.random.randint(1000, 10000, 100)
        
        return pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)
    
    @pytest.mark.asyncio
    async def test_my_indicator_calculation(self, service, sample_data):
        """Test my indicator calculation."""
        # Mock data provider
        with patch('src.common.get_ohlcv', return_value=sample_data):
            request = IndicatorCalculationRequest(
                ticker="TEST",
                indicators=["my_indicator"],
                timeframe="1d",
                period="1y"
            )
            
            result = await service.get_indicators(request)
            indicator = result.get_indicator("my_indicator")
            
            assert indicator is not None
            assert indicator.value is not None
            assert isinstance(indicator.value, float)
            assert indicator.recommendation is not None
    
    def test_my_indicator_parameters(self, service):
        """Test parameter handling."""
        params = service.config_manager.get_parameters("my_indicator")
        
        assert "timeperiod" in params
        assert "multiplier" in params
        assert params["timeperiod"] == 14
        assert params["multiplier"] == 2.0
    
    def test_my_indicator_recommendation(self, service):
        """Test recommendation logic."""
        rec = service.recommendation_engine.get_recommendation(
            "my_indicator", 110.0, {"current_price": 100.0}
        )
        
        assert rec.recommendation == RecommendationType.BUY
        assert rec.confidence > 0.7
        assert "above price threshold" in rec.reason
```

## Adding New Adapters

### 1. Create Adapter Class

Create a new adapter in `src/indicators/adapters/`:

```python
# my_custom_adapter.py
from typing import Dict, Any
import pandas as pd
from src.indicators.adapters.base_adapter import BaseAdapter

class MyCustomAdapter(BaseAdapter):
    """Custom adapter for my calculation backend."""
    
    def __init__(self):
        self._indicators = {
            "rsi": self._compute_rsi,
            "sma": self._compute_sma,
            # Add supported indicators
        }
    
    async def compute(self, name: str, df: pd.DataFrame, 
                     inputs: Dict[str, pd.Series], params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Compute indicator using custom backend."""
        if name not in self._indicators:
            raise ValueError(f"Indicator {name} not supported by MyCustomAdapter")
        
        return await self._indicators[name](df, inputs, params)
    
    def supports(self, indicator_name: str) -> bool:
        """Check if indicator is supported."""
        return indicator_name in self._indicators
    
    async def _compute_rsi(self, df: pd.DataFrame, inputs: Dict[str, pd.Series], 
                          params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Custom RSI implementation."""
        close = inputs["close"]
        timeperiod = params.get("timeperiod", 14)
        
        # Custom RSI calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=timeperiod).mean()
        avg_loss = loss.rolling(window=timeperiod).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return {"value": rsi}
    
    def get_supported_indicators(self) -> List[str]:
        """Get list of supported indicators."""
        return list(self._indicators.keys())
```

### 2. Register Adapter

Add the adapter to the service in `src/indicators/service.py`:

```python
class UnifiedIndicatorService:
    def __init__(self, prefer: Dict[str, int] | None = None):
        self.adapters = {
            "ta-lib": TaLibAdapter(),
            "pandas-ta": PandasTaAdapter(),
            "fundamentals": FundamentalsAdapter(),
            "my-custom": MyCustomAdapter(),  # Add new adapter
        }
```

### 3. Update Registry

Update indicator metadata to include the new adapter:

```python
INDICATOR_META["rsi"] = IndicatorMeta(
    name="rsi",
    kind="tech",
    inputs=["close"],
    outputs=["value"],
    providers=["ta-lib", "pandas-ta", "my-custom"],  # Add new provider
    defaults={"timeperiod": 14}
)
```

## Testing Guidelines

### Test Structure

The testing suite is organized in `src/indicators/tests/`:

```
src/indicators/tests/
├── __init__.py
├── README.md
├── conftest.py                    # Shared fixtures
├── test_core_functionality.py    # Core service tests
├── test_config_validation.py     # Configuration tests
├── test_batch_processing.py      # Batch processing tests
├── test_adapter_integration.py   # Adapter tests
├── test_error_handling_fallbacks.py  # Error handling tests
├── test_migration_compatibility.py   # Migration tests
├── test_performance_benchmarks.py    # Performance tests
└── run_comprehensive_tests.py    # Test runner
```

### Writing Tests

#### Unit Tests

Focus on individual components:

```python
import pytest
from unittest.mock import Mock, patch
from src.indicators.config_manager import UnifiedConfigManager

class TestConfigManager:
    @pytest.fixture
    def config_manager(self):
        return UnifiedConfigManager("test_config.json")
    
    def test_get_parameters_default(self, config_manager):
        """Test getting default parameters."""
        params = config_manager.get_parameters("rsi")
        assert params["timeperiod"] == 14
    
    def test_parameter_override(self, config_manager):
        """Test parameter override functionality."""
        config_manager.set_parameter_override("rsi", "timeperiod", 21)
        params = config_manager.get_parameters("rsi")
        assert params["timeperiod"] == 21
    
    def test_preset_switching(self, config_manager):
        """Test preset switching."""
        success = config_manager.set_preset("aggressive")
        assert success
        
        params = config_manager.get_parameters("rsi")
        assert params["timeperiod"] == 7  # Aggressive preset value
```

#### Integration Tests

Test component interactions:

```python
@pytest.mark.asyncio
class TestServiceIntegration:
    @pytest.fixture
    def service(self):
        return get_unified_indicator_service()
    
    async def test_full_calculation_flow(self, service):
        """Test complete calculation flow."""
        with patch('src.common.get_ohlcv') as mock_data:
            mock_data.return_value = create_sample_ohlcv_data()
            
            request = IndicatorCalculationRequest(
                ticker="AAPL",
                indicators=["rsi", "macd"],
                timeframe="1d",
                period="1y"
            )
            
            result = await service.get_indicators(request)
            
            assert result.ticker == "AAPL"
            assert result.get_indicator("rsi") is not None
            assert result.get_indicator("macd") is not None
```

#### Performance Tests

Test performance characteristics:

```python
@pytest.mark.performance
class TestPerformance:
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self, service):
        """Test batch processing performance."""
        tickers = [f"STOCK{i}" for i in range(50)]
        
        start_time = time.time()
        
        request = BatchIndicatorRequest(
            tickers=tickers,
            indicators=["rsi", "macd"],
            max_concurrent=10
        )
        
        with patch('src.common.get_ohlcv') as mock_data:
            mock_data.return_value = create_sample_ohlcv_data()
            result = await service.get_batch_indicators(request)
        
        duration = time.time() - start_time
        
        assert len(result) == 50
        assert duration < 30.0  # Should complete within 30 seconds
        
        # Calculate throughput
        throughput = len(tickers) / duration
        assert throughput > 5.0  # At least 5 tickers per second
```

### Running Tests

#### Individual Test Files

```bash
# Run specific test file
pytest src/indicators/tests/test_core_functionality.py -v

# Run with coverage
pytest src/indicators/tests/test_core_functionality.py --cov=src.indicators

# Run performance tests
pytest src/indicators/tests/test_performance_benchmarks.py -m performance
```

#### Comprehensive Test Suite

```bash
# Run all tests
python src/indicators/tests/run_comprehensive_tests.py

# Run with specific options
python src/indicators/tests/run_comprehensive_tests.py --performance --integration
```

#### Test Configuration

Configure pytest in `pytest.ini`:

```ini
[tool:pytest]
testpaths = src/indicators/tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow tests
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
```

## Performance Optimization

### 1. Async Operations

Use async/await for I/O operations:

```python
# Good: Non-blocking
async def fetch_data_async(ticker):
    return await asyncio.to_thread(get_ohlcv, ticker, "1d", "1y")

# Bad: Blocking
def fetch_data_sync(ticker):
    return get_ohlcv(ticker, "1d", "1y")
```

### 2. Batch Processing

Process multiple items concurrently:

```python
async def process_batch(tickers, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_ticker(ticker):
        async with semaphore:
            return await calculate_indicators(ticker)
    
    tasks = [process_ticker(ticker) for ticker in tickers]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Caching Strategy

Implement intelligent caching:

```python
class CacheManager:
    def __init__(self, max_size=1000, ttl=3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get_cache_key(self, ticker, indicators, timeframe, period):
        return f"{ticker}:{':'.join(sorted(indicators))}:{timeframe}:{period}"
    
    def get(self, key):
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:
                del self.cache[key]
        return None
    
    def set(self, key, data):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
```

### 4. Memory Management

Optimize memory usage:

```python
def optimize_dataframe(df):
    """Optimize DataFrame memory usage."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
```

## Error Handling Best Practices

### 1. Custom Exception Hierarchy

```python
class IndicatorServiceError(Exception):
    """Base exception for indicator service."""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()

class ConfigurationError(IndicatorServiceError):
    """Configuration-related errors."""
    pass

class DataError(IndicatorServiceError):
    """Data availability or quality errors."""
    pass

class CalculationError(IndicatorServiceError):
    """Calculation errors."""
    pass
```

### 2. Error Recovery Strategies

```python
class ErrorRecoveryStrategy:
    @staticmethod
    def should_retry(error: Exception, attempt: int, max_attempts: int) -> bool:
        """Determine if error should trigger retry."""
        if isinstance(error, ConfigurationError):
            return False  # Don't retry config errors
        
        if isinstance(error, (ConnectionError, TimeoutError)):
            return attempt < max_attempts  # Retry network errors
        
        return False
    
    @staticmethod
    def get_retry_delay(error: Exception, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        base_delay = 1.0
        if isinstance(error, ConnectionError):
            base_delay = 2.0
        
        return base_delay * (2 ** attempt) + random.uniform(0, 1)
```

### 3. Circuit Breaker Implementation

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    async def call(self, func, *args, **kwargs):
        if self.is_open:
            if self._should_attempt_reset():
                self.is_open = False
                self.failure_count = 0
            else:
                raise CircuitBreakerError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self.failure_count = 0  # Reset on success
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            
            raise
    
    def _should_attempt_reset(self):
        return (self.last_failure_time and 
                time.time() - self.last_failure_time > self.recovery_timeout)
```

## Contribution Guidelines

### 1. Code Style

Follow the project's coding conventions:

```python
# Good: Clear, documented function
async def calculate_rsi(close_prices: pd.Series, timeperiod: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Args:
        close_prices: Series of closing prices
        timeperiod: Period for RSI calculation
        
    Returns:
        Series of RSI values
        
    Raises:
        ValueError: If timeperiod is invalid
    """
    if timeperiod <= 0:
        raise ValueError("timeperiod must be positive")
    
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=timeperiod).mean()
    avg_loss = loss.rolling(window=timeperiod).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
```

### 2. Documentation Standards

Document all public APIs:

```python
class MyIndicatorAdapter(BaseAdapter):
    """
    Custom adapter for my indicator calculations.
    
    This adapter provides implementations for custom indicators
    using proprietary calculation methods.
    
    Attributes:
        supported_indicators: List of supported indicator names
        
    Example:
        >>> adapter = MyIndicatorAdapter()
        >>> result = await adapter.compute("my_rsi", df, inputs, params)
    """
    
    async def compute(self, name: str, df: pd.DataFrame, 
                     inputs: Dict[str, pd.Series], params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """
        Compute indicator values.
        
        Args:
            name: Indicator name
            df: OHLCV DataFrame
            inputs: Input series (close, high, low, etc.)
            params: Calculation parameters
            
        Returns:
            Dictionary mapping output names to result series
            
        Raises:
            ValueError: If indicator is not supported
            CalculationError: If calculation fails
        """
        pass
```

### 3. Testing Requirements

All contributions must include tests:

```python
class TestMyFeature:
    """Test suite for my new feature."""
    
    def test_basic_functionality(self):
        """Test basic functionality works correctly."""
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        pass
    
    def test_performance(self):
        """Test performance characteristics."""
        pass
    
    @pytest.mark.asyncio
    async def test_async_behavior(self):
        """Test async behavior if applicable."""
        pass
```

### 4. Pull Request Process

1. **Fork and Branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Implement Changes**
   - Write code following style guidelines
   - Add comprehensive tests
   - Update documentation

3. **Test Thoroughly**
   ```bash
   # Run all tests
   python src/indicators/tests/run_comprehensive_tests.py
   
   # Check coverage
   pytest --cov=src.indicators --cov-report=html
   ```

4. **Submit Pull Request**
   - Clear description of changes
   - Reference related issues
   - Include test results

### 5. Code Review Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Performance impact is acceptable
- [ ] Error handling is appropriate
- [ ] Backward compatibility is maintained

## Debugging and Troubleshooting

### 1. Enable Debug Logging

```python
import logging

# Enable debug logging for indicators
logging.getLogger("src.indicators").setLevel(logging.DEBUG)

# Enable debug logging for specific components
logging.getLogger("src.indicators.service").setLevel(logging.DEBUG)
logging.getLogger("src.indicators.adapters").setLevel(logging.DEBUG)
```

### 2. Performance Profiling

```python
import cProfile
import pstats

async def profile_calculation():
    """Profile indicator calculation performance."""
    service = get_unified_indicator_service()
    
    request = IndicatorCalculationRequest(
        ticker="AAPL",
        indicators=["rsi", "macd", "bbands"],
        timeframe="1d",
        period="2y"
    )
    
    # Profile the calculation
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = await service.get_indicators(request)
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### 3. Memory Usage Analysis

```python
import tracemalloc
import asyncio

async def analyze_memory_usage():
    """Analyze memory usage during calculation."""
    tracemalloc.start()
    
    service = get_unified_indicator_service()
    
    # Take snapshot before
    snapshot1 = tracemalloc.take_snapshot()
    
    # Perform calculation
    request = BatchIndicatorRequest(
        tickers=["AAPL", "GOOGL", "MSFT"] * 10,
        indicators=["rsi", "macd", "bbands"]
    )
    
    result = await service.get_batch_indicators(request)
    
    # Take snapshot after
    snapshot2 = tracemalloc.take_snapshot()
    
    # Analyze difference
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("Top 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
```

### 4. Common Issues and Solutions

#### Issue: High Memory Usage

**Symptoms**: Memory usage grows during batch processing
**Solution**: 
```python
# Optimize DataFrame memory usage
def optimize_memory(df):
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# Process in smaller batches
async def process_large_batch(tickers, batch_size=20):
    results = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_results = await service.get_batch_indicators(
            BatchIndicatorRequest(tickers=batch, ...)
        )
        results.update(batch_results)
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return results
```

#### Issue: Slow Performance

**Symptoms**: Calculations take too long
**Solution**:
```python
# Increase concurrency
config = BatchProcessingConfig(
    max_concurrent=20,  # Increase from default 10
    timeout_per_ticker=60.0
)

# Use performance monitoring
metrics = service.get_performance_metrics()
print(f"Average calculation time: {metrics['avg_duration']}")

# Profile bottlenecks
await profile_calculation()
```

#### Issue: Frequent Timeouts

**Symptoms**: TimeoutError exceptions
**Solution**:
```python
# Increase timeouts
config = BatchProcessingConfig(
    timeout_per_ticker=120.0,  # Increase timeout
    retry_attempts=3
)

# Check circuit breaker status
status = service.get_circuit_breaker_status()
for name, info in status.items():
    if info['is_open']:
        print(f"Circuit breaker {name} is open")
        service.reset_circuit_breaker(name)
```

## Advanced Topics

### 1. Custom Recommendation Engines

```python
class CustomRecommendationEngine(BaseRecommendationEngine):
    """Custom recommendation engine with ML-based scoring."""
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    def get_recommendation(self, indicator_name: str, value: float, 
                          context: Dict[str, Any] = None) -> Recommendation:
        """Get ML-based recommendation."""
        features = self._extract_features(indicator_name, value, context)
        prediction = self.model.predict([features])[0]
        confidence = self.model.predict_proba([features])[0].max()
        
        return Recommendation(
            recommendation=self._map_prediction(prediction),
            confidence=confidence,
            reason=f"ML model prediction: {prediction}"
        )
```

### 2. Custom Data Providers

```python
class CustomDataProvider:
    """Custom data provider for alternative data sources."""
    
    async def get_ohlcv(self, ticker: str, timeframe: str, period: str) -> pd.DataFrame:
        """Fetch OHLCV data from custom source."""
        # Implement custom data fetching logic
        pass
    
    async def get_fundamentals(self, ticker: str) -> Dict[str, float]:
        """Fetch fundamental data from custom source."""
        # Implement custom fundamental data fetching
        pass

# Register custom provider
from src.common import register_data_provider
register_data_provider("custom", CustomDataProvider())
```

### 3. Real-time Indicator Updates

```python
class RealTimeIndicatorService:
    """Service for real-time indicator updates."""
    
    def __init__(self, base_service: UnifiedIndicatorService):
        self.base_service = base_service
        self.subscribers = {}
    
    async def subscribe(self, ticker: str, indicators: List[str], 
                       callback: Callable[[IndicatorSet], None]):
        """Subscribe to real-time indicator updates."""
        if ticker not in self.subscribers:
            self.subscribers[ticker] = []
        
        self.subscribers[ticker].append({
            'indicators': indicators,
            'callback': callback
        })
    
    async def update_ticker(self, ticker: str, new_price_data: Dict[str, float]):
        """Update indicators with new price data."""
        if ticker not in self.subscribers:
            return
        
        for subscription in self.subscribers[ticker]:
            # Calculate updated indicators
            request = IndicatorCalculationRequest(
                ticker=ticker,
                indicators=subscription['indicators']
            )
            
            result = await self.base_service.get_indicators(request)
            subscription['callback'](result)
```

This developer guide provides comprehensive information for working with and extending the Unified Indicator Service. It covers architecture, design decisions, extension points, testing, and contribution guidelines to help developers effectively work with the system.