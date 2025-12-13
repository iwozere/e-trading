# Architecture Review: `src/data/downloader` Folder

## Executive Summary

The `src/data/downloader` folder implements a data downloader system with 17+ downloader implementations for various financial data providers. The architecture uses a factory pattern with a base class, but has several areas that need improvement for better maintainability, extensibility, and consistency.

---

## Current Architecture Overview

### Structure
- **Base Class**: `BaseDataDownloader` (abstract interface)
- **Factory**: `DataDownloaderFactory` (creates downloader instances)
- **Implementations**: 10+ downloaders inheriting from `BaseDataDownloader`
- **Special Cases**: 5+ downloaders NOT inheriting from base (eodhd, tradier, vix, finra, finra_trf)

### Key Components
1. **BaseDataDownloader**: Defines interface (`get_ohlcv`, `get_fundamentals`, `get_supported_intervals`)
2. **DataDownloaderFactory**: Maps provider codes to downloader classes
3. **Downloader Implementations**: Provider-specific logic

---

## Recommendations by Priority

### 🔴 **PRIORITY 1: Critical Issues (Fix Immediately)**

#### 1.1 **Inconsistent Inheritance Pattern**
**Issue**: Several downloaders don't inherit from `BaseDataDownloader`:
- ~~`EODHDApiError` / `eodhd_downloader.py` - standalone functions~~ ✅ **COMPLETED**: Converted to `EODHDDataDownloader`
- ~~`TradierDownloader` - standalone class~~ ✅ **COMPLETED**: Converted to `TradierDataDownloader`
- ~~`vix_downloader.py` - standalone functions~~ ✅ **COMPLETED**: Converted to `VIXDataDownloader`
- ~~`FINRADataDownloader` - standalone class~~ ✅ **COMPLETED**: Consolidated into `FinraDataDownloader`
- ~~`FinraTRFDownloader` - standalone class~~ ✅ **COMPLETED**: Consolidated into `FinraDataDownloader`

**Impact**: 
- Cannot use factory pattern for these downloaders
- Inconsistent API across downloaders
- Difficult to add common functionality

**Recommendation**:
```python
# Refactor to inherit from BaseDataDownloader
class EODHDDataDownloader(BaseDataDownloader):
    def get_supported_intervals(self) -> List[str]:
        return ['1d']  # EODHD specific intervals
    
    def get_ohlcv(self, symbol: str, interval: str, 
                  start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        # Implement using existing eodhd logic
        pass
```

**Action Items**:
- [x] Refactor `EODHDDataDownloader` to inherit from `BaseDataDownloader` ✅ **COMPLETED**: Converted to `EODHDDataDownloader` class
- [x] Refactor `TradierDownloader` to inherit from `BaseDataDownloader` ✅ **COMPLETED**: Converted to `TradierDataDownloader` class
- [x] Refactor `vix_downloader.py` to inherit from `BaseDataDownloader` ✅ **COMPLETED**: Converted to `VIXDataDownloader` class
- [x] Refactor `FINRADataDownloader` to inherit from `BaseDataDownloader` ✅ **COMPLETED**: Consolidated into `FinraDataDownloader` which inherits from `BaseDataDownloader`
- [x] Refactor `FinraTRFDownloader` to inherit from `BaseDataDownloader` ✅ **COMPLETED**: Consolidated into `FinraDataDownloader` which inherits from `BaseDataDownloader`

**Completed Changes (1.1)**:
- ✅ Consolidated `FINRADataDownloader` and `FinraTRFDownloader` into a single `FinraDataDownloader` class
- ✅ `FinraDataDownloader` now inherits from `BaseDataDownloader`
- ✅ Implemented abstract methods: `get_supported_intervals()` (returns empty list) and `get_ohlcv()` (returns empty DataFrame with warning)
- ✅ Updated all callers (see "FinraDataDownloader Callers" section below)
- ✅ Maintained backward compatibility with `create_finra_downloader()` factory function
- ✅ All existing methods from both classes preserved in consolidated class

**FinraDataDownloader Callers**:
All callers have been updated to use the consolidated `FinraDataDownloader` class:

1. **`src/ml/pipeline/p06_emps2/trf_downloader.py`**
   - Function: `download_trf()` (line 66)
   - Usage: Creates `FinraDataDownloader` instance for TRF data downloads
   - Method called: `run()`

2. **`src/ml/pipeline/p06_emps2/emps2_pipeline.py`**
   - Function: `_stage2b_download_trf_data()` (line 343)
   - Usage: Downloads TRF data for volume correction in EMPS2 pipeline
   - Method called: `run()`

3. **`src/ml/pipeline/p06_emps2/backfill_trf_data.py`**
   - Function: `backfill_trf_data()` (line 82)
   - Usage: Downloads TRF data for multiple historical dates
   - Method called: `run()`

4. **`src/ml/pipeline/p04_short_squeeze/core/weekly_screener.py`**
   - Function: `__init__()` (line 63)
   - Usage: Creates FINRA downloader via factory function for short interest data
   - Method called: Uses `create_finra_downloader()` factory function which returns `FinraDataDownloader`
   - Methods used: `get_short_interest_data()`, `get_bulk_short_interest()`, etc.

---

#### 1.2 **Factory Missing Downloaders**
**Issue**: `DataDownloaderFactory` doesn't include:
- ~~EODHD~~ ✅ **COMPLETED**: Added to factory
- ~~Tradier~~ ✅ **COMPLETED**: Added to factory (using code "trdr" to resolve conflict)
- ~~FINRA~~ ✅ **COMPLETED**: Added to factory
- ~~FINRA TRF~~ ✅ **COMPLETED**: Added to factory (maps to same FinraDataDownloader)
- ~~VIX~~ ✅ **COMPLETED**: Added to factory

**Impact**: 
- These downloaders can't be created via factory
- Inconsistent usage patterns across codebase
- Direct imports scattered throughout codebase

**Recommendation**:
```python
# Add to PROVIDER_MAP
PROVIDER_MAP = {
    # ... existing ...
    "eodhd": "eodhd",
    "eod": "eodhd",
    "tradier": "tradier",
    "td": "tradier",  # Note: conflicts with "twelvedata"
    "finra": "finra",
    "finra_trf": "finra",
}

# Add to _get_downloader_class
downloader_classes = {
    # ... existing ...
    "eodhd": EODHDDataDownloader,
    "tradier": TradierDataDownloader,
    "finra": FinraDataDownloader,  # Note: Consolidated from FINRADataDownloader and FinraTRFDownloader
}
```

**Action Items**:
- [x] Add FINRA downloaders to factory ✅ **COMPLETED**: Added `FinraDataDownloader` to factory with codes "finra" and "finra_trf"
- [x] Add EODHD to factory ✅ **COMPLETED**: Added `EODHDDataDownloader` to factory with codes "eodhd" and "eod"
- [x] Add Tradier to factory ✅ **COMPLETED**: Added `TradierDataDownloader` to factory with codes "trdr" and "tradier" (resolved conflict with "td" used by twelvedata)
- [x] Add VIX to factory ✅ **COMPLETED**: Added `VIXDataDownloader` to factory with code "vix"
- [x] Resolve provider code conflicts ✅ **COMPLETED**: Used "trdr" for Tradier to avoid conflict with "td" (twelvedata)
- [x] Update factory instantiation logic ✅ **COMPLETED**: Added instantiation logic for all new downloaders

**Completed Changes (1.2)**:
- ✅ Added `FinraDataDownloader` import to factory
- ✅ Added "finra" and "finra_trf" provider codes to `PROVIDER_MAP` (both map to "finra")
- ✅ Added `EODHDDataDownloader` import to factory
- ✅ Added "eodhd" and "eod" provider codes to `PROVIDER_MAP` (both map to "eodhd")
- ✅ Added `TradierDataDownloader` import to factory
- ✅ Added "trdr" and "tradier" provider codes to `PROVIDER_MAP` (both map to "tradier")
  - Used "trdr" to resolve conflict with "td" used by twelvedata
- ✅ Added `VIXDataDownloader` import to factory
- ✅ Added "vix" provider code to `PROVIDER_MAP`
- ✅ Added all new downloaders to `_get_downloader_class()` method
- ✅ Added instantiation logic in `_create_downloader_instance()` for all new downloaders:
  - **FINRA**: `rate_limit_delay`, `date`, `output_dir`, `output_filename`, `fetch_yfinance_data`
  - **EODHD**: `api_key` (from env or kwargs)
  - **Tradier**: `api_key` (from env or kwargs), `rate_limit_sleep` (default: 0.3)
  - **VIX**: No parameters required
- ✅ Added all new downloaders to `get_provider_info()` method with complete provider information
- ✅ Updated factory documentation to include all new providers in supported providers list

---

#### 1.3 **Hardcoded Configuration Imports**
**Issue**: Downloaders directly import from `config.donotshare.donotshare`:
```python
from config.donotshare.donotshare import POLYGON_KEY
self.api_key = api_key or POLYGON_KEY
```

**Impact**:
- Tight coupling to configuration structure
- Difficult to test (requires config module)
- Cannot use different config sources
- Breaks if config structure changes

**Recommendation**:
```python
# In BaseDataDownloader or factory
class BaseDataDownloader(ABC):
    @staticmethod
    def _get_config_value(key: str, env_var: str, default: Optional[str] = None) -> Optional[str]:
        """Get config value from environment or config module."""
        import os
        # Try environment variable first
        value = os.getenv(env_var)
        if value:
            return value
        
        # Fallback to config module
        try:
            from config.donotshare.donotshare import get_config
            return get_config(key)
        except (ImportError, AttributeError):
            return default
```

**Action Items**:
- [x] Create centralized config accessor ✅ **COMPLETED**: Added `_get_config_value()` static method to `BaseDataDownloader`
- [x] Refactor all downloaders to use centralized config ✅ **COMPLETED**: All downloaders updated
- [x] Support environment variables as primary source ✅ **COMPLETED**: Method prioritizes environment variables
- [x] Add config validation ✅ **COMPLETED**: Method handles ImportError and AttributeError gracefully

**Completed Changes (1.3)**:
- ✅ Added `_get_config_value()` static method to `BaseDataDownloader` class
  - Method prioritizes environment variables (if `env_var` is provided)
  - Falls back to `config.donotshare.donotshare` module using `getattr()`
  - Returns `default` value if not found
  - Handles `ImportError` and `AttributeError` gracefully
- ✅ Updated all downloaders to use `_get_config_value()` instead of direct imports:
  - `AlphaVantageDataDownloader`: Uses `_get_config_value('ALPHA_VANTAGE_KEY', 'ALPHA_VANTAGE_KEY')`
  - `FinnhubDataDownloader`: Uses `_get_config_value('FINNHUB_KEY', 'FINNHUB_KEY')`
  - `PolygonDataDownloader`: Uses `_get_config_value('POLYGON_KEY', 'POLYGON_KEY')`
  - `TwelveDataDataDownloader`: Uses `_get_config_value('TWELVE_DATA_KEY', 'TWELVE_DATA_KEY')`
  - `FMPDataDownloader`: Uses `_get_config_value('FMP_API_KEY', 'FMP_API_KEY')`
  - `TiingoDataDownloader`: Uses `_get_config_value('TIINGO_API_KEY', 'TIINGO_API_KEY')`
  - `AlpacaDataDownloader`: Uses `_get_config_value()` for both `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`
  - `EODHDDataDownloader`: Uses `_get_config_value('EODHD_API_KEY', 'EODHD_API_KEY')`
  - `FinraDataDownloader`: Uses `_get_config_value()` for both `FINRA_API_CLIENT` and `FINRA_API_SECRET`
- ✅ Removed all direct imports from `config.donotshare.donotshare` in downloader files
- ✅ Environment variables are now the primary source (checked first)
- ✅ Config module is fallback (checked second)
- ✅ All changes maintain backward compatibility with existing configuration structure

---

### 🟠 **PRIORITY 2: High Impact (Fix Soon)**

#### 2.1 **Factory Instantiation Logic is Monolithic**
**Issue**: `_create_downloader_instance` has a large if/elif chain (50+ lines):
```python
if provider == "alphavantage":
    # ... 5 lines ...
elif provider == "finnhub":
    # ... 5 lines ...
elif provider == "polygon":
    # ... 5 lines ...
# ... 10 more elif blocks ...
```

**Impact**:
- Hard to maintain (adding new provider requires modifying factory)
- Violates Open/Closed Principle
- Difficult to test individual provider instantiation
- Error-prone (easy to miss a provider)

**Recommendation**: Use Strategy Pattern or Registration Pattern
```python
class DownloaderConfig:
    """Configuration for a downloader."""
    def __init__(self, 
                 api_key_env: Optional[str] = None,
                 secret_key_env: Optional[str] = None,
                 requires_api_key: bool = True,
                 requires_secret: bool = False):
        self.api_key_env = api_key_env
        self.secret_key_env = secret_key_env
        self.requires_api_key = requires_api_key
        self.requires_secret = requires_secret
    
    def create_instance(self, downloader_class: Type[BaseDataDownloader], **kwargs) -> BaseDataDownloader:
        """Create downloader instance with proper config."""
        params = {}
        if self.api_key_env:
            params['api_key'] = kwargs.get('api_key') or os.getenv(self.api_key_env)
        if self.secret_key_env:
            params['secret_key'] = kwargs.get('secret_key') or os.getenv(self.secret_key_env)
        
        # Validate required params
        if self.requires_api_key and not params.get('api_key'):
            raise ValueError(f"API key required for {downloader_class.__name__}")
        
        return downloader_class(**params)

# Register configurations
DOWNLOADER_CONFIGS = {
    "alphavantage": DownloaderConfig(
        api_key_env="ALPHA_VANTAGE_KEY",
        requires_api_key=True
    ),
    "finnhub": DownloaderConfig(
        api_key_env="FINNHUB_KEY",
        requires_api_key=True
    ),
    # ... etc
}
```

**Action Items**:
- [ ] Create `DownloaderConfig` class
- [ ] Move instantiation logic to configuration objects
- [ ] Register all providers with their configs
- [ ] Update factory to use config registry

---

#### 2.2 **No Standardized Error Handling**
**Issue**: Each downloader handles errors differently:
- Some return empty DataFrames
- Some raise exceptions
- Some log and continue
- Some return None

**Impact**:
- Inconsistent behavior for callers
- Difficult to handle errors uniformly
- Poor user experience

**Recommendation**: Standardize error handling in base class
```python
class BaseDataDownloader(ABC):
    class DownloadError(Exception):
        """Base exception for download errors."""
        pass
    
    class RateLimitError(DownloadError):
        """Rate limit exceeded."""
        pass
    
    class InvalidSymbolError(DownloadError):
        """Invalid symbol provided."""
        pass
    
    class NoDataError(DownloadError):
        """No data available for request."""
        pass
    
    def get_ohlcv(self, symbol: str, interval: str, 
                  start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Download with standardized error handling."""
        try:
            return self._fetch_ohlcv(symbol, interval, start_date, end_date, **kwargs)
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                raise self.RateLimitError(f"Rate limit exceeded for {symbol}") from e
            raise self.DownloadError(f"HTTP error downloading {symbol}") from e
        except Exception as e:
            _logger.exception("Error downloading %s:", symbol)
            raise self.DownloadError(f"Failed to download {symbol}") from e
    
    @abstractmethod
    def _fetch_ohlcv(self, symbol: str, interval: str, 
                     start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """Internal method for actual fetching (implemented by subclasses)."""
        pass
```

**Action Items**:
- [ ] Define standard exception hierarchy
- [ ] Add error handling wrapper in base class
- [ ] Refactor all downloaders to use standard exceptions
- [ ] Update callers to handle standard exceptions

---

#### 2.3 **Code Duplication: Chunking Logic**
**Issue**: Multiple downloaders implement similar chunking logic:
- `YahooDataDownloader._download_ohlcv_batched`
- `AlpacaDataDownloader._download_with_chunking`
- `BinanceDataDownloader._calculate_batch_dates`

**Impact**:
- Code duplication
- Bugs fixed in one place not fixed in others
- Inconsistent chunking behavior

**Recommendation**: Extract to base class or utility
```python
class BaseDataDownloader(ABC):
    def _chunk_date_range(self, start_date: datetime, end_date: datetime,
                          max_bars: int, interval: str) -> List[Tuple[datetime, datetime]]:
        """Calculate date chunks respecting bar limits."""
        interval_minutes = self._interval_to_minutes(interval)
        max_minutes = interval_minutes * max_bars
        max_timedelta = timedelta(minutes=max_minutes)
        
        chunks = []
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + max_timedelta, end_date)
            chunks.append((current_start, current_end))
            current_start = current_end
        return chunks
    
    def _interval_to_minutes(self, interval: str) -> int:
        """Convert interval string to minutes."""
        # Default implementation, can be overridden
        mapping = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '1d': 1440, '1w': 10080
        }
        return mapping.get(interval, 1440)
```

**Action Items**:
- [ ] Extract chunking logic to base class
- [ ] Create interval conversion utilities
- [ ] Refactor all downloaders to use shared chunking
- [ ] Add unit tests for chunking logic

---

### 🟡 **PRIORITY 3: Medium Impact (Improve Over Time)**

#### 3.1 **No Retry Mechanism**
**Issue**: Downloaders don't have standardized retry logic for transient failures.

**Recommendation**: Add retry decorator/utility
```python
from functools import wraps
from typing import Callable, Type, Tuple
import time

def retry_on_failure(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (requests.RequestException,)
):
    """Retry decorator for downloader methods."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(self, *args, **kwargs)
                except retryable_exceptions as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor * (2 ** attempt)
                    _logger.warning("Retry %d/%d after %s seconds: %s", 
                                   attempt + 1, max_retries, wait_time, e)
                    time.sleep(wait_time)
        return wrapper
    return decorator
```

**Action Items**:
- [ ] Create retry utility
- [ ] Apply to critical download methods
- [ ] Make retry configurable per provider

---

#### 3.2 **No Health Check Interface**
**Issue**: No standardized way to check if a downloader is working.

**Recommendation**: Add health check to base class
```python
class BaseDataDownloader(ABC):
    def health_check(self) -> Dict[str, Any]:
        """
        Check downloader health.
        
        Returns:
            Dict with 'status', 'latency_ms', 'error' keys
        """
        import time
        start = time.time()
        try:
            # Test with a well-known symbol
            test_symbol = self.get_test_symbol()
            df = self.get_ohlcv(
                test_symbol,
                self.get_supported_intervals()[0],
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
            latency_ms = (time.time() - start) * 1000
            return {
                'status': 'healthy' if not df.empty else 'degraded',
                'latency_ms': latency_ms,
                'error': None
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'latency_ms': (time.time() - start) * 1000,
                'error': str(e)
            }
    
    def get_test_symbol(self) -> str:
        """Get a test symbol for health checks."""
        return "AAPL"  # Default, can be overridden
```

**Action Items**:
- [ ] Add health_check method to base class
- [ ] Implement in all downloaders
- [ ] Add health monitoring endpoint/utility

---

#### 3.3 **Inconsistent Data Format**
**Issue**: Some downloaders return DataFrames with index, some with 'timestamp' column.

**Recommendation**: Standardize on one format
```python
class BaseDataDownloader(ABC):
    def get_ohlcv(self, ...) -> pd.DataFrame:
        """Returns DataFrame with 'timestamp' column (not index)."""
        df = self._fetch_ohlcv(...)
        return self._normalize_dataframe(df)
    
    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to standard format."""
        # Ensure timestamp is a column, not index
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove timezone if present
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        return df[required_cols].sort_values('timestamp')
```

**Action Items**:
- [ ] Define standard DataFrame format
- [ ] Add normalization method to base class
- [ ] Refactor all downloaders to use normalization
- [ ] Update tests to verify format

---

#### 3.4 **Missing Type Hints and Documentation**
**Issue**: Some methods lack type hints, inconsistent docstring formats.

**Recommendation**: 
- Use consistent docstring format (Google or NumPy style)
- Add type hints to all public methods
- Use `Protocol` for interfaces if needed

**Action Items**:
- [ ] Add type hints to all base class methods
- [ ] Standardize docstring format
- [ ] Add type hints to all downloader implementations
- [ ] Use mypy for type checking

---

### 🟢 **PRIORITY 4: Nice to Have (Future Improvements)**

#### 4.1 **Plugin/Registration System**
**Issue**: Adding new downloaders requires modifying factory code.

**Recommendation**: Use plugin registration
```python
# Auto-discovery of downloaders
DOWNLOADER_REGISTRY = {}

def register_downloader(provider_code: str, provider_name: str):
    """Decorator to register downloaders."""
    def decorator(cls: Type[BaseDataDownloader]):
        DOWNLOADER_REGISTRY[provider_code] = {
            'class': cls,
            'name': provider_name
        }
        return cls
    return decorator

@register_downloader("yf", "yahoo")
class YahooDataDownloader(BaseDataDownloader):
    pass
```

**Action Items**:
- [ ] Implement registration decorator
- [ ] Refactor existing downloaders to use registration
- [ ] Update factory to use registry

---

#### 4.2 **Rate Limiting Abstraction**
**Issue**: Each downloader implements rate limiting differently.

**Recommendation**: Create rate limiter utility
```python
from collections import deque
import time

class RateLimiter:
    """Token bucket rate limiter."""
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls = deque()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        # Remove old calls
        while self.calls and self.calls[0] < now - self.period_seconds:
            self.calls.popleft()
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period_seconds - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.calls.popleft()
        
        self.calls.append(time.time())
```

**Action Items**:
- [ ] Create RateLimiter class
- [ ] Integrate into base class
- [ ] Configure per provider

---

#### 4.3 **Metrics and Monitoring**
**Issue**: No metrics collection for downloader performance.

**Recommendation**: Add metrics collection
```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class DownloadMetrics:
    """Metrics for a download operation."""
    provider: str
    symbol: str
    interval: str
    duration_ms: float
    rows_downloaded: int
    success: bool
    error: Optional[str] = None

class BaseDataDownloader(ABC):
    def __init__(self):
        self.metrics: List[DownloadMetrics] = []
    
    def get_ohlcv(self, ...) -> pd.DataFrame:
        start = time.time()
        try:
            df = self._fetch_ohlcv(...)
            self._record_metrics(..., duration_ms=(time.time() - start) * 1000, 
                              rows=len(df), success=True)
            return df
        except Exception as e:
            self._record_metrics(..., duration_ms=(time.time() - start) * 1000,
                              rows=0, success=False, error=str(e))
            raise
```

**Action Items**:
- [ ] Define metrics data structure
- [ ] Add metrics collection to base class
- [ ] Create metrics aggregation/export utility
- [ ] Add dashboard/visualization (optional)

---

#### 4.4 **Async Support**
**Issue**: All downloaders are synchronous.

**Recommendation**: Add async interface (optional, for high-throughput scenarios)
```python
from abc import abstractmethod
import asyncio

class BaseDataDownloader(ABC):
    # Keep sync interface
    def get_ohlcv(self, ...) -> pd.DataFrame:
        return asyncio.run(self.get_ohlcv_async(...))
    
    # Add async interface
    @abstractmethod
    async def get_ohlcv_async(self, ...) -> pd.DataFrame:
        """Async version of get_ohlcv."""
        pass
```

**Action Items**:
- [ ] Evaluate if async is needed
- [ ] Add async methods if beneficial
- [ ] Update factory to support async

---

## Summary of Action Items

### Immediate (Priority 1)
- [ ] Refactor 5 downloaders to inherit from BaseDataDownloader
- [ ] Add missing downloaders to factory
- [ ] Centralize configuration access

### Short Term (Priority 2)
- [ ] Refactor factory instantiation logic
- [ ] Standardize error handling
- [ ] Extract chunking logic

### Medium Term (Priority 3)
- [ ] Add retry mechanism
- [ ] Add health check interface
- [ ] Standardize data format
- [ ] Improve type hints and docs

### Long Term (Priority 4)
- [ ] Plugin registration system
- [ ] Rate limiting abstraction
- [ ] Metrics and monitoring
- [ ] Async support (if needed)

---

## Testing Recommendations

1. **Unit Tests**: Test each downloader in isolation
2. **Integration Tests**: Test factory and downloader interactions
3. **Mock Tests**: Mock API responses for consistent testing
4. **Error Tests**: Test error handling paths
5. **Performance Tests**: Test chunking and rate limiting

---

## Migration Strategy

1. **Phase 1**: Fix critical issues (Priority 1)
   - Refactor non-inheriting downloaders
   - Add to factory
   - Centralize config

2. **Phase 2**: Improve architecture (Priority 2)
   - Refactor factory
   - Standardize errors
   - Extract common logic

3. **Phase 3**: Enhance features (Priority 3)
   - Add retry/health checks
   - Standardize formats
   - Improve docs

4. **Phase 4**: Advanced features (Priority 4)
   - Plugin system
   - Metrics
   - Async (if needed)

---

## Notes

- Consider backward compatibility when refactoring
- Update all callers when changing interfaces
- Add deprecation warnings for breaking changes
- Document migration path for users

