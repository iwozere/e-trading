# Quick Start Guide - E-Trading Data Module

## 🚀 What's New in Phase 4

✅ **File-based cache** replacing Redis dependency  
✅ **Hierarchical structure**: `provider/symbol/interval/year/`  
✅ **Comprehensive testing framework**  
✅ **Updated data providers** (Binance, Yahoo, FMP, Alpha Vantage, etc.)  
✅ **Performance optimization** and data validation  

## 📥 How to Fill the File-Based Cache

### Option 1: Use the Populate Script (Recommended)
```bash
# Test the system first
python src/data/cache/populate_cache.py --test-all

# Populate cache with mock data (for testing)
python src/data/cache/populate_cache.py --populate --symbols BTCUSDT,ETHUSDT --intervals 1h,4h,1d --days 30

# Validate cache structure
python src/data/cache/populate_cache.py --validate-cache
```

### Option 2: Use Data Handler Directly
```python
from src.data import get_data_handler, configure_file_cache
from datetime import datetime, timedelta

# Configure cache
cache = configure_file_cache(cache_dir='d:/data-cache')

# Get handler for Binance
handler = get_data_handler('binance', cache_enabled=True)

# Download and cache data
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=365)

df = handler.get_cached_data('BTCUSDT', '1h', start_date, end_date)
if df is not None:
    print(f"Cached {len(df)} rows")
```

### Option 3: Use Data Source Factory
```python
from src.data import get_data_source_factory, configure_file_cache

# Configure cache
cache = configure_file_cache(cache_dir='d:/data-cache')

# Get factory and create Binance source
factory = get_data_source_factory()
binance_source = factory.create_data_source('binance')

# Download historical data (automatically cached)
df = binance_source.get_data_with_cache('BTCUSDT', '1h', start_date, end_date)
```

## 🧪 How to Test Everything

### Run All Tests
```bash
# Comprehensive test suite
python src/data/tests/run_phase4_tests.py

# Specific test types
python src/data/tests/run_phase4_tests.py --unit-only
python src/data/tests/run_phase4_tests.py --integration-only
python src/data/tests/run_phase4_tests.py --performance-only
```

### Test Individual Components
```bash
# Test file-based cache
python -m pytest src/data/tests/unit/test_file_based_cache.py -v

# Test integration
python -m pytest src/data/tests/integration/test_phase4_integration.py -v

# Test performance
python -m pytest src/data/tests/performance/test_performance_benchmarks.py -v
```

### Manual Testing
```python
# Test cache operations manually
from src.data import configure_file_cache
import pandas as pd
from datetime import datetime

cache = configure_file_cache(cache_dir='d:/data-cache', max_size_gb=1.0)

# Create test data
test_df = pd.DataFrame({
    'open': [100, 101, 102],
    'high': [105, 106, 107],
    'low': [95, 96, 97],
    'close': [101, 102, 103],
    'volume': [1000, 1100, 1200]
}, index=pd.date_range('2023-01-01', periods=3, freq='1H'))

# Test put/get operations
success = cache.put(test_df, 'test_provider', 'TESTSYM', '1h', 
                   start_date=datetime(2023, 1, 1))
print(f"Put success: {success}")

retrieved_df = cache.get('test_provider', 'TESTSYM', '1h', 
                        start_date=datetime(2023, 1, 1))
print(f"Retrieved data: {retrieved_df is not None}")
```

## 🔄 Updated Data Providers

The following providers have been updated to use the new architecture:

✅ **Binance** - Live feed and data feed  
✅ **Yahoo Finance** - Data downloader  
✅ **Financial Modeling Prep (FMP)** - Data downloader  
✅ **Alpha Vantage** - Data downloader  
✅ **CoinGecko** - Data downloader  
✅ **Polygon.io** - Data downloader  
✅ **Finnhub** - Data downloader  

### What Was Updated:
- Removed dependency on `src.model.telegram_bot.Fundamentals`
- Updated to use `src.model.schemas.Fundamentals` and `OptionalFundamentals`
- Integrated with new data validation and caching systems
- Added rate limiting and retry mechanisms

## 📁 Cache Directory Structure

Your cache will be organized as:
```
d:/data-cache/
├── binance/
│   ├── BTCUSDT/
│   │   ├── 1h/
│   │   │   ├── 2023/
│   │   │   │   ├── data.parquet
│   │   │   │   └── metadata.json
│   │   │   └── 2024/
│   │   └── 4h/
│   └── ETHUSDT/
├── yahoo/
│   ├── AAPL/
│   └── MSFT/
└── fmp/
    ├── TSLA/
    └── GOOGL/
```

## ⚡ Performance Features

- **Data compression** with Parquet format
- **Lazy loading** for large datasets
- **Parallel processing** for data operations
- **Memory optimization** for DataFrames
- **Performance monitoring** and metrics

## 🚨 Troubleshooting

### Common Issues:

1. **Import errors**: Make sure you're running from the project root
2. **Cache directory**: Ensure `d:/data-cache` exists or change the path
3. **Data validation failures**: Check your data format matches OHLCV requirements
4. **Rate limiting**: The system automatically handles API rate limits

### Getting Help:

- Check the comprehensive documentation: `src/data/docs/PHASE4_DOCUMENTATION.md`
- Review the refactoring plan: `src/data/docs/REFACTOR.md`
- Run tests to identify specific issues: `python src/data/tests/run_phase4_tests.py`

## 🎯 Next Steps

1. **Test the system**: Run `python src/data/cache/populate_cache.py --test-all`
2. **Populate cache**: Use the populate script with your preferred symbols
3. **Integrate**: Use the new data handlers in your trading strategies
4. **Monitor**: Check performance metrics and cache statistics

---

**Need help?** The system includes comprehensive error handling and logging. Check the console output for detailed information about any issues.
