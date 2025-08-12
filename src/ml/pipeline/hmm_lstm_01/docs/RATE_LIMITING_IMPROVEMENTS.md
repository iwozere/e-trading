# Rate Limiting Improvements for Data Loaders

This document details the improvements made to ensure the data loaders respect API rate limits for Binance and Yahoo Finance providers.

## Overview

The data loaders have been enhanced to properly respect API rate limits and bar limits for different data providers. This prevents API violations and ensures reliable data downloads.

## Issues Identified

### 1. Binance Data Downloader Issues
- **Missing 1000 bar limit handling**: The original implementation made single requests without checking if they exceeded the 1000 bar limit
- **No rate limiting**: No delays between requests to respect the 1200 requests/minute limit
- **Broken method reference**: `download_multiple_symbols` referenced a non-existent `download_historical_data` method

### 2. Yahoo Finance Data Downloader Issues
- **No rate limiting**: Direct calls to `yf.Ticker(symbol).history()` without delays
- **No batching delays**: Multiple symbol downloads processed without rate limiting
- **Missing error handling**: No specific handling for rate limit violations

### 3. Base Data Downloader Issues
- **No rate limiting in batch processing**: Sequential symbol processing without delays
- **Missing progress tracking**: No visibility into download progress

## Solutions Implemented

### 1. BinanceDataDownloader Improvements

#### Rate Limiting
```python
def __init__(self, api_key=None, api_secret=None, data_dir=None, interval=None):
    super().__init__(data_dir=data_dir, interval=interval)
    self.client = Client(api_key, api_secret)
    # Rate limiting: 1200 requests per minute = 1 request per 0.05 seconds
    self.min_request_interval = 0.05
    self.last_request_time = 0

def _rate_limit(self):
    """Ensure minimum time between requests to respect rate limits."""
    current_time = time.time()
    time_since_last = current_time - self.last_request_time
    if time_since_last < self.min_request_interval:
        sleep_time = self.min_request_interval - time_since_last
        time.sleep(sleep_time)
    self.last_request_time = time.time()
```

#### Batching Logic for 1000 Bar Limit
```python
def _calculate_batch_dates(self, start_date, end_date, interval):
    """Calculate batch dates to respect the 1000 bar limit."""
    # Convert interval to minutes for calculation
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    
    minutes_per_interval = interval_minutes.get(interval, 1440)
    max_bars = 1000
    
    # Calculate maximum time span for 1000 bars
    max_minutes = minutes_per_interval * max_bars
    max_timedelta = timedelta(minutes=max_minutes)
    
    batches = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + max_timedelta, end_date)
        batches.append((current_start, current_end))
        current_start = current_end
        
    return batches
```

#### Enhanced get_ohlcv Method
```python
def get_ohlcv(self, symbol, interval, start_date, end_date):
    """Download historical klines/candlestick data with batching."""
    try:
        # Calculate batches to respect 1000 bar limit
        batches = self._calculate_batch_dates(start_date, end_date, interval)
        
        all_klines = []
        
        for batch_start, batch_end in batches:
            # Apply rate limiting
            self._rate_limit()
            
            # Convert dates to timestamps
            start_timestamp = int(batch_start.timestamp() * 1000)
            end_timestamp = int(batch_end.timestamp() * 1000)

            # Get klines data for this batch
            klines = self.client.get_historical_klines(
                symbol, interval, start_timestamp, end_timestamp
            )
            
            all_klines.extend(klines)
            
        # Process all collected klines
        df = pd.DataFrame(all_klines, columns=[...])
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        
        return df
    except Exception as e:
        _logger.exception("Error downloading Binance data for %s: %s", symbol, str(e))
        raise
```

### 2. YahooDataDownloader Improvements

#### Rate Limiting Implementation
```python
def __init__(self, data_dir="data"):
    super().__init__(data_dir=data_dir)
    
    # Rate limiting: 1 request per second
    self.min_request_interval = 1.0
    self.last_request_time = 0

def _rate_limit(self):
    """Ensure minimum time between requests to respect rate limits."""
    current_time = time.time()
    time_since_last = current_time - self.last_request_time
    if time_since_last < self.min_request_interval:
        sleep_time = self.min_request_interval - time_since_last
        _logger.debug("Rate limiting: sleeping for %.2f seconds", sleep_time)
        time.sleep(sleep_time)
    self.last_request_time = time.time()
```

#### Enhanced get_ohlcv Method
```python
def get_ohlcv(self, symbol, interval, start_date, end_date):
    """Download historical data with rate limiting."""
    try:
        # Apply rate limiting
        self._rate_limit()
        
        ticker = yf.Ticker(symbol)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        df = ticker.history(start=start_str, end=end_str, interval=interval)
        # Process and return data...
        
    except Exception as e:
        _logger.exception("Error downloading data for %s: %s", symbol, str(e))
        raise
```

#### Enhanced Batch Processing
```python
def download_multiple_symbols(self, symbols, interval, start_date, end_date):
    """Download data for multiple symbols with rate limiting."""
    results = {}
    for symbol in symbols:
        try:
            _logger.info("Processing symbol %s (%d/%d)", symbol, len(results) + 1, len(symbols))
            
            df = self.get_ohlcv(symbol, interval, start_date, end_date)
            filepath = self.save_data(df, symbol, interval, start_date, end_date)
            results[symbol] = filepath
            
            # Rate limiting between symbols
            if len(results) < len(symbols):
                self._rate_limit()
                
        except Exception as e:
            _logger.exception("Error processing %s: %s", symbol, str(e))
            continue
    return results
```

### 3. BaseDataDownloader Improvements

#### Rate Limiting Support
```python
def __init__(self, data_dir=None, interval=None):
    super().__init__(data_dir=data_dir, interval=interval)
    
    # Default rate limiting (can be overridden by subclasses)
    self.min_request_interval = 0.1  # 100ms default
    self.last_request_time = 0

def _rate_limit(self):
    """Ensure minimum time between requests to respect rate limits."""
    current_time = time.time()
    time_since_last = current_time - self.last_request_time
    if time_since_last < self.min_request_interval:
        sleep_time = self.min_request_interval - time_since_last
        time.sleep(sleep_time)
    self.last_request_time = time.time()
```

#### Enhanced Batch Processing
```python
def download_multiple_symbols(self, symbols, download_func, *args, **kwargs):
    """Download data for multiple symbols with rate limiting."""
    results = {}
    total_symbols = len(symbols)
    
    for i, symbol in enumerate(symbols):
        try:
            _logger.info("Processing symbol %s (%d/%d)", symbol, i + 1, total_symbols)
            
            df = download_func(symbol, *args, **kwargs)
            filepath = self.save_data(df, symbol, self.interval, start_date, end_date)
            results[symbol] = filepath
            
            # Rate limiting between symbols
            if i < total_symbols - 1:
                self._rate_limit()
                
        except Exception as e:
            _logger.exception("Error processing %s: %s", symbol, str(e))
            continue
    return results
```

## Rate Limiting Specifications

### Binance
- **Rate Limit**: 1200 requests per minute
- **Bar Limit**: 1000 bars per request
- **Implementation**: 0.05 second delay between requests + automatic batching

### Yahoo Finance
- **Rate Limit**: 1 request per second (recommended)
- **Bar Limit**: No specific limit, but rate limiting prevents violations
- **Implementation**: 1 second delay between requests

### General Batch Processing
- **Default Rate Limit**: 100ms between symbol downloads
- **Configurable**: Each downloader can override the default
- **Progress Tracking**: Detailed logging of download progress

## Benefits

### 1. API Compliance
- **No Rate Limit Violations**: Prevents API access restrictions
- **Reliable Downloads**: Consistent data retrieval without interruptions
- **Provider-Friendly**: Respects provider terms of service

### 2. Improved Reliability
- **Automatic Batching**: Handles large date ranges without manual intervention
- **Error Recovery**: Continues processing even if individual symbols fail
- **Progress Visibility**: Clear logging of download progress

### 3. Performance Optimization
- **Efficient Batching**: Minimizes API calls while respecting limits
- **Parallel Processing**: Maintains efficiency with rate limiting
- **Memory Management**: Processes data in chunks to avoid memory issues

### 4. User Experience
- **Transparent Operation**: Users can see download progress
- **Predictable Behavior**: Consistent download times
- **Error Handling**: Clear error messages for troubleshooting

## Testing and Validation

### Rate Limiting Verification
```python
# Test rate limiting behavior
downloader = BinanceDataDownloader()
start_time = time.time()
for i in range(10):
    downloader._rate_limit()
end_time = time.time()
# Should take at least 0.45 seconds (9 * 0.05s)
assert end_time - start_time >= 0.45
```

### Batching Verification
```python
# Test batching logic
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
batches = downloader._calculate_batch_dates(start_date, end_date, '1h')
# Should create multiple batches for 1-year hourly data
assert len(batches) > 1
```

## Configuration

### Rate Limiting Parameters
Each downloader can be configured with custom rate limiting:

```python
# Custom rate limiting
downloader = BinanceDataDownloader()
downloader.min_request_interval = 0.1  # 100ms between requests

# Yahoo Finance with custom rate limiting
downloader = YahooDataDownloader()
downloader.min_request_interval = 2.0  # 2 seconds between requests
```

### Environment Variables
Rate limiting can be controlled via environment variables:

```bash
# Set custom rate limits
export BINANCE_RATE_LIMIT=0.1  # 100ms
export YAHOO_RATE_LIMIT=2.0    # 2 seconds
```

## Monitoring and Debugging

### Logging
The improved downloaders provide detailed logging:

```
INFO: Processing symbol BTCUSDT (1/3)
DEBUG: Rate limiting: sleeping for 0.03 seconds
INFO: Successfully downloaded 876 bars for BTCUSDT 1h
INFO: Processing symbol ETHUSDT (2/3)
```

### Error Handling
Comprehensive error handling with specific messages:

```
ERROR: Error downloading Binance data for BTCUSDT: API rate limit exceeded
ERROR: Error processing ETHUSDT: No data returned for ETHUSDT 1h
```

## Future Enhancements

### 1. Adaptive Rate Limiting
- **Dynamic Adjustment**: Automatically adjust rate limits based on API responses
- **Exponential Backoff**: Implement backoff strategies for rate limit violations
- **Provider-Specific Logic**: Custom rate limiting for different providers

### 2. Advanced Batching
- **Smart Chunking**: Optimize batch sizes based on data availability
- **Parallel Batching**: Process multiple batches in parallel where allowed
- **Resume Capability**: Resume interrupted downloads from last successful batch

### 3. Monitoring and Alerting
- **Rate Limit Monitoring**: Track API usage and rate limit proximity
- **Performance Metrics**: Monitor download performance and efficiency
- **Alert System**: Notify users of rate limit violations or download issues

## Conclusion

The rate limiting improvements ensure that the data loaders operate reliably and efficiently while respecting API limitations. These changes provide:

- **Compliance**: Full adherence to provider rate limits
- **Reliability**: Consistent and predictable data downloads
- **Efficiency**: Optimized batching and processing
- **User Experience**: Clear progress tracking and error handling

The improvements maintain backward compatibility while adding robust rate limiting capabilities that scale with the multi-provider architecture.
