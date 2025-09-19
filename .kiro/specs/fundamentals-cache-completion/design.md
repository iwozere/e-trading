# Design Document

## Purpose

This design document outlines the architecture and implementation approach for completing the fundamentals cache system in the E-Trading Data Module. The system will provide a robust, intelligent, and high-performance fundamentals data retrieval system with multi-provider support, advanced caching, and comprehensive error handling.

## Architecture

### High-Level Architecture

The fundamentals cache completion builds upon the existing well-designed infrastructure:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (Trading Strategies, Analytics, Portfolio Management)     │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 DataManager Facade                         │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ get_fundamentals(symbol, providers, force_refresh,      │ │
│  │                 combination_strategy, data_type)       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────┬─────────────────────┬───────────────────────────┘
              │                     │
┌─────────────▼───────────┐ ┌───────▼─────────────────────────┐
│   Enhanced Provider     │ │      Enhanced Cache             │
│   Selection Logic       │ │      Management                 │
│ ┌─────────────────────┐ │ │ ┌─────────────────────────────┐ │
│ │ Symbol Classification│ │ │ │ TTL-based Cache Validation │ │
│ │ Data Type Mapping   │ │ │ │ Automatic Stale Cleanup    │ │
│ │ Provider Failover   │ │ │ │ Quality-based Refresh      │ │
│ │ Rate Limit Handling │ │ │ │ Parallel Cache Operations  │ │
│ └─────────────────────┘ │ │ └─────────────────────────────┘ │
└─────────────────────────┘ └─────────────────────────────────┘
              │                     │
┌─────────────▼───────────────────────────────────────────────┐
│                Multi-Provider Data Fetching                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────┐ │
│  │ FMP         │ │ Yahoo       │ │ Alpha       │ │ Others│ │
│  │ (Priority 1)│ │ (Priority 2)│ │ Vantage     │ │       │ │
│  │             │ │             │ │ (Priority 3)│ │       │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────┘ │
└─────────────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│                Data Combination Engine                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Priority-Based  │  │ Quality-Based   │  │ Consensus   │ │
│  │ Field Selection │  │ Field Selection │  │ Averaging   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────┐
│                Enhanced Cache Layer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ JSON Cache      │  │ Metadata        │  │ Quality     │ │
│  │ (TTL-based)     │  │ Tracking        │  │ Validation  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. Enhanced DataManager Integration

**Current State Analysis:**
- ✅ Basic `get_fundamentals` method exists in DataManager
- ✅ Cache and combiner integration is implemented
- ⚠️ Provider selection logic needs enhancement
- ⚠️ Error handling needs improvement
- ⚠️ Performance optimization needed

**Enhanced Implementation:**
```python
class DataManager:
    def get_fundamentals(self, symbol: str, providers: Optional[List[str]] = None,
                        force_refresh: bool = False, combination_strategy: str = "priority_based",
                        data_type: str = "general") -> Dict[str, Any]:
        """Enhanced fundamentals retrieval with robust error handling."""
        
        # 1. Input validation and normalization
        symbol = self._normalize_symbol(symbol)
        
        # 2. Cache validation with data-type specific TTL
        if not force_refresh:
            cached_data = self._get_cached_fundamentals(symbol, data_type)
            if cached_data:
                return cached_data
        
        # 3. Enhanced provider selection
        selected_providers = self._select_fundamentals_providers(symbol, providers, data_type)
        
        # 4. Parallel data fetching with error handling
        provider_data = self._fetch_fundamentals_parallel(symbol, selected_providers)
        
        # 5. Data combination and validation
        combined_data = self._combine_and_validate_fundamentals(provider_data, combination_strategy, data_type)
        
        # 6. Cache management and cleanup
        self._cache_fundamentals_data(symbol, provider_data, combined_data)
        
        return combined_data
```

#### 2. Enhanced Provider Selection Logic

**Symbol Classification Enhancement:**
```python
class EnhancedProviderSelector:
    def select_fundamentals_providers(self, symbol: str, requested_providers: Optional[List[str]], 
                                    data_type: str) -> List[str]:
        """Select optimal providers for fundamentals data."""
        
        # 1. Symbol classification (US vs International)
        symbol_info = self._classify_symbol_for_fundamentals(symbol)
        
        # 2. Data type specific provider sequence
        if requested_providers:
            return self._validate_requested_providers(requested_providers, symbol_info)
        
        # 3. Configuration-driven provider selection
        provider_sequence = self.combiner.get_provider_sequence(data_type)
        
        # 4. Filter by symbol compatibility and availability
        return self._filter_compatible_providers(provider_sequence, symbol_info)
    
    def _classify_symbol_for_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Enhanced symbol classification for fundamentals."""
        return {
            'symbol': symbol,
            'market': self._detect_market(symbol),  # US, UK, EU, etc.
            'exchange': self._detect_exchange(symbol),
            'symbol_type': self._detect_symbol_type(symbol),  # stock, etf, reit
            'international': self._is_international_symbol(symbol)
        }
```

#### 3. Parallel Data Fetching with Error Handling

**Robust Data Fetching:**
```python
class FundamentalsDataFetcher:
    async def fetch_fundamentals_parallel(self, symbol: str, providers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch fundamentals data from multiple providers in parallel."""
        
        tasks = []
        for provider in providers:
            task = self._fetch_from_provider_with_retry(symbol, provider)
            tasks.append(task)
        
        # Execute with timeout and error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        provider_data = {}
        for provider, result in zip(providers, results):
            if isinstance(result, Exception):
                _logger.error("Provider %s failed for %s: %s", provider, symbol, result)
                continue
            
            if result and self._validate_fundamentals_data(result):
                provider_data[provider] = result
        
        return provider_data
    
    async def _fetch_from_provider_with_retry(self, symbol: str, provider: str) -> Optional[Dict[str, Any]]:
        """Fetch data from a single provider with retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self._wait_for_rate_limit(provider)
                
                # Fetch data
                downloader = self.provider_selector.downloaders.get(provider)
                if not downloader or not hasattr(downloader, 'get_fundamentals'):
                    return None
                
                fundamentals = downloader.get_fundamentals(symbol)
                if fundamentals:
                    return self._normalize_fundamentals_data(fundamentals)
                
            except RateLimitException:
                wait_time = self._calculate_backoff_time(attempt)
                await asyncio.sleep(wait_time)
                continue
            except Exception as e:
                _logger.warning("Attempt %d failed for %s %s: %s", attempt + 1, provider, symbol, e)
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self._calculate_backoff_time(attempt))
        
        return None
```

#### 4. Enhanced Data Combination and Validation

**Advanced Data Combination:**
```python
class EnhancedFundamentalsCombiner:
    def combine_and_validate_fundamentals(self, provider_data: Dict[str, Dict[str, Any]], 
                                        strategy: str, data_type: str) -> Dict[str, Any]:
        """Combine data with enhanced validation."""
        
        # 1. Pre-combination validation
        validated_data = {}
        for provider, data in provider_data.items():
            if self._validate_provider_data(data, provider):
                validated_data[provider] = data
        
        if not validated_data:
            return {}
        
        # 2. Data combination
        combined_data = self.combine_snapshots(validated_data, strategy, data_type)
        
        # 3. Post-combination validation
        if not self._validate_combined_data(combined_data):
            _logger.error("Combined data failed validation")
            return {}
        
        # 4. Cross-provider consistency checks
        consistency_score = self._calculate_consistency_score(validated_data)
        combined_data['_metadata']['consistency_score'] = consistency_score
        
        return combined_data
    
    def _validate_provider_data(self, data: Dict[str, Any], provider: str) -> bool:
        """Validate data from a single provider."""
        
        # Required fields check
        required_fields = self.config.get('data_validation', {}).get('required_fields', [])
        for field in required_fields:
            if field not in data or data[field] is None:
                _logger.warning("Missing required field %s from %s", field, provider)
                return False
        
        # Data quality score check
        quality_score = self._calculate_data_quality(data)
        min_quality = self.config.get('data_validation', {}).get('min_quality_score', 0.8)
        
        if quality_score < min_quality:
            _logger.warning("Data quality too low from %s: %.2f < %.2f", provider, quality_score, min_quality)
            return False
        
        return True
```

#### 5. Advanced Cache Management

**Intelligent Cache Operations:**
```python
class EnhancedFundamentalsCache:
    def get_cached_fundamentals(self, symbol: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Get cached data with intelligent validation."""
        
        # 1. Find latest cache entry
        cache_metadata = self.find_latest_json(symbol, data_type=data_type)
        if not cache_metadata:
            return None
        
        # 2. Validate cache age with data-type specific TTL
        if not self._is_cache_valid_for_data_type(cache_metadata, data_type):
            _logger.debug("Cache expired for %s %s", symbol, data_type)
            return None
        
        # 3. Load and validate cached data
        cached_data = self.read_json(cache_metadata.file_path)
        if not cached_data:
            return None
        
        # 4. Quality-based cache validation
        if self._should_refresh_based_on_quality(cached_data):
            _logger.debug("Cache quality too low for %s, refresh needed", symbol)
            return None
        
        return cached_data
    
    def cache_fundamentals_data(self, symbol: str, provider_data: Dict[str, Dict[str, Any]], 
                              combined_data: Dict[str, Any]) -> None:
        """Cache data with enhanced management."""
        
        timestamp = datetime.now()
        
        # 1. Cache individual provider data
        for provider, data in provider_data.items():
            try:
                self.write_json(symbol, provider, data, timestamp)
                
                # Cleanup stale data for this provider
                removed_files = self.cleanup_stale_data(symbol, provider, timestamp)
                if removed_files:
                    _logger.debug("Cleaned up %d stale files for %s %s", len(removed_files), symbol, provider)
                    
            except Exception as e:
                _logger.error("Failed to cache data for %s %s: %s", symbol, provider, e)
        
        # 2. Cache combined data
        try:
            self.write_json(symbol, 'combined', combined_data, timestamp)
        except Exception as e:
            _logger.error("Failed to cache combined data for %s: %s", symbol, e)
        
        # 3. Trigger background cache maintenance
        self._schedule_cache_maintenance()
```

### Data Flow

#### Enhanced Fundamentals Data Flow

```
1. Request → DataManager.get_fundamentals(symbol, options)
2. Input Validation → Normalize symbol, validate parameters
3. Cache Check → Check for valid cached data with data-type specific TTL
4. Provider Selection → Select optimal providers based on symbol and data type
5. Parallel Fetching → Fetch data from multiple providers with retry logic
6. Data Validation → Validate individual provider data quality
7. Data Combination → Combine data using specified strategy
8. Post-Validation → Validate combined data and consistency
9. Cache Management → Cache new data and cleanup stale entries
10. Response → Return combined fundamentals data with metadata
```

#### Error Handling Flow

```
1. Provider Failure → Log error, try next provider in sequence
2. Rate Limit Hit → Wait with exponential backoff, try alternative providers
3. Data Validation Failure → Log validation errors, exclude from combination
4. Network Timeout → Retry with backoff, fallback to cached data if available
5. All Providers Fail → Return cached data if available, otherwise empty result
6. Cache Corruption → Remove corrupted files, fetch fresh data
7. Configuration Error → Fall back to default configuration, log warnings
```

## Design Decisions

### 1. Asynchronous Data Fetching

**Decision:** Implement parallel data fetching using asyncio

**Rationale:**
- **Performance**: Significantly faster when fetching from multiple providers
- **Efficiency**: Better resource utilization during I/O operations
- **Scalability**: Can handle multiple concurrent requests
- **Timeout Handling**: Better control over request timeouts

**Implementation:**
- Use asyncio.gather() for parallel execution
- Implement proper exception handling for individual providers
- Add configurable timeouts for each provider
- Maintain backward compatibility with synchronous interface

### 2. Enhanced Provider Selection

**Decision:** Implement symbol-aware provider selection with international support

**Rationale:**
- **Data Quality**: Different providers excel with different markets
- **Coverage**: Some providers have better international coverage
- **Reliability**: Provider performance varies by region
- **Cost Optimization**: Use free providers where possible

**Implementation:**
- Detect symbol market and exchange
- Use market-specific provider sequences
- Filter providers by symbol compatibility
- Implement provider capability mapping

### 3. Data-Type Specific TTL

**Decision:** Use different TTL values based on data type (profiles: 14d, ratios: 3d, statements: 90d)

**Rationale:**
- **Data Freshness**: Different data types have different update frequencies
- **API Efficiency**: Reduce unnecessary API calls for stable data
- **Cost Management**: Optimize API usage based on data volatility
- **User Experience**: Ensure critical data is always fresh

**Implementation:**
- Load TTL configuration from fundamentals.json
- Apply data-type specific validation in cache
- Support override TTL for specific use cases
- Implement cache warming for frequently accessed data

### 4. Quality-Based Cache Refresh

**Decision:** Implement intelligent cache refresh based on data quality scores

**Rationale:**
- **Data Reliability**: Ensure high-quality data is always available
- **Proactive Refresh**: Refresh poor-quality data before TTL expires
- **Provider Optimization**: Learn which providers provide better data
- **User Satisfaction**: Reduce likelihood of serving poor-quality data

**Implementation:**
- Calculate quality scores for cached data
- Set quality thresholds for refresh triggers
- Track provider quality over time
- Implement background refresh for low-quality data

### 5. Field-Specific Provider Priorities

**Decision:** Use field-specific provider priorities instead of global priorities

**Rationale:**
- **Data Accuracy**: Different providers excel at different data fields
- **Optimization**: Use best provider for each specific field
- **Flexibility**: Allow fine-grained control over data sources
- **Quality**: Maximize overall data quality through specialization

**Implementation:**
- Load field priorities from configuration
- Apply field-specific selection during combination
- Fall back to general priorities when field-specific not available
- Support nested field paths (e.g., "ttm_metrics.pe_ratio")

### 6. Comprehensive Error Recovery

**Decision:** Implement multi-level error recovery with graceful degradation

**Rationale:**
- **Reliability**: System continues to function despite provider failures
- **User Experience**: Always return some data when possible
- **Monitoring**: Detailed error tracking for system health
- **Maintenance**: Automatic recovery from transient issues

**Implementation:**
- Provider-level retry with exponential backoff
- Automatic failover to backup providers
- Cache fallback when all providers fail
- Detailed error logging and metrics

## Performance Considerations

### 1. Parallel Processing

- **Async Provider Calls**: Fetch from multiple providers simultaneously
- **Connection Pooling**: Reuse HTTP connections across requests
- **Request Batching**: Batch multiple symbol requests when possible
- **Memory Management**: Efficient data structures and cleanup

### 2. Cache Optimization

- **Fast Lookups**: Efficient cache key generation and indexing
- **Lazy Loading**: Load cache data only when needed
- **Background Cleanup**: Asynchronous cache maintenance
- **Compression**: Gzip compression for cache files

### 3. Resource Management

- **Rate Limiting**: Respect provider API limits
- **Memory Limits**: Implement memory usage monitoring
- **Disk Space**: Automatic cache size management
- **CPU Usage**: Optimize data processing algorithms

## Security Considerations

### 1. API Key Management

- **Environment Variables**: Store API keys securely
- **Key Rotation**: Support for regular key rotation
- **Access Control**: Limit API key access to necessary components
- **Audit Logging**: Log API key usage for security monitoring

### 2. Data Validation

- **Input Sanitization**: Validate all input parameters
- **Data Integrity**: Verify data consistency across providers
- **Injection Prevention**: Prevent code injection through data fields
- **Error Information**: Don't expose sensitive information in errors

### 3. Network Security

- **HTTPS Only**: All API communications use encrypted connections
- **Certificate Validation**: Verify SSL certificates
- **Timeout Handling**: Prevent resource exhaustion attacks
- **Rate Limiting**: Prevent abuse of the system

## Monitoring and Observability

### 1. Performance Metrics

- **Response Times**: Track fundamentals retrieval performance
- **Cache Hit Rates**: Monitor cache effectiveness
- **Provider Performance**: Track individual provider response times
- **Error Rates**: Monitor system reliability

### 2. Business Metrics

- **Data Quality Scores**: Track data quality over time
- **Provider Usage**: Monitor which providers are used most
- **Symbol Coverage**: Track which symbols have good data coverage
- **User Satisfaction**: Monitor data freshness and completeness

### 3. Operational Metrics

- **Cache Size**: Monitor cache growth and cleanup
- **API Usage**: Track API call volumes and costs
- **System Resources**: Monitor CPU, memory, and disk usage
- **Error Patterns**: Identify common failure modes

## Integration Patterns

### 1. Backward Compatibility

- **Existing API**: Maintain compatibility with current DataManager interface
- **Provider Methods**: Keep individual provider get_fundamentals methods working
- **Cache Format**: Support reading existing cache files
- **Configuration**: Graceful handling of missing configuration

### 2. Extension Points

- **New Providers**: Easy addition of new data providers
- **Custom Strategies**: Support for custom combination strategies
- **Validation Rules**: Configurable data validation rules
- **Cache Backends**: Support for alternative cache storage

### 3. Testing Integration

- **Mock Providers**: Support for mock data providers in tests
- **Cache Testing**: Tools for testing cache behavior
- **Performance Testing**: Benchmarking tools for performance validation
- **Integration Testing**: End-to-end testing with real providers