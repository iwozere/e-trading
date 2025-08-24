# Indicator System Refactoring Guide

## Overview

This document outlines the plan to unify technical and fundamental indicator collection into a single, reusable system that can be used by all modules in the project. The goal is to eliminate code duplication, improve performance through caching, and provide consistent recommendation logic across the entire application.

## Current State Analysis

### Issues Identified

1. **Duplication of Indicator Logic**: Technical indicators are calculated in multiple places:
   - `src/common/technicals.py` - Main calculation logic
   - `src/frontend/telegram/screener/indicator_calculator.py` - Alert-specific calculations
   - `src/frontend/telegram/screener/enhanced_screener.py` - Screener-specific calculations

2. **Scattered Recommendation Logic**: Recommendation functions are duplicated across:
   - `src/common/technicals.py` - Technical recommendations
   - `src/frontend/telegram/screener/notifications.py` - Fundamental recommendations
   - Multiple other files with similar logic

3. **Inconsistent Data Retrieval**: Each module fetches data independently, leading to:
   - Multiple API calls for the same data
   - Inconsistent data freshness
   - No caching mechanism

4. **Fragmented Data Structures**: Different modules use different data structures for the same information

## Recommended Unified Architecture

### 1. Create a Unified Indicator Service

**File**: `src/common/indicator_service.py`

**Key Features**:
- **Single Data Source**: Centralized data retrieval with caching
- **Unified Calculation Engine**: One place for all technical and fundamental calculations
- **Comprehensive Recommendations**: Unified recommendation engine for all indicators
- **Caching Layer**: Redis/Memory cache for frequently accessed data
- **Batch Processing**: Efficient batch calculations for multiple tickers

### 2. Enhanced Data Models

**File**: `src/models/indicators.py`

**New Unified Models**:
- `IndicatorResult` - Generic indicator result with value, recommendation, and metadata
- `IndicatorSet` - Collection of indicators for a ticker
- `RecommendationEngine` - Centralized recommendation logic
- `IndicatorCache` - Caching layer for indicator results

### 3. Service Architecture

```python
# src/common/indicator_service.py
class IndicatorService:
    def __init__(self):
        self.cache = IndicatorCache()
        self.recommendation_engine = RecommendationEngine()
        self.data_provider = UnifiedDataProvider()
    
    async def get_indicators(self, ticker: str, indicators: List[str]) -> IndicatorSet:
        """Get indicators with caching and unified calculation"""
    
    async def get_batch_indicators(self, tickers: List[str], indicators: List[str]) -> Dict[str, IndicatorSet]:
        """Get indicators for multiple tickers efficiently"""
    
    def calculate_recommendations(self, indicator_set: IndicatorSet) -> Dict[str, Recommendation]:
        """Calculate recommendations for all indicators"""
```

### 4. Unified Recommendation Engine

**File**: `src/common/recommendation_engine.py`

```python
class RecommendationEngine:
    def __init__(self):
        self.technical_rules = TechnicalRecommendationRules()
        self.fundamental_rules = FundamentalRecommendationRules()
    
    def get_recommendation(self, indicator: str, value: float, context: Dict = None) -> Recommendation:
        """Get recommendation for any indicator type"""
    
    def get_composite_recommendation(self, indicator_set: IndicatorSet) -> CompositeRecommendation:
        """Get overall recommendation based on all indicators"""
```

### 5. Caching Strategy

**File**: `src/common/cache_service.py`

```python
class IndicatorCache:
    def __init__(self):
        self.redis_client = Redis()
        self.memory_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes
    
    async def get_cached_indicators(self, ticker: str, indicators: List[str]) -> Optional[IndicatorSet]:
        """Get cached indicators if available and fresh"""
    
    async def cache_indicators(self, ticker: str, indicator_set: IndicatorSet):
        """Cache indicator results with appropriate TTL"""
```

## Implementation Plan

### Phase 1: Core Infrastructure

1. **Create Unified Data Models**
   ```python
   # src/models/indicators.py
   @dataclass
   class IndicatorResult:
       name: str
       value: float
       recommendation: str
       confidence: float
       last_updated: datetime
       source: str
   
   @dataclass
   class IndicatorSet:
       ticker: str
       technical_indicators: Dict[str, IndicatorResult]
       fundamental_indicators: Dict[str, IndicatorResult]
       composite_score: float
       overall_recommendation: str
   ```

2. **Create Recommendation Engine**
   ```python
   # src/common/recommendation_engine.py
   class RecommendationEngine:
       def get_technical_recommendation(self, indicator: str, value: float) -> str
       def get_fundamental_recommendation(self, indicator: str, value: float) -> str
       def get_composite_recommendation(self, indicator_set: IndicatorSet) -> str
   ```

### Phase 2: Service Implementation

1. **Create Indicator Service**
   ```python
   # src/common/indicator_service.py
   class IndicatorService:
       async def get_indicators(self, ticker: str, indicators: List[str]) -> IndicatorSet
       async def get_batch_indicators(self, tickers: List[str], indicators: List[str]) -> Dict[str, IndicatorSet]
       def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]
       def calculate_fundamental_indicators(self, fundamentals: Fundamentals) -> Dict[str, float]
   ```

2. **Implement Caching Layer**
   ```python
   # src/common/cache_service.py
   class IndicatorCache:
       async def get_cached_data(self, key: str) -> Optional[Any]
       async def set_cached_data(self, key: str, data: Any, ttl: int = 300)
       def generate_cache_key(self, ticker: str, indicators: List[str]) -> str
   ```

### Phase 3: Migration

1. **Update Existing Modules**
   - Replace direct calculations in `enhanced_screener.py` with service calls
   - Update `notifications.py` to use unified recommendation engine
   - Migrate `indicator_calculator.py` to use the service

2. **Create Migration Scripts**
   ```python
   # scripts/migrate_to_unified_indicators.py
   def migrate_screener_module():
       """Migrate screener modules to use unified indicator service"""
   
   def migrate_notification_module():
       """Migrate notification modules to use unified recommendation engine"""
   ```

### Phase 4: Optimization

1. **Batch Processing**
   ```python
   # src/common/batch_processor.py
   class BatchIndicatorProcessor:
       async def process_ticker_batch(self, tickers: List[str], indicators: List[str]) -> Dict[str, IndicatorSet]
       def optimize_data_retrieval(self, tickers: List[str]) -> List[str]
   ```

2. **Performance Monitoring**
   ```python
   # src/common/monitoring.py
   class IndicatorServiceMonitor:
       def track_performance(self, operation: str, duration: float)
       def track_cache_hit_rate(self, hit_rate: float)
       def track_api_calls(self, provider: str, count: int)
   ```

## Benefits of This Architecture

### 1. Performance Improvements
- **Reduced API Calls**: Single data source with intelligent caching
- **Batch Processing**: Efficient processing of multiple tickers
- **Caching**: 5-10x faster response times for frequently accessed data

### 2. Consistency
- **Unified Logic**: Single source of truth for all calculations
- **Standardized Recommendations**: Consistent recommendation logic across all modules
- **Data Freshness**: Centralized control over data freshness

### 3. Maintainability
- **Single Point of Update**: Changes to indicator logic only need to be made in one place
- **Clear Separation**: Service layer separates business logic from data access
- **Testability**: Easier to unit test individual components

### 4. Scalability
- **Horizontal Scaling**: Service can be deployed across multiple instances
- **Load Balancing**: Can distribute indicator calculations across multiple workers
- **Resource Optimization**: Efficient resource usage through caching and batching

### 5. Extensibility
- **Easy to Add Indicators**: New indicators can be added to the service without affecting other modules
- **Plugin Architecture**: Recommendation rules can be easily extended
- **Provider Agnostic**: Easy to add new data providers

## Migration Strategy

### Step 1: Create New Infrastructure
1. Create the new service files
2. Implement basic functionality
3. Add comprehensive tests

### Step 2: Gradual Migration
1. Start with one module (e.g., `enhanced_screener.py`)
2. Migrate to use the new service
3. Verify functionality and performance
4. Move to next module

### Step 3: Cleanup
1. Remove duplicate code from old modules
2. Update documentation
3. Performance optimization

### Step 4: Monitoring
1. Add performance monitoring
2. Track usage patterns
3. Optimize based on real-world usage

## File Structure

```
src/
├── models/
│   └── indicators.py              # New unified data models
├── services/
│   ├── indicator_service.py       # Main indicator service
│   ├── recommendation_engine.py   # Unified recommendation logic
│   ├── cache_service.py          # Caching layer
│   ├── batch_processor.py        # Batch processing
│   └── monitoring.py             # Performance monitoring
├── common/
│   ├── technicals.py             # Legacy (to be migrated)
│   └── fundamentals.py           # Legacy (to be migrated)
└── frontend/telegram/screener/
    ├── enhanced_screener.py      # To be updated
    ├── notifications.py          # To be updated
    └── indicator_calculator.py   # To be updated
```

## Technical Specifications

### Indicator Types Supported

**Technical Indicators**:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ADX (Average Directional Index)
- Moving Averages (SMA, EMA)
- Volume indicators (OBV, ADR)

**Fundamental Indicators**:
- P/E Ratio
- P/B Ratio
- P/S Ratio
- PEG Ratio
- ROE (Return on Equity)
- ROA (Return on Assets)
- Debt/Equity Ratio
- Current Ratio
- Quick Ratio
- Operating Margin
- Profit Margin
- Revenue Growth
- Net Income Growth
- Free Cash Flow
- Dividend Yield

### Caching Strategy

**Cache Levels**:
1. **Memory Cache**: Fast access for frequently used indicators (TTL: 5 minutes)
2. **Redis Cache**: Persistent cache for less frequent data (TTL: 30 minutes)
3. **Database Cache**: Long-term storage for historical data

**Cache Keys**:
```
indicator:{ticker}:{indicator_name}:{timeframe}
batch:{ticker_list_hash}:{indicator_list_hash}
```

### Performance Targets

- **Single Indicator**: < 100ms (cached), < 2s (uncached)
- **Batch Processing**: < 5s for 100 tickers
- **Cache Hit Rate**: > 80% for frequently accessed data
- **API Call Reduction**: > 70% reduction in redundant calls

## Testing Strategy

### Unit Tests
- Test individual indicator calculations
- Test recommendation logic
- Test caching mechanisms
- Test error handling

### Integration Tests
- Test service integration
- Test batch processing
- Test cache consistency
- Test performance under load

### Migration Tests
- Test backward compatibility
- Test data consistency
- Test performance improvements
- Test error scenarios

## Monitoring and Metrics

### Key Metrics to Track
- Cache hit rate
- API call frequency
- Response times
- Error rates
- Memory usage
- CPU utilization

### Alerts
- Cache hit rate < 70%
- Response time > 5s
- Error rate > 5%
- Memory usage > 80%

## Risk Mitigation

### Potential Risks
1. **Data Inconsistency**: Mitigated by single data source
2. **Performance Degradation**: Mitigated by caching and optimization
3. **Migration Complexity**: Mitigated by gradual migration approach
4. **Cache Invalidation**: Mitigated by proper TTL and invalidation strategies

### Rollback Plan
1. Keep legacy code during migration
2. Feature flags for gradual rollout
3. Monitoring and alerting for issues
4. Quick rollback capability

## Success Criteria

### Phase 1 Success Metrics
- [ ] All new services implemented and tested
- [ ] Basic functionality working
- [ ] Performance benchmarks met

### Phase 2 Success Metrics
- [ ] One module successfully migrated
- [ ] Performance improvements achieved
- [ ] No regression in functionality

### Phase 3 Success Metrics
- [ ] All modules migrated
- [ ] 70% reduction in code duplication
- [ ] 50% improvement in response times

### Phase 4 Success Metrics
- [ ] 80% cache hit rate achieved
- [ ] 70% reduction in API calls
- [ ] All performance targets met

## Conclusion

This unified architecture will significantly improve the codebase's maintainability, performance, and consistency while providing a solid foundation for future enhancements. The gradual migration approach ensures minimal disruption to existing functionality while delivering immediate benefits through improved performance and reduced code duplication.
