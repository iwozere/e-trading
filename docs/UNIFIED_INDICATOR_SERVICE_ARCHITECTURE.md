# Unified Indicator Service Architecture

## Overview

The Unified Indicator Service represents a complete consolidation of all indicator-related functionality in the e-trading platform. This architecture replaces the previous fragmented system with a single, comprehensive service that provides consistent APIs, enhanced performance, and simplified maintenance.

## Architecture Components

### Core Service Layer

#### UnifiedIndicatorService (`src/indicators/service.py`)
- **Purpose**: Main orchestrator for all indicator operations
- **Responsibilities**:
  - Request routing to appropriate adapters
  - Result aggregation and formatting
  - Error handling and fallback logic
  - Performance monitoring and metrics collection
  - Batch processing coordination

#### ConfigurationManager (`src/indicators/config_manager.py`)
- **Purpose**: Centralized parameter and preset management
- **Responsibilities**:
  - Load and validate configuration from unified JSON format
  - Provide runtime parameter overrides
  - Manage indicator presets (default, conservative, aggressive, day_trading)
  - Handle legacy indicator name mappings

#### RecommendationEngine (`src/indicators/recommendation_engine.py`)
- **Purpose**: Generate trading recommendations based on indicator values
- **Responsibilities**:
  - Calculate individual indicator recommendations
  - Compute composite recommendations from multiple indicators
  - Provide confidence scores and reasoning
  - Support contextual recommendations using related indicators

### Adapter Layer

#### BaseAdapter (`src/indicators/adapters/base_adapter.py`)
- **Purpose**: Abstract interface for all calculation backends
- **Implementations**:
  - `TALibAdapter`: TA-Lib calculations
  - `PandasTAAdapter`: pandas-ta calculations  
  - `BacktraderAdapter`: Backtrader native calculations
  - `FundamentalsAdapter`: Fundamental analysis calculations

#### BacktraderWrappers (`src/indicators/adapters/backtrader_wrappers.py`)
- **Purpose**: Provide Backtrader-compatible indicator classes
- **Features**:
  - Maintains same line-based interface as original Backtrader indicators
  - Uses unified service for calculations
  - Supports all backend types through unified interface

### Data Layer

#### Registry (`src/indicators/registry.py`)
- **Purpose**: Centralized catalog of all available indicators
- **Contents**:
  - Indicator metadata (inputs, outputs, parameters)
  - Supported backends per indicator
  - Default parameter values
  - Validation schemas

#### Models (`src/indicators/models.py`)
- **Purpose**: Data structures for requests, responses, and configurations
- **Key Models**:
  - `IndicatorRequest`: Request specification
  - `IndicatorResult`: Individual indicator result
  - `IndicatorSet`: Collection of results for a ticker
  - `CompositeRecommendation`: Aggregated recommendations

## Configuration Architecture

### Unified Configuration Format

The system uses a single configuration file `config/indicators.json` with the following structure:

```json
{
  "version": "2.0",
  "default_parameters": { ... },
  "presets": { ... },
  "legacy_mappings": { ... },
  "plotter_config": { ... }
}
```

### Configuration Hierarchy

1. **Registry Defaults**: Base parameters from indicator metadata
2. **Global Defaults**: System-wide parameter overrides
3. **Preset Parameters**: Preset-specific configurations
4. **Runtime Overrides**: Dynamic parameter changes

### Legacy Compatibility

The system maintains backward compatibility through:
- **Legacy Mappings**: Old indicator names mapped to canonical names
- **Parameter Translation**: Automatic conversion of old parameter formats
- **Interface Preservation**: Existing method signatures maintained where possible

## Integration Patterns

### Backtrader Integration

```python
# Direct usage in strategies
from src.indicators.adapters.backtrader_wrappers import UnifiedRSI

class MyStrategy(bt.Strategy):
    def __init__(self):
        self.rsi = UnifiedRSI(self.data, timeperiod=14)
```

### Async Service Usage

```python
# Service-level usage
from src.indicators.service import UnifiedIndicatorService

service = UnifiedIndicatorService()
request = IndicatorRequest(
    ticker="BTCUSDT",
    indicators=["rsi", "macd", "bbands"]
)
result = await service.calculate(request)
```

### Batch Processing

```python
# Multiple tickers
requests = [
    IndicatorRequest(ticker="BTCUSDT", indicators=["rsi"]),
    IndicatorRequest(ticker="ETHUSDT", indicators=["macd"])
]
results = await service.calculate_batch(requests)
```

## Performance Optimizations

### Caching Strategy
- **Request-level caching**: Avoid duplicate calculations within request
- **Result caching**: Cache computed values for reuse
- **Configuration caching**: Cache parsed configurations

### Batch Processing
- **Concurrent execution**: Parallel processing of multiple tickers
- **Resource pooling**: Efficient resource utilization
- **Memory management**: Optimized memory usage for large datasets

### Error Handling
- **Circuit breakers**: Prevent cascade failures
- **Graceful degradation**: Partial results when some calculations fail
- **Automatic fallbacks**: Switch backends on calculation errors

## Migration Benefits

### Before (Fragmented System)
- Multiple indicator services with different APIs
- Scattered configuration files
- Inconsistent error handling
- Duplicated functionality
- Complex maintenance

### After (Unified System)
- Single service with consistent API
- Consolidated configuration
- Standardized error handling
- Eliminated duplication
- Simplified maintenance

## Testing Architecture

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component functionality
3. **Performance Tests**: Benchmarking and optimization
4. **Compatibility Tests**: Legacy interface validation
5. **End-to-End Tests**: Complete workflow validation

### Test Coverage
- **Core Functionality**: 95%+ coverage
- **Adapter Layer**: 90%+ coverage
- **Configuration**: 100% coverage
- **Error Handling**: 90%+ coverage

## Future Extensibility

### Adding New Indicators
1. Register in `registry.py`
2. Implement in appropriate adapter
3. Add configuration defaults
4. Create tests
5. Update documentation

### Adding New Backends
1. Implement `BaseAdapter` interface
2. Register in service configuration
3. Add backend-specific tests
4. Update documentation

### Adding New Features
1. Extend service interface
2. Update data models
3. Implement in core service
4. Add configuration support
5. Create comprehensive tests

## Monitoring and Observability

### Metrics Collection
- Request latency and throughput
- Error rates by component
- Cache hit/miss ratios
- Resource utilization

### Logging Strategy
- Structured logging with correlation IDs
- Performance metrics logging
- Error context preservation
- Debug information for troubleshooting

### Health Checks
- Service availability checks
- Backend connectivity validation
- Configuration integrity verification
- Performance threshold monitoring

## Security Considerations

### Input Validation
- Parameter validation against schemas
- Request size limits
- Rate limiting capabilities
- Input sanitization

### Error Information
- Sanitized error messages
- No sensitive data in logs
- Controlled error propagation
- Audit trail maintenance

This unified architecture provides a solid foundation for all indicator-related operations while maintaining flexibility for future enhancements and ensuring smooth operation across all use cases.