# Strategy Framework Documentation

## Overview

This directory contains comprehensive documentation for the e-trading strategy framework, including implementation guides, design documents, and task tracking.

## Documentation Structure

### Core Documentation
- **[README.md](README.md)** - This file, providing an overview of the documentation structure
- **[Requirements.md](Requirements.md)** - Detailed requirements and specifications
- **[Design.md](Design.md)** - System architecture and design decisions
- **[Tasks.md](Tasks.md)** - Implementation tasks and progress tracking

### Implementation Documentation
All implementation details, bug fixes, and technical analysis have been consolidated into the main documentation files above. The detailed analysis includes:

- **Database Logging Configuration**: Process-level configuration for backtesting vs trading
- **TradeRepository Integration**: Complete analysis of database integration and partial exit support
- **Enhanced Trade Tracking**: Trade size tracking improvements and asset type validation
- **Bug Fixes**: Entry price tracking, logging improvements, and size validation fixes

## Strategy Framework Components

### Base Strategy (`BaseStrategy`)
The foundation class for all trading strategies, providing:
- Trade tracking and management
- Position sizing and validation
- Performance monitoring
- Database integration
- Partial exit support
- Asset type validation (stocks vs crypto)

### Entry Mixins
Modular components for trade entry logic:
- `RSIOrBBEntryMixin` - RSI and Bollinger Bands entry signals
- `HMMLSTMEntryMixin` - Machine learning-based entry signals
- `RSIBBVolumeEntryMixin` - Volume-confirmed entry signals

### Exit Mixins
Modular components for trade exit logic:
- `AdvancedATRExitMixin` - Advanced ATR-based trailing stops
- `ATRExitMixin` - Basic ATR exit strategy
- `TrailingStopExitMixin` - Simple trailing stop
- `TimeBasedExitMixin` - Time-based exits
- `FixedRatioExitMixin` - Fixed ratio exits

### Strategy Implementations
- `CustomStrategy` - Configurable strategy using entry/exit mixins
- `HMMLSTMStrategy` - Machine learning-based strategy

## Key Features

### Trade Tracking
- **Complete Lifecycle**: Track trades from entry to exit
- **Partial Exits**: Support for partial position closures
- **Size Validation**: Proper validation for stocks (whole numbers) vs crypto (fractional)
- **Performance Metrics**: Real-time PnL, win rate, drawdown tracking

### Database Integration
- **Persistent Storage**: All trades stored in database
- **Bot Instance Tracking**: Track multiple bot instances
- **Performance Analytics**: Historical performance analysis
- **Partial Exit Support**: Complete tracking of partial exit sequences

### Risk Management
- **Position Sizing**: Configurable position sizing with validation
- **Asset Type Detection**: Automatic detection of crypto vs stock symbols
- **Size Validation**: Enforce proper size constraints per asset type
- **Performance Monitoring**: Real-time risk metrics

## Quick Start

### Basic Strategy Configuration
```python
from src.strategy.custom_strategy import CustomStrategy

# Configure strategy
config = {
    'enable_database_logging': True,
    'bot_type': 'paper',
    'position_size': 0.1,
    'entry_mixins': ['RSIOrBBEntryMixin'],
    'exit_mixins': ['AdvancedATRExitMixin']
}

# Create strategy
strategy = CustomStrategy(config=config)
```

### Database Integration
```python
# Enable database logging for paper/live trading
config = {
    'enable_database_logging': True,
    'bot_instance_name': 'my_trading_bot',
    'bot_type': 'paper'  # or 'live' for live trading
}

# Disable database logging for backtesting (default)
config = {
    'enable_database_logging': False,
    'bot_type': 'optimization'
}
```

### Partial Exit Support
```python
# Partial exits are automatically tracked
# Original position: 0.1 BTC
# Partial exit 1: 0.05 BTC
# Partial exit 2: 0.05 BTC
# All tracked with proper relationships
```

## Architecture

### Strategy Hierarchy
```
BaseStrategy (bt.Strategy)
├── CustomStrategy
│   ├── Entry Mixins
│   └── Exit Mixins
└── HMMLSTMStrategy
    ├── ML Entry Logic
    └── ML Exit Logic
```

### Database Schema
```
Trade
├── Basic Trade Info (symbol, direction, prices)
├── Partial Exit Tracking (position_id, sequence, parent_trade_id)
├── Performance Metrics (pnl, commission, duration)
└── Metadata (strategy config, bot instance)

BotInstance
├── Bot Configuration
├── Status Tracking
└── Performance Metrics

PerformanceMetrics
├── Aggregated Performance
├── Risk Metrics
└── Time Series Data
```

## Development Guidelines

### Adding New Strategies
1. Inherit from `BaseStrategy`
2. Implement required methods
3. Add configuration options
4. Update documentation
5. Add tests

### Adding New Mixins
1. Inherit from `BaseEntryMixin` or `BaseExitMixin`
2. Implement required abstract methods
3. Add to factory classes
4. Update configuration schema
5. Add tests

### Database Changes
1. Update schema in `database.py`
2. Update repository methods
3. Add migration scripts
4. Update documentation
5. Test with existing data

## Testing

### Unit Tests
- Strategy logic testing
- Mixin functionality testing
- Database operations testing
- Configuration validation testing

### Integration Tests
- End-to-end strategy execution
- Database integration testing
- Partial exit scenario testing
- Performance metrics validation

### Backtesting
- Historical data validation
- Performance comparison
- Risk metric validation
- Edge case testing

## Performance Considerations

### Database Optimization
- Proper indexing on frequently queried fields
- Connection pooling for high-frequency trading
- Batch operations for bulk data
- Regular cleanup of old data

### Memory Management
- Efficient trade record storage
- Proper cleanup of completed trades
- Memory monitoring for long-running bots
- Garbage collection optimization

### Execution Performance
- Minimal overhead in trade execution
- Efficient indicator calculations
- Optimized database queries
- Caching of frequently accessed data

## Monitoring and Debugging

### Logging
- Comprehensive trade logging
- Performance metrics logging
- Error tracking and reporting
- Debug information for troubleshooting

### Metrics
- Real-time performance monitoring
- Risk metric tracking
- System health monitoring
- Alert generation for anomalies

### Debugging Tools
- Trade history analysis
- Performance attribution
- Error investigation tools
- Configuration validation

## Contributing

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include type hints
- Write unit tests for new features

### Documentation Standards
- Update relevant documentation files
- Include examples and use cases
- Document configuration options
- Maintain task tracking

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for database operations
- Performance tests for critical paths
- Edge case testing

## Support

### Common Issues
- Check the troubleshooting section in individual documentation files
- Review error logs for specific error messages
- Validate configuration settings
- Check database connectivity

### Getting Help
- Review existing documentation
- Check implementation examples
- Test with minimal configurations
- Create detailed bug reports with logs

## Version History

### v1.0.0 - Initial Implementation
- Basic strategy framework
- Entry/exit mixin system
- Database integration
- Trade tracking

### v1.1.0 - Enhanced Features
- Partial exit support
- Improved trade size tracking
- Asset type validation
- Performance optimizations

### v1.2.0 - Advanced Features
- Machine learning strategies
- Advanced exit strategies
- Comprehensive analytics
- Risk management tools

---

*This documentation is maintained alongside the codebase. Please update relevant sections when making changes to the strategy framework.*
