# Data Module

The data module provides comprehensive data management capabilities for the e-trading platform, serving as the foundation for all market data operations.

## Quick Start

```python
from src.data import get_data_manager

# Get the main data manager instance
data_manager = get_data_manager()

# Retrieve historical OHLCV data with intelligent provider selection
df = data_manager.get_ohlcv("BTCUSDT", "5m", start_date, end_date)

# Get fundamentals data with caching
fundamentals = data_manager.get_fundamentals("AAPL")

# Create live data feed
live_feed = data_manager.get_live_feed("BTCUSDT", "5m")
```

## Key Features

- **Unified Interface**: Single entry point (`DataManager`) for all data operations
- **Intelligent Provider Selection**: Automatic selection of best data provider based on symbol type and timeframe
- **Unified Cache System**: Efficient file-based caching with gzip compression
- **Multi-Provider Support**: Integration with 10+ data providers (Binance, Yahoo, FMP, Alpha Vantage, etc.)
- **Cache Pipeline**: Multi-step data processing pipeline for efficient data management
- **Real-time Data**: WebSocket-based live data feeds
- **Database Integration**: Unified database system with repository pattern

## Architecture

The data module follows a layered architecture with the `DataManager` serving as the main facade:

```
Application Layer
       ↓
DataManager (Facade)
       ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ ProviderSelector│ UnifiedCache    │ Live Data Feeds │
│ (Intelligent)   │ (OHLCV + Fund.) │ (Real-time)     │
└─────────────────┴─────────────────┴─────────────────┘
       ↓
Multiple Data Providers (Binance, FMP, Yahoo, etc.)
```

## Module Structure

```
src/data/
├── __init__.py                 # Main module exports
├── data_manager.py            # DataManager facade and ProviderSelector
├── database_service.py        # Unified database service
├── cache/                     # Cache system
│   ├── unified_cache.py       # Main cache implementation
│   ├── fundamentals_cache.py  # JSON-based fundamentals cache
│   └── pipeline/              # Multi-step data pipeline
├── downloader/                # Data provider implementations
│   ├── base_data_downloader.py
│   ├── binance_data_downloader.py
│   ├── yahoo_data_downloader.py
│   └── ...
├── feed/                      # Live data feeds
│   ├── base_live_data_feed.py
│   ├── binance_live_feed.py
│   └── ...
├── db/                        # Database models and repositories
│   ├── telegram_models.py
│   ├── telegram_repository.py
│   └── ...
├── utils/                     # Utilities and tools
│   ├── populate_cache.py      # Cache population script
│   ├── fill_gaps.py          # Gap filling utility
│   └── ...
└── docs/                      # Documentation
    ├── README.md              # Comprehensive documentation
    ├── Requirements.md        # Dependencies and setup
    ├── Design.md             # Architecture and design decisions
    └── Tasks.md              # Development roadmap
```

## Documentation

For detailed information about the data module:

- **[docs/README.md](docs/README.md)** - Comprehensive module documentation with usage examples
- **[docs/Requirements.md](docs/Requirements.md)** - Dependencies, API keys, and setup instructions
- **[docs/Design.md](docs/Design.md)** - Architecture, design decisions, and technical details
- **[docs/Tasks.md](docs/Tasks.md)** - Development roadmap, completed features, and known issues

## Quick Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** (optional but recommended):
   ```bash
   # Add to your .env file or environment
   ALPHA_VANTAGE_API_KEY=your_key_here
   FMP_API_KEY=your_key_here
   BINANCE_API_KEY=your_key_here
   ```

3. **Initialize cache** (optional):
   ```bash
   python src/data/utils/populate_cache.py --symbols BTCUSDT,AAPL --intervals 5m,1h,1d
   ```

## Contributing

When contributing to the data module:

1. Follow the coding conventions in `CODING_CONVENTIONS.md`
2. Add tests for new functionality in the `tests/` directory
3. Update documentation in the `docs/` folder
4. Ensure all providers follow the `BaseDataDownloader` interface
5. Add new providers to the `ProviderSelector` configuration

## Support

For issues, questions, or contributions, please refer to the main project documentation or create an issue in the project repository.