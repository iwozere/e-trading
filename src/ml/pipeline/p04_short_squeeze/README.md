# Short Squeeze Detection Pipeline

## Overview
The Short Squeeze Detection Pipeline is a comprehensive system that identifies publicly traded companies with high probability of upcoming short squeeze events. The system uses a **hybrid approach** combining volume-based detection with official FINRA short interest data, following a multi-tier scheduling design with weekly universe loading, bi-weekly FINRA data collection, and daily analysis.

## Features
- **Weekly Universe Loading**: FMP-based stock universe selection with market cap filtering
- **Bi-weekly FINRA Data Collection**: Official short interest data from FINRA
- **Daily Volume Detection**: Volume pattern analysis and momentum indicators for early squeeze detection
- **Daily Hybrid Deep Scan**: Combines volume analysis with FINRA data for comprehensive scoring
- **Multi-tier Alert System**: High, medium, and low priority alerts with cooldown logic
- **Ad-hoc Candidate Management**: Manual addition of stocks for monitoring
- **Comprehensive Reporting**: Weekly summaries and daily reports in HTML/CSV formats
- **Performance Monitoring**: Runtime metrics, API usage tracking, and data quality validation
- **Configuration Management**: YAML-based configuration with environment variable support
- **Integration Ready**: Seamless integration with existing platform infrastructure

## Pipeline Scripts & Scheduling

### Available Scripts

#### 1. Weekly Universe Selection ✅
- **Script**: `scripts/run_weekly_screener.py`
- **Timing**: Weekly (Mondays 8:00 AM Europe/Zurich)
- **Purpose**: Load stock universe from FMP based on market cap filtering (NO individual ticker analysis)
- **Status**: **FIXED** - Now only loads universe, no ticker-by-ticker API calls

#### 2. Bi-weekly FINRA Data Collection ✅
- **Script**: `scripts/run_finra_collector.py` 
- **Timing**: Bi-weekly (1st and 15th of month, 9:00 AM Europe/Zurich)
- **Purpose**: Download official FINRA short interest data with incremental loading
- **Status**: **IMPLEMENTED** - Smart date detection, avoids duplicate downloads

#### 3. Daily Volume Detection ❌
- **Script**: `scripts/run_volume_detector.py`
- **Timing**: Daily (9:30 AM Europe/Zurich)
- **Purpose**: Analyze volume patterns and identify squeeze candidates
- **Status**: **NEEDS TO BE CREATED**

#### 4. Daily Hybrid Deep Scan ✅
- **Script**: `scripts/run_daily_deep_scan.py`
- **Timing**: Daily (10:00 AM Europe/Zurich)
- **Purpose**: Combine volume analysis with latest FINRA data for comprehensive scoring
- **Status**: **ENHANCED** - Now integrates latest FINRA data from database for each candidate

#### 5. Ad-hoc Management ✅
- **Script**: `scripts/manage_adhoc_candidates.py`
- **Timing**: As needed
- **Purpose**: Manual candidate management
- **Status**: Available

### Recommended Pipeline Sequence

Weekly: run_weekly_screener.py
Bi-weekly: run_finra_collector.py
Daily 9:30 AM: run_volume_detector.py
Daily 10:00 AM: run_daily_deep_scan.py

```
WEEKLY (Monday 8:00 AM):
└── run_weekly_screener.py → Load universe from FMP (market cap filtering)

BI-WEEKLY (1st & 15th, 9:00 AM):  
└── run_finra_collector.py → Download FINRA short interest data

DAILY (9:30 AM):
└── run_volume_detector.py → Analyze volume patterns, identify candidates

DAILY (10:00 AM):
└── run_daily_deep_scan.py → Hybrid analysis (volume + FINRA + sentiment)

AS NEEDED:
└── manage_adhoc_candidates.py → Manual candidate management
```

## Quick Start

### Test API Connections
```bash
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener --test-connection
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_finra_collector --test-connection
```

### FINRA Data Collection
```bash
# Download missing FINRA data (auto-detects missing dates)
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_finra_collector --verbose

# Force download for specific date
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_finra_collector --date 2024-01-15 --force-download

# Dry run to see what would be downloaded
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_finra_collector --dry-run --verbose
```

### Small Test Run (Recommended)
```bash
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_weekly_screener --dry-run --max-universe 20 --verbose
```

### Daily Deep Scan with FINRA Integration
```bash
# Deep scan with specific tickers (uses latest FINRA data from DB)
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan --tickers AAPL,TSLA,GME --dry-run --progress

# Deep scan all candidates from database
python -m src.ml.pipeline.p04_short_squeeze.scripts.run_daily_deep_scan --dry-run --progress --verbose
```

### Manage Ad-hoc Candidates
```bash
# Add a candidate
python -m src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates add AAPL "High volume spike observed"

# List active candidates
python -m src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates list --details

# Show statistics
python -m src.ml.pipeline.p04_short_squeeze.scripts.manage_adhoc_candidates stats
```

## Architecture

The pipeline follows a modular architecture with the following components:

### Core Modules
- **Universe Loader**: Fetches and filters the initial universe of stocks from FMP
- **FINRA Data Collector**: Downloads official short interest data with smart incremental loading
- **Volume Squeeze Detector**: Analyzes volume patterns and momentum indicators for early detection
- **Daily Deep Scan**: Enhanced hybrid analysis using latest FINRA data from database per candidate
- **Scoring Engine**: Computes squeeze probability scores combining transient + FINRA + structural metrics
- **Alert Engine**: Manages alert generation and cooldown logic

### Data Management
- **Candidate Store**: Persistent storage for candidates and results
- **Ad-hoc Manager**: Handles manually added candidates

### Support Systems
- **Configuration Manager**: YAML-based configuration with validation
- **Logging System**: Structured logging with performance metrics
- **Reporting Engine**: Generate summaries and export data

## Integration

This module integrates with:
- `src.data` - For FMP and Finnhub data providers
- `src.notification` - For Telegram and email alerts
- `src.common` - For shared database and utility functions
- Existing PostgreSQL database system
- Existing configuration management patterns

## Configuration

The pipeline uses YAML configuration files located at:
- `config/pipeline/p04_short_squeeze.yaml` - Main configuration
- Environment variables for sensitive data (API keys, credentials)

Key configuration sections:
- **Scheduling**: Future scheduler integration parameters
- **Screener**: Universe filtering and scoring weights
- **Deep Scan**: Real-time metrics and batch processing
- **Alerting**: Thresholds, cooldowns, and notification channels
- **Performance**: API rate limits and error handling

## Database Schema

The pipeline adds five new tables to the existing PostgreSQL database:
- `ss_snapshot` - Volume detector results and universe snapshots (append-only)
- `ss_finra_short_interest` - **Historical FINRA data** with unique constraint on (ticker, settlement_date)
- `ss_deep_metrics` - Daily hybrid deep scan metrics combining volume and latest FINRA data
- `ss_alerts` - Alert history and cooldown tracking
- `ss_ad_hoc_candidates` - Manually added candidates

### FINRA Data Management
- **Historical Storage**: Multiple rows per ticker for different settlement dates (1 year history)
- **Smart Date Detection**: Handles FINRA's business day logic (15th and end-of-month with weekend adjustments)
- **Latest Data Retrieval**: Queries automatically get most recent data per ticker
- **Incremental Loading**: Only downloads missing settlement dates that actually exist on FINRA servers
- **Data Freshness**: Tracks data age and quality metrics

## Current Status & Known Issues

### ✅ Implemented
- **FINRA Data Collection**: Smart incremental loading with duplicate prevention
- **Database Integration**: Historical FINRA data storage with latest data retrieval
- **Enhanced Deep Scan**: Integrates latest FINRA data for each candidate during analysis
- **Hybrid Scoring**: Combines transient metrics (50%) + FINRA data (30%) + structural metrics (20%)
- Volume-based detection algorithms
- Ad-hoc candidate management system
- Comprehensive logging and error handling

### ⚠️ Known Issues
- **Volume Detector Script**: Still needs to be created for daily volume analysis
- **Data Source Reliability**: FINRA data availability depends on official publication schedule

### 🔧 Next Steps
1. **Volume Detector Implementation**: Complete the daily volume detection script
2. **Alert System Integration**: Connect scoring results to notification system
3. **Performance Optimization**: Batch processing and caching for large universes
4. **Data Quality Monitoring**: Enhanced validation and freshness tracking

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements and dependencies
- [Design](docs/Design.md) - Architecture and design decisions
- [Tasks](docs/Tasks.md) - Implementation roadmap and status