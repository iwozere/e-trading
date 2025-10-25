# Short Squeeze Detection Pipeline

## Overview
The Short Squeeze Detection Pipeline is a comprehensive system that identifies publicly traded companies with high probability of upcoming short squeeze events. The system operates using existing data providers (FMP, Finnhub) and follows a hybrid scheduling design with weekly structural scans and daily focused analysis.

## Features
- **Weekly Screener**: Broad structural analysis to identify high short-interest candidates
- **Daily Deep Scan**: Focused real-time analysis on previously identified candidates  
- **Multi-tier Alert System**: High, medium, and low priority alerts with cooldown logic
- **Ad-hoc Candidate Management**: Manual addition of stocks for monitoring
- **Comprehensive Reporting**: Weekly summaries and daily reports in HTML/CSV formats
- **Performance Monitoring**: Runtime metrics, API usage tracking, and data quality validation
- **Configuration Management**: YAML-based configuration with environment variable support
- **Integration Ready**: Seamless integration with existing platform infrastructure

## Quick Start
Example code showing how to use the pipeline:

```python
from src.ml.pipeline.p04_short_squeeze import PipelineConfig
from src.ml.pipeline.p04_short_squeeze.config.config_manager import ConfigManager
from src.ml.pipeline.p04_short_squeeze.config.logging_config import setup_pipeline_logging

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Set up logging
loggers = setup_pipeline_logging(config.run_id)

# Initialize pipeline components
from src.ml.pipeline.p04_short_squeeze import (
    UniverseLoader, WeeklyScreener, DailyDeepScan, 
    ScoringEngine, AlertEngine, CandidateStore
)

# Example usage will be implemented in subsequent tasks
```

## Architecture

The pipeline follows a modular architecture with the following components:

### Core Modules
- **Universe Loader**: Fetches and filters the initial universe of stocks
- **Weekly Screener**: Performs structural analysis for candidate identification
- **Daily Deep Scan**: Conducts real-time analysis on active candidates
- **Scoring Engine**: Computes squeeze probability scores
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

The pipeline adds four new tables to the existing PostgreSQL database:
- `ss_snapshot` - Weekly screener results (append-only)
- `ss_deep_metrics` - Daily deep scan metrics
- `ss_alerts` - Alert history and cooldown tracking
- `ss_ad_hoc_candidates` - Manually added candidates

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements and dependencies
- [Design](docs/Design.md) - Architecture and design decisions
- [Tasks](docs/Tasks.md) - Implementation roadmap and status