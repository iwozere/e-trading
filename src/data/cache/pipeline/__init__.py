"""
Data Pipeline Package

This package contains the complete data pipeline for downloading and processing market data.

Pipeline Steps:
1. Download 1-minute data from Alpaca (step01_download_alpaca_1m.py)
2. Calculate higher timeframes (step02_calculate_timeframes.py)

Main Components:
- run_pipeline.py: Complete pipeline runner
- step01_download_alpaca_1m.py: Download 1m data from Alpaca
- step02_calculate_timeframes.py: Calculate 5m, 15m, 1h, 4h, 1d timeframes

Usage:
    # Run complete pipeline
    python src/data/cache/pipeline/run_pipeline.py

    # Run individual steps
    python src/data/cache/pipeline/step01_download_alpaca_1m.py
    python src/data/cache/pipeline/step02_calculate_timeframes.py
"""

__version__ = "1.0.0"
__author__ = "E-Trading Platform"

# Pipeline step information
PIPELINE_STEPS = {
    1: {
        'name': 'Download 1-Minute Data',
        'script': 'step01_download_alpaca_1m.py',
        'description': 'Download 1-minute OHLCV data from Alpaca Markets'
    },
    2: {
        'name': 'Calculate Higher Timeframes',
        'script': 'step02_calculate_timeframes.py',
        'description': 'Calculate 5m, 15m, 1h, 4h, 1d timeframes from 1m data'
    }
}

# Supported timeframes
SUPPORTED_TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']

# Trading hours configuration
TRADING_HOURS = {
    'start_hour': 4,  # 4:00 AM ET
    'end_hour': 20,   # 8:00 PM ET
    'duration_hours': 16
}