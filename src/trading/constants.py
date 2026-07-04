"""
Trading Constants
-----------------
Centralized constants for paths and configuration.
"""

from pathlib import Path

# Project root (assumes this file is at src/trading/constants.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directoriess
TRADING_CONFIG_DIR = PROJECT_ROOT / "config" / "trading"
TRADING_LOGS_DIR = PROJECT_ROOT / "logs" / "json"
TRADING_STATE_DIR = PROJECT_ROOT / "data" / "bots"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
for directory in [TRADING_CONFIG_DIR, TRADING_LOGS_DIR, TRADING_STATE_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
