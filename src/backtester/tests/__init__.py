"""
Backtester Tests Module
-----------------------

This module provides a comprehensive testing framework for backtesting trading strategies
with JSON configuration support.

Features:
- JSON-based test configuration
- Dynamic strategy and mixin loading
- Automated backtest execution
- Performance validation and assertions
- Detailed test reports

Usage:
    python -m pytest src/backtester/tests/
    python -m pytest src/backtester/tests/test_custom_strategy.py
"""

from src.backtester.tests.backtester_test_framework import (
    BacktesterTestFramework,
    run_backtest_from_config,
)

__all__ = ["BacktesterTestFramework", "run_backtest_from_config"]
