"""
RSI Indicator using Unified Indicator Service

This module provides a Backtrader-compatible RSI indicator that uses
the unified indicator service for calculation.
"""

from src.indicators.adapters.backtrader_wrappers import UnifiedRSIIndicator

# Export the unified implementation directly
RsiIndicator = UnifiedRSIIndicator
