"""
MACD Indicator using Unified Indicator Service

This module provides a Backtrader-compatible MACD indicator that uses
the unified indicator service for calculation.
"""

from src.indicators.adapters.backtrader_wrappers import UnifiedMACDIndicator

# Export the unified implementation directly
MacdIndicator = UnifiedMACDIndicator
