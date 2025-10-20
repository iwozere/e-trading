"""
Bollinger Bands Indicator using Unified Indicator Service

This module provides a Backtrader-compatible Bollinger Bands indicator that uses
the unified indicator service for calculation.
"""

from src.indicators.adapters.backtrader_wrappers import UnifiedBollingerBandsIndicator

# Export the unified implementation directly
BollingerBandIndicator = UnifiedBollingerBandsIndicator
