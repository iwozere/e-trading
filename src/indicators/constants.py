"""
Unified indicator constants and naming conventions.

This module provides standardized naming conventions, constants, and mappings
for all indicators across the unified indicator system.
"""

from typing import Dict, List, Set
from src.indicators.models import (
    TECHNICAL_INDICATORS,
    FUNDAMENTAL_INDICATORS,
    LEGACY_INDICATOR_NAMES,
    get_canonical_name
)

# Multi-output indicator mappings
MULTI_OUTPUT_INDICATORS = {
    "macd": ["macd", "signal", "hist"],
    "bbands": ["upper", "middle", "lower"],
    "stoch": ["k", "d"],
    "aroon": ["aroon_up", "aroon_down"],
    "ichimoku": ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"],
    "support_resistance": ["resistance", "support"]
}

# Legacy multi-output mappings for backward compatibility
LEGACY_MULTI_OUTPUT_MAPPING = {
    "MACD": "macd",
    "MACD_SIGNAL": "macd",
    "MACD_HISTOGRAM": "macd",
    "BB_UPPER": "bbands",
    "BB_MIDDLE": "bbands",
    "BB_LOWER": "bbands",
    "STOCH_K": "stoch",
    "STOCH_D": "stoch",
    "AROON_UP": "aroon",
    "AROON_DOWN": "aroon"
}

# Parameter aliases for different naming conventions
PARAMETER_ALIASES = {
    # RSI
    "rsi": {
        "timeperiod": ["period", "length", "window"],
        "price": ["close", "source"]
    },

    # MACD
    "macd": {
        "fastperiod": ["fast", "fast_period", "fast_length"],
        "slowperiod": ["slow", "slow_period", "slow_length"],
        "signalperiod": ["signal", "signal_period", "signal_length"]
    },

    # Bollinger Bands
    "bbands": {
        "timeperiod": ["period", "length", "window"],
        "nbdevup": ["std_up", "std_dev_up", "upper_std"],
        "nbdevdn": ["std_down", "std_dev_down", "lower_std"]
    },

    # Stochastic
    "stoch": {
        "fastk_period": ["k_period", "fastk", "k_length"],
        "slowk_period": ["slowk", "slowk_length"],
        "slowd_period": ["slowd", "slowd_length", "d_period"]
    },

    # Moving Averages
    "sma": {
        "timeperiod": ["period", "length", "window"]
    },
    "ema": {
        "timeperiod": ["period", "length", "window"]
    },

    # ADX
    "adx": {
        "timeperiod": ["period", "length", "window"]
    },

    # ATR
    "atr": {
        "timeperiod": ["period", "length", "window"]
    },

    # Williams %R
    "williams_r": {
        "timeperiod": ["period", "length", "window"]
    },

    # CCI
    "cci": {
        "timeperiod": ["period", "length", "window"]
    },

    # ROC
    "roc": {
        "timeperiod": ["period", "length", "window"]
    },

    # MFI
    "mfi": {
        "timeperiod": ["period", "length", "window"]
    }
}

# Default parameter values
DEFAULT_PARAMETERS = {
    "rsi": {"timeperiod": 14},
    "macd": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "bbands": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
    "stoch": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
    "adx": {"timeperiod": 14},
    "plus_di": {"timeperiod": 14},
    "minus_di": {"timeperiod": 14},
    "sma": {"timeperiod": 20},
    "ema": {"timeperiod": 20},
    "cci": {"timeperiod": 14},
    "roc": {"timeperiod": 10},
    "mfi": {"timeperiod": 14},
    "williams_r": {"timeperiod": 14},
    "atr": {"timeperiod": 14},
    "obv": {},
    "adr": {"timeperiod": 14},
    "aroon": {"timeperiod": 14},
    "ichimoku": {"tenkan": 9, "kijun": 26, "senkou": 52},
    "sar": {"acceleration": 0.02, "maximum": 0.2},
    "super_trend": {"length": 10, "multiplier": 3.0},
    "ad": {},
    "adosc": {"fastperiod": 3, "slowperiod": 10},
    "bop": {},
    "eom": {"timeperiod": 14, "scale": 100000000.0},
    "support_resistance": {"lookback_bars": 2, "max_swings": 50}
}

# Input requirements for each indicator
INDICATOR_INPUTS = {
    # Technical indicators
    "rsi": ["close"],
    "macd": ["close"],
    "bbands": ["close"],
    "stoch": ["high", "low", "close"],
    "adx": ["high", "low", "close"],
    "plus_di": ["high", "low", "close"],
    "minus_di": ["high", "low", "close"],
    "sma": ["close"],
    "ema": ["close"],
    "cci": ["high", "low", "close"],
    "roc": ["close"],
    "mfi": ["high", "low", "close", "volume"],
    "williams_r": ["high", "low", "close"],
    "atr": ["high", "low", "close"],
    "obv": ["close", "volume"],
    "adr": ["high", "low"],
    "aroon": ["high", "low"],
    "ichimoku": ["high", "low", "close"],
    "sar": ["high", "low"],
    "super_trend": ["high", "low", "close"],
    "ad": ["high", "low", "close", "volume"],
    "adosc": ["high", "low", "close", "volume"],
    "bop": ["open", "high", "low", "close"],
    "eom": ["high", "low", "volume"],
    "support_resistance": ["high", "low", "close"],

    # Fundamental indicators (no OHLCV inputs required)
    **{name: [] for name in FUNDAMENTAL_INDICATORS.keys()}
}

# Output specifications for each indicator
INDICATOR_OUTPUTS = {
    # Single-output technical indicators
    "rsi": ["value"],
    "adx": ["value"],
    "plus_di": ["value"],
    "minus_di": ["value"],
    "sma": ["value"],
    "ema": ["value"],
    "cci": ["value"],
    "roc": ["value"],
    "mfi": ["value"],
    "williams_r": ["value"],
    "atr": ["value"],
    "obv": ["value"],
    "adr": ["value"],
    "sar": ["value"],
    "ad": ["value"],
    "adosc": ["value"],
    "bop": ["value"],

    # Multi-output technical indicators
    "macd": ["macd", "signal", "hist"],
    "bbands": ["upper", "middle", "lower"],
    "stoch": ["k", "d"],
    "aroon": ["aroon_up", "aroon_down"],
    "ichimoku": ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"],
    "super_trend": ["value", "trend"],
    "eom": ["value"],
    "support_resistance": ["resistance", "support"],

    # Fundamental indicators (all single-output)
    **{name: ["value"] for name in FUNDAMENTAL_INDICATORS.keys()}
}

# Indicator categories
TECHNICAL_INDICATOR_NAMES = set(TECHNICAL_INDICATORS.keys())
FUNDAMENTAL_INDICATOR_NAMES = set(FUNDAMENTAL_INDICATORS.keys())
ALL_INDICATOR_NAMES = TECHNICAL_INDICATOR_NAMES | FUNDAMENTAL_INDICATOR_NAMES

# Provider support mapping
PROVIDER_SUPPORT = {
    # Technical indicators - most support multiple providers
    **{name: ["ta-lib", "pandas-ta"] for name in TECHNICAL_INDICATOR_NAMES},

    # Some indicators only support specific providers
    "ichimoku": ["pandas-ta"],
    "super_trend": ["pandas-ta"],

    # Fundamental indicators only support fundamentals provider
    **{name: ["fundamentals"] for name in FUNDAMENTAL_INDICATOR_NAMES}
}

# Special cases for provider support
PROVIDER_SUPPORT.update({
    "aroon": ["ta-lib", "pandas-ta"],
    "sar": ["ta-lib", "pandas-ta"],
    "ad": ["ta-lib", "pandas-ta"],
    "adosc": ["ta-lib", "pandas-ta"],
    "bop": ["ta-lib", "pandas-ta"],
    "eom": ["custom"],
    "support_resistance": ["custom"]
})


def is_technical_indicator(indicator_name: str) -> bool:
    """Check if an indicator is a technical indicator."""
    canonical_name = get_canonical_name(indicator_name)
    return canonical_name in TECHNICAL_INDICATOR_NAMES


def is_fundamental_indicator(indicator_name: str) -> bool:
    """Check if an indicator is a fundamental indicator."""
    canonical_name = get_canonical_name(indicator_name)
    return canonical_name in FUNDAMENTAL_INDICATOR_NAMES


def is_multi_output_indicator(indicator_name: str) -> bool:
    """Check if an indicator produces multiple outputs."""
    canonical_name = get_canonical_name(indicator_name)
    return canonical_name in MULTI_OUTPUT_INDICATORS


def get_indicator_outputs(indicator_name: str) -> List[str]:
    """Get the output names for an indicator."""
    canonical_name = get_canonical_name(indicator_name)
    return INDICATOR_OUTPUTS.get(canonical_name, ["value"])


def get_indicator_inputs(indicator_name: str) -> List[str]:
    """Get the required inputs for an indicator."""
    canonical_name = get_canonical_name(indicator_name)
    return INDICATOR_INPUTS.get(canonical_name, [])


def get_default_parameters(indicator_name: str) -> Dict[str, any]:
    """Get default parameters for an indicator."""
    canonical_name = get_canonical_name(indicator_name)
    return DEFAULT_PARAMETERS.get(canonical_name, {}).copy()


def get_parameter_aliases(indicator_name: str) -> Dict[str, List[str]]:
    """Get parameter aliases for an indicator."""
    canonical_name = get_canonical_name(indicator_name)
    return PARAMETER_ALIASES.get(canonical_name, {})


def get_supported_providers(indicator_name: str) -> List[str]:
    """Get supported providers for an indicator."""
    canonical_name = get_canonical_name(indicator_name)
    return PROVIDER_SUPPORT.get(canonical_name, [])


def normalize_parameter_name(indicator_name: str, param_name: str) -> str:
    """Normalize a parameter name to the canonical form."""
    canonical_name = get_canonical_name(indicator_name)
    aliases = get_parameter_aliases(canonical_name)

    # Check if param_name is an alias for any canonical parameter
    for canonical_param, alias_list in aliases.items():
        if param_name in alias_list:
            return canonical_param

    # Return as-is if no alias found
    return param_name


def get_all_indicator_names(include_legacy: bool = False) -> Set[str]:
    """Get all indicator names."""
    names = ALL_INDICATOR_NAMES.copy()

    if include_legacy:
        names.update(LEGACY_INDICATOR_NAMES.keys())

    return names


def validate_indicator_name(indicator_name: str) -> bool:
    """Validate that an indicator name is recognized."""
    canonical_name = get_canonical_name(indicator_name)
    return canonical_name in ALL_INDICATOR_NAMES