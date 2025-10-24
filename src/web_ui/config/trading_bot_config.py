"""
Trading Bot Configuration Definitions
------------------------------------

This module contains parameter definitions for all available trading strategy mixins.
Used by the web UI to dynamically generate configuration forms.
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ParameterType(Enum):
    """Parameter types for UI form generation."""
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"
    SELECT = "select"


@dataclass
class ParameterDefinition:
    """Definition of a mixin parameter for UI generation."""
    name: str
    type: ParameterType
    label: str
    default: Any
    min_value: float = None
    max_value: float = None
    step: float = None
    options: List[str] = None
    description: str = ""
    required: bool = False


# Entry Mixin Parameter Definitions
ENTRY_MIXIN_PARAMETERS = {
    'RSIBBVolumeEntryMixin': [
        ParameterDefinition(
            name='e_rsi_period',
            type=ParameterType.NUMBER,
            label='RSI Period',
            default=14,
            min_value=2,
            max_value=100,
            step=1,
            description='Period for RSI calculation'
        ),
        ParameterDefinition(
            name='e_rsi_oversold',
            type=ParameterType.NUMBER,
            label='RSI Oversold Level',
            default=30,
            min_value=0,
            max_value=50,
            step=1,
            description='RSI level considered oversold'
        ),
        ParameterDefinition(
            name='e_bb_period',
            type=ParameterType.NUMBER,
            label='Bollinger Bands Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for Bollinger Bands calculation'
        ),
        ParameterDefinition(
            name='e_bb_dev',
            type=ParameterType.NUMBER,
            label='BB Standard Deviations',
            default=2.0,
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            description='Standard deviation multiplier for Bollinger Bands'
        ),
        ParameterDefinition(
            name='e_vol_ma_period',
            type=ParameterType.NUMBER,
            label='Volume MA Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for volume moving average'
        ),
        ParameterDefinition(
            name='e_min_volume_ratio',
            type=ParameterType.NUMBER,
            label='Min Volume Ratio',
            default=1.1,
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            description='Minimum volume ratio above average'
        ),
        ParameterDefinition(
            name='e_use_bb_touch',
            type=ParameterType.BOOLEAN,
            label='Use BB Touch Signal',
            default=True,
            description='Use Bollinger Band touch for entry signal'
        )
    ],

    'RSIBBEntryMixin': [
        ParameterDefinition(
            name='e_rsi_period',
            type=ParameterType.NUMBER,
            label='RSI Period',
            default=14,
            min_value=2,
            max_value=100,
            step=1,
            description='Period for RSI calculation'
        ),
        ParameterDefinition(
            name='e_rsi_oversold',
            type=ParameterType.NUMBER,
            label='RSI Oversold Level',
            default=30,
            min_value=0,
            max_value=50,
            step=1,
            description='RSI level considered oversold'
        ),
        ParameterDefinition(
            name='e_bb_period',
            type=ParameterType.NUMBER,
            label='Bollinger Bands Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for Bollinger Bands calculation'
        ),
        ParameterDefinition(
            name='e_bb_dev',
            type=ParameterType.NUMBER,
            label='BB Standard Deviations',
            default=2.0,
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            description='Standard deviation multiplier for Bollinger Bands'
        )
    ],

    'RSIOrBBEntryMixin': [
        ParameterDefinition(
            name='e_rsi_period',
            type=ParameterType.NUMBER,
            label='RSI Period',
            default=14,
            min_value=2,
            max_value=100,
            step=1,
            description='Period for RSI calculation'
        ),
        ParameterDefinition(
            name='e_rsi_oversold',
            type=ParameterType.NUMBER,
            label='RSI Oversold Level',
            default=30,
            min_value=0,
            max_value=50,
            step=1,
            description='RSI level considered oversold'
        ),
        ParameterDefinition(
            name='e_bb_period',
            type=ParameterType.NUMBER,
            label='Bollinger Bands Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for Bollinger Bands calculation'
        ),
        ParameterDefinition(
            name='e_bb_dev',
            type=ParameterType.NUMBER,
            label='BB Standard Deviations',
            default=2.0,
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            description='Standard deviation multiplier for Bollinger Bands'
        )
    ],

    'RSIIchimokuEntryMixin': [
        ParameterDefinition(
            name='e_rsi_period',
            type=ParameterType.NUMBER,
            label='RSI Period',
            default=14,
            min_value=2,
            max_value=100,
            step=1,
            description='Period for RSI calculation'
        ),
        ParameterDefinition(
            name='e_rsi_oversold',
            type=ParameterType.NUMBER,
            label='RSI Oversold Level',
            default=30,
            min_value=0,
            max_value=50,
            step=1,
            description='RSI level considered oversold'
        ),
        ParameterDefinition(
            name='e_ichimoku_tenkan',
            type=ParameterType.NUMBER,
            label='Ichimoku Tenkan Period',
            default=9,
            min_value=1,
            max_value=50,
            step=1,
            description='Tenkan-sen (conversion line) period'
        ),
        ParameterDefinition(
            name='e_ichimoku_kijun',
            type=ParameterType.NUMBER,
            label='Ichimoku Kijun Period',
            default=26,
            min_value=1,
            max_value=100,
            step=1,
            description='Kijun-sen (base line) period'
        )
    ],

    'BBVolumeSupertrendEntryMixin': [
        ParameterDefinition(
            name='e_bb_period',
            type=ParameterType.NUMBER,
            label='Bollinger Bands Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for Bollinger Bands calculation'
        ),
        ParameterDefinition(
            name='e_bb_dev',
            type=ParameterType.NUMBER,
            label='BB Standard Deviations',
            default=2.0,
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            description='Standard deviation multiplier for Bollinger Bands'
        ),
        ParameterDefinition(
            name='e_vol_ma_period',
            type=ParameterType.NUMBER,
            label='Volume MA Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for volume moving average'
        ),
        ParameterDefinition(
            name='e_supertrend_period',
            type=ParameterType.NUMBER,
            label='Supertrend Period',
            default=10,
            min_value=1,
            max_value=50,
            step=1,
            description='Period for Supertrend calculation'
        ),
        ParameterDefinition(
            name='e_supertrend_multiplier',
            type=ParameterType.NUMBER,
            label='Supertrend Multiplier',
            default=3.0,
            min_value=1.0,
            max_value=10.0,
            step=0.1,
            description='Multiplier for Supertrend calculation'
        )
    ],

    'RSIVolumeSupertrendEntryMixin': [
        ParameterDefinition(
            name='e_rsi_period',
            type=ParameterType.NUMBER,
            label='RSI Period',
            default=14,
            min_value=2,
            max_value=100,
            step=1,
            description='Period for RSI calculation'
        ),
        ParameterDefinition(
            name='e_rsi_oversold',
            type=ParameterType.NUMBER,
            label='RSI Oversold Level',
            default=30,
            min_value=0,
            max_value=50,
            step=1,
            description='RSI level considered oversold'
        ),
        ParameterDefinition(
            name='e_vol_ma_period',
            type=ParameterType.NUMBER,
            label='Volume MA Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for volume moving average'
        ),
        ParameterDefinition(
            name='e_supertrend_period',
            type=ParameterType.NUMBER,
            label='Supertrend Period',
            default=10,
            min_value=1,
            max_value=50,
            step=1,
            description='Period for Supertrend calculation'
        ),
        ParameterDefinition(
            name='e_supertrend_multiplier',
            type=ParameterType.NUMBER,
            label='Supertrend Multiplier',
            default=3.0,
            min_value=1.0,
            max_value=10.0,
            step=0.1,
            description='Multiplier for Supertrend calculation'
        )
    ]
}

# Exit Mixin Parameter Definitions
EXIT_MIXIN_PARAMETERS = {
    'ATRExitMixin': [
        ParameterDefinition(
            name='x_atr_period',
            type=ParameterType.NUMBER,
            label='ATR Period',
            default=14,
            min_value=1,
            max_value=100,
            step=1,
            description='Period for ATR calculation'
        ),
        ParameterDefinition(
            name='x_sl_multiplier',
            type=ParameterType.NUMBER,
            label='Stop Loss Multiplier',
            default=2.0,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            description='ATR multiplier for stop loss distance'
        )
    ],

    'AdvancedATRExitMixin': [
        ParameterDefinition(
            name='x_atr_period',
            type=ParameterType.NUMBER,
            label='ATR Period',
            default=14,
            min_value=1,
            max_value=100,
            step=1,
            description='Period for ATR calculation'
        ),
        ParameterDefinition(
            name='x_k_init',
            type=ParameterType.NUMBER,
            label='Initial K Factor',
            default=2.5,
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            description='Initial ATR multiplier for stop loss'
        ),
        ParameterDefinition(
            name='x_k_run',
            type=ParameterType.NUMBER,
            label='Running K Factor',
            default=2.0,
            min_value=0.5,
            max_value=10.0,
            step=0.1,
            description='Running ATR multiplier for stop loss'
        )
    ],

    'SimpleATRExitMixin': [
        ParameterDefinition(
            name='x_atr_period',
            type=ParameterType.NUMBER,
            label='ATR Period',
            default=14,
            min_value=1,
            max_value=100,
            step=1,
            description='Period for ATR calculation'
        ),
        ParameterDefinition(
            name='x_sl_multiplier',
            type=ParameterType.NUMBER,
            label='Stop Loss Multiplier',
            default=1.5,
            min_value=0.1,
            max_value=10.0,
            step=0.1,
            description='ATR multiplier for stop loss distance'
        )
    ],

    'FixedRatioExitMixin': [
        ParameterDefinition(
            name='x_take_profit_pct',
            type=ParameterType.NUMBER,
            label='Take Profit %',
            default=5.0,
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            description='Take profit percentage'
        ),
        ParameterDefinition(
            name='x_stop_loss_pct',
            type=ParameterType.NUMBER,
            label='Stop Loss %',
            default=2.0,
            min_value=0.1,
            max_value=50.0,
            step=0.1,
            description='Stop loss percentage'
        )
    ],

    'MACrossoverExitMixin': [
        ParameterDefinition(
            name='x_ma_fast',
            type=ParameterType.NUMBER,
            label='Fast MA Period',
            default=10,
            min_value=1,
            max_value=100,
            step=1,
            description='Fast moving average period'
        ),
        ParameterDefinition(
            name='x_ma_slow',
            type=ParameterType.NUMBER,
            label='Slow MA Period',
            default=20,
            min_value=1,
            max_value=200,
            step=1,
            description='Slow moving average period'
        )
    ],

    'RSIBBExitMixin': [
        ParameterDefinition(
            name='x_rsi_period',
            type=ParameterType.NUMBER,
            label='RSI Period',
            default=14,
            min_value=2,
            max_value=100,
            step=1,
            description='Period for RSI calculation'
        ),
        ParameterDefinition(
            name='x_rsi_overbought',
            type=ParameterType.NUMBER,
            label='RSI Overbought Level',
            default=70,
            min_value=50,
            max_value=100,
            step=1,
            description='RSI level considered overbought'
        ),
        ParameterDefinition(
            name='x_bb_period',
            type=ParameterType.NUMBER,
            label='Bollinger Bands Period',
            default=20,
            min_value=5,
            max_value=100,
            step=1,
            description='Period for Bollinger Bands calculation'
        ),
        ParameterDefinition(
            name='x_bb_dev',
            type=ParameterType.NUMBER,
            label='BB Standard Deviations',
            default=2.0,
            min_value=0.5,
            max_value=5.0,
            step=0.1,
            description='Standard deviation multiplier for Bollinger Bands'
        )
    ],

    'TrailingStopExitMixin': [
        ParameterDefinition(
            name='x_trail_percent',
            type=ParameterType.NUMBER,
            label='Trailing Stop %',
            default=3.0,
            min_value=0.1,
            max_value=20.0,
            step=0.1,
            description='Trailing stop percentage'
        ),
        ParameterDefinition(
            name='x_activation_percent',
            type=ParameterType.NUMBER,
            label='Activation %',
            default=2.0,
            min_value=0.1,
            max_value=20.0,
            step=0.1,
            description='Profit percentage to activate trailing stop'
        )
    ],

    'TimeBasedExitMixin': [
        ParameterDefinition(
            name='x_max_hold_bars',
            type=ParameterType.NUMBER,
            label='Max Hold Bars',
            default=100,
            min_value=1,
            max_value=1000,
            step=1,
            description='Maximum number of bars to hold position'
        )
    ]
}

# Configuration options
BROKER_TYPES = [
    {'value': 'binance', 'label': 'Binance'},
    {'value': 'mock', 'label': 'Mock (Testing)'},
    {'value': 'ibkr', 'label': 'Interactive Brokers'}
]

TRADING_MODES = [
    {'value': 'paper', 'label': 'Paper Trading'},
    {'value': 'live', 'label': 'Live Trading'}
]

DATA_SOURCES = [
    {'value': 'binance', 'label': 'Binance'},
    {'value': 'file', 'label': 'CSV File'},
    {'value': 'yahoo', 'label': 'Yahoo Finance'}
]

INTERVALS = [
    {'value': '1m', 'label': '1 Minute'},
    {'value': '5m', 'label': '5 Minutes'},
    {'value': '15m', 'label': '15 Minutes'},
    {'value': '30m', 'label': '30 Minutes'},
    {'value': '1h', 'label': '1 Hour'},
    {'value': '4h', 'label': '4 Hours'},
    {'value': '1d', 'label': '1 Day'}
]

SYMBOLS = [
    {'value': 'BTCUSDT', 'label': 'Bitcoin (BTC/USDT)'},
    {'value': 'ETHUSDT', 'label': 'Ethereum (ETH/USDT)'},
    {'value': 'LTCUSDT', 'label': 'Litecoin (LTC/USDT)'},
    {'value': 'ADAUSDT', 'label': 'Cardano (ADA/USDT)'},
    {'value': 'DOTUSDT', 'label': 'Polkadot (DOT/USDT)'},
    {'value': 'LINKUSDT', 'label': 'Chainlink (LINK/USDT)'},
    {'value': 'BNBUSDT', 'label': 'Binance Coin (BNB/USDT)'},
    {'value': 'SOLUSDT', 'label': 'Solana (SOL/USDT)'}
]


def get_entry_mixin_parameters(mixin_name: str) -> List[ParameterDefinition]:
    """Get parameter definitions for an entry mixin."""
    return ENTRY_MIXIN_PARAMETERS.get(mixin_name, [])


def get_exit_mixin_parameters(mixin_name: str) -> List[ParameterDefinition]:
    """Get parameter definitions for an exit mixin."""
    return EXIT_MIXIN_PARAMETERS.get(mixin_name, [])


def get_available_entry_mixins() -> List[Dict[str, str]]:
    """Get list of available entry mixins."""
    return [
        {'value': name, 'label': name.replace('EntryMixin', '').replace('Entry', '')}
        for name in ENTRY_MIXIN_PARAMETERS.keys()
    ]


def get_available_exit_mixins() -> List[Dict[str, str]]:
    """Get list of available exit mixins."""
    return [
        {'value': name, 'label': name.replace('ExitMixin', '').replace('Exit', '')}
        for name in EXIT_MIXIN_PARAMETERS.keys()
    ]


def parameter_definition_to_dict(param: ParameterDefinition) -> Dict[str, Any]:
    """Convert ParameterDefinition to dictionary for JSON serialization."""
    return {
        'name': param.name,
        'type': param.type.value,
        'label': param.label,
        'default': param.default,
        'min_value': param.min_value,
        'max_value': param.max_value,
        'step': param.step,
        'options': param.options,
        'description': param.description,
        'required': param.required
    }