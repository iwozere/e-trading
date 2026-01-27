"""
IndicatorFactory - Creates TALib indicators from configuration

This module provides a factory for creating Backtrader TALib indicators
from JSON configuration. It handles indicator creation, parameter validation,
and field mapping.
"""

import backtrader as bt
from typing import Dict, Any, List

from src.notification.logger import setup_logger
from src.indicators.eom_indicator import EOMIndicator
from src.indicators.support_resistance_indicator import SupportResistanceIndicator
from src.indicators.adapters.backtrader_wrappers import UnifiedSuperTrendIndicator

logger = setup_logger(__name__)


class IndicatorFactory:
    """
    Factory for creating TALib indicators from configuration.

    This factory creates bt.talib indicators and maps their outputs to
    strategy-specific field aliases.
    """

    # Mapping of indicator types to their TALib classes and output fields
    INDICATOR_MAP = {
        'RSI': {
            'class': bt.talib.RSI,
            'outputs': ['rsi'],
            'required_params': ['timeperiod'],
            'data_inputs': ['close']  # Only close price needed
        },
        'BBANDS': {
            'class': bt.talib.BBANDS,
            'outputs': ['upperband', 'middleband', 'lowerband'],
            'required_params': ['timeperiod'],
            'data_inputs': ['close']  # Only close price needed
        },
        'ATR': {
            'class': bt.talib.ATR,
            'outputs': ['atr'],
            'required_params': ['timeperiod'],
            'data_inputs': ['high', 'low', 'close']  # Needs HLC
        },
        'MACD': {
            'class': bt.talib.MACD,
            'outputs': ['macd', 'macdsignal', 'macdhist'],
            'required_params': ['fastperiod', 'slowperiod', 'signalperiod'],
            'data_inputs': ['close']  # Only close price needed
        },
        'SMA': {
            'class': bt.talib.SMA,
            'outputs': ['sma'],
            'required_params': ['timeperiod'],
            'data_inputs': ['close']  # Only close price needed
        },
        'EMA': {
            'class': bt.talib.EMA,
            'outputs': ['ema'],
            'required_params': ['timeperiod'],
            'data_inputs': ['close']  # Only close price needed
        },
        'STOCH': {
            'class': bt.talib.STOCH,
            'outputs': ['slowk', 'slowd'],
            'required_params': ['fastk_period', 'slowk_period', 'slowd_period'],
            'data_inputs': ['high', 'low', 'close']  # Needs HLC
        },
        'ICHIMOKU': {
            'class': bt.indicators.Ichimoku,
            'outputs': ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'],
            'required_params': ['tenkan', 'kijun', 'senkou'],
            'data_inputs': ['data']  # Uses full data feed
        },
        'EOM': {
            'class': EOMIndicator,
            'outputs': ['eom'],
            'required_params': ['timeperiod'],
            'data_inputs': ['data']
        },
        'SupportResistance': {
            'class': SupportResistanceIndicator,
            'outputs': ['resistance', 'support'],
            'required_params': ['lookback_bars'],
            'data_inputs': ['data']
        },
        'SUPERTREND': {
            'class': UnifiedSuperTrendIndicator,
            'outputs': ['super_trend', 'direction'],
            'required_params': ['length', 'multiplier'],
            'data_inputs': ['data']
        }
    }

    @staticmethod
    def create_indicators(
        data: bt.feeds.DataBase,
        indicator_configs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create TALib indicators from configuration.

        Args:
            data: Backtrader data feed
            indicator_configs: List of indicator configurations with format:
                [
                    {
                        "type": "RSI",
                        "params": {"timeperiod": 14},
                        "fields_mapping": {"rsi": "entry_rsi"}
                    }
                ]

        Returns:
            Dictionary mapping field aliases to indicator line objects:
                {'entry_rsi': <RSI indicator line>}

        Raises:
            ValueError: If configuration is invalid
        """
        all_indicators = {}
        used_aliases = set()

        for ind_config in indicator_configs:
            # Validate config
            IndicatorFactory.validate_config(ind_config)

            # Create indicator and get mapped outputs
            indicator_outputs = IndicatorFactory._create_single_indicator(
                data, ind_config, all_indicators
            )

            # Check for duplicate aliases
            for alias in indicator_outputs.keys():
                if alias in used_aliases:
                    raise ValueError(
                        f"Duplicate indicator alias '{alias}'. "
                        "Each field mapping must have a unique alias."
                    )
                used_aliases.add(alias)

            # Add to result
            all_indicators.update(indicator_outputs)

        result_aliases = list(all_indicators.keys())
        logger.info(f"Created {len(all_indicators)} indicator outputs from {len(indicator_configs)} indicators. Aliases: {result_aliases}")

        return all_indicators

    @staticmethod
    def _create_single_indicator(
        data: bt.feeds.DataBase,
        ind_config: Dict[str, Any],
        existing_indicators: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create a single indicator and return mapped outputs.

        Args:
            data: Backtrader data feed
            ind_config: Single indicator configuration

        Returns:
            Dictionary mapping aliases to indicator line objects
        """
        ind_type = ind_config['type']
        params = ind_config['params']
        fields_mapping = ind_config['fields_mapping']
        data_field = ind_config.get('data_field')

        # Get indicator metadata
        ind_meta = IndicatorFactory.INDICATOR_MAP[ind_type]
        ind_class = ind_meta['class']
        data_inputs = ind_meta.get('data_inputs', ['close'])  # Default to close if not specified

        # Prepare data inputs for the indicator
        if data_field:
            if existing_indicators and data_field in existing_indicators:
                # Use the specified indicator line as input
                data_args = [existing_indicators[data_field]]
                logger.debug(f"Using derived data field '{data_field}' for {ind_type}")
            elif hasattr(data, data_field):
                # Use the specified field from the data feed (e.g., 'volume', 'open', etc.)
                data_args = [getattr(data, data_field)]
                logger.debug(f"Using data feed field '{data_field}' for {ind_type}")
            else:
                raise ValueError(
                    f"Data field '{data_field}' not found for {ind_type}. "
                    "Ensure the field exists on the data feed (open, high, low, close, volume) "
                    "or is a derived indicator defined earlier in the configuration."
                )
        elif data_inputs == ['data']:
            data_args = [data]
        else:
            data_args = [getattr(data, field) for field in data_inputs]

        # Create the TALib indicator
        try:
            logger.debug(f"Creating {ind_type} with args: {[type(x) for x in data_args]}")
            for i, arg in enumerate(data_args):
                if arg is None:
                    logger.error(f"Argument {i} is None for {ind_type}!")

            indicator = ind_class(*data_args, **params)
            logger.debug(f"Created {ind_type} indicator with params: {params}")
        except Exception as e:
            raise ValueError(
                f"Failed to create {ind_type} indicator with params {params}: {e}"
            )

        # Map outputs to aliases
        result = {}

        for output_field, alias in fields_mapping.items():
            # Validate output field exists
            if output_field not in ind_meta['outputs']:
                raise ValueError(
                    f"Invalid output field '{output_field}' for {ind_type}. "
                    f"Valid fields: {ind_meta['outputs']}"
                )

            # Get the indicator line for this output
            if len(ind_meta['outputs']) == 1:
                # Single output indicator (RSI, ATR, SMA, EMA)
                result[alias] = indicator
            else:
                # Multi-output indicator (BBANDS, MACD, STOCH)
                # Access the specific line by name
                try:
                    indicator_line = getattr(indicator, output_field)
                    result[alias] = indicator_line
                    logger.debug(f"Mapped field '{output_field}' to alias '{alias}'")
                except AttributeError:
                    logger.error(f"Failed to find output field '{output_field}' on {ind_type} object. Available lines: {indicator.lines.getlinealias(0)}")
                    raise

        return result

    @staticmethod
    def validate_config(ind_config: Dict[str, Any]) -> None:
        """
        Validate indicator configuration.

        Args:
            ind_config: Indicator configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required top-level fields
        required_fields = ['type', 'params', 'fields_mapping']
        for field in required_fields:
            if field not in ind_config:
                raise ValueError(
                    f"Indicator config missing required field: '{field}'"
                )

        ind_type = ind_config['type']

        # Check if indicator type is supported
        if ind_type not in IndicatorFactory.INDICATOR_MAP:
            supported = ', '.join(IndicatorFactory.INDICATOR_MAP.keys())
            raise ValueError(
                f"Unsupported indicator type: '{ind_type}'. "
                f"Supported types: {supported}"
            )

        # Validate required parameters
        ind_meta = IndicatorFactory.INDICATOR_MAP[ind_type]
        params = ind_config['params']

        for required_param in ind_meta['required_params']:
            if required_param not in params:
                raise ValueError(
                    f"{ind_type} indicator missing required parameter: '{required_param}'"
                )

        # Validate fields_mapping
        fields_mapping = ind_config['fields_mapping']
        if not fields_mapping:
            raise ValueError(
                f"{ind_type} indicator must have at least one field mapping"
            )

        # Check that mapped output fields are valid
        valid_outputs = ind_meta['outputs']
        for output_field in fields_mapping.keys():
            if output_field not in valid_outputs:
                raise ValueError(
                    f"Invalid output field '{output_field}' for {ind_type}. "
                    f"Valid fields: {valid_outputs}"
                )
