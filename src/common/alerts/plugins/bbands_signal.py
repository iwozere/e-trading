from typing import Any, Dict, List

import pandas as pd

from .base import SignalPlugin


class BollingerBandsPlugin(SignalPlugin):
    """
    Signal plugin for Bollinger Bands.
    Supports complex crossover and touch logic across upper and lower bands.
    """

    @property
    def name(self) -> str:
        return "bbands_signal"

    def schema(self) -> Dict[str, Any]:
        """
        JSON schema for the parameters this plugin expects.
        """
        return {
            "type": "object",
            "required": ["type", "period", "dev_up", "dev_down"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "cross_upper_up",  # crosses upper band from bottom to up
                        "cross_upper_down",  # crosses upper band from up to down
                        "cross_lower_up",  # crosses lower band from bottom to up
                        "cross_lower_down",  # crosses lower band from up to down
                        "touch_upper",  # touches upper band
                        "touch_lower",  # touches lower band
                    ],
                },
                "period": {"type": "integer", "default": 14, "minimum": 1},
                "dev_up": {"type": "number", "default": 2.0},
                "dev_down": {"type": "number", "default": 2.0},
            },
        }

    def get_required_indicators(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Requests the calculation of Bollinger Bands with the given parameters and upper/lower options.
        Wait, AlertEvaluator's _calculate_basic_indicators currently supports only single band selection
        for BOLLINGER (band="upper"|"lower"|"middle").
        We will request both upper and lower bands generically.
        """
        period = int(params.get("period", 14))
        dev_up = float(params.get("dev_up", 2.0))
        dev_down = float(params.get("dev_down", 2.0))

        return [
            {
                "type": "BOLLINGER",
                "output": f"bbands_upper_{period}_{dev_up}",
                "params": {"period": period, "std_dev": dev_up, "band": "upper"},
            },
            {
                "type": "BOLLINGER",
                "output": f"bbands_lower_{period}_{dev_down}",
                "params": {"period": period, "std_dev": dev_down, "band": "lower"},
            },
        ]

    def evaluate(self, params: Dict[str, Any], market_data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> bool:
        """
        Evaluates the boolean signal for the Bollinger Bands conditions.
        Requires at least 2 bars to determine crossovers.
        """
        if len(market_data) < 2:
            return False

        signal_type = params.get("type")
        period = int(params.get("period", 14))
        dev_up = float(params.get("dev_up", 2.0))
        dev_down = float(params.get("dev_down", 2.0))

        upper_key = f"bbands_upper_{period}_{dev_up}"
        lower_key = f"bbands_lower_{period}_{dev_down}"

        if upper_key not in indicators or lower_key not in indicators:
            return False

        upper_band = indicators[upper_key]
        lower_band = indicators[lower_key]

        # Current and previous prices
        curr_close = market_data["close"].iloc[-1]
        prev_close = market_data["close"].iloc[-2]

        # Current and previous band values
        curr_upper = upper_band.iloc[-1]
        prev_upper = upper_band.iloc[-2]

        curr_lower = lower_band.iloc[-1]
        prev_lower = lower_band.iloc[-2]

        # Check conditions
        if signal_type == "cross_upper_up":
            # Price crosses upper band from bottom to up
            return prev_close < prev_upper and curr_close > curr_upper

        elif signal_type == "cross_upper_down":
            # Price crosses upper band from up to down
            return prev_close > prev_upper and curr_close < curr_upper

        elif signal_type == "cross_lower_up":
            # Price crosses lower band from bottom to up
            return prev_close < prev_lower and curr_close > curr_lower

        elif signal_type == "cross_lower_down":
            # Price crosses lower band from up to down
            return prev_close > prev_lower and curr_close < curr_lower

        elif signal_type == "touch_upper":
            # Touches upper band (current High is >= upper band)
            curr_high = market_data["high"].iloc[-1]
            return curr_high >= curr_upper

        elif signal_type == "touch_lower":
            # Touches lower band (current Low is <= lower band)
            curr_low = market_data["low"].iloc[-1]
            return curr_low <= curr_lower

        return False
