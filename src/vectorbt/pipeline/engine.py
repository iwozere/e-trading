import vectorbt as vbt
import pandas as pd
import numpy as np
import operator
from typing import Dict, Any, List, Union, Optional
from src.shared.indicators.adapters import RSI, BBANDS, EMA, SMA, ATR, ADX
from src.notification.logger import setup_logger

_logger = setup_logger(__name__, use_multiprocessing=True)

class IndicatorRegistry:
    """Registry mapping strings to indicator implementation classes."""
    MAPPING = {
        "RSI": RSI,
        "BBANDS": BBANDS,
        "EMA": EMA,
        "SMA": SMA,
        "ATR": ATR,
        "ADX": ADX
    }

    @classmethod
    def get(cls, indicator_type: str):
        if indicator_type not in cls.MAPPING:
            raise ValueError(f"Unknown indicator type: {indicator_type}")
        return cls.MAPPING[indicator_type]

class IndicatorResult:
    """Wrapper for indicator results to handle data and parameters uniformly."""
    def __init__(self, data: Any, params: Dict[str, Any]):
        self.data = data
        self.params = params

    def get_field(self, field: str) -> Any:
        """Retrieve a field from data or params."""
        # 1. Check params first (thresholds, etc.)
        if field in self.params:
            return self.params[field]

        # 2. Check data (indicator outputs)
        if isinstance(self.data, dict):
            if field in self.data:
                return self.data[field]

        # 3. Handle Series/DataFrame aliases
        # If the data is the primary output (Series or multi-symbol DataFrame)
        # and the field is a common alias or the indicator type itself
        if isinstance(self.data, (pd.Series, pd.DataFrame)):
            aliases = ["values", "rsi", "ema", "sma", "atr", "close", "out"]
            if field.lower() in aliases:
                return self.data

            # If it's a DataFrame, check if field is a column (for multi-output like BBANDS if not in dict)
            if isinstance(self.data, pd.DataFrame) and field in self.data.columns:
                return self.data[field]

        # 4. Fallback to attribute access (vbt objects)
        try:
            return getattr(self.data, field)
        except (AttributeError, TypeError):
            raise KeyError(f"Field '{field}' not found in indicator result or parameters")

class LogicEvaluator:
    """Evaluates nested logic trees for entry and exit signals."""

    OPERATORS = {
        "<": operator.lt,
        ">": operator.gt,
        "<=": operator.le,
        ">=": operator.ge,
        "==": operator.eq,
        "!=": operator.ne
    }

    @classmethod
    def evaluate(cls, node: Union[Dict[str, Any], List[Any]], results: Dict[str, IndicatorResult], close: pd.DataFrame) -> pd.DataFrame:
        """
        Recursively evaluate a logic node.

        Args:
            node: Dict representing a logic branch (OR/AND) or a leaf condition.
            results: dict of calculated indicators { "rsi_main": pd.DataFrame, ... }
            close: price data for 'close' references
        """
        if "operator" in node:
            op = node["operator"].upper()
            conditions = [cls.evaluate(c, results, close) for c in node["conditions"]]

            if op == "AND":
                res = conditions[0]
                for c in conditions[1:]:
                    res = res & c
                return res
            elif op == "OR":
                res = conditions[0]
                for c in conditions[1:]:
                    res = res | c
                return res
            elif op == "NOT":
                return ~conditions[0]
            else:
                raise ValueError(f"Unsupported logic operator: {op}")

        # Leaf condition: {"indicator": "id", "field": "rsi", "op": "<", "target": 30}
        indicator_id = node.get("indicator")
        field = node.get("field", "close")
        op_str = node.get("op")
        target = node.get("target")

        # Get the left value for comparison
        if indicator_id == "close" or field == "close":
            left_val = close
        else:
            indicator_res = results.get(indicator_id)
            if indicator_res is None:
                raise ValueError(f"Indicator result for '{indicator_id}' not found")

            left_val = indicator_res.get_field(field)

        # Resolve target value
        if isinstance(target, str) and "." in target:
            target_id, target_field = target.split(".")
            target_indicator = results.get(target_id)
            if target_indicator is None:
                raise ValueError(f"Target indicator '{target_id}' not found")

            right_val = target_indicator.get_field(target_field)
        else:
            right_val = target

        # Execute comparison
        op_func = cls.OPERATORS.get(op_str)
        if not op_func:
            raise ValueError(f"Unsupported comparison operator: {op_str}")

        return op_func(left_val, right_val)

class StrategyEngine:
    """Coordinates indicator calculation and logic evaluation based on JSON config."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.indicators_config = config.get("indicators", {})
        self.logic_config = config.get("logic", {})

    def run(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Executes the strategy.

        Args:
            data: Prices DataFrame (should contain Open/High/Low/Close if needed)
            params: Parameters from Optuna trial
        """
        # Ensure we have a Close series for base operations
        if 'column' in data.columns.names and 'Close' in data.columns.get_level_values('column'):
            close = data.xs('Close', level='column', axis=1)
            high = data.xs('High', level='column', axis=1)
            low = data.xs('Low', level='column', axis=1)
            open_px = data.xs('Open', level='column', axis=1)
        else:
            # Fallback if only Close is provided
            close = data
            high = data
            low = data
            open_px = data

        results = {}

        # 1. Calculate all indicators
        for ind_id, ind_cfg in self.indicators_config.items():
            ind_type = ind_cfg["type"]
            impl = IndicatorRegistry.get(ind_type)
            target_tf = ind_cfg.get("timeframe") # e.g. "1h"

            # Prepare data for this indicator (MTF Resampling if needed)
            if target_tf:
                _logger.debug(f"Resampling {ind_id} to {target_tf}")
                # Use vbt resampling to aggregate
                # Note: vbt.resample(data, '1h').last() etc.
                # However, for simplicity and TA-Lib compatibility, we'll use pandas resample
                resampled_high = high.resample(target_tf).max()
                resampled_low = low.resample(target_tf).min()
                resampled_close = close.resample(target_tf).last()
                calc_high, calc_low, calc_close = resampled_high, resampled_low, resampled_close
            else:
                calc_high, calc_low, calc_close = high, low, close

            # Extract params for this specific instance
            ind_params = {}
            # Start with hardcoded defaults
            if "params" in ind_cfg:
                ind_params.update(ind_cfg["params"])

            # Overwrite with optimization space params
            for p_name in ind_cfg.get("space", {}).keys():
                param_key = f"{ind_id}_{p_name}"
                if param_key in params:
                    ind_params[p_name] = params[param_key]

            # Compute and wrap
            try:
                import inspect
                sig = inspect.signature(impl.compute)

                # Map OHLC components based on signature
                compute_args = {}
                if 'high' in sig.parameters: compute_args['high'] = calc_high
                if 'low' in sig.parameters: compute_args['low'] = calc_low
                if 'close' in sig.parameters: compute_args['close'] = calc_close
                if 'open_px' in sig.parameters: compute_args['open_px'] = calc_high # fallback or use actual

                # Add other window/optuna params
                for k, v in ind_params.items():
                    if k in sig.parameters:
                        compute_args[k] = v

                data_res = impl.compute(**compute_args)

                # Up-sample back to base timeframe if resampled
                if target_tf:
                    if isinstance(data_res, dict):
                        data_res = {k: v.reindex(close.index).ffill() for k, v in data_res.items()}
                    else:
                        data_res = data_res.reindex(close.index).ffill()

                results[ind_id] = IndicatorResult(data_res, ind_params)
            except Exception as e:
                _logger.error(f"Failed to calculate {ind_id}: {e}")
                raise

        # 2. Evaluate Logic
        signals = {}
        for signal_type in ["long_entry", "long_exit", "short_entry", "short_exit"]:
            logic_node = self.logic_config.get(signal_type)
            if logic_node:
                signals[signal_type] = LogicEvaluator.evaluate(logic_node, results, close)
            else:
                # Default to false if not defined
                signals[signal_type] = pd.DataFrame(False, index=close.index, columns=close.columns)

        return {
            "signals": {
                "entries": signals.get("long_entry"),
                "exits": signals.get("long_exit"),
                "short_entries": signals.get("short_entry"),
                "short_exits": signals.get("short_exits")
            },
            "results": results
        }
