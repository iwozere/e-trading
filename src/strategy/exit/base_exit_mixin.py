from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BaseExitMixin(ABC):
    """Base class for all exit mixins"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.strategy = None
        self.params = params or {}
        self.indicators = {}

        self._validate_params()
        self._set_defaults()

    def _validate_params(self):
        required_params = self.get_required_params()
        for param in required_params:
            if param not in self.params:
                raise ValueError(
                    f"Required parameter '{param}' not provided for {self.__class__.__name__}"
                )

    def _set_defaults(self):
        defaults = self.get_default_params()
        for key, value in defaults.items():
            if key not in self.params:
                self.params[key] = value

    @abstractmethod
    def get_required_params(self) -> list:
        return []

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        return {}

    def init_exit(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        self.strategy = strategy

        if additional_params:
            self.params.update(additional_params)
            self._validate_params()

        # Initialize indicators
        self._init_indicators()

    @abstractmethod
    def _init_indicators(self):
        pass

    def are_indicators_ready(self) -> bool:
        """
        Check if indicators are ready to be used.
        Default implementation checks if indicators dictionary exists and has entries.
        Subclasses can override for more specific checks.

        Returns:
            bool: True if indicators are ready, False otherwise
        """
        if not hasattr(self, "indicators"):
            return False

        if not self.indicators:
            return False

        # Check if we have enough data points (basic check)
        if hasattr(self, "strategy") and self.strategy and hasattr(self.strategy, "data"):
            try:
                if len(self.strategy.data) < 2:  # Minimum data requirement
                    return False
            except Exception:
                return False

        return True

    @abstractmethod
    def should_exit(self) -> bool:
        pass

    @abstractmethod
    def get_exit_reason(self) -> str:
        """Get the reason for exit (called after should_exit returns True)."""
        pass

    def on_entry(self, entry_price: float, entry_time, position_size: float, direction: str):
        """Called when a position is entered. Default implementation does nothing."""
        pass

    def get_param(self, key: str, default=None):
        return self.params.get(key, default)

    def get_indicator(self, alias: str) -> Any:
        """
        Get indicator value by alias (for new TALib-based architecture).

        Args:
            alias: Field alias from fields_mapping

        Returns:
            Current indicator value (for current bar)

        Raises:
            RuntimeError: If mixin not attached to strategy
            KeyError: If indicator not found

        Example:
            atr_value = self.get_indicator('exit_atr')  # Current bar value
        """
        if self.strategy is None:
            raise RuntimeError("Mixin not attached to strategy")

        if not hasattr(self.strategy, 'indicators') or alias not in self.strategy.indicators:
            raise KeyError(
                f"Indicator '{alias}' not found in strategy. "
                f"Available indicators: {list(getattr(self.strategy, 'indicators', {}).keys())}"
            )

        indicator = self.strategy.indicators[alias]
        return indicator[0]  # Current bar value

    def get_indicator_prev(self, alias: str, offset: int = 1) -> Any:
        """
        Get previous indicator value by offset.

        Args:
            alias: Field alias from fields_mapping
            offset: Number of bars back (default: 1)

        Returns:
            Previous indicator value

        Example:
            atr_prev = self.get_indicator_prev('exit_atr')  # Previous bar
            atr_prev2 = self.get_indicator_prev('exit_atr', 2)  # 2 bars back
        """
        if self.strategy is None:
            raise RuntimeError("Mixin not attached to strategy")

        if not hasattr(self.strategy, 'indicators') or alias not in self.strategy.indicators:
            raise KeyError(f"Indicator '{alias}' not found in strategy")

        indicator = self.strategy.indicators[alias]
        return indicator[-offset]

    def register_indicator(self, name: str, indicator: Any):
        """
        Register an indicator in the indicators dictionary and set it as a strategy attribute

        Args:
            name: Name of the indicator
            indicator: The indicator instance
        """
        _logger.debug("Registering indicator: %s", name)
        # Store in indicators dictionary
        self.indicators[name] = indicator

        # Also set as strategy attribute if strategy exists
        if hasattr(self, "strategy") and self.strategy is not None:
            setattr(self.strategy, name, indicator)
            _logger.debug("Indicator %s set as strategy attribute", name)
        else:
            _logger.warning(
                f"Cannot set {name} as strategy attribute - strategy not available"
            )

    def are_indicators_ready(self) -> bool:
        """Check if indicators are ready to be used"""
        if not self.indicators:
            return False

        try:
            # Try to access the first value of each indicator
            for indicator in self.indicators.values():
                if hasattr(indicator, "__getitem__"):
                    _ = indicator[0]
                elif hasattr(indicator, "lines"):
                    for line in indicator.lines:
                        _ = line[0]
            return True
        except (IndexError, TypeError):
            return False

    def next(self):
        """Called for each new bar"""
        # Check if we need to reinitialize indicators
        if not self.indicators:
            self._init_indicators()

    def notify_trade(self, trade):
        """Strategy will call this method when SELL order is executed (for long position)"""
