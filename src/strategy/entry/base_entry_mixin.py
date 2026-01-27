"""
Base Entry Mixin Module

This module provides the base class for all entry mixins.
"""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, get_type_hints, List

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class BaseEntryMixin(ABC):
    """Base class for all entry mixins"""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialization of the mixin

        Args:
            params: Dictionary with configuration parameters
        """
        self.strategy = None
        self.params = params or {}
        self.indicators = {}

        # Validate class method implementation
        self._validate_class_methods()

        # Validation of parameters when creating
        self._validate_params()

        # Setting default values
        self._set_defaults()

    def _validate_class_methods(self):
        """Validate that required methods are implemented correctly"""
        # Check get_default_params
        default_params_method = getattr(self.__class__, "get_default_params", None)
        if default_params_method is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement get_default_params method. "
                "This method should return a dictionary of default parameters."
            )

        # Check if method can be called on the class
        try:
            self.__class__.get_default_params()
        except TypeError:
            raise TypeError(
                f"{self.__class__.__name__}.get_default_params must be a class method. "
                "Use @classmethod decorator and 'cls' parameter."
            )

        # Check get_required_params
        required_params_method = getattr(self.__class__, "get_required_params", None)
        if required_params_method is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement get_required_params method. "
                "This method should return a list of required parameter names."
            )

        # Check method signature
        sig = inspect.signature(required_params_method)
        if "self" not in sig.parameters:
            raise TypeError(
                f"{self.__class__.__name__}.get_required_params must be an instance method. "
                "Use 'self' parameter."
            )

        # Check _init_indicators
        init_indicators_method = getattr(self.__class__, "_init_indicators", None)
        if init_indicators_method is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement _init_indicators method. "
                "This method should initialize all required technical indicators."
            )

        # Check method signature
        sig = inspect.signature(init_indicators_method)
        if "self" not in sig.parameters:
            raise TypeError(
                f"{self.__class__.__name__}._init_indicators must be an instance method. "
                "Use 'self' parameter."
            )

        # Check should_enter
        should_enter_method = getattr(self.__class__, "should_enter", None)
        if should_enter_method is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement should_enter method. "
                "This method should return a boolean indicating whether to enter a position."
            )

        # Check method signature
        sig = inspect.signature(should_enter_method)
        if "self" not in sig.parameters:
            raise TypeError(
                f"{self.__class__.__name__}.should_enter must be an instance method. "
                "Use 'self' parameter."
            )

        # Validate method signatures and return types
        self._validate_method_signatures()

    def _validate_method_signatures(self):
        """Validate the signatures and return types of implemented methods"""
        # Validate get_default_params
        if not inspect.ismethod(self.__class__.get_default_params):
            raise TypeError(
                f"{self.__class__.__name__}.get_default_params must be a class method. "
                "Use @classmethod decorator and 'cls' parameter."
            )

        # Validate return type of get_default_params
        return_type = get_type_hints(self.__class__.get_default_params).get("return")
        if return_type != Dict[str, Any]:
            raise TypeError(
                f"{self.__class__.__name__}.get_default_params must return Dict[str, Any]. "
                f"Current return type: {return_type}"
            )

        # Validate get_required_params
        required_params_sig = inspect.signature(self.__class__.get_required_params)
        if "self" not in required_params_sig.parameters:
            raise TypeError(
                f"{self.__class__.__name__}.get_required_params must have 'self' parameter. "
                "Example: def get_required_params(self) -> list:"
            )

        # Validate return type of get_required_params
        return_type = get_type_hints(self.__class__.get_required_params).get("return")
        if return_type != list:
            raise TypeError(
                f"{self.__class__.__name__}.get_required_params must return list. "
                f"Current return type: {return_type}"
            )

        # Validate _init_indicators
        init_indicators_sig = inspect.signature(self.__class__._init_indicators)
        if "self" not in init_indicators_sig.parameters:
            raise TypeError(
                f"{self.__class__.__name__}._init_indicators must have 'self' parameter. "
                "Example: def _init_indicators(self):"
            )

        # Validate return type of _init_indicators
        return_type = get_type_hints(self.__class__._init_indicators).get("return")
        if return_type is not None and return_type != type(None):
            raise TypeError(
                f"{self.__class__.__name__}._init_indicators should not return anything. "
                f"Current return type: {return_type}"
            )

        # Validate should_enter
        should_enter_sig = inspect.signature(self.__class__.should_enter)
        if "self" not in should_enter_sig.parameters:
            raise TypeError(
                f"{self.__class__.__name__}.should_enter must have 'self' parameter. "
                "Example: def should_enter(self) -> bool:"
            )

        # Validate return type of should_enter
        return_type = get_type_hints(self.__class__.should_enter).get("return")
        if return_type != bool:
            raise TypeError(
                f"{self.__class__.__name__}.should_enter must return bool. "
                f"Current return type: {return_type}"
            )

    def _validate_params(self):
        """Validation of mixin parameters"""
        required_params = self.get_required_params()
        for param in required_params:
            if param not in self.params:
                raise ValueError(
                    f"Required parameter '{param}' not provided for {self.__class__.__name__}. "
                    f"Required parameters are: {required_params}"
                )

    def _set_defaults(self):
        """Setting default values for parameters.
        Only sets default if the key or its prefixed versions (e_, x_) are not present.
        """
        defaults = self.__class__.get_default_params()
        for key, value in defaults.items():
            prefixed_e = f"e_{key}"
            prefixed_x = f"x_{key}"

            # If any of the possible names for this parameter are already present, don't set default
            if key not in self.params and prefixed_e not in self.params and prefixed_x not in self.params:
                self.params[key] = value

    @abstractmethod
    def get_required_params(self) -> list:
        """Returns a list of required parameters"""
        return []

    @classmethod
    @abstractmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Returns a dictionary of default parameters"""
        return {}

    @classmethod
    def get_indicator_config(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Returns the default indicator configuration for this mixin.
        Subclasses should override this to define their required indicators.

        Args:
            params: The parameters (e.g. from config or optimized) to use for indicator setup.

        Returns:
            List of indicator configurations.
        """
        return []

    def init_entry(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        """
        Initialization of the mixin with a strategy

        Args:
            strategy: Backtrader strategy instance
            additional_params: Additional parameters for updating
        """
        self.strategy = strategy

        if additional_params:
            self.params.update(additional_params)
            self._validate_params()

        # Initialize indicators
        self._init_indicators()

    @abstractmethod
    def _init_indicators(self):
        """Initialization of technical indicators"""

    def are_indicators_ready(self) -> bool:
        """
        Check if required indicators exist in the strategy registry.
        """
        # New Architecture check: if we have a strategy, it should have the indicators
        if self.strategy and hasattr(self.strategy, 'indicators'):
            # This is a basic check. Subclasses should override if they have specific aliases.
            return True

        # Legacy fallback (to be removed after migration)
        if not hasattr(self, "indicators"):
            return False

        if not self.indicators:
            return False

        return True

    @abstractmethod
    def should_enter(self) -> bool:
        """
        Determines if the mixin should enter a position

        Args:
            strategy: Backtrader strategy instance

        Returns:
            bool: True if the mixin should enter a position
        """

    def get_minimum_lookback(self) -> int:
        """
        Returns the minimum number of bars required for indicators to be ready.
        Subclasses should override this based on their specific indicator periods.

        Returns:
            int: Minimum number of bars (default: 1)
        """
        return 1

    def get_param(self, key: str, default=None):
        """Safe parameter retrieval"""
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
            rsi_value = self.get_indicator('entry_rsi')  # Current bar value
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

    def get_indicator_line(self, alias: str) -> Any:
        """
        Get indicator line object by alias.
        Useful for checking length or accessing history directly.

        Args:
            alias: Field alias from fields_mapping

        Returns:
            Backtrader line object
        """
        if self.strategy is None:
            raise RuntimeError("Mixin not attached to strategy")

        if not hasattr(self.strategy, 'indicators') or alias not in self.strategy.indicators:
            raise KeyError(f"Indicator '{alias}' not found in strategy")

        return self.strategy.indicators[alias]

    def get_indicator_prev(self, alias: str, offset: int = 1) -> Any:
        """
        Get previous indicator value by offset.

        Args:
            alias: Field alias from fields_mapping
            offset: Number of bars back (default: 1)

        Returns:
            Previous indicator value

        Example:
            rsi_prev = self.get_indicator_prev('entry_rsi')  # Previous bar
            rsi_prev2 = self.get_indicator_prev('entry_rsi', 2)  # 2 bars back
        """
        if self.strategy is None:
            raise RuntimeError("Mixin not attached to strategy")

        if not hasattr(self.strategy, 'indicators') or alias not in self.strategy.indicators:
            raise KeyError(f"Indicator '{alias}' not found in strategy")

        indicator = self.strategy.indicators[alias]
        return indicator[-offset]

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Creating an instance from configuration

        Args:
            config: Dictionary with configuration parameters

        Returns:
            RSIBBMixin: Instance of the class
        """
        return cls(params=config)

    def register_indicator(self, name: str, indicator: Any):
        """
        Register an indicator in the indicators dictionary and set it as a strategy attribute

        Args:
            name: Name of the indicator
            indicator: Indicator instance
        """
        _logger.debug("Registering indicator: %s", name)
        if not hasattr(self, "indicators"):
            self.indicators = {}
        self.indicators[name] = indicator

        # Set as strategy attribute if strategy exists
        if hasattr(self, "strategy") and self.strategy is not None:
            # Set the indicator as a strategy attribute
            setattr(self.strategy, name, indicator)
            _logger.debug("Set indicator '%s' as strategy attribute", name)

            # Also set it in the strategy's indicators dictionary if it exists
            if not hasattr(self.strategy, "indicators"):
                self.strategy.indicators = {}
            self.strategy.indicators[name] = indicator
            _logger.debug(
                f"Added indicator '{name}' to strategy's indicators dictionary"
            )
        else:
            _logger.warning(
                f"Strategy not set, indicator '{name}' only stored in indicators dictionary"
            )

    def next(self):
        """Called for each new bar"""
        # Check if we need to reinitialize indicators
        if not self.indicators:
            self._init_indicators()

    def notify_trade(self, trade):
        """Strategy will call this method when BUY order is executed (for long position)"""

    def get_entry_reason(self) -> str:
        """
        Returns the reason for the entry.
        Subclasses should override this to provide specific reasoning.
        """
        return f"{self.__class__.__name__} signal"
