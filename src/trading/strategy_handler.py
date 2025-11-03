#!/usr/bin/env python3
"""
Strategy Handler
----------------

Dynamic strategy factory that instantiates different strategy types based on configuration.
This is the SOLE component responsible for strategy type resolution and instantiation.

Features:
- Strategy registry with plugin architecture
- Dynamic strategy class loading
- Parameter validation for different strategy types
- Fallback to CustomStrategy for unknown types
- Support for CustomStrategy with entry/exit mixins
"""

import importlib
import inspect
from typing import Dict, Any, Type, Optional, List, Tuple
from pathlib import Path

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class StrategyHandler:
    """
    Dynamic strategy factory for loading and validating trading strategies.

    Responsibilities:
    - Maintain strategy registry (CustomStrategy, ML strategies, etc.)
    - Dynamically import and instantiate strategy classes
    - Validate strategy-specific parameters
    - Provide fallback to CustomStrategy for unknown types
    """

    def __init__(self):
        """Initialize the strategy handler with default strategies."""
        self.strategy_registry: Dict[str, Dict[str, Any]] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default built-in strategies."""
        # Register CustomStrategy
        self.register_strategy(
            strategy_type="CustomStrategy",
            module_path="src.strategy.custom_strategy",
            class_name="CustomStrategy",
            description="Custom strategy with configurable entry/exit mixins",
            requires_mixins=True
        )

        # Register AdvancedStrategyFramework (if exists)
        try:
            self.register_strategy(
                strategy_type="AdvancedStrategyFramework",
                module_path="src.strategy.future.composite_strategy_manager",
                class_name="AdvancedStrategyFramework",
                description="Advanced composite strategy framework",
                requires_mixins=False
            )
        except Exception as e:
            _logger.debug("AdvancedStrategyFramework not available: %s", e)

        _logger.info("Registered %d default strategies", len(self.strategy_registry))

    def register_strategy(
        self,
        strategy_type: str,
        module_path: str,
        class_name: str,
        description: str = "",
        requires_mixins: bool = False,
        validator_func: Optional[callable] = None
    ):
        """
        Register a new strategy type.

        Args:
            strategy_type: Strategy type identifier (e.g., "CustomStrategy")
            module_path: Python module path (e.g., "src.strategy.custom_strategy")
            class_name: Class name within the module
            description: Human-readable description
            requires_mixins: Whether this strategy requires entry/exit mixins
            validator_func: Optional custom validation function
        """
        self.strategy_registry[strategy_type] = {
            "module_path": module_path,
            "class_name": class_name,
            "description": description,
            "requires_mixins": requires_mixins,
            "validator_func": validator_func,
            "class_ref": None  # Lazy loaded
        }
        _logger.info("Registered strategy type: %s (%s)", strategy_type, description)

    def get_strategy_class(self, strategy_type: str) -> Type:
        """
        Get strategy class for the given type.

        Args:
            strategy_type: Strategy type identifier

        Returns:
            Strategy class type

        Raises:
            ValueError: If strategy type is invalid or cannot be loaded
        """
        # Normalize strategy type
        strategy_type = strategy_type.strip()

        # Check if registered
        if strategy_type not in self.strategy_registry:
            _logger.warning(
                "Unknown strategy type '%s', falling back to CustomStrategy. "
                "Available types: %s",
                strategy_type,
                list(self.strategy_registry.keys())
            )
            strategy_type = "CustomStrategy"

        strategy_info = self.strategy_registry[strategy_type]

        # Lazy load the class if not already loaded
        if strategy_info["class_ref"] is None:
            try:
                module = importlib.import_module(strategy_info["module_path"])
                strategy_class = getattr(module, strategy_info["class_name"])
                strategy_info["class_ref"] = strategy_class
                _logger.debug("Loaded strategy class: %s.%s",
                             strategy_info["module_path"],
                             strategy_info["class_name"])
            except Exception as e:
                _logger.exception("Failed to load strategy class %s:", strategy_type)
                # Fallback to CustomStrategy if not already trying it
                if strategy_type != "CustomStrategy":
                    _logger.warning("Falling back to CustomStrategy")
                    return self.get_strategy_class("CustomStrategy")
                raise ValueError(f"Failed to load strategy class {strategy_type}: {e}")

        return strategy_info["class_ref"]

    def validate_strategy_config(
        self,
        strategy_type: str,
        strategy_config: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate strategy configuration parameters.

        Args:
            strategy_type: Strategy type identifier
            strategy_config: Strategy configuration dictionary

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Normalize strategy type
        strategy_type = strategy_type.strip()

        # Check if type is registered
        if strategy_type not in self.strategy_registry:
            warnings.append(
                f"Unknown strategy type '{strategy_type}', will fallback to CustomStrategy"
            )
            strategy_type = "CustomStrategy"

        strategy_info = self.strategy_registry[strategy_type]

        # Check for required parameters
        parameters = strategy_config.get("parameters", {})
        if not parameters:
            warnings.append("No strategy parameters provided")

        # Validate mixin-based strategies (CustomStrategy)
        if strategy_info["requires_mixins"]:
            mixin_errors, mixin_warnings = self._validate_mixin_config(parameters)
            errors.extend(mixin_errors)
            warnings.extend(mixin_warnings)

        # Run custom validator if provided
        if strategy_info["validator_func"]:
            try:
                custom_valid, custom_errors, custom_warnings = strategy_info["validator_func"](
                    strategy_config
                )
                if not custom_valid:
                    errors.extend(custom_errors)
                warnings.extend(custom_warnings)
            except Exception as e:
                errors.append(f"Custom validation failed: {e}")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    def _validate_mixin_config(
        self,
        parameters: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate mixin configuration for CustomStrategy.

        Args:
            parameters: Strategy parameters containing entry_logic and exit_logic

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        # Check entry logic
        entry_logic = parameters.get("entry_logic")
        if not entry_logic:
            errors.append("Missing required 'entry_logic' configuration")
        else:
            if not isinstance(entry_logic, dict):
                errors.append("'entry_logic' must be a dictionary")
            elif "name" not in entry_logic:
                errors.append("'entry_logic' missing required 'name' field")
            else:
                # Validate entry mixin name
                entry_name = entry_logic.get("name")
                if not self._is_valid_mixin_name(entry_name):
                    warnings.append(
                        f"Entry mixin '{entry_name}' may not be a valid mixin class"
                    )

                # Check for params
                if "params" not in entry_logic:
                    warnings.append("'entry_logic' has no parameters (params)")

        # Check exit logic
        exit_logic = parameters.get("exit_logic")
        if not exit_logic:
            errors.append("Missing required 'exit_logic' configuration")
        else:
            if not isinstance(exit_logic, dict):
                errors.append("'exit_logic' must be a dictionary")
            elif "name" not in exit_logic:
                errors.append("'exit_logic' missing required 'name' field")
            else:
                # Validate exit mixin name
                exit_name = exit_logic.get("name")
                if not self._is_valid_mixin_name(exit_name):
                    warnings.append(
                        f"Exit mixin '{exit_name}' may not be a valid mixin class"
                    )

                # Check for params
                if "params" not in exit_logic:
                    warnings.append("'exit_logic' has no parameters (params)")

        # Check position size
        position_size = parameters.get("position_size")
        if position_size is not None:
            if not isinstance(position_size, (int, float)):
                errors.append("'position_size' must be a number")
            elif position_size <= 0 or position_size > 1:
                warnings.append(
                    f"'position_size' {position_size} should be between 0 and 1"
                )

        return errors, warnings

    def _is_valid_mixin_name(self, mixin_name: str) -> bool:
        """
        Check if mixin name follows expected naming conventions.

        Args:
            mixin_name: Mixin class name

        Returns:
            True if name appears valid
        """
        if not mixin_name:
            return False

        # Check naming conventions
        # Entry mixins typically end with "EntryMixin"
        # Exit mixins typically end with "ExitMixin"
        valid_suffixes = ["Mixin", "EntryMixin", "ExitMixin", "Strategy"]
        return any(mixin_name.endswith(suffix) for suffix in valid_suffixes)

    def get_registered_strategies(self) -> Dict[str, str]:
        """
        Get list of all registered strategy types with descriptions.

        Returns:
            Dictionary mapping strategy type to description
        """
        return {
            stype: info["description"]
            for stype, info in self.strategy_registry.items()
        }

    def discover_strategies(self, search_paths: List[Path]) -> int:
        """
        Discover and register strategies from specified paths.

        Args:
            search_paths: List of paths to search for strategy modules

        Returns:
            Number of newly discovered strategies
        """
        discovered = 0
        _logger.info("Discovering strategies in paths: %s", search_paths)

        for search_path in search_paths:
            if not search_path.exists():
                _logger.warning("Search path does not exist: %s", search_path)
                continue

            # Find all Python files
            for py_file in search_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    # Convert file path to module path
                    relative_path = py_file.relative_to(Path.cwd())
                    module_path = str(relative_path.with_suffix("")).replace("/", ".")

                    # Try to import and inspect
                    module = importlib.import_module(module_path)

                    # Look for strategy classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it looks like a strategy class
                        if self._is_strategy_class(obj):
                            strategy_type = name
                            if strategy_type not in self.strategy_registry:
                                self.register_strategy(
                                    strategy_type=strategy_type,
                                    module_path=module_path,
                                    class_name=name,
                                    description=f"Auto-discovered strategy: {name}"
                                )
                                discovered += 1

                except Exception as e:
                    _logger.debug("Could not inspect %s: %s", py_file, e)

        _logger.info("Discovered %d new strategies", discovered)
        return discovered

    def _is_strategy_class(self, cls: Type) -> bool:
        """
        Check if a class appears to be a trading strategy.

        Args:
            cls: Class to inspect

        Returns:
            True if class appears to be a strategy
        """
        # Basic heuristic: has certain methods or inherits from known base
        strategy_methods = ["next", "buy", "sell", "notify_order"]
        has_methods = any(hasattr(cls, method) for method in strategy_methods)

        # Check class name
        name_indicators = ["Strategy", "Trading", "Bot"]
        has_name_indicator = any(indicator in cls.__name__ for indicator in name_indicators)

        return has_methods or has_name_indicator


# Create singleton instance for easy import
strategy_handler = StrategyHandler()
