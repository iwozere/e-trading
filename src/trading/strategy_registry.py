"""
Strategy Registry
-----------------
Centralized registry for all available trading strategies in the system.
Bridges StrategyHandler and LiveTradingBot / StrategyInstance.
"""

from typing import Dict, Type, List, Optional, Any
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class StrategyRegistry:
    """
    Registry for trading strategy classes.
    Follows Singleton pattern.
    """
    _instance = None
    _strategies: Dict[str, Type] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StrategyRegistry, cls).__new__(cls)
        return cls._instance

    def register(self, strategy_type: str, strategy_class: Type):
        """Register a new strategy class."""
        self._strategies[strategy_type] = strategy_class
        _logger.debug("Registered strategy type: %s", strategy_type)

    def get(self, strategy_type: str) -> Optional[Type]:
        """Get strategy class by type name."""
        return self._strategies.get(strategy_type)

    def list_strategies(self) -> List[str]:
        """List all registered strategy types."""
        return list(self._strategies.keys())

    def validate_type(self, strategy_type: str) -> bool:
        """Check if a strategy type is registered."""
        return strategy_type in self._strategies

# Global registry instance
strategy_registry = StrategyRegistry()
