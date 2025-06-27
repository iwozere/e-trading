from typing import Any, Dict

from src.broker.broker_factory import get_broker
from src.trading.live_trading_bot import LiveTradingBot
from src.trading.config_validator import ConfigValidator, validate_config_file


def create_trading_bot(
    config: Dict[str, Any], strategy_class: Any, parameters: Dict[str, Any]
) -> Any:
    """
    Centralized function to create a trading bot using config, strategy class, and parameters.
    Uses broker factory.
    """
    broker = get_broker(config)
    # For now, return None since bot_factory doesn't exist
    # TODO: Implement bot factory or use LiveTradingBot directly
    return None


__all__ = [
    'create_trading_bot',
    'LiveTradingBot',
    'ConfigValidator',
    'validate_config_file'
]
