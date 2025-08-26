"""
Bot Manager Module
------------------

This module provides a unified interface for managing trading bot instances in memory. It supports starting, stopping, and tracking bots dynamically by strategy name and configuration. The bot manager is used by both the REST API and the web GUI to ensure consistent bot lifecycle management.

Main Features:
- Dynamically load and start bot classes by strategy name
- Assign unique bot IDs and track running bots and their threads
- Stop bots and clean up resources
- Query status and trade history for all running bots
- Used as a singleton registry for all bot management operations

Functions:
- start_bot(strategy_name, config, bot_id=None): Start a new bot instance
- stop_bot(bot_id): Stop a running bot by ID
- get_status(): Get status of all running bots
- get_trades(bot_id): Get trade history for a bot
- get_running_bots(): List all running bot IDs
"""

import importlib
import threading
from typing import Any, Dict, List, Optional

# In-memory registry of running bots and their threads
running_bots: Dict[str, Any] = {}
bot_threads: Dict[str, threading.Thread] = {}


def start_bot(
    strategy_name: str, config: Dict[str, Any], bot_id: Optional[str] = None
) -> str:
    """
    Start a trading bot for the given strategy. Returns bot_id.
    Args:
        strategy_name: Name of the strategy (used for module/class lookup)
        config: Configuration dictionary for the bot
        bot_id: Optional bot ID to assign
    Returns:
        The bot_id of the started bot
    Raises:
        Exception: If a bot with the same ID is already running
    """
    if not bot_id:
        bot_id = (
            strategy_name
            if strategy_name not in running_bots
            else f"{strategy_name}_{len(running_bots)+1}"
        )
    if bot_id in running_bots:
        raise Exception(f"Bot with id {bot_id} is already running.")
    bot_module = importlib.import_module(f"src.trading.{strategy_name}_bot")
    bot_class = getattr(
        bot_module, "".join([w.capitalize() for w in strategy_name.split("_")]) + "Bot"
    )
    bot_instance = bot_class(config)
    running_bots[bot_id] = bot_instance
    t = threading.Thread(target=bot_instance.run, daemon=True)
    bot_threads[bot_id] = t
    t.start()
    return bot_id


def stop_bot(bot_id: str) -> None:
    """
    Stop a running bot by id.
    Args:
        bot_id: The ID of the bot to stop
    Raises:
        Exception: If no bot with the given ID is running
    """
    if bot_id not in running_bots:
        raise Exception(f"No running bot with id {bot_id}.")
    bot = running_bots[bot_id]
    bot.stop()
    del running_bots[bot_id]
    del bot_threads[bot_id]


def get_status() -> Dict[str, str]:
    """
    Get status of all running bots.
    Returns:
        A dictionary mapping bot IDs to their status ("running")
    """
    return {bot_id: "running" for bot_id in running_bots.keys()}


def get_trades(bot_id: str) -> List[Any]:
    """
    Get trade history for a running bot.
    Args:
        bot_id: The ID of the bot
    Returns:
        A list of trade records (may be empty if no trades or bot not found)
    """
    if bot_id in running_bots:
        return getattr(running_bots[bot_id], "trade_history", [])
    return []


def get_running_bots() -> List[str]:
    """
    Get a list of all running bot ids.
    Returns:
        List of bot IDs
    """
    return list(running_bots.keys())
