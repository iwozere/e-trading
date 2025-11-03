#!/usr/bin/env python3
"""
Enhanced Strategy Manager
------------------------

This module manages multiple strategy instances within a single service.
Each strategy instance can run with different broker configurations (paper/live).

Features:
- Multiple strategy instances in one service
- Per-strategy broker configuration (paper/live)
- Integration with existing BaseTradingBot and CustomStrategy
- Health monitoring and auto-recovery
- Unified logging and notifications
"""

import asyncio
import json
import uuid
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

import backtrader as bt

from src.trading.base_trading_bot import BaseTradingBot
from src.trading.broker.broker_factory import get_broker
from src.trading.broker.broker_manager import BrokerManager
from src.trading.strategy_handler import strategy_handler
from src.data.feed.data_feed_factory import DataFeedFactory
from src.notification.logger import setup_logger
from src.data.db.services.trading_service import trading_service

_logger = setup_logger(__name__)


class StrategyInstance:
    """Represents a single strategy instance with its own configuration and broker."""

    def __init__(self, instance_id: str, config: Dict[str, Any]):
        """Initialize strategy instance."""
        self.instance_id = instance_id
        self.config = config
        self.name = config.get('name', f'Strategy_{instance_id}')

        # Core components
        self.broker = None
        self.trading_bot = None
        self.data_feed = None
        self.cerebro = None

        # Status tracking
        self.status = 'stopped'
        self.start_time = None
        self.error_count = 0
        self.last_error = None
        self.is_running = False
        self.should_stop = False

        # Threading for monitoring
        self.monitor_thread = None

    async def start(self) -> bool:
        """
        Start the strategy instance with full Backtrader integration.

        Refactored from LiveTradingBot.start() to include:
        - Data feed creation
        - Backtrader setup
        - Trading loop execution
        """
        try:
            _logger.info("Starting strategy instance: %s", self.name)

            # Create broker
            broker_config = self.config['broker']
            self.broker = get_broker(broker_config)
            if not self.broker:
                raise Exception("Failed to create broker")

            _logger.info("Created broker for %s: %s (mode: %s)",
                        self.name,
                        broker_config.get('type'),
                        broker_config.get('trading_mode'))

            # Get strategy class using StrategyHandler
            strategy_config = self.config['strategy']
            strategy_class = self._get_strategy_class(strategy_config['type'])
            _logger.info("Loaded strategy class: %s", strategy_class.__name__)

            # Create data feed (from LiveTradingBot._create_data_feed)
            if not self._create_data_feed():
                raise RuntimeError("Failed to create data feed")

            # Setup Backtrader (from LiveTradingBot._setup_backtrader)
            if not self._setup_backtrader(strategy_class):
                raise RuntimeError("Failed to setup Backtrader")

            # Start monitoring thread for data feed health
            self.monitor_thread = threading.Thread(
                target=self._monitor_data_feed,
                daemon=True,
                name=f"Monitor-{self.name}"
            )
            self.monitor_thread.start()
            _logger.info("Started data feed monitor thread for %s", self.name)

            # Set status
            self.status = 'running'
            self.start_time = datetime.now(timezone.utc)
            self.is_running = True

            # Update database status
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "running",
                    started_at=self.start_time
                )
            except Exception as e:
                _logger.warning("Failed to update bot status in DB: %s", e)

            # Start Backtrader in background task
            asyncio.create_task(self._run_backtrader_async())

            _logger.info("✅ Strategy instance %s started successfully", self.name)
            return True

        except Exception as e:
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)
            _logger.exception("❌ Failed to start strategy instance %s:", self.name)

            # Update database with error status
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "error",
                    error_message=str(e)
                )
            except Exception:
                pass

            return False

    async def stop(self) -> bool:
        """
        Stop the strategy instance gracefully.

        Refactored from LiveTradingBot.stop() to include:
        - Data feed shutdown
        - Backtrader cleanup
        - State persistence
        """
        try:
            _logger.info("Stopping strategy instance: %s", self.name)

            self.should_stop = True
            self.is_running = False

            # Stop data feed
            if self.data_feed:
                try:
                    self.data_feed.stop()
                    _logger.info("Stopped data feed for %s", self.name)
                except Exception as e:
                    _logger.warning("Error stopping data feed: %s", e)

            # Stop Cerebro if running
            if self.cerebro:
                try:
                    # Backtrader doesn't have explicit stop, it completes when data ends
                    _logger.info("Backtrader will complete naturally for %s", self.name)
                except Exception as e:
                    _logger.warning("Error with Backtrader cleanup: %s", e)

            # Stop broker connection
            if self.broker:
                try:
                    await self.broker.disconnect()
                    _logger.info("Disconnected broker for %s", self.name)
                except Exception as e:
                    _logger.warning("Error disconnecting broker: %s", e)

            # Update status
            self.status = 'stopped'

            # Update database status
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "stopped"
                )
            except Exception as e:
                _logger.warning("Failed to update bot status in DB: %s", e)

            _logger.info("✅ Strategy instance %s stopped successfully", self.name)
            return True

        except Exception as e:
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)
            _logger.exception("❌ Failed to stop strategy instance %s:", self.name)
            return False

    async def restart(self) -> bool:
        """Restart the strategy instance."""
        _logger.info("Restarting strategy instance: %s", self.name)
        await self.stop()
        await asyncio.sleep(2)  # Brief pause
        return await self.start()

    def get_status(self) -> Dict[str, Any]:
        """Get strategy instance status."""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        return {
            'instance_id': self.instance_id,
            'name': self.name,
            'status': self.status,
            'uptime_seconds': uptime,
            'error_count': self.error_count,
            'last_error': self.last_error,
            'broker_type': self.config['broker'].get('type'),
            'trading_mode': self.config['broker'].get('trading_mode'),
            'symbol': self.config.get('symbol'),
            'strategy_type': self.config['strategy'].get('type')
        }

    def _get_strategy_class(self, strategy_type: str):
        """
        Get strategy class based on type using StrategyHandler.

        Args:
            strategy_type: Strategy type from configuration

        Returns:
            Strategy class to instantiate
        """
        try:
            # Use StrategyHandler for dynamic strategy loading
            return strategy_handler.get_strategy_class(strategy_type)
        except Exception as e:
            _logger.exception("Error getting strategy class for %s:", strategy_type)
            # StrategyHandler already handles fallback to CustomStrategy
            raise

    def _build_bot_config(self) -> Dict[str, Any]:
        """Build configuration for BaseTradingBot."""
        return {
            'trading_pair': self.config.get('symbol', 'BTCUSDT'),
            'initial_balance': self.config['broker'].get('cash', 10000.0),
            'notifications': self.config.get('notifications', {}),
            'risk_management': self.config.get('risk_management', {}),
            'logging': self.config.get('logging', {}),
            'data': self.config.get('data', {}),
            'trading': self.config.get('trading', {})
        }

    async def _start_trading_bot(self):
        """Start the trading bot (placeholder for actual implementation)."""
        # This would integrate with your existing BaseTradingBot.start() method
        # For now, we'll simulate starting
        _logger.info("Trading bot for %s would start here", self.name)
        # self.trading_bot.start()  # Uncomment when ready

    async def _stop_trading_bot(self):
        """Stop the trading bot (placeholder for actual implementation)."""
        # This would integrate with your existing BaseTradingBot.stop() method
        _logger.info("Trading bot for %s would stop here", self.name)
        # self.trading_bot.stop()  # Uncomment when ready

    def _create_data_feed(self) -> bool:
        """
        Create and initialize the data feed.

        Refactored from LiveTradingBot._create_data_feed()
        """
        try:
            data_config = self.config.get('data', {})

            # Ensure required fields
            if 'data_source' not in data_config:
                data_config['data_source'] = self.config['broker'].get('type', 'binance')
            if 'symbol' not in data_config:
                data_config['symbol'] = self.config.get('symbol', 'BTCUSDT')
            if 'interval' not in data_config:
                data_config['interval'] = '1h'
            if 'lookback_bars' not in data_config:
                data_config['lookback_bars'] = 500

            # Add callback for new data notifications (optional)
            def on_new_bar(symbol, timestamp, data):
                self._notify_new_bar(symbol, timestamp, data)
            data_config["on_new_bar"] = on_new_bar

            self.data_feed = DataFeedFactory.create_data_feed(data_config)

            if self.data_feed is None:
                raise ValueError("Failed to create data feed")

            _logger.info("Created data feed for %s: %s (%s)",
                        self.name,
                        data_config.get('symbol'),
                        data_config.get('interval'))
            return True

        except Exception as e:
            _logger.exception("Error creating data feed for %s:", self.name)
            return False

    def _setup_backtrader(self, strategy_class) -> bool:
        """
        Setup Backtrader engine.

        Refactored from LiveTradingBot._setup_backtrader()
        """
        try:
            self.cerebro = bt.Cerebro()

            # Add data feed
            self.cerebro.adddata(self.data_feed)
            _logger.info("Added data feed to Cerebro for %s", self.name)

            # Add strategy with parameters
            strategy_params = self.config['strategy'].get('parameters', {})
            self.cerebro.addstrategy(strategy_class, **strategy_params)
            _logger.info("Added strategy %s to Cerebro", strategy_class.__name__)

            # Setup broker
            if self.broker:
                self.cerebro.broker = self.broker
                _logger.info("Assigned broker to Cerebro")

            # Setup initial cash
            initial_balance = self.config['broker'].get('cash', 10000.0)
            self.cerebro.broker.setcash(initial_balance)
            _logger.info("Set initial cash: $%.2f", initial_balance)

            # Setup commission
            commission = self.config.get('commission', 0.001)  # 0.1% default
            self.cerebro.broker.setcommission(commission=commission)
            _logger.info("Set commission: %.4f", commission)

            _logger.info("✅ Backtrader setup complete for %s", self.name)
            return True

        except Exception as e:
            _logger.exception("Error setting up Backtrader for %s:", self.name)
            return False

    async def _run_backtrader_async(self):
        """
        Run Backtrader engine in async context.

        Refactored from LiveTradingBot._run_backtrader()
        """
        try:
            _logger.info("Starting Backtrader engine for %s...", self.name)

            # Run Backtrader (blocking call, but in separate task)
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_backtrader_sync
            )

            _logger.info("Backtrader engine completed for %s", self.name)

        except Exception as e:
            _logger.exception("Error in Backtrader engine for %s:", self.name)
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)

            # Update database with error
            try:
                trading_service.update_bot_status(
                    int(self.instance_id),
                    "error",
                    error_message=f"Backtrader error: {str(e)}"
                )
            except Exception:
                pass

    def _run_backtrader_sync(self):
        """Synchronous Backtrader execution."""
        results = self.cerebro.run()
        return results

    def _notify_new_bar(self, symbol: str, timestamp, data: Dict[str, Any]):
        """
        Notify about new data bar.

        Refactored from LiveTradingBot._notify_new_bar()
        """
        try:
            _logger.debug(
                "New %s bar: O=%.4f H=%.4f L=%.4f C=%.4f",
                symbol,
                data.get('open', 0),
                data.get('high', 0),
                data.get('low', 0),
                data.get('close', 0)
            )
        except Exception as e:
            _logger.debug("Error notifying new bar: %s", e)

    def _monitor_data_feed(self):
        """
        Monitor data feed health and reconnect if needed.

        Refactored from LiveTradingBot._monitor_data_feed()
        """
        _logger.info("Data feed monitor started for %s", self.name)

        while self.is_running and not self.should_stop:
            try:
                if self.data_feed:
                    status = self.data_feed.get_status()
                    if not status.get("is_connected", False):
                        _logger.warning(
                            "Data feed disconnected for %s, attempting reconnect...",
                            self.name
                        )
                        self._reconnect_data_feed()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                _logger.exception("Error in data feed monitor for %s:", self.name)
                time.sleep(60)

        _logger.info("Data feed monitor stopped for %s", self.name)

    def _reconnect_data_feed(self):
        """
        Reconnect data feed.

        Refactored from LiveTradingBot._reconnect_data_feed()
        """
        try:
            if self.data_feed:
                self.data_feed.stop()
                time.sleep(5)

            if self._create_data_feed():
                _logger.info("Data feed reconnected successfully for %s", self.name)
                # Note: Backtrader reconnection would require strategy restart
                # For now, just log the reconnection
            else:
                _logger.error("Failed to reconnect data feed for %s", self.name)
                self.error_count += 1

        except Exception as e:
            _logger.exception("Error reconnecting data feed for %s:", self.name)


class StrategyManager:
    """
    Bot manager and SOLE CONFIGURATION LOADER.

    This is the ONLY component that loads configurations from the database.
    All other components receive configs through this manager.

    Responsibilities:
    - Load ALL bot configurations from database (ONLY place this happens)
    - Create and manage StrategyInstance objects
    - Handle bot lifecycle (start/stop/restart)
    - Monitor bot health and performance
    - Update database with bot status and metrics
    - DB polling for hot-reload
    """

    def __init__(self):
        """Initialize the strategy manager."""
        self.strategy_instances: Dict[str, StrategyInstance] = {}
        self.broker_manager = BrokerManager()
        self.is_running = False
        self.monitoring_task = None
        self.db_poll_task = None
        self._db_poll_running = False
        self._db_poll_user_id: Optional[int] = None
        self._db_poll_interval: int = 60

    async def load_strategies_from_config(self, config_file: str) -> bool:
        """Load strategy configurations from JSON file."""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                _logger.error("Configuration file not found: %s", config_path)
                return False

            with open(config_path, 'r') as f:
                config = json.load(f)

            strategies = config.get('strategies', [])
            if not strategies:
                _logger.error("No strategies found in configuration")
                return False

            _logger.info("Loading %d strategy configurations", len(strategies))

            for strategy_config in strategies:
                instance_id = strategy_config.get('id') or str(uuid.uuid4())

                # Validate required fields
                required_fields = ['name', 'symbol', 'broker', 'strategy']
                missing_fields = [field for field in required_fields if field not in strategy_config]

                if missing_fields:
                    _logger.error("Strategy %s: Missing required fields: %s", instance_id, missing_fields)
                    continue

                # Create strategy instance
                instance = StrategyInstance(instance_id, strategy_config)
                self.strategy_instances[instance_id] = instance

                _logger.info("Loaded strategy: %s (%s)", instance.name, instance_id)

            _logger.info("Successfully loaded %d strategies", len(self.strategy_instances))
            return True

        except Exception as e:
            _logger.exception("Failed to load strategies from config:")
            return False

    async def start_all_strategies(self) -> int:
        """Start all configured strategy instances."""
        _logger.info("Starting all strategy instances...")

        started_count = 0

        for instance_id, instance in self.strategy_instances.items():
            # Check if strategy is enabled
            if not instance.config.get('enabled', True):
                _logger.info("Strategy %s is disabled, skipping", instance.name)
                continue

            success = await instance.start()
            if success:
                started_count += 1

            # Small delay between starts
            await asyncio.sleep(1)

        _logger.info("Started %d/%d strategy instances", started_count, len(self.strategy_instances))
        return started_count

    async def stop_all_strategies(self):
        """Stop all running strategy instances."""
        _logger.info("Stopping all strategy instances...")

        stop_tasks = []
        for instance in self.strategy_instances.values():
            if instance.status == 'running':
                task = asyncio.create_task(instance.stop())
                stop_tasks.append(task)

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        _logger.info("All strategy instances stopped")

    async def start_strategy(self, instance_id: str) -> bool:
        """Start a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            _logger.error("Strategy instance %s not found", instance_id)
            return False

        return await self.strategy_instances[instance_id].start()

    async def stop_strategy(self, instance_id: str) -> bool:
        """Stop a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            _logger.error("Strategy instance %s not found", instance_id)
            return False

        return await self.strategy_instances[instance_id].stop()

    async def restart_strategy(self, instance_id: str) -> bool:
        """Restart a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            _logger.error("Strategy instance %s not found", instance_id)
            return False

        return await self.strategy_instances[instance_id].restart()

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get status of all strategy instances."""
        return [instance.get_status() for instance in self.strategy_instances.values()]

    def get_strategy_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific strategy instance."""
        if instance_id not in self.strategy_instances:
            return None

        return self.strategy_instances[instance_id].get_status()

    async def start_monitoring(self):
        """Start monitoring all strategy instances."""
        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitor_strategies())

    async def stop_monitoring(self):
        """Stop monitoring."""
        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Stop DB polling if running
        self._db_poll_running = False
        if self.db_poll_task:
            self.db_poll_task.cancel()
            try:
                await self.db_poll_task
            except asyncio.CancelledError:
                pass

    async def _monitor_strategies(self):
        """Monitor strategy instances and handle auto-recovery."""
        while self.is_running:
            try:
                # Check health of all running strategies
                for instance_id, instance in self.strategy_instances.items():
                    if instance.status == 'running':
                        # Here you could add health checks
                        # For now, we'll just log status
                        pass
                    elif instance.status == 'error' and instance.error_count < 3:
                        # Auto-recovery for failed strategies (max 3 attempts)
                        _logger.warning("Attempting auto-recovery for %s", instance.name)
                        await instance.restart()

                # Log periodic status
                running_count = sum(1 for i in self.strategy_instances.values() if i.status == 'running')
                _logger.info("📊 Strategy Monitor: %d/%d running", running_count, len(self.strategy_instances))

                await asyncio.sleep(60)  # Monitor every minute

            except Exception as e:
                _logger.exception("Error in strategy monitoring:")
                await asyncio.sleep(10)

    # -------------------- DB-backed loading and polling --------------------
    async def load_strategies_from_db(self, user_id: Optional[int] = None) -> bool:
        """
        Load strategy configurations from the database - ONLY PLACE THIS HAPPENS.

        This is the SOLE configuration loader. No other component should load from DB.

        Behavior:
        - Reads all bots where status != 'disabled'
        - Validates each bot config using trading_service AND StrategyHandler
        - Builds or updates StrategyInstance entries keyed by bot id
        - Invalid configs are skipped and marked as 'error' in DB

        Args:
            user_id: Optional user ID to filter bots

        Returns:
            True if successfully loaded at least one bot, False otherwise
        """
        try:
            _logger.info("=" * 80)
            _logger.info("LOADING BOT CONFIGURATIONS FROM DATABASE (SOLE CONFIG LOADER)")
            _logger.info("=" * 80)

            bots = trading_service.get_enabled_bots(user_id)
            if not bots:
                _logger.warning("No enabled bots found in DB%s",
                                f" for user_id={user_id}" if user_id else "")
                return False

            _logger.info("Found %d enabled bot(s) in database", len(bots))

            loaded = 0
            for bot in bots:
                instance_id = str(bot["id"])
                bot_name = bot.get("description") or f"Bot {bot['id']}"

                _logger.info("Processing bot: %s (ID: %s)", bot_name, instance_id)

                # Validate database record/config
                is_valid, errors, warnings = trading_service.validate_bot_configuration(bot["id"])
                if not is_valid:
                    _logger.error("Bot %s config invalid, skipping. Errors: %s", bot["id"], errors)
                    try:
                        trading_service.update_bot_status(bot["id"], "error", error_message="; ".join(errors))
                    except Exception:
                        pass
                    continue

                # Log warnings
                for warning in warnings:
                    _logger.warning("Bot %s: %s", bot["id"], warning)

                # Map db record to StrategyInstance config
                si_config = self._db_bot_to_strategy_config(bot)

                # Additional validation using StrategyHandler
                strategy_type = si_config.get("strategy", {}).get("type", "CustomStrategy")
                strategy_config = si_config.get("strategy", {})

                is_strategy_valid, strategy_errors, strategy_warnings = strategy_handler.validate_strategy_config(
                    strategy_type,
                    strategy_config
                )

                if not is_strategy_valid:
                    _logger.error("Bot %s strategy config invalid: %s", bot["id"], strategy_errors)
                    try:
                        trading_service.update_bot_status(
                            bot["id"],
                            "error",
                            error_message="Strategy validation failed: " + "; ".join(strategy_errors)
                        )
                    except Exception:
                        pass
                    continue

                # Log strategy warnings
                for warning in strategy_warnings:
                    _logger.warning("Bot %s strategy: %s", bot["id"], warning)

                # Create or update instance
                if instance_id in self.strategy_instances:
                    # Update existing config
                    _logger.info("Updating existing bot instance: %s", bot_name)
                    self.strategy_instances[instance_id].config = si_config
                else:
                    # Create new instance
                    _logger.info("Creating new bot instance: %s", bot_name)
                    self.strategy_instances[instance_id] = StrategyInstance(instance_id, si_config)

                loaded += 1
                _logger.info("✅ Successfully loaded bot: %s", bot_name)

            _logger.info("=" * 80)
            _logger.info("CONFIGURATION LOADING COMPLETE: %d/%d bots loaded", loaded, len(bots))
            _logger.info("=" * 80)

            return loaded > 0

        except Exception:
            _logger.exception("Failed to load strategies from DB:")
            return False

    def _db_bot_to_strategy_config(self, bot: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a trading_bots row dict into StrategyInstance config shape."""
        cfg = bot.get("config") or {}
        # Prefer fields from config; fallback to DB row
        name = cfg.get("name") or bot.get("description") or f"Bot {bot.get('id')}"
        symbol = cfg.get("symbol") or cfg.get("trading_pair") or cfg.get("pair") or cfg.get("ticker")
        broker_cfg = cfg.get("broker") or {}
        strategy_cfg = cfg.get("strategy") or {}

        # Enabled if DB status not 'disabled' and config doesn't explicitly disable
        enabled = bot.get("status") != "disabled" and cfg.get("enabled", True)

        return {
            "id": str(bot.get("id")),
            "name": name,
            "enabled": enabled,
            "symbol": symbol or "BTCUSDT",
            "broker": broker_cfg,
            "strategy": strategy_cfg,
            "data": cfg.get("data", {}),
            "trading": cfg.get("trading", {}),
            "risk_management": cfg.get("risk_management", {}),
            "notifications": cfg.get("notifications", {}),
        }

    async def start_db_polling(self, user_id: Optional[int] = None, interval_seconds: int = 60):
        """Continuously poll DB and sync strategy processes.

        Actions on each poll:
        - Start new StrategyInstances for bots present in DB with status 'enabled' that aren't running yet.
        - Stop and remove instances for bots that became 'disabled' or were deleted.
        - Refresh configs for bots that changed.
        - Update bot statuses in DB to 'running' when started and 'stopped' when stopped.
        """
        self._db_poll_user_id = user_id
        self._db_poll_interval = max(5, interval_seconds)
        self._db_poll_running = True

        # Initial load
        await self.load_strategies_from_db(user_id)

        async def _poll_loop():
            while self._db_poll_running:
                try:
                    # Fetch enabled bots (status != disabled)
                    enabled_bots = {str(b["id"]): b for b in trading_service.get_enabled_bots(user_id)}

                    # Start or update instances for enabled bots
                    for bot_id, bot in enabled_bots.items():
                        is_valid, errors, _ = trading_service.validate_bot_configuration(int(bot_id))
                        if not is_valid:
                            _logger.error("Bot %s invalid during polling, skipping start. Errors: %s", bot_id, errors)
                            try:
                                trading_service.update_bot_status(int(bot_id), "error", error_message="; ".join(errors))
                            except Exception:
                                pass
                            continue

                        desired_status = bot.get("status", "enabled")
                        exists = bot_id in self.strategy_instances
                        if not exists:
                            # Create instance
                            si_config = self._db_bot_to_strategy_config(bot)
                            self.strategy_instances[bot_id] = StrategyInstance(bot_id, si_config)

                        instance = self.strategy_instances[bot_id]
                        # If bot should be running and isn't, start it
                        if desired_status in ("enabled", "starting") and instance.status != "running":
                            try:
                                trading_service.update_bot_status(int(bot_id), "starting")
                            except Exception:
                                pass
                            ok = await instance.start()
                            try:
                                trading_service.update_bot_status(int(bot_id), "running" if ok else "error",
                                                                 error_message=None if ok else instance.last_error)
                            except Exception:
                                pass

                        # If bot is running but config changed, we could implement hot-reload: simple restart
                        else:
                            # Refresh config in case changed
                            instance.config = self._db_bot_to_strategy_config(bot)

                    # Stop instances for bots no longer enabled
                    to_stop = [iid for iid in list(self.strategy_instances.keys()) if iid not in enabled_bots]
                    for iid in to_stop:
                        instance = self.strategy_instances.get(iid)
                        if instance and instance.status == "running":
                            await instance.stop()
                            try:
                                trading_service.update_bot_status(int(iid), "stopped")
                            except Exception:
                                pass
                        # Remove from manager
                        self.strategy_instances.pop(iid, None)

                    await asyncio.sleep(self._db_poll_interval)

                except asyncio.CancelledError:
                    break
                except Exception:
                    _logger.exception("Error during DB polling loop:")
                    await asyncio.sleep(self._db_poll_interval)

        # Launch poll loop
        self.db_poll_task = asyncio.create_task(_poll_loop())

    async def shutdown(self):
        """Shutdown the strategy manager."""
        _logger.info("Shutting down Enhanced Strategy Manager...")

        await self.stop_monitoring()
        await self.stop_all_strategies()
        await self.broker_manager.shutdown()

        _logger.info("✅ Enhanced Strategy Manager shutdown complete")