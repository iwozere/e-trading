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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.trading.base_trading_bot import BaseTradingBot
from src.trading.broker.broker_factory import get_broker
from src.trading.broker.broker_manager import BrokerManager
from src.strategy.custom_strategy import CustomStrategy
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
        self.broker = None
        self.trading_bot = None
        self.status = 'stopped'
        self.start_time = None
        self.error_count = 0
        self.last_error = None

    async def start(self) -> bool:
        """Start the strategy instance."""
        try:
            _logger.info("Starting strategy instance: %s", self.name)

            # Create broker
            broker_config = self.config['broker']
            self.broker = get_broker(broker_config)

            if not self.broker:
                raise Exception("Failed to create broker")

            # Create strategy class instance
            strategy_config = self.config['strategy']
            strategy_class = self._get_strategy_class(strategy_config['type'])

            # Create trading bot with strategy
            bot_config = self._build_bot_config()
            self.trading_bot = BaseTradingBot(
                config=bot_config,
                strategy_class=strategy_class,
                parameters=strategy_config.get('parameters', {}),
                broker=self.broker,
                paper_trading=broker_config.get('trading_mode') == 'paper',
                bot_id=self.instance_id
            )

            # Start the trading bot
            await self._start_trading_bot()

            self.status = 'running'
            self.start_time = datetime.now(timezone.utc)
            _logger.info("✅ Strategy instance %s started successfully", self.name)
            return True

        except Exception as e:
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)
            _logger.exception("❌ Failed to start strategy instance:", self.name)
            return False

    async def stop(self) -> bool:
        """Stop the strategy instance."""
        try:
            _logger.info("Stopping strategy instance: %s", self.name)

            if self.trading_bot:
                await self._stop_trading_bot()
                self.trading_bot = None

            if self.broker:
                await self.broker.disconnect()
                self.broker = None

            self.status = 'stopped'
            _logger.info("✅ Strategy instance %s stopped successfully", self.name)
            return True

        except Exception as e:
            self.status = 'error'
            self.error_count += 1
            self.last_error = str(e)
            _logger.exception("❌ Failed to stop strategy instance:", self.name)
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
        """Get strategy class based on type."""
        # For now, we'll use CustomStrategy as the base
        # This can be extended to support other strategy types
        if strategy_type.lower() in ['custom', 'customstrategy']:
            return CustomStrategy
        else:
            # Default to CustomStrategy
            _logger.warning("Unknown strategy type %s, using CustomStrategy", strategy_type)
            return CustomStrategy

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


class StrategyManager:
    """Manages multiple strategy instances in a single service."""

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
        """Load strategy configurations from the database trading_bots table.

        Behavior:
        - Reads all bots where status != 'disabled'.
        - Validates each bot config; invalid ones are skipped and marked error.
        - Builds or updates StrategyInstance entries keyed by bot id.
        """
        try:
            bots = trading_service.get_enabled_bots(user_id)
            if not bots:
                _logger.warning("No enabled bots found in DB%s",
                                f" for user_id={user_id}" if user_id else "")

            loaded = 0
            for bot in bots:
                instance_id = str(bot["id"])  # Use DB id as instance id

                # Validate database record/config
                is_valid, errors, warnings = trading_service.validate_bot_configuration(bot["id"])
                if not is_valid:
                    _logger.error("Bot %s config invalid, skipping. Errors: %s", bot["id"], errors)
                    try:
                        trading_service.update_bot_status(bot["id"], "error", error_message="; ".join(errors))
                    except Exception:
                        pass
                    continue

                # Map db record to StrategyInstance config
                si_config = self._db_bot_to_strategy_config(bot)

                if instance_id in self.strategy_instances:
                    # Update existing config
                    self.strategy_instances[instance_id].config = si_config
                else:
                    # Create new instance
                    self.strategy_instances[instance_id] = StrategyInstance(instance_id, si_config)
                loaded += 1

            _logger.info("Loaded %d strategy configs from DB", loaded)
            return True

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