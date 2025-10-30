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

    async def shutdown(self):
        """Shutdown the strategy manager."""
        _logger.info("Shutting down Enhanced Strategy Manager...")

        await self.stop_monitoring()
        await self.stop_all_strategies()
        await self.broker_manager.shutdown()

        _logger.info("✅ Enhanced Strategy Manager shutdown complete")