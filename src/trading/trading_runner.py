#!/usr/bin/env python3
"""
Enhanced Multi-Strategy Trading System Runner
--------------------------------------------

This is the main runner for the enhanced multi-strategy trading system.
It manages multiple trading strategies simultaneously with advanced monitoring,
health checks, and recovery capabilities.

Features:
- Multiple strategies in one service
- Real-time health monitoring
- Automatic recovery and restart
- Performance analytics
- Web dashboard (future)
- Comprehensive logging and notifications

Usage:
    python src/trading/trading_runner.py [config_file]

Examples:
    python src/trading/trading_runner.py
    python src/trading/trading_runner.py config/enhanced_trading/simple_multi_strategy.json
"""

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.broker_manager import BrokerManager
from src.trading.broker.config_manager import ConfigManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class EnhancedMultiStrategyRunner:
    """Enhanced multi-strategy trading system runner."""

    def __init__(self, config_file: Optional[str] = None):
        """Initialize the runner."""
        self.config_file = config_file or "config/enhanced_trading/simple_multi_strategy.json"
        self.config = None
        self.broker_manager = BrokerManager()
        self.config_manager = ConfigManager()
        self.running_strategies = {}
        self.is_running = False
        self.start_time = None

    async def load_config(self) -> bool:
        """Load the multi-strategy configuration."""
        try:
            config_path = Path(self.config_file)
            if not config_path.exists():
                _logger.error("Configuration file not found: %s", config_path)
                return False

            with open(config_path, 'r') as f:
                self.config = json.load(f)

            _logger.info("Loaded configuration: %s", config_path)
            _logger.info("System: %s v%s", self.config['system']['name'], self.config['system']['version'])
            _logger.info("Strategies to run: %s", len(self.config['strategies']))

            return True

        except Exception as e:
            _logger.exception("Failed to load configuration:")
            return False

    async def validate_config(self) -> bool:
        """Validate the configuration."""
        if not self.config:
            return False

        required_sections = ['system', 'strategies']
        for section in required_sections:
            if section not in self.config:
                _logger.error("Missing required section: %s", section)
                return False

        if not self.config['strategies']:
            _logger.error("No strategies defined in configuration")
            return False

        # Validate each strategy
        for i, strategy in enumerate(self.config['strategies']):
            required_fields = ['id', 'name', 'symbol', 'broker_config']
            for field in required_fields:
                if field not in strategy:
                    _logger.error("Strategy %s: Missing required field '%s'", i, field)
                    return False

        _logger.info("Configuration validation passed")
        return True

    async def start_strategy(self, strategy_config: Dict) -> bool:
        """Start a single strategy."""
        strategy_id = strategy_config['id']
        strategy_name = strategy_config['name']

        try:
            _logger.info("Starting strategy: %s (%s)", strategy_name, strategy_id)

            # Check if strategy is enabled
            if not strategy_config.get('enabled', True):
                _logger.info("Strategy %s is disabled, skipping", strategy_name)
                return False

            # Create broker for this strategy
            broker_id = await self.broker_manager.create_broker(
                broker_id=strategy_id,
                config=strategy_config['broker_config']
            )

            if not broker_id:
                _logger.error("Failed to create broker for strategy %s", strategy_name)
                return False

            # Start the broker
            success = await self.broker_manager.start_broker(broker_id)

            if success:
                self.running_strategies[strategy_id] = {
                    'name': strategy_name,
                    'broker_id': broker_id,
                    'config': strategy_config,
                    'start_time': datetime.now(timezone.utc),
                    'status': 'running'
                }
                _logger.info("✅ Strategy %s started successfully", strategy_name)
                return True
            else:
                _logger.error("❌ Failed to start broker for strategy %s", strategy_name)
                return False

        except Exception as e:
            _logger.exception("Error starting strategy %s", strategy_name)
            return False

    async def stop_strategy(self, strategy_id: str) -> bool:
        """Stop a single strategy."""
        if strategy_id not in self.running_strategies:
            _logger.warning("Strategy %s not found in running strategies", strategy_id)
            return False

        strategy_info = self.running_strategies[strategy_id]
        strategy_name = strategy_info['name']
        broker_id = strategy_info['broker_id']

        try:
            _logger.info("Stopping strategy: %s", strategy_name)

            # Stop the broker
            success = await self.broker_manager.stop_broker(broker_id)

            if success:
                strategy_info['status'] = 'stopped'
                _logger.info("✅ Strategy %s stopped successfully", strategy_name)
                return True
            else:
                _logger.error("❌ Failed to stop strategy %s", strategy_name)
                return False

        except Exception as e:
            _logger.exception("Error stopping strategy %s:", strategy_name)
            return False

    async def start_all_strategies(self) -> int:
        """Start all configured strategies."""
        _logger.info("Starting all strategies...")

        started_count = 0

        for strategy_config in self.config['strategies']:
            success = await self.start_strategy(strategy_config)
            if success:
                started_count += 1

            # Small delay between starts
            await asyncio.sleep(1)

        _logger.info("Started %s/%s strategies", started_count, len(self.config['strategies']))
        return started_count

    async def stop_all_strategies(self):
        """Stop all running strategies."""
        _logger.info("Stopping all strategies...")

        stop_tasks = []
        for strategy_id in list(self.running_strategies.keys()):
            task = asyncio.create_task(self.stop_strategy(strategy_id))
            stop_tasks.append(task)

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        self.running_strategies.clear()
        _logger.info("All strategies stopped")

    async def monitor_strategies(self):
        """Monitor running strategies and provide status updates."""
        while self.is_running:
            try:
                if not self.running_strategies:
                    await asyncio.sleep(10)
                    continue

                _logger.info("📊 Strategy Status Report - %s active", len(self.running_strategies))
                _logger.info("-" * 80)

                for strategy_id, strategy_info in self.running_strategies.items():
                    strategy_name = strategy_info['name']
                    broker_id = strategy_info['broker_id']

                    # Get broker status
                    status = await self.broker_manager.get_broker_status(broker_id)

                    if status:
                        uptime = status.get('uptime_seconds', 0)
                        health = status.get('health_status', 'Unknown')

                        _logger.info("🤖 %s:", strategy_name)
                        _logger.info("   Status: %s", status.get('status', 'Unknown'))
                        _logger.info("   Health: %s", health)
                        _logger.info("   Uptime: %.0fs", uptime)

                        # Get portfolio info if available
                        try:
                            broker = self.broker_manager.brokers.get(broker_id)
                            if broker and hasattr(broker, 'get_portfolio'):
                                portfolio = await broker.get_portfolio()
                                if portfolio:
                                    total_value = portfolio.total_value
                                    pnl = portfolio.realized_pnl + portfolio.unrealized_pnl
                                    _logger.info("   Portfolio: $%.2f (P&L: $%.2f)", total_value, pnl)
                        except Exception as e:
                            _logger.debug("   Portfolio error: %s", e)

                        # Check for unhealthy strategies
                        if health != 'healthy':
                            _logger.warning("⚠️  Strategy %s health issue: %s", strategy_name, health)
                    else:
                        _logger.warning("❌ %s: No status available", strategy_name)

                # System summary
                uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                _logger.info("📈 System uptime: %.0fs | Active strategies: %d", uptime, len(self.running_strategies))

                # Wait before next monitoring cycle
                await asyncio.sleep(30)

            except Exception as e:
                _logger.exception("Error in strategy monitoring:")
                await asyncio.sleep(10)

    async def run(self):
        """Main run method."""
        try:
            _logger.info("🚀 Enhanced Multi-Strategy Trading System Starting...")
            _logger.info("=" * 80)

            # Load and validate configuration
            if not await self.load_config():
                return False

            if not await self.validate_config():
                return False

            # Set running flag and start time
            self.is_running = True
            self.start_time = datetime.now(timezone.utc)

            # Start all strategies
            started_count = await self.start_all_strategies()

            if started_count == 0:
                _logger.error("No strategies started successfully")
                return False

            _logger.info("🎯 System running with %d strategies", started_count)
            _logger.info("Press Ctrl+C to stop the system")
            _logger.info("=" * 80)

            # Start monitoring
            monitor_task = asyncio.create_task(self.monitor_strategies())

            # Wait for shutdown signal
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

            return True

        except Exception as e:
            _logger.exception("Error in main run loop:")
            return False
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown."""
        _logger.info("🛑 Shutting down Enhanced Multi-Strategy System...")

        self.is_running = False

        # Stop all strategies
        await self.stop_all_strategies()

        # Shutdown broker manager
        await self.broker_manager.shutdown()

        _logger.info("✅ System shutdown complete")


def setup_signal_handlers(runner):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        _logger.info("Received signal %s, initiating shutdown...", signum)
        asyncio.create_task(runner.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Strategy Trading System')
    parser.add_argument('config', nargs='?',
                       default='config/enhanced_trading/simple_multi_strategy.json',
                       help='Configuration file path')
    parser.add_argument('--setup', action='store_true',
                       help='Run setup first')

    args = parser.parse_args()

    # Run setup if requested
    if args.setup:
        _logger.info("Running setup...")
        import setup_enhanced_trading
        setup_enhanced_trading.main()
        return

    # Create and run the system
    runner = EnhancedMultiStrategyRunner(args.config)
    setup_signal_handlers(runner)

    try:
        success = await runner.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt")
        await runner.shutdown()
        sys.exit(0)
    except Exception as e:
        _logger.exception("Unexpected error:")
        await runner.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())