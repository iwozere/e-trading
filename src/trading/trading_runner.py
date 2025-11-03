#!/usr/bin/env python3
"""
Trading Service Runner
----------------------

Service orchestrator for database-driven multi-bot trading system.
This component DOES NOT load configurations - it delegates to StrategyManager.

Responsibilities:
- Service lifecycle management (start/stop)
- System signal handling (SIGINT, SIGTERM)
- Delegate all bot operations to StrategyManager
- Provide service-level health monitoring wrapper

Architecture:
    trading_runner.py (THIS) → Service orchestrator (NO CONFIG LOADING)
        ↓
    strategy_manager.py → SOLE configuration loader from database
        ↓
    StrategyInstance objects → Individual bot execution

Usage:
    python src/trading/trading_runner.py [--user-id USER_ID]

Examples:
    python src/trading/trading_runner.py
    python src/trading/trading_runner.py --user-id 1
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.trading.strategy_manager import StrategyManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TradingServiceRunner:
    """
    Service orchestrator - delegates all bot management to StrategyManager.

    This class DOES NOT:
    - Load configurations (delegated to StrategyManager)
    - Create bot instances (delegated to StrategyManager)
    - Manage individual bots (delegated to StrategyManager)

    This class ONLY:
    - Manages service lifecycle (start/stop)
    - Handles system signals
    - Coordinates startup/shutdown
    """

    def __init__(self, user_id: Optional[int] = None, db_poll_interval: int = 60):
        """
        Initialize the trading service runner.

        Args:
            user_id: Optional user ID to filter bots
            db_poll_interval: Database polling interval in seconds for hot-reload
        """
        self.strategy_manager = StrategyManager()
        self.user_id = user_id
        self.db_poll_interval = db_poll_interval
        self.is_running = False
        self.start_time = None

    async def start_service(self) -> bool:
        """
        Start the trading service - DELEGATES to StrategyManager.

        Returns:
            True if service started successfully
        """
        try:
            _logger.info("🚀 Starting Trading Service...")
            _logger.info("=" * 80)

            self.is_running = True
            self.start_time = datetime.now(timezone.utc)

            # StrategyManager handles ALL config loading
            _logger.info("Loading bot configurations from database...")
            if not await self.strategy_manager.load_strategies_from_db(self.user_id):
                _logger.error("Failed to load bot configurations from database")
                return False

            # Start all loaded strategies
            _logger.info("Starting all bot instances...")
            started_count = await self.strategy_manager.start_all_strategies()

            if started_count == 0:
                _logger.warning("No bots started successfully")
                return False

            _logger.info("✅ Successfully started %d bot(s)", started_count)

            # Start monitoring
            _logger.info("Starting bot monitoring...")
            await self.strategy_manager.start_monitoring()

            # Start DB polling for hot-reload
            _logger.info("Starting database polling for configuration hot-reload...")
            await self.strategy_manager.start_db_polling(
                user_id=self.user_id,
                interval_seconds=self.db_poll_interval
            )

            _logger.info("=" * 80)
            _logger.info("🎯 Trading Service is running with %d active bot(s)", started_count)
            _logger.info("Press Ctrl+C to stop the service")
            _logger.info("=" * 80)

            return True

        except Exception as e:
            _logger.exception("Error starting trading service:")
            return False

    async def stop_service(self):
        """Gracefully stop all bots - DELEGATES to StrategyManager."""
        _logger.info("🛑 Shutting down Trading Service...")

        self.is_running = False

        # Shutdown strategy manager (which handles all bots)
        await self.strategy_manager.shutdown()

        _logger.info("✅ Trading Service shutdown complete")

    async def run(self):
        """Main service loop - coordinates startup and waits for shutdown signal."""
        try:
            # Start the service
            if not await self.start_service():
                _logger.error("Failed to start trading service")
                return False

            # Wait for shutdown signal
            while self.is_running:
                await asyncio.sleep(1)

            return True

        except Exception as e:
            _logger.exception("Error in main run loop:")
            return False
        finally:
            await self.stop_service()


def setup_signal_handlers(runner):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        _logger.info("Received signal %s, initiating shutdown...", signum)
        runner.is_running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main function - service entry point."""
    parser = argparse.ArgumentParser(
        description='Database-Driven Multi-Bot Trading Service'
    )
    parser.add_argument(
        '--user-id',
        type=int,
        default=None,
        help='User ID to filter bots (optional)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=60,
        help='Database polling interval in seconds for config hot-reload (default: 60)'
    )

    args = parser.parse_args()

    # Create and run the service
    runner = TradingServiceRunner(
        user_id=args.user_id,
        db_poll_interval=args.poll_interval
    )
    setup_signal_handlers(runner)

    try:
        success = await runner.run()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt")
        await runner.stop_service()
        sys.exit(0)
    except Exception as e:
        _logger.exception("Unexpected error:")
        await runner.stop_service()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())