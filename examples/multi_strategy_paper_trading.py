#!/usr/bin/env python3
"""
Multi-Strategy Paper Trading Example
----------------------------------

This example demonstrates how to run multiple trading strategies
simultaneously using the enhanced broker system.
"""

import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.broker_manager import BrokerManager
from src.trading.broker.config_manager import ConfigManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


async def main():
    """Run multiple strategies with enhanced broker system."""

    print("🚀 Multi-Strategy Paper Trading Demo")
    print("=" * 50)

    # Initialize managers
    config_manager = ConfigManager()
    broker_manager = BrokerManager()

    # Strategy configurations
    strategies = [
        {
            "name": "RSI_Strategy_BTCUSDT",
            "symbol": "BTCUSDT",
            "broker_config": {
                "type": "binance",
                "trading_mode": "paper",
                "name": "rsi_btc_bot",
                "cash": 5000.0,
                "paper_trading_config": {
                    "mode": "realistic",
                    "initial_balance": 5000.0,
                    "commission_rate": 0.001,
                    "slippage_model": "linear",
                    "base_slippage": 0.0005
                }
            }
        },
        {
            "name": "BB_Strategy_ETHUSDT",
            "symbol": "ETHUSDT",
            "broker_config": {
                "type": "binance",
                "trading_mode": "paper",
                "name": "bb_eth_bot",
                "cash": 5000.0,
                "paper_trading_config": {
                    "mode": "realistic",
                    "initial_balance": 5000.0,
                    "commission_rate": 0.001,
                    "slippage_model": "linear",
                    "base_slippage": 0.0005
                }
            }
        }
    ]

    # Start brokers for each strategy
    brokers = {}

    for strategy in strategies:
        print(f"\n📊 Starting {strategy['name']}...")

        # Create broker
        broker_id = await broker_manager.create_broker(
            broker_id=strategy['name'],
            config=strategy['broker_config']
        )

        # Start broker
        success = await broker_manager.start_broker(broker_id)

        if success:
            brokers[strategy['name']] = broker_id
            print(f"✅ {strategy['name']} started successfully")
        else:
            print(f"❌ Failed to start {strategy['name']}")

    if not brokers:
        print("❌ No brokers started successfully")
        return

    print(f"\n🎯 Running {len(brokers)} strategies simultaneously...")
    print("Press Ctrl+C to stop all strategies")

    try:
        # Monitor strategies
        while True:
            print(f"\n📈 Strategy Status Report - {asyncio.get_event_loop().time():.0f}s")
            print("-" * 60)

            for strategy_name, broker_id in brokers.items():
                status = await broker_manager.get_broker_status(broker_id)

                if status:
                    print(f"🤖 {strategy_name}:")
                    print(f"   Status: {status.get('status', 'Unknown')}")
                    print(f"   Uptime: {status.get('uptime_seconds', 0):.0f}s")
                    print(f"   Health: {status.get('health_status', 'Unknown')}")

                    # Get portfolio info if available
                    try:
                        broker = broker_manager.brokers.get(broker_id)
                        if broker and hasattr(broker, 'get_portfolio'):
                            portfolio = await broker.get_portfolio()
                            if portfolio:
                                print(f"   Balance: ${portfolio.total_value:.2f}")
                                print(f"   P&L: ${portfolio.realized_pnl + portfolio.unrealized_pnl:.2f}")
                    except Exception as e:
                        print(f"   Portfolio: Error - {e}")
                else:
                    print(f"❌ {strategy_name}: No status available")

            # Wait before next update
            await asyncio.sleep(30)

    except KeyboardInterrupt:
        print(f"\n🛑 Stopping all strategies...")

        # Stop all brokers
        for strategy_name, broker_id in brokers.items():
            print(f"Stopping {strategy_name}...")
            await broker_manager.stop_broker(broker_id)

        print("✅ All strategies stopped")


if __name__ == "__main__":
    asyncio.run(main())