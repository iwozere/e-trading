#!/usr/bin/env python3
"""
Enhanced Binance Broker Demo
----------------------------

This script demonstrates the enhanced Binance broker capabilities including:
- Seamless paper-to-live trading mode switching
- Realistic paper trading simulation
- WebSocket market data integration
- Position notifications
- Comprehensive analytics

Usage:
    python examples/enhanced_binance_broker_demo.py --mode paper
    python examples/enhanced_binance_broker_demo.py --mode live --confirm
"""

import asyncio
import argparse
import json
import time
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.trading.broker.broker_factory import get_broker
from src.trading.broker.binance_utils import create_binance_config_template
from src.trading.broker.base_broker import Order, OrderType, OrderSide
from config.donotshare.donotshare import BINANCE_KEY, BINANCE_SECRET, BINANCE_PAPER_KEY, BINANCE_PAPER_SECRET


async def demo_paper_trading():
    """Demonstrate paper trading capabilities."""
    print("🔄 Starting Enhanced Binance Paper Trading Demo")
    print("=" * 60)

    # Create paper trading configuration
    config = create_binance_config_template('paper')
    config['notifications']['email_enabled'] = False  # Disable for demo
    config['notifications']['telegram_enabled'] = False

    print(f"📋 Configuration:")
    print(f"   Trading Mode: {config['trading_mode']}")
    print(f"   Initial Balance: ${config['cash']:,.2f}")
    print(f"   Commission Rate: {config['paper_trading_config']['commission_rate']:.3f}")
    print(f"   Slippage Model: {config['paper_trading_config']['slippage_model']}")
    print()

    # Create broker
    try:
        broker = get_broker(config)
        print(f"✅ Created Binance broker: {broker.get_name()}")
        print(f"   Trading Mode: {broker.get_trading_mode().value}")
        print(f"   Paper Trading: {broker.is_paper_trading()}")
        print(f"   API URL: {broker.client.API_URL}")
        print()
    except Exception as e:
        print(f"❌ Failed to create broker: {e}")
        return

    # Connect to broker
    try:
        print("🔌 Connecting to Binance...")
        connected = await broker.connect()
        if connected:
            print("✅ Connected successfully")

            # Get account info
            account_info = await broker.get_account_info()
            print(f"📊 Account Info:")
            for key, value in account_info.items():
                if key != 'paper_trading_config':
                    print(f"   {key}: {value}")
            print()
        else:
            print("❌ Connection failed")
            return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return

    # Simulate market data
    print("📈 Simulating market data...")
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    prices = {'BTCUSDT': 45000.0, 'ETHUSDT': 3000.0, 'ADAUSDT': 0.5}

    for symbol, price in prices.items():
        broker.update_market_data_cache(symbol, price)
        print(f"   {symbol}: ${price:,.2f}")
    print()

    # Demonstrate order placement
    print("📝 Placing demo orders...")

    # Market buy order
    market_order = Order(
        symbol='BTCUSDT',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.001
    )

    try:
        order_id = await broker.place_order(market_order)
        print(f"✅ Market buy order placed: {order_id}")
        print(f"   Symbol: {market_order.symbol}")
        print(f"   Quantity: {market_order.quantity}")
        print(f"   Status: {market_order.status.value}")
        print()
    except Exception as e:
        print(f"❌ Market order failed: {e}")

    # Limit sell order
    limit_order = Order(
        symbol='BTCUSDT',
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=0.001,
        price=46000.0  # Above current market price
    )

    try:
        order_id = await broker.place_order(limit_order)
        print(f"✅ Limit sell order placed: {order_id}")
        print(f"   Symbol: {limit_order.symbol}")
        print(f"   Price: ${limit_order.price:,.2f}")
        print(f"   Status: {limit_order.status.value}")
        print()
    except Exception as e:
        print(f"❌ Limit order failed: {e}")

    # Check positions
    print("💼 Current positions:")
    try:
        positions = await broker.get_positions()
        if positions:
            for symbol, position in positions.items():
                print(f"   {symbol}:")
                print(f"     Quantity: {position.quantity}")
                print(f"     Avg Price: ${position.average_price:,.2f}")
                print(f"     Market Value: ${position.market_value:,.2f}")
                print(f"     Unrealized P&L: ${position.unrealized_pnl:,.2f}")
        else:
            print("   No positions")
        print()
    except Exception as e:
        print(f"❌ Error getting positions: {e}")

    # Check portfolio
    print("📊 Portfolio summary:")
    try:
        portfolio = await broker.get_portfolio()
        print(f"   Total Value: ${portfolio.total_value:,.2f}")
        print(f"   Cash: ${portfolio.cash:,.2f}")
        print(f"   Unrealized P&L: ${portfolio.unrealized_pnl:,.2f}")
        print(f"   Realized P&L: ${portfolio.realized_pnl:,.2f}")
        print(f"   Total Trades: {portfolio.total_trades}")
        print()
    except Exception as e:
        print(f"❌ Error getting portfolio: {e}")

    # Simulate price movement and process pending orders
    print("📈 Simulating price movement...")
    new_prices = {'BTCUSDT': 46500.0, 'ETHUSDT': 3100.0, 'ADAUSDT': 0.52}

    for symbol, price in new_prices.items():
        broker.update_market_data_cache(symbol, price)
        print(f"   {symbol}: ${price:,.2f} (↗️)")

    # Process pending orders
    await broker.process_market_data_update()
    print("✅ Processed pending orders")
    print()

    # Get execution quality report
    if hasattr(broker, 'get_execution_quality_report'):
        print("📈 Execution Quality Report:")
        try:
            report = broker.get_execution_quality_report()
            if 'error' not in report:
                print(f"   Total Executions: {report['total_executions']}")
                print(f"   Average Slippage: {report['average_slippage_bps']:.2f} bps")
                print(f"   Average Latency: {report['average_latency_ms']:.1f} ms")
                print(f"   Quality Distribution:")
                for quality, percentage in report['quality_distribution'].items():
                    print(f"     {quality}: {percentage:.1f}%")
            else:
                print(f"   {report['error']}")
            print()
        except Exception as e:
            print(f"❌ Error getting execution report: {e}")

    # Get performance report
    if hasattr(broker, 'get_paper_trading_performance_report'):
        print("📊 Performance Report:")
        try:
            report = broker.get_paper_trading_performance_report()
            if 'error' not in report:
                metrics = report['portfolio_metrics']
                stats = report['trading_statistics']

                print(f"   Total Return: {metrics['total_return_pct']:.2f}%")
                print(f"   Win Rate: {metrics['win_rate_pct']:.1f}%")
                print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
                print(f"   Sharpe Ratio: {stats['sharpe_ratio']:.3f}")
                print(f"   Total Commission: ${metrics['total_commission']:.4f}")
                print(f"   Net P&L: ${metrics['net_pnl']:.2f}")
            else:
                print(f"   {report['error']}")
            print()
        except Exception as e:
            print(f"❌ Error getting performance report: {e}")

    # Disconnect
    try:
        await broker.disconnect()
        print("🔌 Disconnected from Binance")
    except Exception as e:
        print(f"❌ Disconnect error: {e}")

    print("✅ Paper trading demo completed!")


async def demo_live_trading():
    """Demonstrate live trading capabilities (with safety checks)."""
    print("⚠️  LIVE TRADING MODE - REAL MONEY WILL BE USED!")
    print("=" * 60)

    # Create live trading configuration
    config = create_binance_config_template('live')
    config['live_trading_confirmed'] = True  # Required for live trading
    config['notifications']['email_enabled'] = False  # Disable for demo
    config['notifications']['telegram_enabled'] = False

    print(f"📋 Configuration:")
    print(f"   Trading Mode: {config['trading_mode']}")
    print(f"   Initial Balance: ${config['cash']:,.2f}")
    print(f"   Max Position Size: ${config['risk_management']['max_position_size']:,.2f}")
    print(f"   Max Daily Loss: ${config['risk_management']['max_daily_loss']:,.2f}")
    print()

    # Create broker
    try:
        broker = get_broker(config)
        print(f"✅ Created Binance broker: {broker.get_name()}")
        print(f"   Trading Mode: {broker.get_trading_mode().value}")
        print(f"   Paper Trading: {broker.is_paper_trading()}")
        print(f"   API URL: {broker.client.API_URL}")
        print()
    except Exception as e:
        print(f"❌ Failed to create broker: {e}")
        return

    # Connect to broker
    try:
        print("🔌 Connecting to Binance...")
        connected = await broker.connect()
        if connected:
            print("✅ Connected successfully")

            # Get account info
            account_info = await broker.get_account_info()
            print(f"📊 Account Info:")
            for key, value in account_info.items():
                print(f"   {key}: {value}")
            print()
        else:
            print("❌ Connection failed")
            return
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return

    # Get current positions and balances
    print("💼 Current account status:")
    try:
        positions = await broker.get_positions()
        portfolio = await broker.get_portfolio()

        print(f"   Positions: {len(positions)}")
        for symbol, position in positions.items():
            print(f"     {symbol}: {position.quantity}")

        print(f"   Portfolio Value: ${portfolio.total_value:,.2f}")
        print()
    except Exception as e:
        print(f"❌ Error getting account status: {e}")

    # Get Binance-specific info
    try:
        binance_info = await broker.get_binance_specific_info()
        print(f"🔧 Binance-specific info:")
        print(f"   Supported Order Types: {len(binance_info['supported_order_types'])}")
        print(f"   Supported Symbols: {binance_info['supported_symbols_count']}")
        print(f"   Exchange Info Loaded: {binance_info['exchange_info_loaded']}")
        print()
    except Exception as e:
        print(f"❌ Error getting Binance info: {e}")

    print("⚠️  Live trading demo completed (no orders placed for safety)")
    print("   To place actual orders, modify the demo script")

    # Disconnect
    try:
        await broker.disconnect()
        print("🔌 Disconnected from Binance")
    except Exception as e:
        print(f"❌ Disconnect error: {e}")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Enhanced Binance Broker Demo')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (default: paper)')
    parser.add_argument('--confirm', action='store_true',
                       help='Confirm live trading (required for live mode)')

    args = parser.parse_args()

    print("🚀 Enhanced Binance Broker Demo")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.mode == 'live':
        if not args.confirm:
            print("❌ Live trading requires --confirm flag")
            print("   Use: python demo.py --mode live --confirm")
            return

        # Check if live credentials are available
        if not BINANCE_KEY or not BINANCE_SECRET:
            print("❌ Live trading credentials not configured")
            print("   Please set BINANCE_KEY and BINANCE_SECRET in environment")
            return

        await demo_live_trading()
    else:
        # Check if paper credentials are available
        if not BINANCE_PAPER_KEY or not BINANCE_PAPER_SECRET:
            print("❌ Paper trading credentials not configured")
            print("   Please set BINANCE_PAPER_KEY and BINANCE_PAPER_SECRET in environment")
            return

        await demo_paper_trading()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()