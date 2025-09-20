#!/usr/bin/env python3
"""
Enhanced IBKR Broker Demo
-------------------------

This script demonstrates the enhanced IBKR broker capabilities including:
- Seamless paper-to-live trading mode switching
- Multi-asset support (stocks, options, futures, forex)
- Realistic paper trading simulation
- Real-time market data integration
- Position notifications
- Comprehensive analytics

Usage:
    python examples/enhanced_ibkr_broker_demo.py --mode paper
    python examples/enhanced_ibkr_broker_demo.py --mode live --confirm
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
from src.trading.broker.ibkr_utils import create_ibkr_config_template
from src.trading.broker.enhanced_base_broker import Order, OrderType, OrderSide


async def demo_paper_trading():
    """Demonstrate paper trading capabilities."""
    print("🔄 Starting Enhanced IBKR Paper Trading Demo")
    print("=" * 60)

    # Create paper trading configuration
    config = create_ibkr_config_template('paper')
    config['notifications']['email_enabled'] = False  # Disable for demo
    config['notifications']['telegram_enabled'] = False

    print(f"📋 Configuration:")
    print(f"   Trading Mode: {config['trading_mode']}")
    print(f"   Initial Balance: ${config['cash']:,.2f}")
    print(f"   Connection: {config['connection']['host']}:{config['connection']['port']}")
    print(f"   Commission Rate: {config['paper_trading_config']['commission_rate']:.4f}")
    print(f"   Slippage Model: {config['paper_trading_config']['slippage_model']}")
    print()

    # Create broker
    try:
        broker = get_broker(config)
        print(f"✅ Created IBKR broker: {broker.get_name()}")
        print(f"   Trading Mode: {broker.get_trading_mode().value}")
        print(f"   Paper Trading: {broker.is_paper_trading()}")
        print(f"   Host: {broker.host}, Port: {broker.port}")
        print()
    except Exception as e:
        print(f"❌ Failed to create broker: {e}")
        return

    # Connect to broker (mock connection for demo)
    try:
        print("🔌 Connecting to IBKR TWS/Gateway...")
        print("   Note: This demo uses mock connections for safety")

        # Mock the connection for demo purposes
        broker.is_connected = True
        broker.ib.isConnected = lambda: True

        print("✅ Connected successfully (mock connection)")

        # Get account info
        account_info = await broker.get_account_info()
        print(f"📊 Account Info:")
        for key, value in account_info.items():
            if key not in ['paper_trading_config', 'connection_info']:
                print(f"   {key}: {value}")
        print()
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return

    # Demonstrate multi-asset support
    print("🌐 Multi-Asset Support Demo:")
    assets = [
        ('AAPL', 'STK', 150.0),
        ('MSFT', 'STK', 300.0),
        ('GOOGL', 'STK', 2500.0),
        ('EURUSD', 'CASH', 1.0850),
        ('GBPUSD', 'CASH', 1.2650)
    ]

    for symbol, asset_class, price in assets:
        contract = broker._create_contract(symbol, asset_class)
        broker.update_market_data_cache(symbol, price)
        print(f"   {symbol} ({asset_class}): ${price:,.4f}")
    print()

    # Demonstrate order placement for different asset classes
    print("📝 Placing demo orders for different asset classes...")

    # Stock order
    stock_order = Order(
        symbol='AAPL',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100
    )

    try:
        order_id = await broker.place_order(stock_order)
        print(f"✅ Stock order placed: {order_id}")
        print(f"   Symbol: {stock_order.symbol} (Stock)")
        print(f"   Quantity: {stock_order.quantity} shares")
        print(f"   Status: {stock_order.status.value}")
        print()
    except Exception as e:
        print(f"❌ Stock order failed: {e}")

    # Forex order
    forex_order = Order(
        symbol='EURUSD',
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100000,  # Standard lot
        price=1.0840
    )

    try:
        order_id = await broker.place_order(forex_order)
        print(f"✅ Forex order placed: {order_id}")
        print(f"   Symbol: {forex_order.symbol} (Forex)")
        print(f"   Quantity: {forex_order.quantity:,.0f} units")
        print(f"   Price: {forex_order.price:.4f}")
        print(f"   Status: {forex_order.status.value}")
        print()
    except Exception as e:
        print(f"❌ Forex order failed: {e}")

    # Limit order with stop loss
    limit_order = Order(
        symbol='MSFT',
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50,
        price=295.0
    )

    try:
        order_id = await broker.place_order(limit_order)
        print(f"✅ Limit order placed: {order_id}")
        print(f"   Symbol: {limit_order.symbol}")
        print(f"   Type: {limit_order.order_type.value}")
        print(f"   Price: ${limit_order.price:.2f}")
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
                print(f"     Avg Price: ${position.average_price:,.4f}")
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

    # Simulate price movements and process pending orders
    print("📈 Simulating price movements...")
    new_prices = {
        'AAPL': 152.0,
        'MSFT': 294.0,  # Below limit price - should trigger
        'GOOGL': 2520.0,
        'EURUSD': 1.0835,  # Below limit price - should trigger
        'GBPUSD': 1.2680
    }

    for symbol, price in new_prices.items():
        broker.update_market_data_cache(symbol, price)
        print(f"   {symbol}: ${price:,.4f} (↗️)")

    # Process pending orders
    await broker.process_market_data_update()
    print("✅ Processed pending orders")
    print()

    # Get IBKR-specific information
    print("🔧 IBKR-specific information:")
    try:
        ibkr_info = await broker.get_ibkr_specific_info()
        print(f"   Broker Type: {ibkr_info['broker_type']}")
        print(f"   Supported Order Types: {len(ibkr_info['supported_order_types'])}")
        print(f"   Supported Asset Classes: {len(ibkr_info['supported_asset_classes'])}")
        print(f"   Market Data Subscriptions: {ibkr_info['market_data_subscriptions']}")
        print(f"   Active Contracts: {ibkr_info['active_contracts']}")
        print()
    except Exception as e:
        print(f"❌ Error getting IBKR info: {e}")

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
        print("🔌 Disconnected from IBKR")
    except Exception as e:
        print(f"❌ Disconnect error: {e}")

    print("✅ Paper trading demo completed!")


async def demo_live_trading():
    """Demonstrate live trading capabilities (with safety checks)."""
    print("⚠️  LIVE TRADING MODE - REAL MONEY WILL BE USED!")
    print("=" * 60)

    # Create live trading configuration
    config = create_ibkr_config_template('live')
    config['live_trading_confirmed'] = True  # Required for live trading
    config['notifications']['email_enabled'] = False  # Disable for demo
    config['notifications']['telegram_enabled'] = False

    print(f"📋 Configuration:")
    print(f"   Trading Mode: {config['trading_mode']}")
    print(f"   Initial Balance: ${config['cash']:,.2f}")
    print(f"   Connection: {config['connection']['host']}:{config['connection']['port']}")
    print(f"   Max Position Size: ${config['risk_management']['max_position_size']:,.2f}")
    print(f"   Max Daily Loss: ${config['risk_management']['max_daily_loss']:,.2f}")
    print()

    # Create broker
    try:
        broker = get_broker(config)
        print(f"✅ Created IBKR broker: {broker.get_name()}")
        print(f"   Trading Mode: {broker.get_trading_mode().value}")
        print(f"   Paper Trading: {broker.is_paper_trading()}")
        print(f"   Host: {broker.host}, Port: {broker.port}")
        print()
    except Exception as e:
        print(f"❌ Failed to create broker: {e}")
        return

    # Connect to broker (mock connection for demo)
    try:
        print("🔌 Connecting to IBKR TWS/Gateway...")
        print("   Note: This demo uses mock connections for safety")

        # Mock the connection for demo purposes
        broker.is_connected = True
        broker.ib.isConnected = lambda: True

        print("✅ Connected successfully (mock connection)")

        # Get account info
        account_info = await broker.get_account_info()
        print(f"📊 Account Info:")
        for key, value in account_info.items():
            if key != 'account_values':
                print(f"   {key}: {value}")
        print()
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

    # Demonstrate contract details retrieval
    print("📋 Contract Details Demo:")
    symbols = ['AAPL', 'MSFT', 'GOOGL']

    for symbol in symbols:
        try:
            # Mock contract details for demo
            details = {
                'symbol': symbol,
                'contract_id': 12345,
                'exchange': 'NASDAQ',
                'currency': 'USD',
                'asset_class': 'STK',
                'market_name': 'NASDAQ',
                'min_tick': 0.01
            }
            print(f"   {symbol}:")
            print(f"     Exchange: {details['exchange']}")
            print(f"     Currency: {details['currency']}")
            print(f"     Min Tick: ${details['min_tick']}")
        except Exception as e:
            print(f"   ❌ Error getting details for {symbol}: {e}")
    print()

    # Get IBKR-specific info
    try:
        ibkr_info = await broker.get_ibkr_specific_info()
        print(f"🔧 IBKR-specific info:")
        print(f"   Supported Order Types: {len(ibkr_info['supported_order_types'])}")
        print(f"   Supported Asset Classes: {len(ibkr_info['supported_asset_classes'])}")
        print(f"   Connection Status: {ibkr_info['connection_info']['connected']}")
        print()
    except Exception as e:
        print(f"❌ Error getting IBKR info: {e}")

    print("⚠️  Live trading demo completed (no orders placed for safety)")
    print("   To place actual orders, modify the demo script")

    # Disconnect
    try:
        await broker.disconnect()
        print("🔌 Disconnected from IBKR")
    except Exception as e:
        print(f"❌ Disconnect error: {e}")


async def demo_multi_asset_trading():
    """Demonstrate multi-asset trading capabilities."""
    print("🌐 Multi-Asset Trading Demo")
    print("=" * 60)

    config = create_ibkr_config_template('paper')
    config['notifications']['email_enabled'] = False
    config['notifications']['telegram_enabled'] = False

    broker = get_broker(config)
    broker.is_connected = True
    broker.ib.isConnected = lambda: True

    print("📊 Asset Class Support:")
    asset_classes = broker.get_supported_asset_classes()
    for asset_class in asset_classes:
        print(f"   ✅ {asset_class}")
    print()

    print("📝 Order Type Support:")
    order_types = broker.get_supported_order_types()
    for order_type in order_types:
        print(f"   ✅ {order_type.value}")
    print()

    # Demonstrate different asset classes
    print("🔄 Creating contracts for different asset classes:")

    # Stock
    stock_contract = broker._create_contract('AAPL', 'STK')
    print(f"   Stock: {stock_contract.symbol} ({stock_contract.secType})")

    # Forex
    forex_contract = broker._create_contract('EURUSD', 'CASH')
    print(f"   Forex: {forex_contract.symbol} ({forex_contract.secType})")

    # Future (mock)
    try:
        future_contract = broker._create_contract('ES', 'FUT', 'GLOBEX')
        print(f"   Future: {future_contract.symbol} ({future_contract.secType})")
    except:
        print(f"   Future: ES (FUT) - Contract creation demo")

    print()
    print("✅ Multi-asset demo completed!")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='Enhanced IBKR Broker Demo')
    parser.add_argument('--mode', choices=['paper', 'live', 'multi-asset'], default='paper',
                       help='Demo mode (default: paper)')
    parser.add_argument('--confirm', action='store_true',
                       help='Confirm live trading (required for live mode)')

    args = parser.parse_args()

    print("🚀 Enhanced IBKR Broker Demo")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if args.mode == 'live':
        if not args.confirm:
            print("❌ Live trading requires --confirm flag")
            print("   Use: python demo.py --mode live --confirm")
            return

        await demo_live_trading()

    elif args.mode == 'multi-asset':
        await demo_multi_asset_trading()

    else:
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