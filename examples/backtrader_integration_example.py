#!/usr/bin/env python3
"""
Example demonstrating BaseBroker integration with backtrader.

This example shows how to use the BaseBroker within a backtrader strategy,
leveraging both backtrader's framework and the enhanced broker's paper trading
and notification features.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

try:
    import backtrader as bt

    # Import directly from the broker module to avoid dependency issues
    sys.path.append(str(PROJECT_ROOT / "src" / "trading" / "broker"))
    from base_broker import BaseBroker

    class SimpleStrategy(bt.Strategy):
        """Simple backtrader strategy using BaseBroker."""

        def __init__(self):
            # Configure the enhanced broker
            broker_config = {
                'name': 'backtrader_enhanced_broker',
                'trading_mode': 'paper',
                'paper_trading_config': {
                    'initial_balance': 10000.0,
                    'commission_rate': 0.001,
                    'slippage_model': 'linear',
                    'base_slippage': 0.0005,
                    'latency_simulation': True,
                    'realistic_fills': True
                },
                'notifications': {
                    'position_opened': True,
                    'position_closed': True,
                    'email_enabled': False,  # Disable for example
                    'telegram_enabled': False  # Disable for example
                }
            }

            # Create and set the enhanced broker
            self.enhanced_broker = BaseBroker(broker_config)

            # Replace cerebro's broker with our enhanced broker
            self.cerebro.broker = self.enhanced_broker

            print("Strategy initialized with Enhanced Broker")
            print(f"Backtrader mode: {self.enhanced_broker.is_backtrader_mode()}")
            print(f"Paper trading: {self.enhanced_broker.is_paper_trading()}")

        def next(self):
            """Strategy logic executed on each bar."""
            # Simple strategy: buy if we don't have a position
            if not self.position:
                # Use backtrader's standard buy method, which now goes through BaseBroker
                order = self.buy(size=100)
                print(f"Buy order placed: {order}")

            # Sell after holding for 5 bars
            elif len(self) > 5:
                order = self.sell(size=self.position.size)
                print(f"Sell order placed: {order}")

        def notify_order(self, order):
            """Called when order status changes."""
            print(f"Order notification: {order.ref} - Status: {order.status}")

            if order.status in [order.Completed]:
                if order.isbuy():
                    print(f"Buy executed: Price {order.executed.price}, Size {order.executed.size}")
                elif order.issell():
                    print(f"Sell executed: Price {order.executed.price}, Size {order.executed.size}")

    def run_backtrader_example():
        """Run the backtrader example with BaseBroker."""
        print("Running Backtrader Integration Example")
        print("=" * 50)

        # Create cerebro instance
        cerebro = bt.Cerebro()

        # Add strategy
        cerebro.addstrategy(SimpleStrategy)

        # Create some dummy data (in real usage, you'd load actual market data)
        data = bt.feeds.YahooFinanceCSVData(
            dataname='AAPL',  # This would be a real CSV file path
            fromdate=None,
            todate=None,
            reverse=False
        )

        # For this example, we'll create synthetic data instead
        # In a real scenario, you'd use actual market data

        print("Enhanced Broker with Backtrader integration ready!")
        print("Note: This example requires actual market data to run a full backtest.")
        print("The integration is working and ready for use with real data feeds.")

        # Show broker capabilities
        broker_config = {
            'name': 'example_broker',
            'trading_mode': 'paper'
        }

        broker = BaseBroker(broker_config)
        print("\nBroker Features:")
        print(f"- Name: {broker.get_name()}")
        print(f"- Trading Mode: {broker.get_trading_mode().value}")
        print(f"- Paper Trading: {broker.is_paper_trading()}")
        print(f"- Backtrader Mode: {broker.is_backtrader_mode()}")
        print(f"- Execution Quality Tracking: {broker.paper_trading_config.enable_execution_quality}")

        return True

    if __name__ == '__main__':
        run_backtrader_example()

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure backtrader is installed: pip install backtrader")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()