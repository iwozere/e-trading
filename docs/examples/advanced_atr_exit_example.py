"""
Advanced ATR Exit Strategy Example

This example demonstrates how to use the AdvancedATRExitMixin with a custom strategy.
It shows the configuration, initialization, and usage of the sophisticated exit strategy.
"""

import backtrader as bt
import pandas as pd

# Import the exit mixin and factory
from src.strategy.exit.exit_mixin_factory import get_exit_mixin, get_exit_mixin_from_config
from src.strategy.entry.rsi_bb_entry_mixin import RSIBBEntryMixin
from src.strategy.base_strategy import BaseStrategy


class AdvancedATRStrategy(BaseStrategy):
    """
    Example strategy using the Advanced ATR Exit Mixin.

    This strategy combines:
    - RSI + Bollinger Bands entry logic
    - Advanced ATR trailing stop exit
    """

    def __init__(self):
        super().__init__()

        # Initialize entry mixin
        self.entry_mixin = RSIBBEntryMixin({
            'rsi_period': 14,
            'bb_period': 20,
            'bb_dev_factor': 2.0,
            'e_rsi_oversold': 30,
            'e_use_bb_touch': True
        })

        # Initialize exit mixin with advanced configuration
        self.exit_mixin = get_exit_mixin("AdvancedATRExitMixin", {
            # Core parameters
            'anchor': 'high',
            'k_init': 2.5,
            'k_run': 2.0,
            'k_phase2': 1.5,

            # ATR parameters
            'p_fast': 7,
            'p_slow': 21,
            'use_htf_atr': True,
            'alpha_fast': 1.0,
            'alpha_slow': 1.0,
            'alpha_htf': 1.0,

            # Break-even and phases
            'arm_at_R': 1.0,
            'breakeven_offset_atr': 0.0,
            'phase2_at_R': 2.0,

            # Structural ratchet
            'use_swing_ratchet': True,
            'swing_lookback': 10,
            'struct_buffer_atr': 0.25,

            # Time-based tightening
            'tighten_if_stagnant_bars': 20,
            'tighten_k_factor': 0.8,
            'min_bars_between_tighten': 5,

            # Filters
            'min_stop_step': 0.0,
            'noise_filter_atr': 0.0,
            'max_trail_freq': 1,
            'tick_size': 0.01,

            # Partial take-profit
            'pt_levels_R': [1.0, 2.0],
            'pt_sizes': [0.33, 0.33],
            'retune_after_pt': True
        })

        # Initialize mixins with strategy reference
        self.entry_mixin.init_entry(self)
        self.exit_mixin.init_exit(self)

    def next(self):
        """Main strategy logic."""
        super().next()

        # Check if we should enter a position
        if self.position.size == 0 and self.entry_mixin.should_enter():
            # Calculate position size
            confidence = 1.0  # Could be based on signal strength
            self._enter_position('long', confidence=confidence, reason="RSI+BB signal")

        # Check if we should exit
        elif self.position.size != 0:
            should_exit, exit_reason = self.exit_mixin.should_exit()
            if should_exit:
                self.close()
                print(f"Position closed: {exit_reason}")

    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Completed]:
            if order.isbuy():
                # Position opened - notify exit mixin
                self.exit_mixin.on_entry(
                    entry_price=order.executed.price,
                    entry_time=order.executed.dt,
                    position_size=order.executed.size,
                    direction='long'
                )
                print(f"Position opened at {order.executed.price}")

            elif order.issell():
                # Position closed - log exit details
                exit_log = self.exit_mixin.get_exit_log()
                print(f"Exit strategy log: {len(exit_log)} events recorded")


def create_sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')

    # Generate sample OHLCV data
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    return data


def run_example():
    """Run the advanced ATR exit strategy example."""
    print("🚀 Advanced ATR Exit Strategy Example")
    print("=" * 50)

    # Create sample data
    data = create_sample_data()
    print(f"📊 Created sample data: {len(data)} bars")

    # Create Backtrader data feed
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # Use index
        open=0, high=1, low=2, close=3, volume=4,
        openinterest=-1
    )

    # Create and configure Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(AdvancedATRStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Run the strategy
    print("🏃 Running strategy...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()

    # Print results
    print("\n📈 Results:")
    print(f"Initial Portfolio Value: ${initial_value:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {((final_value / initial_value) - 1) * 100:.2f}%")

    # Get strategy instance and show exit mixin details
    strategy = results[0]
    exit_mixin = strategy.exit_mixin

    print("\n🎯 Exit Strategy Details:")
    print(f"Current State: {exit_mixin.get_state()}")
    print(f"Current Stop: ${exit_mixin.get_current_stop():.2f}")
    print(f"Exit Events Logged: {len(exit_mixin.get_exit_log())}")

    # Show some exit log events
    exit_log = exit_mixin.get_exit_log()
    if exit_log:
        print("\n📝 Recent Exit Events:")
        for event in exit_log[-3:]:  # Show last 3 events
            print(f"  - {event['event_type']}: {event.get('details', 'N/A')}")


def demonstrate_configuration_approaches():
    """Demonstrate different ways to configure the exit mixin."""
    print("\n🔧 Configuration Approaches:")
    print("=" * 40)

    # Approach 1: Direct parameter passing
    print("1. Direct Parameter Passing:")
    exit_mixin1 = get_exit_mixin("AdvancedATRExitMixin", {
        'k_init': 3.0,
        'k_run': 2.5,
        'use_swing_ratchet': True
    })
    print(f"   Created mixin with k_init={exit_mixin1.k_init}")

    # Approach 2: Configuration dictionary
    print("\n2. Configuration Dictionary:")
    config = {
        "name": "AdvancedATRExitMixin",
        "params": {
            "anchor": "high",
            "k_init": 2.5,
            "k_run": 2.0,
            "p_fast": 7,
            "p_slow": 21,
            "use_htf_atr": True,
            "arm_at_R": 1.0,
            "phase2_at_R": 2.0,
            "use_swing_ratchet": True,
            "swing_lookback": 10,
            "pt_levels_R": [1.0, 2.0],
            "pt_sizes": [0.33, 0.33]
        }
    }
    exit_mixin2 = get_exit_mixin_from_config(config)
    print(f"   Created mixin with anchor={exit_mixin2.anchor}")

    # Approach 3: Default parameters
    print("\n3. Default Parameters:")
    exit_mixin3 = get_exit_mixin("AdvancedATRExitMixin")
    print(f"   Created mixin with default k_init={exit_mixin3.k_init}")


if __name__ == "__main__":
    # Run the main example
    run_example()

    # Demonstrate configuration approaches
    demonstrate_configuration_approaches()

    print("\n✅ Example completed successfully!")
    print("\n📚 For more information, see:")
    print("   - docs/ADVANCED_ATR_EXIT.md")
    print("   - config/optimizer/exit/AdvancedATRExitMixin.json")
