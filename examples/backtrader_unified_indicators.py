"""
Example: Using Backtrader Indicators with Unified Service

This example demonstrates how to use the simplified Backtrader indicators
that leverage the unified indicator service without backward compatibility.
"""

import backtrader as bt
import pandas as pd
import numpy as np

# Import the unified indicators directly from backtrader wrappers
from src.indicators.adapters.backtrader_wrappers import (
    UnifiedRSIIndicator as RsiIndicator,
    UnifiedBollingerBandsIndicator as BollingerBandIndicator,
    UnifiedMACDIndicator as MacdIndicator
)

# Import the factory for programmatic creation
from src.indicators.indicator_factory import IndicatorFactory


class UnifiedIndicatorStrategy(bt.Strategy):
    """
    Example strategy using unified indicators
    """

    def __init__(self):
        # Create indicators using direct imports
        self.rsi = RsiIndicator(self.data, period=14, backend="bt")
        self.bb = BollingerBandIndicator(self.data, period=20, devfactor=2.0, backend="bt")
        self.macd = MacdIndicator(self.data, fast_period=12, slow_period=26, signal_period=9, backend="bt")

        # Or create using factory
        factory = IndicatorFactory()
        self.rsi_factory = factory.create_backtrader_rsi(self.data, period=21, backend="bt")

    def next(self):
        # Use indicators as normal Backtrader indicators
        current_rsi = self.rsi.rsi[0]
        upper_bb = self.bb.upper[0]
        lower_bb = self.bb.lower[0]
        macd_line = self.macd.macd[0]

        # Example trading logic
        if current_rsi < 30 and self.data.close[0] < lower_bb:
            self.buy()
        elif current_rsi > 70 and self.data.close[0] > upper_bb:
            self.sell()


def create_sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Generate realistic price data
    price = 100
    prices = []

    for _ in dates:
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price *= (1 + change)
        prices.append(price)

    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000, 10000)

        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return pd.DataFrame(data)


def main():
    """
    Main function demonstrating unified indicator usage
    """
    print("Backtrader Unified Indicators Example")
    print("=====================================")

    # Create sample data
    df = create_sample_data()
    print(f"Created sample data with {len(df)} rows")

    # Create Backtrader cerebro
    cerebro = bt.Cerebro()

    # Add data feed
    data_feed = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume'
    )
    cerebro.adddata(data_feed)

    # Add strategy
    cerebro.addstrategy(UnifiedIndicatorStrategy)

    # Set initial cash
    cerebro.broker.setcash(10000)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run backtest
    try:
        results = cerebro.run()
        print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
        print("Backtest completed successfully!")

        # Print some indicator values from the last strategy instance
        strategy = results[0]
        print("\nFinal indicator values:")
        print(f"RSI: {strategy.rsi.rsi[0]:.2f}")
        print(f"Bollinger Upper: {strategy.bb.upper[0]:.2f}")
        print(f"Bollinger Lower: {strategy.bb.lower[0]:.2f}")
        print(f"MACD: {strategy.macd.macd[0]:.4f}")

    except Exception as e:
        print(f"Error during backtest: {e}")
        print("Note: This example requires the unified indicator service to be properly configured.")


if __name__ == "__main__":
    main()