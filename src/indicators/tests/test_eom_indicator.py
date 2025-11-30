"""
Unit tests for EOM (Ease of Movement) Indicator

Tests the calculation, edge cases, and parameter validation of the EOM indicator.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import backtrader as bt
import numpy as np

from src.indicators.eom_indicator import EOMIndicator, EOM
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestEOMIndicator(unittest.TestCase):
    """Test cases for EOM Indicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.cerebro = bt.Cerebro()

    def test_eom_initialization_default_params(self):
        """Test EOM indicator initialization with default parameters"""

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom = EOMIndicator(self.data, timeperiod=14, scale=100000000.0)

        self.cerebro.addstrategy(TestStrategy)

        # Create test data
        data = bt.feeds.PandasData(dataname=self._create_test_data())
        self.cerebro.adddata(data)

        try:
            self.cerebro.run()
            _logger.info("EOM indicator initialized successfully with default params")
        except Exception as e:
            self.fail(f"EOM initialization failed: {e}")

    def test_eom_alias(self):
        """Test that EOM alias works correctly"""

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom = EOM(self.data)

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_test_data())
        self.cerebro.adddata(data)

        try:
            self.cerebro.run()
            _logger.info("EOM alias works correctly")
        except Exception as e:
            self.fail(f"EOM alias initialization failed: {e}")

    def test_eom_calculation_logic(self):
        """Test EOM calculation produces expected values"""

        results = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom = EOMIndicator(self.data, timeperiod=3, scale=1000000.0)

            def next(self):
                if len(self.data) >= 20:
                    results.append({
                        'eom': self.eom[0],
                        'high': self.data.high[0],
                        'low': self.data.low[0],
                        'volume': self.data.volume[0]
                    })

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_test_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # Verify we got results
        self.assertGreater(len(results), 0, "Should have EOM values")

        # Check that EOM values are numeric
        for result in results:
            self.assertIsInstance(result['eom'], (int, float), "EOM should be numeric")
            self.assertFalse(np.isnan(result['eom']), "EOM should not be NaN after warmup")

    def test_eom_bullish_movement(self):
        """Test EOM is positive during bullish price movement"""

        results = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom = EOMIndicator(self.data, timeperiod=3, scale=1000000.0)

            def next(self):
                if len(self.data) >= 20:
                    results.append(self.eom[0])

        self.cerebro.addstrategy(TestStrategy)

        # Create strongly bullish data
        data = bt.feeds.PandasData(dataname=self._create_bullish_test_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # In bullish trend, EOM should tend to be positive
        positive_count = sum(1 for eom in results if eom > 0)
        total_count = len(results)

        _logger.info(f"Bullish test: {positive_count}/{total_count} positive EOM values")
        self.assertGreater(positive_count / total_count, 0.5,
                          "EOM should be mostly positive during bullish movement")

    def test_eom_bearish_movement(self):
        """Test EOM is negative during bearish price movement"""

        results = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom = EOMIndicator(self.data, timeperiod=3, scale=1000000.0)

            def next(self):
                if len(self.data) >= 20:
                    results.append(self.eom[0])

        self.cerebro.addstrategy(TestStrategy)

        # Create strongly bearish data
        data = bt.feeds.PandasData(dataname=self._create_bearish_test_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # In bearish trend, EOM should tend to be negative
        negative_count = sum(1 for eom in results if eom < 0)
        total_count = len(results)

        _logger.info(f"Bearish test: {negative_count}/{total_count} negative EOM values")
        self.assertGreater(negative_count / total_count, 0.5,
                          "EOM should be mostly negative during bearish movement")

    def test_eom_zero_volume_handling(self):
        """Test EOM handles zero volume gracefully"""

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom = EOMIndicator(self.data, timeperiod=3, scale=1000000.0)

            def next(self):
                # Should not crash even with zero volume
                _ = self.eom[0]

        self.cerebro.addstrategy(TestStrategy)

        # Create data with zero volume
        data = bt.feeds.PandasData(dataname=self._create_zero_volume_data())
        self.cerebro.adddata(data)

        try:
            self.cerebro.run()
            _logger.info("EOM handles zero volume without crashing")
        except Exception as e:
            self.fail(f"EOM should handle zero volume gracefully: {e}")

    def test_eom_equal_high_low_handling(self):
        """Test EOM handles equal high/low (zero range) gracefully"""

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom = EOMIndicator(self.data, timeperiod=3, scale=1000000.0)

            def next(self):
                # Should not crash even with zero range
                _ = self.eom[0]

        self.cerebro.addstrategy(TestStrategy)

        # Create data with equal high/low
        data = bt.feeds.PandasData(dataname=self._create_flat_price_data())
        self.cerebro.adddata(data)

        try:
            self.cerebro.run()
            _logger.info("EOM handles zero range without crashing")
        except Exception as e:
            self.fail(f"EOM should handle zero range gracefully: {e}")

    def test_eom_custom_scale_parameter(self):
        """Test EOM with custom scale parameter"""

        results_default = []
        results_custom = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.eom_default = EOMIndicator(self.data, timeperiod=3, scale=100000000.0)
                self.eom_custom = EOMIndicator(self.data, timeperiod=3, scale=1000000.0)

            def next(self):
                if len(self.data) >= 20:
                    results_default.append(self.eom_default[0])
                    results_custom.append(self.eom_custom[0])

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_test_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # Different scales should produce different values
        self.assertEqual(len(results_default), len(results_custom))

        # At least some values should be different
        different_count = sum(1 for d, c in zip(results_default, results_custom) if abs(d - c) > 0.0001)
        self.assertGreater(different_count, 0, "Different scales should produce different values")

    # Helper methods to create test data

    def _create_test_data(self):
        """Create standard test data"""
        import pandas as pd
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        # Create realistic OHLCV data
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 2)
        high = close + np.random.rand(100) * 3
        low = close - np.random.rand(100) * 3
        open_price = close + np.random.randn(100)
        volume = np.random.randint(1000000, 10000000, 100)

        df = pd.DataFrame({
            'datetime': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.set_index('datetime', inplace=True)

        return df

    def _create_bullish_test_data(self):
        """Create bullish trending data"""
        import pandas as pd
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        # Strong uptrend
        close = 100 + np.arange(100) * 0.5 + np.random.randn(100) * 0.5
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 1
        open_price = close - np.random.rand(100) * 0.5
        volume = np.random.randint(5000000, 15000000, 100)

        df = pd.DataFrame({
            'datetime': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.set_index('datetime', inplace=True)

        return df

    def _create_bearish_test_data(self):
        """Create bearish trending data"""
        import pandas as pd
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        # Strong downtrend
        close = 100 - np.arange(100) * 0.5 + np.random.randn(100) * 0.5
        high = close + np.random.rand(100) * 1
        low = close - np.random.rand(100) * 2
        open_price = close + np.random.rand(100) * 0.5
        volume = np.random.randint(5000000, 15000000, 100)

        df = pd.DataFrame({
            'datetime': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.set_index('datetime', inplace=True)

        return df

    def _create_zero_volume_data(self):
        """Create data with zero volume"""
        import pandas as pd
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        close = 100 + np.random.randn(100)
        high = close + 2
        low = close - 2
        open_price = close + np.random.randn(100)
        volume = np.zeros(100)  # Zero volume

        df = pd.DataFrame({
            'datetime': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.set_index('datetime', inplace=True)

        return df

    def _create_flat_price_data(self):
        """Create data with flat prices (high == low)"""
        import pandas as pd
        from datetime import datetime, timedelta

        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        close = np.full(100, 100.0)  # Constant price
        high = close.copy()
        low = close.copy()
        open_price = close.copy()
        volume = np.random.randint(1000000, 10000000, 100)

        df = pd.DataFrame({
            'datetime': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.set_index('datetime', inplace=True)

        return df


if __name__ == '__main__':
    unittest.main()
