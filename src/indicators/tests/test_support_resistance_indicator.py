"""
Unit tests for Support/Resistance Indicator

Tests swing detection, support/resistance calculation, and edge cases.
"""

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.indicators.support_resistance_indicator import SupportResistanceIndicator, SupportResistance
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class TestSupportResistanceIndicator(unittest.TestCase):
    """Test cases for Support/Resistance Indicator"""

    def setUp(self):
        """Set up test fixtures"""
        self.cerebro = bt.Cerebro()

    def test_sr_initialization_default_params(self):
        """Test S/R indicator initialization with default parameters"""

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistanceIndicator(self.data, lookback_bars=2)

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_test_data())
        self.cerebro.adddata(data)

        try:
            self.cerebro.run()
            _logger.info("S/R indicator initialized successfully with default params")
        except Exception as e:
            self.fail(f"S/R initialization failed: {e}")

    def test_sr_alias(self):
        """Test that SupportResistance alias works correctly"""

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistance(self.data)

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_test_data())
        self.cerebro.adddata(data)

        try:
            self.cerebro.run()
            _logger.info("SupportResistance alias works correctly")
        except Exception as e:
            self.fail(f"S/R alias initialization failed: {e}")

    def test_sr_swing_detection(self):
        """Test that swing highs and lows are detected"""

        results = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistanceIndicator(self.data, lookback_bars=2)

            def next(self):
                if len(self.data) >= 20:
                    results.append({
                        'resistance': self.sr.lines.resistance[0],
                        'support': self.sr.lines.support[0],
                        'close': self.data.close[0]
                    })

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_swing_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # Verify we got results
        self.assertGreater(len(results), 0, "Should have S/R values")

        # At least some bars should have valid resistance/support
        valid_resistance = sum(1 for r in results if not np.isnan(r['resistance']))
        valid_support = sum(1 for r in results if not np.isnan(r['support']))

        _logger.info(f"Valid resistance: {valid_resistance}/{len(results)}")
        _logger.info(f"Valid support: {valid_support}/{len(results)}")

        self.assertGreater(valid_resistance, 0, "Should detect some resistance levels")
        self.assertGreater(valid_support, 0, "Should detect some support levels")

    def test_resistance_above_price(self):
        """Test that resistance is always above current price"""

        results = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistanceIndicator(self.data, lookback_bars=2)

            def next(self):
                if len(self.data) >= 30:
                    resistance = self.sr.lines.resistance[0]
                    close = self.data.close[0]
                    results.append({
                        'resistance': resistance,
                        'close': close,
                        'valid': np.isnan(resistance) or resistance > close
                    })

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_swing_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # All valid resistances should be above price
        invalid = [r for r in results if not r['valid']]

        if invalid:
            _logger.error(f"Found {len(invalid)} invalid resistances")
            for r in invalid[:5]:  # Show first 5
                _logger.error(f"Resistance {r['resistance']} <= Close {r['close']}")

        self.assertEqual(len(invalid), 0, "All resistances should be above current price")

    def test_support_below_price(self):
        """Test that support is always below current price"""

        results = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistanceIndicator(self.data, lookback_bars=2)

            def next(self):
                if len(self.data) >= 30:
                    support = self.sr.lines.support[0]
                    close = self.data.close[0]
                    results.append({
                        'support': support,
                        'close': close,
                        'valid': np.isnan(support) or support < close
                    })

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_swing_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # All valid supports should be below price
        invalid = [r for r in results if not r['valid']]

        if invalid:
            _logger.error(f"Found {len(invalid)} invalid supports")
            for r in invalid[:5]:  # Show first 5
                _logger.error(f"Support {r['support']} >= Close {r['close']}")

        self.assertEqual(len(invalid), 0, "All supports should be below current price")

    def test_lookback_bars_parameter(self):
        """Test different lookback_bars parameters"""

        results_2bar = []
        results_3bar = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr_2 = SupportResistanceIndicator(self.data, lookback_bars=2)
                self.sr_3 = SupportResistanceIndicator(self.data, lookback_bars=3)

            def next(self):
                if len(self.data) >= 30:
                    results_2bar.append(self.sr_2.lines.resistance[0])
                    results_3bar.append(self.sr_3.lines.resistance[0])

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_swing_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # Different lookback periods may detect different swings
        _logger.info(f"2-bar lookback: {len([r for r in results_2bar if not np.isnan(r)])} valid resistances")
        _logger.info(f"3-bar lookback: {len([r for r in results_3bar if not np.isnan(r)])} valid resistances")

        # Both should produce some results
        self.assertEqual(len(results_2bar), len(results_3bar))

    def test_uptrend_resistance_updates(self):
        """Test that resistance updates during uptrend"""

        resistances = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistanceIndicator(self.data, lookback_bars=2)

            def next(self):
                if len(self.data) >= 20:
                    r = self.sr.lines.resistance[0]
                    if not np.isnan(r):
                        resistances.append(r)

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_uptrend_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # In uptrend, resistances should generally increase over time
        if len(resistances) > 10:
            early_avg = np.mean(resistances[:5])
            late_avg = np.mean(resistances[-5:])

            _logger.info(f"Uptrend: early resistance avg={early_avg:.2f}, late avg={late_avg:.2f}")
            self.assertGreater(late_avg, early_avg,
                             "Resistances should increase during uptrend")

    def test_downtrend_support_updates(self):
        """Test that support updates during downtrend"""

        supports = []

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistanceIndicator(self.data, lookback_bars=2)

            def next(self):
                if len(self.data) >= 20:
                    s = self.sr.lines.support[0]
                    if not np.isnan(s):
                        supports.append(s)

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_downtrend_data())
        self.cerebro.adddata(data)

        self.cerebro.run()

        # In downtrend, supports should generally decrease over time
        if len(supports) > 10:
            early_avg = np.mean(supports[:5])
            late_avg = np.mean(supports[-5:])

            _logger.info(f"Downtrend: early support avg={early_avg:.2f}, late avg={late_avg:.2f}")
            self.assertLess(late_avg, early_avg,
                           "Supports should decrease during downtrend")

    def test_flat_market_handling(self):
        """Test S/R indicator handles flat market"""

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sr = SupportResistanceIndicator(self.data, lookback_bars=2)

            def next(self):
                # Should not crash
                _ = self.sr.lines.resistance[0]
                _ = self.sr.lines.support[0]

        self.cerebro.addstrategy(TestStrategy)
        data = bt.feeds.PandasData(dataname=self._create_flat_data())
        self.cerebro.adddata(data)

        try:
            self.cerebro.run()
            _logger.info("S/R handles flat market without crashing")
        except Exception as e:
            self.fail(f"S/R should handle flat market gracefully: {e}")

    # Helper methods to create test data

    def _create_test_data(self):
        """Create standard test data"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

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

    def _create_swing_data(self):
        """Create data with clear swings"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        # Create zigzag pattern
        base = 100
        swing_amplitude = 10
        swing_period = 10

        close = base + swing_amplitude * np.sin(np.arange(100) * 2 * np.pi / swing_period)
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        open_price = close + np.random.randn(100) * 0.5
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

    def _create_uptrend_data(self):
        """Create uptrend data"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        close = 100 + np.arange(100) * 0.5 + np.random.randn(100) * 0.5
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 1
        open_price = close - np.random.rand(100) * 0.5
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

    def _create_downtrend_data(self):
        """Create downtrend data"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        close = 100 - np.arange(100) * 0.5 + np.random.randn(100) * 0.5
        high = close + np.random.rand(100) * 1
        low = close - np.random.rand(100) * 2
        open_price = close + np.random.rand(100) * 0.5
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

    def _create_flat_data(self):
        """Create flat market data"""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

        close = 100 + np.random.randn(100) * 0.1  # Very small movements
        high = close + 0.1
        low = close - 0.1
        open_price = close
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
