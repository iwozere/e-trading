#!/usr/bin/env python3
"""
Unit tests for strategy_core module
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from src.strategy.future.strategy_core import (
    BaseStrategy,
    StrategySignal,
    SignalAggregator,
    AggregationMethod,
    MarketRegimeDetector,
    MarketRegime
)


class TestBaseStrategy(unittest.TestCase):
    """Test BaseStrategy abstract class."""

    def test_base_strategy_signal_shape(self):
        """Test that BaseStrategy implementations return correct signal shape."""

        class TestStrategy(BaseStrategy):
            def generate_signal(self):
                return StrategySignal(
                    strategy_name="test",
                    signal_type="buy",
                    confidence=0.8,
                    weight=1.0,
                    timestamp=datetime.now(),
                    metadata={}
                )

        strategy = TestStrategy("test_strategy")
        signal = strategy.generate_signal()

        self.assertEqual(signal.strategy_name, "test")
        self.assertEqual(signal.signal_type, "buy")
        self.assertEqual(signal.confidence, 0.8)
        self.assertEqual(signal.weight, 1.0)
        self.assertIsInstance(signal.timestamp, datetime)
        self.assertIsInstance(signal.metadata, dict)

    def test_strategy_data_setting(self):
        """Test setting data for strategies."""

        class TestStrategy(BaseStrategy):
            def generate_signal(self):
                return StrategySignal(
                    strategy_name="test",
                    signal_type="hold",
                    confidence=0.0,
                    weight=1.0,
                    timestamp=datetime.now(),
                    metadata={}
                )

        strategy = TestStrategy("test_strategy")
        test_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        strategy.set_data(test_data)
        self.assertIsNotNone(strategy.data)
        self.assertEqual(len(strategy.data), 3)

    def test_indicator_management(self):
        """Test adding and retrieving indicators."""

        class TestStrategy(BaseStrategy):
            def generate_signal(self):
                return StrategySignal(
                    strategy_name="test",
                    signal_type="hold",
                    confidence=0.0,
                    weight=1.0,
                    timestamp=datetime.now(),
                    metadata={}
                )

        strategy = TestStrategy("test_strategy")
        mock_indicator = Mock()

        strategy.add_indicator("rsi", mock_indicator)
        retrieved = strategy.get_indicator("rsi")

        self.assertEqual(retrieved, mock_indicator)
        self.assertIsNone(strategy.get_indicator("nonexistent"))


class TestSignalAggregator(unittest.TestCase):
    """Test SignalAggregator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.aggregator = SignalAggregator(AggregationMethod.WEIGHTED_VOTING)

        self.buy_signal = StrategySignal(
            strategy_name="strategy1",
            signal_type="buy",
            confidence=0.8,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={}
        )

        self.sell_signal = StrategySignal(
            strategy_name="strategy2",
            signal_type="sell",
            confidence=0.6,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={}
        )

        self.hold_signal = StrategySignal(
            strategy_name="strategy3",
            signal_type="hold",
            confidence=0.0,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={}
        )

    def test_empty_signals(self):
        """Test aggregation with no signals."""
        result = self.aggregator.aggregate_signals([])

        self.assertEqual(result.signal_type, "hold")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.contributing_strategies), 0)

    def test_weighted_voting(self):
        """Test weighted voting aggregation."""
        signals = [self.buy_signal, self.sell_signal]
        result = self.aggregator.aggregate_signals(signals)

        self.assertEqual(result.signal_type, "buy")  # Higher confidence
        self.assertGreater(result.confidence, 0.0)
        self.assertEqual(len(result.contributing_strategies), 2)

    def test_consensus_voting(self):
        """Test consensus voting aggregation."""
        consensus_aggregator = SignalAggregator(AggregationMethod.CONSENSUS, consensus_threshold=0.6)

        # Two buy signals should create consensus
        buy_signal2 = StrategySignal(
            strategy_name="strategy4",
            signal_type="buy",
            confidence=0.7,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={}
        )

        signals = [self.buy_signal, buy_signal2]
        result = consensus_aggregator.aggregate_signals(signals)

        self.assertEqual(result.signal_type, "buy")
        self.assertGreater(result.confidence, 0.6)

    def test_majority_voting(self):
        """Test majority voting aggregation."""
        majority_aggregator = SignalAggregator(AggregationMethod.MAJORITY)

        # Two buy, one sell should result in buy
        buy_signal2 = StrategySignal(
            strategy_name="strategy4",
            signal_type="buy",
            confidence=0.5,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={}
        )

        signals = [self.buy_signal, buy_signal2, self.sell_signal]
        result = majority_aggregator.aggregate_signals(signals)

        self.assertEqual(result.signal_type, "buy")
        self.assertAlmostEqual(result.confidence, 2/3, places=2)

    def test_weighted_average(self):
        """Test weighted average aggregation."""
        avg_aggregator = SignalAggregator(AggregationMethod.WEIGHTED_AVERAGE)

        signals = [self.buy_signal, self.sell_signal]
        result = avg_aggregator.aggregate_signals(signals)

        # Should be buy (higher weight due to higher confidence)
        self.assertEqual(result.signal_type, "buy")
        self.assertGreater(result.confidence, 0.0)

    def test_invalid_signal_aggregator_method(self):
        """Test that an invalid aggregation method raises ValueError."""
        from src.strategy.future.strategy_core import SignalAggregator, AggregationMethod
        aggregator = SignalAggregator(AggregationMethod.WEIGHTED_VOTING)
        aggregator.method = "invalid_method"
        with self.assertRaises(ValueError):
            aggregator.aggregate_signals([self.buy_signal, self.sell_signal])

    def test_strategy_signal_metadata(self):
        """Test that metadata is correctly passed in signals."""
        class TestStrategy(BaseStrategy):
            def generate_signal(self):
                return StrategySignal(
                    strategy_name="test",
                    signal_type="buy",
                    confidence=0.8,
                    weight=1.0,
                    timestamp=datetime.now(),
                    metadata={"foo": "bar"}
                )
        strategy = TestStrategy("test_strategy")
        signal = strategy.generate_signal()
        self.assertIn("foo", signal.metadata)
        self.assertEqual(signal.metadata["foo"], "bar")


class TestMarketRegimeDetector(unittest.TestCase):
    """Test MarketRegimeDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()

    def test_insufficient_data(self):
        """Test regime detection with insufficient data."""
        data = pd.DataFrame({
            'close': [100, 101, 102]  # Less than lookback_period
        })

        regime = self.detector.detect_regime(data)
        self.assertEqual(regime, MarketRegime.RANGING_STABLE)

    def test_trending_volatile_regime(self):
        """Test detection of trending volatile regime."""
        # Create trending data with high volatility
        np.random.seed(42)
        trend = np.linspace(100, 120, 30)
        noise = np.random.normal(0, 2, 30)  # High volatility
        prices = trend + noise

        data = pd.DataFrame({
            'close': prices
        })

        regime = self.detector.detect_regime(data)
        self.assertEqual(regime, MarketRegime.TRENDING_VOLATILE)

    def test_trending_stable_regime(self):
        """Test detection of trending stable regime."""
        # Create trending data with low volatility
        np.random.seed(42)
        trend = np.linspace(100, 120, 30)
        noise = np.random.normal(0, 0.5, 30)  # Low volatility
        prices = trend + noise

        data = pd.DataFrame({
            'close': prices
        })

        regime = self.detector.detect_regime(data)
        self.assertEqual(regime, MarketRegime.TRENDING_STABLE)

    def test_ranging_volatile_regime(self):
        """Test detection of ranging volatile regime."""
        # Create ranging data with high volatility
        np.random.seed(42)
        base = np.ones(30) * 110
        noise = np.random.normal(0, 2, 30)  # High volatility
        prices = base + noise

        data = pd.DataFrame({
            'close': prices
        })

        regime = self.detector.detect_regime(data)
        self.assertEqual(regime, MarketRegime.RANGING_VOLATILE)


if __name__ == "__main__":
    unittest.main()
