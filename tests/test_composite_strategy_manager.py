#!/usr/bin/env python3
"""
Integration tests for composite_strategy_manager module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock
import tempfile
import json
import os

from src.strategy.future.composite_strategy_manager import (
    StrategyComposer,
    AdvancedStrategyFramework
)
from src.strategy.future.strategy_core import (
    BaseStrategy,
    StrategySignal
)


class TestStrategyComposer(unittest.TestCase):
    """Test StrategyComposer class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock strategies
        self.mock_strategy1 = Mock(spec=BaseStrategy)
        self.mock_strategy1.name = "strategy1"
        self.mock_strategy1.weight = 1.0
        self.mock_strategy1.generate_signal.return_value = StrategySignal(
            strategy_name="strategy1",
            signal_type="buy",
            confidence=0.8,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={}
        )

        self.mock_strategy2 = Mock(spec=BaseStrategy)
        self.mock_strategy2.name = "strategy2"
        self.mock_strategy2.weight = 1.0
        self.mock_strategy2.generate_signal.return_value = StrategySignal(
            strategy_name="strategy2",
            signal_type="sell",
            confidence=0.6,
            weight=1.0,
            timestamp=datetime.now(),
            metadata={}
        )

        self.strategies = [self.mock_strategy1, self.mock_strategy2]

    def test_signal_aggregation(self):
        """Test that composer aggregates signals correctly."""
        composer = StrategyComposer(self.strategies)
        result = composer.aggregate_signals()

        self.assertIsNotNone(result)
        self.assertIn("strategy1", result.contributing_strategies)
        self.assertIn("strategy2", result.contributing_strategies)
        self.assertEqual(len(result.contributing_strategies), 2)

    def test_strategy_signals(self):
        """Test getting individual strategy signals."""
        composer = StrategyComposer(self.strategies)
        signals = composer.get_strategy_signals()

        self.assertEqual(len(signals), 2)
        self.assertIn("strategy1", signals)
        self.assertIn("strategy2", signals)
        self.assertEqual(signals["strategy1"].signal_type, "buy")
        self.assertEqual(signals["strategy2"].signal_type, "sell")

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        composer = StrategyComposer(self.strategies)

        # Update performance
        composer.update_performance("strategy1", {"sharpe": 1.5, "returns": 0.1})
        composer.update_performance("strategy1", {"sharpe": 1.6, "returns": 0.12})

        # Get performance
        performance = composer.get_strategy_performance("strategy1")

        self.assertIn("sharpe", performance)
        self.assertIn("returns", performance)
        self.assertAlmostEqual(performance["sharpe"], 1.55, places=2)
        self.assertAlmostEqual(performance["returns"], 0.11, places=2)

    def test_error_handling(self):
        """Test error handling when strategies fail."""
        # Create a strategy that raises an exception
        failing_strategy = Mock(spec=BaseStrategy)
        failing_strategy.name = "failing_strategy"
        failing_strategy.weight = 1.0
        failing_strategy.generate_signal.side_effect = Exception("Strategy failed")

        strategies = [self.mock_strategy1, failing_strategy]
        composer = StrategyComposer(strategies)

        # Should not raise exception
        result = composer.aggregate_signals()

        self.assertIsNotNone(result)
        self.assertIn("strategy1", result.contributing_strategies)
        self.assertIn("failing_strategy", result.contributing_strategies)

    def test_empty_strategies(self):
        """Test composer with no strategies returns hold signal."""
        composer = StrategyComposer([])
        result = composer.aggregate_signals()
        self.assertEqual(result.signal_type, "hold")
        self.assertEqual(result.confidence, 0.0)


class TestAdvancedStrategyFramework(unittest.TestCase):
    """Test AdvancedStrategyFramework class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config")
        os.makedirs(self.config_path)

        # Create test configuration files
        self.create_test_configs()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def create_test_configs(self):
        """Create test configuration files."""
        # Composite strategies config
        composite_config = {
            "test_composite": {
                "aggregation_method": "weighted_voting",
                "consensus_threshold": 0.6,
                "strategies": [
                    {
                        "name": "rsi_strategy",
                        "type": "rsi_momentum",
                        "weight": 1.0,
                        "params": {
                            "rsi_period": 14,
                            "rsi_oversold": 30,
                            "rsi_overbought": 70
                        }
                    },
                    {
                        "name": "supertrend_strategy",
                        "type": "supertrend",
                        "weight": 1.0,
                        "params": {
                            "atr_period": 14,
                            "multiplier": 3.0
                        }
                    }
                ]
            }
        }

        with open(os.path.join(self.config_path, "composite_strategies.json"), "w") as f:
            json.dump(composite_config, f)

        # Multi-timeframe config
        mtf_config = {
            "test_mtf": {
                "timeframes": {
                    "primary": "1h",
                    "secondary": ["4h", "1d"],
                    "trend_timeframe": "4h",
                    "entry_timeframe": "1h"
                },
                "strategy_config": {
                    "trend_analysis": {
                        "method": "supertrend",
                        "params": {
                            "atr_period": 14,
                            "multiplier": 3.0
                        }
                    },
                    "entry_signals": {
                        "method": "rsi_volume",
                        "params": {
                            "rsi_period": 14,
                            "rsi_oversold": 30,
                            "rsi_overbought": 70,
                            "volume_threshold": 1.5
                        }
                    }
                },
                "rules": {
                    "entry_only_in_trend_direction": True
                }
            }
        }

        with open(os.path.join(self.config_path, "multi_timeframe.json"), "w") as f:
            json.dump(mtf_config, f)

        # Dynamic switching config
        dynamic_config = {
            "regime_mapping": {
                "trending_volatile": "test_composite",
                "trending_stable": "test_mtf",
                "ranging_volatile": "test_composite",
                "ranging_stable": "test_mtf"
            }
        }

        with open(os.path.join(self.config_path, "dynamic_switching.json"), "w") as f:
            json.dump(dynamic_config, f)

    def test_framework_initialization(self):
        """Test framework initialization with config files."""
        framework = AdvancedStrategyFramework(self.config_path)

        # Check that strategies were loaded
        self.assertIn("test_composite", framework.composite_strategies)
        self.assertIn("test_mtf", framework.multi_timeframe_strategies)
        self.assertIsNotNone(framework.dynamic_switching_config)

    def test_composite_signal_generation(self):
        """Test composite signal generation."""
        framework = AdvancedStrategyFramework(self.config_path)

        # Create test data
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })

        data_feeds = {"1h": test_data}

        # Get composite signal
        result = framework.get_composite_signal("test_composite", data_feeds)

        self.assertIsNotNone(result)
        self.assertIsInstance(result.signal_type, str)
        self.assertIsInstance(result.confidence, float)

    def test_dynamic_strategy_selection(self):
        """Test dynamic strategy selection based on market regime."""
        framework = AdvancedStrategyFramework(self.config_path)

        # Create trending volatile data
        np.random.seed(42)
        trend = np.linspace(100, 120, 30)
        noise = np.random.normal(0, 2, 30)
        prices = trend + noise

        test_data = pd.DataFrame({
            'open': prices,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.ones(30) * 1000
        })

        data_feeds = {"1h": test_data}

        # Get dynamic strategy
        strategy_name = framework.get_dynamic_strategy(data_feeds)

        self.assertIn(strategy_name, ["test_composite", "test_mtf"])

    def test_strategy_execution(self):
        """Test strategy execution with error handling."""
        framework = AdvancedStrategyFramework(self.config_path)

        # Create test data
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })

        data_feeds = {"1h": test_data}

        # Execute strategy
        result = framework.execute_strategy("test_composite", data_feeds)

        self.assertIsNotNone(result)
        self.assertIsInstance(result.signal_type, str)
        self.assertIsInstance(result.confidence, float)

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        framework = AdvancedStrategyFramework(self.config_path)

        # Update performance
        framework.update_performance("test_composite", {"sharpe": 1.5, "returns": 0.1})
        framework.update_performance("test_composite", {"sharpe": 1.6, "returns": 0.12})

        # Get performance
        performance = framework.get_strategy_performance("test_composite")

        self.assertIn("sharpe", performance)
        self.assertIn("returns", performance)
        self.assertAlmostEqual(performance["sharpe"], 1.55, places=2)
        self.assertAlmostEqual(performance["returns"], 0.11, places=2)

    def test_framework_strategy_not_found(self):
        """Test framework returns hold signal if strategy not found."""
        framework = AdvancedStrategyFramework(self.config_path)
        data_feeds = {"1h": pd.DataFrame({
            'open': [100, 101, 102],
            'high': [102, 103, 104],
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1100, 1200]
        })}
        result = framework.get_composite_signal("nonexistent_strategy", data_feeds)
        self.assertEqual(result.signal_type, "hold")
        self.assertIn("error", result.metadata)


if __name__ == "__main__":
    unittest.main()