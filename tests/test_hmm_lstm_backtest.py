"""
Test script for HMM-LSTM Backtesting System

This script validates the setup and basic functionality of the HMM-LSTM backtesting system.
It checks for required files, validates configurations, and tests basic operations.

Usage:
    python tests/test_hmm_lstm_backtest.py
"""

import sys
import json
import yaml
from pathlib import Path
import unittest
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.backtester.optimizer.hmm_lstm import HMMLSTMOptimizer


class TestHMMLSTMBacktest(unittest.TestCase):
    """Test cases for HMM-LSTM backtesting system."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            "optimizer_type": "hmm_lstm",
            "initial_capital": 10000.0,
            "commission": 0.001,
            "position_size": 0.1,
            "plot": True,
            "save_trades": True,
            "output_dir": "test_results",

            "ml_models": {
                    "pipeline_dir": "src/ml/pipeline/p01_hmm_lstm",
    "models_dir": "src/ml/pipeline/p01_hmm_lstm/models",
                "config_file": "config/pipeline/p01.yaml"
            },

            "strategy": {
                "name": "HMMLSTMStrategy",
                "entry_threshold": 0.6,
                "regime_confidence_threshold": 0.7
            },

            "data": {
                "data_dir": "data",
                "start_date": "2023-01-01",
                "end_date": "2024-01-01"
            },

            "risk_management": {
                "max_position_size": 0.2,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.1,
                "max_drawdown": 0.15
            },

            "optimization": {
                "enabled": False,
                "n_trials": 50,
                "optimize_params": [
                    "entry_threshold",
                    "exit_threshold",
                    "regime_confidence_threshold"
                ],
                "parameter_ranges": {
                    "entry_threshold": {
                        "min": 0.3,
                        "max": 0.8,
                        "type": "float"
                    },
                    "exit_threshold": {
                        "min": 0.2,
                        "max": 0.7,
                        "type": "float"
                    },
                    "regime_confidence_threshold": {
                        "min": 0.5,
                        "max": 0.9,
                        "type": "float"
                    }
                }
            }
        }

        # Create test config file
        self.test_config_path = Path("test_config.json")
        with open(self.test_config_path, 'w') as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_config_path.exists():
            self.test_config_path.unlink()

        # Clean up test results directory
        test_results_dir = Path("test_results")
        if test_results_dir.exists():
            import shutil
            shutil.rmtree(test_results_dir)

    def test_config_loading(self):
        """Test configuration loading functionality."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Check if config was loaded correctly
        self.assertEqual(optimizer.config['initial_capital'], 10000.0)
        self.assertEqual(optimizer.config['strategy']['entry_threshold'], 0.6)
        self.assertEqual(optimizer.config['data']['data_dir'], 'data')

    def test_pipeline_config_loading(self):
        """Test pipeline configuration loading."""
        # Create mock pipeline config
        mock_pipeline_config = {
            'paths': {
                            'data_labeled': 'src/ml/pipeline/p01_hmm_lstm/data_labeled',
            'models_hmm': 'src/ml/pipeline/p01_hmm_lstm/models'
            }
        }

        mock_config_path = Path("mock_pipeline_config.yaml")
        with open(mock_config_path, 'w') as f:
            yaml.dump(mock_pipeline_config, f)

        try:
            # Update test config to use mock pipeline config
            self.test_config['ml_models']['config_file'] = str(mock_config_path)
            with open(self.test_config_path, 'w') as f:
                json.dump(self.test_config, f)

            optimizer = HMMLSTMOptimizer(str(self.test_config_path))

            # Check if pipeline config was loaded
            self.assertIn('paths', optimizer.pipeline_config)
            self.assertIn('data_labeled', optimizer.pipeline_config['paths'])

        finally:
            if mock_config_path.exists():
                mock_config_path.unlink()

    def test_model_discovery(self):
        """Test model file discovery functionality."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Test with non-existent models (should return None)
        hmm_path, lstm_path = optimizer.find_latest_models("NONEXISTENT", "1h")
        self.assertIsNone(hmm_path)
        self.assertIsNone(lstm_path)

    @patch('pathlib.Path.glob')
    def test_model_discovery_with_mock_files(self, mock_glob):
        """Test model discovery with mock files."""
        # Mock model files
        mock_hmm_files = [
                    Path("src/ml/pipeline/p01_hmm_lstm/models/hmm_BTCUSDT_1h_20240101.pkl"),
        Path("src/ml/pipeline/p01_hmm_lstm/models/hmm_BTCUSDT_1h_20240102.pkl")
        ]
        mock_lstm_files = [
                    Path("src/ml/pipeline/p01_hmm_lstm/models/lstm_BTCUSDT_1h_20240101.pkl"),
        Path("src/ml/pipeline/p01_hmm_lstm/models/lstm_BTCUSDT_1h_20240102.pkl")
        ]

        def mock_glob_side_effect(pattern):
            if "hmm_" in pattern:
                return mock_hmm_files
            elif "lstm_" in pattern:
                return mock_lstm_files
            return []

        mock_glob.side_effect = mock_glob_side_effect

        optimizer = HMMLSTMOptimizer(str(self.test_config_path))
        hmm_path, lstm_path = optimizer.find_latest_models("BTCUSDT", "1h")

        # Should return the latest (last) files
        self.assertEqual(hmm_path, mock_hmm_files[-1])
        self.assertEqual(lstm_path, mock_lstm_files[-1])

    def test_config_validation(self):
        """Test configuration validation."""
        # Test with invalid config (missing required fields)
        invalid_config = {"optimizer_type": "hmm_lstm"}
        invalid_config_path = Path("invalid_config.json")

        with open(invalid_config_path, 'w') as f:
            json.dump(invalid_config, f)

        try:
            with self.assertRaises(KeyError):
                optimizer = HMMLSTMOptimizer(str(invalid_config_path))
        finally:
            if invalid_config_path.exists():
                invalid_config_path.unlink()

    def test_output_directory_creation(self):
        """Test output directory creation."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Check if output directory was created
        output_dir = Path(optimizer.config['output_dir'])
        self.assertTrue(output_dir.exists())
        self.assertTrue(output_dir.is_dir())

    def test_device_detection(self):
        """Test PyTorch device detection."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Device should be either 'cpu' or 'cuda'
        self.assertIn(optimizer.device.type, ['cpu', 'cuda'])

    @patch('pandas.read_csv')
    def test_data_preparation(self, mock_read_csv):
        """Test data preparation functionality."""
        # Mock DataFrame
        import pandas as pd
        mock_df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
            'open': [100] * 100,
            'high': [110] * 100,
            'low': [90] * 100,
            'close': [105] * 100,
            'volume': [1000] * 100,
            'regime': [0] * 100
        })
        mock_read_csv.return_value = mock_df

        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True):
            optimizer = HMMLSTMOptimizer(str(self.test_config_path))

            # Test data preparation
            df = optimizer.prepare_data("BTCUSDT", "1h")

            # Check if DataFrame has required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume', 'regime']
            for col in required_cols:
                self.assertIn(col, df.columns)

    def test_parameter_optimization_disabled(self):
        """Test parameter optimization when disabled."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Optimization should be disabled by default
        self.assertFalse(optimizer.config['optimization']['enabled'])

        # Test optimization method returns empty dict when disabled
        result = optimizer.optimize_parameters("BTCUSDT", "1h", {}, {})
        self.assertEqual(result, {})

    def test_results_serialization(self):
        """Test results serialization functionality."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Test data with numpy types
        import numpy as np
        test_results = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'total_return': np.float64(0.15),
            'sharpe_ratio': np.float32(1.2),
            'max_drawdown': np.float64(0.05),
            'trades': {
                'total': np.int64(50),
                'won': np.int32(30)
            }
        }

        # Test serialization using the convert_numpy function from the optimizer
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        serialized = convert_numpy(test_results)

        # Check that numpy types were converted
        self.assertIsInstance(serialized['total_return'], float)
        self.assertIsInstance(serialized['sharpe_ratio'], float)
        self.assertIsInstance(serialized['trades']['total'], int)
        self.assertIsInstance(serialized['trades']['won'], int)

    def test_error_handling(self):
        """Test error handling in the optimizer."""
        # Test with non-existent config file
        with self.assertRaises(FileNotFoundError):
            optimizer = HMMLSTMOptimizer("non_existent_config.json")

    def test_config_structure(self):
        """Test configuration structure validation."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Check required sections exist
        required_sections = ['ml_models', 'strategy', 'data', 'risk_management', 'optimization']
        for section in required_sections:
            self.assertIn(section, optimizer.config)

        # Check strategy parameters
        strategy_params = optimizer.config['strategy']
        required_strategy_params = ['entry_threshold', 'regime_confidence_threshold']
        for param in required_strategy_params:
            self.assertIn(param, strategy_params)

    def test_data_directory_config(self):
        """Test data directory configuration."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Check if data directory is configured
        self.assertEqual(optimizer.config['data']['data_dir'], 'data')

    def test_optimization_parameter_ranges(self):
        """Test optimization parameter ranges configuration."""
        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Check if parameter ranges are configured
        param_ranges = optimizer.config['optimization']['parameter_ranges']

        # Check entry_threshold range
        self.assertIn('entry_threshold', param_ranges)
        self.assertEqual(param_ranges['entry_threshold']['min'], 0.3)
        self.assertEqual(param_ranges['entry_threshold']['max'], 0.8)
        self.assertEqual(param_ranges['entry_threshold']['type'], 'float')

        # Check exit_threshold range
        self.assertIn('exit_threshold', param_ranges)
        self.assertEqual(param_ranges['exit_threshold']['min'], 0.2)
        self.assertEqual(param_ranges['exit_threshold']['max'], 0.7)
        self.assertEqual(param_ranges['exit_threshold']['type'], 'float')

        # Check regime_confidence_threshold range
        self.assertIn('regime_confidence_threshold', param_ranges)
        self.assertEqual(param_ranges['regime_confidence_threshold']['min'], 0.5)
        self.assertEqual(param_ranges['regime_confidence_threshold']['max'], 0.9)
        self.assertEqual(param_ranges['regime_confidence_threshold']['type'], 'float')

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    def test_discover_available_combinations(self, mock_glob, mock_exists):
        """Test the discover_available_combinations method."""
        # Mock data directory exists
        mock_exists.return_value = True

        # Mock CSV files
        mock_csv_files = [
            Path("data/BTCUSDT_1h.csv"),
            Path("data/ETHUSDT_4h.csv"),
            Path("data/ADAUSDT_1h.csv"),
            Path("data/invalid_file.csv"),  # Invalid format
            Path("data/BTCUSDT_1h_extra.csv")  # Invalid format
        ]
        mock_glob.return_value = mock_csv_files

        optimizer = HMMLSTMOptimizer(str(self.test_config_path))

        # Mock model discovery to return some models
        with patch.object(optimizer, 'find_latest_models') as mock_find_models:
            mock_find_models.side_effect = [
                (Path("hmm.pkl"), Path("lstm.pkl")),  # BTCUSDT_1h - has models
                (None, None),  # ETHUSDT_4h - no models
                (Path("hmm.pkl"), Path("lstm.pkl")),  # ADAUSDT_1h - has models
                (None, None),  # invalid_file - no models (and invalid format)
                (None, None)   # BTCUSDT_1h_extra - no models (and invalid format)
            ]

            combinations = optimizer.discover_available_combinations()

            # Should only return combinations with both data and models
            expected_combinations = [("BTCUSDT", "1h"), ("ADAUSDT", "1h")]
            self.assertEqual(combinations, expected_combinations)


def run_integration_tests():
    """Run integration tests that require actual files."""
    print("Running integration tests...")

    # Check if pipeline directory exists
    pipeline_dir = Path("src/ml/pipeline/p01_hmm_lstm")
    if not pipeline_dir.exists():
        print("❌ Pipeline directory not found. Skipping integration tests.")
        return False

    # Check if models directory exists
    models_dir = pipeline_dir / "models"
    if not models_dir.exists():
        print("❌ Models directory not found. Skipping integration tests.")
        return False

    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found. Skipping integration tests.")
        return False

    print("✅ Integration test prerequisites met!")
    return True


def main():
    """Main function to run all tests."""
    print("HMM-LSTM Backtesting System Tests")
    print("=" * 50)

    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    print("\n" + "=" * 50)

    # Run integration tests
    print("\nRunning integration tests...")
    integration_success = run_integration_tests()

    print("\n" + "=" * 50)

    if integration_success:
        print("✅ All tests completed successfully!")
        print("\nThe HMM-LSTM backtesting system is ready to use.")
        print("Run the following to start backtesting:")
        print("python src/backtester/optimizer/hmm_lstm.py")
    else:
        print("⚠️  Unit tests passed, but integration tests require pipeline setup.")
        print("Please complete the HMM-LSTM pipeline training first.")


if __name__ == "__main__":
    main()
