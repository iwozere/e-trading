"""
HMM-LSTM Backtesting Example

This example demonstrates how to use the HMM-LSTM backtesting system
programmatically to evaluate trained models on historical data.

Prerequisites:
1. Complete HMM-LSTM pipeline training
2. Trained models available in src/ml/pipeline/hmm_lstm_01/models/
3. OHLCV data available in data/{symbol}_{timeframe}.csv

Usage:
    python examples/hmm_lstm_backtest_example.py
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.backtester.optimizer.hmm_lstm import HMMLSTMOptimizer


def create_custom_config():
    """Create a custom configuration for the example."""
    config = {
        "optimizer_type": "hmm_lstm",
        "initial_capital": 10000.0,
        "commission": 0.001,
        "position_size": 0.1,
        "plot": True,
        "save_trades": True,
        "output_dir": "results",

        "ml_models": {
            "pipeline_dir": "src/ml/pipeline/hmm_lstm_01",
            "models_dir": "src/ml/pipeline/hmm_lstm_01/models",
            "config_file": "config/pipeline/x01.yaml"
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
            "enabled": False,  # Disable optimization for this example
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

    return config


def save_config(config, filename="hmm_lstm_example.json"):
    """Save configuration to file."""
    config_path = Path("config/optimizer") / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")
    return config_path


def run_basic_backtest():
    """Run a basic backtest with default settings."""
    print("=== Running Basic HMM-LSTM Backtest ===")

    # Create custom configuration
    config = create_custom_config()
    config_path = save_config(config)

    # Initialize optimizer
    optimizer = HMMLSTMOptimizer(str(config_path))

    # Run backtesting
    optimizer.run()

    print("Basic backtest completed!")


def run_optimization_backtest():
    """Run backtest with parameter optimization."""
    print("=== Running HMM-LSTM Backtest with Optimization ===")

    # Create configuration with optimization enabled
    config = create_custom_config()
    config["optimization"]["enabled"] = True
    config["optimization"]["n_trials"] = 20  # Reduced for faster execution

    config_path = save_config(config, "hmm_lstm_optimize.json")

    # Initialize optimizer
    optimizer = HMMLSTMOptimizer(str(config_path))

    # Run backtesting with optimization
    optimizer.run()

    print("Optimization backtest completed!")


def run_multi_symbol_backtest():
    """Run backtest with all available symbols and timeframes."""
    print("=== Running Multi-Symbol HMM-LSTM Backtest ===")

    # Create configuration (will auto-discover all available combinations)
    config = create_custom_config()
    config_path = save_config(config, "hmm_lstm_multi.json")

    # Initialize optimizer
    optimizer = HMMLSTMOptimizer(str(config_path))

    # Run backtesting (will process all discovered combinations)
    optimizer.run()

    print("Multi-symbol backtest completed!")


def check_prerequisites():
    """Check if prerequisites are met."""
    print("=== Checking Prerequisites ===")

    # Check if models directory exists
    models_dir = Path("src/ml/pipeline/hmm_lstm_01/models")
    if not models_dir.exists():
        print("❌ Models directory not found. Please run the HMM-LSTM pipeline first.")
        return False

        # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found. Please create a 'data' directory with OHLCV files.")
        return False

    # Check for model files
    model_files = list(models_dir.glob("*.pkl"))
    if not model_files:
        print("❌ No model files found. Please run the HMM-LSTM pipeline first.")
        return False

    # Check for OHLCV data files
    data_files = list(data_dir.glob("*.csv"))
    if not data_files:
        print("❌ No OHLCV data files found. Please add CSV files to the 'data' directory.")
        return False

    print("✅ Prerequisites met!")
    print(f"   Found {len(model_files)} model files")
    print(f"   Found {len(data_files)} OHLCV data files")
    return True


def main():
    """Main function to run the example."""
    print("HMM-LSTM Backtesting Example")
    print("=" * 50)

    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease complete the HMM-LSTM pipeline training first:")
        print("cd src/ml/pipeline/hmm_lstm_01")
        print("python run_pipeline.py")
        return

    print("\n" + "=" * 50)

    # Run different examples
    try:
        # Example 1: Basic backtest
        run_basic_backtest()

        print("\n" + "=" * 50)

        # Example 2: Optimization backtest (optional)
        response = input("\nRun optimization backtest? (y/n): ").lower().strip()
        if response == 'y':
            run_optimization_backtest()

        print("\n" + "=" * 50)

        # Example 3: Multi-symbol backtest (optional)
        response = input("\nRun multi-symbol backtest? (y/n): ").lower().strip()
        if response == 'y':
            run_multi_symbol_backtest()

        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("\nCheck the 'results/' directory for output files.")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("\nPlease check the error messages and ensure all prerequisites are met.")


if __name__ == "__main__":
    main()
