"""
HMM-LSTM Backtesting Example

This example demonstrates how to use the updated HMM-LSTM optimizer
with the same structure and quality as custom_optimizer.py.

Prerequisites:
1. Complete the HMM-LSTM pipeline training
2. Ensure you have OHLCV data files in data/ directory
3. Ensure trained models are available in src/ml/pipeline/p01_hmm_lstm/models/
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.backtester.optimizer.hmm_lstm import HMMLSTMOptimizer


def example_basic_backtesting():
    """Example 1: Basic backtesting with default configuration."""
    print("=" * 60)
    print("Example 1: Basic HMM-LSTM Backtesting")
    print("=" * 60)

    try:
        # Initialize optimizer with default config
        optimizer = HMMLSTMOptimizer("config/optimizer/p01_hmm_lstm.json")

        # Run backtesting (automatically discovers available combinations)
        optimizer.run()

        print("Backtesting completed successfully!")
        print("Check the results/ directory for detailed reports and visualizations.")

    except Exception as e:
        print(f"Error in basic backtesting: {e}")
        return None


def example_with_optimization():
    """Example 2: Backtesting with parameter optimization."""
    print("\n" + "=" * 60)
    print("Example 2: HMM-LSTM Backtesting with Optimization")
    print("=" * 60)

    try:
        # Load configuration
        with open("config/optimizer/p01_hmm_lstm.json", 'r') as f:
            config = json.load(f)

        # Enable optimization
        config['optimization']['enabled'] = True
        config['optimization']['n_trials'] = 20  # Reduced for faster execution

        # Save modified config
        temp_config_path = "config/optimizer/p01_hmm_lstm_optimization.json"
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Initialize optimizer with optimization enabled
        optimizer = HMMLSTMOptimizer(temp_config_path)

        # Run backtesting with optimization
        optimizer.run()

        print("Backtesting with optimization completed successfully!")
        print("Check the results/ directory for optimization results.")

        # Clean up temporary config
        Path(temp_config_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"Error in optimization backtesting: {e}")
        return None


def example_custom_configuration():
    """Example 3: Backtesting with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 3: HMM-LSTM Backtesting with Custom Configuration")
    print("=" * 60)

    try:
        # Create custom configuration
        custom_config = {
            "optimizer_type": "hmm_lstm",
            "initial_capital": 50000.0,  # Higher initial capital
            "commission": 0.0005,  # Lower commission
            "position_size": 0.15,  # Larger position size
            "plot": True,
            "save_trades": True,
            "output_dir": "results/custom_hmm_lstm",

            "ml_models": {
                "pipeline_dir": "src/ml/pipeline/p01_hmm_lstm",
                "models_dir": "src/ml/pipeline/p01_hmm_lstm/models",
                "config_file": "config/pipeline/p01.yaml"
            },

            "strategy": {
                "name": "HMMLSTMStrategy",
                "entry_threshold": 0.5,  # More conservative entry
                "regime_confidence_threshold": 0.8  # Higher confidence requirement
            },

            "data": {
                "data_dir": "data",
                "start_date": "2023-06-01",  # Shorter period
                "end_date": "2023-12-31"
            },

            "risk_management": {
                "max_position_size": 0.25,
                "stop_loss_pct": 0.03,  # Tighter stop loss
                "take_profit_pct": 0.08,  # Lower profit target
                "max_drawdown": 0.10
            },

            "optimization": {
                "enabled": False  # No optimization for this example
            }
        }

        # Save custom config
        custom_config_path = "config/optimizer/p01_hmm_lstm_custom.json"
        with open(custom_config_path, 'w') as f:
            json.dump(custom_config, f, indent=2)

        # Initialize optimizer with custom config
        optimizer = HMMLSTMOptimizer(custom_config_path)

        # Run backtesting
        optimizer.run()

        print("Custom configuration backtesting completed successfully!")
        print("Check the results/custom_hmm_lstm/ directory for results.")

        # Clean up custom config
        Path(custom_config_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"Error in custom configuration backtesting: {e}")
        return None


def main():
    """Run all examples."""
    print("HMM-LSTM Backtesting Examples")
    print("=" * 60)
    print("This script demonstrates the updated HMM-LSTM optimizer.")
    print("Make sure you have completed the p01_hmm_lstm pipeline before running.")
    print()

    # Check prerequisites
    models_dir = Path("src/ml/pipeline/p01_hmm_lstm/models")
    data_dir = Path("data")

    if not models_dir.exists():
        print("❌ Models directory not found. Please complete the HMM-LSTM pipeline training first.")
        return

    if not data_dir.exists():
        print("❌ Data directory not found. Please ensure you have OHLCV data files.")
        return

    print("✅ Prerequisites check passed!")
    print()

    # Run examples
    example_basic_backtesting()
    example_with_optimization()
    example_custom_configuration()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("Examples completed! Check the following for results:")
    print("- Basic backtesting: results/")
    print("- Optimization results: results/ (with optimization enabled)")
    print("- Custom configuration: results/custom_hmm_lstm/")
    print()
    print("Key improvements in the updated HMM-LSTM optimizer:")
    print("1. ✅ Consistent results format with custom_optimizer.py")
    print("2. ✅ Same analyzer structure and quality")
    print("3. ✅ Enhanced optimization framework")
    print("4. ✅ Better error handling and logging")
    print("5. ✅ Automatic discovery of available symbol-timeframe combinations")
    print("6. ✅ Comprehensive performance analysis")
    print()
    print("Next steps:")
    print("1. Review the generated reports and visualizations")
    print("2. Analyze the performance metrics")
    print("3. Use the best parameters for live trading")
    print("4. Consider running more extensive optimization")


if __name__ == "__main__":
    main()
