"""
Backtester Test Framework - Usage Examples
------------------------------------------

This module demonstrates various ways to use the backtester test framework.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.backtester.tests.backtester_test_framework import (
    BacktesterTestFramework,
    run_backtest_from_config
)


def example_1_simple_run():
    """
    Example 1: Simple backtest run using convenience function.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Backtest Run")
    print("="*80)

    config_path = "config/backtester/custom_strategy_test.json"

    try:
        results = run_backtest_from_config(config_path, generate_report=True)

        print(results['report'])

        if results['success']:
            print("\n✓ Test PASSED")
        else:
            print("\n✗ Test FAILED")
            print("Failures:", results['validation']['failures'])

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the data file exists at: data/BTCUSDT_1h.csv")


def example_2_step_by_step():
    """
    Example 2: Step-by-step backtest with manual control.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Step-by-Step Backtest")
    print("="*80)

    config_path = "config/backtester/custom_strategy_test.json"

    try:
        # Initialize framework
        framework = BacktesterTestFramework(config_path)
        print(f"✓ Loaded config: {framework.config['test_name']}")

        # Setup backtest
        framework.setup_backtest()
        print("✓ Backtest setup complete")

        # Run backtest
        results = framework.run_backtest()
        print("✓ Backtest execution complete")

        # Display key metrics
        print("\nKey Metrics:")
        print(f"  Initial Value: ${results['initial_value']:,.2f}")
        print(f"  Final Value: ${results['final_value']:,.2f}")
        print(f"  Total Return: {results['total_return']*100:.2f}%")
        print(f"  Total Trades: {results.get('total_trades', 0)}")
        print(f"  Win Rate: {results.get('win_rate', 0)*100:.2f}%")

        # Validate assertions
        validation = framework.validate_assertions()
        print(f"\n✓ Validation complete: {'PASSED' if validation['passed'] else 'FAILED'}")

        # Generate report
        report = framework.generate_report()
        print("\n" + report)

    except Exception as e:
        print(f"Error: {e}")


def example_3_custom_config():
    """
    Example 3: Create and run a custom configuration programmatically.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Configuration")
    print("="*80)

    import json
    import tempfile

    # Create custom config
    custom_config = {
        "test_name": "Programmatic Test Example",
        "description": "Testing RSI+Volume+Supertrend with Trailing Stop",
        "strategy": {
            "type": "CustomStrategy",
            "parameters": {
                "entry_logic": {
                    "name": "RSIVolumeSuperTrendMixin",
                    "params": {
                        "rsi_period": 14,
                        "rsi_threshold": 40,
                        "volume_ma_period": 20,
                        "volume_threshold": 1.5,
                        "supertrend_period": 10,
                        "supertrend_multiplier": 3.0
                    }
                },
                "exit_logic": {
                    "name": "TrailingStopExitMixin",
                    "params": {
                        "trail_percent": 5.0,
                        "min_profit_percent": 2.0
                    }
                },
                "position_size": 0.15
            }
        },
        "data": {
            "file_path": "data/BTCUSDT_1h.csv",
            "symbol": "BTCUSDT",
            "datetime_col": "timestamp",
            "open_col": "open",
            "high_col": "high",
            "low_col": "low",
            "close_col": "close",
            "volume_col": "volume",
            "fromdate": "2023-01-01",
            "todate": "2023-06-30"
        },
        "broker": {
            "cash": 25000.0,
            "commission": 0.001
        },
        "assertions": {
            "min_trades": 1
        }
    }

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(custom_config, f, indent=2)
        temp_config_path = f.name

    print(f"Created temporary config: {temp_config_path}")

    try:
        # Run backtest
        results = run_backtest_from_config(temp_config_path, generate_report=False)

        print(f"\nResults:")
        print(f"  Total Return: {results['results']['total_return']*100:.2f}%")
        print(f"  Total Trades: {results['results'].get('total_trades', 0)}")
        print(f"  Test Status: {'PASSED' if results['success'] else 'FAILED'}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        import os
        os.unlink(temp_config_path)
        print(f"\nCleaned up temporary config")


def example_4_batch_testing():
    """
    Example 4: Run multiple configurations and compare results.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Testing")
    print("="*80)

    config_dir = Path("config/backtester")
    configs = [
        "custom_strategy_test.json",
        "rsi_volume_supertrend_test.json",
        "trailing_stop_test.json"
    ]

    results_list = []

    for config_name in configs:
        config_path = config_dir / config_name

        if not config_path.exists():
            print(f"Skipping {config_name} (not found)")
            continue

        print(f"\nTesting: {config_name}")
        print("-" * 80)

        try:
            results = run_backtest_from_config(str(config_path), generate_report=False)

            result_summary = {
                'config': config_name,
                'test_name': results['results']['test_name'],
                'return_pct': results['results']['total_return'] * 100,
                'total_trades': results['results'].get('total_trades', 0),
                'win_rate': results['results'].get('win_rate', 0) * 100,
                'sharpe_ratio': results['results'].get('sharpe_ratio', 0),
                'max_drawdown_pct': results['results'].get('max_drawdown', 0) * 100,
                'passed': results['success']
            }

            results_list.append(result_summary)

            print(f"  Return: {result_summary['return_pct']:.2f}%")
            print(f"  Trades: {result_summary['total_trades']}")
            print(f"  Win Rate: {result_summary['win_rate']:.2f}%")
            print(f"  Status: {'✓ PASSED' if result_summary['passed'] else '✗ FAILED'}")

        except Exception as e:
            print(f"  Error: {e}")

    # Display summary
    if results_list:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)

        # Sort by return
        results_list.sort(key=lambda x: x['return_pct'], reverse=True)

        print(f"\n{'Config':<35} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Status'}")
        print("-" * 80)

        for result in results_list:
            status = "✓" if result['passed'] else "✗"
            print(f"{result['config']:<35} {result['return_pct']:>8.2f}% "
                  f"{result['total_trades']:>6} {result['win_rate']:>8.2f}%   {status}")

        print()


def example_5_custom_validation():
    """
    Example 5: Add custom validation logic.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Validation")
    print("="*80)

    config_path = "config/backtester/custom_strategy_test.json"

    try:
        framework = BacktesterTestFramework(config_path)
        framework.setup_backtest()
        results = framework.run_backtest()

        print("Running custom validations...")

        # Custom validation rules
        validations = []

        # Check 1: Win rate should be reasonable
        win_rate = results.get('win_rate', 0)
        if win_rate >= 0.4:  # At least 40%
            validations.append(("✓", f"Win rate acceptable: {win_rate*100:.1f}%"))
        else:
            validations.append(("✗", f"Win rate too low: {win_rate*100:.1f}%"))

        # Check 2: Profit factor
        profit_factor = results.get('profit_factor', 0)
        if profit_factor >= 1.5:
            validations.append(("✓", f"Profit factor good: {profit_factor:.2f}"))
        else:
            validations.append(("✗", f"Profit factor low: {profit_factor:.2f}"))

        # Check 3: Reasonable number of trades
        total_trades = results.get('total_trades', 0)
        if 5 <= total_trades <= 100:
            validations.append(("✓", f"Trade count reasonable: {total_trades}"))
        else:
            validations.append(("✗", f"Trade count unusual: {total_trades}"))

        # Display results
        print("\nCustom Validation Results:")
        print("-" * 80)
        for status, message in validations:
            print(f"  {status} {message}")

        # Overall status
        all_passed = all(status == "✓" for status, _ in validations)
        print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BACKTESTER TEST FRAMEWORK - USAGE EXAMPLES")
    print("="*80)
    print("\nThis script demonstrates various ways to use the backtester framework.")
    print("Each example can be run independently or all together.")
    print()

    examples = [
        ("Simple Run", example_1_simple_run),
        ("Step-by-Step", example_2_step_by_step),
        ("Custom Config", example_3_custom_config),
        ("Batch Testing", example_4_batch_testing),
        ("Custom Validation", example_5_custom_validation),
    ]

    # Check if data file exists
    data_file = Path("data/BTCUSDT_1h.csv")
    if not data_file.exists():
        print("⚠️  WARNING: Test data file not found at: data/BTCUSDT_1h.csv")
        print("   Examples will fail without this file.")
        print()
        return

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Running Example {i}: {name}")
        print(f"{'='*80}")

        try:
            func()
        except Exception as e:
            print(f"\n✗ Example failed with error: {e}")

        input("\nPress Enter to continue to next example (or Ctrl+C to exit)...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
