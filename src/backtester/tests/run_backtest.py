#!/usr/bin/env python
"""
Quick Start Script for Running Backtests
-----------------------------------------

This script provides a simple CLI for running backtests using JSON configurations.

Usage:
    # CLI mode
    python src/backtester/tests/run_backtest.py config/backtester/custom_strategy_test.json
    python src/backtester/tests/run_backtest.py --list-configs

    # Debug mode (set parameters in main() and run from IDE)
    # Just set DEBUG_CONFIG and run the script
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.backtester.tests.backtester_test_framework import run_backtest_from_config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


# =============================================================================
# DEBUG CONFIGURATION
# Set these parameters when running from IDE debugger
# =============================================================================
DEBUG_MODE = True  # Set to True to use DEBUG_CONFIG, False for CLI mode
DEBUG_CONFIG = {
    # Path to your config file (relative to project root or absolute)
    'config_path': 'config/backtester/custom_strategy_test.json',

    # Generate report file? (True/False)
    'generate_report': True,

    # Enable verbose output? (True/False)
    'verbose': True,
}
# =============================================================================


def list_available_configs():
    """List all available configuration files."""
    config_dir = Path("config/backtester")

    if not config_dir.exists():
        print("No config directory found at: config/backtester/")
        return

    configs = list(config_dir.glob("*.json"))

    if not configs:
        print("No configuration files found in config/backtester/")
        return

    print("\nAvailable Configurations:")
    print("=" * 80)

    for i, config_path in enumerate(configs, 1):
        print(f"\n{i}. {config_path.name}")

        # Try to load and show basic info
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)

            print(f"   Test Name: {config.get('test_name', 'N/A')}")
            print(f"   Strategy: {config.get('strategy', {}).get('type', 'N/A')}")

            if 'parameters' in config.get('strategy', {}):
                params = config['strategy']['parameters']
                if 'entry_logic' in params:
                    print(f"   Entry: {params['entry_logic']['name']}")
                if 'exit_logic' in params:
                    print(f"   Exit: {params['exit_logic']['name']}")

            print(f"   Data: {config.get('data', {}).get('file_path', 'N/A')}")

        except Exception as e:
            print(f"   (Error reading config: {e})")

    print("\n" + "=" * 80)
    print(f"\nTo run a test:")
    print(f"  python run_backtest.py config/backtester/<config_name>.json")
    print()


def run_backtest(config_path: str, generate_report: bool = True, verbose: bool = False) -> int:
    """
    Run a backtest with the given configuration.

    Args:
        config_path: Path to JSON configuration file
        generate_report: Whether to generate and save report file
        verbose: Enable verbose error output

    Returns:
        Exit code (0 = success, 1 = test failed, 2 = error)
    """
    config_path = Path(config_path)

    # Handle relative paths from project root
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        print(f"\n[ERROR] Configuration file not found: {config_path}")
        print("\nUse --list-configs to see available configurations")
        return 1

    try:
        print("\n" + "=" * 80)
        print("RUNNING BACKTEST")
        print("=" * 80)
        print(f"Config: {config_path}")
        print(f"Mode: {'Debug' if DEBUG_MODE else 'CLI'}")
        print()

        # Run backtest
        results = run_backtest_from_config(
            str(config_path),
            generate_report=generate_report
        )

        # Print report (console-safe version for Windows)
        report_text = results['report']
        # Replace Unicode characters with ASCII alternatives for console output
        console_safe_report = report_text.replace('✓', '[PASS]').replace('✗', '[FAIL]')
        print("\n" + console_safe_report)

        # Print summary (console-safe)
        print("\n" + "=" * 80)
        if results['success']:
            print("[PASS] TEST PASSED")
        else:
            print("[FAIL] TEST FAILED")
            if results['validation'].get('failures'):
                print("\nFailures:")
                for failure in results['validation']['failures']:
                    print(f"  - {failure}")

        print("=" * 80)
        print()

        # Exit with appropriate code
        return 0 if results['success'] else 1

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPossible issues:")
        print("  - Data file not found (check 'file_path' in config)")
        print("  - Config file not found")
        return 2

    except Exception as e:
        print(f"\n[ERROR] {e}")
        if verbose:
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
        else:
            print("\n(Use --verbose for full error details)")
        return 2


def main():
    """Main entry point - supports both CLI and debug mode."""

    # =========================================================================
    # DEBUG MODE: Run with parameters set at the top of the file
    # =========================================================================
    if DEBUG_MODE:
        print("=" * 80)
        print("RUNNING IN DEBUG MODE")
        print("=" * 80)
        print(f"Config: {DEBUG_CONFIG['config_path']}")
        print(f"Generate Report: {DEBUG_CONFIG['generate_report']}")
        print(f"Verbose: {DEBUG_CONFIG['verbose']}")
        print()
        print("Tip: Set DEBUG_MODE = False at the top of the file to use CLI mode")
        print("=" * 80)

        return run_backtest(
            config_path=DEBUG_CONFIG['config_path'],
            generate_report=DEBUG_CONFIG['generate_report'],
            verbose=DEBUG_CONFIG['verbose']
        )

    # =========================================================================
    # CLI MODE: Parse command line arguments
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Run backtest from JSON configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/backtester/tests/run_backtest.py config/backtester/custom_strategy_test.json
  python src/backtester/tests/run_backtest.py config/backtester/rsi_volume_supertrend_test.json --no-report
  python src/backtester/tests/run_backtest.py --list-configs

Debug Mode:
  Set DEBUG_MODE = True and DEBUG_CONFIG at the top of the file, then run from IDE
        """
    )

    parser.add_argument(
        'config',
        nargs='?',
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating report file'
    )
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List all available configuration files'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # List configs mode
    if args.list_configs:
        list_available_configs()
        return 0

    # Validate config argument
    if not args.config:
        parser.print_help()
        print("\nError: config file required (or use --list-configs)")
        return 1

    return run_backtest(
        config_path=args.config,
        generate_report=not args.no_report,
        verbose=args.verbose
    )


if __name__ == "__main__":
    sys.exit(main())
