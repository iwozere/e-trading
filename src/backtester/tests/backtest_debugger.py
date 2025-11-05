"""
Backtest Debugger
-----------------

This module provides debugging utilities for backtests that generate no trades
or unexpected results.

Features:
- Analyze entry/exit conditions
- Check indicator values
- Identify why signals aren't triggering
- Suggest parameter adjustments
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
from src.data.feed.file_data_feed import FileDataFeed
from src.strategy.entry.entry_mixin_factory import ENTRY_MIXIN_REGISTRY
from src.strategy.exit.exit_mixin_factory import EXIT_MIXIN_REGISTRY

_logger = setup_logger(__name__)


class BacktestDebugger:
    """
    Debug helper for backtests with no trades or unexpected results.
    """

    def __init__(self, config_path: str):
        """
        Initialize debugger with test configuration.

        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data from configuration."""
        data_config = self.config['data']
        file_path = Path(data_config['file_path'])

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Load CSV
        df = pd.read_csv(file_path)

        # Parse datetime
        datetime_col = data_config.get('datetime_col', 'datetime')
        df['datetime'] = pd.to_datetime(df[datetime_col])

        # Apply date filters if specified
        if data_config.get('fromdate'):
            df = df[df['datetime'] >= pd.to_datetime(data_config['fromdate'])]
        if data_config.get('todate'):
            df = df[df['datetime'] <= pd.to_datetime(data_config['todate'])]

        # Rename columns to standard names
        column_mapping = {
            data_config.get('open_col', 'open'): 'open',
            data_config.get('high_col', 'high'): 'high',
            data_config.get('low_col', 'low'): 'low',
            data_config.get('close_col', 'close'): 'close',
            data_config.get('volume_col', 'volume'): 'volume',
        }
        df = df.rename(columns=column_mapping)

        self.data = df
        _logger.info("Loaded %d rows of data", len(df))

        return df

    def analyze_entry_conditions(self) -> Dict[str, Any]:
        """
        Analyze entry conditions to understand why signals aren't triggering.

        Returns:
            Dictionary with analysis results
        """
        if self.data is None:
            self.load_data()

        strategy_params = self.config['strategy']['parameters']
        entry_logic = strategy_params['entry_logic']

        print("\n" + "="*80)
        print("ENTRY CONDITION ANALYSIS")
        print("="*80)

        print(f"\nEntry Mixin: {entry_logic['name']}")
        print(f"Parameters: {json.dumps(entry_logic['params'], indent=2)}")

        # Get entry mixin class
        mixin_class = ENTRY_MIXIN_REGISTRY.get(entry_logic['name'])
        if not mixin_class:
            print(f"\n✗ Entry mixin '{entry_logic['name']}' not found!")
            return {'error': 'Mixin not found'}

        # Analyze based on mixin type
        analysis = {'mixin': entry_logic['name'], 'issues': [], 'suggestions': []}

        if 'RSI' in entry_logic['name']:
            analysis.update(self._analyze_rsi_conditions(entry_logic['params']))

        if 'BB' in entry_logic['name'] or 'Bollinger' in entry_logic['name']:
            analysis.update(self._analyze_bb_conditions(entry_logic['params']))

        if 'Volume' in entry_logic['name']:
            analysis.update(self._analyze_volume_conditions(entry_logic['params']))

        if 'SuperTrend' in entry_logic['name']:
            analysis.update(self._analyze_supertrend_conditions(entry_logic['params']))

        # Print summary
        print("\n" + "-"*80)
        print("ANALYSIS SUMMARY")
        print("-"*80)

        if analysis.get('issues'):
            print("\nPotential Issues:")
            for issue in analysis['issues']:
                print(f"  ✗ {issue}")

        if analysis.get('suggestions'):
            print("\nSuggestions:")
            for suggestion in analysis['suggestions']:
                print(f"  → {suggestion}")

        return analysis

    def _analyze_rsi_conditions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze RSI-related conditions."""
        df = self.data.copy()

        # Calculate RSI
        rsi_period = params.get('e_rsi_period') or params.get('rsi_period', 14)
        rsi_oversold = params.get('e_rsi_oversold') or params.get('rsi_oversold', 30)
        rsi_overbought = params.get('e_rsi_overbought') or params.get('rsi_overbought', 70)

        # Simple RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Count signals
        oversold_count = (rsi <= rsi_oversold).sum()

        print(f"\nRSI Analysis:")
        print(f"  Period: {rsi_period}")
        print(f"  Oversold threshold: {rsi_oversold}")
        print(f"  Current RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
        print(f"  Times RSI was oversold (<= {rsi_oversold}): {oversold_count}")
        print(f"  Percentage of bars: {oversold_count/len(df)*100:.2f}%")

        issues = []
        suggestions = []

        if oversold_count == 0:
            issues.append(f"RSI never reached oversold level ({rsi_oversold})")
            suggestions.append(f"Increase RSI oversold threshold (try {int(rsi.quantile(0.1))})")
        elif oversold_count < 10:
            issues.append(f"RSI rarely oversold (only {oversold_count} times)")
            suggestions.append(f"Consider raising threshold to {int(rsi.quantile(0.2))}")

        return {'issues': issues, 'suggestions': suggestions}

    def _analyze_bb_conditions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Bollinger Bands conditions."""
        df = self.data.copy()

        bb_period = params.get('e_bb_period') or params.get('bb_period', 20)
        bb_dev = params.get('e_bb_dev') or params.get('bb_dev', 2.0)

        # Calculate BB
        sma = df['close'].rolling(window=bb_period).mean()
        std = df['close'].rolling(window=bb_period).std()
        bb_upper = sma + (std * bb_dev)
        bb_lower = sma - (std * bb_dev)

        # Count touches
        lower_touches = (df['close'] <= bb_lower).sum()

        print(f"\nBollinger Bands Analysis:")
        print(f"  Period: {bb_period}, Deviation: {bb_dev}")
        print(f"  Lower band touches: {lower_touches}")
        print(f"  Percentage of bars: {lower_touches/len(df)*100:.2f}%")

        issues = []
        suggestions = []

        if lower_touches == 0:
            issues.append(f"Price never touched lower BB (dev={bb_dev})")
            suggestions.append(f"Reduce BB deviation to 1.5 or use different entry logic")
        elif lower_touches < 5:
            issues.append(f"Rare BB lower touches ({lower_touches})")
            suggestions.append(f"Consider reducing deviation or period")

        return {'issues': issues, 'suggestions': suggestions}

    def _analyze_volume_conditions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume conditions."""
        df = self.data.copy()

        vol_ma_period = params.get('e_vol_ma_period') or params.get('volume_ma_period', 20)
        vol_threshold = params.get('e_min_volume_ratio') or params.get('volume_threshold', 1.5)

        # Calculate volume MA
        vol_ma = df['volume'].rolling(window=vol_ma_period).mean()
        vol_ratio = df['volume'] / vol_ma

        # Count high volume bars
        high_vol_count = (vol_ratio >= vol_threshold).sum()

        print(f"\nVolume Analysis:")
        print(f"  MA Period: {vol_ma_period}")
        print(f"  Volume threshold: {vol_threshold}x")
        print(f"  High volume bars: {high_vol_count}")
        print(f"  Percentage of bars: {high_vol_count/len(df)*100:.2f}%")
        print(f"  Typical volume ratio range: {vol_ratio.quantile(0.1):.2f} - {vol_ratio.quantile(0.9):.2f}")

        issues = []
        suggestions = []

        if high_vol_count < 10:
            issues.append(f"Volume rarely exceeds {vol_threshold}x average")
            suggestions.append(f"Reduce threshold to {vol_ratio.quantile(0.7):.2f}x")

        return {'issues': issues, 'suggestions': suggestions}

    def _analyze_supertrend_conditions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SuperTrend conditions."""
        print(f"\nSuperTrend Analysis:")
        print(f"  Period: {params.get('supertrend_period', 10)}")
        print(f"  Multiplier: {params.get('supertrend_multiplier', 3.0)}")
        print(f"  Note: SuperTrend calculation requires ATR - check if trending conditions exist")

        return {'issues': [], 'suggestions': []}

    def suggest_parameter_adjustments(self) -> Dict[str, Any]:
        """
        Suggest parameter adjustments based on data analysis.

        Returns:
            Dictionary with suggested parameters
        """
        if self.data is None:
            self.load_data()

        print("\n" + "="*80)
        print("PARAMETER SUGGESTIONS")
        print("="*80)

        strategy_params = self.config['strategy']['parameters']
        entry_logic = strategy_params['entry_logic']

        suggested_params = entry_logic['params'].copy()

        # Adjust RSI if present
        if 'rsi' in entry_logic['name'].lower():
            df = self.data.copy()
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Suggest oversold level at 20th percentile
            suggested_oversold = int(rsi.quantile(0.20))

            for key in ['e_rsi_oversold', 'rsi_oversold']:
                if key in suggested_params:
                    print(f"  RSI Oversold: {suggested_params[key]} → {suggested_oversold}")
                    suggested_params[key] = suggested_oversold

        # Adjust volume if present
        if 'volume' in entry_logic['name'].lower():
            df = self.data.copy()
            vol_ma = df['volume'].rolling(window=20).mean()
            vol_ratio = df['volume'] / vol_ma

            suggested_vol_threshold = round(vol_ratio.quantile(0.60), 2)

            for key in ['e_min_volume_ratio', 'volume_threshold']:
                if key in suggested_params:
                    print(f"  Volume Threshold: {suggested_params[key]} → {suggested_vol_threshold}")
                    suggested_params[key] = suggested_vol_threshold

        print("\nSuggested Configuration:")
        print(json.dumps(suggested_params, indent=2))

        return suggested_params

    def generate_debug_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive debug report.

        Args:
            output_path: Optional path to save report

        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("BACKTEST DEBUG REPORT")
        report_lines.append("="*80)
        report_lines.append("")

        # Configuration info
        report_lines.append(f"Config File: {self.config_path}")
        report_lines.append(f"Test Name: {self.config.get('test_name', 'N/A')}")
        report_lines.append("")

        # Load data
        if self.data is None:
            self.load_data()

        report_lines.append(f"Data Summary:")
        report_lines.append(f"  Rows: {len(self.data)}")
        report_lines.append(f"  Date Range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
        report_lines.append(f"  Close Price Range: ${self.data['close'].min():.2f} - ${self.data['close'].max():.2f}")
        report_lines.append("")

        # Run analysis
        entry_analysis = self.analyze_entry_conditions()

        report = "\n".join(report_lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nDebug report saved to: {output_path}")

        return report


# =============================================================================
# DEBUG CONFIGURATION
# Set these parameters when running from IDE debugger
# =============================================================================
DEBUG_MODE = True  # Set to True to use DEBUG_CONFIG, False for CLI mode
DEBUG_CONFIG = {
    # Path to your config file (relative to project root or absolute)
    'config_path': 'config/backtester/custom_strategy_test.json',

    # Suggest parameter adjustments? (True/False)
    'suggest_adjustments': True,

    # Save debug report to file? (None or path string)
    'report_path': 'results/debug_report.txt',  # Set to 'results/debug_report.txt' to save report
}
# =============================================================================


def main():
    """CLI interface for debugger - supports both CLI and debug mode."""

    # =========================================================================
    # DEBUG MODE: Run with parameters set at the top of the file
    # =========================================================================
    if DEBUG_MODE:
        print("=" * 80)
        print("RUNNING DEBUGGER IN DEBUG MODE")
        print("=" * 80)
        print(f"Config: {DEBUG_CONFIG['config_path']}")
        print(f"Suggest Adjustments: {DEBUG_CONFIG['suggest_adjustments']}")
        print(f"Report Path: {DEBUG_CONFIG.get('report_path', 'None')}")
        print()
        print("Tip: Set DEBUG_MODE = False at the top to use CLI mode")
        print("=" * 80)

        config_path = DEBUG_CONFIG['config_path']

        # Handle relative paths from project root
        if not Path(config_path).is_absolute():
            config_path = str(project_root / config_path)

        debugger = BacktestDebugger(config_path)

        # Run analysis
        debugger.analyze_entry_conditions()

        # Suggest adjustments if requested
        if DEBUG_CONFIG['suggest_adjustments']:
            debugger.suggest_parameter_adjustments()

        # Generate report if requested
        report_path = DEBUG_CONFIG.get('report_path')
        if report_path and isinstance(report_path, str):
            debugger.generate_debug_report(report_path)

        return

    # =========================================================================
    # CLI MODE: Parse command line arguments
    # =========================================================================
    import argparse

    parser = argparse.ArgumentParser(
        description='Debug backtests with no trades',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/backtester/tests/backtest_debugger.py config/backtester/your_config.json
  python src/backtester/tests/backtest_debugger.py config/backtester/your_config.json --suggest
  python src/backtester/tests/backtest_debugger.py config/backtester/your_config.json --suggest --report results/debug.txt

Debug Mode:
  Set DEBUG_MODE = True and DEBUG_CONFIG at the top of the file, then run from IDE
        """
    )
    parser.add_argument('config', help='Path to JSON configuration file')
    parser.add_argument('--suggest', action='store_true', help='Suggest parameter adjustments')
    parser.add_argument('--report', help='Save debug report to file')

    args = parser.parse_args()

    debugger = BacktestDebugger(args.config)

    # Run analysis
    debugger.analyze_entry_conditions()

    # Suggest adjustments if requested
    if args.suggest:
        debugger.suggest_parameter_adjustments()

    # Generate report if requested
    if args.report:
        debugger.generate_debug_report(args.report)


if __name__ == "__main__":
    main()
