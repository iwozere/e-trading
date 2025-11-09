"""
Backtester Test Framework
-------------------------

This module provides a comprehensive framework for testing trading strategies
using JSON configuration files. It handles:
- Loading test configurations from JSON
- Setting up strategies with mixins
- Running backtests
- Validating results with assertions
- Generating test reports
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import backtrader as bt

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
from src.strategy.custom_strategy import CustomStrategy
from src.data.feed.file_data_feed import FileDataFeed

_logger = setup_logger(__name__)


def _safe_print(text: str) -> str:
    """
    Convert text with Unicode characters to console-safe version for Windows.

    Args:
        text: Text that may contain Unicode characters

    Returns:
        Console-safe text with ASCII alternatives
    """
    # Replace Unicode checkmarks with ASCII alternatives for console
    return text.replace('✓', '[PASS]').replace('✗', '[FAIL]')


class BacktesterTestFramework:
    """
    Framework for running backtests from JSON configuration files.

    This class provides methods to:
    - Load test configuration from JSON
    - Set up Backtrader with the specified strategy and data
    - Execute backtests
    - Validate results against assertions
    - Generate test reports
    """

    def __init__(self, config_path: str):
        """
        Initialize the backtester test framework.

        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cerebro = None
        self.results = None

        _logger.info("Initialized BacktesterTestFramework with config: %s", config_path)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load test configuration from JSON file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Validate required sections
            required_sections = ['strategy', 'data', 'broker']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in config")

            _logger.info("Loaded configuration: %s", config.get('test_name', 'Unnamed Test'))
            return config

        except json.JSONDecodeError as e:
            _logger.exception("Invalid JSON in config file: %s", e)
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            _logger.exception("Error loading configuration: %s", e)
            raise

    def _get_strategy_class(self, strategy_type: str):
        """
        Get strategy class by name.

        Args:
            strategy_type: Name of the strategy class

        Returns:
            Strategy class

        Raises:
            ValueError: If strategy type is not found
        """
        # Map of available strategies
        strategy_map = {
            'CustomStrategy': CustomStrategy,
            # Add more strategies as needed
            # 'HMMLSTMStrategy': HMMLSTMStrategy,
            # 'CNNXGBoostStrategy': CNNXGBoostStrategy,
        }

        if strategy_type not in strategy_map:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(strategy_map.keys())}"
            )

        return strategy_map[strategy_type]

    def _prepare_data_feed(self) -> FileDataFeed:
        """
        Prepare data feed from configuration.

        Returns:
            Configured FileDataFeed instance

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        data_config = self.config['data']
        file_path = Path(data_config['file_path'])

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Prepare data feed parameters
        feed_params = {
            'dataname': str(file_path),
            'symbol': data_config.get('symbol', 'UNKNOWN'),
            'datetime_col': data_config.get('datetime_col', 'datetime'),
            'open_col': data_config.get('open_col', 'open'),
            'high_col': data_config.get('high_col', 'high'),
            'low_col': data_config.get('low_col', 'low'),
            'close_col': data_config.get('close_col', 'close'),
            'volume_col': data_config.get('volume_col', 'volume'),
        }

        # Add optional date filters
        if data_config.get('fromdate'):
            feed_params['fromdate'] = data_config['fromdate']
        if data_config.get('todate'):
            feed_params['todate'] = data_config['todate']

        _logger.info("Preparing data feed from: %s", file_path)
        return FileDataFeed(**feed_params)

    def setup_backtest(self):
        """
        Set up Backtrader cerebro with strategy, data, and broker settings.
        """
        try:
            # Create Backtrader engine
            self.cerebro = bt.Cerebro()

            # Add data feed
            data_feed = self._prepare_data_feed()
            self.cerebro.adddata(data_feed)

            # Get strategy class
            strategy_config = self.config['strategy']
            strategy_class = self._get_strategy_class(strategy_config['type'])

            # Add strategy with parameters
            strategy_params = strategy_config.get('parameters', {})
            self.cerebro.addstrategy(
                strategy_class,
                strategy_config=strategy_params
            )

            # Configure broker
            broker_config = self.config['broker']
            self.cerebro.broker.setcash(broker_config.get('cash', 10000.0))
            self.cerebro.broker.setcommission(commission=broker_config.get('commission', 0.001))

            # Add slippage if specified
            if 'slippage' in broker_config:
                # Note: Backtrader slippage requires more complex setup
                # This is a simplified version
                pass

            # Add analyzers based on configuration
            analyzer_config = self.config.get('analyzers', {})

            if analyzer_config.get('sharpe_ratio', {}).get('enabled', True):
                sharpe_params = analyzer_config.get('sharpe_ratio', {}).get('params', {})
                self.cerebro.addanalyzer(
                    bt.analyzers.SharpeRatio,
                    _name='sharpe',
                    timeframe=bt.TimeFrame.Days if sharpe_params.get('timeframe') == 'annual' else bt.TimeFrame.Days,
                    riskfreerate=sharpe_params.get('riskfreerate', 0.0)
                )

            if analyzer_config.get('drawdown', {}).get('enabled', True):
                self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

            if analyzer_config.get('trades', {}).get('enabled', True):
                self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

            if analyzer_config.get('returns', {}).get('enabled', True):
                self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

            _logger.info("Backtest setup completed successfully")

        except Exception as e:
            _logger.exception("Error setting up backtest: %s", e)
            raise

    def run_backtest(self) -> Dict[str, Any]:
        """
        Execute the backtest and collect results.

        Returns:
            Dictionary with backtest results

        Raises:
            RuntimeError: If backtest execution fails
        """
        try:
            if self.cerebro is None:
                raise RuntimeError("Backtest not set up. Call setup_backtest() first.")

            _logger.info("Starting backtest execution...")
            initial_value = self.cerebro.broker.getvalue()

            # Run backtest
            strategies = self.cerebro.run()
            strategy = strategies[0]

            # Get final value
            final_value = self.cerebro.broker.getvalue()

            # Extract analyzer results
            results = {
                'test_name': self.config.get('test_name', 'Unnamed Test'),
                'description': self.config.get('description', ''),
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': (final_value - initial_value) / initial_value,
                'total_pnl': final_value - initial_value,
            }

            # Extract Sharpe ratio
            if hasattr(strategy.analyzers, 'sharpe'):
                sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
                results['sharpe_ratio'] = sharpe_analysis.get('sharperatio', None)

            # Extract drawdown
            if hasattr(strategy.analyzers, 'drawdown'):
                dd_analysis = strategy.analyzers.drawdown.get_analysis()
                results['max_drawdown'] = dd_analysis.get('max', {}).get('drawdown', 0) / 100
                results['max_drawdown_period'] = dd_analysis.get('max', {}).get('len', 0)

            # Extract trade statistics
            if hasattr(strategy.analyzers, 'trades'):
                trade_analysis = strategy.analyzers.trades.get_analysis()
                results['total_trades'] = trade_analysis.get('total', {}).get('total', 0)
                results['won_trades'] = trade_analysis.get('won', {}).get('total', 0)
                results['lost_trades'] = trade_analysis.get('lost', {}).get('total', 0)
                results['win_rate'] = (
                    results['won_trades'] / results['total_trades']
                    if results['total_trades'] > 0 else 0
                )

                # Average PnL per trade
                won_total = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
                lost_total = trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0)
                results['avg_win'] = (
                    won_total / results['won_trades']
                    if results['won_trades'] > 0 else 0
                )
                results['avg_loss'] = (
                    lost_total / results['lost_trades']
                    if results['lost_trades'] > 0 else 0
                )

                # Profit factor
                if abs(lost_total) > 0:
                    results['profit_factor'] = abs(won_total / lost_total)
                else:
                    results['profit_factor'] = float('inf') if won_total > 0 else 0

            # Extract returns
            if hasattr(strategy.analyzers, 'returns'):
                returns_analysis = strategy.analyzers.returns.get_analysis()
                results['avg_return'] = returns_analysis.get('ravg', None)
                results['rnorm'] = returns_analysis.get('rnorm', None)
                results['rnorm100'] = returns_analysis.get('rnorm100', None)

            self.results = results
            _logger.info("Backtest execution completed successfully")
            _logger.info("Final value: %.2f, Total return: %.2f%%",
                        final_value, results['total_return'] * 100)

            return results

        except Exception as e:
            _logger.exception("Error running backtest: %s", e)
            raise

    def validate_assertions(self) -> Dict[str, Any]:
        """
        Validate backtest results against configured assertions.

        Returns:
            Dictionary with validation results
        """
        if self.results is None:
            raise RuntimeError("No results to validate. Run backtest first.")

        assertions = self.config.get('assertions', {})
        if not assertions:
            _logger.info("No assertions configured, skipping validation")
            return {'passed': True, 'failures': []}

        validation_results = {
            'passed': True,
            'failures': [],
            'checks': []
        }

        try:
            # Check minimum trades
            if 'min_trades' in assertions and assertions['min_trades'] is not None:
                min_trades = assertions['min_trades']
                actual_trades = self.results.get('total_trades', 0)
                passed = actual_trades >= min_trades

                validation_results['checks'].append({
                    'name': 'min_trades',
                    'passed': passed,
                    'expected': f">= {min_trades}",
                    'actual': actual_trades
                })

                if not passed:
                    validation_results['passed'] = False
                    validation_results['failures'].append(
                        f"Minimum trades not met: {actual_trades} < {min_trades}"
                    )

            # Check maximum drawdown
            if 'max_drawdown_pct' in assertions and assertions['max_drawdown_pct'] is not None:
                max_dd_pct = assertions['max_drawdown_pct']
                actual_dd = self.results.get('max_drawdown', 0) * 100
                passed = actual_dd <= max_dd_pct

                validation_results['checks'].append({
                    'name': 'max_drawdown_pct',
                    'passed': passed,
                    'expected': f"<= {max_dd_pct}%",
                    'actual': f"{actual_dd:.2f}%"
                })

                if not passed:
                    validation_results['passed'] = False
                    validation_results['failures'].append(
                        f"Max drawdown exceeded: {actual_dd:.2f}% > {max_dd_pct}%"
                    )

            # Check minimum Sharpe ratio
            if 'min_sharpe_ratio' in assertions and assertions['min_sharpe_ratio'] is not None:
                min_sharpe = assertions['min_sharpe_ratio']
                actual_sharpe = self.results.get('sharpe_ratio', 0)
                if actual_sharpe is None:
                    actual_sharpe = 0
                passed = actual_sharpe >= min_sharpe

                validation_results['checks'].append({
                    'name': 'min_sharpe_ratio',
                    'passed': passed,
                    'expected': f">= {min_sharpe}",
                    'actual': f"{actual_sharpe:.3f}" if actual_sharpe else "N/A"
                })

                if not passed:
                    validation_results['passed'] = False
                    validation_results['failures'].append(
                        f"Minimum Sharpe ratio not met: {actual_sharpe:.3f} < {min_sharpe}"
                    )

            # Check if final value > initial value
            if assertions.get('final_value_greater_than_initial', False):
                passed = self.results['final_value'] > self.results['initial_value']

                validation_results['checks'].append({
                    'name': 'final_value_greater_than_initial',
                    'passed': passed,
                    'expected': f"> {self.results['initial_value']:.2f}",
                    'actual': f"{self.results['final_value']:.2f}"
                })

                if not passed:
                    validation_results['passed'] = False
                    validation_results['failures'].append(
                        f"Final value not greater than initial: "
                        f"{self.results['final_value']:.2f} <= {self.results['initial_value']:.2f}"
                    )

            if validation_results['passed']:
                _logger.info("All assertions passed!")
            else:
                _logger.warning("Assertion failures: %s", validation_results['failures'])

            return validation_results

        except Exception as e:
            _logger.exception("Error validating assertions: %s", e)
            return {
                'passed': False,
                'failures': [f"Validation error: {str(e)}"],
                'checks': []
            }

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a detailed test report.

        Args:
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        if self.results is None:
            raise RuntimeError("No results to report. Run backtest first.")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BACKTESTER TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Test information
        report_lines.append(f"Test Name: {self.results.get('test_name', 'N/A')}")
        report_lines.append(f"Description: {self.results.get('description', 'N/A')}")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Strategy configuration
        strategy_config = self.config['strategy']
        report_lines.append("Strategy Configuration:")
        report_lines.append(f"  Type: {strategy_config['type']}")
        if 'parameters' in strategy_config:
            params = strategy_config['parameters']
            if 'entry_logic' in params:
                report_lines.append(f"  Entry Logic: {params['entry_logic']['name']}")
                # Support both new (logic_params) and legacy (params) formats
                if 'logic_params' in params['entry_logic']:
                    report_lines.append(f"    Logic Params: {params['entry_logic']['logic_params']}")
                    if 'indicators' in params['entry_logic']:
                        report_lines.append(f"    Indicators: {len(params['entry_logic']['indicators'])} configured")
                elif 'params' in params['entry_logic']:
                    report_lines.append(f"    Params: {params['entry_logic']['params']}")
            if 'exit_logic' in params:
                report_lines.append(f"  Exit Logic: {params['exit_logic']['name']}")
                # Support both new (logic_params) and legacy (params) formats
                if 'logic_params' in params['exit_logic']:
                    report_lines.append(f"    Logic Params: {params['exit_logic']['logic_params']}")
                    if 'indicators' in params['exit_logic']:
                        report_lines.append(f"    Indicators: {len(params['exit_logic']['indicators'])} configured")
                elif 'params' in params['exit_logic']:
                    report_lines.append(f"    Params: {params['exit_logic']['params']}")
            report_lines.append(f"  Position Size: {params.get('position_size', 'N/A')}")
            report_lines.append(f"  Use TALib: {params.get('use_talib', 'N/A')}")
        report_lines.append("")

        # Performance metrics
        report_lines.append("Performance Metrics:")
        report_lines.append("-" * 80)
        report_lines.append(f"  Initial Value:    ${self.results['initial_value']:,.2f}")
        report_lines.append(f"  Final Value:      ${self.results['final_value']:,.2f}")
        report_lines.append(f"  Total P&L:        ${self.results['total_pnl']:,.2f}")
        report_lines.append(f"  Total Return:     {self.results['total_return']*100:.2f}%")
        report_lines.append("")

        report_lines.append(f"  Sharpe Ratio:     {self.results.get('sharpe_ratio', 'N/A')}")
        report_lines.append(f"  Max Drawdown:     {self.results.get('max_drawdown', 0)*100:.2f}%")
        report_lines.append(f"  Max DD Period:    {self.results.get('max_drawdown_period', 'N/A')} bars")
        report_lines.append("")

        # Trade statistics
        report_lines.append("Trade Statistics:")
        report_lines.append("-" * 80)
        report_lines.append(f"  Total Trades:     {self.results.get('total_trades', 0)}")
        report_lines.append(f"  Winning Trades:   {self.results.get('won_trades', 0)}")
        report_lines.append(f"  Losing Trades:    {self.results.get('lost_trades', 0)}")
        report_lines.append(f"  Win Rate:         {self.results.get('win_rate', 0)*100:.2f}%")
        report_lines.append(f"  Avg Win:          ${self.results.get('avg_win', 0):,.2f}")
        report_lines.append(f"  Avg Loss:         ${self.results.get('avg_loss', 0):,.2f}")
        report_lines.append(f"  Profit Factor:    {self.results.get('profit_factor', 0):.2f}")
        report_lines.append("")

        # Validation results
        if hasattr(self, '_validation_results'):
            validation = self._validation_results
            report_lines.append("Assertion Validation:")
            report_lines.append("-" * 80)
            report_lines.append(f"  Overall Status:   {'PASSED' if validation['passed'] else 'FAILED'}")

            if validation['checks']:
                report_lines.append("  Checks:")
                for check in validation['checks']:
                    status = "✓" if check['passed'] else "✗"
                    report_lines.append(
                        f"    {status} {check['name']}: "
                        f"Expected {check['expected']}, Got {check['actual']}"
                    )

            if validation['failures']:
                report_lines.append("  Failures:")
                for failure in validation['failures']:
                    report_lines.append(f"    - {failure}")

            report_lines.append("")

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            _logger.info("Report saved to: %s", output_path)

        return report

    def run_full_test(self, generate_report_file: bool = True) -> Dict[str, Any]:
        """
        Run complete test workflow: setup, execute, validate, report.

        Args:
            generate_report_file: Whether to save report to file

        Returns:
            Dictionary with complete test results
        """
        try:
            # Setup
            self.setup_backtest()

            # Execute
            results = self.run_backtest()

            # Validate
            validation = self.validate_assertions()
            self._validation_results = validation

            # Generate report
            if generate_report_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                test_name = self.config.get('test_name', 'test').replace(' ', '_').lower()
                report_path = Path(f"results/backtester_tests/{test_name}_{timestamp}.txt")
                report = self.generate_report(report_path)
            else:
                report = self.generate_report()

            return {
                'results': results,
                'validation': validation,
                'report': report,
                'success': validation['passed']
            }

        except Exception as e:
            _logger.exception("Error in full test execution: %s", e)
            raise


def run_backtest_from_config(config_path: str, generate_report: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run a backtest from a configuration file.

    Args:
        config_path: Path to JSON configuration file
        generate_report: Whether to generate and save a report

    Returns:
        Dictionary with test results
    """
    framework = BacktesterTestFramework(config_path)
    return framework.run_full_test(generate_report_file=generate_report)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run backtest from JSON config')
    parser.add_argument('config', help='Path to JSON configuration file')
    parser.add_argument('--no-report', action='store_true', help='Skip generating report file')

    args = parser.parse_args()

    try:
        results = run_backtest_from_config(args.config, generate_report=not args.no_report)

        print("\n" + results['report'])

        if results['success']:
            print("\n✓ Test PASSED")
            sys.exit(0)
        else:
            print("\n✗ Test FAILED")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Test ERROR: {e}")
        sys.exit(2)
