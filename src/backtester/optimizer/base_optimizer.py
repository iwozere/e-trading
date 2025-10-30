"""
Base Optimizer for Trading Strategy Backtesting and Optimization.

This module provides a base class that all strategy-specific optimizers should inherit from.
It contains common functionality for data handling, result analysis, file operations,
and optimization workflows.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import numpy as np
import pandas as pd
import backtrader as bt
import optuna
import json
import pickle
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
from src.util.config import load_config

warnings.filterwarnings('ignore')

_logger = setup_logger(__name__)


class BaseOptimizer:
    """
    Base class for all trading strategy optimizers.

    This class provides common functionality that all strategy-specific optimizers
    should inherit from, including:
    - Configuration loading and validation
    - Data preparation and validation
    - Result analysis and metrics calculation
    - File operations (saving/loading results)
    - Common optimization workflows
    """

    def __init__(self, config_path: str):
        """
        Initialize the base optimizer.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Set up directories
        self.data_dir = Path(self.config.get('data', {}).get('data_dir', 'data/raw'))
        self.output_dir = Path(self.config.get('output', {}).get('results_dir', 'results'))

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Strategy parameters
        self.strategy_params = self.config.get('strategy', {})
        self.optimization_params = self.config.get('optimization', {})

        # Trading parameters
        self.initial_cash = self.config.get('trading_parameters', {}).get('initial_cash', 100000)
        self.commission = self.config.get('trading_parameters', {}).get('commission', 0.001)

        _logger.info("Base optimizer initialized with config: %s", config_path)

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.

        Returns:
            Configuration dictionary
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            config = load_config(str(self.config_path))
            _logger.info("Loaded configuration from %s", self.config_path)
            return config

        except Exception as e:
            _logger.exception("Error loading configuration:")
            raise

    def _prepare_data(self, data_file: Path) -> pd.DataFrame:
        """
        Prepare data for backtesting.

        Args:
            data_file: Path to data file

        Returns:
            Prepared DataFrame for backtesting
        """
        try:
            # Load data file
            df = pd.read_csv(data_file)

            # Ensure datetime column exists
            if 'timestamp' in df.columns:
                datetime_col = 'timestamp'
            elif 'datetime' in df.columns:
                datetime_col = 'datetime'
            else:
                # Assume first column is datetime
                datetime_col = df.columns[0]

            # Convert to datetime and set as index
            df['datetime'] = pd.to_datetime(df[datetime_col], utc=True)
            df.set_index('datetime', inplace=True)

            # Ensure the index is timezone-naive for Backtrader compatibility
            df.index = df.index.tz_localize(None)

            # Ensure the index is pandas datetime, not numpy float64
            df.index = pd.to_datetime(df.index)

            # Filter by date range if specified
            if 'start_date' in self.config.get('data', {}):
                start_date = pd.to_datetime(self.config['data']['start_date'], utc=True)
                df = df[df.index >= start_date]

            if 'end_date' in self.config.get('data', {}):
                end_date = pd.to_datetime(self.config['data']['end_date'], utc=True)
                df = df[df.index <= end_date]

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in data")

            # Remove any rows with NaN values
            df = df.dropna()

            if len(df) < 50:
                raise ValueError(f"Insufficient data: {len(df)} rows (minimum 50 required)")

            _logger.info("Prepared data: %s rows from %s to %s",
                        len(df), df.index[0], df.index[-1])

            # Debug: Check index type
            _logger.debug("Index type: %s, dtype: %s", type(df.index), df.index.dtype)

            return df

        except Exception as e:
            _logger.exception("Error preparing data:")
            raise

    def _create_backtrader_engine(self, df: pd.DataFrame, strategy_class, strategy_config: Dict[str, Any]) -> bt.Cerebro:
        """
        Create Backtrader engine with data and strategy.

        Args:
            df: Prepared DataFrame
            strategy_class: Strategy class to use
            strategy_config: Strategy configuration

        Returns:
            Configured Backtrader Cerebro instance
        """
        try:
            # Create Backtrader engine
            cerebro = bt.Cerebro()

            # Prepare data feed with robust datetime handling
            # Reset index to make datetime a column instead of index to avoid Backtrader issues
            df_copy = df.copy(deep=True)
            df_copy = df_copy.reset_index()
            df_copy = df_copy.rename(columns={'datetime': 'timestamp'})
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])

            # Add data feed
            data = bt.feeds.PandasData(
                dataname=df_copy,
                datetime=0,  # 0 indicates datetime is in column 0 (timestamp)
                open=1,      # open is now column 1
                high=2,      # high is now column 2
                low=3,       # low is now column 3
                close=4,     # close is now column 4
                volume=5,    # volume is now column 5
                openinterest=None,
                fromdate=df.index[0],
                todate=df.index[-1]
            )
            cerebro.adddata(data)

            # Add strategy
            cerebro.addstrategy(strategy_class, strategy_config=strategy_config)

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

            # Set initial cash
            cerebro.broker.setcash(self.initial_cash)

            # Set commission
            cerebro.broker.setcommission(commission=self.commission)

            return cerebro

        except Exception as e:
            _logger.exception("Error creating Backtrader engine:")
            raise

    def _extract_backtest_results(self, cerebro: bt.Cerebro, strategy) -> Dict[str, Any]:
        """
        Extract results from Backtrader backtest.

        Args:
            cerebro: Backtrader Cerebro instance
            strategy: Strategy instance

        Returns:
            Dictionary with backtest results
        """
        try:
            # Extract results
            sharpe_ratio = strategy.analyzers.sharpe.get_analysis()
            drawdown = strategy.analyzers.drawdown.get_analysis()
            returns = strategy.analyzers.returns.get_analysis()
            trades = strategy.analyzers.trades.get_analysis()

            # Calculate metrics
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash
            max_drawdown = drawdown.get('max', {}).get('drawdown', 0) / 100

            # Extract trade statistics
            total_trades = trades.get('total', {}).get('total', 0)
            won_trades = trades.get('won', {}).get('total', 0)
            lost_trades = trades.get('lost', {}).get('total', 0)
            win_rate = won_trades / total_trades if total_trades > 0 else 0

            # Calculate Sharpe ratio
            sharpe = sharpe_ratio.get('sharperatio', 0)
            if sharpe is None:
                sharpe = 0

            # Compile results
            backtest_results = {
                'initial_cash': self.initial_cash,
                'final_value': final_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'won_trades': won_trades,
                'lost_trades': lost_trades,
                'win_rate': win_rate,
                'equity_curve': getattr(strategy, 'equity_curve', []),
                'equity_dates': getattr(strategy, 'equity_dates', [])
            }

            return backtest_results

        except Exception as e:
            _logger.exception("Error extracting backtest results:")
            raise

    def _create_optuna_study(self, direction: str = 'minimize') -> optuna.Study:
        """
        Create Optuna study for optimization.

        Args:
            direction: Optimization direction ('minimize' or 'maximize')

        Returns:
            Optuna Study instance
        """
        try:
            study = optuna.create_study(
                direction=direction,
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            return study

        except Exception as e:
            _logger.exception("Error creating Optuna study:")
            raise

    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate overall summary of results.

        Args:
            results: List of backtest/optimization results

        Returns:
            Dictionary with summary statistics
        """
        try:
            if not results:
                return {}

            # Extract key metrics
            returns = [r.get('total_return', 0) for r in results]
            sharpes = [r.get('sharpe_ratio', 0) for r in results]
            drawdowns = [r.get('max_drawdown', 0) for r in results]
            trade_counts = [r.get('total_trades', 0) for r in results]
            win_rates = [r.get('win_rate', 0) for r in results]

            # Calculate statistics
            summary = {
                'total_combinations': len(results),
                'successful_combinations': len([r for r in returns if not np.isnan(r)]),
                'average_return': np.mean(returns),
                'median_return': np.median(returns),
                'best_return': np.max(returns),
                'worst_return': np.min(returns),
                'return_std': np.std(returns),
                'average_sharpe': np.mean(sharpes),
                'median_sharpe': np.median(sharpes),
                'best_sharpe': np.max(sharpes),
                'worst_sharpe': np.min(sharpes),
                'average_drawdown': np.mean(drawdowns),
                'median_drawdown': np.median(drawdowns),
                'worst_drawdown': np.max(drawdowns),
                'average_trades': np.mean(trade_counts),
                'average_win_rate': np.mean(win_rates),
                'profitable_combinations': len([r for r in returns if r > 0]),
                'profitable_rate': len([r for r in returns if r > 0]) / len(returns),
                'results': results
            }

            return summary

        except Exception as e:
            _logger.exception("Error generating summary:")
            return {}

    def _save_results(self, results: List[Dict[str, Any]], summary: Dict[str, Any], prefix: str = "results"):
        """
        Save results to files.

        Args:
            results: List of backtest/optimization results
            summary: Overall summary
            prefix: Prefix for file names
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Save detailed results
            results_file = self.output_dir / f"{prefix}_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Save summary
            summary_file = self.output_dir / f"{prefix}_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            # Generate performance report
            self._generate_performance_report(results, summary, timestamp, prefix)

            _logger.info("Results saved to %s", self.output_dir)

        except Exception as e:
            _logger.exception("Error saving results:")

    def _generate_performance_report(self, results: List[Dict[str, Any]], summary: Dict[str, Any],
                                   timestamp: str, prefix: str = "results"):
        """
        Generate performance report.

        Args:
            results: List of backtest/optimization results
            summary: Overall summary
            timestamp: Timestamp for file naming
            prefix: Prefix for file names
        """
        try:
            report_file = self.output_dir / f"{prefix}_performance_report_{timestamp}.txt"

            with open(report_file, 'w') as f:
                f.write(f"{prefix.replace('_', ' ').title()} Backtesting and Optimization Report\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Combinations: {summary.get('total_combinations', 0)}\n")
                f.write(f"Successful Combinations: {summary.get('successful_combinations', 0)}\n\n")

                f.write("Overall Performance Summary:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Return: {summary.get('average_return', 0):.2%}\n")
                f.write(f"Median Return: {summary.get('median_return', 0):.2%}\n")
                f.write(f"Best Return: {summary.get('best_return', 0):.2%}\n")
                f.write(f"Worst Return: {summary.get('worst_return', 0):.2%}\n")
                f.write(f"Return Std Dev: {summary.get('return_std', 0):.2%}\n\n")

                f.write(f"Average Sharpe Ratio: {summary.get('average_sharpe', 0):.3f}\n")
                f.write(f"Median Sharpe Ratio: {summary.get('median_sharpe', 0):.3f}\n")
                f.write(f"Best Sharpe Ratio: {summary.get('best_sharpe', 0):.3f}\n")
                f.write(f"Worst Sharpe Ratio: {summary.get('worst_sharpe', 0):.3f}\n\n")

                f.write(f"Average Max Drawdown: {summary.get('average_drawdown', 0):.2%}\n")
                f.write(f"Median Max Drawdown: {summary.get('median_drawdown', 0):.2%}\n")
                f.write(f"Worst Max Drawdown: {summary.get('worst_drawdown', 0):.2%}\n\n")

                f.write(f"Average Trades: {summary.get('average_trades', 0):.1f}\n")
                f.write(f"Average Win Rate: {summary.get('average_win_rate', 0):.2%}\n")
                f.write(f"Profitable Combinations: {summary.get('profitable_combinations', 0)}/{summary.get('total_combinations', 0)} ({summary.get('profitable_rate', 0):.2%})\n\n")

            _logger.info("Performance report generated: %s", report_file)

        except Exception as e:
            _logger.exception("Error generating performance report:")

    def discover_available_combinations(self) -> List[Dict[str, str]]:
        """
        Discover available data and model combinations.

        This method should be overridden by subclasses to implement
        strategy-specific model discovery logic.

        Returns:
            List of dictionaries with symbol, timeframe, and model information
        """
        raise NotImplementedError("Subclasses must implement discover_available_combinations")

    def run_backtest(self, combination: Dict[str, str], strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest for a specific combination.

        This method should be overridden by subclasses to implement
        strategy-specific backtesting logic.

        Args:
            combination: Dictionary with symbol, timeframe, and model information
            strategy_params: Strategy parameters for this backtest

        Returns:
            Dictionary with backtest results
        """
        raise NotImplementedError("Subclasses must implement run_backtest")

    def optimize_parameters(self, combination: Dict[str, str], n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Optuna.

        This method should be overridden by subclasses to implement
        strategy-specific optimization logic.

        Args:
            combination: Dictionary with symbol, timeframe, and model information
            n_trials: Number of optimization trials

        Returns:
            Dictionary with optimization results
        """
        raise NotImplementedError("Subclasses must implement optimize_parameters")

    def run(self, optimize: bool = False, n_trials: int = 100) -> Dict[str, Any]:
        """
        Run backtesting and optimization for all available combinations.

        This method provides a common workflow that subclasses can use
        or override if needed.

        Args:
            optimize: Whether to run parameter optimization
            n_trials: Number of optimization trials per combination

        Returns:
            Dictionary with overall results
        """
        try:
            _logger.info("Starting backtesting and optimization")

            # Discover available combinations
            available_combinations = self.discover_available_combinations()
            _logger.info("Combinations to process: %d", len(available_combinations))

            all_results = []

            for i, combination in enumerate(available_combinations):
                try:
                    _logger.info("Processing combination %d/%d", i + 1, len(available_combinations))

                    if optimize:
                        # Run optimization
                        results = self.optimize_parameters(combination, n_trials)
                    else:
                        # Run backtest with default parameters
                        default_params = self.strategy_params.copy()
                        results = {
                            'final_results': self.run_backtest(combination, default_params),
                            'best_params': default_params
                        }

                    all_results.append({
                        'combination': combination,
                        'results': results
                    })

                except Exception as e:
                    _logger.exception("Error processing combination:")
                    continue

            # Generate overall summary
            summary = self._generate_summary(all_results)

            # Save results
            self._save_results(all_results, summary, self.__class__.__name__.lower())

            _logger.info("Backtesting and optimization completed")
            _logger.info("Processed combinations: %d", len(all_results))

            return summary

        except Exception as e:
            _logger.exception("Error in main run:")
            raise
