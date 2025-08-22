"""
HMM-LSTM Backtesting Optimizer

This module provides functionality to run backtesting for HMM-LSTM trading strategies.
It handles:
1. Loading trained HMM and LSTM models from the pipeline
2. Preparing data with regime labels and technical indicators
3. Running backtesting with the HMMLSTMStrategy
4. Saving results and generating performance reports
5. Optional parameter optimization using Optuna

Features:
- Integration with existing HMM-LSTM pipeline
- Support for multiple symbols and timeframes
- Regime-aware trading decisions
- Comprehensive performance analysis
- Risk management controls
"""

import os
import sys
import json
import yaml
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import optuna
import backtrader as bt
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.strategy.hmm_lstm_strategy import HMMLSTMStrategy
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class HMMLSTMOptimizer:
    """
    HMM-LSTM Backtesting Optimizer

    This class handles the complete backtesting process for HMM-LSTM strategies,
    including model loading, data preparation, strategy execution, and result analysis.
    """

    def __init__(self, config_path: str = "config/optimizer/p01_hmm_lstm.json"):
        """
        Initialize the HMM-LSTM optimizer.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.pipeline_dir = Path(self.config['ml_models']['pipeline_dir'])
        self.models_dir = Path(self.config['ml_models']['models_dir'])
        self.pipeline_config = self._load_pipeline_config()

        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _logger.info("Using device: %s", self.device)

    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def _load_pipeline_config(self) -> dict:
        """Load pipeline configuration."""
        pipeline_config_path = Path(self.config['ml_models']['config_file'])
        if not pipeline_config_path.exists():
            raise FileNotFoundError(f"Pipeline config not found: {pipeline_config_path}")

        with open(pipeline_config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def find_latest_models(self, symbol: str, timeframe: str) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Find the latest trained HMM and LSTM models for a symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Tuple of (hmm_model_path, lstm_model_path) or (None, None) if not found
        """
        # Find HMM model
        hmm_pattern = f"hmm_{symbol}_{timeframe}_*.pkl"
        hmm_files = list(self.models_dir.glob(hmm_pattern))
        hmm_model_path = sorted(hmm_files)[-1] if hmm_files else None

        # Find LSTM model
        lstm_pattern = f"lstm_{symbol}_{timeframe}_*.pkl"
        lstm_files = list(self.models_dir.glob(lstm_pattern))
        lstm_model_path = sorted(lstm_files)[-1] if lstm_files else None

        if hmm_model_path:
            _logger.info("Found HMM model: %s", hmm_model_path)
        else:
            _logger.warning("No HMM model found for %s %s", symbol, timeframe)

        if lstm_model_path:
            _logger.info("Found LSTM model: %s", lstm_model_path)
        else:
            _logger.warning("No LSTM model found for %s %s", symbol, timeframe)

        return hmm_model_path, lstm_model_path

    def load_models(self, hmm_path: Path, lstm_path: Path) -> Tuple[Dict, Dict]:
        """
        Load trained HMM and LSTM models.

        Args:
            hmm_path: Path to HMM model file
            lstm_path: Path to LSTM model file

        Returns:
            Tuple of (hmm_model_data, lstm_model_data)
        """
        # Load HMM model
        with open(hmm_path, 'rb') as f:
            hmm_data = pickle.load(f)

        # Load LSTM model
        with open(lstm_path, 'rb') as f:
            lstm_data = pickle.load(f)

        _logger.info("Loaded HMM model with %d components", hmm_data['config']['n_components'])
        _logger.info("Loaded LSTM model with input_size=%d, hidden_size=%d",
                    lstm_data['config']['input_size'], lstm_data['config']['hidden_size'])

        return hmm_data, lstm_data

    def prepare_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Prepare OHLCV data for backtesting.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            DataFrame with OHLCV data
        """
        # Load raw OHLCV data
        data_dir = Path(self.config['data']['data_dir'])
        data_file = data_dir / f"{symbol}_{timeframe}.csv"

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # Ensure we have required OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter by date range if specified
        if 'start_date' in self.config['data']:
            start_date = pd.to_datetime(self.config['data']['start_date'])
            df = df[df.index >= start_date]

        if 'end_date' in self.config['data']:
            end_date = pd.to_datetime(self.config['data']['end_date'])
            df = df[df.index <= end_date]

        _logger.info("Loaded data: %s rows, %s columns", len(df), len(df.columns))
        return df

    def run_backtest(self, symbol: str, timeframe: str,
                    hmm_data: Dict, lstm_data: Dict) -> Dict:
        """
        Run backtesting for a single symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            hmm_data: Loaded HMM model data
            lstm_data: Loaded LSTM model data

        Returns:
            Dictionary containing backtest results
        """
        # Prepare data
        df = self.prepare_data(symbol, timeframe)

        # Create Backtrader engine
        cerebro = bt.Cerebro()

        # Add data feed
        data = bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        cerebro.adddata(data)

        # Add strategy
        strategy_params = self.config['strategy']
        cerebro.addstrategy(
            HMMLSTMStrategy,
            hmm_model=hmm_data['model'],
            hmm_scaler=hmm_data['scaler'],
            hmm_features=hmm_data['features'],
            lstm_model=lstm_data['model'],
            lstm_scalers=lstm_data['scalers'],
            lstm_features=lstm_data['features'],
            sequence_length=lstm_data['config']['sequence_length'],
            prediction_threshold=strategy_params['entry_threshold'],
            regime_confidence_threshold=strategy_params['regime_confidence_threshold'],
            profit_target=self.config['risk_management']['take_profit_pct'],
            stop_loss=self.config['risk_management']['stop_loss_pct'],
            trailing_stop=self.config['risk_management'].get('trailing_stop', 0.005)
        )

        # Set initial capital
        cerebro.broker.setcash(self.config['initial_capital'])
        cerebro.broker.setcommission(commission=self.config['commission'])

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Run backtest
        _logger.info("Running backtest for %s %s", symbol, timeframe)
        results = cerebro.run()

        # Extract results
        strat = results[0]
        analyzers = strat.analyzers

        # Compile results
        backtest_results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'initial_capital': self.config['initial_capital'],
            'final_capital': cerebro.broker.getvalue(),
            'total_return': analyzers.returns.get_analysis()['rtot'],
            'annual_return': analyzers.returns.get_analysis()['rnorm100'],
            'sharpe_ratio': analyzers.sharpe.get_analysis()['sharperatio'],
            'max_drawdown': analyzers.drawdown.get_analysis()['max']['drawdown'],
            'trades': analyzers.trades.get_analysis(),
            'strategy_params': strategy_params,
            'timestamp': datetime.now().isoformat()
        }

        return backtest_results

    def optimize_parameters(self, symbol: str, timeframe: str,
                          hmm_data: Dict, lstm_data: Dict) -> Dict:
        """
        Optimize strategy parameters using Optuna.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            hmm_data: Loaded HMM model data
            lstm_data: Loaded LSTM model data

        Returns:
            Dictionary containing optimization results
        """
        if not self.config['optimization']['enabled']:
            return {}

        def objective(trial):
            # Get parameter ranges from config
            param_ranges = self.config['optimization']['parameter_ranges']

            # Suggest parameter values based on config
            suggested_params = {}
            for param_name in self.config['optimization']['optimize_params']:
                if param_name in param_ranges:
                    param_config = param_ranges[param_name]
                    if param_config['type'] == 'float':
                        suggested_params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['min'],
                            param_config['max']
                        )
                    elif param_config['type'] == 'int':
                        suggested_params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['min'],
                            param_config['max']
                        )
                    elif param_config['type'] == 'categorical':
                        suggested_params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )

            # Update config temporarily
            original_params = self.config['strategy'].copy()
            self.config['strategy'].update(suggested_params)

            try:
                # Run backtest with suggested parameters
                results = self.run_backtest(symbol, timeframe, hmm_data, lstm_data)

                # Return negative Sharpe ratio (Optuna minimizes)
                return -results['sharpe_ratio']

            except Exception as e:
                _logger.error("Optimization trial failed: %s", e)
                return float('inf')
            finally:
                # Restore original parameters
                self.config['strategy'] = original_params

        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.config['optimization']['n_trials'])

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        _logger.info("Optimization completed. Best Sharpe ratio: %.4f", -best_value)
        _logger.info("Best parameters: %s", best_params)

        return {
            'best_params': best_params,
            'best_sharpe': -best_value,
            'optimization_history': study.trials_dataframe()
        }

    def save_results(self, results: Dict, symbol: str, timeframe: str):
        """
        Save backtest results to file.

        Args:
            results: Backtest results dictionary
            symbol: Trading symbol
            timeframe: Timeframe
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hmm_lstm_{symbol}_{timeframe}_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert numpy types to native Python types for JSON serialization
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

        results_serializable = convert_numpy(results)

        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2, default=str)

        _logger.info("Results saved to %s", filepath)

    def generate_report(self, all_results: List[Dict]):
        """
        Generate comprehensive performance report.

        Args:
            all_results: List of backtest results for all symbol-timeframe combinations
        """
        if not all_results:
            return

        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Symbol': result['symbol'],
                'Timeframe': result['timeframe'],
                'Total Return (%)': result['total_return'] * 100,
                'Annual Return (%)': result['annual_return'],
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown (%)': result['max_drawdown'],
                'Final Capital': result['final_capital']
            })

        summary_df = pd.DataFrame(summary_data)

        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"hmm_lstm_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)

        # Generate plots
        self._generate_plots(all_results, summary_df)

        _logger.info("Report generated. Summary saved to %s", summary_file)

    def _generate_plots(self, all_results: List[Dict], summary_df: pd.DataFrame):
        """Generate performance plots."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HMM-LSTM Strategy Performance Summary', fontsize=16)

        # 1. Returns comparison
        symbols = summary_df['Symbol'].unique()
        x = np.arange(len(symbols))
        width = 0.35

        axes[0, 0].bar(x - width/2, summary_df['Total Return (%)'], width, label='Total Return')
        axes[0, 0].bar(x + width/2, summary_df['Annual Return (%)'], width, label='Annual Return')
        axes[0, 0].set_xlabel('Symbol')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].set_title('Returns Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(symbols)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Sharpe ratio
        axes[0, 1].bar(symbols, summary_df['Sharpe Ratio'])
        axes[0, 1].set_xlabel('Symbol')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_title('Sharpe Ratio by Symbol')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Max drawdown
        axes[1, 0].bar(symbols, summary_df['Max Drawdown (%)'])
        axes[1, 0].set_xlabel('Symbol')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].set_title('Maximum Drawdown by Symbol')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Risk-return scatter
        axes[1, 1].scatter(summary_df['Max Drawdown (%)'], summary_df['Total Return (%)'],
                          s=100, alpha=0.7)
        for i, symbol in enumerate(symbols):
            axes[1, 1].annotate(symbol,
                              (summary_df.iloc[i]['Max Drawdown (%)'],
                               summary_df.iloc[i]['Total Return (%)']),
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Max Drawdown (%)')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].set_title('Risk-Return Profile')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"hmm_lstm_performance_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        _logger.info("Performance plots saved to %s", plot_file)

    def discover_available_combinations(self) -> List[Tuple[str, str]]:
        """
        Discover available symbol-timeframe combinations that have both models and data.

        Returns:
            List of (symbol, timeframe) tuples that have both models and data
        """
        available_combinations = []
        data_dir = Path(self.config['data']['data_dir'])

        if not data_dir.exists():
            _logger.warning("Data directory not found: %s", data_dir)
            return available_combinations

        # Get all CSV files in data directory
        csv_files = list(data_dir.glob("*.csv"))
        _logger.info("Found %d CSV files in data directory", len(csv_files))

        for csv_file in csv_files:
            # Parse filename to extract symbol and timeframe
            # Expected format: {symbol}_{timeframe}.csv
            filename = csv_file.stem  # Remove .csv extension

            if '_' not in filename:
                _logger.warning("Skipping file with invalid format: %s (expected: symbol_timeframe.csv)", csv_file.name)
                continue

            # Split by last underscore to handle symbols with underscores
            parts = filename.rsplit('_', 1)
            if len(parts) != 2:
                _logger.warning("Skipping file with invalid format: %s (expected: symbol_timeframe.csv)", csv_file.name)
                continue

            symbol, timeframe = parts

            # Check if models exist for this combination
            hmm_path, lstm_path = self.find_latest_models(symbol, timeframe)

            if hmm_path and lstm_path:
                available_combinations.append((symbol, timeframe))
                _logger.info("Found complete setup for %s %s (data + models)", symbol, timeframe)
            else:
                _logger.warning("Skipping %s %s - models not found (data exists: %s)",
                              symbol, timeframe, csv_file.name)

        return available_combinations

    def run(self):
        """
        Run the complete HMM-LSTM backtesting process.
        """
        _logger.info("Starting HMM-LSTM backtesting process")

        all_results = []

        # Auto-discover available combinations from data directory
        available_combinations = self.discover_available_combinations()

        if not available_combinations:
            _logger.error("No valid symbol-timeframe combinations found with both models and data")
            _logger.info("Please ensure you have:")
            _logger.info("1. OHLCV files in %s with format: symbol_timeframe.csv", self.config['data']['data_dir'])
            _logger.info("2. Trained models in %s with format: hmm/lstm_symbol_timeframe_*.pkl", self.models_dir)
            return

        _logger.info("Discovered %d symbol-timeframe combinations from data directory", len(available_combinations))
        combinations_to_process = available_combinations

        for symbol, timeframe in combinations_to_process:
            try:
                _logger.info("Processing %s %s", symbol, timeframe)

                # Find and load models
                hmm_path, lstm_path = self.find_latest_models(symbol, timeframe)
                if not hmm_path or not lstm_path:
                    _logger.warning("Skipping %s %s - models not found", symbol, timeframe)
                    continue

                hmm_data, lstm_data = self.load_models(hmm_path, lstm_path)

                # Run optimization if enabled
                if self.config['optimization']['enabled']:
                    opt_results = self.optimize_parameters(symbol, timeframe, hmm_data, lstm_data)
                    if opt_results:
                        # Update strategy parameters with best values
                        best_params = opt_results['best_params']
                        self.config['strategy'].update(best_params)

                # Run backtest
                results = self.run_backtest(symbol, timeframe, hmm_data, lstm_data)

                # Save individual results
                self.save_results(results, symbol, timeframe)
                all_results.append(results)

                _logger.info("Completed %s %s - Sharpe: %.4f, Return: %.2f%%",
                           symbol, timeframe, results['sharpe_ratio'],
                           results['total_return'] * 100)

            except Exception as e:
                _logger.error("Error processing %s %s: %s", symbol, timeframe, e)
                continue

        # Generate comprehensive report
        if all_results:
            self.generate_report(all_results)
            _logger.info("HMM-LSTM backtesting completed successfully. Processed %d combinations", len(all_results))
        else:
            _logger.warning("HMM-LSTM backtesting completed with no successful results")


def main():
    """Main entry point for HMM-LSTM backtesting."""
    import argparse

    parser = argparse.ArgumentParser(description='HMM-LSTM Backtesting Optimizer')
    parser.add_argument('--config', default='config/optimizer/p01_hmm_lstm.json',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = HMMLSTMOptimizer(args.config)

    # Run backtesting (automatically discovers all available combinations)
    optimizer.run()


if __name__ == "__main__":
    main()
