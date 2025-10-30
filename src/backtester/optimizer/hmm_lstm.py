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
- Comprehensive performance analysis with custom analyzers
- Risk management controls
- Consistent results format with custom_optimizer.py
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

from src.backtester.optimizer.base_optimizer import BaseOptimizer
from src.strategy.hmm_lstm_strategy import HMMLSTMStrategy
from src.backtester.analyzer.bt_analyzers import (CAGR, CalmarRatio,
                                       ConsecutiveWinsLosses,
                                       PortfolioVolatility, ProfitFactor,
                                       SortinoRatio, WinRate)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class HMMLSTMOptimizer(BaseOptimizer):
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
        super().__init__(config_path)

        self.pipeline_dir = Path(self.config['ml_models']['pipeline_dir'])
        self.models_dir = Path(self.config['ml_models']['models_dir'])
        self.pipeline_config = self._load_pipeline_config()

        # Initialize device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _logger.info("Using device: %s", self.device)



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
        # Find HMM model - pattern: hmm_{source}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.pkl
        hmm_dir = self.models_dir / "hmm"
        hmm_pattern = f"hmm_*_{symbol}_{timeframe}_*.pkl"
        hmm_files = list(hmm_dir.glob(hmm_pattern))
        _logger.debug("HMM pattern '%s' found %d files: %s", hmm_pattern, len(hmm_files), [f.name for f in hmm_files])
        hmm_model_path = sorted(hmm_files)[-1] if hmm_files else None

        # Find LSTM model - pattern: lstm_{symbol}_{timeframe}_{timestamp}.pkl
        lstm_dir = self.models_dir / "lstm"
        lstm_pattern = f"lstm_{symbol}_{timeframe}_*.pkl"
        lstm_files = list(lstm_dir.glob(lstm_pattern))
        _logger.debug("LSTM pattern '%s' found %d files: %s", lstm_pattern, len(lstm_files), [f.name for f in lstm_files])
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

        # Log HMM model info (handle different possible structures)
        if 'config' in hmm_data and 'n_components' in hmm_data['config']:
            _logger.info("Loaded HMM model with %d components", hmm_data['config']['n_components'])
        elif 'params' in hmm_data and 'n_components' in hmm_data['params']:
            _logger.info("Loaded HMM model with %d components", hmm_data['params']['n_components'])
        else:
            _logger.info("Loaded HMM model (structure: %s)", list(hmm_data.keys()))

        # Log LSTM model info (handle different possible structures)
        if 'config' in lstm_data and 'input_size' in lstm_data['config']:
            _logger.info("Loaded LSTM model with input_size=%d, hidden_size=%d",
                        lstm_data['config']['input_size'], lstm_data['config']['hidden_size'])
        elif 'model_architecture' in lstm_data:
            arch = lstm_data['model_architecture']
            _logger.info("Loaded LSTM model with input_size=%d, hidden_size=%d",
                        arch.get('input_size', 'unknown'), arch.get('hidden_size', 'unknown'))
        else:
            _logger.info("Loaded LSTM model (structure: %s)", list(lstm_data.keys()))

        return hmm_data, lstm_data

    def _reconstruct_lstm_model(self, lstm_data: Dict) -> torch.nn.Module:
        """
        Reconstruct LSTM model from saved components.

        Args:
            lstm_data: Loaded LSTM model data containing architecture and state dict

        Returns:
            Reconstructed PyTorch LSTM model
        """
        try:
            # Import LSTMModel from strategy module
            from src.strategy.hmm_lstm_strategy import LSTMModel

            # Get model architecture
            if 'model_architecture' not in lstm_data:
                raise ValueError("No model_architecture found in LSTM data")

            arch = lstm_data['model_architecture']

            # Create model with saved architecture
            model = LSTMModel(
                input_size=arch.get('input_size', 50),
                hidden_size=arch.get('hidden_size', 100),
                num_layers=arch.get('num_layers', 2),
                dropout=arch.get('dropout', 0.2),
                output_size=arch.get('output_size', 1),
                n_regimes=arch.get('n_regimes', 3)
            )

            # Load state dict
            if 'model_state_dict' not in lstm_data:
                raise ValueError("No model_state_dict found in LSTM data")

            model.load_state_dict(lstm_data['model_state_dict'])
            model.eval()  # Set to evaluation mode

            _logger.info("Reconstructed LSTM model: input_size=%d, hidden_size=%d, num_layers=%d",
                        arch.get('input_size', 50), arch.get('hidden_size', 100), arch.get('num_layers', 2))

            return model

        except Exception as e:
            _logger.exception("Error reconstructing LSTM model:")
            raise

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

        # Find the data file with the correct naming pattern
        # Pattern: {source}_{symbol}_{timeframe}_{start_date}_{end_date}.csv
        pattern = f"*_{symbol}_{timeframe}_*.csv"
        matching_files = list(data_dir.glob(pattern))

        if not matching_files:
            raise FileNotFoundError(f"No data file found for {symbol}_{timeframe} in {data_dir}")

        # Use the first matching file (or the most recent one)
        data_file = sorted(matching_files)[-1]  # Get the most recent file
        _logger.info("Using data file: %s", data_file.name)

        df = pd.read_csv(data_file)

        # Handle different datetime column names
        datetime_col = None
        for col in ['datetime', 'timestamp', 'date']:
            if col in df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            raise ValueError(f"No datetime column found in {data_file}. Available columns: {df.columns.tolist()}")

        df['datetime'] = pd.to_datetime(df[datetime_col], utc=True)
        df.set_index('datetime', inplace=True)

        # Convert to timezone-naive for compatibility
        df.index = df.index.tz_localize(None)

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

    def run_backtest(self, combination: Dict[str, str], strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtesting for a specific combination.

        Args:
            combination: Dictionary with symbol, timeframe, and model information
            strategy_params: Strategy parameters for this backtest

        Returns:
            Dictionary with backtest results
        """
        try:
            _logger.info("Running backtest for %s %s %s",
                        combination['provider'], combination['symbol'], combination['timeframe'])

            # Prepare data using base class method
            data_file = self.data_dir / combination['data_file']
            df = self._prepare_data(data_file)

            # Load models
            hmm_data, lstm_data = self.load_models(Path(combination['hmm_model']), Path(combination['lstm_model']))

            # Prepare strategy config
            strategy_config = {
                'prediction_threshold': strategy_params.get('entry_threshold', 0.001),
                'regime_confidence_threshold': strategy_params.get('regime_confidence_threshold', 0.4),
                'profit_target': strategy_params.get('profit_target', 0.02),
                'stop_loss': strategy_params.get('stop_loss', 0.01),
                'trailing_stop': strategy_params.get('trailing_stop', 0.005),
                'position_size': strategy_params.get('position_size', 0.1),
                'max_position_size': strategy_params.get('max_position_size', 0.5),
                'models_dir': str(self.models_dir),
                'results_dir': str(self.output_dir)
            }

            # Reconstruct LSTM model from saved components
            lstm_model = self._reconstruct_lstm_model(lstm_data)

            # Create Backtrader engine using base class method
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

            # Add strategy with correct model data structure
            cerebro.addstrategy(
                HMMLSTMStrategy,
                hmm_model=hmm_data['model'],
                hmm_scaler=hmm_data.get('scaler', None),
                hmm_features=hmm_data.get('features', None),
                lstm_model=lstm_model,
                lstm_scalers=lstm_data['scalers'],
                lstm_features=lstm_data['features'],
                sequence_length=lstm_data.get('hyperparameters', {}).get('sequence_length', 60),
                strategy_config=strategy_config
            )

            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

            # Set initial cash and commission
            cerebro.broker.setcash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Run backtest
            results = cerebro.run()
            strategy = results[0]

            # Extract results using base class method
            backtest_results = self._extract_backtest_results(cerebro, strategy)

            # Add combination-specific information
            backtest_results.update({
                'symbol': combination['symbol'],
                'timeframe': combination['timeframe'],
                'provider': combination['provider'],
                'strategy_params': strategy_params
            })

            _logger.info("Backtest completed - Return: %.2f%%, Sharpe: %.3f, Max DD: %.2f%%, Trades: %d",
                        backtest_results['total_return'] * 100, backtest_results['sharpe_ratio'],
                        backtest_results['max_drawdown'] * 100, backtest_results['total_trades'])

            return backtest_results

        except Exception as e:
            _logger.exception("Error in backtest for %s %s %s:",
                         combination['provider'], combination['symbol'], combination['timeframe'])


    def to_dict(self, obj):
        """Convert object to dictionary (same as custom_optimizer.py)."""
        if isinstance(obj, dict):
            return {k: self.to_dict(v) for k, v in obj.items()}
        elif hasattr(obj, "items"):
            return {k: self.to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.to_dict(v) for v in obj]
        else:
            return obj

    def optimize_parameters(self, combination: Dict[str, str], n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize strategy parameters using Optuna.

        Args:
            combination: Dictionary with symbol, timeframe, and model information
            n_trials: Number of optimization trials

        Returns:
            Dictionary with optimization results
        """
        try:
            _logger.info("Starting parameter optimization for %s %s %s",
                        combination['provider'], combination['symbol'], combination['timeframe'])

            def objective(trial):
                # Suggest parameters based on config
                strategy_params = {}

                # Get parameter ranges from config
                param_ranges = self.config.get('optimization', {}).get('parameter_ranges', {})

                for param_name, param_config in param_ranges.items():
                    if param_config['type'] == 'float':
                        strategy_params[param_name] = trial.suggest_float(
                            param_name,
                            param_config['min'],
                            param_config['max']
                        )
                    elif param_config['type'] == 'int':
                        strategy_params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['min'],
                            param_config['max']
                        )
                    elif param_config['type'] == 'categorical':
                        strategy_params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )

                # Run backtest
                results = self.run_backtest(combination, strategy_params)

                # Return negative Sharpe ratio (Optuna minimizes)
                return -results['sharpe_ratio']

            # Create study using base class method
            study = self._create_optuna_study(direction='minimize')

            # Optimize
            study.optimize(objective, n_trials=n_trials)

            # Get best parameters
            best_params = study.best_params
            best_value = -study.best_value  # Convert back to positive Sharpe ratio

            # Run final backtest with best parameters
            final_results = self.run_backtest(combination, best_params)

            optimization_results = {
                'best_params': best_params,
                'best_sharpe': best_value,
                'final_results': final_results,
                'optimization_history': study.trials_dataframe().to_dict('records')
            }

            _logger.info("Optimization completed - Best Sharpe: %.3f", best_value)

            return optimization_results

        except Exception as e:
            _logger.exception("Error in parameter optimization:")
            raise



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

    def discover_available_combinations(self) -> List[Dict[str, str]]:
        """
        Discover available symbol-timeframe combinations that have both models and data.

        Returns:
            List of dictionaries with symbol, timeframe, and model information
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
            # Expected format: {source}_{symbol}_{timeframe}_{start_date}_{end_date}.csv
            # Example: binance_BTCUSDT_1h_20230829_20250828.csv
            filename = csv_file.stem  # Remove .csv extension

            if '_' not in filename:
                _logger.warning("Skipping file with invalid format: %s (expected: source_symbol_timeframe_*.csv)", csv_file.name)
                continue

            # Split by underscores to extract components
            parts = filename.split('_')
            if len(parts) < 3:
                _logger.warning("Skipping file with invalid format: %s (expected: source_symbol_timeframe_*.csv)", csv_file.name)
                continue

            # Extract source, symbol, and timeframe
            source = parts[0]  # binance, yfinance, etc.
            symbol = parts[1]  # BTCUSDT, GOOG, etc.
            timeframe = parts[2]  # 1h, 4h, 15m, 1d, etc.

            # Check if models exist for this combination
            hmm_path, lstm_path = self.find_latest_models(symbol, timeframe)

            if hmm_path and lstm_path:
                available_combinations.append({
                    'provider': source,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_file': csv_file.name,
                    'hmm_model': str(hmm_path),
                    'lstm_model': str(lstm_path)
                })
                _logger.info("Found complete setup for %s %s (data + models)", symbol, timeframe)
            else:
                _logger.warning("Skipping %s %s - models not found (data exists: %s)",
                              symbol, timeframe, csv_file.name)

        return available_combinations

    def run(self, optimize: bool = False, n_trials: int = 100) -> Dict[str, Any]:
        """
        Run the complete HMM-LSTM backtesting process.

        Args:
            optimize: Whether to run parameter optimization
            n_trials: Number of optimization trials per combination

        Returns:
            Dictionary with overall results
        """
        _logger.info("Starting HMM-LSTM backtesting process")

        all_results = []

        # Auto-discover available combinations from data directory
        available_combinations = self.discover_available_combinations()

        if not available_combinations:
            _logger.error("No valid symbol-timeframe combinations found with both models and data")
            _logger.info("Please ensure you have:")
            _logger.info("1. OHLCV files in %s with format: source_symbol_timeframe_*.csv", self.config['data']['data_dir'])
            _logger.info("2. Trained models in %s with format: hmm/lstm_symbol_timeframe_*.pkl", self.models_dir)
            return {}

        _logger.info("Discovered %d symbol-timeframe combinations from data directory", len(available_combinations))

        for i, combination in enumerate(available_combinations):
            try:
                _logger.info("Processing combination %d/%d: %s %s %s",
                            i + 1, len(available_combinations),
                            combination['provider'], combination['symbol'], combination['timeframe'])

                # Load models
                hmm_data, lstm_data = self.load_models(Path(combination['hmm_model']), Path(combination['lstm_model']))

                if optimize:
                    # Run optimization
                    results = self.optimize_parameters(combination, n_trials)
                else:
                    # Run backtest with default parameters
                    default_params = self.config.get('strategy', {})
                    results = {
                        'final_results': self.run_backtest(combination, default_params),
                        'best_params': default_params
                    }

                all_results.append({
                    'combination': combination,
                    'results': results
                })

            except Exception as e:
                _logger.exception("Error processing combination %s %s %s:",
                             combination['provider'], combination['symbol'], combination['timeframe'])
                continue

        # Generate overall summary using base class method
        summary = self._generate_summary(all_results)

        # Save results using base class method
        self._save_results(all_results, summary, "hmm_lstm")

        _logger.info("HMM-LSTM backtesting completed successfully. Processed %d combinations", len(all_results))

        return summary


def main():
    """Main entry point for HMM-LSTM backtesting."""
    import argparse

    parser = argparse.ArgumentParser(description='HMM-LSTM Backtesting Optimizer')
    parser.add_argument('--config', default='config/optimizer/p01_hmm_lstm.json',
                       help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')

    args = parser.parse_args()

    # Initialize optimizer
    optimizer = HMMLSTMOptimizer(args.config)

    # Run backtesting and optimization
    results = optimizer.run(optimize=args.optimize, n_trials=args.n_trials)

    print("HMM-LSTM backtesting completed successfully!")
    print(f"Processed combinations: {results.get('total_combinations', 0)}")
    print(f"Average return: {results.get('average_return', 0):.2%}")
    print(f"Average Sharpe ratio: {results.get('average_sharpe', 0):.3f}")


if __name__ == "__main__":
    main()
