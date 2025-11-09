"""
CNN-XGBoost Backtesting and Optimization System.

This module provides comprehensive backtesting and optimization capabilities for the CNN-XGBoost
trading strategy using models from pipeline p03_cnn_xgboost. It supports parameter optimization
using Optuna and generates detailed performance reports.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import json
import pickle

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.backtester.optimizer.base_optimizer import BaseOptimizer
from src.strategy.cnn_xgboost_strategy import CNNXGBoostStrategy
from src.notification.logger import setup_logger

warnings.filterwarnings('ignore')

_logger = setup_logger(__name__)


class CNNXGBoostOptimizer(BaseOptimizer):
    """
    CNN-XGBoost Backtesting and Optimization System.

    This optimizer provides comprehensive backtesting and optimization capabilities
    for the CNN-XGBoost trading strategy using models from pipeline p03_cnn_xgboost.
    """

    def __init__(self, config_path: str = "config/optimizer/p03_cnn_xgboost.json"):
        """
        Initialize the CNN-XGBoost optimizer.

        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)

        # Set up model directory
        self.models_dir = Path(self.config['models']['models_dir'])

        # Model discovery
        self.available_combinations = self.discover_available_combinations()

        _logger.info("CNN-XGBoost optimizer initialized")
        _logger.info("Available combinations: %d", len(self.available_combinations))

    def discover_available_combinations(self) -> List[Dict[str, str]]:
        """
        Discover available data and model combinations.

        Returns:
            List of dictionaries with symbol, timeframe, and model information
        """
        combinations = []

        try:
            # Find data files
            data_files = list(self.data_dir.glob("*.csv"))

            for csv_file in data_files:
                # Expected format: {provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv
                parts = csv_file.stem.split('_')
                if len(parts) < 3:
                    _logger.warning("Skipping file with invalid format: %s (expected: provider_symbol_timeframe_*.csv)", csv_file.name)
                    continue

                provider = parts[0]
                symbol = parts[1]
                timeframe = parts[2]

                # Check if corresponding models exist
                cnn_model = self._find_cnn_model(provider, symbol, timeframe)
                xgb_models = self._find_xgb_models(symbol, timeframe)

                if cnn_model and xgb_models:
                    combinations.append({
                        'provider': provider,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'data_file': csv_file.name,
                        'cnn_model': cnn_model,
                        'xgb_models': xgb_models
                    })
                    _logger.debug("Found combination: %s %s %s", provider, symbol, timeframe)

        except Exception:
            _logger.exception("Error discovering combinations:")

        return combinations

    def _find_cnn_model(self, provider: str, symbol: str, timeframe: str) -> Optional[str]:
        """
        Find CNN model for the given combination.

        Args:
            provider: Data provider (e.g., 'binance', 'yfinance')
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Path to CNN model file if found, None otherwise
        """
        try:
            cnn_model_dir = self.models_dir / "cnn"

            # Find CNN model - pattern: cnn_{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.pth
            cnn_pattern = f"cnn_{provider}_{symbol}_{timeframe}_*.pth"
            cnn_files = list(cnn_model_dir.glob(cnn_pattern))

            if cnn_files:
                # Return the most recent model
                latest_model = sorted(cnn_files)[-1]
                return str(latest_model)

            _logger.warning("No CNN model found for %s %s %s", provider, symbol, timeframe)
            return None

        except Exception:
            _logger.exception("Error finding CNN model:")
            return None

    def _find_xgb_models(self, symbol: str, timeframe: str) -> Optional[Dict[str, str]]:
        """
        Find XGBoost models for the given combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary of target to model path mappings if found, None otherwise
        """
        try:
            xgb_model_dir = self.models_dir / "xgboost"
            targets = ["target_direction", "target_volatility", "target_trend", "target_magnitude"]

            xgb_models = {}

            for target in targets:
                # Find XGBoost model - pattern: {target}_model.pkl
                model_path = xgb_model_dir / f"{target}_model.pkl"

                if model_path.exists():
                    xgb_models[target] = str(model_path)
                else:
                    _logger.warning("XGBoost model not found for target %s", target)

            # Return models only if all targets are available
            if len(xgb_models) == len(targets):
                return xgb_models

            _logger.warning("Incomplete XGBoost models for %s %s", symbol, timeframe)
            return None

        except Exception:
            _logger.exception("Error finding XGBoost models:")
            return None



    def _load_models(self, combination: Dict[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load CNN and XGBoost models for the given combination.

        Args:
            combination: Dictionary with model information

        Returns:
            Tuple of (cnn_data, xgb_data) dictionaries
        """
        try:
            # Load CNN model data
            cnn_data = {}
            if combination['cnn_model']:
                cnn_model_path = Path(combination['cnn_model'])

                # Load model configuration
                config_path = cnn_model_path.parent / f"{cnn_model_path.stem}_config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        cnn_data['config'] = json.load(f)

                # Load model weights
                cnn_data['model_path'] = str(cnn_model_path)

                # Load scaler
                scaler_path = cnn_model_path.parent / f"{cnn_model_path.stem}_scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        cnn_data['scaler'] = pickle.load(f)

                _logger.info("Loaded CNN model: %s", cnn_model_path.name)

            # Load XGBoost models data
            xgb_data = {}
            if combination['xgb_models']:
                for target, model_path in combination['xgb_models'].items():
                    model_path = Path(model_path)

                    # Load model
                    with open(model_path, 'rb') as f:
                        xgb_data[f'{target}_model'] = pickle.load(f)

                    # Load scaler if available
                    scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
                    if scaler_path.exists():
                        with open(scaler_path, 'rb') as f:
                            xgb_data[f'{target}_scaler'] = pickle.load(f)

                    _logger.info("Loaded XGBoost model for %s: %s", target, model_path.name)

            return cnn_data, xgb_data

        except Exception:
            _logger.exception("Error loading models:")
            raise

    def run_backtest(self, combination: Dict[str, str], strategy_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest for a specific combination.

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
            cnn_data, xgb_data = self._load_models(combination)

            # Prepare strategy config
            strategy_config = {
                'prediction_threshold': strategy_params['prediction_threshold'],
                'direction_weight': strategy_params['direction_weight'],
                'volatility_weight': strategy_params['volatility_weight'],
                'trend_weight': strategy_params['trend_weight'],
                'magnitude_weight': strategy_params['magnitude_weight'],
                'sequence_length': strategy_params.get('sequence_length', 120),
                'cnn_input_channels': strategy_params.get('cnn_input_channels', 5),
                'cnn_num_filters': strategy_params.get('cnn_num_filters', [32, 64, 128]),
                'cnn_kernel_sizes': strategy_params.get('cnn_kernel_sizes', [3, 5, 7]),
                'cnn_dropout_rate': strategy_params.get('cnn_dropout_rate', 0.3),
                'profit_target': strategy_params.get('profit_target', 0.02),
                'stop_loss': strategy_params.get('stop_loss', 0.01),
                'trailing_stop': strategy_params.get('trailing_stop', 0.005),
                'base_position_size': strategy_params.get('base_position_size', 0.1),
                'min_position_size': strategy_params.get('min_position_size', 0.01),
                'max_position_size': strategy_params.get('max_position_size', 0.5),
                'cnn_model_path': cnn_data.get('model_path'),
                'xgb_models_dir': str(Path(combination['xgb_models']['target_direction']).parent),
                'optimized_indicators': strategy_params.get('optimized_indicators', {})
            }

            # Create Backtrader engine using base class method
            cerebro = self._create_backtrader_engine(df, CNNXGBoostStrategy, strategy_config)

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

        except Exception:
            _logger.exception("Error in backtest for %s %s %s:",
                         combination['provider'], combination['symbol'], combination['timeframe'])
            raise

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
                # Suggest parameters
                strategy_params = {
                    'prediction_threshold': trial.suggest_float('prediction_threshold', 0.3, 0.8),
                    'direction_weight': trial.suggest_float('direction_weight', 0.2, 0.6),
                    'volatility_weight': trial.suggest_float('volatility_weight', 0.1, 0.3),
                    'trend_weight': trial.suggest_float('trend_weight', 0.1, 0.3),
                    'magnitude_weight': trial.suggest_float('magnitude_weight', 0.1, 0.3),
                    'profit_target': trial.suggest_float('profit_target', 0.01, 0.05),
                    'stop_loss': trial.suggest_float('stop_loss', 0.005, 0.03),
                    'base_position_size': trial.suggest_float('base_position_size', 0.05, 0.3)
                }

                # Ensure weights sum to 1.0
                total_weight = (strategy_params['direction_weight'] +
                              strategy_params['volatility_weight'] +
                              strategy_params['trend_weight'] +
                              strategy_params['magnitude_weight'])

                if total_weight > 0:
                    strategy_params['direction_weight'] /= total_weight
                    strategy_params['volatility_weight'] /= total_weight
                    strategy_params['trend_weight'] /= total_weight
                    strategy_params['magnitude_weight'] /= total_weight

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

        except Exception:
            _logger.exception("Error in parameter optimization:")
            raise

    def run(self, optimize: bool = False, n_trials: int = 100) -> Dict[str, Any]:
        """
        Run backtesting and optimization for all available combinations.

        Args:
            optimize: Whether to run parameter optimization
            n_trials: Number of optimization trials per combination

        Returns:
            Dictionary with overall results
        """
        try:
            _logger.info("Starting CNN-XGBoost backtesting and optimization")
            _logger.info("Combinations to process: %d", len(self.available_combinations))

            all_results = []

            for i, combination in enumerate(self.available_combinations):
                try:
                    _logger.info("Processing combination %d/%d: %s %s %s",
                                i + 1, len(self.available_combinations),
                                combination['provider'], combination['symbol'], combination['timeframe'])

                    if optimize:
                        # Run optimization
                        results = self.optimize_parameters(combination, n_trials)
                    else:
                        # Run backtest with default parameters
                        default_params = {
                            'prediction_threshold': 0.6,
                            'direction_weight': 0.4,
                            'volatility_weight': 0.2,
                            'trend_weight': 0.2,
                            'magnitude_weight': 0.2,
                            'profit_target': 0.02,
                            'stop_loss': 0.01,
                            'base_position_size': 0.1
                        }
                        results = {
                            'final_results': self.run_backtest(combination, default_params),
                            'best_params': default_params
                        }

                    all_results.append({
                        'combination': combination,
                        'results': results
                    })

                except Exception:
                    _logger.exception("Error processing combination %s %s %s:",
                                 combination['provider'], combination['symbol'], combination['timeframe'])
                    continue

            # Generate overall summary using base class method
            summary = self._generate_summary(all_results)

            # Save results using base class method
            self._save_results(all_results, summary, "cnn_xgboost")

            _logger.info("CNN-XGBoost backtesting and optimization completed")
            _logger.info("Processed combinations: %d", len(all_results))

            return summary

        except Exception:
            _logger.exception("Error in main run:")
            raise




def main():
    """Main function to run CNN-XGBoost backtesting and optimization."""
    try:
        # Load configuration
        config_path = "config/optimizer/p03_cnn_xgboost.json"

        # Create optimizer
        optimizer = CNNXGBoostOptimizer(config_path)

        # Run backtesting and optimization
        results = optimizer.run(optimize=True, n_trials=50)

        print("CNN-XGBoost backtesting and optimization completed successfully!")
        print(f"Processed combinations: {results.get('total_combinations', 0)}")
        print(f"Average return: {results.get('average_return', 0):.2%}")
        print(f"Average Sharpe ratio: {results.get('average_sharpe', 0):.3f}")

    except Exception:
        _logger.exception("Error in main:")
        raise


if __name__ == "__main__":
    main()
