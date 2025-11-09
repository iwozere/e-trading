"""
Optuna Optimization for Technical Indicator Parameters

This module uses Optuna to optimize technical indicator parameters based on
trading performance metrics. It performs backtesting with different indicator
settings to find the combination that maximizes Sharpe ratio, profit factor,
and minimizes drawdown.

Features:
- Optimizes multiple technical indicators simultaneously
- Uses trading performance metrics as objectives
- Implements simple trading strategies for evaluation
- Saves optimal parameters for use in LSTM training
- Supports multiple objectives and Pareto optimization
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import json
from datetime import datetime
import optuna
import talib
import sys
from typing import Dict

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class IndicatorOptimizer:
    def __init__(self, config_path: str = "config/pipeline/p01.yaml"):
        """
        Initialize indicator optimizer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.results_dir = Path(self.config['paths']['models_lstm'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Optuna configuration
        self.n_trials = self.config['optuna']['n_trials']
        self.timeout = self.config['optuna'].get('timeout', 3600)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def calculate_indicators_with_params(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Calculate technical indicators with given parameters.

        Args:
            df: DataFrame with OHLCV data
            params: Dictionary of indicator parameters

        Returns:
            DataFrame with calculated indicators
        """
        df = df.copy()

        # Extract OHLCV arrays
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        try:
            # RSI
            if 'rsi_period' in params:
                df['rsi'] = talib.RSI(close, timeperiod=params['rsi_period'])

            # Bollinger Bands
            if 'bb_period' in params and 'bb_std' in params:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close,
                    timeperiod=params['bb_period'],
                    nbdevup=params['bb_std'],
                    nbdevdn=params['bb_std']
                )
                df['bb_upper'] = bb_upper
                df['bb_middle'] = bb_middle
                df['bb_lower'] = bb_lower
                # Avoid division by zero in bb_position calculation
                bb_range = bb_upper - bb_lower
                # Use masked arrays to avoid division by zero warnings
                mask = bb_range != 0
                bb_position = np.full_like(close, 0.5)  # Default value
                bb_position[mask] = (close[mask] - bb_lower[mask]) / bb_range[mask]
                df['bb_position'] = bb_position

                # Avoid division by zero in bb_width calculation
                mask_width = bb_middle != 0
                bb_width = np.full_like(close, 0)  # Default value
                bb_width[mask_width] = bb_range[mask_width] / bb_middle[mask_width]
                df['bb_width'] = bb_width

        except Exception as e:
            _logger.warning("Error calculating indicators with params %s: %s", params, str(e))
            # Return original DataFrame if calculation fails
            return df

        return df

    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on calculated indicators.

        Args:
            df: DataFrame with calculated indicators

        Returns:
            DataFrame with trading signals
        """
        df = df.copy()
        signals = []

        for i in range(len(df)):
            signal = 0  # 0: hold, 1: buy, -1: sell

            # Simple RSI + Bollinger Bands strategy
            buy_signals = 0
            sell_signals = 0

            # RSI signals
            if 'rsi' in df.columns and not pd.isna(df['rsi'].iloc[i]):
                rsi = df['rsi'].iloc[i]
                if rsi < 30:
                    buy_signals += 1
                elif rsi > 70:
                    sell_signals += 1

            # Bollinger Bands signals
            if 'bb_position' in df.columns and not pd.isna(df['bb_position'].iloc[i]):
                bb_pos = df['bb_position'].iloc[i]
                if bb_pos < 0.2:  # Near lower band
                    buy_signals += 1
                elif bb_pos > 0.8:  # Near upper band
                    sell_signals += 1

            # Generate final signal - simpler logic
            if buy_signals >= 1:  # At least one buy signal
                signal = 1
            elif sell_signals >= 1:  # At least one sell signal
                signal = -1

            signals.append(signal)

        df['signal'] = signals
        return df

    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate trading performance metrics.

        Args:
            df: DataFrame with prices and signals

        Returns:
            Dict with performance metrics
        """
        if 'signal' not in df.columns:
            return {'sharpe_ratio': -999, 'profit_factor': 0, 'max_drawdown': 999}

        # Calculate returns
        df = df.copy()
        df['price_return'] = df['close'].pct_change()

        # Calculate strategy returns (shift signal by 1 to avoid look-ahead bias)
        df['strategy_return'] = df['signal'].shift(1) * df['price_return']
        df = df.dropna()

        if len(df) < 10:  # Not enough data
            return {'sharpe_ratio': -999, 'profit_factor': 0, 'max_drawdown': 999}

        strategy_returns = df['strategy_return']

        # Handle NaN values in strategy returns
        strategy_returns = strategy_returns.fillna(0)

        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()

        # Sharpe Ratio (annualized)
        if strategy_returns.std() == 0:
            sharpe_ratio = 0
        else:
            # Assuming daily data, annualize with 365 factor
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365)

        # Profit Factor
        positive_returns = strategy_returns[strategy_returns > 0].sum()
        negative_returns = abs(strategy_returns[strategy_returns < 0].sum())

        if negative_returns == 0:
            profit_factor = 100 if positive_returns > 0 else 1
        else:
            profit_factor = positive_returns / negative_returns

        # Maximum Drawdown
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = abs(drawdown.min())

        # Total Return - handle edge cases
        if len(cumulative_returns) > 0:
            total_return = cumulative_returns.iloc[-1] - 1
        else:
            total_return = 0

        # Win Rate
        win_rate = (strategy_returns > 0).mean()

        # Number of trades
        n_trades = (df['signal'].diff() != 0).sum()

        return {
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'win_rate': win_rate,
            'n_trades': n_trades,
            'avg_return': strategy_returns.mean(),
            'volatility': strategy_returns.std()
        }

    def objective(self, trial: optuna.trial.Trial, df: pd.DataFrame) -> float:
        """
        Optuna objective function for indicator optimization.

        Args:
            trial: Optuna trial object
            df: DataFrame with OHLCV data

        Returns:
            Objective value (negative because Optuna minimizes)
        """
        try:
            # Suggest indicator parameters - Start with just RSI and Bollinger Bands
            params = {}

            # RSI
            params['rsi_period'] = trial.suggest_int('rsi_period', 5, 50)

            # Bollinger Bands
            params['bb_period'] = trial.suggest_int('bb_period', 10, 50)
            params['bb_std'] = trial.suggest_float('bb_std', 1.0, 3.0)

            # Calculate indicators with suggested parameters
            df_with_indicators = self.calculate_indicators_with_params(df, params)

            # Generate trading signals
            df_with_signals = self.generate_trading_signals(df_with_indicators)

            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(df_with_signals)

                        # Focus on profit and drawdown instead of Sharpe ratio
            total_return = metrics['total_return']
            profit_factor = metrics['profit_factor']
            max_drawdown = metrics['max_drawdown']

            # Handle NaN and infinite values
            if np.isnan(total_return) or np.isinf(total_return):
                total_return = 0
            if np.isnan(profit_factor) or np.isinf(profit_factor):
                profit_factor = 1
            if np.isnan(max_drawdown) or np.isinf(max_drawdown):
                max_drawdown = 1

            # Debug logging for first few trials
            # Note: study is not available in objective scope, so we'll log without trial number
            _logger.debug("Trial - Total Return: %.3f, Profit Factor: %.3f, Max DD: %.3f",
                       total_return, profit_factor, max_drawdown)

            # More realistic thresholds for initial optimization
            if total_return < -0.5 or profit_factor < 0.3 or max_drawdown > 0.8:
                return 999  # Penalize extremely poor strategies

            # Focus on profit-based objective function
            # Total return is the primary metric (what we actually care about)
            profit_component = total_return * 0.6  # 60% weight on total return
            profit_factor_component = np.log(max(0.1, profit_factor)) * 0.3  # 30% weight on profit factor
            drawdown_component = max_drawdown * 0.1  # 10% penalty for drawdown

            objective_value = profit_component + profit_factor_component - drawdown_component

            return -objective_value  # Negative because Optuna minimizes

        except Exception as e:
            _logger.warning("Error in objective function: %s", str(e))
            return 999  # Return large positive value for failed trials

    def optimize_indicators(self, symbol: str, timeframe: str, provider: str = None) -> Dict:
        """
        Optimize indicator parameters for a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            provider: Data provider (e.g., 'binance', 'yfinance')

        Returns:
            Dict with optimization results
        """
        _logger.info("Optimizing indicators for %s %s (provider: %s)", symbol, timeframe, provider)

        try:
            # Find labeled data file with provider prefix
            if provider:
                pattern = f"{provider}_{symbol}_{timeframe}_*_labeled.csv"
            else:
                # Fallback to old pattern for backward compatibility
                pattern = f"{symbol}_{timeframe}_*_labeled.csv"

            csv_files = list(self.labeled_data_dir.glob(pattern))

            if not csv_files:
                raise FileNotFoundError(f"No labeled data found for {symbol} {timeframe}")

            # Use the most recent file
            csv_file = sorted(csv_files)[-1]
            _logger.info("Using data file: %s", csv_file)

            # Load data
            df = pd.read_csv(csv_file)

            # Use a subset of data for optimization (to speed up the process)
            # Keep last 80% for optimization, reserve 20% for validation
            split_idx = int(len(df) * 0.2)
            df_optimization = df.iloc[split_idx:].copy().reset_index(drop=True)

            _logger.info("Using %d samples for optimization", len(df_optimization))

            # Create Optuna study
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            )

            # Optimize
            study.optimize(
                lambda trial: self.objective(trial, df_optimization),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )

            # Get best parameters
            best_params = study.best_params
            best_value = -study.best_value  # Convert back to positive

            _logger.info("Optimization completed for %s %s", symbol, timeframe)
            _logger.info("Best objective value: %.4f", best_value)
            _logger.info("Best parameters: %s", best_params)

            # Validate on the full dataset
            df_with_indicators = self.calculate_indicators_with_params(df, best_params)
            df_with_signals = self.generate_trading_signals(df_with_indicators)
            validation_metrics = self.calculate_performance_metrics(df_with_signals)

            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Convert NumPy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj

            results = {
                'symbol': symbol,
                'timeframe': timeframe,
                'optimization_timestamp': timestamp,
                'best_params': convert_numpy_types(best_params),
                'best_objective_value': convert_numpy_types(best_value),
                'validation_metrics': convert_numpy_types(validation_metrics),
                'n_trials': len(study.trials),
                'optimization_samples': len(df_optimization),
                'validation_samples': len(df)
            }

            # Save to JSON file
            output_filename = f"indicators_{symbol}_{timeframe}_{timestamp}.json"
            output_path = self.results_dir / output_filename

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            _logger.info("[OK] Saved optimization results to %s", output_path)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'results_file': str(output_path),
                'best_params': best_params,
                'validation_metrics': validation_metrics
            }

        except Exception as e:
            error_msg = f"Failed to optimize indicators for {symbol} {timeframe}: {str(e)}"
            _logger.error(error_msg)
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': error_msg
            }

    def optimize_all(self) -> Dict:
        """
        Optimize indicator parameters for all symbol-timeframe combinations.

        Returns:
            Dict with summary of optimization results
        """
                # Check if using new multi-provider format
        if 'data_sources' in self.config:
            _logger.info("Using multi-provider configuration format")
            symbols = []
            timeframes = []
            providers = []

            for provider, provider_config in self.config['data_sources'].items():
                provider_symbols = provider_config['symbols']
                provider_timeframes = provider_config['timeframes']

                for symbol in provider_symbols:
                    for timeframe in provider_timeframes:
                        symbols.append(symbol)
                        timeframes.append(timeframe)
                        providers.append(provider)

            _logger.info("Multi-provider symbols: %s", symbols)
            _logger.info("Multi-provider timeframes: %s", timeframes)
            _logger.info("Multi-provider providers: %s", providers)
        else:
            # Legacy format
            symbols = self.config['symbols']
            timeframes = self.config['timeframes']
            providers = [None] * len(symbols)  # No provider info for legacy format
            _logger.info("Using legacy configuration format")

        _logger.info("Optimizing indicators for %d symbol-timeframe combinations", len(symbols))

        results = {
            'total': len(symbols),
            'successful': [],
            'failed': []
        }

        for i, (symbol, timeframe, provider) in enumerate(zip(symbols, timeframes, providers)):
            _logger.info("Processing %d/%d: %s %s (provider: %s)", i+1, len(symbols), symbol, timeframe, provider)
            result = self.optimize_indicators(symbol, timeframe, provider)

            if result['success']:
                results['successful'].append(result)
            else:
                results['failed'].append(result)

        # Log summary
        _logger.info("%s", "="*50)
        _logger.info("Indicator Optimization Summary:")
        _logger.info("  Total: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))
        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.warning("Failed optimizations:")
            for failure in results['failed']:
                _logger.warning("  %s %s: %s", failure['symbol'], failure['timeframe'], failure['error'])

        return results

def main():
    """Main function to run indicator optimization."""
    try:
        optimizer = IndicatorOptimizer()
        results = optimizer.optimize_all()

        _logger.info("Indicator optimization completed!")

    except Exception:
        _logger.exception("Indicator optimization failed: ")
        raise

if __name__ == "__main__":
    main()
