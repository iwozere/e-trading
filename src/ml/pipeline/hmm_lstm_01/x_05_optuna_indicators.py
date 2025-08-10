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
import logging
import json
from datetime import datetime
import optuna
import talib
import sys
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IndicatorOptimizer:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize indicator optimizer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.results_dir = Path(self.config['paths']['results'])
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

        logger.info(f"Loaded configuration from {self.config_path}")
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
                df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
                df['bb_width'] = (bb_upper - bb_lower) / bb_middle

            # MACD
            if all(param in params for param in ['macd_fast', 'macd_slow', 'macd_signal']):
                macd, macd_signal, macd_hist = talib.MACD(
                    close,
                    fastperiod=params['macd_fast'],
                    slowperiod=params['macd_slow'],
                    signalperiod=params['macd_signal']
                )
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_hist

            # Moving Averages
            if 'ema_fast' in params:
                df['ema_fast'] = talib.EMA(close, timeperiod=params['ema_fast'])
            if 'ema_slow' in params:
                df['ema_slow'] = talib.EMA(close, timeperiod=params['ema_slow'])
            if 'sma_period' in params:
                df['sma'] = talib.SMA(close, timeperiod=params['sma_period'])

            # ATR
            if 'atr_period' in params:
                df['atr'] = talib.ATR(high, low, close, timeperiod=params['atr_period'])

            # Stochastic
            if 'stoch_k' in params and 'stoch_d' in params:
                stoch_k, stoch_d = talib.STOCH(
                    high, low, close,
                    fastk_period=params['stoch_k'],
                    slowk_period=params['stoch_d'],
                    slowd_period=params['stoch_d']
                )
                df['stoch_k'] = stoch_k
                df['stoch_d'] = stoch_d

            # Williams %R
            if 'williams_period' in params:
                df['williams_r'] = talib.WILLR(high, low, close, timeperiod=params['williams_period'])

            # Money Flow Index
            if 'mfi_period' in params:
                df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=params['mfi_period'])

        except Exception as e:
            logger.warning(f"Error calculating indicators with params {params}: {str(e)}")
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

            # Multi-indicator strategy
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

            # MACD signals
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                if (not pd.isna(df['macd'].iloc[i]) and not pd.isna(df['macd_signal'].iloc[i])):
                    macd = df['macd'].iloc[i]
                    macd_signal = df['macd_signal'].iloc[i]

                    if i > 0:  # Check for crossover
                        prev_macd = df['macd'].iloc[i-1]
                        prev_signal = df['macd_signal'].iloc[i-1]

                        # Bullish crossover
                        if macd > macd_signal and prev_macd <= prev_signal:
                            buy_signals += 1
                        # Bearish crossover
                        elif macd < macd_signal and prev_macd >= prev_signal:
                            sell_signals += 1

            # EMA signals
            if all(col in df.columns for col in ['ema_fast', 'ema_slow']):
                if (not pd.isna(df['ema_fast'].iloc[i]) and not pd.isna(df['ema_slow'].iloc[i])):
                    if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]:
                        buy_signals += 0.5
                    else:
                        sell_signals += 0.5

            # Stochastic signals
            if all(col in df.columns for col in ['stoch_k', 'stoch_d']):
                if (not pd.isna(df['stoch_k'].iloc[i]) and not pd.isna(df['stoch_d'].iloc[i])):
                    stoch_k = df['stoch_k'].iloc[i]
                    stoch_d = df['stoch_d'].iloc[i]

                    if stoch_k < 20 and stoch_d < 20:
                        buy_signals += 1
                    elif stoch_k > 80 and stoch_d > 80:
                        sell_signals += 1

            # Generate final signal
            if buy_signals > sell_signals and buy_signals >= 2:
                signal = 1
            elif sell_signals > buy_signals and sell_signals >= 2:
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

        # Total Return
        total_return = cumulative_returns.iloc[-1] - 1

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
            # Suggest indicator parameters
            params = {}

            # RSI
            params['rsi_period'] = trial.suggest_int('rsi_period', 5, 50)

            # Bollinger Bands
            params['bb_period'] = trial.suggest_int('bb_period', 10, 50)
            params['bb_std'] = trial.suggest_float('bb_std', 1.0, 3.0)

            # MACD
            params['macd_fast'] = trial.suggest_int('macd_fast', 5, 20)
            params['macd_slow'] = trial.suggest_int('macd_slow', 20, 50)
            params['macd_signal'] = trial.suggest_int('macd_signal', 5, 20)

            # EMA
            params['ema_fast'] = trial.suggest_int('ema_fast', 5, 30)
            params['ema_slow'] = trial.suggest_int('ema_slow', 20, 100)

            # ATR
            params['atr_period'] = trial.suggest_int('atr_period', 5, 30)

            # Stochastic
            params['stoch_k'] = trial.suggest_int('stoch_k', 5, 25)
            params['stoch_d'] = trial.suggest_int('stoch_d', 3, 15)

            # Williams %R
            params['williams_period'] = trial.suggest_int('williams_period', 5, 30)

            # MFI
            params['mfi_period'] = trial.suggest_int('mfi_period', 5, 30)

            # SMA
            params['sma_period'] = trial.suggest_int('sma_period', 10, 100)

            # Ensure logical constraints
            if params['macd_fast'] >= params['macd_slow']:
                params['macd_fast'] = params['macd_slow'] - 1

            if params['ema_fast'] >= params['ema_slow']:
                params['ema_fast'] = params['ema_slow'] - 1

            # Calculate indicators with suggested parameters
            df_with_indicators = self.calculate_indicators_with_params(df, params)

            # Generate trading signals
            df_with_signals = self.generate_trading_signals(df_with_indicators)

            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(df_with_signals)

            # Multi-objective: combine Sharpe ratio, profit factor, and drawdown
            sharpe_ratio = metrics['sharpe_ratio']
            profit_factor = metrics['profit_factor']
            max_drawdown = metrics['max_drawdown']

            # Normalize and combine metrics
            # Higher Sharpe ratio is better
            # Higher profit factor is better
            # Lower max drawdown is better

            if sharpe_ratio < -10 or profit_factor < 0.5 or max_drawdown > 0.5:
                return -999  # Penalize poor strategies

            # Combined objective (to be maximized, so we'll return negative)
            objective_value = (sharpe_ratio * 0.5 +
                             np.log(profit_factor) * 0.3 -
                             max_drawdown * 0.2)

            return -objective_value  # Negative because Optuna minimizes

        except Exception as e:
            logger.warning(f"Error in objective function: {str(e)}")
            return 999  # Return large positive value for failed trials

    def optimize_indicators(self, symbol: str, timeframe: str) -> Dict:
        """
        Optimize indicator parameters for a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with optimization results
        """
        logger.info(f"Optimizing indicators for {symbol} {timeframe}")

        try:
            # Find labeled data file
            pattern = f"labeled_{symbol}_{timeframe}_*.csv"
            csv_files = list(self.labeled_data_dir.glob(pattern))

            if not csv_files:
                raise FileNotFoundError(f"No labeled data found for {symbol} {timeframe}")

            # Use the most recent file
            csv_file = sorted(csv_files)[-1]
            logger.info(f"Using data file: {csv_file}")

            # Load data
            df = pd.read_csv(csv_file)

            # Use a subset of data for optimization (to speed up the process)
            # Keep last 80% for optimization, reserve 20% for validation
            split_idx = int(len(df) * 0.2)
            df_optimization = df.iloc[split_idx:].copy().reset_index(drop=True)

            logger.info(f"Using {len(df_optimization)} samples for optimization")

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

            logger.info(f"Optimization completed for {symbol} {timeframe}")
            logger.info(f"Best objective value: {best_value:.4f}")
            logger.info(f"Best parameters: {best_params}")

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

            logger.info(f"[OK] Saved optimization results to {output_path}")

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
            logger.error(error_msg)
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
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']

        logger.info(f"Optimizing indicators for {len(symbols)} symbols x {len(timeframes)} timeframes")

        results = {
            'total': len(symbols) * len(timeframes),
            'successful': [],
            'failed': []
        }

        for symbol in symbols:
            for timeframe in timeframes:
                result = self.optimize_indicators(symbol, timeframe)

                if result['success']:
                    results['successful'].append(result)
                else:
                    results['failed'].append(result)

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Indicator Optimization Summary:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"{'='*50}")

        if results['failed']:
            logger.warning("Failed optimizations:")
            for failure in results['failed']:
                logger.warning(f"  {failure['symbol']} {failure['timeframe']}: {failure['error']}")

        return results

def main():
    """Main function to run indicator optimization."""
    try:
        optimizer = IndicatorOptimizer()
        results = optimizer.optimize_all()

        logger.info("Indicator optimization completed!")

    except Exception as e:
        logger.error(f"Indicator optimization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
