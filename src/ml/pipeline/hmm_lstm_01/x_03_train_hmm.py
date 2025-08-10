"""
HMM Training for Market Regime Detection

This module trains Hidden Markov Models to detect market regimes using
preprocessed OHLCV data and technical indicators. The HMM identifies
different market states (e.g., trending, ranging, volatile) that will
be used as features for the LSTM model.

Features:
- Trains HMM with configurable number of components (regimes)
- Uses rolling training windows for temporal adaptation
- Saves trained models with timestamp for versioning
- Validates regime detection quality
- Supports multiple symbols and timeframes
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import sys
from typing import Dict, List, Optional, Tuple
import talib

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class HMMTrainer:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize HMM trainer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.processed_data_dir = Path(self.config['paths']['data_processed'])
        self.models_dir = Path(self.config['paths']['models_hmm'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # HMM configuration
        self.n_components = self.config['hmm']['n_components']
        self.train_window_days = self.config['hmm']['train_window_days']
        #self.covariance_type = self.config['hmm'].get('covariance_type', 'diag')
        self.covariance_type = self.config['hmm'].get('covariance_type', 'full')
        self.algorithm = self.config['hmm'].get('algorithm', 'viterbi')
        self.n_iter = self.config['hmm'].get('n_iter', 100)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    @staticmethod
    def build_indicator_config(timeframe: str, normalized_timeframe_in_minutes: int = 240) -> Dict:
        """
        Build indicator_config dictionary with parameters adapted to the timeframe.

        Args:
            timeframe: str, e.g. '5m', '15m', '1h', '4h', '1d'

        Returns:
            Dict with indicator names as keys and their parameters as nested dicts
        """
        # Default periods per indicator (can be adjusted)
        base_periods = {
            'rsi': 14,
            'atr': 14,
            'ema_fast': 12,
            'ema_slow': 26,
            'bbands': 20
        }

        # Adjust periods depending on timeframe scale
        if timeframe.endswith('m'):
            multiplier = int(timeframe[:-1])  # e.g. '5m' -> 5
        elif timeframe.endswith('h'):
            multiplier = int(timeframe[:-1]) * 60  # convert hours to minutes
        elif timeframe.endswith('d'):
            multiplier = int(timeframe[:-1]) * 60 * 24  # convert days to minutes
        else:
            multiplier = 1  # fallback

        # Scale indicator periods inversely to timeframe
        # Shorter timeframes need longer periods, longer timeframes need shorter periods
        # This makes sense because:
        # - 5m bars: need more bars (larger period) to capture meaningful patterns
        # - 1d bars: each bar already represents significant time, so smaller periods suffice
        # Use square root scaling to avoid extreme values while maintaining the inverse relationship
        scale_factor = (normalized_timeframe_in_minutes / multiplier) ** 0.5

        def scaled_period(base):
            val = max(2, int(round(base * scale_factor)))  # Minimum period of 2 for TA-Lib
            return val

        config = {
            "rsi": {"timeperiod": scaled_period(base_periods['rsi'])},
            "atr": {"timeperiod": scaled_period(base_periods['atr'])},
            "ema_spread": {
                "fastperiod": scaled_period(base_periods['ema_fast']),
                "slowperiod": scaled_period(base_periods['ema_slow']),
            },
            "bbands": {"timeperiod": scaled_period(base_periods['bbands']), "nbdevup": 2, "nbdevdn": 2}
        }

        return config

    def select_features_for_hmm(self, df: pd.DataFrame, indicator_config: Dict) -> List[str]:
        """
        Select optimal features for HMM training, adding indicators from TA-Lib if needed.

        Args:
            df: Preprocessed DataFrame with OHLCV data
            indicator_config: dict specifying indicators to include and their parameters
                Example:
                {
                    "rsi": {"timeperiod": 14},
                    "atr": {"timeperiod": 14},
                    "ema_spread": {"fastperiod": 12, "slowperiod": 26},
                    "bbands": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2}
                }

        Returns:
            List[str]: Selected feature column names
        """

        base_features = ["log_return"]

        # === Compute indicators with TA-Lib if missing ===
        if "rsi" in indicator_config:
            tp = indicator_config["rsi"].get("timeperiod", 14)
            rsi_col = f"rsi_{tp}"
            if rsi_col not in df.columns:
                df[rsi_col] = talib.RSI(df["close"], timeperiod=tp)

        if "atr" in indicator_config:
            tp = indicator_config["atr"].get("timeperiod", 14)
            atr_col = f"atr_{tp}"
            if atr_col not in df.columns:
                df[atr_col] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=tp)

        if "ema_spread" in indicator_config:
            fast = indicator_config["ema_spread"].get("fastperiod", 12)
            slow = indicator_config["ema_spread"].get("slowperiod", 26)
            if "ema_spread" not in df.columns:
                ema_fast = talib.EMA(df["close"], timeperiod=fast)
                ema_slow = talib.EMA(df["close"], timeperiod=slow)
                df["ema_spread"] = (ema_fast - ema_slow) / df["close"]

        if "bbands" in indicator_config:
            tp = indicator_config["bbands"].get("timeperiod", 20)
            bb_position_col = f"bb_position_{tp}"
            bb_width_col = f"bb_width_{tp}"
            if bb_position_col not in df.columns or bb_width_col not in df.columns:
                up, mid, low = talib.BBANDS(
                    df["close"],
                    timeperiod=tp,
                    nbdevup=indicator_config["bbands"].get("nbdevup", 2),
                    nbdevdn=indicator_config["bbands"].get("nbdevdn", 2)
                )
                df[bb_position_col] = (df["close"] - low) / (up - low)
                df[bb_width_col] = (up - low) / df["close"]

        # === Feature groups ===
        volume_features = [col for col in ["volume", "volume_sma_5"] if col in df.columns]
        volatility_features = [col for col in df.columns if col.startswith("atr_") or col.startswith("close_std_")]
        momentum_features = [col for col in df.columns if col.startswith("rsi_") or col.startswith("macd")]
        trend_features = ["ema_spread"] if "ema_spread" in df.columns else []
        bb_features = [col for col in df.columns if col.startswith("bb_position_") or col.startswith("bb_width_")]

        # === Select and filter ===
        selected_features = (
            base_features +
            volume_features[:1] +       # Limit volume features
            volatility_features[:1] +
            momentum_features[:1] +
            trend_features +
            bb_features
        )

        available_features = [feat for feat in selected_features if feat in df.columns]
        _logger.info("Selected %d features for HMM: %s", len(available_features), available_features)
        return available_features

    def prepare_training_data(self, df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, StandardScaler]:
        """
        Prepare data for HMM training with robust preprocessing for small timeframe noise.
        """
        feature_data = df[features].copy()

        # Fill and replace infinities
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        # Clip extreme outliers
        for col in feature_data.columns:
            if np.issubdtype(feature_data[col].dtype, np.number):
                q1 = feature_data[col].quantile(0.01)
                q99 = feature_data[col].quantile(0.99)
                feature_data[col] = feature_data[col].clip(q1, q99)

        # Scale for HMM
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)

        _logger.info("Prepared training data: %s samples, %d features", scaled_data.shape[0], scaled_data.shape[1])
        return scaled_data, scaler

    def train_hmm_model(self, X: np.ndarray) -> hmm.GaussianHMM:
        """
        Train HMM model on prepared data.

        Args:
            X: Scaled feature matrix

        Returns:
            Trained HMM model
        """
        _logger.info("Training HMM with %d components on %d samples", self.n_components, X.shape[0])

        # Initialize and train HMM
        model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
            random_state=42
        )

        # Fit the model
        model.fit(X)

        # Log training results
        _logger.info("HMM training completed")
        _logger.info("Log likelihood: %.2f", model.score(X))
        _logger.info("Converged: %s", model.monitor_.converged)

        return model

    def validate_hmm_model(self, model: hmm.GaussianHMM, X: np.ndarray, df: pd.DataFrame) -> Dict:
        """
        Validate the trained HMM model.

        Args:
            model: Trained HMM model
            X: Scaled feature matrix used for training
            df: Original DataFrame with timestamps

        Returns:
            Dict with validation metrics
        """
        # Predict regimes
        regimes = model.predict(X)

        # Calculate regime statistics
        regime_counts = pd.Series(regimes).value_counts().sort_index()
        regime_transitions = np.sum(np.diff(regimes) != 0)

        # Calculate regime persistence (average duration)
        regime_durations = []
        current_regime = regimes[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = regimes[i]
                current_duration = 1
        regime_durations.append(current_duration)

        # Calculate validation metrics
        validation_metrics = {
            'n_samples': len(regimes),
            'n_components': self.n_components,
            'log_likelihood': model.score(X),
            'regime_counts': regime_counts.to_dict(),
            'regime_percentages': (regime_counts / len(regimes) * 100).to_dict(),
            'n_transitions': regime_transitions,
            'transition_rate': regime_transitions / len(regimes),
            'avg_regime_duration': np.mean(regime_durations),
            'min_regime_duration': np.min(regime_durations),
            'max_regime_duration': np.max(regime_durations)
        }

        _logger.info("Validation metrics:")
        _logger.info("  Regime distribution: %s", validation_metrics['regime_percentages'])
        _logger.info("  Transition rate: %.4f", validation_metrics['transition_rate'])
        _logger.info("  Avg regime duration: %.2f", validation_metrics['avg_regime_duration'])

        return validation_metrics

    def analyze_regime_characteristics(self, df: pd.DataFrame, regimes: np.ndarray) -> List[str]:
        """
        Analyze the characteristics of each regime to assign proper labels.

        Args:
            df: DataFrame with price data
            regimes: Array of regime predictions

        Returns:
            List of regime labels in order of regime_id
        """
        regime_stats = []

        for regime_id in range(self.n_components):
            mask = regimes == regime_id
            if not np.any(mask):
                regime_stats.append({'regime_id': regime_id, 'avg_return': 0, 'volatility': 0})
                continue

            # Get data for this regime
            regime_data = df.iloc[mask]

            # Calculate average log return
            if 'log_return' in regime_data.columns:
                avg_return = regime_data['log_return'].mean()
                volatility = regime_data['log_return'].std()
            else:
                # Calculate from price if log_return not available
                price_returns = regime_data['close'].pct_change().dropna()
                avg_return = price_returns.mean()
                volatility = price_returns.std()

            regime_stats.append({
                'regime_id': regime_id,
                'avg_return': avg_return,
                'volatility': volatility
            })

        # Sort regimes by average return to assign labels
        regime_stats.sort(key=lambda x: x['avg_return'])

        # Assign labels based on sorted order
        if self.n_components == 3:
            labels = [''] * self.n_components
            for i, stat in enumerate(regime_stats):
                if i == 0:
                    labels[stat['regime_id']] = 'Bearish'
                elif i == 1:
                    labels[stat['regime_id']] = 'Sideways'
                else:
                    labels[stat['regime_id']] = 'Bullish'
        elif self.n_components == 2:
            labels = [''] * self.n_components
            for i, stat in enumerate(regime_stats):
                if i == 0:
                    labels[stat['regime_id']] = 'Bearish'
                else:
                    labels[stat['regime_id']] = 'Bullish'
        else:
            labels = [f'Regime {i}' for i in range(self.n_components)]

        # Log regime characteristics
        _logger.info("Regime characteristics:")
        for stat in regime_stats:
            _logger.info("  Regime %d (%s): avg_return=%.6f, volatility=%.6f",
                       stat['regime_id'], labels[stat['regime_id']], stat['avg_return'], stat['volatility'])

        return labels

    def visualize_regimes(self, model: hmm.GaussianHMM, X: np.ndarray, df: pd.DataFrame,
                         features: List[str], symbol: str, timeframe: str, save_path: Path) -> None:
        """
        Create visualizations of detected regimes.

        Args:
            model: Trained HMM model
            X: Scaled feature matrix
            df: Original DataFrame
            features: Feature names
            symbol: Trading symbol
            timeframe: Timeframe
            save_path: Path to save visualizations
        """
        try:
            # Predict regimes
            regimes = model.predict(X)

            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(60, 12))
            fig.suptitle(f'HMM Regime Detection - {symbol} {timeframe}', fontsize=16)

            # Prepare data for plotting (use recent data to avoid overcrowding)
            plot_data = df.iloc[-min(1000, len(df)):].copy()
            plot_regimes = regimes[-min(1000, len(regimes)):]

            # Convert timestamp to datetime if needed
            if 'timestamp' in plot_data.columns:
                plot_data['timestamp'] = pd.to_datetime(plot_data['timestamp'])
                x_axis = plot_data['timestamp']
            else:
                x_axis = range(len(plot_data))

            # Plot 1: Price with regime overlay
            axes[0].set_title('Price with Detected Regimes')

            # Color map for regimes
            colors = ['red', 'green', 'blue', 'orange', 'purple'][:self.n_components]

            # Analyze regime characteristics and assign proper labels
            regime_labels = self.analyze_regime_characteristics(df, regimes)

            for regime_id in range(self.n_components):
                mask = plot_regimes == regime_id
                if np.any(mask):
                    axes[0].scatter(x_axis[mask], plot_data['close'].iloc[mask],
                                  c=colors[regime_id], alpha=0.6, s=1,
                                  label=f'{regime_labels[regime_id]} (Regime {regime_id})')

            axes[0].plot(x_axis, plot_data['close'], 'k-', alpha=0.3, linewidth=0.5)
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Log returns with regime overlay
            axes[1].set_title('Log Returns with Detected Regimes')

            for regime_id in range(self.n_components):
                mask = plot_regimes == regime_id
                if np.any(mask):
                    axes[1].scatter(x_axis[mask], plot_data['log_return'].iloc[mask],
                                  c=colors[regime_id], alpha=0.6, s=1,
                                  label=f'{regime_labels[regime_id]} (Regime {regime_id})')

            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1].set_ylabel('Log Return')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Plot 3: Regime timeline
            axes[2].set_title('Regime Timeline')
            axes[2].plot(x_axis, plot_regimes, 'o-', markersize=1, linewidth=1)
            axes[2].set_ylabel('Regime')
            axes[2].set_xlabel('Time')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yticks(range(self.n_components))
            axes[2].set_yticklabels(regime_labels)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            _logger.info("Saved regime visualization to %s", save_path)

        except Exception as e:
            _logger.warning("Failed to create regime visualization: %s", str(e))

    def save_model(self, model: hmm.GaussianHMM, scaler: StandardScaler, features: List[str],
                   validation_metrics: Dict, symbol: str, timeframe: str) -> Path:
        """
        Save trained HMM model and associated metadata.

        Args:
            model: Trained HMM model
            scaler: Fitted feature scaler
            features: List of features used
            validation_metrics: Validation metrics
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hmm_{symbol}_{timeframe}_{timestamp}.pkl"
        filepath = self.models_dir / filename

        # Prepare model package
        model_package = {
            'model': model,
            'scaler': scaler,
            'features': features,
            'validation_metrics': validation_metrics,
            'config': {
                'n_components': self.n_components,
                'covariance_type': self.covariance_type,
                'algorithm': self.algorithm,
                'n_iter': self.n_iter
            },
            'metadata': {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': timestamp,
                'training_date': datetime.now().isoformat()
            }
        }

        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)

        _logger.info("Saved HMM model to %s", filepath)
        return filepath

    def train_symbol_timeframe(self, symbol: str, timeframe: str) -> Dict:
        """
        Train HMM for a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with training results
        """
        _logger.info("Training HMM for %s %s", symbol, timeframe)

        try:
            # Find processed data file
            pattern = f"processed_{symbol}_{timeframe}_*.csv"
            csv_files = list(self.processed_data_dir.glob(pattern))

            if not csv_files:
                raise FileNotFoundError(f"No processed data found for {symbol} {timeframe}")

            # Use the most recent file if multiple exist
            csv_file = sorted(csv_files)[-1]
            _logger.info("Using data file: %s", csv_file)

            # Load data
            df = pd.read_csv(csv_file)

            # Apply training window if specified
            if self.train_window_days > 0:
                # Calculate the start date for training window
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    cutoff_date = df['timestamp'].max() - pd.Timedelta(days=self.train_window_days)
                    df = df[df['timestamp'] >= cutoff_date].copy()
                    _logger.info("Applied %d-day training window: %d samples", self.train_window_days, len(df))

            # Select features
            features = self.select_features_for_hmm(df, HMMTrainer.build_indicator_config(timeframe, 240))

            if not features:
                raise ValueError("No suitable features found for HMM training")

            # Prepare training data
            X, scaler = self.prepare_training_data(df, features)

            # Train model
            model = self.train_hmm_model(X)

            # Validate model
            validation_metrics = self.validate_hmm_model(model, X, df)

            # Create visualization
            viz_filename = f"hmm_regimes_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            viz_path = self.models_dir / viz_filename
            self.visualize_regimes(model, X, df, features, symbol, timeframe, viz_path)

            # Save model
            model_path = self.save_model(model, scaler, features, validation_metrics, symbol, timeframe)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'model_path': str(model_path),
                'validation_metrics': validation_metrics,
                'n_samples': len(df),
                'features': features
            }

        except Exception as e:
            error_msg = f"Failed to train HMM for {symbol} {timeframe}: {str(e)}"
            _logger.error(error_msg)
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': error_msg
            }

    def train_all(self) -> Dict:
        """
        Train HMM models for all symbol-timeframe combinations.

        Returns:
            Dict with summary of training results
        """
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']

        _logger.info("Training HMM models for %d symbols x %d timeframes", len(symbols), len(timeframes))

        results = {
            'total': len(symbols) * len(timeframes),
            'successful': [],
            'failed': [],
            'overall_success': True
        }

        for symbol in symbols:
            for timeframe in timeframes:
                result = self.train_symbol_timeframe(symbol, timeframe)

                if result['success']:
                    results['successful'].append(result)
                else:
                    results['failed'].append(result)
                    # Mark overall as failed if any training fails
                    results['overall_success'] = False

        # Log summary
        _logger.info("\n%s", "="*50)
        _logger.info("HMM Training Summary:")
        _logger.info("  Total: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))
        _logger.info("  Overall success: %s", results['overall_success'])
        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.error("Failed training:")
            for failure in results['failed']:
                _logger.error("  %s %s: %s", failure['symbol'], failure['timeframe'], failure['error'])

        return results

def main():
    """Main function to run HMM training."""
    try:
        trainer = HMMTrainer()
        results = trainer.train_all()

        if results['overall_success']:
            _logger.info("HMM training completed successfully!")
        else:
            _logger.error("HMM training completed with failures!")
            # Raise exception to signal pipeline failure
            raise RuntimeError(f"HMM training failed: {len(results['failed'])} out of {results['total']} training tasks failed")

    except Exception as e:
        _logger.exception("HMM training failed")
        raise

if __name__ == "__main__":
    main()
