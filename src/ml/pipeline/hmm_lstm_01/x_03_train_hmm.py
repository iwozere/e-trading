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
import logging
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        self.covariance_type = self.config['hmm'].get('covariance_type', 'diag')
        self.algorithm = self.config['hmm'].get('algorithm', 'viterbi')
        self.n_iter = self.config['hmm'].get('n_iter', 100)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def select_features_for_hmm(self, df: pd.DataFrame) -> List[str]:
        """
        Select optimal features for HMM training.

        Args:
            df: Preprocessed DataFrame

        Returns:
            List of selected feature column names
        """
        # Base features that are typically good for regime detection
        base_features = ['log_return']

        # Volume-based features
        volume_features = []
        if 'volume' in df.columns:
            volume_features.extend(['volume'])
        if 'volume_sma_5' in df.columns:
            volume_features.extend(['volume_sma_5'])

        # Volatility features
        volatility_features = []
        if 'atr_14' in df.columns:
            volatility_features.extend(['atr_14'])
        if 'close_std_10' in df.columns:
            volatility_features.extend(['close_std_10'])

        # Price momentum features
        momentum_features = []
        if 'rsi_14' in df.columns:
            momentum_features.extend(['rsi_14'])
        if 'macd' in df.columns:
            momentum_features.extend(['macd'])

        # Trend features
        trend_features = []
        if 'ema_12' in df.columns and 'ema_26' in df.columns:
            # Create EMA spread feature
            df['ema_spread'] = (df['ema_12'] - df['ema_26']) / df['close']
            trend_features.extend(['ema_spread'])

        # Bollinger Bands features
        bb_features = []
        if 'bb_position_20' in df.columns:
            bb_features.extend(['bb_position_20'])
        if 'bb_width_20' in df.columns:
            bb_features.extend(['bb_width_20'])

        # Combine all features
        selected_features = (base_features +
                           volume_features[:1] +  # Limit volume features
                           volatility_features[:1] +
                           momentum_features[:1] +
                           trend_features +
                           bb_features)

        # Filter to only include features that exist in the DataFrame
        available_features = [feat for feat in selected_features if feat in df.columns]

        logger.info(f"Selected {len(available_features)} features for HMM: {available_features}")
        return available_features

    def prepare_training_data(self, df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, StandardScaler]:
        """
        Prepare data for HMM training.

        Args:
            df: Preprocessed DataFrame
            features: List of feature column names

        Returns:
            Tuple of (scaled feature matrix, fitted scaler)
        """
        # Extract features
        feature_data = df[features].copy()

        # Handle any remaining missing values
        feature_data = feature_data.ffill().bfill()
        feature_data = feature_data.fillna(0)

        # Handle infinite and extremely large values
        feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
        feature_data = feature_data.ffill().bfill()
        feature_data = feature_data.fillna(0)

        # Clip extreme values to prevent numerical issues
        for col in feature_data.columns:
            if feature_data[col].dtype in ['float64', 'float32']:
                # Get the 1st and 99th percentiles
                q1 = feature_data[col].quantile(0.01)
                q99 = feature_data[col].quantile(0.99)
                # Clip values outside this range
                feature_data[col] = feature_data[col].clip(lower=q1, upper=q99)

        # Scale features for HMM
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)

        logger.info(f"Prepared training data: {scaled_data.shape} with features {features}")
        return scaled_data, scaler

    def train_hmm_model(self, X: np.ndarray) -> hmm.GaussianHMM:
        """
        Train HMM model on prepared data.

        Args:
            X: Scaled feature matrix

        Returns:
            Trained HMM model
        """
        logger.info(f"Training HMM with {self.n_components} components on {X.shape[0]} samples")

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
        logger.info(f"HMM training completed")
        logger.info(f"Log likelihood: {model.score(X):.2f}")
        logger.info(f"Converged: {model.monitor_.converged}")

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

        logger.info(f"Validation metrics:")
        logger.info(f"  Regime distribution: {validation_metrics['regime_percentages']}")
        logger.info(f"  Transition rate: {validation_metrics['transition_rate']:.4f}")
        logger.info(f"  Avg regime duration: {validation_metrics['avg_regime_duration']:.2f}")

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
        logger.info(f"Regime characteristics:")
        for stat in regime_stats:
            logger.info(f"  Regime {stat['regime_id']} ({labels[stat['regime_id']]}): "
                       f"avg_return={stat['avg_return']:.6f}, volatility={stat['volatility']:.6f}")

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

            logger.info(f"Saved regime visualization to {save_path}")

        except Exception as e:
            logger.warning(f"Failed to create regime visualization: {str(e)}")

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

        logger.info(f"Saved HMM model to {filepath}")
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
        logger.info(f"Training HMM for {symbol} {timeframe}")

        try:
            # Find processed data file
            pattern = f"processed_{symbol}_{timeframe}_*.csv"
            csv_files = list(self.processed_data_dir.glob(pattern))

            if not csv_files:
                raise FileNotFoundError(f"No processed data found for {symbol} {timeframe}")

            # Use the most recent file if multiple exist
            csv_file = sorted(csv_files)[-1]
            logger.info(f"Using data file: {csv_file}")

            # Load data
            df = pd.read_csv(csv_file)

            # Apply training window if specified
            if self.train_window_days > 0:
                # Calculate the start date for training window
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    cutoff_date = df['timestamp'].max() - pd.Timedelta(days=self.train_window_days)
                    df = df[df['timestamp'] >= cutoff_date].copy()
                    logger.info(f"Applied {self.train_window_days}-day training window: {len(df)} samples")

            # Select features
            features = self.select_features_for_hmm(df)

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
            logger.error(error_msg)
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

        logger.info(f"Training HMM models for {len(symbols)} symbols x {len(timeframes)} timeframes")

        results = {
            'total': len(symbols) * len(timeframes),
            'successful': [],
            'failed': []
        }

        for symbol in symbols:
            for timeframe in timeframes:
                result = self.train_symbol_timeframe(symbol, timeframe)

                if result['success']:
                    results['successful'].append(result)
                else:
                    results['failed'].append(result)

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"HMM Training Summary:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"{'='*50}")

        if results['failed']:
            logger.warning("Failed training:")
            for failure in results['failed']:
                logger.warning(f"  {failure['symbol']} {failure['timeframe']}: {failure['error']}")

        return results

def main():
    """Main function to run HMM training."""
    try:
        trainer = HMMTrainer()
        results = trainer.train_all()

        logger.info("HMM training completed!")

    except Exception as e:
        logger.error(f"HMM training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
