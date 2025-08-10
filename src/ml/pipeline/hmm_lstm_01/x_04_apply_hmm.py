"""
HMM Application for Market Regime Labeling

This module applies trained HMM models to preprocessed data to generate
regime labels for each time step. These regime labels will be used as
features in the LSTM model for price prediction.

Features:
- Loads trained HMM models and applies them to data
- Generates regime labels with confidence scores
- Handles missing or invalid data gracefully
- Saves labeled data for LSTM training
- Supports batch processing of multiple symbols/timeframes
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import pickle
from datetime import datetime
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class HMMApplicator:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize HMM applicator with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.processed_data_dir = Path(self.config['paths']['data_processed'])
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.models_dir = Path(self.config['paths']['models_hmm'])

        # Create labeled data directory
        self.labeled_data_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def find_latest_model(self, symbol: str, timeframe: str) -> Optional[Path]:
        """
        Find the latest trained HMM model for a symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Path to latest model file or None if not found
        """
        pattern = f"hmm_{symbol}_{timeframe}_*.pkl"
        model_files = list(self.models_dir.glob(pattern))

        if not model_files:
            _logger.warning("No HMM model found for %s %s", symbol, timeframe)
            return None

        # Return the most recent model
        latest_model = sorted(model_files)[-1]
        _logger.info("Found latest model: %s", latest_model)
        return latest_model

    def load_model(self, model_path: Path) -> Dict:
        """
        Load trained HMM model and metadata.

        Args:
            model_path: Path to model pickle file

        Returns:
            Dict containing model, scaler, features, and metadata
        """
        try:
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)

            _logger.info("Loaded HMM model from %s", model_path)
            _logger.info("Model features: %s", model_package['features'])
            _logger.info("Model components: %d", model_package['config']['n_components'])

            return model_package

        except Exception as e:
            _logger.error("Failed to load model from %s: %s", model_path, str(e))
            raise

    def prepare_features(self, df: pd.DataFrame, required_features: List[str]) -> np.ndarray:
        """
        Prepare features for HMM prediction using the same features as training.

        Args:
            df: Preprocessed DataFrame
            required_features: List of features required by the model

        Returns:
            Prepared feature matrix
        """
        # Check if all required features are available
        missing_features = [feat for feat in required_features if feat not in df.columns]

        if missing_features:
            _logger.warning("Missing features: %s", missing_features)
            _logger.info("Attempting to create missing features...")
            # Try to create missing features if possible
            df = self._create_missing_features(df, missing_features)

            # Check again
            missing_features = [feat for feat in required_features if feat not in df.columns]
            if missing_features:
                raise ValueError(f"Cannot create missing features: {missing_features}")

        # Extract features
        feature_data = df[required_features].copy()

        # Handle missing values
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

        return feature_data.values

    def _create_missing_features(self, df: pd.DataFrame, missing_features: List[str]) -> pd.DataFrame:
        """
        Attempt to create missing features that might be derivable.

        Args:
            df: DataFrame
            missing_features: List of missing feature names

        Returns:
            DataFrame with additional features created if possible
        """
        df = df.copy()

        for feature in missing_features:
            try:
                if feature == 'ema_spread' and 'ema_12' in df.columns and 'ema_26' in df.columns:
                    df['ema_spread'] = (df['ema_12'] - df['ema_26']) / df['close']
                    _logger.info("Created missing feature: %s", feature)
                elif feature == 'ema_spread':
                    _logger.warning("Cannot create %s: missing required columns 'ema_12' or 'ema_26'", feature)
                    _logger.info("Available columns: %s", list(df.columns))
                else:
                    _logger.warning("Cannot create missing feature: %s", feature)
            except Exception as e:
                _logger.warning("Error creating feature %s: %s", feature, str(e))

        return df

    def predict_regimes(self, model_package: Dict, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes using the trained HMM model.

        Args:
            model_package: Loaded model package
            df: Preprocessed DataFrame

        Returns:
            Tuple of (regime predictions, posterior probabilities)
        """
        model = model_package['model']
        scaler = model_package['scaler']
        features = model_package['features']

        # Prepare features
        X = self.prepare_features(df, features)

        # Scale features using the same scaler as training
        X_scaled = scaler.transform(X)

        # Predict regimes
        regimes = model.predict(X_scaled)

        # Get posterior probabilities for confidence assessment
        try:
            posteriors = model.predict_proba(X_scaled)
        except Exception as e:
            _logger.warning("Could not compute posterior probabilities: %s", str(e))
            # Create dummy probabilities
            posteriors = np.zeros((len(regimes), model.n_components))
            for i, regime in enumerate(regimes):
                posteriors[i, regime] = 1.0

        _logger.info("Predicted regimes for %d samples", len(regimes))
        _logger.info("Regime distribution: %s", np.bincount(regimes, minlength=model.n_components))

        return regimes, posteriors

    def add_regime_features(self, df: pd.DataFrame, regimes: np.ndarray, posteriors: np.ndarray) -> pd.DataFrame:
        """
        Add regime-related features to the DataFrame.

        Args:
            df: Original DataFrame
            regimes: Regime predictions
            posteriors: Posterior probabilities

        Returns:
            DataFrame with additional regime features
        """
        df = df.copy()

        # Add primary regime label
        df['regime'] = regimes

        # Add posterior probabilities for each regime
        n_components = posteriors.shape[1]
        for i in range(n_components):
            df[f'regime_prob_{i}'] = posteriors[:, i]

        # Add regime confidence (max posterior probability)
        df['regime_confidence'] = np.max(posteriors, axis=1)

        # Add regime stability features
        df['regime_changed'] = (df['regime'] != df['regime'].shift(1)).astype(int)

        # Add regime duration (how long in current regime)
        regime_duration = np.zeros(len(regimes))
        current_regime = regimes[0]
        current_duration = 1

        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                current_regime = regimes[i]
                current_duration = 1
            regime_duration[i] = current_duration

        df['regime_duration'] = regime_duration

        _logger.info("Added regime features: regime, regime_prob_*, regime_confidence, regime_changed, regime_duration")

        return df

    def validate_regime_quality(self, df: pd.DataFrame) -> Dict:
        """
        Validate the quality of regime predictions.

        Args:
            df: DataFrame with regime predictions

        Returns:
            Dict with quality metrics
        """
        regimes = df['regime'].values

        # Basic statistics
        regime_counts = pd.Series(regimes).value_counts().sort_index()
        n_transitions = np.sum(np.diff(regimes) != 0)

        # Regime persistence analysis
        regime_durations = df['regime_duration'].values
        avg_duration = np.mean(regime_durations)

        # Confidence analysis
        avg_confidence = df['regime_confidence'].mean()
        low_confidence_pct = (df['regime_confidence'] < 0.6).mean() * 100

        # Regime balance (how evenly distributed are the regimes)
        regime_balance = 1 - (regime_counts.max() - regime_counts.min()) / len(regimes)

        quality_metrics = {
            'n_samples': len(regimes),
            'n_unique_regimes': len(regime_counts),
            'regime_distribution': regime_counts.to_dict(),
            'regime_percentages': (regime_counts / len(regimes) * 100).to_dict(),
            'n_transitions': n_transitions,
            'transition_rate': n_transitions / len(regimes),
            'avg_regime_duration': avg_duration,
            'max_regime_duration': np.max(regime_durations),
            'avg_confidence': avg_confidence,
            'low_confidence_percentage': low_confidence_pct,
            'regime_balance': regime_balance
        }

        _logger.info("Regime quality metrics:")
        _logger.info("  Average confidence: %.3f", avg_confidence)
        _logger.info("  Transition rate: %.4f", quality_metrics['transition_rate'])
        _logger.info("  Average duration: %.2f", avg_duration)
        _logger.info("  Regime balance: %.3f", regime_balance)

        return quality_metrics

    def apply_hmm_to_file(self, symbol: str, timeframe: str) -> Dict:
        """
        Apply HMM model to a specific symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dict with application results
        """
        _logger.info("Applying HMM for %s %s", symbol, timeframe)

        try:
            # Find and load model
            model_path = self.find_latest_model(symbol, timeframe)
            if model_path is None:
                raise FileNotFoundError(f"No trained HMM model found for {symbol} {timeframe}")

            model_package = self.load_model(model_path)

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
            original_shape = df.shape

            # Predict regimes
            regimes, posteriors = self.predict_regimes(model_package, df)

            # Add regime features
            df_labeled = self.add_regime_features(df, regimes, posteriors)

            # Validate regime quality
            quality_metrics = self.validate_regime_quality(df_labeled)

            # Save labeled data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"labeled_{symbol}_{timeframe}_{timestamp}.csv"
            output_path = self.labeled_data_dir / output_filename

            df_labeled.to_csv(output_path, index=False)

            _logger.info("[OK] Saved labeled data: %s -> %s to %s", original_shape, df_labeled.shape, output_path)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': True,
                'input_file': str(csv_file),
                'output_file': str(output_path),
                'model_file': str(model_path),
                'original_shape': original_shape,
                'labeled_shape': df_labeled.shape,
                'quality_metrics': quality_metrics
            }

        except Exception as e:
            error_msg = f"Failed to apply HMM for {symbol} {timeframe}: {str(e)}"
            _logger.error(error_msg)
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'success': False,
                'error': error_msg
            }

    def apply_all(self) -> Dict:
        """
        Apply HMM models to all symbol-timeframe combinations.

        Returns:
            Dict with summary of application results
        """
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']

        _logger.info("Applying HMM models for %d symbols x %d timeframes", len(symbols), len(timeframes))

        results = {
            'total': len(symbols) * len(timeframes),
            'successful': [],
            'failed': []
        }

        for symbol in symbols:
            for timeframe in timeframes:
                result = self.apply_hmm_to_file(symbol, timeframe)

                if result['success']:
                    results['successful'].append(result)
                else:
                    results['failed'].append(result)

        # Log summary
        _logger.info("\n%s", "="*50)
        _logger.info("HMM Application Summary:")
        _logger.info("  Total: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))
        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.warning("Failed applications:")
            for failure in results['failed']:
                _logger.warning("  %s %s: %s", failure['symbol'], failure['timeframe'], failure['error'])

        return results

    def list_labeled_files(self) -> List[Path]:
        """List all labeled CSV files."""
        csv_files = list(self.labeled_data_dir.glob("labeled_*.csv"))
        return sorted(csv_files)

def main():
    """Main function to run HMM application."""
    try:
        applicator = HMMApplicator()
        results = applicator.apply_all()

        _logger.info("HMM application completed!")

        # List created files
        labeled_files = applicator.list_labeled_files()
        _logger.info("Created %d labeled data files:", len(labeled_files))
        for file in labeled_files[-5:]:  # Show last 5 files
            _logger.info("  %s", file.name)

    except Exception as e:
        _logger.error("HMM application failed: %s", str(e))
        raise

if __name__ == "__main__":
    main()
