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
    def __init__(self, config_path: str = "config/pipeline/p01.yaml"):
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

            # Check if model uses old fixed-period features
            features = model_package['features']
            fixed_period_features = [feat for feat in features
                                   if any(indicator in feat for indicator in ['rsi_', 'bb_', 'macd', 'ema_', 'atr_', 'stoch', 'williams', 'mfi', 'sma'])]

            if fixed_period_features:
                _logger.warning("Model uses fixed-period indicators: %s", fixed_period_features)
                _logger.warning("This model was trained with the old preprocessing approach.")
                _logger.warning("Consider retraining with the new dynamic feature approach for better performance.")

            return model_package

        except Exception:
            _logger.exception("Failed to load model from %s: ", model_path)
            raise

    def prepare_features(self, df: pd.DataFrame, required_features: List[str], timeframe: str = "4h") -> np.ndarray:
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
            _logger.info("Attempting to create missing features dynamically...")

            # Check if these are fixed-period indicators that we can compute dynamically
            fixed_period_indicators = [feat for feat in missing_features
                                     if any(indicator in feat for indicator in ['rsi_', 'bb_', 'macd', 'ema_', 'atr_', 'stoch', 'williams', 'mfi', 'sma'])]

            if fixed_period_indicators:
                # Extract timeframe from the model metadata or use a default
                # For now, we'll need to pass timeframe to this method
                # This is a temporary fix - in the future, timeframe should be stored in model metadata
                _logger.info("Computing indicators dynamically using the same logic as HMM training")
                # Note: We need to get timeframe from somewhere - for now, we'll use a default
                # In the apply_hmm_to_file method, we'll pass the timeframe
                df = self._compute_indicators_dynamically(df, timeframe)
            else:
                # Try to create other missing features if possible
                df = self._create_missing_features(df, missing_features)

            # Check again
            missing_features = [feat for feat in required_features if feat not in df.columns]
            if missing_features:
                error_msg = f"Cannot create missing features: {missing_features}"
                raise ValueError(error_msg)

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

        # Return as DataFrame to preserve feature names for StandardScaler
        return feature_data

    def _create_missing_features(self, df: pd.DataFrame, missing_features: List[str]) -> pd.DataFrame:
        """
        Attempt to create missing features that might be derivable.

        Note: This method is primarily for backward compatibility. The new pipeline
        uses dynamic feature generation with optimized parameters, so fixed-period
        indicators like 'rsi_14' should not be expected in the data.

        Args:
            df: DataFrame
            missing_features: List of missing feature names

        Returns:
            DataFrame with additional features created if possible
        """
        df = df.copy()

        for feature in missing_features:
            try:
                if feature == 'ema_spread' and 'ema_fast' in df.columns and 'ema_slow' in df.columns:
                    df['ema_spread'] = (df['ema_fast'] - df['ema_slow']) / df['close']
                    _logger.info("Created missing feature: %s", feature)
                elif feature == 'ema_spread':
                    _logger.warning("Cannot create %s: missing required columns 'ema_fast' or 'ema_slow'", feature)
                    _logger.info("Available columns: %s", list(df.columns))
                elif any(indicator in feature for indicator in ['rsi_', 'bb_', 'macd', 'ema_', 'atr_', 'stoch', 'williams', 'mfi', 'sma']):
                    # These are fixed-period indicators that are no longer generated
                    # by the new preprocessing approach. The HMM model needs to be
                    # retrained with the new dynamic features.
                    _logger.error("Missing fixed-period indicator: %s", feature)
                    _logger.error("This indicates the HMM model was trained with old fixed-period features.")
                    _logger.error("The model needs to be retrained with the new dynamic feature approach.")
                    _logger.error("Available features: %s", [col for col in df.columns if any(indicator in col for indicator in ['rsi', 'bb', 'macd', 'ema', 'atr', 'stoch', 'williams', 'mfi', 'sma'])])
                    _logger.error("Please run the pipeline from stage 3 (HMM training) to regenerate models with optimized features.")
                else:
                    _logger.warning("Cannot create missing feature: %s", feature)
            except Exception as e:
                _logger.warning("Error creating feature %s: %s", feature, str(e))

        return df

    def _compute_indicators_dynamically(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Compute indicators dynamically using the same logic as HMM training.

        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe string (e.g., '4h')

        Returns:
            DataFrame with computed indicators
        """
        df = df.copy()

        # Convert timeframe to minutes
        if timeframe.endswith('m'):
            timeframe_minutes = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            timeframe_minutes = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            timeframe_minutes = int(timeframe[:-1]) * 60 * 24
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")

        # Build indicator config using the same logic as HMM training
        base_periods = {
            'rsi': 14,
            'atr': 14,
            'ema_fast': 12,
            'ema_slow': 26,
            'bbands': 20
        }

        # Adjust periods depending on timeframe scale
        if timeframe.endswith('m'):
            multiplier = int(timeframe[:-1])
        elif timeframe.endswith('h'):
            multiplier = int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            multiplier = int(timeframe[:-1]) * 60 * 24
        else:
            multiplier = 1

        # Scale indicator periods inversely to timeframe
        scale_factor = (timeframe_minutes / multiplier) ** 0.5

        def scaled_period(base):
            val = max(2, int(round(base * scale_factor)))
            return val

        # Compute indicators
        import talib

        # RSI
        rsi_period = scaled_period(base_periods['rsi'])
        rsi_col = f"rsi_{rsi_period}"
        if rsi_col not in df.columns:
            df[rsi_col] = talib.RSI(df["close"], timeperiod=rsi_period)
            _logger.info("Computed %s with period %d", rsi_col, rsi_period)

        # Bollinger Bands
        bb_period = scaled_period(base_periods['bbands'])
        bb_position_col = f"bb_position_{bb_period}"
        bb_width_col = f"bb_width_{bb_period}"
        if bb_position_col not in df.columns or bb_width_col not in df.columns:
            up, mid, low = talib.BBANDS(
                df["close"],
                timeperiod=bb_period,
                nbdevup=2,
                nbdevdn=2
            )
            df[bb_position_col] = (df["close"] - low) / (up - low)
            df[bb_width_col] = (up - low) / df["close"]
            _logger.info("Computed %s and %s with period %d", bb_position_col, bb_width_col, bb_period)

        # EMA Spread
        if "ema_spread" not in df.columns:
            ema_fast_period = scaled_period(base_periods['ema_fast'])
            ema_slow_period = scaled_period(base_periods['ema_slow'])
            ema_fast = talib.EMA(df["close"], timeperiod=ema_fast_period)
            ema_slow = talib.EMA(df["close"], timeperiod=ema_slow_period)
            df["ema_spread"] = (ema_fast - ema_slow) / df["close"]
            _logger.info("Computed ema_spread with periods %d, %d", ema_fast_period, ema_slow_period)

        return df

    def predict_regimes(self, model_package: Dict, df: pd.DataFrame, timeframe: str = "4h") -> Tuple[np.ndarray, np.ndarray]:
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
        X = self.prepare_features(df, features, timeframe)

        # Scale features using the same scaler as training
        # X is now a DataFrame, so we pass it directly to preserve feature names
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
            regimes, posteriors = self.predict_regimes(model_package, df, timeframe)

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

    def apply_hmm_to_file_multi_provider(self, symbol: str, timeframe: str, provider: str) -> Dict:
        """
        Apply HMM model to a specific symbol-timeframe combination with provider information.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            provider: Data provider (e.g., 'binance', 'yfinance')

        Returns:
            Dict with application results
        """
        _logger.info("Applying HMM for %s %s from %s", symbol, timeframe, provider)

        try:
            # Find processed data file with provider prefix
            pattern = f"processed_{provider}_{symbol}_{timeframe}_*.csv"
            csv_files = list(self.processed_data_dir.glob(pattern))

            if not csv_files:
                # Try legacy pattern without provider prefix
                pattern = f"processed_{symbol}_{timeframe}_*.csv"
                csv_files = list(self.processed_data_dir.glob(pattern))

                if not csv_files:
                    raise FileNotFoundError(f"No processed data found for {provider} {symbol} {timeframe}")

            # Use the most recent file if multiple exist
            csv_file = sorted(csv_files)[-1]
            _logger.info("Using data file: %s", csv_file)

            # Find and load the corresponding HMM model
            model_path = self.find_latest_model(symbol, timeframe)
            if not model_path:
                raise FileNotFoundError(f"No HMM model found for {provider} {symbol} {timeframe}")

            model_package = self.load_model(model_path)
            _logger.info("Loaded HMM model: %s", model_path)

            # Load processed data
            df = pd.read_csv(csv_file)
            original_shape = df.shape
            _logger.info("Loaded data: %s", original_shape)

            # Predict regimes
            regimes, posteriors = self.predict_regimes(model_package, df, timeframe)

            # Add regime features
            df_labeled = self.add_regime_features(df, regimes, posteriors)

            # Validate regime quality
            quality_metrics = self.validate_regime_quality(df_labeled)

            # Save labeled data with provider prefix
            output_filename = f"labeled_{provider}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_path = self.labeled_data_dir / output_filename
            df_labeled.to_csv(output_path, index=False)

            _logger.info("[OK] Saved labeled data: %s -> %s to %s", original_shape, df_labeled.shape, output_path)

            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'provider': provider,
                'success': True,
                'input_file': str(csv_file),
                'output_file': str(output_path),
                'model_file': str(model_path),
                'original_shape': original_shape,
                'labeled_shape': df_labeled.shape,
                'quality_metrics': quality_metrics
            }

        except Exception as e:
            error_msg = f"Failed to apply HMM for {provider} {symbol} {timeframe}: {str(e)}"
            _logger.error(error_msg)
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'provider': provider,
                'success': False,
                'error': error_msg
            }

    def apply_all(self) -> Dict:
        """
        Apply HMM models to all symbol-timeframe combinations.

        Returns:
            Dict with summary of application results
        """
        # Check if using multi-provider configuration
        if 'data_sources' in self.config:
            _logger.info("Using multi-provider configuration")
            results = self._apply_all_multi_provider()
        else:
            # Legacy configuration
            _logger.info("Using legacy configuration")
            symbols = self.config['symbols']
            timeframes = self.config['timeframes']
            results = self._apply_all_legacy(symbols, timeframes)

        return results

    def _apply_all_multi_provider(self) -> Dict:
        """
        Apply HMM models using multi-provider configuration.

        Returns:
            Dict with summary of application results
        """
        data_sources = self.config['data_sources']
        total_tasks = 0
        all_tasks = []

        # Count total tasks and collect all symbol-timeframe-provider combinations
        for provider, config in data_sources.items():
            symbols = config['symbols']
            timeframes = config['timeframes']
            total_tasks += len(symbols) * len(timeframes)

            for symbol in symbols:
                for timeframe in timeframes:
                    all_tasks.append((symbol, timeframe, provider))

        _logger.info("Applying HMM models for %d tasks across %d providers", total_tasks, len(data_sources))

        results = {
            'total': total_tasks,
            'successful': [],
            'failed': []
        }

        for symbol, timeframe, provider in all_tasks:
            result = self.apply_hmm_to_file_multi_provider(symbol, timeframe, provider)

            if result['success']:
                results['successful'].append(result)
            else:
                results['failed'].append(result)

        return results

    def _apply_all_legacy(self, symbols: List[str], timeframes: List[str]) -> Dict:
        """
        Apply HMM models using legacy configuration.

        Args:
            symbols: List of symbols
            timeframes: List of timeframes

        Returns:
            Dict with summary of application results
        """
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

        return results

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

        # Check if any applications failed
        if results['failed']:
            error_msg = f"HMM application failed: {len(results['failed'])} out of {results['total']} applications failed"
            _logger.error(error_msg)
            raise RuntimeError(error_msg)

        _logger.info("HMM application completed successfully!")

        # List created files
        labeled_files = applicator.list_labeled_files()
        _logger.info("Created %d labeled data files:", len(labeled_files))
        for file in labeled_files[-5:]:  # Show last 5 files
            _logger.info("  %s", file.name)

    except Exception:
        _logger.exception("HMM application failed: ")
        raise

if __name__ == "__main__":
    main()
