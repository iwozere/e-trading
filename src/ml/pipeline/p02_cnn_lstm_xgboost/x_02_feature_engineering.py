"""
Feature Engineering for CNN-LSTM-XGBoost Trading Pipeline

This module handles feature engineering for the CNN-LSTM-XGBoost pipeline, including:
- Technical indicators calculation using TA-Lib
- Data normalization and preprocessing
- Sequence preparation for LSTM input
- Feature validation and quality checks

Features:
- Configurable technical indicators via YAML
- Multiple normalization methods (MinMax, Standard, Robust)
- Missing data handling strategies
- Outlier detection and handling
- Sequence preparation for time series modeling
"""

import pandas as pd
import numpy as np
import yaml
import talib
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
        """
        Initialize FeatureEngineer with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Directory setup
        self.raw_data_dir = Path(self.config['paths']['data_raw'])
        self.labeled_data_dir = Path(self.config['paths']['data_labeled'])
        self.labeled_data_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.technical_indicators_config = self.config.get('technical_indicators', {})
        self.feature_engineering_config = self.config.get('feature_engineering', {})
        self.cnn_lstm_config = self.config.get('cnn_lstm', {})
        self.evaluation_config = self.config.get('evaluation', {})

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        _logger.info("Calculating technical indicators...")

        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Extract price and volume data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # RSI
        rsi_period = self.technical_indicators_config.get('rsi_period', 14)
        df['RSI'] = talib.RSI(close, timeperiod=rsi_period)

        # MACD
        macd_fast = self.technical_indicators_config.get('macd_fast', 12)
        macd_slow = self.technical_indicators_config.get('macd_slow', 26)
        macd_signal = self.technical_indicators_config.get('macd_signal', 9)
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
        df['MACD'] = macd
        df['MACD_SIGNAL'] = macdsignal
        df['MACD_HIST'] = macdhist

        # Bollinger Bands
        bb_period = self.technical_indicators_config.get('bb_period', 20)
        bb_std = self.technical_indicators_config.get('bb_std', 2)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std)
        df['BB_UPPER'] = bb_upper
        df['BB_MIDDLE'] = bb_middle
        df['BB_LOWER'] = bb_lower
        df['BB_WIDTH'] = (bb_upper - bb_lower) / bb_middle  # Bollinger Band Width
        df['BB_POSITION'] = (close - bb_lower) / (bb_upper - bb_lower)  # Bollinger Band Position

        # ATR (Average True Range)
        atr_period = self.technical_indicators_config.get('atr_period', 14)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=atr_period)

        # ADX (Average Directional Index)
        adx_period = self.technical_indicators_config.get('adx_period', 14)
        df['ADX'] = talib.ADX(high, low, close, timeperiod=adx_period)

        # OBV (On-Balance Volume)
        df['OBV'] = talib.OBV(close, volume)

        # Moving Averages
        sma_periods = self.technical_indicators_config.get('sma_periods', [10, 20, 50])
        for period in sma_periods:
            df[f'SMA_{period}'] = talib.SMA(close, timeperiod=period)

        ema_periods = self.technical_indicators_config.get('ema_periods', [10, 20, 50])
        for period in ema_periods:
            df[f'EMA_{period}'] = talib.EMA(close, timeperiod=period)

        # Stochastic Oscillator
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

        # Williams %R
        df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        # Commodity Channel Index
        df['CCI'] = talib.CCI(high, low, close, timeperiod=14)

        # Money Flow Index
        df['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)

        # Rate of Change
        df['ROC'] = talib.ROC(close, timeperiod=10)

        # Price Rate of Change
        df['PROC'] = (close - np.roll(close, 10)) / np.roll(close, 10)

        # Volume Rate of Change
        df['VROC'] = (volume - np.roll(volume, 10)) / np.roll(volume, 10)

        # Price and Volume features
        df['PRICE_CHANGE'] = df['close'].pct_change()
        df['VOLUME_CHANGE'] = df['volume'].pct_change()
        df['HIGH_LOW_RATIO'] = df['high'] / df['low']
        df['CLOSE_OPEN_RATIO'] = df['close'] / df['open']

        # Log returns
        df['LOG_RETURNS'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility (rolling standard deviation of returns)
        df['VOLATILITY'] = df['LOG_RETURNS'].rolling(window=20).std()

        _logger.info("Technical indicators calculation completed")
        return df

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the DataFrame.

        Args:
            df: DataFrame with potential missing values

        Returns:
            DataFrame with missing values handled
        """
        strategy = self.feature_engineering_config.get('missing_data_strategy', 'forward_fill')

        _logger.info("Handling missing data using strategy: %s", strategy)

        if strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif strategy == 'interpolate':
            df = df.interpolate(method='linear')
        elif strategy == 'drop':
            df = df.dropna()
        else:
            _logger.warning("Unknown missing data strategy: %s, using forward_fill", strategy)
            df = df.fillna(method='ffill')

        # Remove any remaining NaN values at the beginning
        df = df.dropna()

        _logger.info("Missing data handling completed. DataFrame shape: %s", df.shape)
        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in the DataFrame.

        Args:
            df: DataFrame with potential outliers

        Returns:
            DataFrame with outliers handled
        """
        strategy = self.feature_engineering_config.get('outlier_handling', 'iqr')

        if strategy == 'none':
            return df

        _logger.info("Handling outliers using strategy: %s", strategy)

        # Select numeric columns for outlier handling
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if strategy == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

            elif strategy == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                # Cap values with z-score > 3
                df[column] = df[column].mask(z_scores > 3, df[column].median())

        _logger.info("Outlier handling completed")
        return df

    def normalize_data(self, df: pd.DataFrame, scaler=None) -> Tuple[pd.DataFrame, object]:
        """
        Normalize the data using the specified method.

        Args:
            df: DataFrame to normalize
            scaler: Pre-fitted scaler (optional)

        Returns:
            Tuple of (normalized_dataframe, fitted_scaler)
        """
        normalization_method = self.feature_engineering_config.get('normalization', 'minmax')

        _logger.info("Normalizing data using method: %s", normalization_method)

        # Select features to normalize (exclude datetime index and target variables)
        feature_columns = [col for col in df.columns if col not in ['datetime', 'date', 'target']]

        if scaler is None:
            if normalization_method == 'minmax':
                scaler = MinMaxScaler()
            elif normalization_method == 'standard':
                scaler = StandardScaler()
            elif normalization_method == 'robust':
                scaler = RobustScaler()
            else:
                _logger.warning("Unknown normalization method: %s, using MinMaxScaler", normalization_method)
                scaler = MinMaxScaler()

            # Fit the scaler
            scaler.fit(df[feature_columns])

        # Transform the data
        normalized_features = scaler.transform(df[feature_columns])

        # Create new DataFrame with normalized features
        normalized_df = df.copy()
        normalized_df[feature_columns] = normalized_features

        _logger.info("Data normalization completed")
        return normalized_df, scaler

    def prepare_sequences(self, df: pd.DataFrame, target_column: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM input.

        Args:
            df: DataFrame with features
            target_column: Column to use as target

        Returns:
            Tuple of (X, y) where X is input sequences and y is target values
        """
        time_steps = self.cnn_lstm_config.get('time_steps', 20)

        _logger.info("Preparing sequences with time_steps: %s", time_steps)

        # Select feature columns (exclude datetime index and target)
        feature_columns = [col for col in df.columns if col not in ['datetime', 'date', target_column]]

        # Prepare features and target
        features = df[feature_columns].values
        target = df[target_column].values

        X, y = [], []

        for i in range(len(features) - time_steps):
            X.append(features[i:i + time_steps])
            y.append(target[i + time_steps])

        X = np.array(X)
        y = np.array(y)

        _logger.info("Sequence preparation completed. X shape: %s, y shape: %s", X.shape, y.shape)
        return X, y

    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Input features
            y: Target values

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        test_split = self.evaluation_config.get('test_split', 0.2)
        validation_split = self.evaluation_config.get('validation_split', 0.2)

        _logger.info("Splitting data with test_split: %s, validation_split: %s", test_split, validation_split)

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, shuffle=False
        )

        # Second split: train vs val
        val_split_adjusted = validation_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split_adjusted, shuffle=False
        )

        _logger.info("Data splitting completed:")
        _logger.info("  Train: %s samples", X_train.shape[0])
        _logger.info("  Validation: %s samples", X_val.shape[0])
        _logger.info("  Test: %s samples", X_test.shape[0])

        return X_train, X_val, X_test, y_train, y_val, y_test

    def process_single_file(self, filepath: Path) -> Dict[str, any]:
        """
        Process a single data file.

        Args:
            filepath: Path to the CSV file

        Returns:
            Dictionary with processing results
        """
        try:
            _logger.info("Processing file: %s", filepath.name)

            # Load data
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)

            # Handle missing data
            df = self.handle_missing_data(df)

            # Handle outliers
            df = self.handle_outliers(df)

            # Normalize data
            df_normalized, scaler = self.normalize_data(df)

            # Prepare sequences
            X, y = self.prepare_sequences(df_normalized)

            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

            # Save processed data
            output_filename = f"processed_{filepath.stem}.npz"
            output_path = self.labeled_data_dir / output_filename

            np.savez_compressed(
                output_path,
                X_train=X_train,
                X_val=X_val,
                X_test=X_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                feature_names=df_normalized.columns.tolist(),
                scaler_params={
                    'scale_': scaler.scale_,
                    'mean_': scaler.mean_,
                    'var_': scaler.var_,
                    'min_': scaler.min_,
                    'data_min_': scaler.data_min_,
                    'data_max_': scaler.data_max_,
                    'data_range_': scaler.data_range_
                } if hasattr(scaler, 'scale_') else {
                    'scale_': scaler.scale_,
                    'min_': scaler.min_,
                    'data_min_': scaler.data_min_,
                    'data_max_': scaler.data_max_,
                    'data_range_': scaler.data_range_
                }
            )

            # Save processed DataFrame
            df_output_filename = f"features_{filepath.stem}.csv"
            df_output_path = self.labeled_data_dir / df_output_filename
            df_normalized.to_csv(df_output_path)

            _logger.info("Successfully processed %s", filepath.name)

            return {
                'success': True,
                'filename': filepath.name,
                'output_file': output_filename,
                'features_file': df_output_filename,
                'X_shape': X.shape,
                'y_shape': y.shape,
                'feature_count': len(df_normalized.columns),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test)
            }

        except Exception as e:
            _logger.exception("Error processing %s:", filepath.name)
            return {
                'success': False,
                'filename': filepath.name,
                'error': str(e)
            }

    def run(self) -> Dict[str, List[Dict]]:
        """
        Run the feature engineering process.

        Returns:
            Dictionary with 'success' and 'failed' lists of processing results
        """
        _logger.info("Starting feature engineering process...")

        # Get all CSV files in the raw data directory
        csv_files = list(self.raw_data_dir.glob("*.csv"))

        if not csv_files:
            _logger.warning("No CSV files found in raw data directory")
            return {'success': [], 'failed': []}

        _logger.info("Found %s CSV files to process", len(csv_files))

        # Process each file
        successful_processing = []
        failed_processing = []

        for filepath in csv_files:
            result = self.process_single_file(filepath)

            if result['success']:
                successful_processing.append(result)
            else:
                failed_processing.append(result)

        # Summary
        _logger.info("Feature engineering completed:")
        _logger.info("  Successfully processed: %d files", len(successful_processing))
        _logger.info("  Failed processing: %d files", len(failed_processing))

        if failed_processing:
            _logger.warning("Failed processing: %s", [r['filename'] for r in failed_processing])

        return {
            'success': successful_processing,
            'failed': failed_processing
        }

def main():
    """Main entry point for feature engineering."""
    import argparse

    parser = argparse.ArgumentParser(description='Feature engineering for CNN-LSTM-XGBoost pipeline')
    parser.add_argument('--config', default='config/pipeline/x02.yaml', help='Configuration file path')
    parser.add_argument('--file', help='Process specific file only')

    args = parser.parse_args()

    try:
        feature_engineer = FeatureEngineer(args.config)

        if args.file:
            # Process specific file
            filepath = Path(args.file)
            if not filepath.exists():
                _logger.error("File not found: %s", filepath)
                sys.exit(1)

            result = feature_engineer.process_single_file(filepath)
            print("Processing result:", result)
        else:
            # Process all files
            results = feature_engineer.run()
            print("Feature engineering results:", results)

    except Exception:
        _logger.exception("Feature engineering failed:")
        sys.exit(1)

if __name__ == "__main__":
    main()
