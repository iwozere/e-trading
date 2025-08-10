"""
Data Preprocessing for HMM-LSTM Trading Pipeline

This module preprocesses raw OHLCV data by adding features, computing technical
indicators, normalizing data, and preparing it for HMM training and LSTM modeling.

Features:
- Computes log returns and rolling statistics
- Adds technical indicators with default parameters
- Normalizes features using configurable methods
- Handles missing values and outliers
- Saves processed data for next pipeline stage
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import talib
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

class DataPreprocessor:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize DataPreprocessor with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.raw_data_dir = Path(self.config['paths']['data_raw'])
        self.processed_data_dir = Path(self.config['paths']['data_processed'])
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def add_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add log returns column if not present."""
        if 'log_return' not in df.columns:
            # Calculate log returns with proper handling of zeros and negatives
            price_ratio = df['close'] / df['close'].shift(1)
            price_ratio = price_ratio.replace([np.inf, -np.inf, 0], np.nan)
            df['log_return'] = np.log(price_ratio)
            df['log_return'] = df['log_return'].fillna(0)

        return df

    def add_rolling_statistics(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling statistics for price and volume.

        Args:
            df: Input DataFrame
            windows: List of rolling window sizes

        Returns:
            DataFrame with additional rolling statistics columns
        """
        for window in windows:
            # Rolling means
            df[f'close_sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()

            # Rolling standard deviations
            df[f'close_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'volume_std_{window}'] = df['volume'].rolling(window=window).std()

            # Rolling min/max
            df[f'high_max_{window}'] = df['high'].rolling(window=window).max()
            df[f'low_min_{window}'] = df['low'].rolling(window=window).min()

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators with default parameters.
        These parameters will be optimized later by Optuna.

        Args:
            df: Input DataFrame with OHLCV data

        Returns:
            DataFrame with additional technical indicator columns
        """
        # Convert to numpy arrays for talib
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        open_price = df['open'].values
        volume = df['volume'].values

        try:
            # RSI (Relative Strength Index)
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_21'] = talib.RSI(close, timeperiod=21)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper_20'] = bb_upper
            df['bb_middle_20'] = bb_middle
            df['bb_lower_20'] = bb_lower
            df['bb_width_20'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position_20'] = (close - bb_lower) / (bb_upper - bb_lower)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist

            # Moving Averages
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['sma_200'] = talib.SMA(close, timeperiod=200)

            # Average True Range (ATR)
            df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

            # Commodity Channel Index
            df['cci_14'] = talib.CCI(high, low, close, timeperiod=14)

            # Money Flow Index
            df['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)

            # Volume indicators
            df['ad'] = talib.AD(high, low, close, volume)  # Accumulation/Distribution
            df['obv'] = talib.OBV(close, volume)  # On-Balance Volume

            # Price-based features
            df['hl_ratio'] = (high - low) / close
            df['oc_ratio'] = (close - open_price) / open_price
            df['high_close_ratio'] = (high - close) / close
            df['low_close_ratio'] = (close - low) / close

        except Exception as e:
            logger.warning(f"Error calculating technical indicators: {str(e)}")
            # Continue without technical indicators if calculation fails

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: Input DataFrame with timestamp column

        Returns:
            DataFrame with additional time features
        """
        if 'timestamp' in df.columns:
            # Convert timestamp to datetime if it's not already
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Extract time components
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month

            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with handled missing values
        """
        # Forward fill then backward fill for price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Forward fill volume
        if 'volume' in df.columns:
            df['volume'] = df['volume'].ffill().fillna(0)

        # Fill technical indicators with their median values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in price_cols + ['volume', 'timestamp']:
                df[col] = df[col].fillna(df[col].median())

        return df

    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from the dataset.

        Args:
            df: Input DataFrame
            method: Method for outlier detection ('iqr' or 'zscore')
            factor: Factor for outlier threshold

        Returns:
            DataFrame with outliers handled
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['timestamp']:
                continue

            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                # Cap outliers instead of removing them to preserve time series structure
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df[col] = df[col].mask(z_scores > factor, df[col].median())

        return df

    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features using specified method.

        Args:
            df: Input DataFrame
            method: Normalization method ('minmax', 'standard', 'robust')

        Returns:
            Tuple of (normalized DataFrame, scaler_dict for inverse transform)
        """
        # Columns to exclude from normalization
        exclude_cols = ['timestamp', 'regime']  # regime will be added later by HMM

        # Get numeric columns to normalize
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        scalers = {}
        df_normalized = df.copy()

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        if cols_to_normalize:
            # Fit scaler and transform
            scaled_data = scaler.fit_transform(df[cols_to_normalize])
            df_normalized[cols_to_normalize] = scaled_data

            scalers['feature_scaler'] = scaler
            scalers['normalized_columns'] = cols_to_normalize

        return df_normalized, scalers

    def process_file(self, input_path: Path, output_path: Path) -> Dict:
        """
        Process a single CSV file through the preprocessing pipeline.

        Args:
            input_path: Path to input CSV file
            output_path: Path to save processed CSV file

        Returns:
            Dict with processing statistics
        """
        logger.info(f"Processing {input_path.name}")

        try:
            # Load data
            df = pd.read_csv(input_path)
            original_shape = df.shape

            # Apply preprocessing steps
            df = self.add_log_returns(df)

            if self.config['features']['rolling_stats']:
                df = self.add_rolling_statistics(df)

            if self.config['features']['technical_indicators']:
                df = self.add_technical_indicators(df)

            df = self.add_time_features(df)
            df = self.handle_missing_values(df)
            df = self.remove_outliers(df)

            # Normalize if specified
            scalers = None
            if self.config['features']['normalization']:
                df, scalers = self.normalize_features(df, self.config['features']['normalization'])

            # Remove rows with any remaining NaN values
            df = df.dropna()

            # Save processed data
            df.to_csv(output_path, index=False)

            # Prepare statistics
            stats = {
                'input_file': str(input_path),
                'output_file': str(output_path),
                'original_shape': original_shape,
                'processed_shape': df.shape,
                'columns_added': df.shape[1] - original_shape[1],
                'rows_removed': original_shape[0] - df.shape[0],
                'success': True
            }

            logger.info(f"[OK] {input_path.name}: {original_shape} -> {df.shape} (+{stats['columns_added']} cols, -{stats['rows_removed']} rows)")

            return stats

        except Exception as e:
            error_msg = f"Failed to process {input_path.name}: {str(e)}"
            logger.error(error_msg)
            return {
                'input_file': str(input_path),
                'success': False,
                'error': error_msg
            }

    def process_all(self) -> Dict:
        """
        Process all CSV files in the raw data directory.

        Returns:
            Dict with summary of processing results
        """
        # Find all CSV files in raw data directory
        csv_files = list(self.raw_data_dir.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {self.raw_data_dir}")
            return {'total': 0, 'successful': [], 'failed': []}

        logger.info(f"Found {len(csv_files)} CSV files to process")

        results = {
            'total': len(csv_files),
            'successful': [],
            'failed': []
        }

        for csv_file in csv_files:
            # Generate output filename
            output_filename = f"processed_{csv_file.name}"
            output_path = self.processed_data_dir / output_filename

            # Process file
            stats = self.process_file(csv_file, output_path)

            if stats['success']:
                results['successful'].append(stats)
            else:
                results['failed'].append(stats)

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Preprocessing Summary:")
        logger.info(f"  Total files: {results['total']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"{'='*50}")

        if results['failed']:
            logger.warning("Failed processing:")
            for failure in results['failed']:
                logger.warning(f"  {failure['input_file']}: {failure['error']}")

        return results

def main():
    """Main function to run preprocessing."""
    try:
        preprocessor = DataPreprocessor()
        results = preprocessor.process_all()

        logger.info("Preprocessing completed!")

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
