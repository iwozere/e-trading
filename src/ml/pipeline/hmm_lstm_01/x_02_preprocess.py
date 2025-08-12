"""
Data Preprocessing for HMM-LSTM Trading Pipeline

This module preprocesses raw OHLCV data by adding features, computing technical
indicators using TA-Lib, normalizing data, and preparing it for HMM training and LSTM modeling.

Features:
- Computes log returns and rolling statistics
- Adds TA-Lib technical indicators with timeframe-optimized parameters
- Normalizes features using configurable methods
- Handles missing values and outliers
- Saves processed data for next pipeline stage
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import talib
import sys
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def build_indicator_config(timeframe: str) -> Dict[str, Dict[str, int]]:
    """
    Build timeframe-specific indicator configuration for optimal HMM feature extraction.

    Args:
        timeframe (str): Timeframe string (e.g., '5m', '15m', '1h', '4h', '1d')

    Returns:
        Dict: Configuration for each indicator with optimized parameters
    """
    # Base configuration for different timeframes
    configs = {
        # Intraday timeframes (5m, 15m, 1h) - shorter periods for responsiveness
        '5m': {
            'RSI': {'timeperiod': 14},
            'ATR': {'timeperiod': 14},
            'BB': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'EMA_SPREAD': {'fast': 12, 'slow': 26},
            'VOL_ZSCORE': {'window': 20}
        },
        '15m': {
            'RSI': {'timeperiod': 14},
            'ATR': {'timeperiod': 14},
            'BB': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'EMA_SPREAD': {'fast': 12, 'slow': 26},
            'VOL_ZSCORE': {'window': 20}
        },
        '1h': {
            'RSI': {'timeperiod': 14},
            'ATR': {'timeperiod': 14},
            'BB': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'EMA_SPREAD': {'fast': 12, 'slow': 26},
            'VOL_ZSCORE': {'window': 20}
        },
        # Daily timeframe - longer periods for trend detection
        '1d': {
            'RSI': {'timeperiod': 14},
            'ATR': {'timeperiod': 14},
            'BB': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'EMA_SPREAD': {'fast': 12, 'slow': 26},
            'VOL_ZSCORE': {'window': 20}
        },
        # 4h timeframe - medium-term analysis
        '4h': {
            'RSI': {'timeperiod': 14},
            'ATR': {'timeperiod': 14},
            'BB': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'EMA_SPREAD': {'fast': 12, 'slow': 26},
            'VOL_ZSCORE': {'window': 20}
        }
    }

    # Return configuration for the specified timeframe, or default to 1h
    return configs.get(timeframe, configs['1h'])


def select_features_for_hmm(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Select and calculate features for HMM from OHLCV + log_return data using TA-Lib.

    Args:
        df (pd.DataFrame): Must contain ['open', 'high', 'low', 'close', 'volume', 'log_return']
        timeframe (str): e.g. '5m', '15m', '1h', '4h', '1d'

    Returns:
        pd.DataFrame: Numeric features for HMM
    """
    # Ensure column names are lowercase
    df = df.rename(columns=str.lower).copy()

    # Validate that log_return already exists
    if "log_return" not in df.columns:
        raise ValueError("Expected 'log_return' column in input data, but it was not found.")

    # Build indicator configuration for this timeframe
    indicator_config = build_indicator_config(timeframe)

    # RSI
    df["rsi"] = talib.RSI(
        df["close"],
        timeperiod=indicator_config["RSI"]["timeperiod"]
    )

    # ATR
    df["atr"] = talib.ATR(
        df["high"], df["low"], df["close"],
        timeperiod=indicator_config["ATR"]["timeperiod"]
    )

    # ATR as % of close (volatility normalization)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        df["close"],
        timeperiod=indicator_config["BB"]["timeperiod"],
        nbdevup=indicator_config["BB"]["nbdevup"],
        nbdevdn=indicator_config["BB"]["nbdevdn"],
        matype=0
    )
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower

    # EMA Spread (fast EMA - slow EMA)
    fast_ema = talib.EMA(df["close"], timeperiod=indicator_config["EMA_SPREAD"]["fast"])
    slow_ema = talib.EMA(df["close"], timeperiod=indicator_config["EMA_SPREAD"]["slow"])
    df["ema_spread"] = fast_ema - slow_ema

    # Volume Z-Score (liquidity abnormality detection)
    df["volume_zscore"] = (
        (df["volume"] - df["volume"].rolling(indicator_config["VOL_ZSCORE"]["window"]).mean())
        / df["volume"].rolling(indicator_config["VOL_ZSCORE"]["window"]).std()
    )

    # Drop rows with NaN (from indicator warm-up period)
    df = df.dropna()

    # Only keep numeric columns, but exclude old rolling statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Filter out old rolling statistics columns
    exclude_patterns = [
        'close_sma_', 'volume_sma_', 'close_std_', 'volume_std_',
        'high_max_', 'low_min_'
    ]

    filtered_cols = []
    for col in numeric_cols:
        should_exclude = any(pattern in col for pattern in exclude_patterns)
        if not should_exclude:
            filtered_cols.append(col)

    _logger.info("Selected %d features for HMM (excluded %d rolling statistics columns)",
                len(filtered_cols), len(numeric_cols) - len(filtered_cols))

    return df[filtered_cols]


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

        _logger.info("Loaded configuration from %s", self.config_path)
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

    def add_technical_indicators(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Add TA-Lib technical indicators optimized for the given timeframe.

        Args:
            df: Input DataFrame with OHLCV data
            timeframe: Timeframe string (e.g., '5m', '15m', '1h', '4h', '1d')

        Returns:
            DataFrame with technical indicators added
        """
        try:
            # Use the new TA-Lib based feature selection
            df_with_indicators = select_features_for_hmm(df, timeframe)
            _logger.info("Added TA-Lib technical indicators for timeframe %s", timeframe)
            return df_with_indicators
        except Exception as e:
            _logger.exception("Error adding technical indicators: %s", str(e))
            # Return original dataframe if technical indicators fail
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
        # Create a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()

        # Forward fill then backward fill for price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean.loc[:, col] = df_clean[col].ffill().bfill()

        # Forward fill volume
        if 'volume' in df_clean.columns:
            df_clean.loc[:, 'volume'] = df_clean['volume'].ffill().fillna(0)

        # Fill technical indicators with their median values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in price_cols + ['volume', 'timestamp']:
                df_clean.loc[:, col] = df_clean[col].fillna(df_clean[col].median())

        return df_clean

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
        # Create a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['timestamp']:
                continue

            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR

                # Cap outliers instead of removing them to preserve time series structure
                # Ensure dtype compatibility before clipping
                clipped_values = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                df_clean.loc[:, col] = clipped_values.astype(df_clean[col].dtype)

            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean.loc[:, col] = df_clean[col].mask(z_scores > factor, df_clean[col].median())

        return df_clean

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

    def extract_timeframe_from_filename(self, filename: str) -> str:
        """
        Extract timeframe from filename.

        Args:
            filename: Filename (e.g., 'binance_BTCUSDT_1h_20230801_20240801.csv')

        Returns:
            Timeframe string (e.g., '1h', '4h', '1d')
        """
        # Split filename by underscores and look for timeframe
        parts = filename.replace('.csv', '').split('_')

        # Look for timeframe in the parts
        for part in parts:
            if part in ['5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo']:
                return part

        # Default to 1h if not found
        _logger.warning("Could not extract timeframe from filename %s, using default '1h'", filename)
        return '1h'

    def process_file(self, input_path: Path, output_path: Path) -> Dict:
        """
        Process a single CSV file through the preprocessing pipeline.

        Args:
            input_path: Path to input CSV file
            output_path: Path to save processed CSV file

        Returns:
            Dict with processing statistics
        """
        _logger.info("Processing %s", input_path.name)

        try:
            # Load data
            df = pd.read_csv(input_path)
            original_shape = df.shape

            # Extract timeframe from filename for technical indicators
            timeframe = self.extract_timeframe_from_filename(input_path.name)
            _logger.info("Extracted timeframe '%s' from filename %s", timeframe, input_path.name)

            # Apply preprocessing steps
            df = self.add_log_returns(df)

            if self.config['features']['rolling_stats']:
                df = self.add_rolling_statistics(df)

            if self.config['features']['technical_indicators']:
                df = self.add_technical_indicators(df, timeframe)

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
                'timeframe': timeframe,
                'success': True
            }

            _logger.info("[OK] %s (%s): %s -> %s (+%d cols, -%d rows)",
                        input_path.name, timeframe, original_shape, df.shape,
                        stats['columns_added'], stats['rows_removed'])

            return stats

        except Exception as e:
            error_msg = f"Failed to process {input_path.name}"
            _logger.exception(error_msg)
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
            _logger.warning("No CSV files found in %s", self.raw_data_dir)
            return {'total': 0, 'successful': [], 'failed': []}

        _logger.info("Found %d CSV files to process", len(csv_files))

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
        _logger.info("\n%s", "="*50)
        _logger.info("Preprocessing Summary:")
        _logger.info("  Total files: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))
        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.warning("Failed processing:")
            for failure in results['failed']:
                _logger.warning("  %s: %s", failure['input_file'], failure['error'])

        return results

def main():
    """Main function to run preprocessing."""
    try:
        preprocessor = DataPreprocessor()
        results = preprocessor.process_all()

        _logger.info("Preprocessing completed!")

    except Exception as e:
        _logger.exception("Preprocessing failed")
        raise

if __name__ == "__main__":
    main()
