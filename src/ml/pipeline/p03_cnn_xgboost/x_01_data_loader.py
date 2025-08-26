#!/usr/bin/env python3
"""
Stage 1: Data Loader for CNN + XGBoost Pipeline

This module handles data loading, preprocessing, and validation for the CNN + XGBoost pipeline.
It processes OHLCV data from CSV files and prepares it for CNN training and feature extraction.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import glob
import re
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataLoader:
    """Data loader for CNN + XGBoost pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract configuration parameters
        self.data_config = config["data"]
        self.symbols = self.data_config["symbols"]
        self.timeframes = self.data_config["timeframes"]
        self.start_date = self.data_config["start_date"]
        self.end_date = self.data_config["end_date"]
        self.min_records = self.data_config["min_records"]
        self.max_missing_pct = self.data_config["max_missing_pct"]

        # Directories
        self.input_dir = Path(self.data_config["input_dir"])
        self.processed_dir = Path(self.data_config["processed_dir"])
        self.file_pattern = self.data_config["file_pattern"]

        # Create output directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Data quality metrics
        self.quality_metrics = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_records": 0,
            "cleaned_records": 0,
            "missing_data_removed": 0,
            "outliers_removed": 0
        }

    def run(self) -> Dict[str, Any]:
        """
        Run the data loading and preprocessing pipeline.

        Returns:
            Dictionary containing processing results and metrics
        """
        self.logger.info("Starting data loading and preprocessing")

        try:
            # Discover data files
            data_files = self._discover_data_files()
            self.logger.info(f"Found {len(data_files)} data files")

            # Process each file
            processed_files = []
            for file_path in data_files:
                try:
                    result = self._process_file(file_path)
                    if result:
                        processed_files.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    self.quality_metrics["failed_files"] += 1

            # Generate quality report
            quality_report = self._generate_quality_report()

            # Save processing summary
            self._save_processing_summary(processed_files, quality_report)

            self.logger.info(f"Data loading completed. Processed {len(processed_files)} files")

            return {
                "processed_files": len(processed_files),
                "failed_files": self.quality_metrics["failed_files"],
                "total_records": self.quality_metrics["total_records"],
                "cleaned_records": self.quality_metrics["cleaned_records"],
                "quality_report": quality_report,
                "processed_file_paths": [f["output_path"] for f in processed_files]
            }

        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise

    def _discover_data_files(self) -> List[Path]:
        """
        Discover data files based on configuration.

        Returns:
            List of file paths matching the pattern
        """
        files = []

        for symbol in self.symbols:
            for timeframe in self.timeframes:
                # Create pattern for file discovery
                pattern = self.file_pattern.format(
                    provider="*",
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date="*",
                    end_date="*"
                )

                # Search for files
                search_pattern = str(self.input_dir / pattern)
                matching_files = glob.glob(search_pattern)

                for file_path in matching_files:
                    path = Path(file_path)
                    if self._validate_file_name(path):
                        files.append(path)

        # Sort files by name for consistent processing order
        files.sort()

        return files

    def _validate_file_name(self, file_path: Path) -> bool:
        """
        Validate if file name matches expected pattern.

        Args:
            file_path: Path to the file

        Returns:
            True if file name is valid, False otherwise
        """
        try:
            # Extract components from filename
            filename = file_path.stem
            parts = filename.split('_')

            if len(parts) < 5:
                return False

            # Check if symbol and timeframe match configuration
            symbol = parts[1]
            timeframe = parts[2]

            if symbol not in self.symbols or timeframe not in self.timeframes:
                return False

            # Validate date format
            start_date = parts[3]
            end_date = parts[4]

            datetime.strptime(start_date, "%Y%m%d")
            datetime.strptime(end_date, "%Y%m%d")

            return True

        except (ValueError, IndexError):
            return False

    def _process_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Process a single data file.

        Args:
            file_path: Path to the input file

        Returns:
            Dictionary containing processing results or None if failed
        """
        self.logger.debug(f"Processing file: {file_path}")

        try:
            # Load data
            df = self._load_data(file_path)
            if df is None or df.empty:
                self.logger.warning(f"Empty or invalid data in {file_path}")
                return None

            # Validate data quality
            if not self._validate_data_quality(df):
                self.logger.warning(f"Data quality check failed for {file_path}")
                return None

            # Clean data
            df_cleaned = self._clean_data(df)

            # Add derived features
            df_features = self._add_derived_features(df_cleaned)

            # Save processed data
            output_path = self._save_processed_data(df_features, file_path)

            # Update metrics
            self.quality_metrics["processed_files"] += 1
            self.quality_metrics["total_records"] += len(df)
            self.quality_metrics["cleaned_records"] += len(df_features)

            return {
                "input_path": str(file_path),
                "output_path": str(output_path),
                "original_records": len(df),
                "cleaned_records": len(df_features),
                "missing_removed": len(df) - len(df_cleaned),
                "outliers_removed": len(df_cleaned) - len(df_features)
            }

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return None

    def _load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame or None if failed
        """
        try:
            # Load CSV with proper date parsing
            df = pd.read_csv(
                file_path,
                parse_dates=['timestamp'],
                index_col='timestamp',
                infer_datetime_format=True
            )

            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in {file_path}")
                return None

            # Sort by timestamp
            df = df.sort_index()

            return df

        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality requirements.

        Args:
            df: DataFrame to validate

        Returns:
            True if data meets quality requirements, False otherwise
        """
        # Check minimum records
        if len(df) < self.min_records:
            self.logger.warning(f"Insufficient records: {len(df)} < {self.min_records}")
            return False

        # Check for missing data
        missing_pct = (df.isnull().sum() / len(df)) * 100
        if missing_pct.max() > self.max_missing_pct:
            self.logger.warning(f"Too much missing data: {missing_pct.max():.2f}% > {self.max_missing_pct}%")
            return False

        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        if (df[price_columns] <= 0).any().any():
            self.logger.warning("Found negative or zero prices")
            return False

        # Check for volume consistency
        if (df['volume'] < 0).any():
            self.logger.warning("Found negative volume")
            return False

        # Check OHLC consistency
        if not self._check_ohlc_consistency(df):
            self.logger.warning("OHLC consistency check failed")
            return False

        return True

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> bool:
        """
        Check OHLC (Open, High, Low, Close) consistency.

        Args:
            df: DataFrame with OHLC data

        Returns:
            True if OHLC data is consistent, False otherwise
        """
        # High should be >= Low
        if (df['high'] < df['low']).any():
            return False

        # High should be >= Open and Close
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            return False

        # Low should be <= Open and Close
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            return False

        return True

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by removing outliers and handling missing values.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()

        # Remove rows with any missing values
        initial_count = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        self.quality_metrics["missing_data_removed"] += initial_count - len(df_cleaned)

        # Remove outliers using IQR method
        df_cleaned = self._remove_outliers(df_cleaned)

        return df_cleaned

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using IQR method.

        Args:
            df: DataFrame to clean

        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()

        # Calculate IQR for price columns
        price_columns = ['open', 'high', 'low', 'close']

        for col in price_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Remove outliers
            outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            df_clean = df_clean[~outlier_mask]

        # Handle volume outliers (more lenient)
        Q1 = df_clean['volume'].quantile(0.25)
        Q3 = df_clean['volume'].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3 * IQR  # More lenient for volume
        upper_bound = Q3 + 3 * IQR

        outlier_mask = (df_clean['volume'] < lower_bound) | (df_clean['volume'] > upper_bound)
        df_clean = df_clean[~outlier_mask]

        self.quality_metrics["outliers_removed"] += len(df) - len(df_clean)

        return df_clean

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to the DataFrame.

        Args:
            df: Cleaned DataFrame

        Returns:
            DataFrame with additional features
        """
        df_features = df.copy()

        # Calculate log returns
        df_features['log_return'] = np.log(df_features['close'] / df_features['close'].shift(1))

        # Calculate price ratios
        df_features['high_low_ratio'] = df_features['high'] / df_features['low']
        df_features['close_open_ratio'] = df_features['close'] / df_features['open']

        # Calculate volume features
        df_features['volume_sma_20'] = df_features['volume'].rolling(window=20).mean()
        df_features['volume_ratio'] = df_features['volume'] / df_features['volume_sma_20']

        # Calculate price volatility
        df_features['price_volatility'] = df_features['log_return'].rolling(window=20).std()

        # Calculate moving averages
        df_features['sma_20'] = df_features['close'].rolling(window=20).mean()
        df_features['ema_12'] = df_features['close'].ewm(span=12).mean()

        # Price position relative to moving averages
        df_features['price_vs_sma20'] = (df_features['close'] - df_features['sma_20']) / df_features['sma_20']
        df_features['price_vs_ema12'] = (df_features['close'] - df_features['ema_12']) / df_features['ema_12']

        # Remove NaN values from derived features
        df_features = df_features.dropna()

        return df_features

    def _save_processed_data(self, df: pd.DataFrame, original_file: Path) -> Path:
        """
        Save processed data to parquet format.

        Args:
            df: Processed DataFrame
            original_file: Original file path for naming

        Returns:
            Path to saved file
        """
        # Create output filename
        filename = original_file.stem + "_processed.parquet"
        output_path = self.processed_dir / filename

        # Save as parquet
        df.to_parquet(output_path, index=True, compression='snappy')

        self.logger.debug(f"Saved processed data to {output_path}")

        return output_path

    def _generate_quality_report(self) -> Dict[str, Any]:
        """
        Generate data quality report.

        Returns:
            Dictionary containing quality metrics
        """
        return {
            "processing_summary": {
                "total_files_found": self.quality_metrics["total_files"],
                "files_processed": self.quality_metrics["processed_files"],
                "files_failed": self.quality_metrics["failed_files"],
                "success_rate": self.quality_metrics["processed_files"] / max(1, self.quality_metrics["total_files"]) * 100
            },
            "data_quality": {
                "total_records": self.quality_metrics["total_records"],
                "cleaned_records": self.quality_metrics["cleaned_records"],
                "missing_data_removed": self.quality_metrics["missing_data_removed"],
                "outliers_removed": self.quality_metrics["outliers_removed"],
                "data_retention_rate": self.quality_metrics["cleaned_records"] / max(1, self.quality_metrics["total_records"]) * 100
            },
            "configuration": {
                "symbols": self.symbols,
                "timeframes": self.timeframes,
                "date_range": f"{self.start_date} to {self.end_date}",
                "min_records": self.min_records,
                "max_missing_pct": self.max_missing_pct
            }
        }

    def _save_processing_summary(self, processed_files: List[Dict[str, Any]], quality_report: Dict[str, Any]):
        """
        Save processing summary to file.

        Args:
            processed_files: List of processed file results
            quality_report: Quality report dictionary
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "quality_report": quality_report,
            "processed_files": processed_files
        }

        summary_file = self.processed_dir / "processing_summary.json"

        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Processing summary saved to {summary_file}")


def main():
    """Main function for standalone execution."""
    import yaml

    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run data loader
    data_loader = DataLoader(config)
    result = data_loader.run()

    print("Data loading completed successfully!")
    print(f"Processed files: {result['processed_files']}")
    print(f"Total records: {result['total_records']}")
    print(f"Cleaned records: {result['cleaned_records']}")


if __name__ == "__main__":
    main()
