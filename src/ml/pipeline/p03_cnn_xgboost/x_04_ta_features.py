"""
Technical Analysis Features Stage for CNN + XGBoost Pipeline.

This module calculates technical indicators from OHLCV data and combines them with
CNN embeddings to create feature-rich datasets for XGBoost classification.
"""

import sys
from pathlib import Path

# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import talib

from src.notification.logger import setup_logger
from src.util.config import load_config
from src.ml.pipeline.p03_cnn_xgboost.utils.data_validation import convert_targets_to_numeric, log_data_quality_report
_logger = setup_logger(__name__)


class TAFeatureEngineer:
    """
    Technical Analysis Feature Engineer for the CNN + XGBoost pipeline.

    Calculates technical indicators from OHLCV data and combines them with
    CNN embeddings to create comprehensive feature sets for XGBoost training.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the TA feature engineer.

        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.ta_config = config.get("technical_analysis", {})

        _logger.info("Initializing TA feature engineer")

        # Create output directories
        self.features_dir = Path("data/features")
        self.features_dir.mkdir(parents=True, exist_ok=True)

        # Define technical indicators to calculate
        self.indicators = [
            "rsi", "macd", "macd_signal", "macd_histogram", "bb_upper", "bb_middle", "bb_lower",
            "sma_20", "ema_fast", "price_vs_sma20", "price_vs_ema_fast",
            "stoch_k", "stoch_d", "adx", "plus_di", "minus_di", "obv", "atr", "cci", "roc", "mfi"
        ]

    def run(self) -> Dict[str, Any]:
        """
        Execute the TA feature engineering stage.

        Returns:
            Dictionary containing feature engineering results and metadata
        """
        _logger.info("Starting TA feature engineering stage")

        try:
            # Discover labeled data files
            labeled_files = self._discover_labeled_data()
            if not labeled_files:
                raise ValueError("No labeled data files found")

            _logger.info("Found %d labeled data files", len(labeled_files))

            # Process each file
            processing_results = self._process_all_files(labeled_files)

            # Save processing summary
            self._save_processing_summary(processing_results)

            _logger.info("TA feature engineering stage completed successfully")
            return processing_results

        except Exception as e:
            _logger.exception("Error in TA feature engineering stage: %s", e)
            raise

    def _discover_labeled_data(self) -> List[Path]:
        """
        Discover labeled data files from the embedding generation stage.

        Returns:
            List of paths to labeled data files
        """
        labeled_dir = Path("data/labeled")
        if not labeled_dir.exists():
            raise FileNotFoundError(f"Labeled data directory not found: {labeled_dir}")

        # Look for CSV files with embeddings
        labeled_files = list(labeled_dir.glob("*_labeled.csv"))

        if not labeled_files:
            # Fallback to any CSV files
            labeled_files = list(labeled_dir.glob("*.csv"))

        return labeled_files

    def _process_all_files(self, labeled_files: List[Path]) -> Dict[str, Any]:
        """
        Process all labeled data files to add technical indicators.

        Args:
            labeled_files: List of paths to labeled data files

        Returns:
            Dictionary containing processing results
        """
        _logger.info("Processing %d labeled data files", len(labeled_files))

        results = {
            "files_processed": 0,
            "total_features": 0,
            "failed_files": [],
            "file_results": []
        }

        for file_path in labeled_files:
            try:
                file_result = self._process_single_file(file_path)
                results["files_processed"] += 1
                results["total_features"] += file_result["features_count"]
                results["file_results"].append(file_result)

                _logger.info("Processed %s: %d features", file_path.name, file_result["features_count"])

            except Exception as e:
                _logger.warning("Failed to process %s: %s", file_path, e)
                results["failed_files"].append({
                    "file": str(file_path),
                    "error": str(e)
                })

        _logger.info("TA feature engineering completed: %d files processed, %d total features",
                    results["files_processed"], results["total_features"])

        return results

    def _process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single labeled data file to add technical indicators.

        Args:
            file_path: Path to the labeled data file

        Returns:
            Dictionary containing file processing results
        """
        # Load labeled data
        df = pd.read_csv(file_path)

        # Extract OHLCV data
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in ohlcv_cols):
            raise ValueError(f"Missing OHLCV columns in {file_path}")

        # Calculate technical indicators
        ta_features = self._calculate_technical_indicators(df)

        # Combine with original data and embeddings
        feature_df = self._combine_features(df, ta_features)

        # Save feature-rich data
        output_path = self._save_feature_data(feature_df, file_path)

        return {
            "input_file": str(file_path),
            "output_file": str(output_path),
            "features_count": len(ta_features.columns),
            "original_rows": len(df),
            "feature_rows": len(feature_df),
            "indicators_calculated": list(ta_features.columns)
        }

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with technical indicators
        """
        ta_features = pd.DataFrame(index=df.index)

        # Extract OHLCV arrays
        open_prices = df["open"].values
        high_prices = df["high"].values
        low_prices = df["low"].values
        close_prices = df["close"].values
        volumes = df["volume"].values

        # RSI
        if "rsi" in self.indicators:
            ta_features["rsi"] = talib.RSI(close_prices, timeperiod=14)

        # MACD
        if "macd" in self.indicators:
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            ta_features["macd"] = macd
            ta_features["macd_signal"] = macd_signal
            ta_features["macd_histogram"] = macd_hist

        # Bollinger Bands
        if "bb_upper" in self.indicators:
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices)
            ta_features["bb_upper"] = bb_upper
            ta_features["bb_lower"] = bb_lower
            ta_features["bb_position"] = (close_prices - bb_lower) / (bb_upper - bb_lower)

        # ATR (Average True Range)
        if "atr" in self.indicators:
            ta_features["atr"] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

        # Moving Averages
        if "sma_20" in self.indicators:
            ta_features["sma_20"] = talib.SMA(close_prices, timeperiod=20)
            ta_features["price_vs_sma20"] = close_prices / ta_features["sma_20"]

        if "ema_fast" in self.indicators:
            ta_features["ema_fast"] = talib.EMA(close_prices, timeperiod=12)
            ta_features["price_vs_ema_fast"] = close_prices / ta_features["ema_fast"]

        # Volume Ratio
        if "volume_ratio" in self.indicators:
            volume_sma = talib.SMA(volumes, timeperiod=20)
            ta_features["volume_ratio"] = volumes / volume_sma

        # Stochastic
        if "stoch_k" in self.indicators:
            stoch_k, stoch_d = talib.STOCH(high_prices, low_prices, close_prices)
            ta_features["stoch_k"] = stoch_k
            ta_features["stoch_d"] = stoch_d

        # Additional price-based features
        ta_features["price_change"] = close_prices / np.roll(close_prices, 1) - 1
        ta_features["high_low_ratio"] = high_prices / low_prices
        ta_features["close_open_ratio"] = close_prices / open_prices

        # Volatility features
        ta_features["price_volatility"] = talib.STDDEV(close_prices, timeperiod=20)

        # Momentum features
        ta_features["momentum"] = talib.MOM(close_prices, timeperiod=10)
        ta_features["roc"] = talib.ROC(close_prices, timeperiod=10)

        # Handle NaN values
        ta_features = ta_features.ffill().fillna(0)

        return ta_features

    def _combine_features(self,
                         original_df: pd.DataFrame,
                         ta_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combine original data, embeddings, and technical indicators.

        Args:
            original_df: Original labeled DataFrame with embeddings
            ta_features: DataFrame with technical indicators

        Returns:
            Combined feature DataFrame
        """
        # Identify embedding columns
        embedding_cols = [col for col in original_df.columns if col.startswith("embedding_")]

        # Select relevant columns from original data
        base_cols = ["open", "high", "low", "close", "volume", "date", "timestamp"]
        base_cols = [col for col in base_cols if col in original_df.columns]

        # Combine base data, embeddings, and TA features
        combined_df = pd.concat([
            original_df[base_cols + embedding_cols],
            ta_features
        ], axis=1)

        # Add metadata columns if they exist
        metadata_cols = ["sequence_start_idx", "sequence_end_idx"]
        for col in metadata_cols:
            if col in original_df.columns:
                combined_df[col] = original_df[col]

        # Create target variables for multiple targets strategy
        combined_df = self._create_target_variables(combined_df)

        return combined_df

    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create multiple target variables for the XGBoost stage.

        Args:
            df: Feature DataFrame

        Returns:
            DataFrame with target variables added
        """
        close_prices = df["close"].values

        # Target 1: Price direction (next period)
        df["target_direction"] = 0
        df.loc[df["close"].shift(-1) > df["close"], "target_direction"] = 1

        # Target 2: Volatility regime (high/low volatility)
        returns = df["close"].pct_change()
        volatility = returns.rolling(window=20).std()
        volatility_median = volatility.median()
        df["target_volatility"] = (volatility > volatility_median).astype(int)

        # Target 3: Trend strength (strong/weak trend)
        sma_short = df["close"].rolling(window=10).mean()
        sma_long = df["close"].rolling(window=30).mean()
        trend_strength = abs(sma_short - sma_long) / sma_long
        trend_median = trend_strength.median()
        df["target_trend"] = (trend_strength > trend_median).astype(int)

        # Target 4: Price movement magnitude (large/small moves)
        price_changes = abs(df["close"].pct_change())
        change_median = price_changes.median()
        df["target_magnitude"] = (price_changes > change_median).astype(int)

        # Handle NaN values in targets and ensure proper data types
        target_cols = ["target_direction", "target_volatility", "target_trend", "target_magnitude"]

                # Convert targets to proper numeric types
        df = convert_targets_to_numeric(df, target_cols)

        return df

    def _save_feature_data(self, feature_df: pd.DataFrame, original_file_path: Path) -> Path:
        """
        Save feature-rich data to the features directory.

        Args:
            feature_df: Feature DataFrame
            original_file_path: Path to original labeled data file

        Returns:
            Path to saved feature data file
        """
        # Create output filename
        base_name = original_file_path.stem.replace("_labeled", "")
        output_filename = f"{base_name}_features.csv"
        output_path = self.features_dir / output_filename

        # Save as CSV
        feature_df.to_csv(output_path, index=False)

        _logger.debug("Saved feature data to %s", output_path)

        return output_path

    def _save_processing_summary(self, processing_results: Dict[str, Any]) -> None:
        """
        Save processing summary to file.

        Args:
            processing_results: Results from feature engineering
        """
        summary_path = self.features_dir / "ta_feature_engineering_summary.json"

        summary = {
            "stage": "ta_feature_engineering",
            "status": "completed",
            "timestamp": pd.Timestamp.now().isoformat(),
            "indicators_calculated": self.indicators,
            "processing_results": processing_results
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        _logger.info("Saved processing summary to %s", summary_path)


def engineer_ta_features(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to engineer technical analysis features.

    Args:
        config: Pipeline configuration dictionary

    Returns:
        Dictionary containing feature engineering results
    """
    engineer = TAFeatureEngineer(config)
    return engineer.run()


if __name__ == "__main__":
    # Load configuration
    config_path = Path("config/pipeline/p03.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Run TA feature engineering
    results = engineer_ta_features(config)
    _logger.info("TA Feature Engineering Results: %s", results)
