"""
Data Loader for HMM-LSTM Trading Pipeline

This module handles downloading OHLCV data for multiple symbols and timeframes
as specified in the pipeline configuration. It uses the existing data downloader
infrastructure to fetch data from various providers (primarily Binance for crypto).

Features:
- Downloads data for multiple symbols and timeframes in parallel
- Saves data with consistent naming convention: {symbol}_{timeframe}_{start_date}_{end_date}.csv
- Configurable via YAML configuration file
- Progress tracking and logging
- Error handling for failed downloads
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.common import get_ohlcv, analyze_period_interval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config/pipeline/x01.yaml"):
        """
        Initialize DataLoader with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_dir = Path(self.config['paths']['data_raw'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def _generate_filename(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime) -> str:
        """Generate consistent filename for downloaded data."""
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"

    def download_symbol_timeframe(self, symbol: str, timeframe: str) -> tuple:
        """
        Download data for a single symbol-timeframe combination.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '5m', '1h', '4h')

        Returns:
            tuple: (symbol, timeframe, success, filepath_or_error)
        """
        try:
            # Calculate date range
            period = self.config['data']['period']
            provider = self.config['data']['provider']

            start_date, end_date = analyze_period_interval(period, timeframe)

            logger.info(f"Downloading {symbol} {timeframe} from {start_date} to {end_date}")

            # Download data using common utility
            df = get_ohlcv(
                ticker=symbol,
                interval=timeframe,
                period=period,
                provider=provider
            )

            if df.empty:
                raise ValueError(f"No data returned for {symbol} {timeframe}")

            # Generate filename and save
            filename = self._generate_filename(symbol, timeframe, start_date, end_date)
            filepath = self.data_dir / filename

            # Add log_return column if not present
            if 'log_return' not in df.columns:
                df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: 0 if x <= 0 else x).pipe(lambda x: x.apply(lambda y: 0 if y == 0 else pd.np.log(y)))

            # Save to CSV
            df.to_csv(filepath, index=False)

            logger.info(f"✓ Successfully saved {symbol} {timeframe} to {filepath}")
            logger.info(f"  Data shape: {df.shape}, Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

            return symbol, timeframe, True, str(filepath)

        except Exception as e:
            error_msg = f"Failed to download {symbol} {timeframe}: {str(e)}"
            logger.error(error_msg)
            return symbol, timeframe, False, error_msg

    def download_all(self, max_workers: int = 4) -> dict:
        """
        Download data for all symbol-timeframe combinations in configuration.

        Args:
            max_workers: Maximum number of parallel download threads

        Returns:
            dict: Summary of download results
        """
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']

        # Create list of all symbol-timeframe combinations
        tasks = [(symbol, tf) for symbol in symbols for tf in timeframes]

        logger.info(f"Starting download for {len(symbols)} symbols x {len(timeframes)} timeframes = {len(tasks)} files")

        results = {
            'successful': [],
            'failed': [],
            'total': len(tasks)
        }

        # Download in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.download_symbol_timeframe, symbol, tf): (symbol, tf)
                for symbol, tf in tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_task):
                symbol, timeframe, success, result = future.result()

                if success:
                    results['successful'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'filepath': result
                    })
                else:
                    results['failed'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'error': result
                    })

        # Log summary
        logger.info(f"\n{'='*50}")
        logger.info(f"Download Summary:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"{'='*50}")

        if results['failed']:
            logger.warning("Failed downloads:")
            for failure in results['failed']:
                logger.warning(f"  {failure['symbol']} {failure['timeframe']}: {failure['error']}")

        return results

    def list_downloaded_files(self) -> list:
        """List all downloaded CSV files in the data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))
        return sorted(csv_files)

    def verify_data_quality(self, filepath: Path) -> dict:
        """
        Verify the quality of a downloaded CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            dict: Quality metrics and issues
        """
        try:
            df = pd.read_csv(filepath)

            # Required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            # Basic checks
            checks = {
                'filepath': str(filepath),
                'shape': df.shape,
                'missing_columns': missing_cols,
                'null_values': df[required_cols].isnull().sum().to_dict() if not missing_cols else {},
                'date_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else None,
                'duplicate_timestamps': df['timestamp'].duplicated().sum() if 'timestamp' in df.columns else 0,
                'zero_volume_count': (df['volume'] == 0).sum() if 'volume' in df.columns else 0,
                'invalid_ohlc': 0  # Count of rows where high < low or close outside [low, high]
            }

            # Check OHLC validity
            if not missing_cols:
                invalid_high_low = (df['high'] < df['low']).sum()
                invalid_close_high = (df['close'] > df['high']).sum()
                invalid_close_low = (df['close'] < df['low']).sum()
                checks['invalid_ohlc'] = invalid_high_low + invalid_close_high + invalid_close_low

            return checks

        except Exception as e:
            return {
                'filepath': str(filepath),
                'error': str(e)
            }

def main():
    """Main function to run data loader."""
    try:
        loader = DataLoader()

        # Download all data
        results = loader.download_all(max_workers=6)

        # Verify data quality for successful downloads
        logger.info("\nVerifying data quality...")
        for item in results['successful']:
            filepath = Path(item['filepath'])
            quality = loader.verify_data_quality(filepath)

            if 'error' in quality:
                logger.error(f"Quality check failed for {filepath}: {quality['error']}")
            else:
                issues = []
                if quality['missing_columns']:
                    issues.append(f"Missing columns: {quality['missing_columns']}")
                if quality['duplicate_timestamps'] > 0:
                    issues.append(f"Duplicate timestamps: {quality['duplicate_timestamps']}")
                if quality['invalid_ohlc'] > 0:
                    issues.append(f"Invalid OHLC: {quality['invalid_ohlc']}")

                if issues:
                    logger.warning(f"{filepath.name}: {', '.join(issues)}")
                else:
                    logger.info(f"✓ {filepath.name}: Quality OK - {quality['shape'][0]} rows")

        logger.info("Data loading completed!")

    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
