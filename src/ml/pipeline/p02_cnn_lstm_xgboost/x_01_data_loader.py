"""
Data Loader for CNN-LSTM-XGBoost Trading Pipeline

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
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time
from typing import Dict, List, Tuple

# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.common import get_ohlcv
from src.notification.logger import setup_logger
from src.data.downloader.data_downloader_factory import DataDownloaderFactory
_logger = setup_logger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config/pipeline/x02.yaml"):
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

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def _map_provider_name_to_code(self, provider_name: str) -> str:
        """
        Map provider name from configuration to provider code for get_ohlcv.

        Args:
            provider_name: Provider name from config (e.g., 'binance', 'yfinance')

        Returns:
            Provider code for get_ohlcv (e.g., 'bnc', 'yf')
        """
        # Direct mappings for common provider names
        provider_mapping = {
            'binance': 'bnc',
            'yfinance': 'yf',
            'yahoo': 'yf',
            'alphavantage': 'av',
            'finnhub': 'fh',
            'polygon': 'pg',
            'twelvedata': 'td',
            'coingecko': 'cg'
        }

        # Check direct mapping first
        if provider_name.lower() in provider_mapping:
            return provider_mapping[provider_name.lower()]

        # If not found in direct mapping, check if it's already a valid provider code
        if DataDownloaderFactory._normalize_provider(provider_name):
            return provider_name

        # If still not found, try to normalize using the factory
        normalized = DataDownloaderFactory._normalize_provider(provider_name)
        if normalized:
            # Find the shortest provider code for this normalized name
            for code, norm_name in DataDownloaderFactory.PROVIDER_MAP.items():
                if norm_name == normalized:
                    return code

        # If all else fails, return the original name and let get_ohlcv handle the error
        _logger.warning("Unknown provider name: %s, using as-is", provider_name)
        return provider_name

    def _generate_filename(self, symbol: str, timeframe: str, start_date: datetime, end_date: datetime, provider: str = None) -> str:
        """
        Generate filename for downloaded data.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '5m', '1h', '1d')
            start_date: Start date
            end_date: End date
            provider: Data provider name (optional)

        Returns:
            Generated filename
        """
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        if provider:
            return f"{provider}_{symbol}_{timeframe}_{start_str}_{end_str}.csv"
        else:
            return f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"

    def _download_single_dataset(self, provider: str, symbol: str, timeframe: str) -> Tuple[str, bool, str]:
        """
        Download data for a single symbol-timeframe combination.

        Args:
            provider: Data provider name
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Tuple of (filename, success, error_message)
        """
        try:
            provider_code = self._map_provider_name_to_code(provider)

            # Calculate date range
            period = self.config['data']['period']
            end_date = datetime.now()

            # Parse period and calculate start date
            if period.endswith('y'):
                years = int(period[:-1])
                start_date = end_date - timedelta(days=years * 365)
            elif period.endswith('m'):
                months = int(period[:-1])
                start_date = end_date - timedelta(days=months * 30)
            elif period.endswith('d'):
                days = int(period[:-1])
                start_date = end_date - timedelta(days=days)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year

            # Generate filename
            filename = self._generate_filename(symbol, timeframe, start_date, end_date, provider)
            filepath = self.data_dir / filename

            # Check if file already exists
            if filepath.exists():
                _logger.info("File already exists: %s", filename)
                return filename, True, ""

            # Download data
            _logger.info("Downloading %s %s %s data...", provider, symbol, timeframe)

            # Use the existing get_ohlcv function
            df = get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                provider=provider_code,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if df is None or df.empty:
                error_msg = f"No data returned for {provider} {symbol} {timeframe}"
                _logger.warning(error_msg)
                return filename, False, error_msg

            # Save to CSV
            df.to_csv(filepath, index=True)
            _logger.info("Successfully downloaded %s (%d rows)", filename, len(df))

            # Rate limiting delay
            delay = self.config['data'].get('rate_limit_delay', 1.0)
            if delay > 0:
                time.sleep(delay)

            return filename, True, ""

        except Exception as e:
            error_msg = f"Error downloading {provider} {symbol} {timeframe}: {str(e)}"
            _logger.error(error_msg)
            return f"{provider}_{symbol}_{timeframe}.csv", False, error_msg

    def _get_all_download_tasks(self) -> List[Tuple[str, str, str]]:
        """
        Get all download tasks from configuration.

        Returns:
            List of (provider, symbol, timeframe) tuples
        """
        tasks = []
        data_sources = self.config.get('data_sources', {})

        for provider, config in data_sources.items():
            symbols = config.get('symbols', [])
            timeframes = config.get('timeframes', [])

            for symbol in symbols:
                for timeframe in timeframes:
                    tasks.append((provider, symbol, timeframe))

        return tasks

    def run(self) -> Dict[str, List[str]]:
        """
        Run the data loading process.

        Returns:
            Dictionary with 'success' and 'failed' lists of filenames
        """
        _logger.info("Starting data loading process...")

        # Get all download tasks
        tasks = self._get_all_download_tasks()

        if not tasks:
            _logger.warning("No download tasks found in configuration")
            return {'success': [], 'failed': []}

        _logger.info("Found %d download tasks", len(tasks))

        # Results tracking
        successful_downloads = []
        failed_downloads = []

        # Parallel downloading
        max_workers = self.config['data'].get('parallel_downloads', 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._download_single_dataset, provider, symbol, timeframe): (provider, symbol, timeframe)
                for provider, symbol, timeframe in tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_task):
                provider, symbol, timeframe = future_to_task[future]

                try:
                    filename, success, error_msg = future.result()

                    if success:
                        successful_downloads.append(filename)
                    else:
                        failed_downloads.append(filename)
                        _logger.error("Failed to download %s %s %s: %s", provider, symbol, timeframe, error_msg)

                except Exception:
                    filename = f"{provider}_{symbol}_{timeframe}.csv"
                    failed_downloads.append(filename)
                    _logger.exception("Exception downloading %s %s %s:", provider, symbol, timeframe)

        # Summary
        _logger.info("Data loading completed:")
        _logger.info("  Successful downloads: %d", len(successful_downloads))
        _logger.info("  Failed downloads: %d", len(failed_downloads))

        if failed_downloads:
            _logger.warning("Failed downloads: %s", failed_downloads)

        return {
            'success': successful_downloads,
            'failed': failed_downloads
        }

    def validate_downloaded_data(self) -> Dict[str, List[str]]:
        """
        Validate downloaded data files.

        Returns:
            Dictionary with 'valid' and 'invalid' lists of filenames
        """
        _logger.info("Validating downloaded data...")

        valid_files = []
        invalid_files = []

        for filepath in self.data_dir.glob("*.csv"):
            try:
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)

                # Basic validation
                if len(df) < 100:  # Minimum data points
                    _logger.warning("File %s has insufficient data: %d rows", filepath.name, len(df))
                    invalid_files.append(filepath.name)
                    continue

                # Check for required columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    _logger.warning("File %s missing columns: %s", filepath.name, missing_columns)
                    invalid_files.append(filepath.name)
                    continue

                # Check for missing values
                missing_pct = df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns))
                if missing_pct > 0.1:  # More than 10% missing values
                    _logger.warning("File %s has too many missing values: %.2f%%", filepath.name, missing_pct * 100)
                    invalid_files.append(filepath.name)
                    continue

                valid_files.append(filepath.name)

            except Exception:
                _logger.exception("Error validating file %s:", filepath.name)
                invalid_files.append(filepath.name)

        _logger.info("Data validation completed:")
        _logger.info("  Valid files: %d", len(valid_files))
        _logger.info("  Invalid files: %d", len(invalid_files))

        return {
            'valid': valid_files,
            'invalid': invalid_files
        }

def main():
    """Main entry point for data loading."""
    import argparse

    parser = argparse.ArgumentParser(description='Download OHLCV data for CNN-LSTM-XGBoost pipeline')
    parser.add_argument('--config', default='config/pipeline/p02.yaml', help='Configuration file path')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing data')

    args = parser.parse_args()

    try:
        data_loader = DataLoader(args.config)

        if args.validate_only:
            results = data_loader.validate_downloaded_data()
        else:
            results = data_loader.run()
            # Also validate the downloaded data
            validation_results = data_loader.validate_downloaded_data()
            results['validation'] = validation_results

        print("Data loading results:", results)

    except Exception:
        _logger.exception("Data loading failed:")
        sys.exit(1)

if __name__ == "__main__":
    main()
