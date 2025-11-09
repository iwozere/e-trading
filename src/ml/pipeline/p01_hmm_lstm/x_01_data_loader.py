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
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.common import get_ohlcv, analyze_period_interval
from src.notification.logger import setup_logger
from src.data.downloader.data_downloader_factory import DataDownloaderFactory
_logger = setup_logger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config/pipeline/p01.yaml"):
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

            _logger.info("Downloading %s %s from %s to %s", symbol, timeframe, start_date, end_date)

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
                df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: 0 if x <= 0 else x).pipe(lambda x: x.apply(lambda y: 0 if y == 0 else np.log(y)))

            # Save to CSV
            df.to_csv(filepath, index=False)

            _logger.info("[OK] Successfully saved %s %s to %s", symbol, timeframe, filepath)
            _logger.info("  Data shape: %s, Date range: %s to %s", df.shape, df['timestamp'].min(), df['timestamp'].max())

            return symbol, timeframe, True, str(filepath)

        except Exception as e:
            error_msg = f"Failed to download {symbol} {timeframe}: {str(e)}"
            _logger.error(error_msg)
            return symbol, timeframe, False, error_msg

    def download_symbol_timeframe_multi_provider(self, symbol: str, timeframe: str, provider: str) -> tuple:
        """
        Download data for a single symbol-timeframe combination from a specific provider.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            timeframe: Timeframe (e.g., '5m', '1h', '4h')
            provider: Data provider (e.g., 'binance', 'yfinance')

        Returns:
            tuple: (symbol, timeframe, provider, success, filepath_or_error)
        """
        try:
            # Calculate date range
            period = self.config['data']['period']

            start_date, end_date = analyze_period_interval(period, timeframe)

            # Map provider name to provider code
            provider_code = self._map_provider_name_to_code(provider)
            _logger.info("Downloading %s %s from %s provider (mapped to %s) (%s to %s)",
                        symbol, timeframe, provider, provider_code, start_date, end_date)

            # Download data using common utility with specific provider
            df = get_ohlcv(
                ticker=symbol,
                interval=timeframe,
                period=period,
                provider=provider_code
            )

            if df.empty:
                raise ValueError(f"No data returned for {symbol} {timeframe} from {provider}")

            # Generate filename with provider prefix and save
            filename = f"{provider}_{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            filepath = self.data_dir / filename

            # Add log_return column if not present
            if 'log_return' not in df.columns:
                df['log_return'] = (df['close'] / df['close'].shift(1)).apply(lambda x: 0 if x <= 0 else x).pipe(lambda x: x.apply(lambda y: 0 if y == 0 else np.log(y)))

            # Save to CSV
            df.to_csv(filepath, index=False)

            _logger.info("[OK] Successfully saved %s %s from %s to %s", symbol, timeframe, provider, filepath)
            _logger.info("  Data shape: %s, Date range: %s to %s", df.shape, df['timestamp'].min(), df['timestamp'].max())

            return symbol, timeframe, provider, True, str(filepath)

        except Exception as e:
            error_msg = f"Failed to download {symbol} {timeframe} from {provider}: {str(e)}"
            _logger.error(error_msg)
            return symbol, timeframe, provider, False, error_msg

    def download_all(self, max_workers: int = 4) -> dict:
        """
        Download data for all symbol-timeframe combinations in configuration.

        Args:
            max_workers: Maximum number of parallel download threads

        Returns:
            dict: Summary of download results
        """
        # Check for multi-provider configuration
        if 'data_sources' in self.config:
            return self._download_all_multi_provider(max_workers)
        else:
            # Fallback to legacy configuration
            return self._download_all_legacy(max_workers)

    def _download_all_multi_provider(self, max_workers: int = 4) -> dict:
        """
        Download data using the new multi-provider configuration.
        """
        data_sources = self.config['data_sources']

        _logger.info("Starting multi-provider data download")

        results = {
            'total': 0,
            'successful': [],
            'failed': [],
            'overall_success': True,
            'providers': {}
        }

        for provider, config in data_sources.items():
            _logger.info("Processing provider: %s", provider)
            symbols = config.get('symbols', [])
            timeframes = config.get('timeframes', [])

            provider_results = {
                'total': len(symbols) * len(timeframes),
                'successful': [],
                'failed': [],
                'success': True
            }

            results['total'] += provider_results['total']

            # Create download tasks for this provider
            tasks = []
            for symbol in symbols:
                for timeframe in timeframes:
                    tasks.append((symbol, timeframe, provider))

            # Download in parallel for this provider
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(self.download_symbol_timeframe_multi_provider, symbol, timeframe, provider): (symbol, timeframe, provider)
                    for symbol, timeframe, provider in tasks
                }

                for future in as_completed(future_to_task):
                    symbol, timeframe, provider = future_to_task[future]
                    try:
                        result = future.result()
                        if result[2]:  # success
                            provider_results['successful'].append(result)
                            results['successful'].append(result)
                            _logger.info("✓ Downloaded %s %s from %s", symbol, timeframe, provider)
                        else:
                            provider_results['failed'].append(result)
                            results['failed'].append(result)
                            _logger.error("✗ Failed to download %s %s from %s: %s", symbol, timeframe, provider, result[3])
                            provider_results['success'] = False
                            results['overall_success'] = False
                    except Exception as e:
                        error_result = (symbol, timeframe, provider, False, str(e))
                        provider_results['failed'].append(error_result)
                        results['failed'].append(error_result)
                        _logger.exception("✗ Exception downloading %s %s from %s: ", symbol, timeframe, provider)
                        provider_results['success'] = False
                        results['overall_success'] = False

            results['providers'][provider] = provider_results

        # Log summary
        _logger.info("\n%s", "="*50)
        _logger.info("Multi-Provider Data Download Summary:")
        _logger.info("  Total: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))
        _logger.info("  Overall success: %s", results['overall_success'])

        for provider, provider_results in results['providers'].items():
            _logger.info("  %s: %d/%d successful", provider,
                        len(provider_results['successful']), provider_results['total'])

        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.error("Failed downloads:")
            for failure in results['failed']:
                # Handle both tuple format (multi-provider) and dict format (legacy)
                if isinstance(failure, tuple):
                    # Multi-provider format: (symbol, timeframe, provider, success, error)
                    symbol, timeframe, provider, success, error = failure
                    _logger.error("  %s %s (%s): %s", symbol, timeframe, provider, error)
                else:
                    # Legacy format: dict with 'symbol', 'timeframe', 'error' keys
                    _logger.error("  %s %s: %s", failure['symbol'], failure['timeframe'], failure['error'])

        return results

    def _download_all_legacy(self, max_workers: int = 4) -> dict:
        """
        Download data using the legacy configuration format.
        """
        symbols = self.config['symbols']
        timeframes = self.config['timeframes']

        # Create list of all symbol-timeframe combinations
        tasks = [(symbol, tf) for symbol in symbols for tf in timeframes]

        _logger.info("Starting download for %d symbols x %d timeframes = %d files (legacy mode)", len(symbols), len(timeframes), len(tasks))

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
        _logger.info("\n%s", "="*50)
        _logger.info("Download Summary (Legacy):")
        _logger.info("  Total: %d", results['total'])
        _logger.info("  Successful: %d", len(results['successful']))
        _logger.info("  Failed: %d", len(results['failed']))
        _logger.info("%s", "="*50)

        if results['failed']:
            _logger.warning("Failed downloads:")
            for failure in results['failed']:
                # Handle both tuple format (multi-provider) and dict format (legacy)
                if isinstance(failure, tuple):
                    # Multi-provider format: (symbol, timeframe, provider, success, error)
                    symbol, timeframe, provider, success, error = failure
                    _logger.warning("  %s %s (%s): %s", symbol, timeframe, provider, error)
                else:
                    # Legacy format: dict with 'symbol', 'timeframe', 'error' keys
                    _logger.warning("  %s %s: %s", failure['symbol'], failure['timeframe'], failure['error'])

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
        _logger.info("\nVerifying data quality...")
        for item in results['successful']:
            # Handle both tuple format (multi-provider) and dict format (legacy)
            if isinstance(item, tuple):
                # Multi-provider format: (symbol, timeframe, provider, success, filepath)
                symbol, timeframe, provider, success, filepath = item
                filepath = Path(filepath)
            else:
                # Legacy format: dict with 'filepath' key
                filepath = Path(item['filepath'])

            quality = loader.verify_data_quality(filepath)

            if 'error' in quality:
                _logger.error("Quality check failed for %s: %s", filepath, quality['error'])
            else:
                issues = []
                if quality['missing_columns']:
                    issues.append(f"Missing columns: {quality['missing_columns']}")
                if quality['duplicate_timestamps'] > 0:
                    issues.append(f"Duplicate timestamps: {quality['duplicate_timestamps']}")
                if quality['invalid_ohlc'] > 0:
                    issues.append(f"Invalid OHLC: {quality['invalid_ohlc']}")

                if issues:
                    _logger.warning("%s: %s", filepath.name, ', '.join(issues))
                else:
                    _logger.info("[OK] %s: Quality OK - %d rows", filepath.name, quality['shape'][0])

        _logger.info("Data loading completed!")

    except Exception:
        _logger.exception("Data loading failed")
        raise

if __name__ == "__main__":
    main()
