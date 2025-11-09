"""
Data Loader for CNN + XGBoost Trading Pipeline

This module handles downloading OHLCV data for multiple symbols and timeframes
as specified in the pipeline configuration. It uses the existing data downloader
infrastructure to fetch data from various providers.

Features:
- Downloads data for multiple symbols and timeframes in parallel
- Saves data with consistent naming convention: {provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv
- Configurable via YAML configuration file
- Progress tracking and logging
- Error handling for failed downloads
- Adds log_return column to downloaded data
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import yaml
import sys


# Add project root to path to import common utilities
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from src.notification.logger import setup_logger
from src.common import get_ohlcv, analyze_period_interval
from src.data.downloader.data_downloader_factory import DataDownloaderFactory

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

_logger = setup_logger(__name__)

class DataLoader:
    def __init__(self, config_path: str = "config/pipeline/p03.yaml"):
        """
        Initialize DataLoader with configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Load data configuration
        self.data_config = self.config["data"]

        # Load provider configurations
        if "data_sources" in self.config:
            self.provider_configs = self.config["data_sources"]
        else:
            # Legacy configuration - create single provider config
            symbols = self.config.get("symbols", [])
            timeframes = self.config.get("timeframes", [])
            provider = self.data_config.get("provider", "binance")
            self.provider_configs = {
                provider: {
                    "symbols": symbols,
                    "timeframes": timeframes
                }
            }

        # Data download period (will be overridden by provider-specific periods)
        self.default_period = self.data_config.get("period", "2y")

        # Directories
        self.input_dir = Path(self.config["paths"]["data_raw"])
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        _logger.info("Loaded configuration from %s", self.config_path)
        return config

    def run(self) -> Dict[str, Any]:
        """
        Run the data loading pipeline (download only).

        Returns:
            Dictionary containing download results
        """
        _logger.info("Starting data loading (download only)")

        try:
            # Download data for all providers
            download_results = self._download_all_data()

            # Generate summary
            total_downloads = sum(len(results["successful"]) for results in download_results.values())
            total_failures = sum(len(results["failed"]) for results in download_results.values())

            _logger.info("Data loading completed. Total downloads: %d, Total failures: %d",
                        total_downloads, total_failures)

            return {
                "providers": download_results,
                "total_downloads": total_downloads,
                "total_failures": total_failures,
                "download_summary": download_results
            }

        except Exception:
            _logger.exception("Data loading failed")
            raise

    def _download_all_data(self) -> Dict[str, Any]:
        """
        Download data for all providers according to configuration.

        Returns:
            Dictionary containing download results for each provider
        """
        _logger.info("Starting data download for all providers")

        download_results = {}

        for provider, provider_config in self.provider_configs.items():
            _logger.info("Downloading data for provider: %s", provider)

            provider_results = self._download_provider_data(provider, provider_config)
            download_results[provider] = provider_results

        return download_results

    def _download_provider_data(self, provider: str, provider_config: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Download data for a specific provider.

        Args:
            provider: Provider name
            provider_config: Provider configuration with symbols and timeframes

        Returns:
            Dictionary containing download results
        """
        symbols = provider_config["symbols"]
        timeframes = provider_config["timeframes"]

        results = {
            "total": len(symbols) * len(timeframes),
            "successful": [],
            "failed": [],
            "success": True
        }

        # Create download tasks
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append((symbol, timeframe, provider))

        # Download in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_task = {
                executor.submit(self._download_single_dataset, symbol, timeframe, provider): (symbol, timeframe, provider)
                for symbol, timeframe, provider in tasks
            }

            for future in as_completed(future_to_task):
                symbol, timeframe, provider = future_to_task[future]
                try:
                    result = future.result()
                    if result[1]:  # success
                        results["successful"].append(result)
                        _logger.info("✓ Downloaded %s %s from %s", symbol, timeframe, provider)
                    else:
                        results["failed"].append(result)
                        _logger.error("✗ Failed to download %s %s from %s: %s", symbol, timeframe, provider, result[2])
                        results["success"] = False
                except Exception as e:
                    error_result = (symbol, timeframe, provider, False, str(e))
                    results["failed"].append(error_result)
                    _logger.exception("✗ Exception downloading %s %s from %s", symbol, timeframe, provider)
                    results["success"] = False

        return results

    def _download_single_dataset(self, symbol: str, timeframe: str, provider: str) -> Tuple[str, bool, str]:
        """
        Download data for a single symbol-timeframe combination.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            provider: Data provider

        Returns:
            Tuple of (filepath, success, error_message)
        """
        try:
            # Map provider name to provider code
            provider_code = self._map_provider_name_to_code(provider)

            # Get provider-specific period
            provider_config = self.provider_configs[provider]
            period = provider_config.get("period", self.default_period)

            # Calculate date range
            start_date, end_date = analyze_period_interval(period, timeframe)

            _logger.info("Downloading %s %s from %s provider (%s to %s)",
                        symbol, timeframe, provider, start_date, end_date)

            # Download data using common utility
            df = get_ohlcv(
                ticker=symbol,
                interval=timeframe,
                period=period,
                provider=provider_code
            )

            if df.empty:
                raise ValueError(f"No data returned for {symbol} {timeframe} from {provider}")

            # Generate filename and save
            filename = f"{provider}_{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            filepath = self.input_dir / filename

            # Add log_return column if not present
            if 'log_return' not in df.columns:
                df['log_return'] = (df['close'] / df['close'].shift(1)).apply(
                    lambda x: 0 if x <= 0 else x).pipe(
                    lambda x: x.apply(lambda y: 0 if y == 0 else np.log(y)))

            # Save to CSV
            df.to_csv(filepath, index=False)

            _logger.info("Successfully saved %s %s from %s to %s", symbol, timeframe, provider, filepath)

            return str(filepath), True, ""

        except Exception as e:
            error_msg = f"Failed to download {symbol} {timeframe} from {provider}: {str(e)}"
            _logger.error(error_msg)
            return "", False, error_msg

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


def main():
    """Main function to run data loader."""
    try:
        loader = DataLoader()
        results = loader.run()

        _logger.info("Data loading completed successfully!")
        _logger.info("Total downloads: %d", results['total_downloads'])
        _logger.info("Total failures: %d", results['total_failures'])

        for provider, provider_results in results['providers'].items():
            _logger.info("Provider %s: Successful: %d, Failed: %d",
                        provider, len(provider_results['successful']), len(provider_results['failed']))

    except Exception:
        _logger.exception("Data loading failed")
        raise


if __name__ == "__main__":
    main()
