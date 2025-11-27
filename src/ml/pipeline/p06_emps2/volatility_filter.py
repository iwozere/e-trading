"""
Volatility Filter

Applies volatility-based filters using intraday data from Yahoo Finance.
Calculates ATR using TA-Lib and filters by price, ATR/Price ratio, and price range.
"""

from pathlib import Path
import sys
from typing import List, Dict
from datetime import datetime, timedelta

import pandas as pd
import talib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p06_emps2.config import EMPS2FilterConfig

_logger = setup_logger(__name__)


class VolatilityFilter:
    """
    Applies volatility-based filters to stock universe.

    Uses Yahoo Finance intraday data to calculate:
    - ATR (Average True Range) using TA-Lib
    - ATR/Price ratio
    - Price range over lookback period

    Filters for stocks showing early volatility expansion.
    """

    def __init__(self, downloader: YahooDataDownloader, config: EMPS2FilterConfig):
        """
        Initialize volatility filter.

        Args:
            downloader: Yahoo Finance data downloader instance
            config: Filter configuration
        """
        self.downloader = downloader
        self.config = config

        # Results directory (dated)
        today = datetime.now().strftime('%Y-%m-%d')
        self._results_dir = Path("results") / "emps2" / today
        self._results_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Volatility Filter initialized: ATR/Price>%.1f%%, range>%.1f%%, lookback=%dd",
                    config.min_volatility_threshold * 100,
                    config.min_price_range * 100,
                    config.lookback_days)

    def apply_filters(self, tickers: List[str]) -> List[str]:
        """
        Apply volatility filters to ticker list.

        Args:
            tickers: List of ticker symbols

        Returns:
            List of tickers passing volatility filters
        """
        try:
            _logger.info("Applying volatility filters to %d tickers", len(tickers))

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days)

            # Download intraday data in batch
            _logger.info("Downloading %s data for %d tickers (%s to %s)",
                        self.config.interval, len(tickers),
                        start_date.date(), end_date.date())

            ohlcv_data = self.downloader.get_ohlcv_batch(
                tickers,
                self.config.interval,
                start_date,
                end_date
            )

            # Apply filters
            passed_tickers = []
            results_data = []

            for ticker in tickers:
                try:
                    df = ohlcv_data.get(ticker)

                    if df is None or df.empty:
                        _logger.debug("No data for %s", ticker)
                        continue

                    # Check minimum bars requirement
                    if len(df) < 20:
                        _logger.debug("%s has insufficient data (%d bars)", ticker, len(df))
                        continue

                    # Apply filters
                    passed, metrics = self._check_volatility_filters(ticker, df)

                    if passed:
                        passed_tickers.append(ticker)
                        results_data.append(metrics)

                except Exception:
                    _logger.exception("Error processing %s:", ticker)
                    continue

            _logger.info("After volatility filtering: %d tickers (%.1f%%)",
                        len(passed_tickers),
                        100.0 * len(passed_tickers) / len(tickers) if tickers else 0)

            # Save results
            self._save_results(results_data)

            return passed_tickers

        except Exception:
            _logger.exception("Error applying volatility filters:")
            return []

    def _check_volatility_filters(self, ticker: str, df: pd.DataFrame) -> tuple:
        """
        Check if ticker passes volatility filters.

        Args:
            ticker: Ticker symbol
            df: OHLCV DataFrame

        Returns:
            Tuple of (passed: bool, metrics: dict)
        """
        try:
            # Ensure data is sorted by timestamp
            df = df.sort_values('timestamp').copy()

            # Get latest price
            last_price = df['close'].iloc[-1]

            # Price filter
            if last_price < self.config.min_price:
                return False, {}

            # Calculate ATR using TA-Lib
            atr = self._compute_atr(df)

            if atr is None or pd.isna(atr):
                return False, {}

            # ATR/Price ratio
            atr_ratio = atr / last_price

            if atr_ratio < self.config.min_volatility_threshold:
                return False, {}

            # Price range filter
            price_high = df['high'].max()
            price_low = df['low'].min()
            price_range = (price_high - price_low) / price_low

            if price_range < self.config.min_price_range:
                return False, {}

            # Passed all filters
            metrics = {
                'ticker': ticker,
                'last_price': last_price,
                'atr': atr,
                'atr_ratio': atr_ratio,
                'price_range': price_range,
                'price_high': price_high,
                'price_low': price_low,
                'bars_count': len(df)
            }

            _logger.debug("%s passed: price=$%.2f, ATR/Price=%.3f, range=%.3f",
                         ticker, last_price, atr_ratio, price_range)

            return True, metrics

        except Exception:
            _logger.exception("Error checking filters for %s:", ticker)
            return False, {}

    def _compute_atr(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR using TA-Lib.

        Args:
            df: OHLCV DataFrame with columns: high, low, close

        Returns:
            Latest ATR value
        """
        try:
            # TA-Lib expects numpy arrays
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Calculate ATR
            atr_values = talib.ATR(high, low, close, timeperiod=self.config.atr_period)

            # Return latest ATR (skip NaN values)
            valid_atr = atr_values[~pd.isna(atr_values)]

            if len(valid_atr) == 0:
                return None

            return float(valid_atr[-1])

        except Exception:
            _logger.exception("Error computing ATR:")
            return None

    def _save_results(self, results_data: List[dict]) -> None:
        """
        Save volatility filter results to CSV.

        Args:
            results_data: List of metrics dictionaries
        """
        try:
            if not results_data:
                _logger.warning("No results to save")
                return

            df = pd.DataFrame(results_data)

            # Sort by ATR ratio (highest volatility first)
            df = df.sort_values('atr_ratio', ascending=False)

            output_path = self._results_dir / "volatility_filtered.csv"
            df.to_csv(output_path, index=False)

            _logger.info("Saved volatility filter results to: %s", output_path)

            # Log top candidates
            _logger.info("Top 10 by ATR/Price ratio:")
            for _, row in df.head(10).iterrows():
                _logger.info("  %s: ATR/Price=%.3f, Range=%.3f, Price=$%.2f",
                            row['ticker'], row['atr_ratio'], row['price_range'], row['last_price'])

        except Exception:
            _logger.exception("Error saving volatility filter results:")


def create_volatility_filter(
    downloader: YahooDataDownloader,
    config: EMPS2FilterConfig
) -> VolatilityFilter:
    """
    Factory function to create volatility filter.

    Args:
        downloader: Yahoo Finance data downloader instance
        config: Filter configuration

    Returns:
        VolatilityFilter instance
    """
    return VolatilityFilter(downloader, config)
