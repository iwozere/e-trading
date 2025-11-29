"""
Volatility Filter

Applies volatility-based filters using intraday data from Yahoo Finance.
Calculates ATR using TA-Lib and filters by price, ATR/Price ratio, and price range.

Enhanced with P05 EMPS indicators:
- Volume Z-Score: Detects unusual volume spikes
- RV Ratio: Measures volatility regime shifts (short-term vs long-term)
"""

from pathlib import Path
import sys
from typing import List, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
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

        _logger.info("Volatility Filter initialized: ATR/Price>%.1f%%, range>%.1f%%, lookback=%dd, vol_zscore>%.1f, rv_ratio>%.1f",
                    config.min_volatility_threshold * 100,
                    config.min_price_range * 100,
                    config.lookback_days,
                    config.min_vol_zscore,
                    config.min_rv_ratio)

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

            # Volume Z-Score (from P05 EMPS)
            vol_zscore = self._compute_volume_zscore(df)

            if vol_zscore < self.config.min_vol_zscore:
                return False, {}

            # RV Ratio - Realized Volatility acceleration (from P05 EMPS)
            rv_ratio, rv_short, rv_long = self._compute_rv_ratio(df)

            if rv_ratio < self.config.min_rv_ratio:
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
                'vol_zscore': vol_zscore,
                'rv_ratio': rv_ratio,
                'rv_short': rv_short,
                'rv_long': rv_long,
                'bars_count': len(df)
            }

            _logger.debug("%s passed: price=$%.2f, ATR/Price=%.3f, range=%.3f, vol_zscore=%.2f, rv_ratio=%.2f",
                         ticker, last_price, atr_ratio, price_range, vol_zscore, rv_ratio)

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

    def _compute_volume_zscore(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """
        Calculate Volume Z-Score (from P05 EMPS).

        Detects unusual volume spikes by measuring standard deviations
        from rolling average.

        Args:
            df: OHLCV DataFrame with 'volume' column
            lookback: Rolling window size (default: 20 bars)

        Returns:
            Latest volume z-score (higher = more unusual volume)
        """
        try:
            if len(df) < lookback + 1:
                return 0.0

            volume = df['volume'].values

            # Calculate rolling mean and std
            vol_mean = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean()
            vol_std = pd.Series(volume).rolling(window=lookback, min_periods=lookback).std()

            # Z-score = (current - mean) / std
            vol_zscore = (volume - vol_mean) / vol_std

            # Return latest z-score
            valid_zscore = vol_zscore[~pd.isna(vol_zscore)]

            if len(valid_zscore) == 0:
                return 0.0

            return float(valid_zscore.iloc[-1])

        except Exception:
            _logger.exception("Error computing volume z-score:")
            return 0.0

    def _compute_rv_ratio(self, df: pd.DataFrame,
                          short_window: int = 5,
                          long_window: int = 40) -> tuple:
        """
        Calculate Realized Volatility Ratio (from P05 EMPS).

        Measures volatility regime shifts by comparing short-term to
        long-term realized volatility. Values >1.5 indicate acceleration.

        Args:
            df: OHLCV DataFrame with 'close' column
            short_window: Short-term window (default: 5 bars = ~75 min for 15m)
            long_window: Long-term window (default: 40 bars = ~10 hours for 15m)

        Returns:
            Tuple of (rv_ratio, rv_short, rv_long)
            - rv_ratio: Short/Long ratio (>1.5 = acceleration)
            - rv_short: Short-term annualized volatility
            - rv_long: Long-term annualized volatility
        """
        try:
            if len(df) < long_window + 1:
                return 1.0, 0.0, 0.0

            # Calculate log returns
            close = df['close'].values
            log_returns = np.log(close[1:] / close[:-1])

            # Determine bars per day based on interval
            # 15m = 26 bars/day (6.5 trading hours)
            # 5m = 78 bars/day
            # 1h = 6.5 bars/day
            interval_map = {
                '5m': 78,
                '15m': 26,
                '30m': 13,
                '1h': 6.5
            }
            bars_per_day = interval_map.get(self.config.interval, 26)

            # Calculate realized volatility (annualized)
            # RV = std(returns) * sqrt(252 * bars_per_day)
            if len(log_returns) >= long_window:
                rv_short = np.std(log_returns[-short_window:]) * np.sqrt(252 * bars_per_day)
                rv_long = np.std(log_returns[-long_window:]) * np.sqrt(252 * bars_per_day)
            else:
                return 1.0, 0.0, 0.0

            # Avoid division by zero
            if rv_long == 0 or np.isnan(rv_long):
                return 1.0, rv_short, 0.0

            rv_ratio = rv_short / rv_long

            return float(rv_ratio), float(rv_short), float(rv_long)

        except Exception:
            _logger.exception("Error computing RV ratio:")
            return 1.0, 0.0, 0.0

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
