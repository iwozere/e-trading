"""
Volatility Filter

Applies volatility-based filters using intraday data from Yahoo Finance.
Calculates ATR using TA-Lib and filters by price, ATR/Price ratio, and price range.

Enhanced with EMPS accumulation indicators:
- Volume Z-Score: Detects unusual volume spikes
- Volume/Volatility Ratio: Detects accumulation (high volume + low price volatility)
"""

from pathlib import Path
import sys
from typing import List, Dict, Optional
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
from src.ml.pipeline.p06_emps2.trf_downloader import get_trf_correction_factor

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

    def __init__(self, downloader: YahooDataDownloader, config: EMPS2FilterConfig, target_date: Optional[str] = None):
        """
        Initialize volatility filter.

        Args:
            downloader: Yahoo Finance data downloader instance
            config: Filter configuration
            target_date: Target trading date (YYYY-MM-DD). Defaults to today.
        """
        self.downloader = downloader
        self.config = config

        # Results directory (dated)
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        self._results_dir = Path("results") / "emps2" / target_date
        self._results_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Volatility Filter initialized: ATR/Price>%.1f%%, range>%.1f%%, lookback=%dd, vol_zscore>%.1f, vol_rv_ratio>%.1f",
                    config.min_volatility_threshold * 100,
                    config.min_price_range * 100,
                    config.lookback_days,
                    config.min_vol_zscore,
                    config.min_vol_rv_ratio)

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

            # Load TRF volume correction factors
            trf_corrections = self._load_trf_volume_corrections()
            if trf_corrections:
                _logger.info("Using TRF volume corrections")
            else:
                _logger.warning("No TRF data found - using raw volume data")

            # Apply filters
            passed_tickers = []
            results_data = []
            diagnostic_data = []  # Track ALL tickers for diagnostics

            for ticker in tickers:
                try:
                    df = ohlcv_data.get(ticker)

                    if df is None or df.empty:
                        _logger.debug("No data for %s", ticker)
                        diagnostic_data.append({
                            'ticker': ticker,
                            'status': 'FAILED',
                            'reason': 'no_data',
                            'last_price': None,
                            'atr': None,
                            'atr_ratio': None,
                            'price_range': None,
                            'vol_zscore': None,
                            'vol_rv_ratio': None,
                            'rv_short': None,
                            'rv_long': None,
                            'bars_count': 0,
                            'trf_correction_factor': None
                        })
                        continue

                    # Check minimum bars requirement
                    if len(df) < 20:
                        _logger.debug("%s has insufficient data (%d bars)", ticker, len(df))
                        diagnostic_data.append({
                            'ticker': ticker,
                            'status': 'FAILED',
                            'reason': 'insufficient_bars',
                            'last_price': None,
                            'atr': None,
                            'atr_ratio': None,
                            'price_range': None,
                            'vol_zscore': None,
                            'vol_rv_ratio': None,
                            'rv_short': None,
                            'rv_long': None,
                            'bars_count': len(df),
                            'trf_correction_factor': None
                        })
                        continue

                    # Apply TRF volume correction if available
                    if ticker in trf_corrections:
                        trf_factor = trf_corrections[ticker]
                        df = self._apply_trf_volume_correction(df, trf_factor)
                        _logger.debug("%s: Applied TRF correction factor %.3f", ticker, trf_factor)
                    else:
                        trf_factor = None

                    # Apply filters
                    passed, metrics, reason = self._check_volatility_filters(ticker, df)

                    # Add TRF correction factor to metrics
                    if metrics:
                        metrics['trf_correction_factor'] = trf_factor

                    if passed:
                        passed_tickers.append(ticker)
                        results_data.append(metrics)
                        # Add to diagnostics with PASSED status
                        diag_entry = metrics.copy()
                        diag_entry['status'] = 'PASSED'
                        diag_entry['reason'] = 'all_filters_passed'
                        diagnostic_data.append(diag_entry)
                    else:
                        # Add to diagnostics with failure reason
                        diag_entry = metrics.copy() if metrics else {
                            'ticker': ticker,
                            'last_price': None,
                            'atr': None,
                            'atr_ratio': None,
                            'price_range': None,
                            'vol_zscore': None,
                            'vol_rv_ratio': None,
                            'rv_short': None,
                            'rv_long': None,
                            'bars_count': len(df)
                        }
                        diag_entry['status'] = 'FAILED'
                        diag_entry['reason'] = reason
                        diagnostic_data.append(diag_entry)

                except Exception:
                    _logger.exception("Error processing %s:", ticker)
                    diagnostic_data.append({
                        'ticker': ticker,
                        'status': 'ERROR',
                        'reason': 'exception',
                        'last_price': None,
                        'atr': None,
                        'atr_ratio': None,
                        'price_range': None,
                        'vol_zscore': None,
                        'vol_rv_ratio': None,
                        'rv_short': None,
                        'rv_long': None,
                        'bars_count': 0
                    })
                    continue

            _logger.info("After volatility filtering: %d tickers (%.1f%%)",
                        len(passed_tickers),
                        100.0 * len(passed_tickers) / len(tickers) if tickers else 0)

            # Save results (passing tickers only)
            self._save_results(results_data)

            # Save diagnostics (all tickers)
            self._save_diagnostics(diagnostic_data)

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
            Tuple of (passed: bool, metrics: dict, reason: str)
            - passed: True if all filters passed
            - metrics: Dictionary with all calculated metrics
            - reason: Specific filter that failed (or 'all_filters_passed')
        """
        try:
            # Ensure data is sorted by timestamp
            df = df.sort_values('timestamp').copy()

            # Get latest price
            last_price = df['close'].iloc[-1]

            # Calculate all metrics first (for diagnostics)
            # Calculate ATR using TA-Lib
            atr = self._compute_atr(df)
            atr_ratio = None

            if atr is not None and not pd.isna(atr) and last_price > 0:
                atr_ratio = atr / last_price

            # Price range filter
            price_high = df['high'].max()
            price_low = df['low'].min()

            if price_low > 0:
                price_range = (price_high - price_low) / price_low
            else:
                price_range = 0.0

            # Volume Z-Score (from P05 EMPS)
            vol_zscore = self._compute_volume_zscore(df)

            # Volume/Volatility Ratio - Detects accumulation (high volume + low price volatility)
            vol_rv_ratio, rv_short, rv_long = self._compute_volume_volatility_ratio(df)

            # Build metrics dictionary with all calculated values
            metrics = {
                'ticker': ticker,
                'last_price': last_price,
                'atr': atr,
                'atr_ratio': atr_ratio,
                'price_range': price_range,
                'price_high': price_high,
                'price_low': price_low,
                'vol_zscore': vol_zscore,
                'vol_rv_ratio': vol_rv_ratio,
                'rv_short': rv_short,
                'rv_long': rv_long,
                'bars_count': len(df)
            }

            # Now apply filters with specific failure reasons
            # Price filter
            if last_price < self.config.min_price:
                return False, metrics, f'price_too_low (${last_price:.2f} < ${self.config.min_price})'

            # ATR check
            if atr is None or pd.isna(atr):
                _logger.warning("ATR calculation failed for %s. Data length: %d, Price range: %.2f-%.2f, OHLC Sample:\n%s",
                                ticker, len(df), price_low, price_high, df[['high', 'low', 'close']].tail())
                return False, metrics, 'atr_calculation_failed'

            # ATR/Price ratio
            if atr_ratio < self.config.min_volatility_threshold:
                return False, metrics, f'atr_ratio_too_low ({atr_ratio:.4f} < {self.config.min_volatility_threshold})'

            # Price range filter
            if price_range < self.config.min_price_range:
                return False, metrics, f'price_range_too_low ({price_range:.4f} < {self.config.min_price_range})'

            # Volume Z-Score
            if vol_zscore < self.config.min_vol_zscore:
                return False, metrics, f'vol_zscore_too_low ({vol_zscore:.2f} < {self.config.min_vol_zscore})'

            # Volume/Volatility Ratio (accumulation detection)
            if vol_rv_ratio < self.config.min_vol_rv_ratio:
                return False, metrics, f'vol_rv_ratio_too_low ({vol_rv_ratio:.2f} < {self.config.min_vol_rv_ratio})'

            # Passed all filters
            _logger.debug("%s passed: price=$%.2f, ATR/Price=%.3f, range=%.3f, vol_zscore=%.2f, vol_rv_ratio=%.2f",
                         ticker, last_price, atr_ratio, price_range, vol_zscore, vol_rv_ratio)

            return True, metrics, 'all_filters_passed'

        except Exception:
            _logger.exception("Error checking filters for %s:", ticker)
            return False, {}, 'exception_during_calculation'

    def _compute_atr(self, df: pd.DataFrame) -> float:
        """
        Calculate ATR using TA-Lib.

        Args:
            df: OHLCV DataFrame with columns: high, low, close

        Returns:
            Latest ATR value
        """
        try:
            # Debug: Print DataFrame info
            _logger.debug("DataFrame shape: %s", df.shape)
            _logger.debug("DataFrame head:\n%s", df[['high', 'low', 'close']].head())

            # Check for required columns
            required_columns = ['high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                _logger.error("Missing required columns in DataFrame. Available columns: %s", df.columns.tolist())
                return None

            # Check for NaN values in input data
            if df[['high', 'low', 'close']].isna().any().any():
                _logger.warning("NaN values found in price data. Filling with previous values.")
                df[['high', 'low', 'close']] = df[['high', 'low', 'close']].ffill()

            # Ensure we have enough data points
            min_required_bars = self.config.atr_period + 1  # ATR needs at least period + 1 bars
            if len(df) < min_required_bars:
                _logger.warning("Insufficient data points for ATR calculation. Need at least %d, got %d",
                            min_required_bars, len(df))
                return None

            # TA-Lib expects numpy arrays
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # Debug: Print first few values
            _logger.debug("First few values - High: %s, Low: %s, Close: %s", high[:5], low[:5], close[:5])

            # Calculate ATR
            atr_values = talib.ATR(high, low, close, timeperiod=self.config.atr_period)

            # Debug: Print ATR values
            _logger.debug("ATR values: %s", atr_values)

            # Return latest ATR (skip NaN values)
            valid_atr = atr_values[~pd.isna(atr_values)]

            if len(valid_atr) == 0:
                _logger.warning("No valid ATR values calculated")
                return None

            return float(valid_atr[-1])

        except Exception as e:
            _logger.exception("Error computing ATR: %s", str(e))
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

    def _compute_volume_volatility_ratio(self, df: pd.DataFrame,
                                          short_window: int = 5,
                                          long_window: int = 40,
                                          vol_window: int = 20) -> tuple:
        """
        Calculate Volume/Volatility Ratio for accumulation detection.

        Detects high volume during low price volatility - a key indicator
        of stealth accumulation before explosive moves.

        High ratio = High volume + Low price volatility = Accumulation phase
        Low ratio = Normal volume or high volatility = Not accumulating

        Args:
            df: OHLCV DataFrame with 'close' and 'volume' columns
            short_window: Short-term volatility window (default: 5 bars)
            long_window: Long-term volatility window (default: 40 bars)
            vol_window: Volume z-score window (default: 20 bars)

        Returns:
            Tuple of (vol_rv_ratio, rv_short, rv_long)
            - vol_rv_ratio: Volume Z-Score / RV_short (higher = accumulation)
            - rv_short: Short-term realized volatility (for reference)
            - rv_long: Long-term realized volatility (for reference)
        """
        try:
            if len(df) < max(long_window + 1, vol_window + 1):
                _logger.debug(f"Insufficient data for volatility ratio calculation: {len(df)} bars < {max(long_window + 1, vol_window + 1)}")
                return 0.0, 0.0, 0.0

            # Calculate log returns for price volatility
            close = df['close'].values
            log_returns = np.log(close[1:] / close[:-1])

            # Determine bars per day based on interval
            interval_map = {
                '5m': 78,
                '15m': 26,
                '30m': 13,
                '1h': 6.5
            }
            bars_per_day = interval_map.get(self.config.interval, 26)

            # Calculate realized volatility (short-term)
            if len(log_returns) >= long_window:
                rv_short = np.std(log_returns[-short_window:]) * np.sqrt(252 * bars_per_day)
                rv_long = np.std(log_returns[-long_window:]) * np.sqrt(252 * bars_per_day)
            else:
                _logger.debug(f"Insufficient log returns data for RV calculation: {len(log_returns)} < {long_window}")
                return 0.0, 0.0, 0.0

            # Calculate volume z-score
            volume = df['volume'].values
            if len(volume) >= vol_window:
                vol_mean = np.mean(volume[-vol_window:])
                vol_std = np.std(volume[-vol_window:])

                if vol_std > 0:
                    current_vol_zscore = (volume[-1] - vol_mean) / vol_std
                else:
                    current_vol_zscore = 0.0
            else:
                _logger.debug(f"Insufficient volume data for z-score calculation: {len(volume)} < {vol_window}")
                current_vol_zscore = 0.0

            # Calculate Volume/Volatility Ratio
            # High vol_rv_ratio = High volume (accumulation) + Low price volatility (quiet)
            if rv_short > 0 and not np.isnan(rv_short):
                vol_rv_ratio = current_vol_zscore / rv_short
            else:
                _logger.debug(f"Invalid RV short for vol_rv_ratio: rv_short={rv_short}, nan={np.isnan(rv_short)}, rv_long={rv_long}, current_vol_zscore={current_vol_zscore}")
                vol_rv_ratio = 0.0

            return float(vol_rv_ratio), float(rv_short), float(rv_long)

        except Exception:
            _logger.exception("Error computing Volume/Volatility ratio:")
            return 0.0, 0.0, 0.0

    def _load_trf_volume_corrections(self) -> Dict[str, float]:
        """
        Load TRF data and calculate volume correction factors.

        TRF data shows off-exchange (dark pool) volume that's not fully captured
        in yfinance intraday data. We calculate a correction factor to augment
        intraday volume with the dark pool component.

        Returns:
            Dictionary mapping ticker -> correction_factor
            correction_factor = (yfinance_volume + trf_volume) / yfinance_volume
        """
        try:
            # Get the date from the lookback period
            target_date = datetime.now() - timedelta(days=self.config.lookback_days)

            # Get the correction factor using the new function
            correction_factor = get_trf_correction_factor("", target_date)  # Empty ticker will be handled by the function

            # If we got a valid correction factor, return it in the expected format
            if correction_factor != 1.0:
                _logger.info(f"Using TRF correction factor: {correction_factor:.3f}")
                return {"*": correction_factor}  # Use "*" as a wildcard for all tickers
            else:
                _logger.warning("No valid TRF correction factor found")
                return {}

        except Exception as e:
            _logger.error(f"Error getting TRF correction factor: {str(e)}")
            return {}

    def _apply_trf_volume_correction(self, df: pd.DataFrame, correction_factor: float) -> pd.DataFrame:
        """
        Apply TRF volume correction factor to intraday OHLCV data.

        IMPORTANT: TRF data is only available for previous days, so we only
        apply corrections to historical bars. Today's bars use raw yfinance volume.

        Args:
            df: OHLCV DataFrame with 'volume' and 'timestamp' columns
            correction_factor: Multiplier to apply to volume

        Returns:
            DataFrame with corrected volume
        """
        try:
            df = df.copy()
            today = datetime.now().date()

            # Only apply correction to historical bars (not today's)
            historical_mask = df['timestamp'].dt.date < today

            if historical_mask.any():
                df.loc[historical_mask, 'volume'] = df.loc[historical_mask, 'volume'] * correction_factor
                _logger.debug("Applied TRF correction (%.3f) to %d historical bars",
                            correction_factor, historical_mask.sum())

            today_bars = (~historical_mask).sum()
            if today_bars > 0:
                _logger.debug("Kept %d today's bars unchanged (no TRF data yet)", today_bars)

            return df

        except Exception as e:
            _logger.exception("Error applying TRF volume correction:")
            return df


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

            output_path = self._results_dir / "04_volatility_filtered.csv"
            df.to_csv(output_path, index=False)

            _logger.info("Saved volatility filter results to: %s", output_path)

            # Log top candidates
            _logger.info("Top 10 by ATR/Price ratio:")
            for _, row in df.head(10).iterrows():
                _logger.info("  %s: ATR/Price=%.3f, Range=%.3f, Price=$%.2f",
                            row['ticker'], row['atr_ratio'], row['price_range'], row['last_price'])

        except Exception:
            _logger.exception("Error saving volatility filter results:")

    def _save_diagnostics(self, diagnostic_data: List[dict]) -> None:
        """
        Save diagnostic data for ALL tickers (passed and failed).

        This CSV shows calculated metrics for every ticker and the specific
        filter that caused rejection, helping diagnose filter effectiveness.

        Args:
            diagnostic_data: List of diagnostic dictionaries with status and reason
        """
        try:
            if not diagnostic_data:
                _logger.warning("No diagnostic data to save")
                return

            df = pd.DataFrame(diagnostic_data)

            # Reorder columns for better readability
            column_order = [
                'ticker', 'status', 'reason',
                'last_price', 'atr', 'atr_ratio', 'price_range',
                'vol_zscore', 'vol_rv_ratio', 'rv_short', 'rv_long',
                'price_high', 'price_low', 'bars_count', 'trf_correction_factor'
            ]

            # Only include columns that exist
            available_cols = [col for col in column_order if col in df.columns]
            df = df[available_cols]

            # Sort by status (PASSED first, then FAILED, then ERROR)
            status_order = {'PASSED': 0, 'FAILED': 1, 'ERROR': 2}
            df['_sort_order'] = df['status'].map(status_order)
            df = df.sort_values(['_sort_order', 'ticker'])
            df = df.drop(columns=['_sort_order'])

            output_path = self._results_dir / "05_volatility_diagnostics.csv"
            df.to_csv(output_path, index=False)

            _logger.info("Saved volatility diagnostics to: %s", output_path)

            # Log summary statistics
            status_counts = df['status'].value_counts()
            _logger.info("Diagnostic Summary:")
            for status, count in status_counts.items():
                _logger.info("  %s: %d tickers", status, count)

            # Log failure reason breakdown
            if 'FAILED' in status_counts:
                _logger.info("Failure Reasons:")
                failed_df = df[df['status'] == 'FAILED']
                reason_counts = failed_df['reason'].value_counts()
                for reason, count in reason_counts.head(10).items():
                    _logger.info("  %s: %d tickers", reason, count)

        except Exception:
            _logger.exception("Error saving volatility diagnostics:")


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
