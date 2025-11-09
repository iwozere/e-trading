#!/usr/bin/env python3
"""
Pipeline Step 2: Calculate Higher Timeframes from 1-Minute Data

This script calculates 5m, 15m, 1h, 4h, and 1d timeframes from 1-minute data with intelligence.
Data is saved in yearly format with JSON metadata for efficient storage and retrieval.

Intelligence Features:
- Trading day: 4:00 AM to 8:00 PM ET (16 hours)
- Missing data handling: If 1m bars are completely missing for a timeframe interval, skip that bar
- Partial data handling: If only some 1m bars exist, accumulate them into the timeframe bar
- Yearly storage: Save data as YYYY.csv.gz files with metadata
- Incremental processing: Only recalculate what's needed
- Gap preservation: Maintains gaps in data as they exist in 1m data

Timeframes:
- 5m: 5-minute bars (3 x 1m bars expected)
- 15m: 15-minute bars (15 x 1m bars expected)
- 1h: 1-hour bars (60 x 1m bars expected)
- 4h: 4-hour bars (240 x 1m bars expected)
- 1d: Daily bars (960 x 1m bars expected, 4 AM to 8 PM)

Output Structure:
DATA_CACHE_DIR/ohlcv/TICKER/
├── 5m/
│   ├── 2020.csv.gz
│   ├── 2021.csv.gz
│   └── metadata.json
├── 15m/
│   ├── 2020.csv.gz
│   └── metadata.json
└── ...

Requirements:
- Step 1 must be completed (1m data available)
- pandas, numpy for calculations

Usage:
    python src/data/cache/pipeline/step02_calculate_timeframes.py
    python src/data/cache/pipeline/step02_calculate_timeframes.py --tickers AAPL,MSFT
    python src/data/cache/pipeline/step02_calculate_timeframes.py --timeframes 5m,15m,1h
    python src/data/cache/pipeline/step02_calculate_timeframes.py --force-refresh
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import argparse
import json
import time
from datetime import datetime, timedelta
from typing import List, Set, Optional, Dict, Any
import pandas as pd

from src.notification.logger import setup_logger

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"

_logger = setup_logger(__name__)


class TimeframeCalculator:
    """
    Pipeline Step 2: Calculate higher timeframes from 1-minute data with intelligence.

    This class handles:
    - Loading 1-minute data from step 1
    - Calculating 5m, 15m, 1h, 4h, 1d timeframes
    - Intelligent handling of missing data
    - Yearly storage with metadata
    - Incremental processing
    """

    # Trading session: 4:00 AM to 8:00 PM ET (16 hours)
    TRADING_START_HOUR = 4
    TRADING_END_HOUR = 20

    # Timeframe definitions (in minutes)
    TIMEFRAMES = {
        '5m': 5,
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 960  # 16 hours * 60 minutes
    }

    def __init__(self, cache_dir: str = None):
        """
        Initialize the timeframe calculator.

        Args:
            cache_dir: Cache directory path (defaults to DATA_CACHE_DIR)
        """
        self.cache_dir = Path(cache_dir or DATA_CACHE_DIR)
        self.ohlcv_dir = self.cache_dir / "ohlcv"

        # Pipeline statistics
        self.stats = {
            'total_tickers': 0,
            'successful_tickers': [],
            'failed_tickers': [],
            'skipped_tickers': [],
            'timeframes_processed': {},
            'total_bars_calculated': 0,
            'processing_time': 0,
            'errors': {}
        }

    def discover_tickers_with_1m_data(self) -> Set[str]:
        """
        Discover tickers that have 1-minute data available.

        Returns:
            Set of ticker symbols with 1m data
        """
        tickers = set()

        if not self.ohlcv_dir.exists():
            _logger.warning("OHLCV cache directory does not exist: %s", self.ohlcv_dir)
            return tickers

        for ticker_dir in self.ohlcv_dir.iterdir():
            if ticker_dir.is_dir() and not ticker_dir.name.startswith('_'):
                ticker = ticker_dir.name.upper()

                # Check if 1m data exists
                csv_gz_file = ticker_dir / f"{ticker}-1m.csv.gz"
                csv_file = ticker_dir / f"{ticker}-1m.csv"

                if csv_gz_file.exists() or csv_file.exists():
                    tickers.add(ticker)

        _logger.info("Discovered %d tickers with 1m data: %s", len(tickers), sorted(tickers))
        return tickers

    def load_1m_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load 1-minute data for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            DataFrame with 1m data or None if not available
        """
        ticker_dir = self.ohlcv_dir / ticker
        csv_gz_file = ticker_dir / f"{ticker}-1m.csv.gz"
        csv_file = ticker_dir / f"{ticker}-1m.csv"

        try:
            if csv_gz_file.exists():
                df = pd.read_csv(csv_gz_file, compression='gzip')
                _logger.debug("Loaded 1m data for %s from gzipped file (%d rows)", ticker, len(df))
            elif csv_file.exists():
                df = pd.read_csv(csv_file)
                _logger.debug("Loaded 1m data for %s from CSV file (%d rows)", ticker, len(df))
            else:
                _logger.warning("No 1m data found for %s", ticker)
                return None

            # Ensure proper datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Filter to trading hours (4 AM to 8 PM ET)
            df = self.filter_trading_hours(df)

            return df

        except Exception:
            _logger.exception("Error loading 1m data for %s:", ticker)
            return None

    def filter_trading_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to trading hours (4 AM to 8 PM ET).

        Args:
            df: DataFrame with datetime index

        Returns:
            Filtered DataFrame
        """
        # Filter to trading hours
        mask = (df.index.hour >= self.TRADING_START_HOUR) & (df.index.hour < self.TRADING_END_HOUR)
        return df[mask]

    def calculate_timeframe_bars(self, df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Calculate higher timeframe bars from 1-minute data with intelligence.

        Args:
            df_1m: 1-minute DataFrame
            timeframe: Target timeframe (5m, 15m, 1h, 4h, 1d)

        Returns:
            DataFrame with calculated timeframe bars
        """
        if timeframe not in self.TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        minutes = self.TIMEFRAMES[timeframe]

        if timeframe == '1d':
            # Daily bars: group by date, 4 AM to 8 PM
            return self._calculate_daily_bars(df_1m)
        else:
            # Intraday bars: resample with intelligence
            return self._calculate_intraday_bars(df_1m, minutes, timeframe)

    def _calculate_daily_bars(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily bars (4 AM to 8 PM ET).

        Args:
            df_1m: 1-minute DataFrame

        Returns:
            DataFrame with daily bars
        """
        if df_1m.empty:
            return pd.DataFrame()

        # Group by date (considering 4 AM start)
        # Shift timestamps by 4 hours so that 4 AM becomes midnight for grouping
        df_shifted = df_1m.copy()
        df_shifted.index = df_shifted.index - timedelta(hours=4)

        # Group by date
        daily_groups = df_shifted.groupby(df_shifted.index.date)

        daily_bars = []

        for date, group in daily_groups:
            if len(group) == 0:
                continue

            # Calculate OHLCV for the day
            bar = {
                'timestamp': pd.Timestamp(date) + timedelta(hours=4),  # 4 AM of the trading day
                'open': group['open'].iloc[0],
                'high': group['high'].max(),
                'low': group['low'].min(),
                'close': group['close'].iloc[-1],
                'volume': group['volume'].sum(),
                'bar_count': len(group)  # Number of 1m bars used
            }

            daily_bars.append(bar)

        if not daily_bars:
            return pd.DataFrame()

        df_daily = pd.DataFrame(daily_bars)
        df_daily.set_index('timestamp', inplace=True)
        df_daily.sort_index(inplace=True)

        _logger.debug("Calculated %d daily bars from %d 1m bars", len(df_daily), len(df_1m))
        return df_daily

    def _calculate_intraday_bars(self, df_1m: pd.DataFrame, minutes: int, timeframe: str) -> pd.DataFrame:
        """
        Calculate intraday bars with intelligence.

        Args:
            df_1m: 1-minute DataFrame
            minutes: Timeframe in minutes
            timeframe: Timeframe name for logging

        Returns:
            DataFrame with calculated bars
        """
        if df_1m.empty:
            return pd.DataFrame()

        # Resample to the target timeframe
        freq = f"{minutes}min"  # min = minutes in pandas (T is deprecated)

        # Use custom aggregation with intelligence
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Resample and aggregate
        resampled = df_1m.resample(freq, label='left', closed='left').agg(agg_dict)

        # Add bar count for intelligence
        bar_counts = df_1m.resample(freq, label='left', closed='left').size()
        resampled['bar_count'] = bar_counts

        # Remove bars with no data (NaN values)
        resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])

        # Apply intelligence: only keep bars with at least 1 underlying bar
        # (This preserves gaps as they exist in 1m data)
        resampled = resampled[resampled['bar_count'] > 0]

        # Remove the bar_count column from final output (keep for logging)
        expected_bars = minutes
        actual_bars = resampled['bar_count'].mean() if len(resampled) > 0 else 0

        _logger.debug("Calculated %d %s bars from %d 1m bars (avg %.1f bars per %s, expected %d)",
                     len(resampled), timeframe, len(df_1m), actual_bars, timeframe, expected_bars)

        # Drop bar_count for final output
        resampled = resampled.drop(columns=['bar_count'])

        return resampled

    def get_existing_timeframe_data(self, ticker: str, timeframe: str) -> Dict[int, pd.DataFrame]:
        """
        Load existing timeframe data by year.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe (5m, 15m, etc.)

        Returns:
            Dictionary mapping year to DataFrame
        """
        ticker_dir = self.ohlcv_dir / ticker / timeframe
        existing_data = {}

        if not ticker_dir.exists():
            return existing_data

        # Load all yearly files
        for year_file in ticker_dir.glob("*.csv.gz"):
            try:
                year = int(year_file.stem)  # Extract year from filename
                df = pd.read_csv(year_file, compression='gzip')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                existing_data[year] = df
                _logger.debug("Loaded existing %s data for %s year %d (%d bars)",
                             timeframe, ticker, year, len(df))
            except Exception as e:
                _logger.warning("Error loading %s data for %s year %s: %s",
                               timeframe, ticker, year_file.stem, e)

        return existing_data

    def save_timeframe_data(self, ticker: str, timeframe: str, df: pd.DataFrame) -> bool:
        """
        Save timeframe data by year with metadata.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            df: DataFrame with timeframe data

        Returns:
            True if successful
        """
        if df.empty:
            _logger.debug("No data to save for %s %s", ticker, timeframe)
            return True

        try:
            ticker_dir = self.ohlcv_dir / ticker / timeframe
            ticker_dir.mkdir(parents=True, exist_ok=True)

            # Group data by year
            yearly_data = {}
            for year, year_group in df.groupby(df.index.year):
                yearly_data[year] = year_group

            # Save each year separately
            for year, year_df in yearly_data.items():
                year_file = ticker_dir / f"{year}.csv.gz"

                # Reset index to save timestamp as column
                year_df_save = year_df.reset_index()
                year_df_save.to_csv(year_file, index=False, compression='gzip')

                _logger.debug("Saved %s %s data for year %d (%d bars) to %s",
                             ticker, timeframe, year, len(year_df), year_file)

            # Save metadata
            metadata = {
                'ticker': ticker,
                'timeframe': timeframe,
                'last_updated': datetime.now().isoformat(),
                'years_available': sorted(yearly_data.keys()),
                'total_bars': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if len(df) > 0 else None,
                    'end': df.index.max().isoformat() if len(df) > 0 else None
                },
                'trading_hours': {
                    'start_hour': self.TRADING_START_HOUR,
                    'end_hour': self.TRADING_END_HOUR
                }
            }

            metadata_file = ticker_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            _logger.info("Saved %s %s data: %d bars across %d years",
                        ticker, timeframe, len(df), len(yearly_data))
            return True

        except Exception:
            _logger.exception("Error saving %s %s data:", ticker, timeframe)
            return False

    def needs_recalculation(self, ticker: str, timeframe: str, df_1m: pd.DataFrame) -> bool:
        """
        Determine if timeframe data needs recalculation.

        Args:
            ticker: Ticker symbol
            timeframe: Timeframe
            df_1m: Current 1m data

        Returns:
            True if recalculation is needed
        """
        if df_1m.empty:
            return False

        # Check if metadata exists
        metadata_file = self.ohlcv_dir / ticker / timeframe / "metadata.json"
        if not metadata_file.exists():
            _logger.debug("No metadata for %s %s, recalculation needed", ticker, timeframe)
            return True

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if 1m data is newer than last calculation
            last_updated = pd.to_datetime(metadata['last_updated'])
            latest_1m_data = df_1m.index.max()

            if latest_1m_data > last_updated:
                _logger.debug("1m data newer than %s %s calculation, recalculation needed",
                             ticker, timeframe)
                return True

            # Check if date range has changed significantly
            current_start = df_1m.index.min()
            current_end = df_1m.index.max()

            if metadata['date_range']['start']:
                existing_start = pd.to_datetime(metadata['date_range']['start'])
                existing_end = pd.to_datetime(metadata['date_range']['end'])

                # If new data extends significantly beyond existing range
                if (current_start < existing_start - timedelta(days=1) or
                    current_end > existing_end + timedelta(days=1)):
                    _logger.debug("Date range changed for %s %s, recalculation needed",
                                 ticker, timeframe)
                    return True

            _logger.debug("No recalculation needed for %s %s", ticker, timeframe)
            return False

        except Exception as e:
            _logger.warning("Error checking metadata for %s %s: %s", ticker, timeframe, e)
            return True  # Recalculate if we can't determine

    def process_ticker(self, ticker: str, timeframes: List[str], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Process a single ticker for all timeframes.

        Args:
            ticker: Ticker symbol
            timeframes: List of timeframes to calculate
            force_refresh: If True, recalculate all data

        Returns:
            Dictionary with processing results
        """
        result = {
            'ticker': ticker,
            'success': False,
            'timeframes_processed': [],
            'timeframes_skipped': [],
            'total_bars_calculated': 0,
            'error': None,
            'processing_time': 0
        }

        start_time = time.time()

        try:
            # Load 1m data
            df_1m = self.load_1m_data(ticker)
            if df_1m is None or df_1m.empty:
                result['error'] = "No 1m data available"
                return result

            _logger.info("Processing %s with %d 1m bars from %s to %s",
                        ticker, len(df_1m), df_1m.index.min().date(), df_1m.index.max().date())

            # Process each timeframe
            for timeframe in timeframes:
                try:
                    # Check if recalculation is needed
                    if not force_refresh and not self.needs_recalculation(ticker, timeframe, df_1m):
                        result['timeframes_skipped'].append(timeframe)
                        _logger.debug("Skipping %s %s (up to date)", ticker, timeframe)
                        continue

                    # Calculate timeframe bars
                    df_timeframe = self.calculate_timeframe_bars(df_1m, timeframe)

                    if df_timeframe.empty:
                        _logger.warning("No %s bars calculated for %s", timeframe, ticker)
                        continue

                    # Save timeframe data
                    if self.save_timeframe_data(ticker, timeframe, df_timeframe):
                        result['timeframes_processed'].append(timeframe)
                        result['total_bars_calculated'] += len(df_timeframe)
                        self.stats['total_bars_calculated'] += len(df_timeframe)

                        # Update timeframe statistics
                        if timeframe not in self.stats['timeframes_processed']:
                            self.stats['timeframes_processed'][timeframe] = 0
                        self.stats['timeframes_processed'][timeframe] += len(df_timeframe)

                        _logger.info("✅ %s %s: %d bars calculated", ticker, timeframe, len(df_timeframe))
                    else:
                        _logger.error("Failed to save %s %s data", ticker, timeframe)

                except Exception:
                    _logger.exception("Error processing %s %s:", ticker, timeframe)

            # Success if we processed timeframes OR if everything was up to date
            success = (len(result['timeframes_processed']) > 0 or
                      len(result['timeframes_skipped']) > 0)

            result.update({
                'success': success,
                'processing_time': time.time() - start_time
            })

        except Exception as e:
            result.update({
                'error': str(e),
                'processing_time': time.time() - start_time
            })

        return result

    def calculate_all_timeframes(self, tickers: List[str], timeframes: List[str],
                               force_refresh: bool = False) -> Dict[str, Any]:
        """
        Calculate timeframes for all tickers with comprehensive pipeline statistics.

        Args:
            tickers: List of ticker symbols
            timeframes: List of timeframes to calculate
            force_refresh: If True, recalculate all data

        Returns:
            Dictionary with comprehensive pipeline results
        """
        pipeline_start_time = time.time()
        self.stats['total_tickers'] = len(tickers)

        _logger.info("=" * 60)
        _logger.info("PIPELINE STEP 2: CALCULATE TIMEFRAMES")
        _logger.info("=" * 60)
        _logger.info("Tickers: %d symbols", len(tickers))
        _logger.info("Timeframes: %s", ', '.join(timeframes))
        _logger.info("Force refresh: %s", force_refresh)
        _logger.info("Trading hours: %02d:00 - %02d:00 ET", self.TRADING_START_HOUR, self.TRADING_END_HOUR)
        _logger.info("=" * 60)

        results = {}

        for i, ticker in enumerate(tickers, 1):
            _logger.info("Processing ticker %d/%d: %s", i, len(tickers), ticker)

            result = self.process_ticker(ticker, timeframes, force_refresh)
            results[ticker] = result

            # Update statistics
            if result['success']:
                self.stats['successful_tickers'].append(ticker)
                if result['timeframes_processed']:
                    _logger.info("✅ %s: %s processed", ticker, ', '.join(result['timeframes_processed']))
                if result['timeframes_skipped']:
                    _logger.info("⏭️ %s: %s skipped (up to date)", ticker, ', '.join(result['timeframes_skipped']))
                if not result['timeframes_processed'] and not result['timeframes_skipped']:
                    _logger.info("✅ %s: No processing needed", ticker)
            else:
                self.stats['failed_tickers'].append(ticker)
                error = result.get('error', 'Unknown error')
                self.stats['errors'][ticker] = error
                _logger.error("❌ %s: %s", ticker, error)

        # Calculate final statistics
        self.stats['processing_time'] = time.time() - pipeline_start_time

        # Print comprehensive pipeline summary
        self.print_pipeline_summary(timeframes)

        return {
            'results': results,
            'statistics': self.stats
        }

    def print_pipeline_summary(self, timeframes: List[str]):
        """Print comprehensive pipeline summary with statistics."""
        _logger.info("=" * 80)
        _logger.info("PIPELINE STEP 2 SUMMARY")
        _logger.info("=" * 80)

        # Overall statistics
        total = self.stats['total_tickers']
        successful = len(self.stats['successful_tickers'])
        failed = len(self.stats['failed_tickers'])

        _logger.info("📊 PROCESSING STATISTICS:")
        _logger.info("   Total tickers processed: %d", total)
        _logger.info("   ✅ Successfully processed: %d", successful)
        _logger.info("   ❌ Failed: %d", failed)
        _logger.info("   📈 Success rate: %.1f%%", successful / total * 100 if total > 0 else 0)

        _logger.info("\n📈 CALCULATION STATISTICS:")
        _logger.info("   Total bars calculated: %d", self.stats['total_bars_calculated'])
        _logger.info("   Processing time: %.1f seconds", self.stats['processing_time'])

        # Timeframe statistics
        if self.stats['timeframes_processed']:
            _logger.info("\n📊 TIMEFRAME STATISTICS:")
            for timeframe in timeframes:
                count = self.stats['timeframes_processed'].get(timeframe, 0)
                _logger.info("   %s: %d bars calculated", timeframe, count)

        # Successful tickers
        if self.stats['successful_tickers']:
            _logger.info("\n✅ SUCCESSFULLY PROCESSED TICKERS (%d):", len(self.stats['successful_tickers']))
            for ticker in sorted(self.stats['successful_tickers']):
                _logger.info("   %s", ticker)

        # Failed tickers with reasons
        if self.stats['failed_tickers']:
            _logger.info("\n❌ FAILED TICKERS (%d):", len(self.stats['failed_tickers']))
            for ticker in sorted(self.stats['failed_tickers']):
                error = self.stats['errors'].get(ticker, 'Unknown error')
                _logger.info("   %s: %s", ticker, error)

        _logger.info("=" * 80)
        _logger.info("PIPELINE STEP 2 COMPLETED")
        _logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Pipeline Step 2: Calculate higher timeframes from 1-minute data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate all timeframes for all tickers with 1m data
  python src/data/cache/pipeline/step02_calculate_timeframes.py

  # Calculate specific timeframes
  python src/data/cache/pipeline/step02_calculate_timeframes.py --timeframes 5m,15m,1h

  # Process specific tickers
  python src/data/cache/pipeline/step02_calculate_timeframes.py --tickers AAPL,MSFT,GOOGL

  # Force refresh all calculations
  python src/data/cache/pipeline/step02_calculate_timeframes.py --force-refresh

Timeframes Available:
  5m   - 5-minute bars
  15m  - 15-minute bars
  1h   - 1-hour bars
  4h   - 4-hour bars
  1d   - Daily bars (4 AM to 8 PM ET)

Output Structure:
  DATA_CACHE_DIR/ohlcv/TICKER/TIMEFRAME/YYYY.csv.gz
        """
    )

    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to process (default: discover from 1m data)"
    )

    parser.add_argument(
        "--timeframes",
        type=str,
        default="5m,15m,1h,4h,1d",
        help="Comma-separated list of timeframes to calculate (default: 5m,15m,1h,4h,1d)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DATA_CACHE_DIR,
        help=f"Cache directory path (default: {DATA_CACHE_DIR})"
    )

    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh: recalculate all timeframe data"
    )

    args = parser.parse_args()

    try:
        # Initialize calculator
        calculator = TimeframeCalculator(args.cache_dir)

        # Parse timeframes
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]
        invalid_timeframes = [tf for tf in timeframes if tf not in calculator.TIMEFRAMES]
        if invalid_timeframes:
            _logger.error("Invalid timeframes: %s. Valid: %s",
                         invalid_timeframes, list(calculator.TIMEFRAMES.keys()))
            sys.exit(1)

        # Get tickers
        if args.tickers:
            tickers = [t.strip().upper() for t in args.tickers.split(",")]
            _logger.info("Using specified tickers: %s", tickers)
        else:
            discovered_tickers = calculator.discover_tickers_with_1m_data()
            if not discovered_tickers:
                _logger.error("No tickers with 1m data found")
                sys.exit(1)
            tickers = sorted(discovered_tickers)

        # Run pipeline step 2
        pipeline_results = calculator.calculate_all_timeframes(
            tickers, timeframes, args.force_refresh
        )

        # Exit with appropriate code based on results
        stats = pipeline_results['statistics']
        if len(stats['failed_tickers']) == 0:
            _logger.info("Pipeline Step 2 completed successfully - all tickers processed")
            sys.exit(0)
        elif len(stats['successful_tickers']) > 0:
            _logger.warning("Pipeline Step 2 completed with some failures")
            sys.exit(0)  # Continue pipeline even with some failures
        else:
            _logger.error("Pipeline Step 2 failed - no tickers processed successfully")
            sys.exit(1)

    except KeyboardInterrupt:
        _logger.info("Pipeline Step 2 cancelled by user")
        sys.exit(1)
    except Exception:
        _logger.exception("Pipeline Step 2 fatal error:")
        sys.exit(1)


if __name__ == "__main__":
    main()