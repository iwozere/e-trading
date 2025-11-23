#!/usr/bin/env python3
"""
Sample Data Generator for EMPS Backtesting

Generates synthetic or downloads real historical data for backtesting EMPS.

Usage:
    # Generate synthetic explosive move data
    python generate_sample_data.py --mode synthetic --ticker GME --output data/GME_5m_synthetic.csv

    # Download real historical data (requires FMP API)
    python generate_sample_data.py --mode fmp --ticker GME --start 2021-01-25 --end 2021-01-29 --output data/GME_5m.csv
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger

logger = setup_logger(__name__)


def generate_synthetic_explosive_move(
    ticker: str,
    start_date: str,
    num_days: int = 5,
    interval_minutes: int = 5
) -> pd.DataFrame:
    """
    Generate synthetic intraday data simulating an explosive move.

    Args:
        ticker: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        num_days: Number of trading days
        interval_minutes: Bar interval in minutes

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Generating synthetic explosive move data for {ticker}")

    start = datetime.strptime(start_date, '%Y-%m-%d')
    bars = []

    # Trading hours: 9:30 AM - 4:00 PM ET (6.5 hours = 390 minutes)
    bars_per_day = 390 // interval_minutes

    base_price = 20.0
    base_volume = 1_000_000

    for day in range(num_days):
        current_date = start + timedelta(days=day)

        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        logger.info(f"  Generating day {day+1}: {current_date.strftime('%Y-%m-%d')}")

        # Price trajectory for the day (explosive move builds up)
        if day == 0:
            # Day 1: Normal trading
            day_multiplier = 1.0
            vol_multiplier = 1.0
        elif day == 1:
            # Day 2: Starting to heat up
            day_multiplier = 1.5
            vol_multiplier = 2.0
        elif day == 2:
            # Day 3: Explosive move begins
            day_multiplier = 3.0
            vol_multiplier = 10.0
        elif day == 3:
            # Day 4: Peak explosion
            day_multiplier = 8.0
            vol_multiplier = 30.0
        else:
            # Day 5: Comedown
            day_multiplier = 4.0
            vol_multiplier = 15.0

        day_start_price = base_price * day_multiplier

        for bar in range(bars_per_day):
            # Calculate timestamp
            market_open = current_date.replace(hour=9, minute=30, second=0)
            bar_time = market_open + timedelta(minutes=bar * interval_minutes)

            # Intraday price movement (add volatility)
            intraday_trend = np.sin(bar / bars_per_day * np.pi) * 0.05  # Sine wave
            noise = np.random.normal(0, 0.02)  # Random noise

            price = day_start_price * (1 + intraday_trend + noise)

            # OHLC
            bar_volatility = price * 0.01 * (1 + np.random.random())
            open_price = price + np.random.uniform(-bar_volatility, bar_volatility)
            high_price = price + abs(np.random.uniform(0, bar_volatility * 2))
            low_price = price - abs(np.random.uniform(0, bar_volatility * 2))
            close_price = price + np.random.uniform(-bar_volatility, bar_volatility)

            # Ensure OHLC relationship
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)

            # Volume (higher during explosive move)
            volume = int(base_volume * vol_multiplier * (1 + np.random.random()))

            bars.append({
                'timestamp': bar_time.strftime('%Y-%m-%d %H:%M:%S'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })

    df = pd.DataFrame(bars)
    logger.info(f"Generated {len(df)} bars across {num_days} days")

    return df


def download_fmp_historical(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = '5min'
) -> pd.DataFrame:
    """
    Download real historical data from FMP.

    Args:
        ticker: Stock ticker
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Bar interval (5min, 15min, 30min, 1hour)

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Downloading FMP historical data for {ticker}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Interval: {interval}")

    try:
        from src.data.downloader.fmp_data_downloader import FMPDataDownloader

        fmp = FMPDataDownloader()

        if not fmp.test_connection():
            logger.error("FMP connection failed")
            return pd.DataFrame()

        # Note: FMP historical intraday endpoint may have limitations
        # This is a placeholder - actual implementation depends on FMP API

        logger.warning("FMP historical intraday download not yet implemented")
        logger.warning("Use synthetic mode or manually download data")

        return pd.DataFrame()

    except Exception as e:
        logger.exception(f"Error downloading from FMP: {e}")
        return pd.DataFrame()


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Generate sample data for EMPS backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['synthetic', 'fmp'],
        required=True,
        help='Data generation mode: synthetic or fmp download'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2021-01-25',
        help='Start date (YYYY-MM-DD) - default: 2021-01-25'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD) - only for FMP mode'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=5,
        help='Number of days - only for synthetic mode (default: 5)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        choices=['5', '15', '30', '60'],
        default='5',
        help='Bar interval in minutes (default: 5)'
    )

    args = parser.parse_args()

    # Generate or download data
    if args.mode == 'synthetic':
        df = generate_synthetic_explosive_move(
            ticker=args.ticker,
            start_date=args.start,
            num_days=args.days,
            interval_minutes=int(args.interval)
        )
    elif args.mode == 'fmp':
        if not args.end:
            logger.error("--end date required for FMP mode")
            return 1

        df = download_fmp_historical(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            interval=f"{args.interval}min"
        )
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

    if df.empty:
        logger.error("No data generated/downloaded")
        return 1

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")

    # Print sample
    logger.info("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    logger.info(f"\nLast 5 rows:")
    print(df.tail().to_string(index=False))

    logger.info(f"\nTotal rows: {len(df)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
