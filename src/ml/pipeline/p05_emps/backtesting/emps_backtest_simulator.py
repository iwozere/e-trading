#!/usr/bin/env python3
"""
EMPS Backtesting Simulator

Simulates scheduled EMPS pipeline runs on historical data.
Reads CSV files with 5m or 15m bars and simulates daily scans at predefined times.

Usage:
    # Simulate market open scan (09:30 ET / 14:30 UTC)
    python emps_backtest_simulator.py --data-file GME_5m.csv --scan-time "14:30" --ticker GME

    # Simulate market close scan (16:00 ET / 21:00 UTC)
    python emps_backtest_simulator.py --data-file AMC_15m.csv --scan-time "21:00" --ticker AMC

    # Simulate both scans
    python emps_backtest_simulator.py --data-file BBBY_5m.csv --scan-time "14:30,21:00" --ticker BBBY

Features:
- Reads historical intraday data from CSV
- Simulates daily scheduled pipeline runs
- Logs EMPS signals at each scan time
- Generates summary report with detection statistics
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import pytz

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p05_emps.emps import compute_emps_from_intraday, DEFAULTS

logger = setup_logger(__name__)


@dataclass
class ScanResult:
    """Result from a single EMPS scan."""
    scan_datetime: datetime
    scan_time_utc: str
    emps_score: float
    explosion_flag: bool
    hard_flag: bool
    vol_zscore: float
    vwap_dev: float
    rv_ratio: float
    liquidity_score: float
    close_price: float
    volume: int
    bars_analyzed: int


@dataclass
class BacktestSummary:
    """Summary statistics from backtesting."""
    ticker: str
    data_file: str
    total_days: int
    total_scans: int
    explosion_flags: int
    hard_flags: int
    max_emps_score: float
    avg_emps_score: float
    first_explosion_date: Optional[datetime]
    days_with_explosions: int
    detection_rate: float


class EMPSBacktestSimulator:
    """
    Simulates scheduled EMPS pipeline runs on historical data.

    This simulator reads historical intraday data and runs EMPS calculations
    at predefined times (e.g., market open, market close) to simulate how
    the pipeline would have performed in real-time.
    """

    def __init__(
        self,
        data_file: str,
        ticker: str,
        scan_times_utc: List[str],
        market_cap: Optional[float] = None,
        float_shares: Optional[float] = None,
        avg_volume: Optional[float] = None,
        emps_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize backtest simulator.

        Args:
            data_file: Path to CSV file with historical intraday data
            ticker: Stock ticker symbol
            scan_times_utc: List of scan times in UTC (e.g., ["14:30", "21:00"])
            market_cap: Market capitalization
            float_shares: Float shares
            avg_volume: Average daily volume
            emps_params: Optional EMPS parameters
        """
        self.data_file = Path(data_file)
        self.ticker = ticker.upper()
        self.scan_times_utc = self._parse_scan_times(scan_times_utc)
        self.market_cap = market_cap
        self.float_shares = float_shares
        self.avg_volume = avg_volume
        self.emps_params = emps_params or DEFAULTS

        self.df_data: Optional[pd.DataFrame] = None
        self.scan_results: List[ScanResult] = []

        logger.info(f"Initialized EMPS Backtest Simulator for {self.ticker}")
        logger.info(f"Data file: {self.data_file}")
        logger.info(f"Scan times (UTC): {[t.strftime('%H:%M') for t in self.scan_times_utc]}")

    def _parse_scan_times(self, scan_times: List[str]) -> List[time]:
        """Parse scan time strings to time objects."""
        times = []
        for t in scan_times:
            try:
                hour, minute = map(int, t.split(':'))
                times.append(time(hour, minute))
            except Exception as e:
                logger.error(f"Invalid scan time format: {t}. Use HH:MM format.")
                raise ValueError(f"Invalid scan time: {t}") from e
        return times

    def load_data(self) -> bool:
        """
        Load historical intraday data from CSV.

        Expected CSV format:
            timestamp,open,high,low,close,volume
            2021-01-25 09:30:00,36.5,37.2,36.0,37.0,1500000
            ...

        Returns:
            True if loaded successfully
        """
        try:
            logger.info(f"Loading data from {self.data_file}...")

            if not self.data_file.exists():
                logger.error(f"Data file not found: {self.data_file}")
                return False

            # Read CSV
            df = pd.read_csv(self.data_file)

            # Validate columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.error(f"Available columns: {df.columns.tolist()}")
                return False

            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Ensure sorted by time
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Rename columns to match EMPS expectations
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            self.df_data = df

            logger.info(f"Loaded {len(df)} bars")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            logger.info(f"Interval detected: {self._detect_interval(df)}")

            return True

        except Exception as e:
            logger.exception(f"Error loading data: {e}")
            return False

    def _detect_interval(self, df: pd.DataFrame) -> str:
        """Detect bar interval from timestamps."""
        if len(df) < 2:
            return "unknown"

        time_diffs = df['timestamp'].diff().dropna()
        median_diff = time_diffs.median()

        if median_diff <= timedelta(minutes=5):
            return "5m"
        elif median_diff <= timedelta(minutes=15):
            return "15m"
        elif median_diff <= timedelta(minutes=30):
            return "30m"
        elif median_diff <= timedelta(hours=1):
            return "1h"
        else:
            return f"{median_diff}"

    def run_backtest(self) -> BacktestSummary:
        """
        Run backtest simulation.

        Simulates daily scans at predefined times and logs EMPS signals.

        Returns:
            BacktestSummary with results
        """
        if self.df_data is None or self.df_data.empty:
            logger.error("No data loaded. Call load_data() first.")
            return None

        logger.info("="*70)
        logger.info("STARTING EMPS BACKTEST SIMULATION")
        logger.info("="*70)

        # Get unique trading days
        self.df_data['date'] = self.df_data['timestamp'].dt.date
        trading_days = sorted(self.df_data['date'].unique())

        logger.info(f"Simulating {len(trading_days)} trading days")
        logger.info(f"Scans per day: {len(self.scan_times_utc)}")
        logger.info(f"Total scans: {len(trading_days) * len(self.scan_times_utc)}")
        logger.info("")

        # Run scans for each day
        for day_idx, trading_day in enumerate(trading_days, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Day {day_idx}/{len(trading_days)}: {trading_day}")
            logger.info(f"{'='*70}")

            # Run scans at each scheduled time
            for scan_time in self.scan_times_utc:
                self._run_single_scan(trading_day, scan_time)

        # Generate summary
        summary = self._generate_summary(trading_days)

        # Print summary
        self._print_summary(summary)

        return summary

    def _run_single_scan(self, trading_day, scan_time_utc: time):
        """
        Run EMPS scan at a specific time on a specific day.

        Args:
            trading_day: Date to scan
            scan_time_utc: Time (UTC) to run scan
        """
        # Create scan datetime
        scan_datetime = datetime.combine(trading_day, scan_time_utc)
        scan_datetime = pytz.UTC.localize(scan_datetime)

        logger.info(f"\n[SCAN] {scan_datetime.strftime('%Y-%m-%d %H:%M UTC')}")

        # Get data up to scan time
        df_up_to_scan = self.df_data[
            self.df_data['timestamp'] <= scan_datetime
        ].copy()

        if df_up_to_scan.empty:
            logger.warning("  [SKIP] No data available at scan time")
            return

        # Determine lookback window (need enough bars for EMPS calculation)
        max_lookback = max(
            self.emps_params.get('vol_lookback', 20),
            self.emps_params.get('vwap_lookback', 20),
            self.emps_params.get('rv_long_window', 40)
        )

        # Get recent bars (use last 200 bars or all available)
        lookback_bars = min(200, len(df_up_to_scan))
        df_recent = df_up_to_scan.tail(lookback_bars).copy()

        if len(df_recent) < max_lookback:
            logger.warning(f"  [SKIP] Insufficient bars ({len(df_recent)} < {max_lookback})")
            return

        logger.info(f"  Analyzing {len(df_recent)} bars")

        try:
            # Compute EMPS
            df_emps = compute_emps_from_intraday(
                df_recent,
                market_cap=self.market_cap,
                float_shares=self.float_shares,
                avg_volume=self.avg_volume,
                ticker=self.ticker,
                params=self.emps_params
            )

            if df_emps.empty:
                logger.warning("  [ERROR] EMPS calculation returned empty DataFrame")
                return

            # Get latest (current) EMPS values
            latest = df_emps.iloc[-1]

            # Create scan result
            result = ScanResult(
                scan_datetime=scan_datetime,
                scan_time_utc=scan_time_utc.strftime('%H:%M'),
                emps_score=float(latest['emps_score']),
                explosion_flag=bool(latest['explosion_flag']),
                hard_flag=bool(latest['hard_flag']),
                vol_zscore=float(latest['vol_zscore']),
                vwap_dev=float(latest['vwap_dev']),
                rv_ratio=float(latest['rv_ratio']),
                liquidity_score=float(latest['liquidity_score']),
                close_price=float(latest['Close']),
                volume=int(latest['Volume']),
                bars_analyzed=len(df_recent)
            )

            self.scan_results.append(result)

            # Log result
            self._log_scan_result(result)

        except Exception as e:
            logger.exception(f"  [ERROR] EMPS calculation failed: {e}")

    def _log_scan_result(self, result: ScanResult):
        """Log scan result in readable format."""

        # Determine signal strength
        if result.explosion_flag:
            if result.hard_flag:
                signal = "[!!!HARD EXPLOSION!!!]"
            else:
                signal = "[!EXPLOSION!]"
        elif result.emps_score >= 0.5:
            signal = "[ELEVATED]"
        else:
            signal = "[NORMAL]"

        logger.info(f"  {signal}")
        logger.info(f"    EMPS Score: {result.emps_score:.3f}")
        logger.info(f"    Explosion Flag: {result.explosion_flag}")
        logger.info(f"    Hard Flag: {result.hard_flag}")
        logger.info(f"    Components:")
        logger.info(f"      Vol Z-Score: {result.vol_zscore:.2f}")
        logger.info(f"      VWAP Dev: {result.vwap_dev:.3f} ({result.vwap_dev*100:.1f}%)")
        logger.info(f"      RV Ratio: {result.rv_ratio:.2f}")
        logger.info(f"      Liquidity: {result.liquidity_score:.3f}")
        logger.info(f"    Price: ${result.close_price:.2f}")
        logger.info(f"    Volume: {result.volume:,}")

    def _generate_summary(self, trading_days: List) -> BacktestSummary:
        """Generate summary statistics from backtest."""

        if not self.scan_results:
            return BacktestSummary(
                ticker=self.ticker,
                data_file=str(self.data_file),
                total_days=len(trading_days),
                total_scans=0,
                explosion_flags=0,
                hard_flags=0,
                max_emps_score=0.0,
                avg_emps_score=0.0,
                first_explosion_date=None,
                days_with_explosions=0,
                detection_rate=0.0
            )

        df_results = pd.DataFrame([vars(r) for r in self.scan_results])

        explosion_flags = int(df_results['explosion_flag'].sum())
        hard_flags = int(df_results['hard_flag'].sum())
        max_emps_score = float(df_results['emps_score'].max())
        avg_emps_score = float(df_results['emps_score'].mean())

        # Find first explosion
        explosions = df_results[df_results['explosion_flag'] == True]
        first_explosion_date = None
        if not explosions.empty:
            first_explosion_date = explosions.iloc[0]['scan_datetime']

        # Count days with explosions
        days_with_explosions = 0
        if not explosions.empty:
            explosions['date'] = pd.to_datetime(explosions['scan_datetime']).dt.date
            days_with_explosions = int(explosions['date'].nunique())

        # Detection rate
        detection_rate = days_with_explosions / len(trading_days) if trading_days else 0.0

        return BacktestSummary(
            ticker=self.ticker,
            data_file=str(self.data_file),
            total_days=len(trading_days),
            total_scans=len(self.scan_results),
            explosion_flags=explosion_flags,
            hard_flags=hard_flags,
            max_emps_score=max_emps_score,
            avg_emps_score=avg_emps_score,
            first_explosion_date=first_explosion_date,
            days_with_explosions=days_with_explosions,
            detection_rate=detection_rate
        )

    def _print_summary(self, summary: BacktestSummary):
        """Print backtest summary."""

        logger.info("\n" + "="*70)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*70)

        logger.info(f"\nTicker: {summary.ticker}")
        logger.info(f"Data File: {summary.data_file}")
        logger.info(f"Total Trading Days: {summary.total_days}")
        logger.info(f"Total Scans: {summary.total_scans}")

        logger.info(f"\nDetection Statistics:")
        logger.info(f"  Explosion Flags: {summary.explosion_flags}")
        logger.info(f"  Hard Flags: {summary.hard_flags}")
        logger.info(f"  Days with Explosions: {summary.days_with_explosions}")
        logger.info(f"  Detection Rate: {summary.detection_rate*100:.1f}% of days")

        logger.info(f"\nEMPS Scores:")
        logger.info(f"  Max EMPS Score: {summary.max_emps_score:.3f}")
        logger.info(f"  Avg EMPS Score: {summary.avg_emps_score:.3f}")

        if summary.first_explosion_date:
            logger.info(f"\nFirst Explosion Detected:")
            logger.info(f"  {summary.first_explosion_date.strftime('%Y-%m-%d %H:%M UTC')}")

        logger.info("\n" + "="*70)

    def export_results(self, output_file: str):
        """
        Export scan results to CSV.

        Args:
            output_file: Path to output CSV file
        """
        if not self.scan_results:
            logger.warning("No results to export")
            return

        df_results = pd.DataFrame([vars(r) for r in self.scan_results])

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_results.to_csv(output_path, index=False)
        logger.info(f"Results exported to: {output_path}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="EMPS Backtesting Simulator - Simulate scheduled pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Market open scan (09:30 ET = 14:30 UTC)
  python emps_backtest_simulator.py --data-file data/GME_5m.csv --scan-time "14:30" --ticker GME

  # Market close scan (16:00 ET = 21:00 UTC)
  python emps_backtest_simulator.py --data-file data/AMC_15m.csv --scan-time "21:00" --ticker AMC

  # Both scans
  python emps_backtest_simulator.py --data-file data/BBBY_5m.csv --scan-time "14:30,21:00" --ticker BBBY

  # With output file
  python emps_backtest_simulator.py --data-file data/GME_5m.csv --scan-time "14:30" --ticker GME --output results.csv

Note: Times are in UTC. Convert from ET: ET + 5 hours = UTC (EST) or ET + 4 hours = UTC (EDT)
        """
    )

    # Required arguments
    parser.add_argument(
        '--data-file',
        type=str,
        required=True,
        help='Path to CSV file with historical intraday data'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol'
    )
    parser.add_argument(
        '--scan-time',
        type=str,
        required=True,
        help='Scan time(s) in UTC (HH:MM format, comma-separated for multiple). '
             'Example: "14:30" or "14:30,21:00"'
    )

    # Optional arguments
    parser.add_argument(
        '--market-cap',
        type=float,
        help='Market capitalization (optional)'
    )
    parser.add_argument(
        '--float-shares',
        type=float,
        help='Float shares (optional)'
    )
    parser.add_argument(
        '--avg-volume',
        type=float,
        help='Average daily volume (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for results (optional)'
    )
    parser.add_argument(
        '--emps-threshold',
        type=float,
        default=0.6,
        help='EMPS explosion threshold (default: 0.6)'
    )

    args = parser.parse_args()

    # Parse scan times
    scan_times = [t.strip() for t in args.scan_time.split(',')]

    # Custom EMPS params
    emps_params = {
        **DEFAULTS,
        'combined_score_thresh': args.emps_threshold
    }

    # Initialize simulator
    simulator = EMPSBacktestSimulator(
        data_file=args.data_file,
        ticker=args.ticker,
        scan_times_utc=scan_times,
        market_cap=args.market_cap,
        float_shares=args.float_shares,
        avg_volume=args.avg_volume,
        emps_params=emps_params
    )

    # Load data
    if not simulator.load_data():
        logger.error("Failed to load data")
        return 1

    # Run backtest
    summary = simulator.run_backtest()

    if summary is None:
        logger.error("Backtest failed")
        return 1

    # Export results if requested
    if args.output:
        simulator.export_results(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
