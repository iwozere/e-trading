"""
Rolling Memory Module - 10-Day Accumulation Tracker

Scans historical daily results to identify:
- Phase 1: Quiet Accumulation (5+ appearances in 10 days)
- Phase 2: Early Public Signal (Phase 1 + volume/sentiment acceleration)
"""

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.notification.logger import setup_logger
from src.ml.pipeline.p06_emps2.config import RollingMemoryConfig

_logger = setup_logger(__name__)


class RollingMemoryScanner:
    """
    Scans historical EMPS2 results to detect accumulation patterns.

    Workflow:
    1. Scan last N days of results folders
    2. Aggregate tickers by appearance frequency
    3. Detect Phase 1 (persistent accumulation)
    4. Detect Phase 2 (acceleration signals)
    5. Generate watchlists and alerts
    """

    def __init__(
        self,
        config: RollingMemoryConfig,
        results_base_path: Path,
        verbose: bool = True
    ):
        """
        Initialize rolling memory scanner.

        Args:
            config: Rolling memory configuration
            results_base_path: Base path to results/emps2/ folder
            verbose: Enable verbose logging
        """
        self.config = config
        self.results_base_path = results_base_path
        self.verbose = verbose

    def scan_historical_results(
        self,
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Scan historical results from last N days.

        Args:
            lookback_days: Override config lookback period

        Returns:
            DataFrame with columns:
            - scan_date, ticker, market_cap, vol_zscore, vol_rv_ratio,
              atr_ratio, last_price, avg_volume, etc.
        """
        lookback = lookback_days or self.config.lookback_days

        # Calculate date range
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=lookback)

        _logger.info(
            "Scanning historical results from %s to %s (%d days)",
            start_date, today, lookback
        )

        all_results = []

        # Scan each day's folder
        for days_back in range(lookback + 1):
            scan_date = today - timedelta(days=days_back)
            date_str = scan_date.strftime('%Y-%m-%d')
            day_folder = self.results_base_path / date_str

            if not day_folder.exists():
                _logger.debug("No results for %s", date_str)
                continue

            # Load 04_volatility_filtered.csv (has all technical indicators)
            vol_file = day_folder / '04_volatility_filtered.csv'
            if vol_file.exists():
                try:
                    df = pd.read_csv(vol_file)
                    df['scan_date'] = scan_date
                    all_results.append(df)
                    _logger.debug("Loaded %d tickers from %s", len(df), date_str)
                except Exception:
                    _logger.exception("Error loading %s", vol_file)

        if not all_results:
            _logger.warning("No historical results found in last %d days", lookback)
            return pd.DataFrame()

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)
        _logger.info(
            "Loaded %d total records from %d days",
            len(combined_df), len(all_results)
        )

        return combined_df

    def calculate_appearance_frequency(
        self,
        historical_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate how many times each ticker appeared in lookback period.

        Args:
            historical_df: Combined historical data

        Returns:
            DataFrame with:
            - ticker, appearance_count, first_seen, last_seen,
              avg_vol_zscore, avg_vol_rv_ratio, latest_price, etc.
        """
        if historical_df.empty:
            return pd.DataFrame()

        # Group by ticker and aggregate
        agg_dict = {
            'scan_date': ['count', 'min', 'max'],
        }

        # Add metrics if they exist
        if 'vol_zscore' in historical_df.columns:
            agg_dict['vol_zscore'] = ['mean', 'max', 'last']
        if 'vol_rv_ratio' in historical_df.columns:
            agg_dict['vol_rv_ratio'] = ['mean', 'max', 'last']
        if 'atr_ratio' in historical_df.columns:
            agg_dict['atr_ratio'] = ['mean', 'last']
        if 'last_price' in historical_df.columns:
            agg_dict['last_price'] = 'last'
        if 'market_cap' in historical_df.columns:
            agg_dict['market_cap'] = 'last'
        if 'avg_volume' in historical_df.columns:
            agg_dict['avg_volume'] = 'last'

        freq_df = historical_df.groupby('ticker').agg(agg_dict).reset_index()

        # Flatten column names
        new_columns = ['ticker']
        for col in freq_df.columns[1:]:
            if isinstance(col, tuple):
                if col[1] == 'count':
                    new_columns.append('appearance_count')
                elif col[1] == 'min':
                    new_columns.append('first_seen')
                elif col[1] == 'max' and col[0] == 'scan_date':
                    new_columns.append('last_seen')
                elif col[1] == 'mean':
                    new_columns.append(f'avg_{col[0]}')
                elif col[1] == 'max':
                    new_columns.append(f'max_{col[0]}')
                elif col[1] == 'last':
                    new_columns.append(f'latest_{col[0]}')
                else:
                    new_columns.append(f'{col[0]}_{col[1]}')
            else:
                new_columns.append(col)

        freq_df.columns = new_columns

        # Sort by appearance count
        freq_df = freq_df.sort_values('appearance_count', ascending=False)

        _logger.info(
            "Calculated frequency for %d unique tickers", len(freq_df)
        )

        if len(freq_df) > 0:
            _logger.info(
                "Top ticker: %s with %d appearances",
                freq_df.iloc[0]['ticker'],
                freq_df.iloc[0]['appearance_count']
            )

        return freq_df

    def detect_phase1_candidates(
        self,
        frequency_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect Phase 1: Quiet Accumulation.

        Criteria:
        - Appeared 5+ times (configurable)
        - Persistent pattern over lookback period

        Args:
            frequency_df: Frequency analysis results

        Returns:
            Phase 1 watchlist DataFrame
        """
        if frequency_df.empty:
            return pd.DataFrame()

        phase1_df = frequency_df[
            frequency_df['appearance_count'] >= self.config.phase1_min_appearances
        ].copy()

        phase1_df['phase'] = 'Phase 1: Quiet Accumulation'

        _logger.info(
            "Detected %d Phase 1 candidates (%d+ appearances)",
            len(phase1_df), self.config.phase1_min_appearances
        )

        return phase1_df

    def detect_phase2_candidates(
        self,
        phase1_df: pd.DataFrame,
        current_scan_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect Phase 2: Early Public Signal.

        Criteria:
        - Already in Phase 1 watchlist
        - Volume Z-Score accelerating (>3.0)
        - Sentiment rising (if available)
        - Virality increasing

        Args:
            phase1_df: Phase 1 watchlist
            current_scan_df: Today's scan results (with sentiment if available)

        Returns:
            Phase 2 alerts DataFrame
        """
        if phase1_df.empty or current_scan_df.empty:
            _logger.info("No Phase 1 candidates or current scan data for Phase 2 detection")
            return pd.DataFrame()

        # Prepare current scan columns
        merge_cols = ['ticker']
        if 'vol_zscore' in current_scan_df.columns:
            merge_cols.append('vol_zscore')
        if 'sentiment_score' in current_scan_df.columns:
            merge_cols.append('sentiment_score')
        if 'mentions_24h' in current_scan_df.columns:
            merge_cols.append('mentions_24h')
        if 'virality_index' in current_scan_df.columns:
            merge_cols.append('virality_index')

        # Merge Phase 1 tickers with today's scan
        phase2_df = phase1_df.merge(
            current_scan_df[merge_cols],
            on='ticker',
            how='inner',
            suffixes=('_history', '_today')
        )

        if phase2_df.empty:
            _logger.info("No Phase 1 tickers found in current scan")
            return pd.DataFrame()

        # Build filter conditions
        conditions = []

        # Volume acceleration (required)
        if 'vol_zscore' in phase2_df.columns:
            conditions.append(phase2_df['vol_zscore'] >= self.config.phase2_min_vol_zscore)

        # Sentiment OR virality (at least one must be true)
        sentiment_conditions = []
        if 'sentiment_score' in phase2_df.columns:
            sentiment_conditions.append(
                phase2_df['sentiment_score'] >= self.config.phase2_min_sentiment
            )
        if 'virality_index' in phase2_df.columns:
            sentiment_conditions.append(
                phase2_df['virality_index'] >= self.config.phase2_min_virality
            )

        # Combine conditions
        if conditions and sentiment_conditions:
            # Volume acceleration AND (sentiment OR virality)
            final_condition = conditions[0]
            if len(sentiment_conditions) > 1:
                final_condition = final_condition & (sentiment_conditions[0] | sentiment_conditions[1])
            elif len(sentiment_conditions) == 1:
                final_condition = final_condition & sentiment_conditions[0]

            phase2_df = phase2_df[final_condition].copy()
        elif conditions:
            # Only volume condition
            phase2_df = phase2_df[conditions[0]].copy()
        else:
            # No valid conditions, return empty
            _logger.warning("No valid Phase 2 detection conditions available")
            return pd.DataFrame()

        phase2_df['phase'] = 'Phase 2: Early Public Signal'
        phase2_df['alert_priority'] = 'HIGH'

        _logger.info(
            "Detected %d Phase 2 transitions (HOT candidates) 🔥",
            len(phase2_df)
        )

        return phase2_df

    def generate_outputs(
        self,
        frequency_df: pd.DataFrame,
        phase1_df: pd.DataFrame,
        phase2_df: pd.DataFrame,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Generate output files.

        Args:
            frequency_df: Rolling candidates frequency data
            phase1_df: Phase 1 watchlist
            phase2_df: Phase 2 alerts
            output_dir: Output directory

        Returns:
            Dict of {file_type: file_path}
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = {}

        # 1. Rolling candidates (all tickers in 10-day window)
        if self.config.save_rolling_candidates and not frequency_df.empty:
            rolling_file = output_dir / '07_rolling_candidates.csv'
            frequency_df.to_csv(rolling_file, index=False)
            output_files['rolling_candidates'] = rolling_file
            _logger.info("Saved rolling candidates: %s", rolling_file)

        # 2. Phase 1 watchlist
        if self.config.save_phase1_watchlist and not phase1_df.empty:
            phase1_file = output_dir / '08_phase1_watchlist.csv'
            phase1_df.to_csv(phase1_file, index=False)
            output_files['phase1_watchlist'] = phase1_file
            _logger.info("Saved Phase 1 watchlist: %s", phase1_file)

        # 3. Phase 2 alerts (HOT)
        if self.config.save_phase2_alerts and not phase2_df.empty:
            phase2_file = output_dir / '09_phase2_alerts.csv'
            phase2_df.to_csv(phase2_file, index=False)
            output_files['phase2_alerts'] = phase2_file
            _logger.info("Saved Phase 2 alerts: %s ⚠️", phase2_file)

        return output_files
