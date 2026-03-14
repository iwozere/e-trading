"""
Rolling Memory Module for P10 EMPS3
Phase 1.5 (Early Warning) Detection
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.notification.logger import setup_logger
from src.ml.pipeline.p10_emps3.config import EMPS3RollingMemoryConfig

_logger = setup_logger(__name__)

class EMPS3RollingMemoryScanner:
    def __init__(
        self,
        config: EMPS3RollingMemoryConfig,
        results_base_path: Path,
        target_date: str,
        verbose: bool = True
    ):
        self.config = config
        self.results_base_path = results_base_path
        self.target_date = target_date
        self.verbose = verbose

    def scan_historical_results(self, lookback_days: Optional[int] = None) -> pd.DataFrame:
        lookback = lookback_days or self.config.lookback_days
        today = datetime.strptime(self.target_date, '%Y-%m-%d').date()
        
        all_results = []

        for days_back in range(lookback + 1):
            scan_date = today - timedelta(days=days_back)
            date_str = scan_date.strftime('%Y-%m-%d')
            day_folder = self.results_base_path / date_str

            if not day_folder.exists():
                continue

            diag_file = day_folder / '08_absorption_diagnostics.csv'
            
            if diag_file.exists():
                try:
                    df = pd.read_csv(diag_file)
                    df['scan_date'] = scan_date
                    all_results.append(df)
                except Exception:
                    _logger.exception("Error loading %s", diag_file)

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def detect_phase1_5_candidates(self, historical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Phase 1.5 (Early Warning)
        Criteria:
        - Appeared 3+ times in last 5 days
        - ATR is trending downwards
        - Volume Z-Score is trending upwards
        """
        if historical_df.empty:
            return pd.DataFrame()

        # Group by ticker and aggregate
        # We need to sort by date for trend analysis
        historical_df = historical_df.sort_values('scan_date')
        
        # Filter to only the passed filters? 
        # Actually it's better to detect from passed items, or everything in diag.
        # We assume if they are in the diagnostic log with PASSED status they are valid
        valid_df = historical_df[historical_df['status'] == 'PASSED']

        if valid_df.empty:
            return pd.DataFrame()

        freq_df = valid_df.groupby('ticker').agg({
            'scan_date': ['count', 'min', 'max']
        }).reset_index()
        
        freq_df.columns = ['ticker', 'appearance_count', 'first_seen', 'last_seen']

        phase_1_5_candidates = []

        for _, row in freq_df[freq_df['appearance_count'] >= self.config.phase1_5_min_appearances].iterrows():
            ticker = row['ticker']
            ticker_data = valid_df[valid_df['ticker'] == ticker].sort_values('scan_date')

            if len(ticker_data) < 2:
                continue

            # Check trends using slope of linear regression
            x = np.arange(len(ticker_data))
            
            atr_ratios = ticker_data['atr_ratio'].values
            vol_zscores = ticker_data['vol_zscore'].values
            
            atr_slope, _ = np.polyfit(x, atr_ratios, 1) if len(x) > 1 else (0, 0)
            vol_slope, _ = np.polyfit(x, vol_zscores, 1) if len(x) > 1 else (0, 0)
            
            if atr_slope < 0 and vol_slope > 0:
                # Meets trending criteria
                candidate = {
                    'ticker': ticker,
                    'appearance_count': row['appearance_count'],
                    'atr_slope': atr_slope,
                    'vol_zscore_slope': vol_slope,
                    'latest_date': row['last_seen'],
                    'phase': 'Phase 1.5: Early Warning',
                    'alert_priority': 'HIGH'
                }
                phase_1_5_candidates.append(candidate)

        df = pd.DataFrame(phase_1_5_candidates)
        _logger.info("Detected %d Phase 1.5 candidates", len(df))
        return df

    def generate_outputs(
        self,
        phase1_5_df: pd.DataFrame,
        output_dir: Path
    ) -> Dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_files = {}

        if not phase1_5_df.empty:
            phase1_5_file = output_dir / '07_phase1_5_alerts.csv'
            phase1_5_df.to_csv(phase1_5_file, index=False)
            output_files['phase1_5_alerts'] = phase1_5_file
            _logger.info("Saved Phase 1.5 alerts: %s", phase1_5_file)

        return output_files
