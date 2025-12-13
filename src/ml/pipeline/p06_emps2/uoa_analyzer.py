# src/ml/pipeline/p06_emps2/uoa_analyzer.py
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Optional
from src.notification.logger import setup_logger
from src.data.downloader.eodhd_downloader import EODHDDataDownloader

_logger = setup_logger(__name__)

class UOAAnalyzer:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.eodhd_downloader = EODHDDataDownloader()

    def get_yesterday_str(self) -> str:
        """Get yesterday's date in YYYY-MM-DD format"""
        return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    def get_todays_candidates(self) -> List[str]:
        """Get tickers from today's 07_rolling_candidates.csv"""
        today = datetime.now().strftime('%Y-%m-%d')
        candidates_file = self.results_dir / today / "07_rolling_candidates.csv"

        if not candidates_file.exists():
            _logger.warning(f"No rolling candidates file found at {candidates_file}")
            return []

        try:
            df = pd.read_csv(candidates_file)
            return df['ticker'].tolist()
        except Exception as e:
            _logger.error(f"Error reading candidates file: {e}")
            return []

    def analyze_uoa(self, tickers: List[str], target_date: Optional[str] = None) -> pd.DataFrame:
        """Download and calculate UOA data for given tickers"""
        if not tickers:
            _logger.warning("No tickers provided for UOA analysis")
            return pd.DataFrame()

        target_date = target_date or self.get_yesterday_str()
        _logger.info(f"Starting UOA analysis for {len(tickers)} tickers on {target_date}")

        all_data = []
        for ticker in tickers:
            try:
                df = self.eodhd_downloader.download_for_date([ticker], target_date)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                _logger.error(f"Error processing {ticker}: {str(e)}")
                continue

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def save_uoa_data(self, uoa_df: pd.DataFrame, target_date: Optional[str] = None) -> Path:
        """Save UOA data to the appropriate directory"""
        if uoa_df.empty:
            _logger.warning("No UOA data to save")
            return None

        target_date = target_date or self.get_yesterday_str()
        output_dir = self.results_dir / target_date
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "uoa.csv"
        uoa_df.to_csv(output_file, index=False)
        _logger.info(f"Saved UOA data to {output_file}")
        return output_file

    def run(self) -> Optional[Path]:
        """Run the UOA analysis pipeline"""
        tickers = self.get_todays_candidates()
        if not tickers:
            _logger.warning("No tickers found for UOA analysis")
            return None

        target_date = self.get_yesterday_str()
        uoa_df = self.analyze_uoa(tickers, target_date)

        if uoa_df.empty:
            _logger.warning("No UOA data generated")
            return None

        return self.save_uoa_data(uoa_df, target_date)