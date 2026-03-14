"""
EMPS3 Pipeline Orchestrator

Main orchestration logic for the EMPS3 pipeline.
Coordinates all stages: universe download, fundamental filtering, accumulation analysis, rolling memory.
"""

from pathlib import Path
import sys
from typing import Optional
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.data_manager import DataManager
from src.ml.pipeline.p10_emps3.config import EMPS3PipelineConfig
from src.ml.pipeline.p06_emps2.universe_downloader import NasdaqUniverseDownloader
from src.ml.pipeline.p06_emps2.fundamental_filter import FundamentalFilter
from src.ml.pipeline.p10_emps3.accumulation_analyzer import AccumulationAnalyzer
from src.ml.pipeline.p10_emps3.rolling_memory import EMPS3RollingMemoryScanner
from src.ml.pipeline.p10_emps3.alerts import EMPS3AlertSender
from src.ml.pipeline.p06_emps2.sentiment_filter import SentimentFilter

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class EMPS3Pipeline:
    def __init__(self, config: Optional[EMPS3PipelineConfig] = None, target_date: Optional[str] = None):
        self.config = config or EMPS3PipelineConfig.create_default()

        if target_date is None:
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.strftime('%Y-%m-%d')

        self.target_date = target_date
        self._results_dir = Path("results") / "p10_emps3" / target_date
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._results_base_path = Path("results") / "p10_emps3"

        self._setup_pipeline_logging()

        self.universe_downloader = NasdaqUniverseDownloader(
            self.config.universe_config,
            target_date=target_date
        )

        self.finnhub = FinnhubDataDownloader()
        self.fundamental_filter = FundamentalFilter(
            self.finnhub,
            self.config.filter_config, # Will rely on duck typing since EMPS3FilterConfig maps to needed fields min_price, min_avg_volume, etc
            target_date=target_date,
            cache_enabled=self.config.fundamental_cache_enabled,
            cache_ttl_days=self.config.fundamental_cache_ttl_days,
            checkpoint_enabled=self.config.checkpoint_enabled,
            checkpoint_interval=self.config.checkpoint_interval
        )

        self.data_manager = DataManager()
        self.accumulation_analyzer = AccumulationAnalyzer(
            self.data_manager,
            self.config.filter_config,
            target_date=target_date
        )

        self.rolling_memory = EMPS3RollingMemoryScanner(
            config=self.config.rolling_memory_config,
            results_base_path=self._results_base_path,
            target_date=target_date,
            verbose=self.config.verbose_logging
        )

        self.sentiment_filter = SentimentFilter(
            self.config.sentiment_config,
            target_date=target_date
        )
        # Override results directory so sentiment output goes to the p10_emps3 folder
        self.sentiment_filter._results_dir = self._results_dir

        self.alert_sender = EMPS3AlertSender(user_id=self.config.user_id) if self.config.rolling_memory_config.send_alerts else None

        _logger.info("EMPS3 Pipeline initialized (target_date: %s, results: %s)",
                     self.target_date, self._results_dir)

    def _setup_pipeline_logging(self) -> None:
        log_file = self._results_dir / "pipeline.log"
        file_handler = RotatingFileHandler(str(log_file), maxBytes=100 * 1024 * 1024, backupCount=3, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s")
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        self._pipeline_log_handler = file_handler

    def _cleanup_pipeline_logging(self) -> None:
        if hasattr(self, '_pipeline_log_handler'):
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._pipeline_log_handler)
            self._pipeline_log_handler.close()

    def run(self, force_refresh: bool = False) -> pd.DataFrame:
        try:
            start_time = datetime.now()
            _logger.info("="*70)
            _logger.info("EMPS3 Pipeline Starting (Precursor Mode)")
            _logger.info("="*70)

            # Stage 1: Universe
            universe_tickers = self.universe_downloader.download_universe(force_refresh)
            if not universe_tickers:
                return pd.DataFrame()

            # Stage 2: Fundamental Filters
            fundamental_df = self.fundamental_filter.apply_filters(universe_tickers, force_refresh)
            if fundamental_df.empty:
                return pd.DataFrame()

            # Stage 2b: Download TRF 
            self._stage2b_download_trf_data()

            # Stage 3: Accumulation Analysis (Coiled Spring effect)
            accum_tickers = self.accumulation_analyzer.apply_filters(fundamental_df['ticker'].tolist())

            accumulation_csv = self._results_dir / "07_prebreakout_watchlist.csv"
            if accumulation_csv.exists():
                accum_df = pd.read_csv(accumulation_csv)
            else:
                accum_df = pd.DataFrame({'ticker': accum_tickers})
            
            # Stage 4: Sentiment Filter (Final filtering step before alerts/memory)
            # Fetch sentiment only for the highly filtered accum_tickers (expected ~10)
            if self.config.sentiment_config.enabled and not accum_df.empty:
                sentiment_df = self.sentiment_filter.apply_filters(accum_df['ticker'].tolist())
                # Merge sentiment data back into accum_df
                if not sentiment_df.empty:
                    final_df = accum_df.merge(sentiment_df, on='ticker', how='inner')
                else:
                    final_df = pd.DataFrame(columns=accum_df.columns.tolist() + ['mentions_24h', 'sentiment_score'])
            else:
                final_df = accum_df.copy()

            final_watchlist_csv = self._results_dir / "09_final_sentiment_watchlist.csv"
            if not final_df.empty:
                final_df.to_csv(final_watchlist_csv, index=False)
                _logger.info("Saved final sentiment-filtered watchlist to %s", final_watchlist_csv)
            
            # Stage 5: Phase 1.5 Early Warning Detection
            historical_df = self.rolling_memory.scan_historical_results()
            phase1_5_df = self.rolling_memory.detect_phase1_5_candidates(historical_df)
            
            if self.config.rolling_memory_config.save_rolling_candidates:
                self.rolling_memory.generate_outputs(phase1_5_df, self._results_dir)

            # Stage 6: Alerts
            if self.config.rolling_memory_config.send_alerts and self.config.rolling_memory_config.alert_on_phase_1_5:
                csv_path = self._results_dir / "07_phase1_5_alerts.csv"
                self.alert_sender.send_phase1_5_alert(phase1_5_df, csv_path)

            elapsed = (datetime.now() - start_time).total_seconds()
            _logger.info("="*70)
            _logger.info("EMPS3 Pipeline Completed in %.1f seconds", elapsed)
            _logger.info("Final universe size: %d tickers", len(final_df))
            _logger.info("="*70)

            return final_df

        except Exception:
            _logger.exception("Error running EMPS3 pipeline:")
            return pd.DataFrame()
        finally:
            self._cleanup_pipeline_logging()

    def _stage2b_download_trf_data(self) -> None:
        try:
            from src.data.downloader.finra_data_downloader import FinraDataDownloader
            trf_file = self._results_dir / "trf.csv"

            if trf_file.exists():
                _logger.info("TRF data exists.")
                return

            _logger.info("Downloading TRF data for %s", self.target_date)
            downloader = FinraDataDownloader(
                date=self.target_date,
                output_dir=str(self._results_dir),
                output_filename="trf.csv",
                fetch_yfinance_data=True
            )
            downloader.run()
        except Exception:
            _logger.exception("Error in TRF data acquisition:")
