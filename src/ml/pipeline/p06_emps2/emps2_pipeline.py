"""
EMPS2 Pipeline Orchestrator

Main orchestration logic for the EMPS2 pipeline.
Coordinates all stages: universe download, fundamental filtering, volatility filtering.
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
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p06_emps2.config import EMPS2PipelineConfig
from src.ml.pipeline.p06_emps2.universe_downloader import NasdaqUniverseDownloader
from src.ml.pipeline.p06_emps2.fundamental_filter import FundamentalFilter
from src.ml.pipeline.p06_emps2.volatility_filter import VolatilityFilter
from src.ml.pipeline.p06_emps2.rolling_memory import RollingMemoryScanner
from src.ml.pipeline.p06_emps2.sentiment_filter import SentimentFilter
from src.ml.pipeline.p06_emps2.alerts import EMPS2AlertSender
from src.ml.pipeline.p06_emps2.uoa_analyzer import UOAAnalyzer

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class EMPS2Pipeline:
    """
    EMPS2 Pipeline - Enhanced Explosive Move Pre-Screener.

    Multi-stage filtering pipeline:
    1. Download NASDAQ universe (~8000 tickers)
    2. Apply fundamental filters (~500-1000 tickers)
    3. Apply volatility filters (~50-200 tickers)
    4. Rolling memory analysis (14-day accumulation tracking)
    5. UOA analysis (unusual options activity)
    6. Save final results and send alerts
    7. Sentiment data collection (social momentum metrics)

    All output saved to results/emps2/YYYY-MM-DD/
    """

    def __init__(self, config: Optional[EMPS2PipelineConfig] = None, target_date: Optional[str] = None):
        """
        Initialize EMPS2 pipeline.

        Args:
            config: Optional pipeline configuration (uses defaults if None)
            target_date: Target trading date (YYYY-MM-DD). Defaults to yesterday.
                        Allows manual override for backfilling or testing.
        """
        self.config = config or EMPS2PipelineConfig.create_default()

        # Target date defaults to yesterday (for end-of-day data)
        if target_date is None:
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            target_date = yesterday.strftime('%Y-%m-%d')

        self.target_date = target_date

        # Results directory (dated for target trading day)
        self._results_dir = Path("results") / "emps2" / target_date
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Results base path (for rolling memory to access historical data)
        self._results_base_path = Path("results") / "emps2"

        # Set up per-scan logging to pipeline.log in results directory
        self._setup_pipeline_logging()

        # Initialize components (pass target_date for consistent folder usage)
        self.universe_downloader = NasdaqUniverseDownloader(
            self.config.universe_config,
            target_date=target_date
        )

        self.finnhub = FinnhubDataDownloader()
        self.fundamental_filter = FundamentalFilter(
            self.finnhub,
            self.config.filter_config,
            target_date=target_date
        )

        self.yahoo = YahooDataDownloader()
        self.volatility_filter = VolatilityFilter(
            self.yahoo,
            self.config.filter_config,
            target_date=target_date
        )

        # Rolling memory scanner
        self.rolling_memory = RollingMemoryScanner(
            config=self.config.rolling_memory_config,
            results_base_path=self._results_base_path,
            target_date=target_date,
            verbose=self.config.verbose_logging
        )

        # Alert sender
        self.alert_sender = EMPS2AlertSender(user_id=self.config.user_id) if self.config.rolling_memory_config.send_alerts else None

        # Sentiment filter
        self.sentiment_filter = SentimentFilter(
            self.config.sentiment_config,
            target_date=target_date
        ) if self.config.sentiment_config.enabled else None

        _logger.info("EMPS2 Pipeline initialized (target_date: %s, results: %s)",
                    self.target_date, self._results_dir)

    def _setup_pipeline_logging(self) -> None:
        """
        Set up per-scan logging to pipeline.log in results directory.

        This adds a file handler to the root logger that captures all pipeline
        logs to results/emps2/YYYY-MM-DD/pipeline.log for this specific scan.
        """
        log_file = self._results_dir / "pipeline.log"

        # Create file handler with rotation (max 100MB, 3 backups)
        file_handler = RotatingFileHandler(
            str(log_file),
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        # Use detailed formatter to match project standards
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to root logger so all modules log to this file
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        # Store handler reference for cleanup
        self._pipeline_log_handler = file_handler

        _logger.info("Per-scan logging initialized: %s", log_file)

    def _cleanup_pipeline_logging(self) -> None:
        """
        Clean up the per-scan log handler.

        Removes the pipeline log handler from the root logger to prevent
        logs from subsequent scans being written to this scan's log file.
        """
        if hasattr(self, '_pipeline_log_handler'):
            root_logger = logging.getLogger()
            root_logger.removeHandler(self._pipeline_log_handler)
            self._pipeline_log_handler.close()
            _logger.debug("Per-scan logging handler cleaned up")

    def run(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Run complete EMPS2 pipeline.

        Args:
            force_refresh: If True, bypass caches and force fresh data

        Returns:
            DataFrame with final filtered universe
        """
        try:
            start_time = datetime.now()
            _logger.info("="*70)
            _logger.info("EMPS2 Pipeline Starting")
            _logger.info("="*70)

            # Stage 1: Download universe
            universe_tickers = self._stage1_download_universe(force_refresh)

            if not universe_tickers:
                _logger.error("Failed to download universe")
                return pd.DataFrame()

            # Stage 2: Fundamental filtering
            fundamental_df = self._stage2_fundamental_filter(universe_tickers, force_refresh)

            if fundamental_df.empty:
                _logger.error("No tickers passed fundamental filters")
                return pd.DataFrame()

            # Stage 2b: Download TRF data (for volume correction in Stage 3)
            self._stage2b_download_trf_data()

            # Stage 3: Volatility filtering
            volatility_df = self._stage3_volatility_filter(
                fundamental_df['ticker'].tolist()
            )

            if volatility_df.empty:
                _logger.error("No tickers passed volatility filters")
                return pd.DataFrame()

            # Stage 4: Rolling Memory Analysis (10-day accumulation tracking)
            phase1_df, phase2_df = self._stage4_rolling_memory_analysis(volatility_df)
            _logger.info("After Stage 4: Phase1=%d, Phase2=%d", len(phase1_df), len(phase2_df))

            # Run stage 5: UOA Analysis
            self._stage5_uoa_analysis()

            # Stage 6: Create final results (before sentiment collection)
            final_df = self._stage6_create_final_results(
                fundamental_df,
                volatility_df,
                phase1_df,
                phase2_df
            )

            # Stage 7: Sentiment Data Collection (social momentum metrics)
            sentiment_df = self._stage7_sentiment_data_collection(phase2_df)

            # Generate summary
            if self.config.generate_summary:
                self._generate_summary(
                    len(universe_tickers),
                    len(fundamental_df),
                    len(volatility_df),
                    len(phase1_df) if not phase1_df.empty else 0,
                    len(phase2_df) if not phase2_df.empty else 0,
                    start_time
                )

            elapsed = (datetime.now() - start_time).total_seconds()
            _logger.info("="*70)
            _logger.info("EMPS2 Pipeline Completed in %.1f seconds", elapsed)
            _logger.info("Final universe size: %d tickers", len(final_df))
            _logger.info("="*70)

            return final_df

        except Exception:
            _logger.exception("Error running EMPS2 pipeline:")
            return pd.DataFrame()

        finally:
            # Always clean up the log handler
            self._cleanup_pipeline_logging()

    def _stage1_download_universe(self, force_refresh: bool) -> list:
        """
        Stage 1: Download NASDAQ universe.

        Args:
            force_refresh: Force fresh download

        Returns:
            List of ticker symbols
        """
        _logger.info("-"*70)
        _logger.info("Stage 1: Downloading NASDAQ Universe")
        _logger.info("-"*70)

        tickers = self.universe_downloader.download_universe(force_refresh)

        _logger.info("Stage 1 complete: %d tickers", len(tickers))
        return tickers

    def _stage2_fundamental_filter(self, tickers: list, force_refresh: bool = False) -> pd.DataFrame:
        """
        Stage 2: Apply fundamental filters.

        Args:
            tickers: List of ticker symbols
            force_refresh: Force fresh data fetch (bypass cache)

        Returns:
            DataFrame with filtered tickers
        """
        _logger.info("-"*70)
        _logger.info("Stage 2: Applying Fundamental Filters")
        _logger.info("-"*70)

        df = self.fundamental_filter.apply_filters(tickers, force_refresh)

        _logger.info("Stage 2 complete: %d tickers", len(df))
        return df

    def _stage2b_download_trf_data(self) -> None:
        """
        Stage 2b: Download TRF data for volume correction.

        Downloads FINRA TRF data for yesterday to provide dark pool volume
        corrections for the volatility filter stage.

        Smart behavior:
        - Checks if TRF file already exists in today's results folder
        - If exists: Uses existing data (no re-download)
        - If missing: Downloads from FINRA API
        - This allows daily pipeline runs to accumulate historical TRF data
          without redundant downloads
        """
        _logger.info("-"*70)
        _logger.info("Stage 2b: TRF Data Acquisition")
        _logger.info("-"*70)

        try:
            from src.data.downloader.finra_data_downloader import FinraDataDownloader
            from datetime import timedelta

            # TRF data should match our target_date (already yesterday)
            trf_dir = self._results_dir  # Use same directory as pipeline target
            trf_file = trf_dir / "trf.csv"

            if trf_file.exists():
                _logger.info("TRF data already exists for %s: %s", self.target_date, trf_file)

                # Validate the existing file
                try:
                    import pandas as pd
                    trf_df = pd.read_csv(trf_file)

                    if trf_df.empty:
                        _logger.warning("Existing TRF file is empty, will re-download")
                        trf_file.unlink()  # Delete empty file
                    else:
                        ticker_count = len(trf_df["ticker"].unique())
                        _logger.info("Using existing TRF data: %d tickers", ticker_count)
                        _logger.info("Stage 2b complete: Using cached TRF data")
                        return

                except Exception as e:
                    _logger.warning("Failed to read existing TRF file: %s, will re-download", str(e))
                    trf_file.unlink()  # Delete corrupted file

            # TRF file doesn't exist or was invalid, download it
            _logger.info("Downloading TRF data for %s", self.target_date)

            downloader = FinraDataDownloader(
                date=self.target_date,
                output_dir=str(trf_dir),  # Save in target date's directory
                output_filename="trf.csv",
                fetch_yfinance_data=True  # Include yfinance validation
            )

            result_df = downloader.run()

            if result_df.empty:
                _logger.warning("No TRF data downloaded for %s (market may be closed)", self.target_date)
                _logger.info("Stage 2b complete: No TRF data available")
            else:
                ticker_count = len(result_df["ticker"].unique())
                _logger.info("Successfully downloaded TRF data for %s: %d tickers",
                           self.target_date, ticker_count)
                _logger.info("Stage 2b complete: TRF data ready for volume correction")

        except Exception:
            _logger.exception("Error in TRF data acquisition:")
            _logger.warning("Continuing without TRF volume corrections")

    def _stage3_volatility_filter(self, tickers: list) -> pd.DataFrame:
        """
        Stage 3: Apply volatility filters.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with volatility-filtered tickers and metrics
        """
        _logger.info("-"*70)
        _logger.info("Stage 3: Applying Volatility Filters")
        _logger.info("-"*70)

        # Apply volatility filter (saves to 04_volatility_filtered.csv)
        filtered_tickers = self.volatility_filter.apply_filters(tickers)

        # Read back the saved CSV to get the full DataFrame with metrics
        volatility_csv = self._results_dir / "05_volatility_filtered.csv"
        if volatility_csv.exists():
            filtered_df = pd.read_csv(volatility_csv)
        else:
            # Fallback: create simple DataFrame from tickers
            filtered_df = pd.DataFrame({'ticker': filtered_tickers})

        _logger.info("Stage 3 complete: %d tickers", len(filtered_df))
        return filtered_df

    def _stage4_rolling_memory_analysis(
        self,
        volatility_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stage 4: Rolling Memory Analysis and Phase Detection.

        Scans historical results to detect:
        - Phase 1: Quiet Accumulation (5+ appearances in 14 days)
        - Phase 2: Early Public Signal (Phase 1 + acceleration)

        Args:
            volatility_df: DataFrame from volatility filter stage

        Returns:
            Tuple of (phase1_df, phase2_df)
        """
        _logger.info("-"*70)
        _logger.info("Stage 4: Rolling Memory Analysis")
        _logger.info("-"*70)

        if not self.config.rolling_memory_config.enabled:
            _logger.info("Rolling memory disabled, skipping phase detection")
            return pd.DataFrame(), pd.DataFrame()

        # Scan historical results
        _logger.info("Scanning historical results for rolling memory analysis...")
        historical_df = self.rolling_memory.scan_historical_results()
        _logger.info("Historical data found: %d records", len(historical_df))

        if historical_df.empty:
            _logger.warning("No historical data found, skipping phase detection")
            _logger.warning("This might be the first run - phase detection requires historical data")
            return pd.DataFrame(), pd.DataFrame()

        # Calculate appearance frequency
        frequency_df = self.rolling_memory.calculate_appearance_frequency(historical_df)
        _logger.info("Frequency analysis: %d unique tickers", len(frequency_df))

        # Detect Phase 1 (quiet accumulation)
        phase1_df = self.rolling_memory.detect_phase1_candidates(frequency_df)
        _logger.info("Phase 1 detection: %d candidates", len(phase1_df))

        # Detect Phase 2 (early public signal)
        phase2_df = self.rolling_memory.detect_phase2_candidates(
            phase1_df=phase1_df,
            current_scan_df=volatility_df
        )
        _logger.info("Phase 2 detection: %d candidates", len(phase2_df))

        # Generate outputs
        output_files = self.rolling_memory.generate_outputs(
            frequency_df=frequency_df,
            phase1_df=phase1_df,
            phase2_df=phase2_df,
            output_dir=self._results_dir
        )

        # Send alerts
        if self.alert_sender:
            if not phase2_df.empty and self.config.rolling_memory_config.alert_on_phase2:
                self.alert_sender.send_phase2_alert(
                    phase2_df=phase2_df,
                    phase2_csv_path=output_files.get('phase2_alerts')
                )

            if not phase1_df.empty and self.config.rolling_memory_config.alert_on_phase1:
                self.alert_sender.send_phase1_alert(
                    phase1_df=phase1_df,
                    phase1_csv_path=output_files.get('phase1_watchlist')
                )

        _logger.info("Stage 4 complete: Phase 1=%d, Phase 2=%d",
                    len(phase1_df), len(phase2_df))

        return phase1_df, phase2_df

    def _stage7_sentiment_data_collection(self, phase2_df: pd.DataFrame) -> pd.DataFrame:
        """
        Stage 7: Sentiment Data Collection.

        Collects sentiment data for all Phase 2 candidates without filtering.
        Saves social momentum metrics to separate file: 10_sentiments.csv

        Args:
            phase2_df: Phase 2 alerts DataFrame

        Returns:
            DataFrame with sentiment metrics for all tickers
        """
        _logger.info("-"*70)
        _logger.info("Stage 7: Sentiment Data Collection")
        _logger.info("-"*70)

        # Check if sentiment is enabled
        if not self.config.sentiment_config.enabled:
            _logger.info("Sentiment data collection is disabled in config")
            return pd.DataFrame()

        if phase2_df.empty:
            _logger.info("No Phase 2 candidates to collect sentiment data for")
            return pd.DataFrame()

        # Extract tickers from Phase 2
        tickers = phase2_df['ticker'].tolist()
        _logger.info("Collecting sentiment data for %d Phase 2 candidates", len(tickers))

        try:
            # Create a new SentimentFilter instance with minimal filtering
            from src.ml.pipeline.p06_emps2.sentiment_filter import SentimentFilter
            from src.ml.pipeline.p06_emps2.config import SentimentFilterConfig

            # Configure to collect all data without filtering
            config = SentimentFilterConfig(
                min_mentions_24h=0,      # No minimum mentions
                min_sentiment_score=-1.0,  # Include all sentiment scores
                max_bot_pct=1.0,         # Include all bot percentages
                min_virality_index=0.0,   # No minimum virality
                min_unique_authors=0,     # No minimum authors
                enabled=True
            )

            # Initialize a temporary sentiment filter with the permissive config
            sentiment_filter = SentimentFilter(config)

            # Apply the filter to collect data (won't filter anything out with these settings)
            sentiment_df = sentiment_filter.apply_filters(tickers)

            _logger.info("Collected sentiment data for %d/%d tickers", len(sentiment_df), len(tickers))

            # Save sentiment data to separate file
            if not sentiment_df.empty:
                sentiment_output_path = self._results_dir / "10_sentiments.csv"
                sentiment_df.to_csv(sentiment_output_path, index=False)
                _logger.info("Saved sentiment data to: %s (%d tickers)", sentiment_output_path, len(sentiment_df))
            else:
                _logger.warning("No sentiment data collected to save")

            return sentiment_df

        except Exception as e:
            _logger.exception("Error collecting sentiment data:")
            return pd.DataFrame()


    def _stage6_create_final_results(
        self,
        fundamental_df: pd.DataFrame,
        volatility_df: pd.DataFrame,
        phase1_df: pd.DataFrame,
        phase2_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Stage 6: Create final results DataFrame.

        Args:
            fundamental_df: DataFrame with fundamental data
            volatility_df: DataFrame with volatility metrics
            phase1_df: Phase 1 watchlist (optional)
            phase2_df: Phase 2 alerts (optional)

        Returns:
            Final filtered DataFrame
        """
        _logger.info("-"*70)
        _logger.info("Stage 6: Creating Final Results")
        _logger.info("-"*70)

        # Use volatility_df as the final results (already has all metrics)
        final_df = volatility_df.copy()

        # Add phase information if available
        if not phase1_df.empty:
            phase1_tickers = set(phase1_df['ticker'].tolist())
            final_df['in_phase1'] = final_df['ticker'].isin(phase1_tickers)
        else:
            final_df['in_phase1'] = False

        if not phase2_df.empty:
            phase2_tickers = set(phase2_df['ticker'].tolist())
            final_df['in_phase2'] = final_df['ticker'].isin(phase2_tickers)
            final_df['alert_priority'] = final_df['ticker'].apply(
                lambda x: 'HIGH' if x in phase2_tickers else 'NORMAL'
            )
        else:
            final_df['in_phase2'] = False
            final_df['alert_priority'] = 'NORMAL'

        # Add scan metadata
        final_df['scan_date'] = self.target_date
        final_df['scan_timestamp'] = datetime.now().isoformat()  # When scan was run

        # Save final results
        output_path = self._results_dir / "09_final_universe.csv"
        final_df.to_csv(output_path, index=False)
        _logger.info("Saved final results to: %s", output_path)

        _logger.info("Stage 6 complete: %d tickers in final universe", len(final_df))
        return final_df

    def _stage5_uoa_analysis(self) -> None:
        """Stage 5: Perform UOA analysis on rolling candidates"""
        _logger.info("-"*70)
        _logger.info("Stage 5: UOA Analysis")
        _logger.info("-"*70)

        if not self.config.enable_uoa_analysis:
            _logger.info("UOA analysis is disabled in config")
            return

        try:
            uoa_analyzer = UOAAnalyzer(self._results_dir.parent)
            uoa_file = uoa_analyzer.run()

            if uoa_file and uoa_file.exists():
                _logger.info("UOA analysis complete. Results saved to: %s", uoa_file)
            else:
                _logger.warning("No UOA data was generated")

        except Exception as e:
            _logger.exception("UOA analysis failed")

    def _generate_summary(
        self,
        initial_count: int,
        fundamental_count: int,
        volatility_count: int,
        phase1_count: int,
        phase2_count: int,
        start_time: datetime
    ) -> None:
        """
        Generate and save pipeline summary.

        Args:
            initial_count: Initial universe size
            fundamental_count: After fundamental filtering
            volatility_count: After volatility filtering
            phase1_count: Phase 1 candidates count
            phase2_count: Phase 2 alerts count
            start_time: Pipeline start time
        """
        try:
            elapsed = (datetime.now() - start_time).total_seconds()

            summary = {
                'pipeline': 'EMPS2',
                'version': '2.2',  # Updated version with target_date support
                'target_date': self.target_date,  # Target trading day
                'scan_timestamp': datetime.now().isoformat(),  # When scan was executed
                'elapsed_seconds': elapsed,
                'stages': {
                    'stage1_universe': {
                        'count': initial_count,
                        'percentage': 100.0
                    },
                    'stage2_fundamental': {
                        'count': fundamental_count,
                        'percentage': 100.0 * fundamental_count / initial_count if initial_count > 0 else 0,
                        'removed': initial_count - fundamental_count
                    },
                    'stage2b_trf': {
                        'enabled': True,
                        'trf_file_exists': (self._results_dir / "trf.csv").exists(),
                        'description': 'TRF volume corrections for dark pool activity'
                    },
                    'stage3_volatility': {
                        'count': volatility_count,
                        'percentage': 100.0 * volatility_count / fundamental_count if fundamental_count > 0 else 0,
                        'removed': fundamental_count - volatility_count,
                        'uses_trf_corrections': (self._results_dir / "trf.csv").exists()
                    },
                    'stage4_rolling_memory': {
                        'enabled': self.config.rolling_memory_config.enabled,
                        'phase1_count': phase1_count,
                        'phase2_count': phase2_count,
                        'lookback_days': self.config.rolling_memory_config.lookback_days
                    }
                },
                'final_universe': {
                    'count': volatility_count,
                    'percentage_of_initial': 100.0 * volatility_count / initial_count if initial_count > 0 else 0,
                    'phase1_candidates': phase1_count,
                    'phase2_alerts': phase2_count
                },
                'config': {
                    'min_price': self.config.filter_config.min_price,
                    'min_avg_volume': self.config.filter_config.min_avg_volume,
                    'min_market_cap': self.config.filter_config.min_market_cap,
                    'max_market_cap': self.config.filter_config.max_market_cap,
                    'max_float': self.config.filter_config.max_float,
                    'min_volatility_threshold': self.config.filter_config.min_volatility_threshold,
                    'min_price_range': self.config.filter_config.min_price_range,
                    'min_vol_zscore': self.config.filter_config.min_vol_zscore,
                    'min_vol_rv_ratio': self.config.filter_config.min_vol_rv_ratio,
                    'lookback_days': self.config.filter_config.lookback_days,
                    'interval': self.config.filter_config.interval,
                    'atr_period': self.config.filter_config.atr_period,
                    'rolling_memory_enabled': self.config.rolling_memory_config.enabled,
                    'phase1_min_appearances': self.config.rolling_memory_config.phase1_min_appearances
                }
            }

            # Save summary as JSON
            summary_path = self._results_dir / "summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            _logger.info("Saved pipeline summary to: %s", summary_path)

            # Log summary
            _logger.info("")
            _logger.info("Pipeline Summary:")
            _logger.info("  Initial universe: %d tickers", initial_count)
            _logger.info("  After fundamental: %d tickers (%.1f%%)",
                        fundamental_count,
                        summary['stages']['stage2_fundamental']['percentage'])
            _logger.info("  After volatility: %d tickers (%.1f%%)",
                        volatility_count,
                        summary['stages']['stage3_volatility']['percentage'])
            if self.config.rolling_memory_config.enabled:
                _logger.info("  Phase 1 candidates: %d tickers", phase1_count)
                _logger.info("  Phase 2 alerts: %d tickers 🔥", phase2_count)
            _logger.info("  Final universe: %d tickers (%.2f%% of initial)",
                        volatility_count,
                        summary['final_universe']['percentage_of_initial'])
            _logger.info("  Total time: %.1f seconds", elapsed)

        except Exception:
            _logger.exception("Error generating summary:")


def create_pipeline(config: Optional[EMPS2PipelineConfig] = None,
                   target_date: Optional[str] = None) -> EMPS2Pipeline:
    """
    Factory function to create EMPS2 pipeline.

    Args:
        config: Optional pipeline configuration
        target_date: Optional target trading date (YYYY-MM-DD). Defaults to yesterday.

    Returns:
        EMPS2Pipeline instance
    """
    return EMPS2Pipeline(config, target_date)
