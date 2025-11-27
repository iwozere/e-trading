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

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p06_emps2.config import EMPS2PipelineConfig
from src.ml.pipeline.p06_emps2.universe_downloader import NasdaqUniverseDownloader
from src.ml.pipeline.p06_emps2.fundamental_filter import FundamentalFilter
from src.ml.pipeline.p06_emps2.volatility_filter import VolatilityFilter

_logger = setup_logger(__name__)


class EMPS2Pipeline:
    """
    EMPS2 Pipeline - Enhanced Explosive Move Pre-Screener.

    Multi-stage filtering pipeline:
    1. Download NASDAQ universe (~8000 tickers)
    2. Apply fundamental filters (~500-1000 tickers)
    3. Apply volatility filters (~50-200 tickers)
    4. Save final results

    All output saved to results/emps2/YYYY-MM-DD/
    """

    def __init__(self, config: Optional[EMPS2PipelineConfig] = None):
        """
        Initialize EMPS2 pipeline.

        Args:
            config: Optional pipeline configuration (uses defaults if None)
        """
        self.config = config or EMPS2PipelineConfig.create_default()

        # Results directory (dated)
        today = datetime.now().strftime('%Y-%m-%d')
        self._results_dir = Path("results") / "emps2" / today
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.universe_downloader = NasdaqUniverseDownloader(self.config.universe_config)

        self.finnhub = FinnhubDataDownloader()
        self.fundamental_filter = FundamentalFilter(
            self.finnhub,
            self.config.filter_config
        )

        self.yahoo = YahooDataDownloader()
        self.volatility_filter = VolatilityFilter(
            self.yahoo,
            self.config.filter_config
        )

        _logger.info("EMPS2 Pipeline initialized (results: %s)", self._results_dir)

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

            # Stage 3: Volatility filtering
            volatility_tickers = self._stage3_volatility_filter(
                fundamental_df['ticker'].tolist()
            )

            if not volatility_tickers:
                _logger.error("No tickers passed volatility filters")
                return pd.DataFrame()

            # Stage 4: Create final results
            final_df = self._stage4_create_final_results(
                fundamental_df,
                volatility_tickers
            )

            # Generate summary
            if self.config.generate_summary:
                self._generate_summary(
                    len(universe_tickers),
                    len(fundamental_df),
                    len(volatility_tickers),
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

    def _stage3_volatility_filter(self, tickers: list) -> list:
        """
        Stage 3: Apply volatility filters.

        Args:
            tickers: List of ticker symbols

        Returns:
            List of tickers passing filters
        """
        _logger.info("-"*70)
        _logger.info("Stage 3: Applying Volatility Filters")
        _logger.info("-"*70)

        filtered_tickers = self.volatility_filter.apply_filters(tickers)

        _logger.info("Stage 3 complete: %d tickers", len(filtered_tickers))
        return filtered_tickers

    def _stage4_create_final_results(
        self,
        fundamental_df: pd.DataFrame,
        volatility_tickers: list
    ) -> pd.DataFrame:
        """
        Stage 4: Create final results DataFrame.

        Args:
            fundamental_df: DataFrame with fundamental data
            volatility_tickers: List of tickers passing volatility filters

        Returns:
            Final filtered DataFrame
        """
        _logger.info("-"*70)
        _logger.info("Stage 4: Creating Final Results")
        _logger.info("-"*70)

        # Filter fundamental data to only include volatility-passed tickers
        final_df = fundamental_df[fundamental_df['ticker'].isin(volatility_tickers)].copy()

        # Add scan metadata
        final_df['scan_date'] = datetime.now().strftime('%Y-%m-%d')
        final_df['scan_timestamp'] = datetime.now().isoformat()

        # Save final results
        output_path = self._results_dir / "prefiltered_universe.csv"
        final_df.to_csv(output_path, index=False)
        _logger.info("Saved final results to: %s", output_path)

        _logger.info("Stage 4 complete: %d tickers in final universe", len(final_df))
        return final_df

    def _generate_summary(
        self,
        initial_count: int,
        fundamental_count: int,
        volatility_count: int,
        start_time: datetime
    ) -> None:
        """
        Generate and save pipeline summary.

        Args:
            initial_count: Initial universe size
            fundamental_count: After fundamental filtering
            volatility_count: After volatility filtering
            start_time: Pipeline start time
        """
        try:
            elapsed = (datetime.now() - start_time).total_seconds()

            summary = {
                'pipeline': 'EMPS2',
                'version': '1.0',
                'scan_date': datetime.now().strftime('%Y-%m-%d'),
                'scan_timestamp': datetime.now().isoformat(),
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
                    'stage3_volatility': {
                        'count': volatility_count,
                        'percentage': 100.0 * volatility_count / fundamental_count if fundamental_count > 0 else 0,
                        'removed': fundamental_count - volatility_count
                    }
                },
                'final_universe': {
                    'count': volatility_count,
                    'percentage_of_initial': 100.0 * volatility_count / initial_count if initial_count > 0 else 0
                },
                'config': {
                    'min_price': self.config.filter_config.min_price,
                    'min_avg_volume': self.config.filter_config.min_avg_volume,
                    'min_market_cap': self.config.filter_config.min_market_cap,
                    'max_market_cap': self.config.filter_config.max_market_cap,
                    'max_float': self.config.filter_config.max_float,
                    'min_volatility_threshold': self.config.filter_config.min_volatility_threshold,
                    'min_price_range': self.config.filter_config.min_price_range,
                    'lookback_days': self.config.filter_config.lookback_days,
                    'interval': self.config.filter_config.interval,
                    'atr_period': self.config.filter_config.atr_period
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
            _logger.info("  Final universe: %d tickers (%.2f%% of initial)",
                        volatility_count,
                        summary['final_universe']['percentage_of_initial'])
            _logger.info("  Total time: %.1f seconds", elapsed)

        except Exception:
            _logger.exception("Error generating summary:")


def create_pipeline(config: Optional[EMPS2PipelineConfig] = None) -> EMPS2Pipeline:
    """
    Factory function to create EMPS2 pipeline.

    Args:
        config: Optional pipeline configuration

    Returns:
        EMPS2Pipeline instance
    """
    return EMPS2Pipeline(config)
