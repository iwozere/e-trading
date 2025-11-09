#!/usr/bin/env python3
"""
Populate calculated fields in ss_finra_short_interest table.

This script calculates and populates:
1. short_interest_pct = (short_interest_shares / float_shares) * 100
2. days_to_cover = short_interest_shares / average_daily_volume

For days_to_cover calculation, it fetches volume data from FMP API.
"""

import sys
import time
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.db.core.database import session_scope
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.data.downloader.fmp_data_downloader import FMPDataDownloader

_logger = setup_logger(__name__)


class FINRAFieldPopulator:
    """Populates calculated fields in FINRA short interest table."""

    def __init__(self, batch_size: int = 50, api_delay: float = 0.2):
        """
        Initialize the populator.

        Args:
            batch_size: Number of tickers to process in each batch
            api_delay: Delay between API calls in seconds
        """
        self.batch_size = batch_size
        self.api_delay = api_delay
        self.fmp_downloader = FMPDataDownloader()
        self.volume_cache: Dict[str, Optional[float]] = {}

    def populate_short_interest_pct(self) -> Dict[str, int]:
        """
        Calculate and populate short_interest_pct field.

        Returns:
            Dictionary with update statistics
        """
        _logger.info("Starting short_interest_pct calculation...")

        try:
            with session_scope() as session:
                # Execute the calculation directly in SQL for efficiency
                # Note: We cap at 100% to avoid constraint violations, but log extreme cases
                result = session.execute("""
                    UPDATE ss_finra_short_interest
                    SET
                        short_interest_pct = CASE
                            WHEN float_shares > 0 THEN
                                -- Cap at 100% due to database constraint (log extreme cases separately)
                                LEAST(100.0, ROUND((short_interest_shares::numeric / float_shares::numeric) * 100, 4))
                            ELSE NULL
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE
                        short_interest_shares IS NOT NULL
                        AND float_shares IS NOT NULL
                        AND float_shares > 0
                        AND short_interest_pct IS NULL
                """)

                # Log extreme cases that were capped
                extreme_cases = session.execute("""
                    SELECT ticker, settlement_date, short_interest_shares, float_shares,
                           ROUND((short_interest_shares::numeric / float_shares::numeric) * 100, 4) as actual_pct
                    FROM ss_finra_short_interest
                    WHERE short_interest_shares IS NOT NULL
                      AND float_shares IS NOT NULL
                      AND float_shares > 0
                      AND (short_interest_shares::numeric / float_shares::numeric) * 100 > 100
                    ORDER BY actual_pct DESC
                    LIMIT 10
                """).fetchall()

                if extreme_cases:
                    _logger.warning("Found %d cases with short interest >100%% (capped due to DB constraint):", len(extreme_cases))
                    for case in extreme_cases:
                        _logger.warning("  %s (%s): %.2f%% short interest",
                                      case.ticker, case.settlement_date, case.actual_pct)

                updated_count = result.rowcount
                session.commit()

                # Get statistics
                stats_result = session.execute("""
                    SELECT
                        COUNT(*) as total_records,
                        COUNT(short_interest_pct) as records_with_pct,
                        MIN(short_interest_pct) as min_pct,
                        MAX(short_interest_pct) as max_pct,
                        AVG(short_interest_pct) as avg_pct
                    FROM ss_finra_short_interest
                """).fetchone()

                stats = {
                    'updated_count': updated_count,
                    'total_records': stats_result.total_records,
                    'records_with_pct': stats_result.records_with_pct,
                    'min_pct': float(stats_result.min_pct) if stats_result.min_pct else None,
                    'max_pct': float(stats_result.max_pct) if stats_result.max_pct else None,
                    'avg_pct': float(stats_result.avg_pct) if stats_result.avg_pct else None,
                }

                _logger.info("Short interest percentage calculation completed:")
                _logger.info("  Updated records: %d", updated_count)
                _logger.info("  Total records with pct: %d/%d",
                           stats['records_with_pct'], stats['total_records'])
                if stats['avg_pct']:
                    _logger.info("  Average short interest: %.2f%%", stats['avg_pct'])

                return stats

        except Exception as e:
            _logger.exception("Error calculating short interest percentages:")
            return {'updated_count': 0, 'error': str(e)}

    def get_tickers_needing_volume_data(self) -> List[str]:
        """Get list of tickers that need volume data for days_to_cover calculation."""
        try:
            with session_scope() as session:
                result = session.execute("""
                    SELECT DISTINCT ticker
                    FROM ss_finra_short_interest
                    WHERE short_interest_shares IS NOT NULL
                      AND short_interest_shares > 0
                      AND days_to_cover IS NULL
                    ORDER BY ticker
                """).fetchall()

                tickers = [row.ticker for row in result]
                _logger.info("Found %d tickers needing volume data", len(tickers))
                return tickers

        except Exception:
            _logger.exception("Error getting tickers needing volume data:")
            return []

    def fetch_volume_data(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch 30-day average volume data for tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to average volume
        """
        _logger.info("Fetching volume data for %d tickers...", len(tickers))

        if not self.fmp_downloader.test_connection():
            _logger.error("FMP API connection failed")
            return {}

        volume_data = {}
        end_date = date.today()
        start_date = end_date - timedelta(days=45)  # Get extra days to ensure 30 trading days

        # Process in batches to avoid API rate limits
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i:i + self.batch_size]
            _logger.info("Processing batch %d/%d (%d tickers)",
                        i // self.batch_size + 1,
                        (len(tickers) + self.batch_size - 1) // self.batch_size,
                        len(batch))

            try:
                # Get OHLCV data for the batch
                ohlcv_batch = self.fmp_downloader.get_ohlcv_batch(
                    tickers=batch,
                    start_date=start_date,
                    end_date=end_date
                )

                for ticker in batch:
                    if ticker in ohlcv_batch:
                        df = ohlcv_batch[ticker]

                        if not df.empty and 'volume' in df.columns:
                            # Calculate average volume over the last 30 trading days
                            volumes = df['volume'].tail(30)

                            if len(volumes) > 0:
                                avg_volume = float(volumes.mean())
                                volume_data[ticker] = avg_volume
                                _logger.debug("30-day avg volume for %s: %s",
                                            ticker, f"{avg_volume:,.0f}")
                            else:
                                _logger.debug("No volume data for %s", ticker)
                        else:
                            _logger.debug("Empty OHLCV data for %s", ticker)
                    else:
                        _logger.warning("No OHLCV data for %s", ticker)

                # Rate limiting
                if i + self.batch_size < len(tickers):
                    time.sleep(self.api_delay)

            except Exception:
                _logger.exception("Error fetching volume data for batch:")
                continue

        success_count = len(volume_data)
        _logger.info("Volume data fetch completed: %d/%d tickers successful",
                    success_count, len(tickers))

        return volume_data

    def populate_days_to_cover(self, volume_data: Dict[str, float]) -> Dict[str, int]:
        """
        Calculate and populate days_to_cover field using volume data.

        Args:
            volume_data: Dictionary mapping ticker to average volume

        Returns:
            Dictionary with update statistics
        """
        _logger.info("Starting days_to_cover calculation for %d tickers...", len(volume_data))

        if not volume_data:
            _logger.warning("No volume data available for days_to_cover calculation")
            return {'updated_count': 0}

        try:
            updated_count = 0

            with session_scope() as session:
                service = ShortSqueezeService(session)

                # Update each ticker with volume data
                for ticker, avg_volume in volume_data.items():
                    if avg_volume <= 0:
                        continue

                    result = session.execute("""
                        UPDATE ss_finra_short_interest
                        SET
                            days_to_cover = ROUND(short_interest_shares::numeric / %s, 2),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE
                            ticker = %s
                            AND short_interest_shares IS NOT NULL
                            AND short_interest_shares > 0
                            AND days_to_cover IS NULL
                    """, (avg_volume, ticker))

                    updated_count += result.rowcount

                session.commit()

                # Get statistics
                stats_result = session.execute("""
                    SELECT
                        COUNT(*) as total_records,
                        COUNT(days_to_cover) as records_with_days,
                        MIN(days_to_cover) as min_days,
                        MAX(days_to_cover) as max_days,
                        AVG(days_to_cover) as avg_days
                    FROM ss_finra_short_interest
                """).fetchone()

                stats = {
                    'updated_count': updated_count,
                    'total_records': stats_result.total_records,
                    'records_with_days': stats_result.records_with_days,
                    'min_days': float(stats_result.min_days) if stats_result.min_days else None,
                    'max_days': float(stats_result.max_days) if stats_result.max_days else None,
                    'avg_days': float(stats_result.avg_days) if stats_result.avg_days else None,
                }

                _logger.info("Days to cover calculation completed:")
                _logger.info("  Updated records: %d", updated_count)
                _logger.info("  Total records with days_to_cover: %d/%d",
                           stats['records_with_days'], stats['total_records'])
                if stats['avg_days']:
                    _logger.info("  Average days to cover: %.2f days", stats['avg_days'])

                return stats

        except Exception as e:
            _logger.exception("Error calculating days to cover:")
            return {'updated_count': 0, 'error': str(e)}

    def run_full_population(self) -> Dict[str, any]:
        """
        Run the complete field population process.

        Returns:
            Dictionary with complete statistics
        """
        _logger.info("Starting FINRA calculated fields population...")
        start_time = datetime.now()

        results = {
            'start_time': start_time,
            'short_interest_pct': {},
            'days_to_cover': {},
            'volume_data_count': 0
        }

        try:
            # Step 1: Calculate short_interest_pct
            _logger.info("Step 1: Calculating short interest percentages...")
            results['short_interest_pct'] = self.populate_short_interest_pct()

            # Step 2: Get tickers needing volume data
            _logger.info("Step 2: Identifying tickers needing volume data...")
            tickers_needing_volume = self.get_tickers_needing_volume_data()

            if tickers_needing_volume:
                # Step 3: Fetch volume data
                _logger.info("Step 3: Fetching volume data from FMP...")
                volume_data = self.fetch_volume_data(tickers_needing_volume)
                results['volume_data_count'] = len(volume_data)

                # Step 4: Calculate days_to_cover
                _logger.info("Step 4: Calculating days to cover...")
                results['days_to_cover'] = self.populate_days_to_cover(volume_data)
            else:
                _logger.info("No tickers need volume data - skipping steps 3 and 4")
                results['days_to_cover'] = {'updated_count': 0, 'message': 'No tickers needed updates'}

            end_time = datetime.now()
            results['end_time'] = end_time
            results['duration'] = (end_time - start_time).total_seconds()

            _logger.info("FINRA field population completed successfully in %.2f seconds",
                        results['duration'])

            return results

        except Exception as e:
            _logger.exception("Error in full population process:")
            results['error'] = str(e)
            return results


def main():
    """Main entry point."""
    print("FINRA Calculated Fields Populator")
    print("=" * 50)

    populator = FINRAFieldPopulator()
    results = populator.run_full_population()

    # Print summary
    print("\n📊 POPULATION SUMMARY")
    print("-" * 30)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
        return 1

    # Short interest percentage results
    pct_stats = results.get('short_interest_pct', {})
    print(f"Short Interest %: {pct_stats.get('updated_count', 0)} records updated")

    # Days to cover results
    days_stats = results.get('days_to_cover', {})
    print(f"Days to Cover: {days_stats.get('updated_count', 0)} records updated")
    print(f"Volume Data: {results.get('volume_data_count', 0)} tickers fetched")

    if 'duration' in results:
        print(f"Duration: {results['duration']:.2f} seconds")

    print("\n✅ Population completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())