"""
Weekly Screener Module for Short Squeeze Detection Pipeline

This module performs weekly structural analysis to identify short squeeze candidates
based on short interest, days to cover, and other structural metrics.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.ml.pipeline.p04_short_squeeze.config.data_classes import ScreenerConfig
from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics, Candidate, CandidateSource
from src.ml.pipeline.p04_short_squeeze.core.volume_squeeze_detector import create_volume_squeeze_detector
from src.data.downloader.finra_data_downloader import create_finra_downloader
from src.data.db.services.short_squeeze_service import ShortSqueezeService
# FINRA service functionality is now part of ShortSqueezeService

_logger = setup_logger(__name__)


@dataclass
class ScreenerResults:
    """Results from weekly screener run."""
    run_id: str
    run_date: datetime
    total_universe: int
    candidates_found: int
    top_candidates: List[Candidate]
    data_quality_metrics: Dict[str, Any]
    runtime_metrics: Dict[str, Any]


class WeeklyScreener:
    """
    Weekly screener for short squeeze detection.

    Performs structural analysis on the stock universe to identify candidates
    with high short squeeze potential based on short interest and other metrics.
    """

    def __init__(self, fmp_downloader: FMPDataDownloader, config: ScreenerConfig):
        """
        Initialize Weekly Screener with FINRA and volume analysis.

        Args:
            fmp_downloader: FMP data downloader instance
            config: Screener configuration with filters and scoring weights
        """
        self.fmp_downloader = fmp_downloader
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize FINRA downloader
        self.finra_downloader = create_finra_downloader()
        # FINRA service is now accessed through ShortSqueezeService

        # Initialize volume squeeze detector
        self.volume_detector = create_volume_squeeze_detector(fmp_downloader)

        _logger.info("Weekly Screener initialized with run_id=%s (FINRA + Volume Analysis)", self.run_id)

    def run_screener(self, universe: List[str]) -> ScreenerResults:
        """
        Run the weekly screener on the provided universe.

        Args:
            universe: List of ticker symbols to screen

        Returns:
            ScreenerResults with candidates and metrics
        """
        try:
            start_time = datetime.now()
            _logger.info("Starting weekly screener run %s with %d tickers", self.run_id, len(universe))

            # Initialize metrics tracking
            data_quality_metrics = {
                'total_tickers': len(universe),
                'successful_fetches': 0,
                'failed_fetches': 0,
                'valid_short_interest': 0,
                'valid_float_data': 0,
                'api_calls_made': 0
            }

            # Update FINRA data first
            self._update_finra_data(data_quality_metrics)

            # Screen candidates using hybrid approach (FINRA + Volume)
            candidates = self._screen_universe_hybrid(universe, data_quality_metrics)

            # Filter and rank candidates
            filtered_candidates = self._filter_candidates(candidates)
            top_candidates = self._select_top_candidates(filtered_candidates)

            # Calculate runtime metrics
            end_time = datetime.now()
            runtime_metrics = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'tickers_per_second': len(universe) / max((end_time - start_time).total_seconds(), 1),
                'candidates_found': len(candidates),
                'candidates_after_filter': len(filtered_candidates),
                'top_candidates_selected': len(top_candidates)
            }

            # Store results in database
            self._store_results(top_candidates, data_quality_metrics, runtime_metrics)

            results = ScreenerResults(
                run_id=self.run_id,
                run_date=start_time,
                total_universe=len(universe),
                candidates_found=len(candidates),
                top_candidates=top_candidates,
                data_quality_metrics=data_quality_metrics,
                runtime_metrics=runtime_metrics
            )

            _logger.info("Weekly screener completed: %d candidates found, %d top candidates selected",
                        len(candidates), len(top_candidates))

            return results

        except Exception:
            _logger.exception("Error in weekly screener run:")
            raise

    def _update_finra_data(self, metrics: Dict[str, Any]) -> None:
        """
        Update FINRA short interest data.

        Args:
            metrics: Data quality metrics to update
        """
        try:
            _logger.info("Updating FINRA short interest data...")

            # Get latest FINRA data
            finra_df = self.finra_downloader.get_short_interest_data()

            if finra_df is not None and not finra_df.empty:
                # Store in database using ShortSqueezeService
                service = ShortSqueezeService()
                # Convert DataFrame to list of dictionaries for the service
                finra_data_list = []
                for _, row in finra_df.iterrows():
                    finra_data_list.append({
                        'ticker': row.get('Symbol', '').strip().upper(),
                        'settlement_date': row.get('report_date'),
                        'short_interest_shares': int(row.get('ShortVolume', 0)),
                        'raw_data': row.to_dict()
                    })
                records_stored = service.store_finra_data(finra_data_list)

                metrics['finra_records_stored'] = records_stored
                _logger.info("Updated FINRA data: %d records stored", records_stored)
            else:
                _logger.warning("No FINRA data available for update")
                metrics['finra_records_stored'] = 0

        except Exception:
            _logger.exception("Error updating FINRA data:")
            metrics['finra_records_stored'] = 0

    def _screen_universe_hybrid(self, universe: List[str], metrics: Dict[str, Any]) -> List[Candidate]:
        """
        Screen universe using hybrid approach: FINRA short interest + volume analysis.

        Args:
            universe: List of ticker symbols to screen
            metrics: Data quality metrics dictionary to update

        Returns:
            List of candidates from hybrid screening
        """
        try:
            _logger.info("Starting hybrid screening (FINRA + Volume) of %d tickers", len(universe))

            # Step 1: Volume-based screening to get initial candidates
            volume_candidates = self._screen_with_volume_analysis(universe, metrics)

            # Step 2: Enhance with FINRA short interest data
            enhanced_candidates = self._enhance_with_finra_data(volume_candidates, metrics)

            # Step 3: Combine and re-score
            final_candidates = self._combine_and_rescore(enhanced_candidates, metrics)

            _logger.info("Hybrid screening completed: %d candidates found", len(final_candidates))
            return final_candidates

        except Exception:
            _logger.exception("Error in hybrid screening:")
            # Fallback to volume-only screening
            return self._screen_with_volume_analysis(universe, metrics)

    def _screen_with_volume_analysis(self, universe: List[str], metrics: Dict[str, Any]) -> List[Candidate]:
        """
        Screen universe using volume-based squeeze detection.

        Args:
            universe: List of ticker symbols to screen
            metrics: Data quality metrics to update

        Returns:
            List of volume-based candidates
        """
        try:
            _logger.info("Running volume-based screening on %d tickers", len(universe))

            # Use volume detector to screen universe
            volume_results = self.volume_detector.screen_universe(
                universe, min_score=0.3  # Minimum volume score threshold
            )

            # Convert to candidates list
            candidates = []
            for candidate, indicators in volume_results:
                # Update candidate source
                candidate.source = CandidateSource.VOLUME_SCREENER
                candidates.append(candidate)

            # Update metrics
            metrics['volume_screening_processed'] = len(universe)
            metrics['volume_candidates_found'] = len(candidates)

            _logger.info("Volume screening found %d candidates", len(candidates))
            return candidates

        except Exception:
            _logger.exception("Error in volume screening:")
            metrics['volume_screening_processed'] = 0
            metrics['volume_candidates_found'] = 0
            return []

    def _enhance_with_finra_data(self, candidates: List[Candidate], metrics: Dict[str, Any]) -> List[Candidate]:
        """
        Enhance candidates with FINRA short interest data.

        Args:
            candidates: List of volume-based candidates
            metrics: Data quality metrics to update

        Returns:
            List of enhanced candidates
        """
        try:
            if not candidates:
                return candidates

            _logger.info("Enhancing %d candidates with FINRA data", len(candidates))

            # Get symbols from candidates
            symbols = [c.ticker for c in candidates]

            # Get FINRA data for all symbols using ShortSqueezeService
            service = ShortSqueezeService()
            finra_data = service.get_bulk_finra_short_interest(symbols)

            enhanced_candidates = []
            finra_enhanced_count = 0

            for candidate in candidates:
                try:
                    finra_info = finra_data.get(candidate.ticker.upper())

                    if finra_info:
                        # Update structural metrics with FINRA data
                        enhanced_metrics = self._update_structural_metrics_with_finra(
                            candidate.structural_metrics, finra_info
                        )

                        if enhanced_metrics:
                            # Create enhanced candidate
                            enhanced_candidate = Candidate(
                                ticker=candidate.ticker,
                                screener_score=candidate.screener_score,
                                structural_metrics=enhanced_metrics,
                                last_updated=datetime.now(),
                                source=CandidateSource.HYBRID_SCREENER
                            )
                            enhanced_candidates.append(enhanced_candidate)
                            finra_enhanced_count += 1
                        else:
                            # Keep original candidate if enhancement fails
                            enhanced_candidates.append(candidate)
                    else:
                        # No FINRA data available, keep original
                        enhanced_candidates.append(candidate)

                except Exception as e:
                    _logger.warning("Error enhancing candidate %s: %s", candidate.ticker, e)
                    enhanced_candidates.append(candidate)

            # Update metrics
            metrics['finra_enhanced_candidates'] = finra_enhanced_count
            metrics['finra_data_availability'] = len(finra_data)

            _logger.info("Enhanced %d/%d candidates with FINRA data",
                        finra_enhanced_count, len(candidates))

            return enhanced_candidates

        except Exception:
            _logger.exception("Error enhancing with FINRA data:")
            return candidates

    def _update_structural_metrics_with_finra(self, original_metrics: StructuralMetrics,
                                            finra_info: Dict[str, Any]) -> Optional[StructuralMetrics]:
        """
        Update structural metrics with FINRA short interest data.

        Args:
            original_metrics: Original structural metrics
            finra_info: FINRA short interest information

        Returns:
            Updated StructuralMetrics or None if update fails
        """
        try:
            # Extract FINRA data
            short_interest_pct = finra_info.get('estimated_short_interest_pct', 0)
            short_volume = finra_info.get('short_volume', 0)

            # Calculate days to cover using average volume
            avg_volume = original_metrics.avg_volume_14d
            days_to_cover = short_volume / max(avg_volume, 1)

            # Create updated metrics
            updated_metrics = StructuralMetrics(
                short_interest_pct=short_interest_pct,
                days_to_cover=days_to_cover,
                float_shares=original_metrics.float_shares,
                avg_volume_14d=original_metrics.avg_volume_14d,
                market_cap=original_metrics.market_cap
            )

            return updated_metrics

        except Exception as e:
            _logger.warning("Error updating structural metrics with FINRA data: %s", e)
            return None

    def _combine_and_rescore(self, candidates: List[Candidate], metrics: Dict[str, Any]) -> List[Candidate]:
        """
        Combine volume and FINRA scores for final ranking.

        Args:
            candidates: List of enhanced candidates
            metrics: Data quality metrics to update

        Returns:
            List of re-scored candidates
        """
        try:
            if not candidates:
                return candidates

            _logger.info("Re-scoring %d candidates with combined metrics", len(candidates))

            rescored_candidates = []

            for candidate in candidates:
                try:
                    # Calculate combined score
                    volume_score = candidate.screener_score  # Original volume-based score

                    # FINRA-based score components
                    finra_score = 0.0
                    if candidate.structural_metrics.short_interest_pct > 0:
                        # Normalize short interest (cap at 50%)
                        si_normalized = min(candidate.structural_metrics.short_interest_pct / 0.5, 1.0)

                        # Normalize days to cover (cap at 20 days)
                        dtc_normalized = min(candidate.structural_metrics.days_to_cover / 20.0, 1.0)

                        # Combined FINRA score
                        finra_score = (si_normalized * 0.6 + dtc_normalized * 0.4)

                    # Weighted combination (60% volume, 40% FINRA when available)
                    if finra_score > 0:
                        combined_score = volume_score * 0.6 + finra_score * 0.4
                        source = CandidateSource.HYBRID_SCREENER
                    else:
                        combined_score = volume_score
                        source = CandidateSource.VOLUME_SCREENER

                    # Create re-scored candidate
                    rescored_candidate = Candidate(
                        ticker=candidate.ticker,
                        screener_score=combined_score,
                        structural_metrics=candidate.structural_metrics,
                        last_updated=datetime.now(),
                        source=source
                    )

                    rescored_candidates.append(rescored_candidate)

                except Exception as e:
                    _logger.warning("Error re-scoring candidate %s: %s", candidate.ticker, e)
                    rescored_candidates.append(candidate)

            # Sort by combined score
            rescored_candidates.sort(key=lambda c: c.screener_score, reverse=True)

            _logger.info("Re-scoring completed for %d candidates", len(rescored_candidates))
            return rescored_candidates

        except Exception:
            _logger.exception("Error in combine and rescore:")
            return candidates

    def _screen_universe_batch(self, universe: List[str], metrics: Dict[str, Any]) -> List[Candidate]:
        """
        Screen the entire universe using batch processing for efficiency.

        Args:
            universe: List of ticker symbols to screen
            metrics: Data quality metrics dictionary to update

        Returns:
            List of candidates that meet criteria
        """
        try:
            _logger.info("Starting batch screening of %d tickers", len(universe))

            # Use batch method to get all data at once
            batch_data = self.fmp_downloader.get_short_squeeze_batch_data(universe)

            # Update API call metrics
            metrics['api_calls_made'] += len(universe) * 3  # Approximate API calls made

            candidates = []

            for ticker, ticker_data in batch_data.items():
                try:
                    candidate = self._process_ticker_data(ticker, ticker_data, metrics)
                    if candidate:
                        candidates.append(candidate)

                except Exception as e:
                    _logger.warning("Error processing ticker %s: %s", ticker, e)
                    metrics['failed_fetches'] += 1
                    continue

            # Update metrics for tickers that had no data
            missing_tickers = set(universe) - set(batch_data.keys())
            metrics['failed_fetches'] += len(missing_tickers)

            _logger.info("Batch screening completed: %d candidates from %d tickers",
                        len(candidates), len(batch_data))

            return candidates

        except Exception:
            _logger.exception("Error in batch screening:")
            # Fallback to individual screening
            return self._screen_universe_individual(universe, metrics)

    def _process_ticker_data(self, ticker: str, ticker_data: Dict[str, Any],
                           metrics: Dict[str, Any]) -> Optional[Candidate]:
        """
        Process batch ticker data to create candidate.

        Args:
            ticker: Ticker symbol
            ticker_data: Combined data from batch call
            metrics: Data quality metrics to update

        Returns:
            Candidate if ticker meets criteria, None otherwise
        """
        try:
            profile = ticker_data.get('profile')
            short_interest_data = ticker_data.get('shortInterest')

            if not profile or not short_interest_data:
                return None

            # Extract structural metrics from batch data
            structural_metrics = self._extract_structural_metrics_from_batch(
                ticker, profile, short_interest_data
            )

            if not structural_metrics:
                return None

            # Update quality metrics
            metrics['successful_fetches'] += 1
            if structural_metrics.short_interest_pct > 0:
                metrics['valid_short_interest'] += 1
            if structural_metrics.float_shares > 0:
                metrics['valid_float_data'] += 1

            # Check if ticker meets minimum criteria
            if not self._meets_minimum_criteria(structural_metrics):
                return None

            # Calculate screener score
            screener_score = self.calculate_screener_score(structural_metrics)

            # Create candidate
            candidate = Candidate(
                ticker=ticker,
                screener_score=screener_score,
                structural_metrics=structural_metrics,
                last_updated=datetime.now(),
                source=CandidateSource.SCREENER
            )

            _logger.debug("Processed %s: score=%.3f, SI=%.1f%%, DTC=%.1f",
                         ticker, screener_score, structural_metrics.short_interest_pct * 100,
                         structural_metrics.days_to_cover)

            return candidate

        except Exception as e:
            _logger.warning("Error processing ticker data for %s: %s", ticker, e)
            return None

    def _extract_structural_metrics_from_batch(self, ticker: str, profile: Dict[str, Any],
                                             short_data: Dict[str, Any]) -> Optional[StructuralMetrics]:
        """
        Extract structural metrics from batch API data.

        Args:
            ticker: Ticker symbol
            profile: Company profile data
            short_data: Short interest data

        Returns:
            StructuralMetrics if data is valid, None otherwise
        """
        try:
            # Extract short interest
            short_interest = short_data.get('shortInterest', 0)
            shares_outstanding = profile.get('sharesOutstanding', 0)

            if not shares_outstanding or shares_outstanding <= 0:
                return None

            short_interest_pct = short_interest / shares_outstanding

            # Extract float shares
            float_shares = profile.get('floatShares', shares_outstanding)
            if float_shares <= 0:
                float_shares = shares_outstanding

            # Extract market cap
            market_cap = profile.get('mktCap', 0)
            if market_cap <= 0:
                return None

            # Extract volume data from profile (use average volume if available)
            avg_volume_14d = profile.get('volAvg', profile.get('avgVolume', 0))
            if avg_volume_14d <= 0:
                return None

            # Calculate days to cover
            days_to_cover = short_interest / max(avg_volume_14d, 1)

            # Create structural metrics
            structural_metrics = StructuralMetrics(
                short_interest_pct=short_interest_pct,
                days_to_cover=days_to_cover,
                float_shares=int(float_shares),
                avg_volume_14d=int(avg_volume_14d),
                market_cap=int(market_cap)
            )

            return structural_metrics

        except Exception as e:
            _logger.warning("Error extracting structural metrics for %s: %s", ticker, e)
            return None

    def _screen_universe_individual(self, universe: List[str], metrics: Dict[str, Any]) -> List[Candidate]:
        """
        Fallback method: Screen universe using individual API calls.

        Args:
            universe: List of ticker symbols to screen
            metrics: Data quality metrics dictionary to update

        Returns:
            List of candidates
        """
        _logger.warning("Using fallback individual screening method")

        candidates = []
        for i, ticker in enumerate(universe, 1):
            try:
                candidate = self._screen_ticker_individual(ticker, metrics)
                if candidate:
                    candidates.append(candidate)

                # Log progress every 100 tickers
                if i % 100 == 0:
                    _logger.info("Screened %d/%d tickers, found %d candidates",
                               i, len(universe), len(candidates))

            except Exception as e:
                _logger.warning("Error screening ticker %s: %s", ticker, e)
                metrics['failed_fetches'] += 1
                continue

        return candidates

    def _screen_ticker_individual(self, ticker: str, metrics: Dict[str, Any]) -> Optional[Candidate]:
        """
        Screen a single ticker for short squeeze potential.

        Args:
            ticker: Ticker symbol to screen
            metrics: Data quality metrics dictionary to update

        Returns:
            Candidate if ticker meets criteria, None otherwise
        """
        try:
            # Get short interest data
            short_data = self.fmp_downloader.get_short_interest_data(ticker)
            metrics['api_calls_made'] += 1

            if not short_data:
                return None

            # Get float shares data
            float_data = self.fmp_downloader.get_float_shares_data(ticker)
            metrics['api_calls_made'] += 1

            if not float_data:
                return None

            # Get recent volume data for average calculation
            volume_data = self._get_average_volume(ticker)
            metrics['api_calls_made'] += 1

            if not volume_data:
                return None

            # Extract and validate metrics
            structural_metrics = self._extract_structural_metrics(
                ticker, short_data, float_data, volume_data
            )

            if not structural_metrics:
                return None

            # Update quality metrics
            metrics['successful_fetches'] += 1
            if structural_metrics.short_interest_pct > 0:
                metrics['valid_short_interest'] += 1
            if structural_metrics.float_shares > 0:
                metrics['valid_float_data'] += 1

            # Check if ticker meets minimum criteria
            if not self._meets_minimum_criteria(structural_metrics):
                return None

            # Calculate screener score
            screener_score = self.calculate_screener_score(structural_metrics)

            # Create candidate
            candidate = Candidate(
                ticker=ticker,
                screener_score=screener_score,
                structural_metrics=structural_metrics,
                last_updated=datetime.now(),
                source=CandidateSource.SCREENER
            )

            _logger.debug("Screened %s: score=%.3f, SI=%.1f%%, DTC=%.1f",
                         ticker, screener_score, structural_metrics.short_interest_pct * 100,
                         structural_metrics.days_to_cover)

            return candidate

        except Exception as e:
            _logger.warning("Error screening ticker %s: %s", ticker, e)
            return None

    def _extract_structural_metrics(self, ticker: str, short_data: Dict[str, Any],
                                   float_data: Dict[str, Any], volume_data: Dict[str, Any]) -> Optional[StructuralMetrics]:
        """
        Extract structural metrics from API data.

        Args:
            ticker: Ticker symbol
            short_data: Short interest data from FMP
            float_data: Float shares data from FMP
            volume_data: Volume data dictionary

        Returns:
            StructuralMetrics if data is valid, None otherwise
        """
        try:
            # Extract short interest
            short_interest = short_data.get('shortInterest', 0)
            shares_outstanding = float_data.get('sharesOutstanding', 0)

            if not shares_outstanding or shares_outstanding <= 0:
                return None

            short_interest_pct = short_interest / shares_outstanding

            # Extract float shares
            float_shares = float_data.get('floatShares', shares_outstanding)
            if float_shares <= 0:
                float_shares = shares_outstanding

            # Extract market cap
            market_cap = float_data.get('marketCap', 0)
            if market_cap <= 0:
                return None

            # Extract volume data
            avg_volume_14d = volume_data.get('avg_volume', 0)
            if avg_volume_14d <= 0:
                return None

            # Calculate days to cover
            days_to_cover = short_interest / max(avg_volume_14d, 1)

            # Create structural metrics
            structural_metrics = StructuralMetrics(
                short_interest_pct=short_interest_pct,
                days_to_cover=days_to_cover,
                float_shares=int(float_shares),
                avg_volume_14d=int(avg_volume_14d),
                market_cap=int(market_cap)
            )

            return structural_metrics

        except Exception as e:
            _logger.warning("Error extracting structural metrics for %s: %s", ticker, e)
            return None

    def _get_average_volume(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get 14-day average volume for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Dictionary with volume data or None if failed
        """
        try:
            # Get 14 days of volume data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=20)  # Get extra days to ensure 14 trading days

            df = self.fmp_downloader.get_ohlcv(ticker, '1d', start_date, end_date)

            if df is None or df.empty:
                return None

            # Calculate average volume over the last 14 trading days
            recent_volumes = df['volume'].tail(14)
            if len(recent_volumes) < 7:  # Need at least 7 days of data
                return None

            avg_volume = recent_volumes.mean()

            return {
                'avg_volume': avg_volume,
                'days_of_data': len(recent_volumes)
            }

        except Exception as e:
            _logger.warning("Error getting average volume for %s: %s", ticker, e)
            return None

    def _meets_minimum_criteria(self, metrics: StructuralMetrics) -> bool:
        """
        Check if structural metrics meet minimum criteria for consideration.

        Args:
            metrics: Structural metrics to check

        Returns:
            True if metrics meet minimum criteria, False otherwise
        """
        try:
            # Check short interest percentage
            if metrics.short_interest_pct < self.config.filters.si_percent_min:
                return False

            # Check days to cover
            if metrics.days_to_cover < self.config.filters.days_to_cover_min:
                return False

            # Check float size (exclude very large floats)
            if metrics.float_shares > self.config.filters.float_max:
                return False

            return True

        except Exception as e:
            _logger.warning("Error checking minimum criteria: %s", e)
            return False

    def calculate_screener_score(self, metrics: StructuralMetrics) -> float:
        """
        Calculate screener score based on structural metrics.

        Args:
            metrics: Structural metrics

        Returns:
            Screener score between 0 and 1
        """
        try:
            # Normalize metrics to 0-1 scale
            normalized_metrics = self._normalize_metrics(metrics)

            # Apply weights
            weighted_score = (
                normalized_metrics['short_interest_pct'] * self.config.scoring.short_interest_pct +
                normalized_metrics['days_to_cover'] * self.config.scoring.days_to_cover +
                normalized_metrics['float_ratio'] * self.config.scoring.float_ratio +
                normalized_metrics['volume_consistency'] * self.config.scoring.volume_consistency
            )

            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, weighted_score))

            return score

        except Exception as e:
            _logger.warning("Error calculating screener score: %s", e)
            return 0.0

    def _normalize_metrics(self, metrics: StructuralMetrics) -> Dict[str, float]:
        """
        Normalize structural metrics to 0-1 scale.

        Args:
            metrics: Structural metrics to normalize

        Returns:
            Dictionary of normalized metrics
        """
        try:
            # Short interest percentage (already 0-1, but cap at reasonable max)
            si_normalized = min(metrics.short_interest_pct / 0.5, 1.0)  # Cap at 50%

            # Days to cover (normalize using sigmoid-like function)
            dtc_normalized = min(metrics.days_to_cover / 20.0, 1.0)  # Cap at 20 days

            # Float ratio (smaller float is better for squeezes)
            # Use inverse relationship - smaller float gets higher score
            max_float = 100_000_000  # 100M shares
            float_ratio = max(0.0, 1.0 - (metrics.float_shares / max_float))

            # Volume consistency (higher volume is better for liquidity)
            # This is a placeholder - in practice, you might want to compare
            # against market averages or use more sophisticated metrics
            min_volume = 100_000
            max_volume = 10_000_000
            volume_normalized = min(
                max(0.0, (metrics.avg_volume_14d - min_volume) / (max_volume - min_volume)),
                1.0
            )

            return {
                'short_interest_pct': si_normalized,
                'days_to_cover': dtc_normalized,
                'float_ratio': float_ratio,
                'volume_consistency': volume_normalized
            }

        except Exception as e:
            _logger.warning("Error normalizing metrics: %s", e)
            return {
                'short_interest_pct': 0.0,
                'days_to_cover': 0.0,
                'float_ratio': 0.0,
                'volume_consistency': 0.0
            }

    def _filter_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Apply additional filtering to candidates.

        Args:
            candidates: List of candidates to filter

        Returns:
            Filtered list of candidates
        """
        try:
            if not candidates:
                return []

            _logger.info("Filtering %d candidates", len(candidates))

            # Sort by screener score
            sorted_candidates = sorted(candidates, key=lambda c: c.screener_score, reverse=True)

            # Apply score threshold (keep candidates with score > 0.3)
            score_threshold = 0.3
            filtered_candidates = [c for c in sorted_candidates if c.screener_score >= score_threshold]

            _logger.info("After filtering: %d candidates remain", len(filtered_candidates))
            return filtered_candidates

        except Exception:
            _logger.exception("Error filtering candidates:")
            return candidates

    def _select_top_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Select top K candidates for daily deep scan.

        Args:
            candidates: List of filtered candidates

        Returns:
            Top K candidates
        """
        try:
            if not candidates:
                return []

            # Sort by screener score (descending)
            sorted_candidates = sorted(candidates, key=lambda c: c.screener_score, reverse=True)

            # Select top K
            top_k = self.config.filters.top_k_candidates
            top_candidates = sorted_candidates[:top_k]

            _logger.info("Selected top %d candidates from %d filtered candidates",
                        len(top_candidates), len(candidates))

            return top_candidates

        except Exception:
            _logger.exception("Error selecting top candidates:")
            return candidates[:self.config.filters.top_k_candidates]

    def _store_results(self, candidates: List[Candidate], data_quality_metrics: Dict[str, Any],
                      runtime_metrics: Dict[str, Any]) -> None:
        """
        Store screener results in database.

        Args:
            candidates: List of top candidates
            data_quality_metrics: Data quality metrics
            runtime_metrics: Runtime performance metrics
        """
        try:
            service = ShortSqueezeService()

            # Convert candidates to dict format
            results = []
            for candidate in candidates:
                results.append({
                    'ticker': candidate.ticker,
                    'screener_score': candidate.screener_score,
                    'short_interest_pct': candidate.structural_metrics.short_interest_pct,
                    'days_to_cover': candidate.structural_metrics.days_to_cover,
                    'market_cap': candidate.structural_metrics.market_cap,
                    'float_shares': candidate.structural_metrics.float_shares,
                    'avg_volume_14d': candidate.structural_metrics.avg_volume_14d,
                    'data_quality': data_quality_metrics.get('successful_fetches', 0) / max(data_quality_metrics.get('total_tickers', 1), 1)
                })

            # Store screener snapshot
            service.save_screener_results(results=results, run_date=datetime.now().date())

            _logger.info("Stored screener results in database: %d candidates", len(candidates))

        except Exception:
            _logger.exception("Error storing screener results:")
            raise


def create_weekly_screener(fmp_downloader: FMPDataDownloader,
                          config: ScreenerConfig) -> WeeklyScreener:
    """
    Factory function to create Weekly Screener.

    Args:
        fmp_downloader: FMP data downloader instance
        config: Screener configuration

    Returns:
        Configured Weekly Screener instance
    """
    return WeeklyScreener(fmp_downloader, config)


# Example usage
if __name__ == "__main__":
    from src.ml.pipeline.p04_short_squeeze.config.data_classes import ScreenerConfig, UniverseConfig
    from src.ml.pipeline.p04_short_squeeze.core.universe_loader import create_universe_loader

    # Create FMP downloader
    fmp_downloader = FMPDataDownloader()

    # Create configurations
    universe_config = UniverseConfig()
    screener_config = ScreenerConfig()

    # Test connection
    if fmp_downloader.test_connection():
        print("✅ FMP API connection successful")

        # Load universe
        universe_loader = create_universe_loader(fmp_downloader, universe_config)
        universe = universe_loader.load_universe()

        if universe:
            print(f"✅ Loaded universe of {len(universe)} stocks")

            # Create screener
            screener = create_weekly_screener(fmp_downloader, screener_config)

            # Run screener on a small sample for testing
            test_universe = universe[:10]  # Test with first 10 tickers
            print(f"Testing screener with {len(test_universe)} tickers")

            results = screener.run_screener(test_universe)

            print("✅ Screener completed:")
            print(f"  - Candidates found: {results.candidates_found}")
            print(f"  - Top candidates: {len(results.top_candidates)}")
            print(f"  - Runtime: {results.runtime_metrics['duration_seconds']:.2f} seconds")

            # Show top candidates
            for i, candidate in enumerate(results.top_candidates[:5], 1):
                print(f"  {i}. {candidate.ticker}: score={candidate.screener_score:.3f}, "
                      f"SI={candidate.structural_metrics.short_interest_pct*100:.1f}%, "
                      f"DTC={candidate.structural_metrics.days_to_cover:.1f}")
        else:
            print("❌ Failed to load universe")
    else:
        print("❌ FMP API connection failed")