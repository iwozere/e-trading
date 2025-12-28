"""
Daily Deep Scan Module for Short Squeeze Detection Pipeline

This module performs daily analysis on identified candidates with real-time metrics
including volume spikes, sentiment analysis, and options data.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from dataclasses import dataclass

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.ml.pipeline.p04_short_squeeze.config.data_classes import DeepScanConfig, SentimentConfig
from src.ml.pipeline.p04_short_squeeze.core.models import (
    Candidate, TransientMetrics, ScoredCandidate, CandidateSource
)
from src.data.db.services.short_squeeze_service import ShortSqueezeService

# Sentiment module import (with feature flag)
try:
    from src.common.sentiments.collect_sentiment_async import collect_sentiment_batch
    SENTIMENT_MODULE_AVAILABLE = True
except ImportError:
    _logger.warning("Sentiment module not available, will use legacy Finnhub sentiment")
    SENTIMENT_MODULE_AVAILABLE = False

_logger = setup_logger(__name__)


@dataclass
class DeepScanResults:
    """Results from daily deep scan run."""
    run_id: str
    run_date: date
    candidates_processed: int
    scored_candidates: List[ScoredCandidate]
    data_quality_metrics: Dict[str, Any]
    runtime_metrics: Dict[str, Any]


class DailyDeepScan:
    """
    Daily deep scan for short squeeze detection.

    Performs focused analysis on previously identified candidates using
    real-time transient metrics like volume spikes, sentiment, and options data.
    """

    def __init__(self, fmp_downloader: FMPDataDownloader,
                 finnhub_downloader: FinnhubDataDownloader,
                 config: DeepScanConfig,
                 sentiment_config: Optional[SentimentConfig] = None):
        """
        Initialize Daily Deep Scan.

        Args:
            fmp_downloader: FMP data downloader instance
            finnhub_downloader: Finnhub data downloader instance
            config: Deep scan configuration with metrics and scoring weights
            sentiment_config: Optional sentiment module configuration
        """
        self.fmp_downloader = fmp_downloader
        self.finnhub_downloader = finnhub_downloader
        self.config = config
        self.sentiment_config = sentiment_config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Feature flag for enhanced sentiment
        self.use_enhanced_sentiment = (
            SENTIMENT_MODULE_AVAILABLE and
            sentiment_config is not None
        )

        if self.use_enhanced_sentiment:
            _logger.info("Daily Deep Scan initialized with enhanced multi-source sentiment (run_id=%s)", self.run_id)
        else:
            _logger.info("Daily Deep Scan initialized with legacy Finnhub sentiment (run_id=%s)", self.run_id)

    def run_deep_scan(self, candidates: Optional[List[Candidate]] = None) -> DeepScanResults:
        """
        Run the daily deep scan on active candidates.

        Args:
            candidates: Optional list of candidates to scan. If None, loads from database

        Returns:
            DeepScanResults with scored candidates and metrics
        """
        try:
            start_time = datetime.now()
            scan_date = start_time.date()

            _logger.info("Starting daily deep scan run %s for date %s", self.run_id, scan_date)

            # Load candidates if not provided
            if candidates is None:
                candidates = self._load_active_candidates()

            if not candidates:
                _logger.warning("No active candidates found for deep scan")
                return DeepScanResults(
                    run_id=self.run_id,
                    run_date=scan_date,
                    candidates_processed=0,
                    scored_candidates=[],
                    data_quality_metrics={},
                    runtime_metrics={}
                )

            _logger.info("Processing %d candidates for deep scan", len(candidates))

            # Initialize metrics tracking
            data_quality_metrics = {
                'total_candidates': len(candidates),
                'successful_scans': 0,
                'failed_scans': 0,
                'valid_volume_data': 0,
                'valid_sentiment_data': 0,
                'valid_options_data': 0,
                'valid_borrow_rates': 0,
                'finra_data_available': 0,
                'api_calls_fmp': 0,
                'api_calls_finnhub': 0
            }

            # Process candidates in batches
            scored_candidates = []
            batch_size = self.config.batch_size

            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                batch_results = self._process_batch(batch, data_quality_metrics)
                scored_candidates.extend(batch_results)

                _logger.info("Processed batch %d-%d: %d candidates scored",
                           i + 1, min(i + batch_size, len(candidates)), len(batch_results))

                # Add delay between batches to respect API rate limits
                if i + batch_size < len(candidates):
                    import time
                    time.sleep(self.config.api_delay_seconds)

            # Store results in database
            self._store_results(scored_candidates, scan_date, data_quality_metrics)

            # Calculate runtime metrics
            end_time = datetime.now()
            runtime_metrics = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'candidates_per_second': len(candidates) / max((end_time - start_time).total_seconds(), 1),
                'successful_scans': data_quality_metrics['successful_scans'],
                'failed_scans': data_quality_metrics['failed_scans']
            }

            results = DeepScanResults(
                run_id=self.run_id,
                run_date=scan_date,
                candidates_processed=len(candidates),
                scored_candidates=scored_candidates,
                data_quality_metrics=data_quality_metrics,
                runtime_metrics=runtime_metrics
            )

            _logger.info("Daily deep scan completed: %d candidates processed, %d scored",
                        len(candidates), len(scored_candidates))

            return results

        except Exception:
            _logger.exception("Error in daily deep scan run:")
            raise

    def _load_active_candidates(self) -> List[Candidate]:
        """
        Load active candidates from database for deep scan.

        Returns:
            List of active candidates
        """
        try:
            service = ShortSqueezeService()

            # Get candidates from latest screener run
            screener_candidates = service.get_candidates_for_deep_scan()

            # Get active ad-hoc candidates
            adhoc_candidates = service.get_active_adhoc_candidates()

            # Combine and convert to Candidate objects
            all_candidates = []

            # Convert screener candidates
            for db_candidate in screener_candidates:
                candidate = self._convert_db_candidate_to_model(db_candidate)
                if candidate:
                    all_candidates.append(candidate)

            # Convert ad-hoc candidates
            for db_adhoc in adhoc_candidates:
                candidate = self._convert_db_adhoc_to_candidate(db_adhoc)
                if candidate:
                    all_candidates.append(candidate)

            _logger.info("Loaded %d active candidates (%d from screener, %d ad-hoc)",
                       len(all_candidates), len(screener_candidates), len(adhoc_candidates))

            return all_candidates

        except Exception:
            _logger.exception("Error loading active candidates:")
            return []

    def _convert_db_candidate_to_model(self, db_candidate) -> Optional[Candidate]:
        """Convert database candidate to model Candidate."""
        try:
            from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics

            # Handle both dictionary and object formats
            if isinstance(db_candidate, dict):
                # Dictionary format from service
                ticker = db_candidate.get('ticker', 'Unknown')
                screener_score = db_candidate.get('screener_score', 0.0)
                short_interest_pct = db_candidate.get('short_interest_pct', 0.0)
                days_to_cover = db_candidate.get('days_to_cover', 0.0)
                market_cap = db_candidate.get('market_cap', 0)
            else:
                # Object format from database model
                ticker = db_candidate.ticker
                screener_score = float(db_candidate.screener_score) if db_candidate.screener_score else 0.0
                short_interest_pct = float(db_candidate.short_interest_pct) if db_candidate.short_interest_pct else 0.0
                days_to_cover = float(db_candidate.days_to_cover) if db_candidate.days_to_cover else 0.0
                market_cap = int(db_candidate.market_cap) if db_candidate.market_cap else 0

            # Create structural metrics with safe defaults
            structural_metrics = StructuralMetrics(
                short_interest_pct=float(short_interest_pct) if short_interest_pct else 0.0,
                days_to_cover=float(days_to_cover) if days_to_cover else 0.0,
                float_shares=int(market_cap / 100) if market_cap else 1000000,  # Estimate float from market cap
                avg_volume_14d=100000,  # Default volume
                market_cap=int(market_cap) if market_cap else 0
            )

            candidate = Candidate(
                ticker=ticker,
                screener_score=float(screener_score) if screener_score else 0.0,
                structural_metrics=structural_metrics,
                last_updated=datetime.now(),  # Use current time as default
                source=CandidateSource.SCREENER
            )

            return candidate

        except Exception as e:
            _logger.warning("Error converting DB candidate %s: %s",
                          ticker if 'ticker' in locals() else 'Unknown', e)
            return None

    def _convert_db_adhoc_to_candidate(self, db_adhoc) -> Optional[Candidate]:
        """Convert database ad-hoc candidate to model Candidate."""
        try:
            # For ad-hoc candidates, we need to create placeholder structural metrics
            # These will be updated during the deep scan process
            from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics

            # Handle both dictionary and object formats
            if isinstance(db_adhoc, dict):
                ticker = db_adhoc.get('ticker', 'Unknown')
            else:
                ticker = db_adhoc.ticker

            structural_metrics = StructuralMetrics(
                short_interest_pct=0.0,  # Will be updated during scan
                days_to_cover=0.0,
                float_shares=1000000,  # Default float
                avg_volume_14d=100000,  # Default volume
                market_cap=1
            )

            candidate = Candidate(
                ticker=ticker,
                screener_score=0.0,  # Ad-hoc candidates start with 0 score
                structural_metrics=structural_metrics,
                last_updated=datetime.now(),  # Use current time as default
                source=CandidateSource.ADHOC
            )

            return candidate

        except Exception as e:
            _logger.warning("Error converting DB ad-hoc candidate %s: %s",
                          ticker if 'ticker' in locals() else 'Unknown', e)
            return None

    def _enhance_candidate_with_finra_data(self, candidate: Candidate,
                                         metrics: Dict[str, Any]) -> Candidate:
        """
        Enhance candidate with latest FINRA data from database.

        Args:
            candidate: Original candidate
            metrics: Data quality metrics dictionary to update

        Returns:
            Enhanced candidate with updated FINRA data or original if no data available
        """
        try:
            ticker = candidate.ticker

            # Get latest FINRA data from database
            service = ShortSqueezeService()
            finra_data = service.get_latest_finra_short_interest(ticker)

            if not finra_data:
                _logger.debug("No FINRA data available for %s, using original candidate", ticker)
                return candidate

            # Update metrics tracking
            metrics['finra_data_available'] = metrics.get('finra_data_available', 0) + 1

            # Extract FINRA metrics
            short_interest_pct = finra_data.get('short_interest_pct', 0.0)
            if short_interest_pct is None:
                short_interest_pct = 0.0
            else:
                # Convert from percentage (0-100) to ratio (0-1) if needed
                if short_interest_pct > 1.0:
                    short_interest_pct = short_interest_pct / 100.0

            days_to_cover = finra_data.get('days_to_cover', 0.0)
            if days_to_cover is None:
                days_to_cover = 0.0

            short_interest_shares = finra_data.get('short_interest_shares', 0)
            data_age_days = finra_data.get('data_age_days', 0)

            # Create enhanced structural metrics
            from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics

            enhanced_metrics = StructuralMetrics(
                short_interest_pct=float(short_interest_pct),
                days_to_cover=float(days_to_cover),
                float_shares=candidate.structural_metrics.float_shares,  # Keep original
                avg_volume_14d=candidate.structural_metrics.avg_volume_14d,  # Keep original
                market_cap=candidate.structural_metrics.market_cap  # Keep original
            )

            # Create enhanced candidate
            enhanced_candidate = Candidate(
                ticker=candidate.ticker,
                screener_score=candidate.screener_score,
                structural_metrics=enhanced_metrics,
                last_updated=datetime.now(),
                source=candidate.source
            )

            _logger.debug("Enhanced %s with FINRA data: SI=%.1f%%, DTC=%.1f, age=%d days",
                         ticker, short_interest_pct * 100, days_to_cover, data_age_days)

            return enhanced_candidate

        except Exception as e:
            _logger.warning("Error enhancing candidate %s with FINRA data: %s",
                          candidate.ticker, e)
            return candidate

    async def _get_historical_mentions_async(self, ticker: str) -> Optional[float]:
        """
        Get 7-day average mention count for growth calculation.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Average mentions over past 7 days, or None if insufficient data
        """
        try:
            # This would ideally be a service method, but for now we can access via UoW
            # TODO: Move this query to ShortSqueezeService
            service = ShortSqueezeService()

            # For now, return None - this needs to be implemented in the service layer
            # The sentiment collection will work without historical data
            return None

        except Exception as e:
            _logger.warning("Failed to get historical mentions for %s: %s", ticker, e)
            return None

    def _collect_batch_sentiment(self, candidates: List[Candidate],
                                 metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect sentiment data for a batch of candidates using multi-source sentiment module.

        Args:
            candidates: List of candidates to collect sentiment for
            metrics: Data quality metrics dictionary to update

        Returns:
            Dictionary mapping ticker -> SentimentFeatures
        """
        try:
            import asyncio

            ticker_list = [c.ticker for c in candidates]
            _logger.info("Collecting batch sentiment for %d tickers...", len(ticker_list))

            # Build sentiment config from dataclass
            sentiment_config_dict = None
            if self.sentiment_config:
                sentiment_config_dict = {
                    'providers': {
                        'stocktwits': self.sentiment_config.providers.stocktwits,
                        'reddit_pushshift': self.sentiment_config.providers.reddit_pushshift,
                        'news': self.sentiment_config.providers.news,
                        'google_trends': self.sentiment_config.providers.google_trends,
                        'twitter': self.sentiment_config.providers.twitter,
                        'discord': self.sentiment_config.providers.discord,
                        'hf_enabled': self.sentiment_config.providers.hf_enabled
                    },
                    'batching': {
                        'concurrency': self.sentiment_config.batching.concurrency,
                        'rate_limit_delay_sec': self.sentiment_config.batching.rate_limit_delay_sec
                    },
                    'weights': {
                        'stocktwits': self.sentiment_config.weights.stocktwits,
                        'reddit': self.sentiment_config.weights.reddit,
                        'news': self.sentiment_config.weights.news,
                        'google_trends': self.sentiment_config.weights.google_trends,
                        'heuristic_vs_hf': self.sentiment_config.weights.heuristic_vs_hf
                    },
                    'cache': {
                        'enabled': self.sentiment_config.cache.enabled,
                        'ttl_seconds': self.sentiment_config.cache.ttl_seconds,
                        'redis_enabled': self.sentiment_config.cache.redis_enabled
                    }
                }

            # Run async sentiment collection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                sentiment_map = loop.run_until_complete(
                    collect_sentiment_batch(
                        tickers=ticker_list,
                        lookback_hours=24,
                        config=sentiment_config_dict,
                        history_lookup=self._get_historical_mentions_async,
                        output_format="dataclass"
                    )
                )
            finally:
                loop.close()

            # Count successes and update metrics
            success_count = sum(1 for v in sentiment_map.values() if v)
            _logger.info("Batch sentiment collection complete: %d/%d successful",
                        success_count, len(ticker_list))

            # Log notable signals
            for ticker, features in sentiment_map.items():
                if features:
                    if features.virality_index > 0.7:
                        _logger.info("  %s: HIGH VIRALITY detected (%.2f)", ticker, features.virality_index)
                    if features.mentions_growth_7d and features.mentions_growth_7d > 2.0:
                        _logger.info("  %s: MENTION SURGE detected (+%.0f%%)",
                                   ticker, features.mentions_growth_7d * 100)
                    if features.bot_pct > 0.5:
                        _logger.warning("  %s: High bot activity detected (%.0f%%)",
                                      ticker, features.bot_pct * 100)

            # Update metrics (API calls are harder to track in batch, use estimate)
            metrics['api_calls_sentiment_batch'] = metrics.get('api_calls_sentiment_batch', 0) + 1

            return sentiment_map

        except Exception as e:
            _logger.exception("Batch sentiment collection failed: %s", e)
            return {}  # Return empty map, will use fallback

    def _process_batch(self, candidates: List[Candidate],
                      metrics: Dict[str, Any]) -> List[ScoredCandidate]:
        """
        Process a batch of candidates for deep scan.

        Args:
            candidates: List of candidates to process
            metrics: Data quality metrics dictionary to update

        Returns:
            List of scored candidates
        """
        try:
            scored_candidates = []

            # BATCH SENTIMENT COLLECTION (if enhanced sentiment enabled)
            sentiment_map = {}
            if self.use_enhanced_sentiment:
                sentiment_map = self._collect_batch_sentiment(candidates, metrics)

            # Process each candidate with pre-fetched sentiment data
            for candidate in candidates:
                try:
                    # Pass sentiment data to scanner
                    scored_candidate = self._scan_candidate(
                        candidate, metrics, sentiment_map.get(candidate.ticker)
                    )
                    if scored_candidate:
                        scored_candidates.append(scored_candidate)
                        metrics['successful_scans'] += 1
                    else:
                        metrics['failed_scans'] += 1

                except Exception as e:
                    _logger.warning("Error scanning candidate %s: %s", candidate.ticker, e)
                    metrics['failed_scans'] += 1
                    continue

            return scored_candidates

        except Exception:
            _logger.exception("Error processing batch:")
            return []

    def _scan_candidate(self, candidate: Candidate,
                       metrics: Dict[str, Any],
                       sentiment_features: Optional[Any] = None) -> Optional[ScoredCandidate]:
        """
        Perform deep scan on a single candidate.

        Args:
            candidate: Candidate to scan
            metrics: Data quality metrics dictionary to update
            sentiment_features: Optional pre-fetched sentiment features from batch collection

        Returns:
            ScoredCandidate if successful, None otherwise
        """
        try:
            ticker = candidate.ticker
            _logger.debug("Deep scanning candidate: %s", ticker)

            # Enhance candidate with latest FINRA data from database
            enhanced_candidate = self._enhance_candidate_with_finra_data(candidate, metrics)

            # Calculate transient metrics (with pre-fetched sentiment if available)
            transient_metrics = self.calculate_transient_metrics(ticker, metrics, sentiment_features)

            if not transient_metrics:
                _logger.warning("Failed to calculate transient metrics for %s", ticker)
                return None

            # Calculate final squeeze score using enhanced candidate data
            squeeze_score = self._calculate_preliminary_squeeze_score(
                enhanced_candidate.screener_score, transient_metrics, enhanced_candidate.structural_metrics
            )

            # Create scored candidate with enhanced data
            scored_candidate = ScoredCandidate(
                candidate=enhanced_candidate,
                transient_metrics=transient_metrics,
                squeeze_score=squeeze_score,
                alert_level=None  # Will be determined by alert engine
            )

            _logger.debug("Scanned %s: squeeze_score=%.3f, volume=%.2f, sentiment=%.2f, virality=%.2f, SI=%.1f%%, DTC=%.1f",
                         ticker, squeeze_score, transient_metrics.volume_spike,
                         transient_metrics.sentiment_24h, transient_metrics.virality_index,
                         enhanced_candidate.structural_metrics.short_interest_pct * 100,
                         enhanced_candidate.structural_metrics.days_to_cover)

            return scored_candidate

        except Exception as e:
            _logger.warning("Error scanning candidate %s: %s", candidate.ticker, e)
            return None

    def calculate_transient_metrics(self, ticker: str,
                                   metrics: Dict[str, Any],
                                   sentiment_features: Optional[Any] = None) -> Optional[TransientMetrics]:
        """
        Calculate transient metrics for a ticker.

        Args:
            ticker: Ticker symbol
            metrics: Data quality metrics dictionary to update
            sentiment_features: Optional pre-fetched sentiment features from batch collection

        Returns:
            TransientMetrics if successful, None otherwise
        """
        try:
            # Calculate volume spike ratio
            volume_spike = self._calculate_volume_spike_ratio(ticker)
            if volume_spike is not None:
                metrics['valid_volume_data'] += 1
                metrics['api_calls_fmp'] += 1

            # Enhanced sentiment metrics (if available from batch collection)
            if sentiment_features:
                sentiment_24h = sentiment_features.sentiment_normalized
                mentions_24h = sentiment_features.mentions_24h
                mentions_growth_7d = sentiment_features.mentions_growth_7d
                virality_index = sentiment_features.virality_index
                bot_pct = sentiment_features.bot_pct
                sentiment_data_quality = sentiment_features.data_quality

                metrics['valid_sentiment_data'] += 1
                # API calls already counted in batch collection
            else:
                # Fallback to legacy Finnhub sentiment
                sentiment_24h = self._get_sentiment_score(ticker)
                mentions_24h = 0
                mentions_growth_7d = None
                virality_index = 0.0
                bot_pct = 0.0
                sentiment_data_quality = {}

                if sentiment_24h is not None:
                    metrics['valid_sentiment_data'] += 1
                    metrics['api_calls_finnhub'] += 1

            # Get call/put ratio
            call_put_ratio = self._get_call_put_ratio(ticker)
            if call_put_ratio is not None:
                metrics['valid_options_data'] += 1
                metrics['api_calls_finnhub'] += 1

            # Get borrow fee percentage
            borrow_fee_pct = self._get_borrow_fee_percentage(ticker)
            if borrow_fee_pct is not None:
                metrics['valid_borrow_rates'] += 1
                metrics['api_calls_finnhub'] += 1

            # Create transient metrics with enhanced sentiment fields
            transient_metrics = TransientMetrics(
                volume_spike=volume_spike or 1.0,  # Default to 1.0 (no spike)
                call_put_ratio=call_put_ratio,
                sentiment_24h=sentiment_24h or 0.0,  # Default to neutral
                borrow_fee_pct=borrow_fee_pct,
                # Enhanced sentiment metrics
                mentions_24h=mentions_24h,
                mentions_growth_7d=mentions_growth_7d,
                virality_index=virality_index,
                bot_pct=bot_pct,
                sentiment_data_quality=sentiment_data_quality
            )

            return transient_metrics

        except Exception as e:
            _logger.warning("Error calculating transient metrics for %s: %s", ticker, e)
            return None

    def _calculate_volume_spike_ratio(self, ticker: str) -> Optional[float]:
        """
        Calculate volume spike ratio (current vs 14-day average).

        Args:
            ticker: Ticker symbol

        Returns:
            Volume spike ratio or None if calculation failed
        """
        try:
            # Get recent volume data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=20)  # Get extra days for 14 trading days

            df = self.fmp_downloader.get_ohlcv(ticker, '1d', start_date, end_date)

            if df is None or df.empty or len(df) < 15:
                _logger.warning("Insufficient volume data for %s", ticker)
                return None

            # Get current day volume (most recent)
            current_volume = df['volume'].iloc[-1]

            # Calculate 14-day average volume (excluding current day)
            avg_volume_14d = df['volume'].iloc[-15:-1].mean()

            if avg_volume_14d <= 0:
                return None

            volume_spike = current_volume / avg_volume_14d

            _logger.debug("Volume spike for %s: current=%d, avg_14d=%d, ratio=%.2f",
                         ticker, current_volume, avg_volume_14d, volume_spike)

            return volume_spike

        except Exception as e:
            _logger.warning("Error calculating volume spike for %s: %s", ticker, e)
            return None

    def _get_sentiment_score(self, ticker: str) -> Optional[float]:
        """
        Get 24-hour sentiment score.

        Args:
            ticker: Ticker symbol

        Returns:
            Sentiment score (-1 to 1) or None if failed
        """
        try:
            if self.finnhub_downloader is None:
                return None  # Finnhub not available

            sentiment_data = self.finnhub_downloader.aggregate_24h_sentiment(ticker)

            if not sentiment_data:
                return None

            sentiment_score = sentiment_data.get('sentiment_score_24h', 0.0)

            _logger.debug("Sentiment for %s: score=%.3f", ticker, sentiment_score)

            return sentiment_score

        except Exception as e:
            _logger.warning("Error getting sentiment for %s: %s", ticker, e)
            return None

    def _get_call_put_ratio(self, ticker: str) -> Optional[float]:
        """
        Get call-to-put ratio from options data.

        Args:
            ticker: Ticker symbol

        Returns:
            Call/put ratio or None if failed
        """
        try:
            if self.finnhub_downloader is None:
                return None  # Finnhub not available

            options_data = self.finnhub_downloader.get_options_data(ticker)

            if not options_data:
                return None

            call_put_ratio = self.finnhub_downloader.calculate_call_put_ratio(options_data)

            if call_put_ratio is not None:
                _logger.debug("Call/put ratio for %s: %.3f", ticker, call_put_ratio)

            return call_put_ratio

        except Exception as e:
            _logger.warning("Error getting call/put ratio for %s: %s", ticker, e)
            return None

    def _get_borrow_fee_percentage(self, ticker: str) -> Optional[float]:
        """
        Get stock borrow fee percentage.

        Args:
            ticker: Ticker symbol

        Returns:
            Borrow fee percentage or None if failed
        """
        try:
            if self.finnhub_downloader is None:
                return None  # Finnhub not available

            borrow_data = self.finnhub_downloader.get_borrow_rates_data(ticker)

            if not borrow_data:
                return None

            borrow_fee = borrow_data.get('fee_rate_percentage')

            if borrow_fee is not None:
                _logger.debug("Borrow fee for %s: %.3f%%", ticker, borrow_fee)

            return borrow_fee

        except Exception as e:
            _logger.warning("Error getting borrow fee for %s: %s", ticker, e)
            return None

    def _calculate_preliminary_squeeze_score(self, screener_score: float,
                                           transient_metrics: TransientMetrics,
                                           structural_metrics: 'StructuralMetrics') -> float:
        """
        Calculate preliminary squeeze score combining screener, transient, and FINRA metrics.

        Note: This is a simplified implementation. The full scoring will be done
        by the dedicated Scoring Engine module.

        Args:
            screener_score: Score from weekly screener
            transient_metrics: Transient metrics from deep scan
            structural_metrics: Enhanced structural metrics including FINRA data

        Returns:
            Preliminary squeeze score (0-1)
        """
        try:
            # Simple weighted combination (will be replaced by proper scoring engine)
            weights = self.config.scoring

            # Normalize transient metrics
            volume_score = min(transient_metrics.volume_spike / 5.0, 1.0)  # Cap at 5x volume
            sentiment_score = (transient_metrics.sentiment_24h + 1) / 2  # Convert -1,1 to 0,1

            # Call/put ratio score (higher ratio = more bullish = higher squeeze potential)
            callput_score = 0.5  # Default neutral
            if transient_metrics.call_put_ratio is not None:
                callput_score = min(transient_metrics.call_put_ratio / 2.0, 1.0)  # Cap at 2.0 ratio

            # Borrow fee score (higher fee = harder to short = higher squeeze potential)
            borrow_score = 0.5  # Default neutral
            if transient_metrics.borrow_fee_pct is not None:
                borrow_score = min(transient_metrics.borrow_fee_pct / 10.0, 1.0)  # Cap at 10%

            # FINRA-based structural scores
            # Short interest percentage score (higher = better for squeeze)
            si_score = min(structural_metrics.short_interest_pct / 0.3, 1.0)  # Cap at 30%

            # Days to cover score (higher = better for squeeze)
            dtc_score = min(structural_metrics.days_to_cover / 10.0, 1.0)  # Cap at 10 days

            # Weighted combination of transient metrics
            transient_score = (
                volume_score * weights.volume_spike +
                sentiment_score * weights.sentiment_24h +
                callput_score * weights.call_put_ratio +
                borrow_score * weights.borrow_fee
            )

            # Weighted combination of FINRA structural metrics
            finra_score = (si_score * 0.6 + dtc_score * 0.4)  # Weight SI more heavily

            # Final combination: 50% transient, 30% FINRA, 20% original screener
            final_score = (
                0.5 * transient_score +
                0.3 * finra_score +
                0.2 * screener_score
            )

            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, final_score))

            return final_score

        except Exception as e:
            _logger.warning("Error calculating preliminary squeeze score: %s", e)
            return screener_score  # Fallback to screener score

    def _store_results(self, scored_candidates: List[ScoredCandidate],
                      scan_date: date, data_quality_metrics: Dict[str, Any]) -> None:
        """
        Store deep scan results in database.

        Args:
            scored_candidates: List of scored candidates
            scan_date: Date of the scan
            data_quality_metrics: Data quality metrics
        """
        try:
            service = ShortSqueezeService()

            # Convert scored candidates to dict format
            results = []
            for sc in scored_candidates:
                results.append({
                    'ticker': sc.candidate.ticker,
                    'squeeze_score': sc.squeeze_score,
                    'volume_spike': sc.transient_metrics.volume_spike,
                    'sentiment_24h': sc.transient_metrics.sentiment_24h,
                    'call_put_ratio': sc.transient_metrics.call_put_ratio,
                    'borrow_fee_pct': sc.transient_metrics.borrow_fee_pct,
                    'alert_level': sc.alert_level.value if sc.alert_level else None,
                })

            # Store deep scan results
            service.save_deep_scan_results(results=results, scan_date=scan_date)

            _logger.info("Stored deep scan results in database: %d candidates",
                       len(scored_candidates))

        except Exception:
            _logger.exception("Error storing deep scan results:")
            raise


def create_daily_deep_scan(fmp_downloader: FMPDataDownloader,
                          finnhub_downloader: FinnhubDataDownloader,
                          config: DeepScanConfig) -> DailyDeepScan:
    """
    Factory function to create Daily Deep Scan.

    Args:
        fmp_downloader: FMP data downloader instance
        finnhub_downloader: Finnhub data downloader instance
        config: Deep scan configuration

    Returns:
        Configured Daily Deep Scan instance
    """
    return DailyDeepScan(fmp_downloader, finnhub_downloader, config)


# Example usage
if __name__ == "__main__":
    from src.ml.pipeline.p04_short_squeeze.config.data_classes import DeepScanConfig

    # Create downloaders
    fmp_downloader = FMPDataDownloader()
    finnhub_downloader = FinnhubDataDownloader("your_FINNHUB_API_KEY")

    # Create configuration
    deep_scan_config = DeepScanConfig()

    # Test connections
    if fmp_downloader.test_connection():
        print("✅ FMP API connection successful")

        # Create deep scan
        deep_scan = create_daily_deep_scan(fmp_downloader, finnhub_downloader, deep_scan_config)

        # Run deep scan (will load candidates from database)
        results = deep_scan.run_deep_scan()

        print("✅ Deep scan completed:")
        print(f"  - Candidates processed: {results.candidates_processed}")
        print(f"  - Scored candidates: {len(results.scored_candidates)}")
        print(f"  - Runtime: {results.runtime_metrics.get('duration_seconds', 0):.2f} seconds")

        # Show top scored candidates
        top_candidates = sorted(results.scored_candidates,
                              key=lambda c: c.squeeze_score, reverse=True)[:5]

        for i, scored_candidate in enumerate(top_candidates, 1):
            candidate = scored_candidate.candidate
            metrics = scored_candidate.transient_metrics
            print(f"  {i}. {candidate.ticker}: squeeze_score={scored_candidate.squeeze_score:.3f}, "
                  f"volume_spike={metrics.volume_spike:.2f}, sentiment={metrics.sentiment_24h:.2f}")
    else:
        print("❌ FMP API connection failed")