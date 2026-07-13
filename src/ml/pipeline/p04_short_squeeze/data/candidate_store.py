"""
Short Squeeze Detection Pipeline Candidate Store

This module provides the CandidateStore class for managing candidate lifecycle,
screener snapshot storage and retrieval, and deep scan results storage.
"""
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics, TransientMetrics, Candidate, ScoredCandidate, Alert, AdHocCandidate, AlertLevel, CandidateSource
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class CandidateStore:
    """
    Candidate Store for Short Squeeze Detection Pipeline.

    Provides high-level operations for candidate lifecycle management,
    screener snapshot storage and retrieval, and deep scan results storage.
    Uses the centralized database infrastructure through ShortSqueezeService.
    """

    def __init__(self):
        """Initialize the candidate store."""
        pass

    def store_screener_snapshot(self, candidates: List[Dict[str, Any]],
        run_date: date) ->int:
        """
        Store screener snapshot with candidate data.

        Args:
            candidates: List of candidate dictionaries with screener results
            run_date: Date of the screener run

        Returns:
            Number of candidates stored
        """
        _logger.info('Storing screener snapshot for %s with %d candidates',
            run_date, len(candidates))
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.save_screener_results(candidates, run_date)

    def retrieve_screener_snapshot(self, run_date: Optional[date]=None,
        limit: int=50) ->List[Dict[str, Any]]:
        """
        Retrieve screener snapshot for a specific date or latest run.

        Args:
            run_date: Specific date to retrieve, or None for latest
            limit: Maximum number of candidates to return

        Returns:
            List of candidate dictionaries from screener
        """
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        if run_date is None:
            return service.get_top_candidates_by_screener_score(limit)
        else:
            repo = service.repos.short_squeeze
            snapshots = repo.screener_snapshots.get_top_candidates(run_date
                , limit)
            results = []
            for snapshot in snapshots:
                results.append({
                    'ticker': snapshot.ticker,
                    'screener_score': float(snapshot.screener_score) if snapshot.screener_score else 0.0,
                    'short_interest_pct': float(snapshot.short_interest_pct) if snapshot.short_interest_pct else None,
                    'days_to_cover': float(snapshot.days_to_cover) if snapshot.days_to_cover else None,
                    'float_shares': snapshot.float_shares,
                    'avg_volume_14d': snapshot.avg_volume_14d,
                    'market_cap': snapshot.market_cap,
                    'run_date': snapshot.run_date,
                    'data_quality': float(snapshot.data_quality) if snapshot.data_quality else None,
                })
            return results

    def store_deep_scan_results(self, results: List[Dict[str, Any]],
        scan_date: date) ->int:
        """
        Store daily deep scan results with date-based updates.

        Args:
            results: List of deep scan result dictionaries
            scan_date: Date of the deep scan

        Returns:
            Number of results stored/updated
        """
        _logger.info('Storing deep scan results for %s with %d results',
            scan_date, len(results))
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.save_deep_scan_results(results, scan_date)

    def retrieve_deep_scan_results(self, scan_date: Optional[date]=None,
        limit: int=20) ->List[Dict[str, Any]]:
        """
        Retrieve deep scan results for a specific date.

        Args:
            scan_date: Date to retrieve results for, or None for today
            limit: Maximum number of results to return

        Returns:
            List of deep scan result dictionaries
        """
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_top_squeeze_scores(scan_date, limit)

    def get_active_candidates(self) ->List[str]:
        """
        Get all active candidates for deep scan processing.

        Combines screener candidates and active ad-hoc candidates.

        Returns:
            List of ticker symbols for active candidates
        """
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_candidates_for_deep_scan_tickers()

    def create_candidate(self, ticker: str, screener_score: float,
        structural_metrics: StructuralMetrics, source: CandidateSource=
        CandidateSource.SCREENER) ->Candidate:
        """
        Create a new candidate with lifecycle management.

        Args:
            ticker: Stock ticker symbol
            screener_score: Score from screener analysis
            structural_metrics: Structural metrics for the candidate
            source: Source of the candidate (screener or adhoc)

        Returns:
            Created Candidate object
        """
        candidate = Candidate(ticker=ticker, screener_score=screener_score,
            structural_metrics=structural_metrics, last_updated=datetime.
            now(), source=source)
        _logger.info('Created candidate: %s (score: %.3f, source: %s)',
            ticker, screener_score, source.value)
        return candidate

    def update_candidate(self, ticker: str, **updates) ->bool:
        """
        Update candidate information.

        Args:
            ticker: Stock ticker symbol
            **updates: Fields to update

        Returns:
            True if candidate was updated, False if not found
        """
        _logger.info('Update requested for candidate %s: %s', ticker, updates)
        return True

    def expire_candidate(self, ticker: str) ->bool:
        """
        Expire a candidate (mark as inactive).

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was expired, False if not found
        """
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.remove_adhoc_candidate(ticker)

    def get_candidate_history(self, ticker: str, days: int=30) ->Dict[str, Any
        ]:
        """
        Get comprehensive candidate history including screener and deep scan data.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history to retrieve

        Returns:
            Dictionary with candidate history data
        """
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.get_ticker_analysis(ticker, days)

    def cleanup_expired_candidates(self, days_to_keep: int=90) ->Dict[str, int
        ]:
        """
        Clean up old candidate data beyond retention period.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Dictionary with cleanup statistics
        """
        _logger.info('Starting candidate data cleanup (keeping %d days)',
            days_to_keep)
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        return service.cleanup_old_data(days_to_keep)

    def get_candidate_statistics(self) ->Dict[str, Any]:
        """
        Get candidate store statistics and health metrics.

        Returns:
            Dictionary with statistics about candidates and pipeline health
        """
        # Service manages sessions internally via UoW pattern
        service = ShortSqueezeService()
        stats = service.get_pipeline_statistics()
        stats.update({"store_type": "CandidateStore", "last_updated": datetime.now()})
        return stats

    def validate_candidate_data(self, candidate_data: Dict[str, Any]) ->Tuple[
        bool, List[str]]:
        """
        Validate candidate data before storage.

        Args:
            candidate_data: Dictionary with candidate information

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_fields = ['ticker', 'screener_score']
        for field in required_fields:
            if field not in candidate_data or candidate_data[field] is None:
                errors.append(f'Missing required field: {field}')
        if 'ticker' in candidate_data:
            ticker = candidate_data['ticker']
            if not isinstance(ticker, str) or len(ticker.strip()) == 0:
                errors.append('Ticker must be a non-empty string')
            elif len(ticker) > 10:
                errors.append('Ticker must be 10 characters or less')
        if 'screener_score' in candidate_data:
            score = candidate_data['screener_score']
            if not isinstance(score, (int, float)) or score < 0 or score > 1:
                errors.append('Screener score must be a number between 0 and 1'
                    )
        if 'structural_metrics' in candidate_data:
            metrics = candidate_data['structural_metrics']
            if isinstance(metrics, dict):
                try:
                    StructuralMetrics(**metrics)
                except (ValueError, TypeError) as e:
                    errors.append(f'Invalid structural metrics: {str(e)}')
        is_valid = len(errors) == 0
        if not is_valid:
            _logger.warning('Candidate data validation failed: %s', errors)
        return is_valid, errors

    def batch_store_candidates(self, candidates: List[Dict[str, Any]],
        run_date: date, validate: bool=True) ->Tuple[int, List[str]]:
        """
        Store multiple candidates in batch with optional validation.

        Args:
            candidates: List of candidate dictionaries
            run_date: Date of the batch operation
            validate: Whether to validate each candidate before storage

        Returns:
            Tuple of (number_stored, list_of_errors)
        """
        errors = []
        valid_candidates = []
        if validate:
            for i, candidate in enumerate(candidates):
                is_valid, validation_errors = self.validate_candidate_data(
                    candidate)
                if is_valid:
                    valid_candidates.append(candidate)
                else:
                    errors.extend([f'Candidate {i}: {error}' for error in
                        validation_errors])
        else:
            valid_candidates = candidates
        if valid_candidates:
            try:
                stored_count = self.store_screener_snapshot(valid_candidates,
                    run_date)
                _logger.info('Batch stored %d candidates (rejected %d)',
                    stored_count, len(candidates) - len(valid_candidates))
                return stored_count, errors
            except Exception as e:
                error_msg = f'Failed to store candidates: {str(e)}'
                errors.append(error_msg)
                _logger.error(error_msg)
                return 0, errors
        else:
            _logger.warning('No valid candidates to store')
            return 0, errors
