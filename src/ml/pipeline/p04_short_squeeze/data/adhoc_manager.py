"""
Short Squeeze Detection Pipeline Ad-hoc Candidate Manager

This module provides the AdHocManager class for managing manually added candidates,
including TTL-based expiration, activation/deactivation, and integration with deep scan processing.
"""
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.db.core.database import session_scope
from src.data.db.services.short_squeeze_service import ShortSqueezeService
from src.ml.pipeline.p04_short_squeeze.core.models import AdHocCandidate
from src.notification.logger import setup_logger
_logger = setup_logger(__name__)


class AdHocManager:
    """
    Ad-hoc Candidate Manager for Short Squeeze Detection Pipeline.

    Manages manually added candidates for monitoring, including TTL-based expiration,
    activation/deactivation functionality, and integration with deep scan processing.
    """

    def __init__(self, default_ttl_days: int=7):
        """
        Initialize the ad-hoc candidate manager.

        Args:
            default_ttl_days: Default time-to-live in days for new candidates
        """
        self.default_ttl_days = default_ttl_days

    def add_candidate(self, ticker: str, reason: str, ttl_days: Optional[
        int]=None) ->bool:
        """
        Add a new ad-hoc candidate for monitoring.

        Args:
            ticker: Stock ticker symbol
            reason: Reason for adding the candidate
            ttl_days: Time-to-live in days, or None to use default

        Returns:
            True if candidate was added successfully, False otherwise
        """
        if ttl_days is None:
            ttl_days = self.default_ttl_days
        ticker = ticker.upper().strip()
        if not ticker:
            _logger.error('Cannot add ad-hoc candidate: ticker cannot be empty'
                )
            return False
        if not reason or not reason.strip():
            _logger.error(
                'Cannot add ad-hoc candidate %s: reason cannot be empty',
                ticker)
            return False
        _logger.info('Adding ad-hoc candidate: %s (reason: %s, TTL: %d days)',
            ticker, reason, ttl_days)
        with session_scope() as session:
            service = ShortSqueezeService(session)
            return service.add_adhoc_candidate(ticker, reason, ttl_days)

    def remove_candidate(self, ticker: str) ->bool:
        """
        Remove (deactivate) an ad-hoc candidate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was removed successfully, False otherwise
        """
        ticker = ticker.upper().strip()
        if not ticker:
            _logger.error(
                'Cannot remove ad-hoc candidate: ticker cannot be empty')
            return False
        _logger.info('Removing ad-hoc candidate: %s', ticker)
        with session_scope() as session:
            service = ShortSqueezeService(session)
            return service.remove_adhoc_candidate(ticker)

    def get_active_candidates(self) ->List[AdHocCandidate]:
        """
        Get all active ad-hoc candidates.

        Returns:
            List of active AdHocCandidate objects
        """
        with session_scope() as session:
            service = ShortSqueezeService(session)
            repo = service.repos.short_squeeze
            active_models = repo.adhoc_candidates.get_active_candidates()
            candidates = []
            for model in active_models:
                candidate = AdHocCandidate(ticker=model.ticker, reason=
                    model.reason or '', first_seen=model.first_seen,
                    expires_at=model.expires_at, active=model.active,
                    promoted_by_screener=model.promoted_by_screener)
                candidates.append(candidate)
            _logger.debug('Retrieved %d active ad-hoc candidates', len(
                candidates))
            return candidates

    def get_candidate(self, ticker: str) ->Optional[AdHocCandidate]:
        """
        Get a specific ad-hoc candidate by ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            AdHocCandidate object if found, None otherwise
        """
        ticker = ticker.upper().strip()
        with session_scope() as session:
            service = ShortSqueezeService(session)
            repo = service.repos.short_squeeze
            model = repo.adhoc_candidates.get_candidate(ticker)
            if model:
                return AdHocCandidate(ticker=model.ticker, reason=model.
                    reason or '', first_seen=model.first_seen, expires_at=
                    model.expires_at, active=model.active,
                    promoted_by_screener=model.promoted_by_screener)
            return None

    def activate_candidate(self, ticker: str) ->bool:
        """
        Activate a previously deactivated ad-hoc candidate.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was activated, False otherwise
        """
        ticker = ticker.upper().strip()
        with session_scope() as session:
            service = ShortSqueezeService(session)
            repo = service.repos.short_squeeze
            model = repo.adhoc_candidates.get_candidate(ticker)
            if not model:
                _logger.warning(
                    'Cannot activate ad-hoc candidate %s: not found', ticker)
                return False
            if model.active:
                _logger.info('Ad-hoc candidate %s is already active', ticker)
                return True
            success = service.add_adhoc_candidate(ticker, model.reason or
                'Reactivated', self.default_ttl_days)
            if success:
                _logger.info('Activated ad-hoc candidate: %s', ticker)
            else:
                _logger.error('Failed to activate ad-hoc candidate: %s', ticker
                    )
            return success

    def deactivate_candidate(self, ticker: str) ->bool:
        """
        Deactivate an ad-hoc candidate without removing it from the database.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was deactivated, False otherwise
        """
        return self.remove_candidate(ticker)

    def expire_candidates(self) ->List[str]:
        """
        Expire ad-hoc candidates that have passed their TTL.

        Returns:
            List of ticker symbols that were expired
        """
        _logger.info('Running ad-hoc candidate expiration process')
        with session_scope() as session:
            service = ShortSqueezeService(session)
            expired_tickers = service.expire_adhoc_candidates()
            if expired_tickers:
                _logger.info('Expired %d ad-hoc candidates: %s', len(
                    expired_tickers), expired_tickers)
            else:
                _logger.debug('No ad-hoc candidates expired')
            return expired_tickers

    def extend_ttl(self, ticker: str, additional_days: int) ->bool:
        """
        Extend the TTL of an ad-hoc candidate.

        Args:
            ticker: Stock ticker symbol
            additional_days: Number of additional days to extend TTL

        Returns:
            True if TTL was extended, False otherwise
        """
        ticker = ticker.upper().strip()
        if additional_days <= 0:
            _logger.error(
                'Cannot extend TTL for %s: additional_days must be positive',
                ticker)
            return False
        with session_scope() as session:
            service = ShortSqueezeService(session)
            repo = service.repos.short_squeeze
            model = repo.adhoc_candidates.get_candidate(ticker)
            if not model:
                _logger.warning(
                    'Cannot extend TTL for ad-hoc candidate %s: not found',
                    ticker)
                return False
            if not model.active:
                _logger.warning(
                    'Cannot extend TTL for ad-hoc candidate %s: not active',
                    ticker)
                return False
            current_expires = model.expires_at or datetime.now()
            new_expires = current_expires + timedelta(days=additional_days)
            model.expires_at = new_expires
            session.commit()
            _logger.info(
                'Extended TTL for ad-hoc candidate %s by %d days (new expiry: %s)'
                , ticker, additional_days, new_expires)
            return True

    def promote_by_screener(self, ticker: str) ->bool:
        """
        Mark an ad-hoc candidate as promoted by screener.

        This indicates that the candidate was later identified by the screener,
        validating the manual addition.

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if candidate was marked as promoted, False otherwise
        """
        ticker = ticker.upper().strip()
        with session_scope() as session:
            service = ShortSqueezeService(session)
            repo = service.repos.short_squeeze
            success = repo.adhoc_candidates.promote_by_screener(ticker)
            if success:
                session.commit()
                _logger.info(
                    'Marked ad-hoc candidate %s as promoted by screener',
                    ticker)
            else:
                _logger.warning(
                    'Failed to mark ad-hoc candidate %s as promoted: not found'
                    , ticker)
            return success

    def get_candidates_for_deep_scan(self) ->List[str]:
        """
        Get active ad-hoc candidate tickers for deep scan processing.

        Returns:
            List of ticker symbols for active ad-hoc candidates
        """
        active_candidates = self.get_active_candidates()
        tickers = [candidate.ticker for candidate in active_candidates]
        _logger.debug('Found %d active ad-hoc candidates for deep scan: %s',
            len(tickers), tickers)
        return tickers

    def bulk_add_candidates(self, candidates: List[Dict[str, Any]]) ->Tuple[
        int, List[str]]:
        """
        Add multiple ad-hoc candidates in bulk.

        Args:
            candidates: List of dictionaries with 'ticker', 'reason', and optional 'ttl_days'

        Returns:
            Tuple of (number_added, list_of_errors)
        """
        added_count = 0
        errors = []
        for i, candidate_data in enumerate(candidates):
            try:
                ticker = candidate_data.get('ticker', '').strip()
                reason = candidate_data.get('reason', '').strip()
                ttl_days = candidate_data.get('ttl_days', self.default_ttl_days
                    )
                if not ticker:
                    errors.append(f'Candidate {i}: ticker is required')
                    continue
                if not reason:
                    errors.append(f'Candidate {i}: reason is required')
                    continue
                if self.add_candidate(ticker, reason, ttl_days):
                    added_count += 1
                else:
                    errors.append(f'Candidate {i}: failed to add {ticker}')
            except Exception as e:
                errors.append(f'Candidate {i}: {str(e)}')
        _logger.info('Bulk add completed: %d added, %d errors', added_count,
            len(errors))
        return added_count, errors

    def get_expiring_candidates(self, days_ahead: int=3) ->List[AdHocCandidate
        ]:
        """
        Get candidates that will expire within the specified number of days.

        Args:
            days_ahead: Number of days to look ahead for expiring candidates

        Returns:
            List of AdHocCandidate objects that will expire soon
        """
        cutoff_date = datetime.now() + timedelta(days=days_ahead)
        active_candidates = self.get_active_candidates()
        expiring_candidates = []
        for candidate in active_candidates:
            if candidate.expires_at and candidate.expires_at <= cutoff_date:
                expiring_candidates.append(candidate)
        _logger.debug('Found %d candidates expiring within %d days', len(
            expiring_candidates), days_ahead)
        return expiring_candidates

    def get_statistics(self) ->Dict[str, Any]:
        """
        Get ad-hoc candidate statistics.

        Returns:
            Dictionary with statistics about ad-hoc candidates
        """
        active_candidates = self.get_active_candidates()
        total_active = len(active_candidates)
        promoted_count = sum(1 for c in active_candidates if c.
            promoted_by_screener)
        expiring_soon = len(self.get_expiring_candidates(3))
        now = datetime.now()
        ages = [(now - c.first_seen).days for c in active_candidates if c.
            first_seen]
        avg_age_days = sum(ages) / len(ages) if ages else 0
        stats = {'total_active': total_active, 'promoted_by_screener':
            promoted_count, 'expiring_within_3_days': expiring_soon,
            'average_age_days': round(avg_age_days, 1), 'default_ttl_days':
            self.default_ttl_days, 'last_updated': datetime.now()}
        _logger.debug('Ad-hoc candidate statistics: %s', stats)
        return stats

    def validate_candidate_data(self, candidate_data: Dict[str, Any]) ->Tuple[
        bool, List[str]]:
        """
        Validate ad-hoc candidate data before adding.

        Args:
            candidate_data: Dictionary with candidate information

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        ticker = candidate_data.get('ticker', '').strip()
        reason = candidate_data.get('reason', '').strip()
        if not ticker:
            errors.append('Ticker is required')
        elif len(ticker) > 10:
            errors.append('Ticker must be 10 characters or less')
        if not reason:
            errors.append('Reason is required')
        elif len(reason) > 1000:
            errors.append('Reason must be 1000 characters or less')
        ttl_days = candidate_data.get('ttl_days')
        if ttl_days is not None:
            if not isinstance(ttl_days, int) or ttl_days <= 0:
                errors.append('TTL days must be a positive integer')
            elif ttl_days > 365:
                errors.append('TTL days cannot exceed 365')
        is_valid = len(errors) == 0
        if not is_valid:
            _logger.warning('Ad-hoc candidate data validation failed: %s',
                errors)
        return is_valid, errors
