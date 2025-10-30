"""
Volume-Based Squeeze Detector

This module provides volume and momentum-based squeeze detection as a complement
to traditional short interest analysis. It identifies potential squeeze candidates
based on volume spikes, price momentum, and technical indicators.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import numpy as np
from dataclasses import dataclass

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.data.downloader.fmp_data_downloader import FMPDataDownloader
from src.ml.pipeline.p04_short_squeeze.core.models import StructuralMetrics, Candidate, CandidateSource

_logger = setup_logger(__name__)


@dataclass
class VolumeMetrics:
    """Volume-based metrics for squeeze detection."""
    avg_volume_14d: float
    current_volume: float
    volume_spike_ratio: float
    volume_trend_7d: float
    volume_consistency: float

    def __post_init__(self):
        """Validate volume metrics."""
        if self.avg_volume_14d < 0:
            raise ValueError("Average volume must be non-negative")
        if self.current_volume < 0:
            raise ValueError("Current volume must be non-negative")
        if self.volume_spike_ratio < 0:
            raise ValueError("Volume spike ratio must be non-negative")


@dataclass
class MomentumMetrics:
    """Price momentum metrics for squeeze detection."""
    price_change_1d: float
    price_change_3d: float
    price_change_7d: float
    price_volatility: float
    momentum_score: float

    def __post_init__(self):
        """Validate momentum metrics."""
        if self.price_volatility < 0:
            raise ValueError("Price volatility must be non-negative")


@dataclass
class SqueezeIndicators:
    """Combined squeeze indicators."""
    volume_score: float
    momentum_score: float
    float_score: float
    combined_score: float
    squeeze_probability: str  # 'HIGH', 'MEDIUM', 'LOW'

    def __post_init__(self):
        """Validate squeeze indicators."""
        for score in [self.volume_score, self.momentum_score, self.float_score, self.combined_score]:
            if not 0 <= score <= 1:
                raise ValueError(f"Scores must be between 0 and 1, got {score}")


class VolumeSqueezeDetector:
    """
    Volume-based squeeze detector.

    Identifies potential squeeze candidates using volume spikes, price momentum,
    and float analysis when short interest data is not available.
    """

    def __init__(self, fmp_downloader: FMPDataDownloader):
        """
        Initialize Volume Squeeze Detector.

        Args:
            fmp_downloader: FMP data downloader instance
        """
        self.fmp_downloader = fmp_downloader

        # Scoring weights
        self.weights = {
            'volume_spike': 0.4,
            'momentum': 0.3,
            'float_size': 0.2,
            'consistency': 0.1
        }

        # Thresholds
        self.thresholds = {
            'high_volume_spike': 3.0,      # 3x average volume
            'medium_volume_spike': 2.0,    # 2x average volume
            'high_momentum': 0.15,         # 15% price increase
            'medium_momentum': 0.08,       # 8% price increase
            'small_float': 50_000_000,     # 50M shares
            'medium_float': 100_000_000    # 100M shares
        }

        _logger.info("Volume Squeeze Detector initialized")

    def analyze_ticker(self, ticker: str) -> Optional[Tuple[Candidate, SqueezeIndicators]]:
        """
        Analyze a ticker for volume-based squeeze potential.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (Candidate, SqueezeIndicators) or None if analysis failed
        """
        try:
            _logger.debug("Analyzing volume squeeze potential for %s", ticker)

            # Get company profile for basic data
            profile = self.fmp_downloader.get_company_profile(ticker)
            if not profile:
                return None

            # Get volume metrics
            volume_metrics = self._calculate_volume_metrics(ticker)
            if not volume_metrics:
                return None

            # Get momentum metrics
            momentum_metrics = self._calculate_momentum_metrics(ticker)
            if not momentum_metrics:
                return None

            # Calculate squeeze indicators
            squeeze_indicators = self._calculate_squeeze_indicators(
                volume_metrics, momentum_metrics, profile
            )

            # Create structural metrics (using available data)
            structural_metrics = self._create_structural_metrics(profile, volume_metrics)
            if not structural_metrics:
                return None

            # Create candidate
            candidate = Candidate(
                ticker=ticker,
                screener_score=squeeze_indicators.combined_score,
                structural_metrics=structural_metrics,
                last_updated=datetime.now(),
                source=CandidateSource.VOLUME_SCREENER
            )

            _logger.debug("Volume analysis for %s: score=%.3f, probability=%s",
                         ticker, squeeze_indicators.combined_score, squeeze_indicators.squeeze_probability)

            return candidate, squeeze_indicators

        except Exception as e:
            _logger.warning("Error analyzing ticker %s: %s", ticker, e)
            return None

    def _calculate_volume_metrics(self, ticker: str) -> Optional[VolumeMetrics]:
        """
        Calculate volume-based metrics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            VolumeMetrics or None if calculation failed
        """
        try:
            # Get 30 days of volume data to ensure we have enough trading days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            df = self.fmp_downloader.get_ohlcv(ticker, '1d', start_date, end_date)
            if df is None or df.empty or len(df) < 14:
                return None

            volumes = df['volume'].values

            # Calculate metrics
            avg_volume_14d = np.mean(volumes[-14:])  # Last 14 days
            current_volume = volumes[-1]  # Most recent day

            # Volume spike ratio
            volume_spike_ratio = current_volume / max(avg_volume_14d, 1)

            # 7-day volume trend (recent vs older)
            recent_7d = np.mean(volumes[-7:])
            older_7d = np.mean(volumes[-14:-7])
            volume_trend_7d = recent_7d / max(older_7d, 1)

            # Volume consistency (lower std dev = more consistent)
            volume_std = np.std(volumes[-14:])
            volume_consistency = 1.0 / (1.0 + volume_std / max(avg_volume_14d, 1))

            return VolumeMetrics(
                avg_volume_14d=avg_volume_14d,
                current_volume=current_volume,
                volume_spike_ratio=volume_spike_ratio,
                volume_trend_7d=volume_trend_7d,
                volume_consistency=volume_consistency
            )

        except Exception as e:
            _logger.warning("Error calculating volume metrics for %s: %s", ticker, e)
            return None

    def _calculate_momentum_metrics(self, ticker: str) -> Optional[MomentumMetrics]:
        """
        Calculate price momentum metrics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            MomentumMetrics or None if calculation failed
        """
        try:
            # Get 14 days of price data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)

            df = self.fmp_downloader.get_ohlcv(ticker, '1d', start_date, end_date)
            if df is None or df.empty or len(df) < 7:
                return None

            closes = df['close'].values

            # Price changes
            price_change_1d = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
            price_change_3d = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
            price_change_7d = (closes[-1] - closes[-8]) / closes[-8] if len(closes) >= 8 else 0

            # Price volatility (standard deviation of returns)
            returns = np.diff(closes) / closes[:-1]
            price_volatility = np.std(returns) if len(returns) > 0 else 0

            # Momentum score (weighted combination of price changes)
            momentum_score = (
                price_change_1d * 0.5 +
                price_change_3d * 0.3 +
                price_change_7d * 0.2
            )

            return MomentumMetrics(
                price_change_1d=price_change_1d,
                price_change_3d=price_change_3d,
                price_change_7d=price_change_7d,
                price_volatility=price_volatility,
                momentum_score=momentum_score
            )

        except Exception as e:
            _logger.warning("Error calculating momentum metrics for %s: %s", ticker, e)
            return None

    def _calculate_squeeze_indicators(self, volume_metrics: VolumeMetrics,
                                    momentum_metrics: MomentumMetrics,
                                    profile: Dict[str, Any]) -> SqueezeIndicators:
        """
        Calculate combined squeeze indicators.

        Args:
            volume_metrics: Volume metrics
            momentum_metrics: Momentum metrics
            profile: Company profile data

        Returns:
            SqueezeIndicators with combined analysis
        """
        try:
            # Volume score (0-1)
            volume_score = self._normalize_volume_score(volume_metrics)

            # Momentum score (0-1)
            momentum_score = self._normalize_momentum_score(momentum_metrics)

            # Float score (smaller float = higher score)
            float_score = self._calculate_float_score(profile)

            # Combined score
            combined_score = (
                volume_score * self.weights['volume_spike'] +
                momentum_score * self.weights['momentum'] +
                float_score * self.weights['float_size'] +
                volume_metrics.volume_consistency * self.weights['consistency']
            )

            # Determine squeeze probability
            if combined_score >= 0.7:
                squeeze_probability = 'HIGH'
            elif combined_score >= 0.5:
                squeeze_probability = 'MEDIUM'
            else:
                squeeze_probability = 'LOW'

            return SqueezeIndicators(
                volume_score=volume_score,
                momentum_score=momentum_score,
                float_score=float_score,
                combined_score=combined_score,
                squeeze_probability=squeeze_probability
            )

        except Exception as e:
            _logger.warning("Error calculating squeeze indicators: %s", e)
            # Return default low-probability indicators
            return SqueezeIndicators(
                volume_score=0.0,
                momentum_score=0.0,
                float_score=0.0,
                combined_score=0.0,
                squeeze_probability='LOW'
            )

    def _normalize_volume_score(self, volume_metrics: VolumeMetrics) -> float:
        """Normalize volume metrics to 0-1 score."""
        try:
            # Volume spike component (most important)
            spike_score = min(volume_metrics.volume_spike_ratio / 5.0, 1.0)  # Cap at 5x

            # Volume trend component
            trend_score = min(volume_metrics.volume_trend_7d / 2.0, 1.0)  # Cap at 2x

            # Combined volume score
            volume_score = spike_score * 0.7 + trend_score * 0.3

            return max(0.0, min(1.0, volume_score))

        except Exception:
            return 0.0

    def _normalize_momentum_score(self, momentum_metrics: MomentumMetrics) -> float:
        """Normalize momentum metrics to 0-1 score."""
        try:
            # Use momentum score but cap extreme values
            momentum_score = momentum_metrics.momentum_score

            # Normalize to 0-1 range (assuming max reasonable momentum is 50%)
            normalized = (momentum_score + 0.1) / 0.6  # Shift and scale

            return max(0.0, min(1.0, normalized))

        except Exception:
            return 0.0

    def _calculate_float_score(self, profile: Dict[str, Any]) -> float:
        """Calculate float-based score (smaller float = higher squeeze potential)."""
        try:
            market_cap = profile.get('mktCap', 0)
            price = profile.get('price', 1)

            # Estimate shares outstanding from market cap and price
            shares_outstanding = market_cap / max(price, 1) if price > 0 else 0

            # Assume float is roughly 80% of shares outstanding (typical)
            estimated_float = shares_outstanding * 0.8

            # Score based on float size (smaller = better for squeezes)
            if estimated_float <= self.thresholds['small_float']:
                return 1.0
            elif estimated_float <= self.thresholds['medium_float']:
                return 0.7
            elif estimated_float <= 200_000_000:  # 200M
                return 0.4
            else:
                return 0.1

        except Exception:
            return 0.5  # Default middle score

    def _create_structural_metrics(self, profile: Dict[str, Any],
                                 volume_metrics: VolumeMetrics) -> Optional[StructuralMetrics]:
        """Create structural metrics from available data."""
        try:
            market_cap = profile.get('mktCap', 0)
            price = profile.get('price', 1)

            if market_cap <= 0 or price <= 0:
                return None

            # Estimate shares outstanding and float
            shares_outstanding = market_cap / price
            estimated_float = int(shares_outstanding * 0.8)  # Assume 80% float

            # Use placeholder values for short interest (will be updated with FINRA data)
            short_interest_pct = 0.0  # Will be updated with real data
            days_to_cover = 0.0  # Will be calculated with real short interest

            return StructuralMetrics(
                short_interest_pct=short_interest_pct,
                days_to_cover=days_to_cover,
                float_shares=estimated_float,
                avg_volume_14d=int(volume_metrics.avg_volume_14d),
                market_cap=int(market_cap)
            )

        except Exception as e:
            _logger.warning("Error creating structural metrics: %s", e)
            return None

    def screen_universe(self, universe: List[str], min_score: float = 0.3) -> List[Tuple[Candidate, SqueezeIndicators]]:
        """
        Screen entire universe for volume-based squeeze candidates.

        Args:
            universe: List of ticker symbols to screen
            min_score: Minimum combined score to include in results

        Returns:
            List of (Candidate, SqueezeIndicators) tuples sorted by score
        """
        try:
            _logger.info("Screening %d tickers for volume-based squeeze potential", len(universe))

            results = []

            for i, ticker in enumerate(universe, 1):
                try:
                    analysis = self.analyze_ticker(ticker)
                    if analysis:
                        candidate, indicators = analysis
                        if indicators.combined_score >= min_score:
                            results.append((candidate, indicators))

                    # Log progress
                    if i % 50 == 0:
                        _logger.info("Screened %d/%d tickers, found %d candidates",
                                   i, len(universe), len(results))

                except Exception as e:
                    _logger.warning("Error screening ticker %s: %s", ticker, e)
                    continue

            # Sort by combined score (descending)
            results.sort(key=lambda x: x[1].combined_score, reverse=True)

            _logger.info("Volume screening completed: %d candidates found", len(results))
            return results

        except Exception as e:
            _logger.exception("Error in volume screening:")
            return []


def create_volume_squeeze_detector(fmp_downloader: FMPDataDownloader) -> VolumeSqueezeDetector:
    """
    Factory function to create Volume Squeeze Detector.

    Args:
        fmp_downloader: FMP data downloader instance

    Returns:
        Configured Volume Squeeze Detector instance
    """
    return VolumeSqueezeDetector(fmp_downloader)


# Example usage
if __name__ == "__main__":
    from src.data.downloader.fmp_data_downloader import FMPDataDownloader

    # Create FMP downloader
    fmp_downloader = FMPDataDownloader()

    # Test connection
    if fmp_downloader.test_connection():
        print("✅ FMP API connection successful")

        # Create volume detector
        detector = create_volume_squeeze_detector(fmp_downloader)

        # Test with known volatile stocks
        test_tickers = ['GME', 'AMC', 'TSLA', 'AAPL', 'NVDA']

        print(f"Testing volume squeeze detection on {len(test_tickers)} tickers...")

        for ticker in test_tickers:
            analysis = detector.analyze_ticker(ticker)
            if analysis:
                candidate, indicators = analysis
                print(f"{ticker}: Score={indicators.combined_score:.3f}, "
                      f"Probability={indicators.squeeze_probability}, "
                      f"Volume={indicators.volume_score:.3f}, "
                      f"Momentum={indicators.momentum_score:.3f}")
            else:
                print(f"{ticker}: Analysis failed")

    else:
        print("❌ FMP API connection failed")