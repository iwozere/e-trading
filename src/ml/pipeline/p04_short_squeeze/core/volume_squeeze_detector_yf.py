"""
Volume-Based Squeeze Detector using yfinance

This module provides volume and momentum-based squeeze detection using yfinance
instead of FMP API to avoid rate limiting issues.
"""

from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import statistics
import numpy as np
import pandas as pd
from dataclasses import dataclass
import yfinance as yf

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
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


class VolumeSqueezeDetectorYF:
    """
    Volume-based squeeze detector using yfinance.

    Identifies potential squeeze candidates using volume spikes, price momentum,
    and float analysis without requiring FMP API.
    """

    def __init__(self):
        """Initialize Volume Squeeze Detector with yfinance."""
        # Scoring weights
        self.weights = {
            'volume_spike': 0.4,
            'momentum': 0.3,
            'float_size': 0.2,
            'consistency': 0.1
        }

        # Thresholds
        self.min_volume_spike = 2.0  # 2x average volume
        self.min_momentum_score = 0.3
        self.lookback_days = 30

    def get_company_info_yf(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company information using yfinance.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company info or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract relevant information
            return {
                'ticker': ticker,
                'market_cap': info.get('marketCap'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'avg_volume': info.get('averageVolume'),
                'avg_volume_10d': info.get('averageVolume10days'),
                'sector': info.get('sector'),
                'industry': info.get('industry')
            }
        except Exception as e:
            _logger.warning("Failed to get company info for %s: %s", ticker, e)
            return None

    def get_ohlcv_yf(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data using yfinance.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                _logger.debug("No data returned for %s", ticker)
                return None

            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                _logger.warning("Missing required columns for %s", ticker)
                return None

            return df[required_cols]

        except Exception as e:
            _logger.warning("Failed to get OHLCV data for %s: %s", ticker, e)
            return None

    def calculate_volume_metrics(self, ticker: str) -> Optional[VolumeMetrics]:
        """
        Calculate volume metrics for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            VolumeMetrics object or None if calculation fails
        """
        try:
            # Get historical data (30 days to have enough for calculations)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            df = self.get_ohlcv_yf(ticker, start_date, end_date)
            if df is None or len(df) < 14:
                _logger.debug("Insufficient data for %s", ticker)
                return None

            # Calculate metrics
            volumes = df['volume'].values
            current_volume = float(volumes[-1])
            avg_volume_14d = float(np.mean(volumes[-14:]))

            if avg_volume_14d == 0:
                return None

            volume_spike_ratio = current_volume / avg_volume_14d

            # 7-day volume trend
            recent_avg = np.mean(volumes[-7:])
            older_avg = np.mean(volumes[-14:-7])
            volume_trend_7d = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

            # Volume consistency (lower coefficient of variation = more consistent)
            volume_std = np.std(volumes[-14:])
            volume_consistency = 1 - min(volume_std / avg_volume_14d, 1.0) if avg_volume_14d > 0 else 0

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

    def calculate_momentum_metrics(self, ticker: str) -> Optional[MomentumMetrics]:
        """
        Calculate price momentum metrics.

        Args:
            ticker: Stock ticker symbol

        Returns:
            MomentumMetrics object or None if calculation fails
        """
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            df = self.get_ohlcv_yf(ticker, start_date, end_date)
            if df is None or len(df) < 7:
                return None

            closes = df['close'].values

            # Price changes
            price_change_1d = (closes[-1] - closes[-2]) / closes[-2] if len(closes) >= 2 else 0
            price_change_3d = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
            price_change_7d = (closes[-1] - closes[-8]) / closes[-8] if len(closes) >= 8 else 0

            # Volatility (14-day standard deviation of returns)
            if len(closes) >= 14:
                returns = np.diff(closes[-14:]) / closes[-14:-1]
                price_volatility = float(np.std(returns))
            else:
                returns = np.diff(closes) / closes[:-1]
                price_volatility = float(np.std(returns))

            # Momentum score (weighted combination of price changes)
            momentum_score = (
                0.3 * max(0, price_change_1d) +
                0.4 * max(0, price_change_3d) +
                0.3 * max(0, price_change_7d)
            )
            momentum_score = min(momentum_score, 1.0)  # Cap at 1.0

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

    def calculate_float_score(self, ticker: str) -> float:
        """
        Calculate float size score (smaller float = higher squeeze potential).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Float score between 0 and 1
        """
        try:
            info = self.get_company_info_yf(ticker)
            if not info:
                return 0.5  # Default neutral score

            market_cap = info.get('market_cap')
            float_shares = info.get('float_shares')

            if not market_cap or not float_shares:
                return 0.5

            # Smaller float = higher score
            # Using log scale: float < 10M = 1.0, float > 100M = 0.0
            if float_shares < 10_000_000:
                return 1.0
            elif float_shares > 100_000_000:
                return 0.0
            else:
                # Linear interpolation between 10M and 100M
                return 1.0 - ((float_shares - 10_000_000) / 90_000_000)

        except Exception:
            return 0.5

    def calculate_squeeze_indicators(self, ticker: str) -> Optional[SqueezeIndicators]:
        """
        Calculate combined squeeze indicators.

        Args:
            ticker: Stock ticker symbol

        Returns:
            SqueezeIndicators object or None if calculation fails
        """
        try:
            # Get individual metrics
            volume_metrics = self.calculate_volume_metrics(ticker)
            momentum_metrics = self.calculate_momentum_metrics(ticker)

            if not volume_metrics or not momentum_metrics:
                return None

            # Calculate scores
            volume_score = min(volume_metrics.volume_spike_ratio / 5.0, 1.0)  # Normalize to 0-1
            momentum_score = momentum_metrics.momentum_score
            float_score = self.calculate_float_score(ticker)

            # Combined score (weighted average)
            combined_score = (
                self.weights['volume_spike'] * volume_score +
                self.weights['momentum'] * momentum_score +
                self.weights['float_size'] * float_score +
                self.weights['consistency'] * volume_metrics.volume_consistency
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
            _logger.warning("Error calculating squeeze indicators for %s: %s", ticker, e)
            return None

    def analyze_ticker(self, ticker: str) -> Optional[Tuple[Candidate, SqueezeIndicators]]:
        """
        Analyze a single ticker for volume squeeze potential.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (Candidate, SqueezeIndicators) or None if not a candidate
        """
        try:
            _logger.debug("Analyzing volume squeeze potential for %s", ticker)

            # Calculate indicators
            indicators = self.calculate_squeeze_indicators(ticker)
            if not indicators:
                return None

            # Check if meets minimum thresholds
            volume_metrics = self.calculate_volume_metrics(ticker)
            if not volume_metrics:
                return None

            if volume_metrics.volume_spike_ratio < self.min_volume_spike:
                return None

            if indicators.momentum_score < self.min_momentum_score:
                return None

            # Create candidate
            candidate = Candidate(
                ticker=ticker,
                source=CandidateSource.VOLUME_DETECTOR,
                detection_date=datetime.now().date(),
                confidence_score=indicators.combined_score
            )

            return (candidate, indicators)

        except Exception as e:
            _logger.warning("Error analyzing ticker %s: %s", ticker, e)
            return None

    def screen_universe(self, tickers: List[str], min_score: float = 0.3) -> List[Tuple[Candidate, SqueezeIndicators]]:
        """
        Screen universe of tickers for volume squeeze candidates.

        Args:
            tickers: List of ticker symbols to screen
            min_score: Minimum combined score threshold

        Returns:
            List of (Candidate, SqueezeIndicators) tuples
        """
        candidates = []

        _logger.info("Screening %d tickers for volume squeeze candidates", len(tickers))

        for i, ticker in enumerate(tickers):
            if i % 50 == 0:
                _logger.info("Progress: %d/%d tickers analyzed, %d candidates found",
                           i, len(tickers), len(candidates))

            try:
                result = self.analyze_ticker(ticker)
                if result:
                    candidate, indicators = result
                    if indicators.combined_score >= min_score:
                        candidates.append(result)
                        _logger.info("Found candidate: %s (score=%.3f, volume_spike=%.2fx)",
                                   ticker, indicators.combined_score,
                                   self.calculate_volume_metrics(ticker).volume_spike_ratio)

            except Exception as e:
                _logger.warning("Error screening ticker %s: %s", ticker, e)
                continue

        _logger.info("Screening complete: %d candidates found from %d tickers",
                    len(candidates), len(tickers))

        # Sort by combined score
        candidates.sort(key=lambda x: x[1].combined_score, reverse=True)

        return candidates


def create_volume_squeeze_detector_yf() -> VolumeSqueezeDetectorYF:
    """
    Factory function to create Volume Squeeze Detector using yfinance.

    Returns:
        VolumeSqueezeDetectorYF instance
    """
    _logger.info("Creating Volume Squeeze Detector with yfinance")
    return VolumeSqueezeDetectorYF()
