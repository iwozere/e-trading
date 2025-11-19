"""
Short Squeeze Detection Pipeline Scoring Engine

This module implements the scoring engine that combines structural and transient metrics
to calculate comprehensive squeeze probability scores.
"""

from pathlib import Path
import sys
from typing import Dict, Any
import math

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

from src.ml.pipeline.p04_short_squeeze.core.models import (
    StructuralMetrics, TransientMetrics, ScoredCandidate, Candidate
)
from src.ml.pipeline.p04_short_squeeze.config.data_classes import ScoringConfig, DeepScanWeights
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ScoringEngine:
    """
    Scoring engine for calculating squeeze probability scores.

    Combines structural and transient metrics using configurable weights
    and normalization methods to produce final squeeze scores.
    """

    def __init__(self, config: ScoringConfig, deep_scan_weights: DeepScanWeights):
        """
        Initialize the scoring engine.

        Args:
            config: Scoring configuration
            deep_scan_weights: Weights for deep scan metrics
        """
        self.config = config
        self.weights = deep_scan_weights
        self._logger = setup_logger(f"{__name__}.ScoringEngine")

        # Validate configuration
        self._validate_config()

        _logger.info("ScoringEngine initialized with normalization method: %s",
                    self.config.normalization_method)

    def _validate_config(self) -> None:
        """Validate scoring configuration."""
        if self.config.weight_validation:
            # Check that weights sum to approximately 1.0
            total_weight = (
                self.weights.volume_spike +
                self.weights.sentiment_24h +
                self.weights.call_put_ratio +
                self.weights.borrow_fee
            )

            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Deep scan weights must sum to 1.0, got {total_weight}")

        # Validate score bounds
        if len(self.config.score_bounds) != 2:
            raise ValueError("Score bounds must be a tuple of (min, max)")

        min_bound, max_bound = self.config.score_bounds
        if min_bound >= max_bound:
            raise ValueError("Score bounds min must be less than max")

    def calculate_squeeze_score(self,
                              structural: StructuralMetrics,
                              transient: TransientMetrics) -> float:
        """
        Calculate comprehensive squeeze probability score.

        Args:
            structural: Structural metrics from screener
            transient: Transient metrics from deep scan

        Returns:
            Squeeze probability score between 0 and 1
        """
        try:
            # Extract and normalize structural metrics
            structural_score = self._calculate_structural_score(structural)

            # Extract and normalize transient metrics
            transient_metrics = self._extract_transient_metrics(transient)
            normalized_transient = self._normalize_metrics(transient_metrics)

            # Apply weights to transient metrics
            weighted_transient_score = self._apply_weights(normalized_transient)

            # Combine structural and transient scores
            # Use 60% transient (current conditions) and 40% structural (baseline)
            final_score = 0.6 * weighted_transient_score + 0.4 * structural_score

            # Apply bounds checking
            final_score = self._validate_score_bounds(final_score)

            self._logger.debug(
                "Calculated squeeze score: structural=%.3f, transient=%.3f, final=%.3f",
                structural_score, weighted_transient_score, final_score
            )

            return final_score

        except Exception:
            self._logger.exception("Error calculating squeeze score:")
            # Return minimum score on error
            return self.config.score_bounds[0]

    def _calculate_structural_score(self, structural: StructuralMetrics) -> float:
        """
        Calculate normalized structural score from screener metrics.

        Args:
            structural: Structural metrics

        Returns:
            Normalized structural score (0-1)
        """
        # Use the screener score as base, but enhance with key structural indicators

        # Short interest percentage contribution (higher is better)
        si_score = min(structural.short_interest_pct / 0.5, 1.0)  # Cap at 50%

        # Days to cover contribution (higher is better, but with diminishing returns)
        dtc_score = min(structural.days_to_cover / 10.0, 1.0)  # Cap at 10 days

        # Float ratio (smaller float relative to market cap is better)
        # Calculate as inverse of float/market_cap ratio
        float_ratio = structural.float_shares / max(structural.market_cap / 1000000, 1)  # Per million market cap
        float_score = max(0, 1.0 - min(float_ratio / 100, 1.0))  # Inverse relationship

        # Volume consistency (higher average volume provides more liquidity for squeeze)
        # Normalize by market cap to account for company size
        volume_per_mcap = structural.avg_volume_14d / max(structural.market_cap / 1000000, 1)
        volume_score = min(volume_per_mcap / 50, 1.0)  # Cap at 50 volume per million market cap

        # Weighted combination of structural factors
        structural_score = (
            0.4 * si_score +
            0.3 * dtc_score +
            0.2 * float_score +
            0.1 * volume_score
        )

        return max(0.0, min(1.0, structural_score))

    def _extract_transient_metrics(self, transient: TransientMetrics) -> Dict[str, float]:
        """
        Extract transient metrics into a dictionary for processing.

        Args:
            transient: Transient metrics

        Returns:
            Dictionary of metric name to value
        """
        metrics = {
            'volume_spike': transient.volume_spike,
            'sentiment_24h': transient.sentiment_24h,
            'call_put_ratio': transient.call_put_ratio or 0.0,  # Default to 0 if None
            'borrow_fee': transient.borrow_fee_pct or 0.0,  # Default to 0 if None
            'virality_index': transient.virality_index,
            'mentions_growth_7d': transient.mentions_growth_7d or 0.0,  # Default to 0 if None
            'bot_pct': transient.bot_pct
        }

        return metrics

    def normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Public interface for metric normalization.

        Args:
            metrics: Dictionary of metric name to value

        Returns:
            Dictionary of normalized metrics (0-1 scale)
        """
        return self._normalize_metrics(metrics)

    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize metrics to 0-1 scale using configured method.

        Args:
            metrics: Dictionary of metric name to value

        Returns:
            Dictionary of normalized metrics
        """
        normalized = {}

        for metric_name, value in metrics.items():
            if self.config.normalization_method == "minmax":
                normalized[metric_name] = self._minmax_normalize(metric_name, value)
            elif self.config.normalization_method == "sigmoid":
                normalized[metric_name] = self._sigmoid_normalize(metric_name, value)
            else:
                raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")

        return normalized

    def _minmax_normalize(self, metric_name: str, value: float) -> float:
        """
        Min-max normalization for specific metrics.

        Args:
            metric_name: Name of the metric
            value: Raw metric value

        Returns:
            Normalized value (0-1)
        """
        # Define expected ranges for each metric
        ranges = {
            'volume_spike': (1.0, 10.0),  # 1x to 10x volume spike
            'sentiment_24h': (-1.0, 1.0),  # Already normalized
            'call_put_ratio': (0.0, 5.0),  # 0 to 5 call/put ratio
            'borrow_fee': (0.0, 50.0),  # 0% to 50% borrow fee
            'virality_index': (0.0, 1.0),  # Already normalized
            'mentions_growth_7d': (-1.0, 10.0),  # -100% to +1000% growth
            'bot_pct': (0.0, 1.0)  # Already normalized (inverse)
        }

        if metric_name not in ranges:
            self._logger.warning("Unknown metric for normalization: %s", metric_name)
            return 0.0

        min_val, max_val = ranges[metric_name]

        # Handle already normalized metrics
        if metric_name == 'sentiment_24h':
            # Convert from [-1, 1] to [0, 1]
            return (value + 1.0) / 2.0
        elif metric_name in ['virality_index', 'bot_pct']:
            # Already in [0, 1] range
            return value
        elif metric_name == 'mentions_growth_7d':
            # Growth metric: 0 = no change, positive = growth, negative = decline
            # Normalize so 0 growth = 0.5, 10x growth = 1.0, -100% = 0.0
            normalized = (value - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))

        # Standard min-max normalization
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def _sigmoid_normalize(self, metric_name: str, value: float) -> float:
        """
        Sigmoid normalization for specific metrics.

        Args:
            metric_name: Name of the metric
            value: Raw metric value

        Returns:
            Normalized value (0-1)
        """
        # Define sigmoid parameters for each metric
        params = {
            'volume_spike': (3.0, 2.0),  # center=3x, steepness=2
            'sentiment_24h': (0.0, 4.0),  # center=0, steepness=4
            'call_put_ratio': (1.5, 1.5),  # center=1.5, steepness=1.5
            'borrow_fee': (10.0, 0.2)  # center=10%, steepness=0.2
        }

        if metric_name not in params:
            self._logger.warning("Unknown metric for sigmoid normalization: %s", metric_name)
            return 0.0

        center, steepness = params[metric_name]

        # Handle sentiment which is already normalized
        if metric_name == 'sentiment_24h':
            # Apply sigmoid and convert to [0, 1]
            sigmoid_val = 1 / (1 + math.exp(-steepness * (value - center)))
            return sigmoid_val

        # Standard sigmoid normalization
        try:
            sigmoid_val = 1 / (1 + math.exp(-steepness * (value - center)))
            return sigmoid_val
        except OverflowError:
            # Handle extreme values
            return 1.0 if value > center else 0.0

    def _apply_weights(self, normalized_metrics: Dict[str, float]) -> float:
        """
        Apply configured weights to normalized metrics.

        Args:
            normalized_metrics: Dictionary of normalized metrics

        Returns:
            Weighted score
        """
        weighted_score = (
            self.weights.volume_spike * normalized_metrics.get('volume_spike', 0.0) +
            self.weights.sentiment_24h * normalized_metrics.get('sentiment_24h', 0.0) +
            self.weights.call_put_ratio * normalized_metrics.get('call_put_ratio', 0.0) +
            self.weights.borrow_fee * normalized_metrics.get('borrow_fee', 0.0)
        )

        return weighted_score

    def _validate_score_bounds(self, score: float) -> float:
        """
        Validate and enforce score bounds.

        Args:
            score: Raw calculated score

        Returns:
            Score within configured bounds
        """
        min_bound, max_bound = self.config.score_bounds

        if score < min_bound:
            self._logger.debug("Score %.3f below minimum bound %.3f, clamping", score, min_bound)
            return min_bound
        elif score > max_bound:
            self._logger.debug("Score %.3f above maximum bound %.3f, clamping", score, max_bound)
            return max_bound

        return score

    def score_candidate(self, candidate: Candidate, transient: TransientMetrics) -> ScoredCandidate:
        """
        Score a candidate with transient metrics.

        Args:
            candidate: Candidate with structural metrics
            transient: Transient metrics from deep scan

        Returns:
            Scored candidate with squeeze score
        """
        try:
            squeeze_score = self.calculate_squeeze_score(
                candidate.structural_metrics,
                transient
            )

            scored_candidate = ScoredCandidate(
                candidate=candidate,
                transient_metrics=transient,
                squeeze_score=squeeze_score
            )

            self._logger.debug(
                "Scored candidate %s: squeeze_score=%.3f",
                candidate.ticker, squeeze_score
            )

            return scored_candidate

        except Exception as e:
            self._logger.error("Error scoring candidate %s: %s", candidate.ticker, e)
            # Return candidate with minimum score on error
            return ScoredCandidate(
                candidate=candidate,
                transient_metrics=transient,
                squeeze_score=self.config.score_bounds[0]
            )

    def get_score_breakdown(self,
                          structural: StructuralMetrics,
                          transient: TransientMetrics) -> Dict[str, Any]:
        """
        Get detailed breakdown of score calculation for analysis.

        Args:
            structural: Structural metrics
            transient: Transient metrics

        Returns:
            Dictionary with score breakdown details
        """
        try:
            # Calculate components
            structural_score = self._calculate_structural_score(structural)
            transient_metrics = self._extract_transient_metrics(transient)
            normalized_transient = self._normalize_metrics(transient_metrics)
            weighted_transient_score = self._apply_weights(normalized_transient)
            final_score = 0.6 * weighted_transient_score + 0.4 * structural_score
            final_score = self._validate_score_bounds(final_score)

            return {
                'final_score': final_score,
                'structural_score': structural_score,
                'transient_score': weighted_transient_score,
                'raw_transient_metrics': transient_metrics,
                'normalized_transient_metrics': normalized_transient,
                'weights': {
                    'volume_spike': self.weights.volume_spike,
                    'sentiment_24h': self.weights.sentiment_24h,
                    'call_put_ratio': self.weights.call_put_ratio,
                    'borrow_fee': self.weights.borrow_fee
                },
                'combination_weights': {
                    'transient': 0.6,
                    'structural': 0.4
                }
            }

        except Exception as e:
            self._logger.exception("Error generating score breakdown:")
            return {
                'final_score': self.config.score_bounds[0],
                'error': str(e)
            }

    def calculate_virality_score_with_bot_penalty(self, virality_index: float, bot_pct: float) -> float:
        """
        Calculate virality score with bot activity penalty.

        Higher bot percentage reduces the effective virality score.

        Args:
            virality_index: Raw virality index (0-1)
            bot_pct: Bot percentage (0-1)

        Returns:
            Adjusted virality score (0-1)
        """
        # Apply bot penalty: reduce score proportionally to bot activity
        # Max 60% penalty for 100% bot activity
        bot_penalty = 1.0 - (bot_pct * 0.6)

        adjusted_score = virality_index * bot_penalty

        return max(0.0, min(1.0, adjusted_score))

    def apply_mentions_growth_boost(self, base_score: float, mentions_growth_7d: float) -> float:
        """
        Apply boost to final score based on mention growth.

        Rapid mention growth is a strong early indicator of squeeze activity.

        Args:
            base_score: Base calculated score (0-1)
            mentions_growth_7d: 7-day mention growth ratio (e.g., 3.0 = 300% growth)

        Returns:
            Boosted score (0-1)
        """
        if mentions_growth_7d <= 1.0:
            # No boost for growth <= 100%
            return base_score

        # Apply progressive boost for high growth
        # 2x growth = +5% boost, 5x = +10%, 10x+ = +15%
        if mentions_growth_7d >= 10.0:
            boost = 0.15
        elif mentions_growth_7d >= 5.0:
            boost = 0.10
        elif mentions_growth_7d >= 3.0:
            boost = 0.07
        else:  # 1x - 3x
            boost = 0.05

        boosted_score = base_score * (1.0 + boost)

        return min(1.0, boosted_score)  # Cap at 1.0