# src/common/sentiments/processing/calibration.py
"""
Per-source sentiment calibration (sentiment-spec-rev2.md §2.5.6).

Raw sentiment scores are not comparable across platforms: Bluesky finance chatter skews
promotional-positive, Hacker News skews critical-negative. Blending raw values imports that
platform bias directly into any downstream score. This module answers the question that actually
matters -- *is this ticker unusually positive for this platform?* -- by converting a raw score to
a z-score against that provider's own trailing distribution before it is blended with anything
else.

This module is pure/DB-agnostic by design (mirrors ``collect_sentiment_async.py``'s
``history_lookup`` callback pattern): callers own persistence and inject a
``CalibrationStats`` lookup, so the standalone sentiment collector never takes a hard DB
dependency.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List, Literal

CalibrationStatus = Literal["ok", "insufficient_history"]


@dataclass(frozen=True)
class CalibrationStats:
    """Pooled trailing-window sentiment distribution for one provider."""

    mean: float
    std: float
    n_obs: int


@dataclass(frozen=True)
class DailyObservation:
    """One day's raw-score distribution for one provider, as persisted in ``ss_sentiment_calibration``."""

    provider: str
    day: str  # ISO date, "YYYY-MM-DD"
    mean_score: float
    std_score: float
    n_obs: int


def compute_daily_stats(scores: List[float]) -> tuple[float, float, int]:
    """
    Compute today's (mean, std, n) for one provider's raw scores across a batch.

    Args:
        scores: Raw ``sentiment_score`` values (-1..+1) collected across tickers in one batch run.

    Returns:
        ``(mean, std, n)``. ``std`` is 0.0 for n < 2 (population std of a single point is
        undefined; the pooling in :func:`pool_daily_observations` treats it as zero-variance
        rather than dropping the observation).
    """
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0
    mean = statistics.fmean(scores)
    std = statistics.pstdev(scores) if n >= 2 else 0.0
    return mean, std, n


def pool_daily_observations(observations: List[DailyObservation]) -> CalibrationStats | None:
    """
    Pool a trailing window of daily (mean, std, n) rows into one distribution.

    Uses the standard pooled-mean / pooled-variance formulas (weighted by each day's sample
    count) rather than a naive average-of-averages, so days with more observations correctly
    dominate the trailing estimate.

    Args:
        observations: Daily calibration rows for one provider, typically the trailing
            ``window_days`` (spec default 30) from ``ss_sentiment_calibration``.

    Returns:
        Pooled :class:`CalibrationStats`, or ``None`` if there are no observations at all.
    """
    total_n = sum(o.n_obs for o in observations)
    if total_n == 0:
        return None

    pooled_mean = sum(o.mean_score * o.n_obs for o in observations) / total_n

    # Pooled variance: within-day variance plus between-day mean spread, both weighted by n.
    pooled_var = sum(
        o.n_obs * (o.std_score**2 + (o.mean_score - pooled_mean) ** 2) for o in observations
    ) / total_n

    return CalibrationStats(mean=pooled_mean, std=pooled_var**0.5, n_obs=total_n)


def calibration_status(stats: CalibrationStats | None, min_observations: int) -> CalibrationStatus:
    """
    Return ``"ok"`` once a provider has accumulated at least ``min_observations`` trailing
    observations, else ``"insufficient_history"`` -- callers fall back to raw scores in the
    latter case (spec §2.5.6).
    """
    if stats is None or stats.n_obs < min_observations:
        return "insufficient_history"
    return "ok"


def calibrate_score(raw_score: float, stats: CalibrationStats | None, min_observations: int, eps: float = 1e-6) -> float:
    """
    Convert a raw per-ticker sentiment score to a z-score against its provider's trailing
    distribution (spec §2.5.6: ``calibrated = (raw - source_mean_30d) / (source_std_30d + eps)``).

    Falls back to the raw score, unchanged, when history is insufficient (also unchanged when
    ``stats`` is ``None``, e.g. a provider with no calibration history at all yet) -- callers
    should pair this with :func:`calibration_status` to record ``data_quality.calibration``.

    Note the z-score is not bounded to [-1, 1] like the raw score is; it is a "how many standard
    deviations from this platform's own norm" value, on purpose (that's the whole point of
    calibrating before blending).
    """
    if calibration_status(stats, min_observations) == "insufficient_history" or stats is None:
        return raw_score
    return (raw_score - stats.mean) / (stats.std + eps)
