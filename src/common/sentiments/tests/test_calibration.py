# src/common/sentiments/tests/test_calibration.py
"""
Unit tests for src.common.sentiments.processing.calibration.

Covers:
- compute_daily_stats / pool_daily_observations against a synthetic known-mean/std distribution
- calibration_status insufficient-history fallback
- calibrate_score z-score math, including the insufficient-history raw-score fallback
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.sentiments.processing.calibration import (
    CalibrationStats,
    DailyObservation,
    calibrate_score,
    calibration_status,
    compute_daily_stats,
    pool_daily_observations,
)


class TestComputeDailyStats:
    def test_empty(self):
        assert compute_daily_stats([]) == (0.0, 0.0, 0)

    def test_single_value_zero_std(self):
        mean, std, n = compute_daily_stats([0.5])
        assert mean == 0.5
        assert std == 0.0
        assert n == 1

    def test_known_distribution(self):
        # Population mean/std of [1, 2, 3, 4, 5] are 3.0 and sqrt(2) (~1.41421356).
        mean, std, n = compute_daily_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mean == 3.0
        assert abs(std - 2**0.5) < 1e-9
        assert n == 5


class TestPoolDailyObservations:
    def test_no_observations(self):
        assert pool_daily_observations([]) is None

    def test_zero_n_obs_rows_pool_to_none(self):
        obs = [DailyObservation(provider="bluesky", day="2026-08-01", mean_score=0.0, std_score=0.0, n_obs=0)]
        assert pool_daily_observations(obs) is None

    def test_single_day_matches_its_own_stats(self):
        obs = [DailyObservation(provider="bluesky", day="2026-08-01", mean_score=0.2, std_score=0.5, n_obs=100)]
        stats = pool_daily_observations(obs)
        assert stats is not None
        assert stats.mean == 0.2
        assert abs(stats.std - 0.5) < 1e-9
        assert stats.n_obs == 100

    def test_pooling_weights_by_sample_count(self):
        # A big low-mean day should dominate a tiny high-mean day.
        obs = [
            DailyObservation(provider="bluesky", day="2026-08-01", mean_score=0.0, std_score=0.1, n_obs=990),
            DailyObservation(provider="bluesky", day="2026-08-02", mean_score=1.0, std_score=0.1, n_obs=10),
        ]
        stats = pool_daily_observations(obs)
        assert stats is not None
        assert stats.n_obs == 1000
        # Weighted mean: (0.0*990 + 1.0*10) / 1000 = 0.01
        assert abs(stats.mean - 0.01) < 1e-9

    def test_identical_days_reduce_to_their_own_std(self):
        # No between-day spread (means identical) -> pooled std equals the shared within-day std.
        obs = [
            DailyObservation(provider="hackernews", day="2026-08-01", mean_score=0.1, std_score=0.3, n_obs=50),
            DailyObservation(provider="hackernews", day="2026-08-02", mean_score=0.1, std_score=0.3, n_obs=50),
        ]
        stats = pool_daily_observations(obs)
        assert stats is not None
        assert abs(stats.mean - 0.1) < 1e-9
        assert abs(stats.std - 0.3) < 1e-9


class TestCalibrationStatus:
    def test_none_stats_is_insufficient(self):
        assert calibration_status(None, min_observations=200) == "insufficient_history"

    def test_below_threshold_is_insufficient(self):
        stats = CalibrationStats(mean=0.0, std=0.2, n_obs=199)
        assert calibration_status(stats, min_observations=200) == "insufficient_history"

    def test_at_threshold_is_ok(self):
        stats = CalibrationStats(mean=0.0, std=0.2, n_obs=200)
        assert calibration_status(stats, min_observations=200) == "ok"

    def test_above_threshold_is_ok(self):
        stats = CalibrationStats(mean=0.0, std=0.2, n_obs=5000)
        assert calibration_status(stats, min_observations=200) == "ok"


class TestCalibrateScore:
    def test_no_history_returns_raw_score_unchanged(self):
        assert calibrate_score(0.42, None, min_observations=200) == 0.42

    def test_insufficient_history_returns_raw_score_unchanged(self):
        stats = CalibrationStats(mean=0.5, std=0.1, n_obs=50)
        assert calibrate_score(0.42, stats, min_observations=200) == 0.42

    def test_z_score_matches_hand_computed_value(self):
        # raw=0.7, mean=0.5, std=0.1 -> z = (0.7 - 0.5) / (0.1 + eps) ≈ 1.99998
        stats = CalibrationStats(mean=0.5, std=0.1, n_obs=1000)
        calibrated = calibrate_score(0.7, stats, min_observations=200, eps=1e-6)
        assert abs(calibrated - 2.0) < 1e-3

    def test_score_at_mean_calibrates_to_zero(self):
        stats = CalibrationStats(mean=0.3, std=0.2, n_obs=1000)
        calibrated = calibrate_score(0.3, stats, min_observations=200)
        assert abs(calibrated) < 1e-6

    def test_negative_z_score_for_below_mean_platform_norm(self):
        # A raw score that is neutral in absolute terms can be unusually *negative* for a
        # platform that skews positive (e.g. Bluesky) -- this is the whole point of calibrating
        # per source before blending (spec §2.5.6).
        stats = CalibrationStats(mean=0.6, std=0.1, n_obs=1000)
        calibrated = calibrate_score(0.0, stats, min_observations=200)
        assert calibrated < -5.0

    def test_zero_std_does_not_divide_by_zero(self):
        stats = CalibrationStats(mean=0.5, std=0.0, n_obs=1000)
        # Should not raise, and should still resolve via eps.
        calibrated = calibrate_score(0.6, stats, min_observations=200, eps=1e-6)
        assert calibrated > 0
