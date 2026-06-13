"""
37-scenario threshold calibration suite for AccumulationAnalyzer._check_accumulation.

Encodes the funnel statistics from 37 historical production runs
(2026-03-13 → 2026-05-19, signal-analysis.md) as executable regression tests.
Each scenario targets exactly one filter gate, holding all other metrics
comfortably inside their passing ranges.

Scenario groups:
  A (5)  — degenerate / invalid data
  B (4)  — vol_zscore gate          (config.min_vol_zscore = 1.5)
  C (3)  — price_range_1d gate      (config.max_price_impact = 0.05)
  D (3)  — atr_ratio gate           (config.max_atr_ratio = 0.04)
  E (4)  — absorption ratio gate    (config.min_vol_rv_ratio = 1.5)
  F (4)  — price_change_1d gate     (hardcoded 0.035)
  G (4)  — SMA20 distance gate      (config.max_distance_from_sma20 = 0.10)
  H (4)  — resistance gate          (config.max_distance_from_resistance = 0.15)
  I (6)  — threshold grid sweep     (tight vs relaxed configs)
                                     Total: 37
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p06_emps2.accumulation_analyzer import AccumulationAnalyzer
from src.ml.pipeline.p06_emps2.config import EMPS2FilterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> EMPS2FilterConfig:
    """Return the current (post-recalibration) EMPS2FilterConfig with optional overrides."""
    cfg = EMPS2FilterConfig()
    cfg.interval = "1h"
    cfg.atr_period = 14
    cfg.min_vol_zscore = 1.5
    cfg.min_vol_rv_ratio = 1.5
    cfg.max_price_impact = 0.05
    cfg.max_atr_ratio = 0.04
    cfg.max_distance_from_resistance = 0.15
    cfg.max_distance_from_sma20 = 0.10
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_analyzer(tmp_path: Path, **cfg_overrides) -> AccumulationAnalyzer:
    return AccumulationAnalyzer(
        data_manager=MagicMock(),
        config=_make_config(**cfg_overrides),
        results_dir=tmp_path,
    )


def _make_daily(
    n: int = 60,
    price: float = 50.0,
    vol_mult: float = 4.0,
    hist_spread: float = 0.01,
    last_spread: float = 0.01,
    price_slope: float = 0.0,
    local_high_boost: float = 0.0,
    price_change_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Synthetic daily OHLCV with fine-grained metric controls.

    Metric approximations (see signal-analysis.md §6.1):
      vol_zscore       > 0 when vol_mult > 1; < 0 when vol_mult < 1
      price_range_1d  ≈ last_spread            (last bar H-L / L)
      atr_ratio       ≈ 2 * hist_spread        (Wilder ATR at warmup)
      dist_sma_20     ≈ 9.5*s / (1+49.5*s)    where s=price_slope
      dist_local_high ≈ local_high_boost / (1+local_high_boost)
      price_change_1d  = price_change_pct       (last two closes differ by this)

    Volumes: n-1 bars at 1_000_000; last bar at 1_000_000 * vol_mult.
    With deterministic equal baselines, vol_zscore sign equals sign(vol_mult - 1).
    """
    dates = pd.date_range("2025-01-01", periods=n, freq="1D")
    # Geometric (per-bar %) growth so atr_ratio ≈ hist_spread + price_slope
    # regardless of bar index.  Linear growth inflates early-bar TR ratios.
    closes = np.array([price * (1.0 + price_slope) ** i for i in range(n)])

    if price_change_pct != 0.0:
        closes[-2] = closes[-1] / (1.0 + price_change_pct)

    highs = closes * (1.0 + hist_spread)
    lows = closes * (1.0 - hist_spread)
    # Override last bar so price_range_1d ≠ 2*hist_spread
    highs[-1] = closes[-1] * (1.0 + last_spread / 2.0)
    lows[-1] = closes[-1] * (1.0 - last_spread / 2.0)

    if local_high_boost > 0.0:
        boost_idx = n - 10  # sits inside the last-20-bar window
        highs[boost_idx] = closes[boost_idx] * (1.0 + local_high_boost)

    volume = np.full(n, 1_000_000.0)
    volume[-1] = 1_000_000.0 * vol_mult

    return pd.DataFrame({
        "timestamp": dates,
        "open":   closes,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volume,
    })


def _make_intraday(
    n: int = 40,
    price: float = 50.0,
    noise_pct: float = 0.001,
    zero_at: int = -1,
) -> pd.DataFrame:
    """
    Synthetic 1h intraday OHLCV.

    rv ≈ noise_pct * sqrt(252 * 6.5) ≈ noise_pct * 40.5
    ar = vol_zscore / rv → high noise_pct lowers AR below the 1.5 threshold.
    """
    rng = np.random.default_rng(1)
    dates = pd.date_range("2025-01-01 09:30", periods=n, freq="1h")
    close = price + rng.normal(0, price * noise_pct, size=n)
    close = np.clip(close, 0.001, None)
    if zero_at >= 0:
        close[zero_at] = 0.0
    return pd.DataFrame({
        "timestamp": dates,
        "open":   close,
        "high":   close * 1.001,
        "low":    close * 0.999,
        "close":  close,
        "volume": np.full(n, 50_000.0),
    })


def _check(
    tmp_path: Path,
    df_daily: pd.DataFrame,
    df_intra: pd.DataFrame,
    ticker: str = "TEST",
    **cfg_overrides,
) -> tuple:
    """Run _check_accumulation and return (passed, metrics, reason)."""
    analyzer = _make_analyzer(tmp_path, **cfg_overrides)
    return analyzer._check_accumulation(ticker, df_intra, df_daily)


# ---------------------------------------------------------------------------
# Group A — degenerate / invalid data (5 scenarios)
# ---------------------------------------------------------------------------

class TestGroupA_DegenerateData:

    def test_a1_nan_rv_from_zero_intraday_close(self, tmp_path):
        """Zero close price in intraday bars produces NaN log-return → rv=NaN → rejected."""
        daily = _make_daily(vol_mult=4.0, last_spread=0.01, hist_spread=0.01)
        intra = _make_intraday(n=40, zero_at=15)  # zero inside rv window
        passed, _, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "nan_metrics"

    def test_a2_negative_vol_zscore_volume_drought(self, tmp_path):
        """Last bar volume at 5% of baseline → deeply negative zscore → rejected."""
        daily = _make_daily(vol_mult=0.05)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "low_volume_zscore"
        assert metrics["vol_zscore"] < 0

    def test_a3_xrxdw_penny_warrant_regression(self, tmp_path):
        """
        XRXDW regression: ~$0.11 warrant with zero intraday close produces NaN rv.
        Pre-fix: NaN passed all filters silently. Post-fix: rejected as nan_metrics.
        """
        daily = _make_daily(price=0.11, vol_mult=3.0)
        intra = _make_intraday(n=40, price=0.11, zero_at=15)
        passed, _, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "nan_metrics"

    def test_a4_extremely_low_price_nan_atr(self, tmp_path):
        """Near-zero price ($0.001) causes ATR computation to produce NaN/zero atr_ratio."""
        daily = _make_daily(price=0.001, vol_mult=4.0)
        intra = _make_intraday(price=0.001, zero_at=10)
        passed, _, reason = _check(tmp_path, daily, intra)
        assert not passed

    def test_a5_intraday_too_few_bars_for_rv_window(self, tmp_path):
        """
        Only 15 intraday bars available; rv_bars_target = int(6.5*5) = 32.
        When len(df_intra) < rv_bars_target, rv=0 → ar=0 → rejected.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday(n=15)  # 15 < 32 rv_bars_target
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert metrics["absorption_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group B — vol_zscore gate (4 scenarios)
# ---------------------------------------------------------------------------

class TestGroupB_VolZscoreGate:

    def test_b1_vol_clearly_below_average(self, tmp_path):
        """vol_mult=0.10 → large negative zscore → rejected as low_volume_zscore."""
        daily = _make_daily(vol_mult=0.10)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "low_volume_zscore"
        assert metrics["vol_zscore"] < 0

    def test_b2_vol_at_half_of_average(self, tmp_path):
        """vol_mult=0.50 → negative zscore → rejected as low_volume_zscore."""
        daily = _make_daily(vol_mult=0.50)
        intra = _make_intraday()
        passed, _, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "low_volume_zscore"

    def test_b3_vol_three_times_average_passes_gate(self, tmp_path):
        """vol_mult=3.0 → positive zscore well above 1.5 → passes vol gate."""
        daily = _make_daily(vol_mult=3.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday()
        passed, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["vol_zscore"] > 1.5

    def test_b4_vol_ten_times_average_passes_gate(self, tmp_path):
        """vol_mult=10.0 → very high positive zscore → passes vol gate."""
        daily = _make_daily(vol_mult=10.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["vol_zscore"] > 1.5


# ---------------------------------------------------------------------------
# Group C — price_range_1d gate (3 scenarios)
# ---------------------------------------------------------------------------

class TestGroupC_PriceRangeGate:

    def test_c1_wide_last_bar_12_pct_range(self, tmp_path):
        """Last bar H-L spread 12% → price_range_1d ≈ 0.12 >> 0.05 → poor_price_compression."""
        # hist_spread=0.01 keeps atr_ratio ≈ 0.02 (< 0.04) so only price_range fires
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.12)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "poor_price_compression"
        assert metrics["price_range_1d"] >= 0.05

    def test_c2_wide_last_bar_8_pct_range(self, tmp_path):
        """Last bar H-L spread 8% → price_range_1d ≈ 0.08 > 0.05 → poor_price_compression."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.08)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "poor_price_compression"
        assert metrics["price_range_1d"] >= 0.05

    def test_c3_tight_last_bar_2_pct_passes_price_gate(self, tmp_path):
        """Last bar H-L spread 2% → price_range_1d ≈ 0.02 < 0.05 → passes price_range gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.02)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["price_range_1d"] < 0.05


# ---------------------------------------------------------------------------
# Group D — atr_ratio gate (3 scenarios)
# ---------------------------------------------------------------------------

class TestGroupD_AtrRatioGate:

    def test_d1_high_atr_spread_10_pct(self, tmp_path):
        """
        hist_spread=0.05 → ATR ≈ 0.10*price → atr_ratio ≈ 0.10 > 0.04.
        last_spread=0.02 keeps price_range_1d=0.02 < 0.05 (only ATR fires).
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.05, last_spread=0.02)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "poor_price_compression"
        assert metrics["atr_ratio"] > 0.04

    def test_d2_moderate_atr_spread_5_pct(self, tmp_path):
        """
        hist_spread=0.025 → atr_ratio ≈ 0.05 > 0.04.
        last_spread=0.015 keeps price_range_1d < 0.05.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.025, last_spread=0.015)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "poor_price_compression"
        assert metrics["atr_ratio"] > 0.04

    def test_d3_low_atr_spread_3_pct_passes_gate(self, tmp_path):
        """hist_spread=0.015 → atr_ratio ≈ 0.03 < 0.04 → passes ATR gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.015, last_spread=0.015)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["atr_ratio"] < 0.04


# ---------------------------------------------------------------------------
# Group E — absorption ratio gate (4 scenarios)
# ---------------------------------------------------------------------------

class TestGroupE_AbsorptionRatioGate:

    def test_e1_high_intraday_noise_lowers_ar(self, tmp_path):
        """
        noise_pct=0.12 → rv ≈ 0.12*40.5 ≈ 4.86 → ar = zscore/rv ≈ 4.36/4.86 ≈ 0.90 < 1.5.
        vol gate passes (mult=4.0), price gates pass (spreads=0.01), AR fails.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday(noise_pct=0.12)
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "low_absorption_ratio"
        assert metrics["absorption_ratio"] < 1.5

    def test_e2_zero_rv_too_few_intraday_bars(self, tmp_path):
        """15 intraday bars < rv_bars_target=32 → rv=0.0 → ar=0.0 → low_absorption_ratio."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday(n=15)
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert metrics["absorption_ratio"] == pytest.approx(0.0)

    def test_e3_low_intraday_noise_high_ar(self, tmp_path):
        """noise_pct=0.001 → rv ≈ 0.04 → ar ≈ 107 >> 1.5 → passes AR gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday(noise_pct=0.001)
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["absorption_ratio"] > 1.5

    def test_e4_moderate_noise_still_passes_ar(self, tmp_path):
        """
        noise_pct=0.05 → rv ≈ 2.0 → ar = 4.36/2.0 ≈ 2.18 > 1.5.
        Confirms the AR gate is not over-sensitive to moderate intraday volatility.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday(noise_pct=0.05)
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["absorption_ratio"] > 1.5


# ---------------------------------------------------------------------------
# Group F — price_change_1d gate (4 scenarios)
# ---------------------------------------------------------------------------

class TestGroupF_PriceChangeGate:

    def test_f1_large_gap_5_pct(self, tmp_path):
        """5% gap between yesterday's close and today's → price_change_too_high."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_change_pct=0.05)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "price_change_too_high"
        assert metrics["price_change_1d"] > 0.035

    def test_f2_borderline_gap_4_pct(self, tmp_path):
        """4% gap → price_change_1d=0.04 > 0.035 → price_change_too_high."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_change_pct=0.04)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "price_change_too_high"
        assert metrics["price_change_1d"] > 0.035

    def test_f3_small_gap_2_pct_passes(self, tmp_path):
        """2% gap → price_change_1d=0.02 < 0.035 → passes price_change gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_change_pct=0.02)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["price_change_1d"] < 0.035

    def test_f4_no_gap_passes(self, tmp_path):
        """Flat close (price_change_pct=0) → price_change_1d=0 → passes gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_change_pct=0.0)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["price_change_1d"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group G — SMA20 distance gate (4 scenarios)
# ---------------------------------------------------------------------------

class TestGroupG_SMA20DistanceGate:
    """
    Uses geometric (per-bar %) price slope so atr_ratio ≈ hist_spread + slope,
    which stays below 0.04 when both are small.

    dist_sma_20 ≈ exp(9.5 * slope) − 1  (geometric sum approximation)

    For threshold 0.10:
      slope = 0.019 → dist ≈ 0.20  (fails clearly)
      slope = 0.012 → dist ≈ 0.12  (fails)
      slope = 0.008 → dist ≈ 0.078 (passes)
      slope = 0.000 → dist = 0.0   (passes)

    ATR-safe because atr_ratio ≈ 0.01 + slope ≤ 0.029 < 0.04.
    price_change_1d ≈ slope < 0.035 for all these values.
    """

    def test_g1_rising_trend_far_above_sma20(self, tmp_path):
        """price_slope=0.019 → dist_sma_20 ≈ 0.19 >> 0.10 → too_far_from_sma20."""
        slope = 0.019
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_slope=slope)
        # Intraday price must match the daily last-close so atr_ratio = ATR/intraday_close
        # uses the same scale.  last_price in _check_accumulation comes from df_intra.
        intra = _make_intraday(price=50.0 * (1.0 + slope) ** 59)
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "too_far_from_sma20"
        assert metrics["dist_sma_20"] > 0.10

    def test_g2_moderate_trend_still_too_far(self, tmp_path):
        """price_slope=0.012 → dist_sma_20 ≈ 0.12 > 0.10 → too_far_from_sma20."""
        slope = 0.012
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_slope=slope)
        intra = _make_intraday(price=50.0 * (1.0 + slope) ** 59)
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "too_far_from_sma20"
        assert metrics["dist_sma_20"] > 0.10

    def test_g3_gentle_trend_within_sma20_band(self, tmp_path):
        """price_slope=0.008 → dist_sma_20 ≈ 0.078 < 0.10 → passes SMA20 gate."""
        slope = 0.008
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_slope=slope)
        intra = _make_intraday(price=50.0 * (1.0 + slope) ** 59)
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["dist_sma_20"] < 0.10

    def test_g4_flat_price_zero_sma20_distance(self, tmp_path):
        """price_slope=0 → price == SMA20 → dist_sma_20=0 → passes gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            price_slope=0.0)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["dist_sma_20"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Group H — resistance (local high) gate (4 scenarios)
# ---------------------------------------------------------------------------

class TestGroupH_ResistanceGate:
    """
    With local_high_boost=b (boost applied to bar n-10 within the last-20 window):
      max(high[-20:]) ≈ price * (1+b)
      dist_local_high ≈ b / (1+b)

    For threshold 0.15:
      b=0.25 → dist ≈ 0.20  (fails)
      b=0.10 → dist ≈ 0.091 (passes)
    """

    def test_h1_very_far_from_local_high(self, tmp_path):
        """local_high_boost=0.30 → dist_local_high ≈ 0.23 > 0.15 → too_far_from_local_high."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            local_high_boost=0.30)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "too_far_from_local_high"
        assert metrics["dist_local_high"] > 0.15

    def test_h2_moderately_far_from_local_high(self, tmp_path):
        """local_high_boost=0.20 → dist_local_high ≈ 0.167 > 0.15 → too_far_from_local_high."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            local_high_boost=0.20)
        intra = _make_intraday()
        passed, metrics, reason = _check(tmp_path, daily, intra)
        assert not passed
        assert reason == "too_far_from_local_high"
        assert metrics["dist_local_high"] > 0.15

    def test_h3_near_local_high_passes_gate(self, tmp_path):
        """local_high_boost=0.10 → dist_local_high ≈ 0.091 < 0.15 → passes gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            local_high_boost=0.10)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["dist_local_high"] < 0.15

    def test_h4_at_local_high_passes_gate(self, tmp_path):
        """No high boost → dist_local_high ≈ hist_spread (tiny) → passes gate."""
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            local_high_boost=0.0)
        intra = _make_intraday()
        _, metrics, _ = _check(tmp_path, daily, intra)
        assert metrics["dist_local_high"] < 0.15


# ---------------------------------------------------------------------------
# Group I — threshold grid sweep (6 scenarios)
# ---------------------------------------------------------------------------

class TestGroupI_ThresholdGridSweep:
    """
    Tests the same data against different threshold configs to document
    which threshold values produce passes vs rejections.

    These are the calibration scenarios: they validate that the chosen
    post-recalibration defaults (atr 0.04, resistance 0.15, vol 1.5)
    accept more candidates than the old defaults (atr 0.02, resistance 0.05)
    while tighter alternatives correctly reject borderline data.
    """

    def _good_daily(self) -> pd.DataFrame:
        """Well-formed daily data that passes all default gates."""
        return _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01)

    def _good_intraday(self) -> pd.DataFrame:
        return _make_intraday(noise_pct=0.001)

    def test_i1_default_config_good_data_passes_all_gates(self, tmp_path):
        """
        Canonical good candidate with post-recalibration defaults.
        Must pass all 7 gates — this is the smoke test for the calibrated config.
        """
        passed, metrics, reason = _check(tmp_path, self._good_daily(), self._good_intraday())
        assert passed, f"Good candidate failed gate: {reason} | metrics: {metrics}"
        assert metrics["absorption_ratio"] > 1.5
        assert metrics["vol_zscore"] > 1.5

    def test_i2_old_atr_threshold_002_rejects_atr_of_003(self, tmp_path):
        """
        Pre-recalibration: atr_ratio threshold=0.02. Data with atr≈0.03 was rejected.
        Signal-analysis.md §3.4: this cut 68% of valid setups.
        With new threshold=0.04, same data passes.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.015, last_spread=0.015)
        intra = _make_intraday()

        # Old threshold: atr_ratio=0.03 → rejected
        passed_old, _, reason_old = _check(tmp_path, daily, intra, max_atr_ratio=0.02)
        assert not passed_old
        assert reason_old == "poor_price_compression"

        # New threshold: atr_ratio=0.03 < 0.04 → accepted at this gate
        _, metrics_new, _ = _check(tmp_path, daily, intra, max_atr_ratio=0.04)
        assert metrics_new["atr_ratio"] < 0.04

    def test_i3_relaxed_atr_006_accepts_higher_atr(self, tmp_path):
        """
        If we relax atr_ratio to 0.06, data with atr≈0.05 passes.
        Useful upper bound: above 0.06 the stock is too volatile to be 'coiled'.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.025, last_spread=0.015)
        intra = _make_intraday()

        # Default threshold 0.04: atr≈0.05 → rejected
        passed_default, _, reason_default = _check(tmp_path, daily, intra, max_atr_ratio=0.04)
        assert not passed_default
        assert reason_default == "poor_price_compression"

        # Relaxed threshold 0.06: accepted at ATR gate
        _, metrics_relaxed, _ = _check(tmp_path, daily, intra, max_atr_ratio=0.06)
        assert metrics_relaxed["atr_ratio"] < 0.06

    def test_i4_old_resistance_gate_005_rejects_local_high_of_010(self, tmp_path):
        """
        Pre-recalibration: max_distance_from_resistance=0.05 (vs 52w high).
        With new gate at 0.15 (20-day high), a stock 10% below local high is accepted.
        """
        # local_high_boost=0.10 → dist ≈ 0.091
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            local_high_boost=0.10)
        intra = _make_intraday()

        # Old tight gate: dist 0.091 > 0.05 → rejected
        passed_old, _, reason_old = _check(tmp_path, daily, intra,
                                           max_distance_from_resistance=0.05)
        assert not passed_old
        assert reason_old == "too_far_from_local_high"

        # New gate 0.15: dist 0.091 < 0.15 → accepted
        _, metrics_new, _ = _check(tmp_path, daily, intra,
                                   max_distance_from_resistance=0.15)
        assert metrics_new["dist_local_high"] < 0.15

    def test_i5_relaxed_resistance_020_accepts_stock_17_pct_below_high(self, tmp_path):
        """
        local_high_boost=0.20 → dist_local_high ≈ 0.167.
        Default gate 0.15 rejects; relaxed gate 0.20 accepts.
        Signal-analysis.md: if gate relaxed beyond 0.20 signal quality degrades.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01,
                            local_high_boost=0.20)
        intra = _make_intraday()

        # Default 0.15: rejected
        passed_default, _, reason_default = _check(tmp_path, daily, intra,
                                                   max_distance_from_resistance=0.15)
        assert not passed_default
        assert reason_default == "too_far_from_local_high"

        # Relaxed 0.20: accepted at resistance gate
        _, metrics_relaxed, _ = _check(tmp_path, daily, intra,
                                       max_distance_from_resistance=0.20)
        assert metrics_relaxed["dist_local_high"] < 0.20

    def test_i6_tight_vol_zscore_500_rejects_typical_signal(self, tmp_path):
        """
        Raising min_vol_zscore above the typical zscore (≈4.36 with deterministic volumes)
        to 5.0 correctly rejects data that the default threshold 1.5 accepts.
        Documents the upper bound: zscore threshold > 4.5 kills virtually all signals.
        """
        daily = _make_daily(vol_mult=4.0, hist_spread=0.01, last_spread=0.01)
        intra = _make_intraday()

        # Default 1.5: passes vol gate
        _, metrics_default, _ = _check(tmp_path, daily, intra, min_vol_zscore=1.5)
        assert metrics_default["vol_zscore"] > 1.5

        # Tight 5.0: zscore ≈ 4.36 < 5.0 → rejected
        passed_tight, _, reason_tight = _check(tmp_path, daily, intra, min_vol_zscore=5.0)
        assert not passed_tight
        assert reason_tight == "low_volume_zscore"
