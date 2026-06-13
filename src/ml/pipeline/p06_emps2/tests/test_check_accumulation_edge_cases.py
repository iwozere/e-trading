"""
Unit and regression tests for AccumulationAnalyzer._check_accumulation.

Verifies:
1. NaN metrics (zero-price intraday bar) are rejected — not silently passed
2. Negative vol_zscore is rejected as 'low_volume_zscore'
3. XRXDW regression: low-price warrant with NaN metrics no longer passes silently
4. A well-formed candidate passes all accumulation gates via apply_filters()
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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

def _make_config() -> EMPS2FilterConfig:
    cfg = EMPS2FilterConfig()
    cfg.interval = "1h"
    cfg.atr_period = 14
    cfg.min_vol_zscore = 1.5
    cfg.min_vol_rv_ratio = 1.5
    cfg.max_price_impact = 0.05
    cfg.max_atr_ratio = 0.04
    cfg.max_distance_from_resistance = 0.15
    cfg.max_distance_from_sma20 = 0.10
    return cfg


def _make_analyzer(tmp_path: Path) -> AccumulationAnalyzer:
    return AccumulationAnalyzer(
        data_manager=MagicMock(),
        config=_make_config(),
        results_dir=tmp_path,
    )


def _make_daily_df(n: int = 60, price: float = 50.0, last_volume_multiplier: float = 4.0) -> pd.DataFrame:
    """
    Synthetic daily OHLCV with a tight 1% H-L range.

    last_volume_multiplier: multiplied against the mean of bars 0..n-2 to set
    the last bar's volume.  > 1 → volume spike (positive zscore), < 1 → dip
    (negative zscore).
    """
    dates = pd.date_range("2025-01-01", periods=n, freq="1D")
    close = np.full(n, price)
    rng = np.random.default_rng(0)
    volume = rng.integers(900_000, 1_100_000, size=n).astype(float)
    volume[-1] = volume[:-1].mean() * last_volume_multiplier

    return pd.DataFrame({
        "timestamp": dates,
        "open":   close,
        "high":   close * 1.005,   # price_range_1d ≈ 0.01 < 0.05 ✓
        "low":    close * 0.995,
        "close":  close,
        "volume": volume,
    })


def _make_intraday_df(n: int = 40, price: float = 50.0, zero_at: int = -1) -> pd.DataFrame:
    """
    Synthetic 1-hour intraday OHLCV with tiny Gaussian noise on close.

    zero_at: when >= 0, sets close[zero_at] = 0 so that the log-return
    sequence contains -inf / inf entries → rv becomes NaN.
    """
    dates = pd.date_range("2025-01-01 09:30", periods=n, freq="1h")
    rng = np.random.default_rng(1)
    close = price + rng.normal(0, price * 0.001, size=n)   # ≈ 0.1% noise
    close = np.clip(close, 0.001, None)

    if zero_at >= 0:
        close[zero_at] = 0.0   # forces log(0/prev) = -inf → std = NaN

    return pd.DataFrame({
        "timestamp": dates,
        "open":   close,
        "high":   close * 1.001,
        "low":    close * 0.999,
        "close":  close,
        "volume": np.ones(n) * 50_000.0,
    })


# ---------------------------------------------------------------------------
# Test 1: NaN metrics must be rejected, not passed
# ---------------------------------------------------------------------------

def test_nan_metrics_are_rejected(tmp_path):
    """
    A zero close value in the intraday series causes RV to be NaN.
    The explicit NaN guard (accumulation_analyzer.py:321) must catch this
    and return (False, metrics, 'nan_metrics') instead of passing the ticker.
    """
    analyzer = _make_analyzer(tmp_path)
    df_daily    = _make_daily_df(n=60, last_volume_multiplier=4.0)
    # zero_at=10 falls inside the rv_bars_target window (last 32 of 40 bars)
    df_intraday = _make_intraday_df(n=40, zero_at=10)

    passed, metrics, reason = analyzer._check_accumulation("TESTNAN", df_intraday, df_daily)

    assert not passed, "Ticker with NaN RV must not pass accumulation filters"
    assert reason == "nan_metrics", (
        f"Expected reason 'nan_metrics', got '{reason}'. "
        "The NaN guard at the top of _check_accumulation may be missing or incorrect."
    )


# ---------------------------------------------------------------------------
# Test 2: Negative vol_zscore must be rejected
# ---------------------------------------------------------------------------

def test_negative_vol_zscore_rejects(tmp_path):
    """
    When the last bar's volume is far below the 20-period mean, vol_zscore
    is negative.  The ticker must be rejected as 'low_volume_zscore' and the
    absorption ratio must be 0.0 (since vol_zscore <= 0 → AR guard kicks in).
    """
    analyzer = _make_analyzer(tmp_path)
    # 0.05× the mean → deeply negative zscore
    df_daily    = _make_daily_df(n=60, last_volume_multiplier=0.05)
    df_intraday = _make_intraday_df(n=40)

    passed, metrics, reason = analyzer._check_accumulation("LOWVOL", df_intraday, df_daily)

    assert not passed, "Negative vol_zscore must not pass filters"
    assert reason == "low_volume_zscore", (
        f"Expected 'low_volume_zscore', got '{reason}'"
    )
    assert metrics["vol_zscore"] < 0, (
        f"Expected negative vol_zscore in diagnostics, got {metrics['vol_zscore']}"
    )
    assert metrics["absorption_ratio"] == 0.0, (
        "AR must be 0.0 when vol_zscore <= 0 (guard in accumulation_analyzer.py:261)"
    )


# ---------------------------------------------------------------------------
# Test 3: XRXDW regression — low-price warrant with NaN metrics is rejected
# ---------------------------------------------------------------------------

def test_low_price_warrant_nan_metrics_regression(tmp_path):
    """
    Regression for the XRXDW bug (see signal-analysis.md).

    Pre-fix: a ~$0.11 warrant produced NaN rv; because NaN <= threshold
    evaluates to False in Python/NumPy, it silently passed every filter gate.

    Post-fix: the explicit NaN guard (accumulation_analyzer.py:321) rejects
    it with reason 'nan_metrics'.  This test must ALWAYS pass.
    """
    analyzer = _make_analyzer(tmp_path)
    df_daily    = _make_daily_df(n=40, price=0.11, last_volume_multiplier=3.0)
    # rv window = close[-32:] (last 32 of 40 bars, indices 8-39).
    # zero_at must be >= 8 to fall inside the window and force NaN rv.
    df_intraday = _make_intraday_df(n=40, price=0.11, zero_at=15)

    passed, metrics, reason = analyzer._check_accumulation("XRXDW", df_intraday, df_daily)

    assert not passed, (
        "XRXDW-like warrant with NaN RV must be rejected — "
        "regression for the silent-pass bug"
    )
    assert reason == "nan_metrics", (
        f"Expected 'nan_metrics' (XRXDW regression), got '{reason}'"
    )


# ---------------------------------------------------------------------------
# Test 4: Well-formed candidate passes all gates via apply_filters()
# ---------------------------------------------------------------------------

@patch("src.ml.pipeline.p06_emps2.accumulation_analyzer.download_trf")
def test_apply_filters_passes_good_candidate(mock_download_trf, tmp_path):
    """
    Integration smoke-test: when DataManager returns synthetic OHLCV with
    - a strong volume spike (vol_zscore >> 1.5)
    - stable intraday close prices (small RV → AR >> 1.5)
    - tight daily range (price_range_1d < 0.05)
    - ATR-to-price ratio < 0.04

    apply_filters() must return a non-empty DataFrame containing the ticker.

    This validates that the threshold recalibration (max_atr_ratio 0.02→0.04,
    max_price_impact 0.03→0.05, 20-day high gate at 0.15) actually produces
    signals on normal market data.
    """
    # Suppress TRF network call cleanly
    mock_trf_path = MagicMock()
    mock_trf_path.exists.return_value = False
    mock_download_trf.return_value = mock_trf_path

    df_daily    = _make_daily_df(n=60, price=50.0, last_volume_multiplier=4.0)
    df_intraday = _make_intraday_df(n=40, price=50.0)

    mock_dm = MagicMock()

    def _get_ohlcv_batch(tickers, interval, start, end):
        df = df_intraday if interval == "1h" else df_daily
        return {t: df.copy() for t in tickers}

    mock_dm.get_ohlcv_batch.side_effect = _get_ohlcv_batch

    cfg = _make_config()
    analyzer = AccumulationAnalyzer(
        data_manager=mock_dm,
        config=cfg,
        results_dir=tmp_path,
        target_date="2025-03-01",
        chunk_size=5,
        checkpoint_enabled=False,
    )

    result = analyzer.apply_filters(["GOOD"])

    assert not result.empty, (
        "Expected at least 1 PASSED candidate with the recalibrated thresholds. "
        "If this fails, a filter threshold may have regressed to its pre-fix value."
    )
    assert "GOOD" in result["ticker"].values, (
        "Ticker 'GOOD' should have passed all accumulation filters"
    )
