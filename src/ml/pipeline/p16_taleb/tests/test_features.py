"""
Unit tests for src/features.py.

Uses synthetic price data to validate all feature formulas.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_pipeline_dir = Path(__file__).resolve().parents[1]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from src.features import build_features


def _make_df(
    n: int = 300,
    trend: float = 0.0002,
    vix_level: float = 20.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic master DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 4000.0 * np.cumprod(1 + rng.normal(trend, 0.01, n))
    vix = np.clip(vix_level + rng.normal(0, 3, n), 5, 80)
    rate_3m = np.full(n, 0.04)
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1e8,
            "vix": vix,
            "rate_3m": rate_3m,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class TestBuildFeatures:
    def test_returns_all_required_columns(self):
        df = _make_df()
        out = build_features(df)
        required = [
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "ret_63d",
            "drawdown",
            "vix_ma20",
            "vol_ratio",
            "vix_regime",
            "stress_flag",
        ]
        for col in required:
            assert col in out.columns, f"Missing column: {col}"

    def test_drawdown_non_positive(self):
        """Drawdown is always ≤ 0 by construction."""
        out = build_features(_make_df())
        assert (out["drawdown"] <= 1e-10).all(), "Drawdown should be ≤ 0"

    def test_drawdown_zero_at_all_time_high(self):
        """On each all-time high, drawdown = 0."""
        df = _make_df(trend=0.005)  # strong uptrend → frequent all-time highs
        out = build_features(df)
        # At least some rows should have drawdown very close to 0
        assert (out["drawdown"].abs() < 1e-9).any()

    def test_drawdown_formula(self):
        """Verify drawdown = (close - cummax) / cummax."""
        df = _make_df()
        out = build_features(df)
        expected = (df["close"] - df["close"].cummax()) / df["close"].cummax()
        pd.testing.assert_series_equal(out["drawdown"], expected, check_names=False, rtol=1e-9)

    def test_vix_ma20_is_rolling_mean(self):
        """vix_ma20 must equal vix.rolling(20).mean()."""
        df = _make_df()
        out = build_features(df)
        expected = df["vix"].rolling(20).mean()
        pd.testing.assert_series_equal(out["vix_ma20"], expected, check_names=False)

    def test_vol_ratio_equals_vix_over_ma20(self):
        df = _make_df()
        out = build_features(df)
        expected = df["vix"] / df["vix"].rolling(20).mean()
        pd.testing.assert_series_equal(out["vol_ratio"], expected, check_names=False)

    def test_stress_flag_is_boolean(self):
        out = build_features(_make_df())
        assert out["stress_flag"].dtype == bool

    def test_stress_flag_triggers_on_high_vix(self):
        """stress_flag is True when VIX > 30."""
        df = _make_df(vix_level=35.0)
        out = build_features(df)
        # With VIX ~35, most rows should be stressed
        assert out["stress_flag"].sum() > 0

    def test_vix_regime_categories(self):
        """vix_regime must be one of the five expected labels."""
        out = build_features(_make_df())
        valid_labels = {"low", "normal", "elevated", "high", "extreme"}
        actual = set(out["vix_regime"].dropna().astype(str).unique())
        assert actual.issubset(valid_labels), f"Unexpected regime labels: {actual - valid_labels}"

    def test_gdelt_available_false_when_no_avgtone(self):
        """Without avgtone column, gdelt_available must be False everywhere."""
        df = _make_df()
        assert "avgtone" not in df.columns
        out = build_features(df)
        assert not out["gdelt_available"].any()

    def test_gdelt_features_when_avgtone_present(self):
        """gdelt_tone_ma5 and gdelt_available computed when avgtone is in df."""
        df = _make_df()
        df["avgtone"] = -1.5
        out = build_features(df)
        assert "gdelt_tone_ma5" in out.columns
        assert out["gdelt_available"].sum() == len(df)

    def test_no_mutation_of_input(self):
        """build_features must not mutate the input DataFrame."""
        df = _make_df()
        cols_before = set(df.columns)
        _ = build_features(df)
        assert set(df.columns) == cols_before
