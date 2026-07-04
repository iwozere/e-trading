"""
Tests for shared/risk_overlay/VixPositionSizer (Phase 7.1).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _make_vix_series(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    vix = 15.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    vix = np.clip(vix, 9.0, 80.0)
    return pd.Series(vix, index=pd.date_range("2022-01-01", periods=n, freq="B"))


class TestVixPositionSizer:
    def _make_sizer(self):
        from src.ml.pipeline.shared.risk_overlay.vix_position_sizer import (
            VixPositionSizer,
            VixPositionSizerConfig,
        )

        config = VixPositionSizerConfig(z_lookback=20)
        return VixPositionSizer(config)

    def test_exposures_are_between_0_and_1(self):
        sizer = self._make_sizer()
        vix = _make_vix_series()
        exposures = sizer.compute_exposures(vix)
        assert exposures.dropna().between(0.0, 1.0).all(), "Exposures outside [0, 1]"

    def test_exposures_length_matches_input(self):
        sizer = self._make_sizer()
        vix = _make_vix_series(150)
        exposures = sizer.compute_exposures(vix)
        assert len(exposures) == 150

    def test_first_bar_is_zero_due_to_lag(self):
        sizer = self._make_sizer()
        vix = _make_vix_series()
        exposures = sizer.compute_exposures(vix)
        # First bar has NaN lagged Z-score → exposure = 0.0
        assert exposures.iloc[0] == 0.0 or np.isnan(exposures.iloc[0])

    def test_compute_z_scores_length_matches(self):
        sizer = self._make_sizer()
        vix = _make_vix_series(100)
        z = sizer.compute_z_scores(vix)
        assert len(z) == 100

    def test_scale_position_size_returns_scaled_value(self):
        sizer = self._make_sizer()
        vix = _make_vix_series(200)
        exposures = sizer.compute_exposures(vix)
        # Pick a date where exposure > 0
        nonzero_dates = exposures[exposures > 0].index
        if len(nonzero_dates) == 0:
            pytest.skip("No non-zero exposure dates in this random sample")
        date = nonzero_dates[0]
        base = 1000.0
        scaled = sizer.scale_position_size(base, vix, date)
        expected = base * float(exposures.loc[date])
        assert abs(scaled - expected) < 1e-6

    def test_scale_position_size_unknown_date_returns_base(self):
        sizer = self._make_sizer()
        vix = _make_vix_series(50)
        base = 500.0
        result = sizer.scale_position_size(base, vix, "2099-01-01")
        assert result == base
