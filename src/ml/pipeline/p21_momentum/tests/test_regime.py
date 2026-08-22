"""Unit tests for src.ml.pipeline.p21_momentum.strategy.regime."""

from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from src.ml.pipeline.p21_momentum.strategy.regime import PriorRegimeState, compute_regime


def _flat_series(value: float, n: int) -> pd.Series:
    return pd.Series([value] * n)


class TestComputeRegimeBasics(unittest.TestCase):
    def test_normal_market_scalar_1(self):
        # SPX steadily up, no bear signal
        spx = pd.Series(np.linspace(3000, 4500, 300))
        vix = _flat_series(15.0, 30)
        prior = PriorRegimeState(scalar_applied=1.0, months_at_state=5)
        result = compute_regime(spx, vix, prior, as_of=date(2026, 8, 31))
        self.assertFalse(result.bear)
        self.assertEqual(result.scalar_applied, 1.0)

    def test_20d_vix_smoothing_not_spot(self):
        # Single-day VIX spike at the very end must not dominate the 20-day average
        spx = pd.Series(np.linspace(3000, 2500, 300))  # bear market (declining)
        vix_values = [15.0] * 29 + [80.0]  # one spot spike
        vix = pd.Series(vix_values)
        prior = PriorRegimeState(scalar_applied=0.60, months_at_state=1)
        result = compute_regime(spx, vix, prior, as_of=date(2026, 8, 31))
        # 20-day avg = (19*15 + 80)/20 = 18.25, well under threshold 28
        self.assertLess(result.vix_20d_avg, 28.0)
        self.assertFalse(result.high_vol)


class TestHysteresis(unittest.TestCase):
    def _bear_high_vol_inputs(self):
        spx = pd.Series(np.linspace(3000, 2000, 300))  # clear bear
        vix = _flat_series(40.0, 30)  # high vol
        return spx, vix

    def _normal_inputs(self):
        spx = pd.Series(np.linspace(2000, 3000, 300))  # clear bull
        vix = _flat_series(15.0, 30)
        return spx, vix

    def test_downward_change_applies_immediately(self):
        spx, vix = self._bear_high_vol_inputs()
        prior = PriorRegimeState(scalar_applied=1.0, months_at_state=3)  # was normal
        result = compute_regime(spx, vix, prior, as_of=date(2026, 8, 31))
        self.assertEqual(result.scalar_applied, result.scalar_raw)
        self.assertEqual(result.scalar_applied, 0.25)  # bear + high_vol
        self.assertEqual(result.months_at_state, 1)

    def test_upward_change_blocked_on_first_month(self):
        spx, vix = self._normal_inputs()
        prior = PriorRegimeState(scalar_applied=0.25, months_at_state=2, recent_raw_states=[(True, True)])
        result = compute_regime(spx, vix, prior, as_of=date(2026, 8, 31))
        # raw wants to go to 1.0, but only 1 month in new state so far (this one) -> blocked
        self.assertEqual(result.scalar_raw, 1.0)
        self.assertEqual(result.scalar_applied, 0.25)  # stays at prior applied
        self.assertEqual(result.months_at_state, 3)

    def test_upward_change_confirmed_on_second_month(self):
        spx, vix = self._normal_inputs()
        # Prior month's raw state already matched today's raw state (False, False)
        # for one month -> this is the 2nd consecutive month -> confirmed
        prior = PriorRegimeState(scalar_applied=0.25, months_at_state=1, recent_raw_states=[(False, False)])
        result = compute_regime(spx, vix, prior, as_of=date(2026, 8, 31))
        self.assertEqual(result.scalar_applied, 1.0)
        self.assertEqual(result.months_at_state, 1)

    def test_same_scalar_increments_months_at_state(self):
        spx, vix = self._normal_inputs()
        prior = PriorRegimeState(scalar_applied=1.0, months_at_state=7)
        result = compute_regime(spx, vix, prior, as_of=date(2026, 8, 31))
        self.assertEqual(result.scalar_applied, 1.0)
        self.assertEqual(result.months_at_state, 8)


if __name__ == "__main__":
    unittest.main()
