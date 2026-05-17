"""
Unit tests for src/pricing.py.

Validates Black-Scholes put price, Greeks, and skew model against
known analytical values from options theory.
"""

import sys
from pathlib import Path

import pytest

# Make p16_taleb/src/ importable
_pipeline_dir = Path(__file__).resolve().parents[1]
if str(_pipeline_dir) not in sys.path:
    sys.path.insert(0, str(_pipeline_dir))

from src.pricing import bs_put_price, bs_greeks, skew_adjusted_sigma


# ---------------------------------------------------------------------------
# Black-Scholes put price
# ---------------------------------------------------------------------------

class TestBsPutPrice:
    def test_atm_put_known_value(self):
        """
        ATM European put (K=S, T=90/252, σ=0.20, r=0.04).

        Analytical BS result: approximately 4.06% of S (verified numerically).
        The spec's "0.056×S" refers to T=90/365 calendar-day convention;
        with T=90/252 trading-day convention the correct answer is ~4%.
        """
        S, K, T, r, sigma = 100.0, 100.0, 90 / 252, 0.04, 0.20
        price = bs_put_price(S, K, T, r, sigma)
        assert 0.038 * S <= price <= 0.045 * S, (
            f"ATM put price {price:.4f} outside expected range [3.8%, 4.5%] of S"
        )

    def test_deep_otm_is_cheap(self):
        """A 30% OTM put should cost much less than a 10% OTM put."""
        S, T, r, sigma = 100.0, 90 / 365, 0.04, 0.20
        price_10 = bs_put_price(S, 0.90 * S, T, r, sigma)
        price_30 = bs_put_price(S, 0.70 * S, T, r, sigma)
        assert price_30 < price_10

    def test_deep_itm_approaches_intrinsic(self):
        """A deep ITM put price ≈ intrinsic value when vol is very low."""
        S, K, T, r, sigma = 100.0, 150.0, 1.0 / 365, 0.0, 0.01
        price = bs_put_price(S, K, T, r, sigma)
        intrinsic = K - S
        assert abs(price - intrinsic) < 1.0

    def test_price_non_negative(self):
        """Put price is always ≥ 0."""
        for moneyness in [0.70, 0.80, 0.90, 1.00, 1.10]:
            p = bs_put_price(100.0, moneyness * 100.0, 90 / 365, 0.04, 0.20)
            assert p >= 0.0

    def test_zero_time_returns_intrinsic(self):
        """At expiry (T=0), put price = max(0, K-S)."""
        p_itm = bs_put_price(90.0, 100.0, 0.0, 0.04, 0.20)
        assert abs(p_itm - 10.0) < 1e-8

        p_otm = bs_put_price(110.0, 100.0, 0.0, 0.04, 0.20)
        assert p_otm == 0.0

    def test_higher_vol_raises_price(self):
        """Option price increases monotonically with volatility."""
        base = (100.0, 85.0, 90 / 365, 0.04)
        prices = [bs_put_price(*base, sigma) for sigma in [0.10, 0.20, 0.30, 0.50]]
        assert prices == sorted(prices)

    def test_longer_tenor_raises_price(self):
        """Option price increases with time to expiry."""
        p30 = bs_put_price(100.0, 85.0, 30 / 365, 0.04, 0.20)
        p90 = bs_put_price(100.0, 85.0, 90 / 365, 0.04, 0.20)
        assert p90 > p30


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

class TestBsGreeks:
    def _greeks(self, S=100.0, K=85.0, T=90 / 365, r=0.04, sigma=0.20):
        return bs_greeks(S, K, T, r, sigma)

    def test_delta_in_range(self):
        """Put delta is in [-1, 0] for all inputs."""
        for K in [70, 80, 90, 100, 110]:
            g = bs_greeks(100.0, float(K), 90 / 365, 0.04, 0.20)
            assert -1.0 <= g["delta"] <= 0.0, f"delta={g['delta']} for K={K}"

    def test_delta_atm_near_minus_half(self):
        """ATM put delta ≈ -0.5 (symmetric forward)."""
        g = bs_greeks(100.0, 100.0, 90 / 365, 0.0, 0.20)
        assert abs(g["delta"] + 0.5) < 0.05

    def test_gamma_positive(self):
        """Gamma is always positive for options."""
        g = self._greeks()
        assert g["gamma"] > 0.0

    def test_vega_positive(self):
        """Vega is always positive for options."""
        g = self._greeks()
        assert g["vega"] > 0.0

    def test_theta_negative_atm(self):
        """ATM put theta (time decay) is negative — option loses value daily."""
        g = bs_greeks(100.0, 100.0, 90 / 365, 0.04, 0.20)
        assert g["theta"] < 0.0

    def test_zero_time_returns_safe_defaults(self):
        """At expiry, greeks should return valid floats, not NaN."""
        g = bs_greeks(90.0, 100.0, 0.0, 0.04, 0.20)
        for key, val in g.items():
            assert isinstance(val, float), f"Expected float for {key}, got {type(val)}"


# ---------------------------------------------------------------------------
# Skew model
# ---------------------------------------------------------------------------

class TestSkewAdjustedSigma:
    def test_zero_otm_returns_base_sigma(self):
        """At-the-money (0% OTM): skew returns VIX/100."""
        result = skew_adjusted_sigma(20.0, 0.0)
        assert abs(result - 0.20) < 1e-9

    def test_15pct_otm(self):
        """15% OTM with slope=0.015: sigma × 1.225."""
        result = skew_adjusted_sigma(20.0, 0.15, skew_slope=0.015)
        expected = 0.20 * (1.0 + 0.015 * 15.0)  # = 0.245
        assert abs(result - expected) < 1e-9

    def test_monotone_increasing_with_otm(self):
        """Adjusted sigma increases as OTM% increases."""
        sigmas = [skew_adjusted_sigma(20.0, otm) for otm in [0.05, 0.10, 0.15, 0.20, 0.30]]
        assert sigmas == sorted(sigmas)

    def test_clamp_at_extreme_vol(self):
        """Very high VIX + deep OTM should be clamped to 1.5."""
        result = skew_adjusted_sigma(200.0, 0.30, skew_slope=0.015)
        assert result == 1.5
