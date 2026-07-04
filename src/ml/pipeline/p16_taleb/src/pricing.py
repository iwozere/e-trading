"""
Option pricing engine for the P16 Taleb barbell pipeline.

Provides Black-Scholes European put pricing, Greeks, and a linear
volatility skew model for deep OTM puts.
"""

import logging

import numpy as np
from scipy.stats import norm

_logger = logging.getLogger(__name__)

_SIGMA_CAP = 1.5  # clamp extreme implied vols for numerical stability


def skew_adjusted_sigma(
    vix_level: float,
    otm_pct: float,
    skew_slope: float = 0.015,
) -> float:
    """
    Linear skew model: scale VIX-implied vol by the OTM put premium.

    Args:
        vix_level: VIX index level (e.g. 20.0 for 20% implied vol).
        otm_pct:   OTM fraction as a decimal (0.15 = 15% OTM).
        skew_slope: Slope of the linear skew per OTM percentage point.
                    Default 0.015 → +1.5% vol per 1% OTM.

    Returns:
        Skew-adjusted sigma as a decimal (e.g. 0.245 for 24.5% vol).
    """
    base_sigma = vix_level / 100.0
    adjusted = base_sigma * (1.0 + skew_slope * otm_pct * 100.0)
    if adjusted > _SIGMA_CAP:
        _logger.warning(
            "skew_adjusted_sigma clamped: %.3f -> %.1f (vix=%.1f, otm_pct=%.2f)",
            adjusted,
            _SIGMA_CAP,
            vix_level,
            otm_pct,
        )
        return _SIGMA_CAP
    return adjusted


def _d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> tuple[float, float]:
    """Compute d1 and d2 for Black-Scholes."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black-Scholes European put price.

    Args:
        S:     Spot price of the underlying.
        K:     Strike price.
        T:     Time to expiry in years (e.g. 90/365).
        r:     Continuously compounded risk-free rate (decimal, e.g. 0.04).
        sigma: Implied volatility (decimal, e.g. 0.20).

    Returns:
        Theoretical put price in the same units as S.
    """
    if T <= 0:
        return max(0.0, K - S)
    if sigma <= 0:
        return max(0.0, K * np.exp(-r * T) - S)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(max(0.0, price))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> dict:
    """
    Black-Scholes Greeks for a European put.

    Args:
        S:     Spot price.
        K:     Strike price.
        T:     Time to expiry in years.
        r:     Risk-free rate (decimal).
        sigma: Implied volatility (decimal).

    Returns:
        Dict with keys delta, gamma, theta, vega, rho.
        theta is per calendar day.
        vega is per 1 percentage-point change in vol (per 1%).
        rho is per 1 percentage-point change in rate (per 1%).
    """
    if T <= 0 or sigma <= 0:
        intrinsic_delta = -1.0 if K > S else 0.0
        return {"delta": intrinsic_delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-r * T)

    delta = norm.cdf(d1) - 1.0  # put delta ∈ [-1, 0]
    gamma = pdf_d1 / (S * sigma * sqrt_T)
    # theta per calendar year (then divided by 365 for per-day)
    theta_annual = -(S * pdf_d1 * sigma) / (2.0 * sqrt_T) + r * K * discount * norm.cdf(-d2)
    theta = theta_annual / 365.0
    vega = S * pdf_d1 * sqrt_T / 100.0  # per 1% vol move
    rho = -K * T * discount * norm.cdf(-d2) / 100.0  # per 1% rate move

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }
