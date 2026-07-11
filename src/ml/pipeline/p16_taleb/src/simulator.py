"""
Barbell strategy simulation engine for the P16 Taleb pipeline.

Simulates systematic monthly purchase of deep OTM put options using
Black-Scholes pricing. Returns one row per rebalance period with full
P&L accounting.
"""

import logging

import numpy as np
import pandas as pd

from .pricing import bs_put_price, skew_adjusted_sigma

_logger = logging.getLogger(__name__)

_MAX_PUT_PCT_OF_S = 0.15  # discard if put_price > 15% of spot (likely data error)


def simulate_barbell(
    df: pd.DataFrame,
    moneyness: float = 0.85,
    T_days: int = 90,
    rebalance_days: int = 21,
    put_budget_pct: float = 0.02,
    initial_capital: float = 100_000.0,
    r_col: str = "rate_3m",
    sigma_col: str = "vix",
    skew_slope: float = 0.015,
    use_market_prices: bool = False,
    options_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Simulate systematic monthly put purchases over the master DataFrame.

    On each rebalance date:
      1. Price a European put at K = moneyness × S using Black-Scholes.
      2. Record spend = put_budget_pct × initial_capital.
      3. At expiry (T_days calendar days later, rolled to next trading day):
         compute payoff = max(0, K − S_expiry) × num_contracts.

    Args:
        df:               Master daily DataFrame with DatetimeIndex.
                          Required columns: close, vix (or sigma_col), rate_3m (or r_col).
        moneyness:        K / S ratio (0.85 = 15% OTM).
        T_days:           Option tenor in calendar days.
        rebalance_days:   Trading-day interval between purchases.
        put_budget_pct:   Fraction of initial_capital spent per period.
        initial_capital:  Starting portfolio value in dollars.
        r_col:            Column name for the risk-free rate (decimal, e.g. 0.04).
        sigma_col:        Column name for the VIX proxy (raw index level, e.g. 20.0).
        skew_slope:       Linear skew slope parameter for skew_adjusted_sigma().
        use_market_prices: Placeholder for Tier-1 data path (not yet implemented).
        options_df:       Real options data (only used when use_market_prices=True).

    Returns:
        DataFrame with DatetimeIndex (purchase date) and columns:
        expiry_date, S, K, T, moneyness, otm_pct, sigma_used, vix, rate_3m,
        put_price, put_price_pct_of_S, budget_spent, num_contracts, price_source,
        S_at_expiry, drawdown_at_expiry, intrinsic_value_at_expiry,
        payoff, pnl, pnl_pct, cum_cost, cum_payoff, cum_pnl.
    """
    if df.empty:
        return pd.DataFrame()

    if use_market_prices:
        _logger.warning("use_market_prices=True not yet implemented; falling back to BS")

    # Pre-compute cumulative high for drawdown at expiry
    cum_peak = df["close"].cummax()
    trading_days = pd.DatetimeIndex(df.index)

    # Rebalance dates: every rebalance_days trading days, starting from the first row
    purchase_indices = list(range(0, len(trading_days), rebalance_days))

    otm_pct = 1.0 - moneyness
    T_years = T_days / 365.0
    budget_per_period = put_budget_pct * initial_capital

    rows = []
    skipped = 0

    # Extract float arrays once at the boundary (also avoids per-row Scalar typing)
    close_arr = df["close"].to_numpy(dtype=float)
    sigma_arr = df[sigma_col].to_numpy(dtype=float) if sigma_col in df.columns else None
    r_arr = df[r_col].to_numpy(dtype=float) if r_col in df.columns else None

    for idx in purchase_indices:
        purchase_date = trading_days[idx]
        S = float(close_arr[idx])
        vix_val = float(sigma_arr[idx]) if sigma_arr is not None and not np.isnan(sigma_arr[idx]) else 20.0
        r = float(r_arr[idx]) if r_arr is not None and not np.isnan(r_arr[idx]) else 0.04

        K = moneyness * S
        sigma = skew_adjusted_sigma(vix_val, otm_pct, skew_slope)

        put_price = bs_put_price(S, K, T_years, r, sigma)

        # Guard checks
        if put_price <= 0 or put_price > S * _MAX_PUT_PCT_OF_S:
            _logger.debug(
                "Skipping period %s: put_price=%.4f (S=%.2f, K=%.2f, sigma=%.3f)",
                purchase_date.date(),
                put_price,
                S,
                K,
                sigma,  # type: ignore[union-attr]
            )
            skipped += 1
            continue

        put_price_pct = put_price / S * 100.0
        num_contracts = budget_per_period / put_price
        budget_spent = num_contracts * put_price  # = budget_per_period (no rounding)

        # Find expiry trading day: first trading day >= purchase_date + T_days
        expiry_calendar = purchase_date + pd.Timedelta(days=T_days)
        expiry_pos = int(trading_days.searchsorted(expiry_calendar))
        if expiry_pos >= len(trading_days):
            expiry_pos = len(trading_days) - 1
        expiry_date = trading_days[expiry_pos]

        S_expiry = float(df["close"].loc[expiry_date])
        peak_at_expiry = float(cum_peak.loc[expiry_date])
        drawdown_at_expiry = (S_expiry - peak_at_expiry) / peak_at_expiry

        intrinsic_value = max(0.0, K - S_expiry)
        payoff = intrinsic_value * num_contracts
        pnl = payoff - budget_spent
        pnl_pct = pnl / initial_capital * 100.0

        rows.append(
            {
                "date": purchase_date,
                "expiry_date": expiry_date,
                "S": S,
                "K": K,
                "T": T_years,
                "moneyness": moneyness,
                "otm_pct": otm_pct * 100.0,
                "sigma_used": sigma,
                "vix": vix_val,
                "rate_3m": r,
                "put_price": put_price,
                "put_price_pct_of_S": put_price_pct,
                "budget_spent": budget_spent,
                "num_contracts": num_contracts,
                "price_source": "bs",
                "S_at_expiry": S_expiry,
                "drawdown_at_expiry": drawdown_at_expiry,
                "intrinsic_value_at_expiry": intrinsic_value,
                "payoff": payoff,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    if not rows:
        _logger.warning("No simulation rows produced (all %d periods skipped)", skipped)
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("date")
    result["cum_cost"] = result["budget_spent"].cumsum()
    result["cum_payoff"] = result["payoff"].cumsum()
    result["cum_pnl"] = result["pnl"].cumsum()

    _logger.info(
        "simulate_barbell: moneyness=%.2f T=%dd → %d periods, %d skipped, "
        "total_cost=$%.0f, total_payoff=$%.0f, net_pnl=$%.0f",
        moneyness,
        T_days,
        len(result),
        skipped,
        result["cum_cost"].iloc[-1],
        result["cum_payoff"].iloc[-1],
        result["cum_pnl"].iloc[-1],
    )
    return result
