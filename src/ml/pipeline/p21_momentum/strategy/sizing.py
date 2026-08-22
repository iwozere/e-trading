"""
P21 Momentum — Position sizing (docs/pipeline-specification.md §7).

Inverse-volatility weights with iterative capping. **The cap is computed
off full NAV, not off sleeve size** — 1% of $250k = $2,500 — this is a
deliberate, non-obvious choice (spec §7 note: "This reflects the operator's
original constraint"), not a bug.
"""

from __future__ import annotations

from typing import Dict

from src.ml.pipeline.p21_momentum.config import (
    MAX_POSITION_PCT,
    NAV_TOTAL_USD,
    SHARES_DECIMALS,
    SIZING_MAX_ITERATIONS,
    SLEEVE_TARGET_PCT,
)


def size_positions(
    vol_by_ticker: Dict[str, float],
    nav_total: float = NAV_TOTAL_USD,
    sleeve_pct: float = SLEEVE_TARGET_PCT,
    max_pos_pct: float = MAX_POSITION_PCT,
    regime_scalar: float = 1.0,
    max_iterations: int = SIZING_MAX_ITERATIONS,
) -> Dict[str, float]:
    """
    Allocate the sleeve across selected names by inverse volatility, capped per-name.

    Args:
        vol_by_ticker: {ticker: vol} for every selected position (vol from
            strategy/signal.py's SignalResult.vol).
        nav_total: Total portfolio NAV (spec: $250,000).
        sleeve_pct: Fraction of NAV allocated to the momentum sleeve (spec: 0.20).
        max_pos_pct: Per-position cap **as a fraction of nav_total**, not of
            sleeve_usd (spec §7's explicit, deliberate choice).
        regime_scalar: Regime overlay scalar (§8) — when < 1, the entire
            sleeve scales down proportionally; released funds go to cash.
        max_iterations: Cap-and-redistribute loop bound (spec: 10 — the loop
            is deterministic, no randomness, so this only bounds worst-case
            iterations, not behavior).

    Returns:
        {ticker: dollar_allocation}, summing to sleeve_usd (+/- rounding).
        Empty dict if vol_by_ticker is empty.
    """
    if not vol_by_ticker:
        return {}

    sleeve_usd = nav_total * sleeve_pct * regime_scalar
    cap_usd = nav_total * max_pos_pct  # cap is off TOTAL NAV, not sleeve size

    inv_vol = {t: 1.0 / v for t, v in vol_by_ticker.items()}
    capped: Dict[str, float] = {}
    free = dict(inv_vol)
    remaining = sleeve_usd

    for _ in range(max_iterations):
        total_w = sum(free.values())
        if total_w == 0:
            break
        alloc = {t: remaining * w / total_w for t, w in free.items()}
        over = {t: v for t, v in alloc.items() if v > cap_usd}
        if not over:
            capped.update(alloc)
            break
        for t in over:
            capped[t] = cap_usd
            free.pop(t)
            remaining -= cap_usd

    return capped


def shares_from_allocation(allocation_usd: Dict[str, float], adj_open_price: Dict[str, float]) -> Dict[str, float]:
    """
    Convert dollar allocations to fractional share counts at the fill price.

    Args:
        allocation_usd: {ticker: dollar_allocation} from size_positions().
        adj_open_price: {ticker: adjusted open price at the fill date}.

    Returns:
        {ticker: shares}, rounded to SHARES_DECIMALS (4, IBKR fractional
        granularity per spec §7). Tickers missing a price are omitted.
    """
    shares: Dict[str, float] = {}
    for ticker, usd in allocation_usd.items():
        price = adj_open_price.get(ticker)
        if price is None or price <= 0:
            continue
        shares[ticker] = round(usd / price, SHARES_DECIMALS)
    return shares
