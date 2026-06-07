"""
Position Sizing Module

Implements position sizing algorithms:
- Kelly Criterion
- Fixed Fractional
"""


_KELLY_FRACTION = 0.25  # Fractional Kelly multiplier (quarter-Kelly is common in practice)
_KELLY_MAX = 0.20       # Hard cap: never risk more than 20% of capital on one trade

def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
    """
    Calculate the optimal fraction of capital to risk per trade using the Kelly Criterion.

    Full Kelly is notoriously over-aggressive and sensitive to edge-estimate error. This
    implementation applies a fractional-Kelly multiplier (_KELLY_FRACTION) and a hard cap
    (_KELLY_MAX) so the result is safe to wire directly into live position sizing.

    Args:
        win_prob (float): Probability of winning (0 < win_prob < 1)
        win_loss_ratio (float): Ratio of average win to average loss (>0)
    Returns:
        float: Fractional, capped Kelly fraction (always in [0, _KELLY_MAX])
    """
    full_kelly = win_prob - (1 - win_prob) / win_loss_ratio
    capped = min(max(0.0, full_kelly) * _KELLY_FRACTION, _KELLY_MAX)
    return capped

def fixed_fractional(
    account_equity: float,
    risk_per_trade: float,
    stop_loss_pct: float,
    entry_price: float,
) -> float:
    """
    Calculate position size using Fixed Fractional method.

    Args:
        account_equity (float): Total account equity
        risk_per_trade (float): Fraction of equity to risk per trade (e.g., 0.01 for 1%)
        stop_loss_pct (float): Stop loss as a fraction of entry price (e.g., 0.02 for 2%)
        entry_price (float): Current entry price of the asset

    Returns:
        float: Position size in asset units (not notional dollars)
    """
    if stop_loss_pct <= 0 or entry_price <= 0:
        return 0.0
    risk_amount = account_equity * risk_per_trade
    position_size = risk_amount / (stop_loss_pct * entry_price)
    return position_size
