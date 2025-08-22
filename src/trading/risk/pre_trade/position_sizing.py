"""
Position Sizing Module

Implements position sizing algorithms:
- Kelly Criterion
- Fixed Fractional
"""


def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
    """
    Calculate the optimal fraction of capital to risk per trade using the Kelly Criterion.
    Args:
        win_prob (float): Probability of winning (0 < win_prob < 1)
        win_loss_ratio (float): Ratio of average win to average loss (>0)
    Returns:
        float: Optimal fraction of capital to risk (can be negative if edge is negative)
    """
    kelly = win_prob - (1 - win_prob) / win_loss_ratio
    return max(0.0, kelly)

def fixed_fractional(account_equity: float, risk_per_trade: float, stop_loss_pct: float) -> float:
    """
    Calculate position size using Fixed Fractional method.
    Args:
        account_equity (float): Total account equity
        risk_per_trade (float): Fraction of equity to risk per trade (e.g., 0.01 for 1%)
        stop_loss_pct (float): Stop loss as a fraction of entry price (e.g., 0.02 for 2%)
    Returns:
        float: Position size (number of units/contracts)
    """
    if stop_loss_pct <= 0:
        return 0.0
    risk_amount = account_equity * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return position_size 
