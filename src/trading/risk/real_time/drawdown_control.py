"""
Drawdown Control Module

Implements circuit breaker logic for drawdown control.
"""
from typing import List

def check_drawdown(equity_curve: List[float], max_drawdown_pct: float) -> bool:
    """
    Check if drawdown exceeds the maximum allowed percentage.
    Args:
        equity_curve (List[float]): List of account equity values over time
        max_drawdown_pct (float): Maximum allowed drawdown as a fraction (e.g., 0.2 for 20%)
    Returns:
        bool: True if within limit, False if drawdown exceeded
    """
    peak = equity_curve[0]
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        if drawdown > max_drawdown_pct:
            return False
    return True 
