"""
Volatility Scaling Module

Implements volatility-adjusted position sizing.
"""
import numpy as np
from typing import List

def volatility_scaled_position(account_equity: float, target_vol: float, returns: List[float], min_size: float = 0.0, max_size: float = None) -> float:
    """
    Calculate position size based on target volatility.
    Args:
        account_equity (float): Total account equity
        target_vol (float): Target annualized volatility (e.g., 0.15 for 15%)
        returns (List[float]): List of recent returns (daily or per period)
        min_size (float): Minimum position size
        max_size (float): Maximum position size (optional)
    Returns:
        float: Volatility-adjusted position size
    """
    if len(returns) == 0:
        return min_size
    realized_vol = np.std(returns) * np.sqrt(252)  # Annualized
    if realized_vol == 0:
        return min_size
    position_size = account_equity * (target_vol / realized_vol)
    if max_size is not None:
        position_size = min(position_size, max_size)
    position_size = max(position_size, min_size)
    return position_size 
