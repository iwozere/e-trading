"""
Exposure Limits Module

Implements position and portfolio exposure limits.
"""
from typing import Dict

def check_position_limit(current_position: float, max_position: float) -> bool:
    """
    Check if a new position exceeds the maximum allowed position size.
    Args:
        current_position (float): Current position size
        max_position (float): Maximum allowed position size
    Returns:
        bool: True if within limit, False if exceeded
    """
    return abs(current_position) <= max_position

def check_portfolio_limit(current_exposures: Dict[str, float], max_portfolio_exposure: float) -> bool:
    """
    Check if total portfolio exposure exceeds the allowed limit.
    Args:
        current_exposures (Dict[str, float]): Mapping of asset to exposure
        max_portfolio_exposure (float): Maximum allowed total exposure
    Returns:
        bool: True if within limit, False if exceeded
    """
    total_exposure = sum(abs(v) for v in current_exposures.values())
    return total_exposure <= max_portfolio_exposure 
