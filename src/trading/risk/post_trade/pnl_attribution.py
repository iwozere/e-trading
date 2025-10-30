"""
PnL Attribution Module

Implements P&L attribution for post-trade analysis.
"""
from typing import List, Dict

def pnl_attribution(trades: List[Dict], group_by: str = "symbol") -> Dict[str, float]:
    """
    Attribute P&L to different categories (e.g., by symbol, strategy).
    Args:
        trades (List[Dict]): List of trade records (must include 'pnl' and group_by key)
        group_by (str): Key to group P&L by (default 'symbol')
    Returns:
        Dict[str, float]: Mapping from group to total P&L
    """
    result = {}
    for trade in trades:
        key = trade.get(group_by, "unknown")
        result[key] = result.get(key, 0.0) + trade.get("pnl", 0.0)
    return result 
