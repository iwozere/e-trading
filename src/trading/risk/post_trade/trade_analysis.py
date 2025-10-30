"""
Trade Analysis Module

Implements trade quality metrics for post-trade analysis.
"""
from typing import List, Dict

def trade_quality_metrics(trades: List[Dict]) -> Dict[str, float]:
    """
    Calculate trade quality metrics (win rate, avg win/loss, expectancy).
    Args:
        trades (List[Dict]): List of trade records (must include 'pnl')
    Returns:
        Dict[str, float]: Metrics including win_rate, avg_win, avg_loss, expectancy
    """
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] < 0]
    total = len(trades)
    win_rate = len(wins) / total if total > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy
    } 
