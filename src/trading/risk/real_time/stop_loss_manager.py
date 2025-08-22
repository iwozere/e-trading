"""
Stop Loss Manager Module

Implements dynamic stop-loss adjustment.
"""

def dynamic_stop_loss(entry_price: float, current_price: float, initial_stop: float, trailing_pct: float = 0.02) -> float:
    """
    Adjust stop-loss dynamically using a trailing percentage.
    Args:
        entry_price (float): Entry price of the position
        current_price (float): Current market price
        initial_stop (float): Initial stop-loss price
        trailing_pct (float): Trailing stop as a fraction (default 2%)
    Returns:
        float: New stop-loss price
    """
    if current_price > entry_price:
        # For long positions, trail stop up
        new_stop = max(initial_stop, current_price * (1 - trailing_pct))
    else:
        # For short or losing positions, keep initial stop
        new_stop = initial_stop
    return new_stop 
