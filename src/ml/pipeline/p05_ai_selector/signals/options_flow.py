"""
Phase 5 stub — requires Tradier API access. Currently returns empty.

This module will be implemented in Phase 5 when Tradier API access is available.
It will provide unusual put/call volume signals as an additional input to Stage 2.
"""

from datetime import date
from typing import Dict, List


def get_unusual_activity(tickers: List[str], as_of_date: date) -> Dict[str, bool]:
    """
    Return unusual options activity flags per ticker.

    Phase 5 stub — always returns empty dict until Tradier API is available.

    Args:
        tickers: List of ticker symbols.
        as_of_date: Reference date.

    Returns:
        Empty dict (stub).
    """
    return {}
