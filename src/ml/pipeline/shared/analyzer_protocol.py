"""
ScreenerAnalyzer protocol — the interface that every Stage 3 analyzer must satisfy.

Both VolatilityFilter (P06 default) and AccumulationAnalyzer (accumulation mode)
already expose apply_filters(), so no internal changes to those classes are needed.
"""

from typing import List, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ScreenerAnalyzer(Protocol):
    """
    Protocol for pluggable Stage 3 screener analyzers.

    Any class that implements apply_filters(tickers) -> pd.DataFrame
    satisfies this protocol without explicit inheritance.
    """

    def apply_filters(self, tickers: List[str]) -> pd.DataFrame:
        """
        Apply the analyzer's filter logic to a list of tickers.

        Args:
            tickers: List of ticker symbols to evaluate.

        Returns:
            DataFrame containing only tickers that passed the filter,
            with diagnostic metrics as columns.
        """
        ...
