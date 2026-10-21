"""Shared fixtures-as-functions for signal ledger tests."""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd


def sessions(n: int, start: str = "2026-01-05") -> pd.DatetimeIndex:
    """Return n consecutive business days."""
    return pd.bdate_range(start, periods=n)


def panel(index: pd.DatetimeIndex, series: Dict[str, Iterable[float]]) -> pd.DataFrame:
    """Build a session-indexed price frame from per-ticker lists."""
    return pd.DataFrame({ticker: list(values) for ticker, values in series.items()}, index=index)


def flat(n: int, value: float = 100.0) -> List[float]:
    """n identical prices."""
    return [value] * n


def ramp(n: int, start: float, step: float) -> List[float]:
    """n prices rising by step."""
    return [start + step * i for i in range(n)]
