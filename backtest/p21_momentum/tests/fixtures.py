"""
P21 Momentum backtest tests — shared synthetic-panel fixtures.

Extracted from ``test_runner.py`` so ``test_metrics.py``, ``test_stress_windows.py``,
``test_robustness.py``, ``test_cost_sensitivity.py``, and ``test_phase0_report.py`` don't each
redefine the same synthetic OHLCV generator (CLAUDE.md §8: helper functions to avoid duplication).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

SECTORS = ["Tech", "Health", "Financials", "Energy", "Industrials"]


def make_ohlcv_df(start: str, end: str, seed: int, drift: float = 0.0004, vol: float = 0.02) -> pd.DataFrame:
    """One ticker's synthetic daily OHLCV, geometric-random-walk close with a fixed seed."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=vol, size=len(idx))
    close = 100.0 * np.cumprod(1 + rets)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [2_000_000] * len(idx),
        }
    )


def make_universe_panel(n_tickers: int, start: str, end: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """A synthetic {ticker: OHLCV} panel of n_tickers names plus MTUM/SPY/^GSPC/^VIX, with sectors."""
    panel = {}
    sector_by_ticker = {}
    for i in range(n_tickers):
        ticker = f"T{i:03d}"
        panel[ticker] = make_ohlcv_df(start, end, seed=i)
        sector_by_ticker[ticker] = SECTORS[i % len(SECTORS)]
    panel["MTUM"] = make_ohlcv_df(start, end, seed=9001)
    panel["SPY"] = make_ohlcv_df(start, end, seed=9002)
    panel["^GSPC"] = make_ohlcv_df(start, end, seed=9003, drift=0.0003, vol=0.01)
    vix_df = make_ohlcv_df(start, end, seed=9004, drift=0.0, vol=0.01)
    vix_df["close"] = 15.0
    vix_df["open"] = 15.0
    panel["^VIX"] = vix_df
    return panel, sector_by_ticker
