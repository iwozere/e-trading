"""
Survivor-bias study — how a stock's price history ends, and what that costs a holder (docs/Design.md).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.ml.research.survivor_bias import config


def classify_end(symbol: str, is_active: bool, last_date: Optional[date], store_date: date, raw_close: pd.Series,
                 total_close: pd.Series) -> str:
    """
    Classify the end of one stock's history.

    Args:
        symbol: Traded symbol.
        is_active: The symbol list marks the stock as trading.
        last_date: Date of its last bar.
        store_date: Latest date in the store.
        raw_close: Raw (as-traded) daily closes, ascending.
        total_close: Total-return adjusted daily closes, ascending, same index.

    Returns:
        ``live``, ``ordinary`` (delisted, last close is a fair exit) or ``distressed``.
    """
    if is_active or last_date is None or last_date >= store_date - timedelta(days=config.DELISTED_GAP_DAYS):
        return config.LIVE
    if len(symbol) == 5 and symbol.endswith("Q"):
        return config.DISTRESSED
    raw = raw_close.dropna()
    if not raw.empty and float(raw.iloc[-1]) < config.DISTRESS_PRICE:
        return config.DISTRESSED
    total = total_close.dropna()
    if len(total) > config.DISTRESS_LOOKBACK_BARS:
        change = float(total.iloc[-1]) / float(total.iloc[-1 - config.DISTRESS_LOOKBACK_BARS]) - 1.0
        if change <= config.DISTRESS_DROP:
            return config.DISTRESSED
    return config.ORDINARY


def terminal_factor(end_kind: str, version: str) -> float:
    """
    Multiplier on the last close a holder receives at delisting.

    Args:
        end_kind: Result of ``classify_end``.
        version: ``last_close`` or ``haircut``.

    Returns:
        1.0, or 0.70 for a distressed delisting under ``haircut``.

    Raises:
        ValueError: Unknown version.
    """
    if version not in config.DELISTING_VERSIONS:
        raise ValueError(f"unknown delisting version {version!r}")
    if version == config.WITH_HAIRCUT and end_kind == config.DISTRESSED:
        return 1.0 + config.HAIRCUT
    return 1.0


def apply_to_trades(trades: pd.DataFrame, end_kinds: Dict[int, str], version: str, return_column: str = "net_return") -> pd.DataFrame:
    """
    Apply the delisting rule to trades that end with the stock's delisting.

    Args:
        trades: Trade rows with ``security_id``, ``delisted_in_trade`` and ``return_column``.
        end_kinds: ``security_id`` -> end kind.
        version: ``last_close`` or ``haircut``.
        return_column: Column to adjust.

    Returns:
        Copy with the adjusted return column and an ``end_kind`` column.
    """
    out = trades.copy()
    if out.empty:
        out["end_kind"] = pd.Series(dtype=str)
        return out
    out["end_kind"] = out["security_id"].map(end_kinds).fillna(config.LIVE)
    factors = np.array([terminal_factor(kind, version) for kind in out["end_kind"]], dtype=float)
    hit = out["delisted_in_trade"].fillna(False).astype(bool).to_numpy()
    factors = np.where(hit, factors, 1.0)
    out[return_column] = (1.0 + out[return_column].astype(float)) * factors - 1.0
    return out
