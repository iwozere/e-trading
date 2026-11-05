"""
P24 Falling Knife — portfolio layer (spec §6.3).

Replays a trade list (one variant, one timeframe) as an equal-weight book with ``max_positions`` slots. Each day exits
are processed before entries; when more signals arrive than there are free slots, the highest ``score`` wins
(tie-break: ``security_id``). A position gets ``equity / max_positions`` at entry, capped at ``liquidity_cap_frac`` of
the 20-bar median dollar volume. Cash earns nothing. Open positions are carried at cost, so the equity curve is a
realized-P&L curve (trade files hold no daily marks); its drawdown understates the intra-trade drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.ml.pipeline.p24_falling_knife.config import PortfolioParams

REQUIRED_COLUMNS = ("security_id", "entry_date", "exit_date", "net_return", "score")


@dataclass
class PortfolioResult:
    """Outcome of one portfolio replay."""

    equity: pd.Series
    taken: pd.DataFrame
    skipped: int
    max_concurrent: int

    def summary(self) -> Dict[str, float]:
        """Headline numbers: total return, CAGR, realized max drawdown, trade counts, slot use."""
        if self.equity.empty:
            return {"total_return": np.nan, "cagr": np.nan, "max_drawdown": np.nan, "taken": 0, "skipped": self.skipped,
                    "max_concurrent": 0}
        start, end = self.equity.iloc[0], self.equity.iloc[-1]
        years = max((self.equity.index[-1] - self.equity.index[0]).days / 365.25, 1e-9)
        drawdown = self.equity / self.equity.cummax() - 1.0
        return {"total_return": end / start - 1.0, "cagr": (end / start) ** (1.0 / years) - 1.0,
                "max_drawdown": float(drawdown.min()), "taken": len(self.taken), "skipped": self.skipped,
                "max_concurrent": self.max_concurrent}


def simulate_portfolio(trades: pd.DataFrame, params: PortfolioParams) -> PortfolioResult:
    """
    Replay trades through a slot-limited, equal-weight book.

    Args:
        trades: Trade rows with ``security_id``, ``entry_date``, ``exit_date``, ``net_return``, ``score`` and optionally
            ``dollar_volume_20``.
        params: Slots, liquidity cap, starting cash.

    Returns:
        Realized equity curve (indexed by date), the trades taken with their allocation, the number of signals that found
        no slot (or no cash), and the most positions held at once.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in trades.columns]
    if missing:
        raise ValueError(f"trades lack columns {missing}")
    if trades.empty:
        return PortfolioResult(pd.Series(dtype=float), trades.iloc[0:0], 0, 0)
    frame = trades.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"]).dt.date
    frame["exit_date"] = pd.to_datetime(frame["exit_date"]).dt.date
    frame = frame.sort_values(["entry_date", "score", "security_id"], ascending=[True, False, True], kind="mergesort")
    entries: Dict[date, List[int]] = {}
    for pos, day in enumerate(frame["entry_date"]):
        entries.setdefault(day, []).append(pos)
    days = sorted(set(frame["entry_date"]) | set(frame["exit_date"]))
    exit_day = frame["exit_date"].to_numpy()
    returns = frame["net_return"].to_numpy(dtype=float)
    dollar = frame["dollar_volume_20"].to_numpy(dtype=float) if "dollar_volume_20" in frame.columns else np.full(len(frame), np.nan)

    cash = params.init_cash
    open_pos: Dict[int, float] = {}
    taken: List[Tuple[int, float]] = []
    skipped = 0
    max_concurrent = 0
    curve: List[Tuple[date, float]] = []
    for day in days:
        for pos in [p for p in open_pos if exit_day[p] == day]:
            cash += open_pos.pop(pos) * (1.0 + returns[pos])
        equity = cash + sum(open_pos.values())
        for pos in entries.get(day, []):
            if len(open_pos) >= params.max_positions or exit_day[pos] < day:
                skipped += 1
                continue
            size = equity / params.max_positions
            if np.isfinite(dollar[pos]) and dollar[pos] > 0:
                size = min(size, params.liquidity_cap_frac * dollar[pos])
            size = min(size, cash)
            if size <= 0:
                skipped += 1
                continue
            cash -= size
            open_pos[pos] = size
            taken.append((pos, size))
        max_concurrent = max(max_concurrent, len(open_pos))
        for pos in [p for p in open_pos if exit_day[p] == day]:  # same-day round trips (e.g. delisted at entry)
            cash += open_pos.pop(pos) * (1.0 + returns[pos])
        curve.append((day, cash + sum(open_pos.values())))

    equity_curve = pd.Series({pd.Timestamp(d): v for d, v in curve}, dtype=float)
    picked = frame.iloc[[p for p, _ in taken]].copy()
    picked["allocation"] = [s for _, s in taken]
    return PortfolioResult(equity_curve, picked.reset_index(drop=True), skipped, max_concurrent)
