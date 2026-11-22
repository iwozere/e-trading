"""
Survivor-bias study — monthly strategies on the panel and their period statistics (docs/Design.md, "Strategies").

Every strategy returns one row per formation month: the portfolio return over the next month, the turnover cost
already taken off, and the number of names held.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.ml.pipeline.p24_falling_knife.config import CostModel
from src.ml.research.survivor_bias import config
from src.ml.research.survivor_bias.delisting import terminal_factor
from src.ml.research.survivor_bias.panel import sources_agree

_COSTS = CostModel()


def eligible_rows(panel: pd.DataFrame, members: Set[int], version: str, clip: bool = False) -> pd.DataFrame:
    """
    Stock-months that can be bought at month end, with the next-month return under a delisting version.

    Args:
        panel: Month-end panel (``panel.build_panel``).
        members: Security ids of the universe.
        version: ``last_close`` or ``haircut``.
        clip: Clip stock-month returns at ``CLIP_MONTHLY_RETURN``.

    Returns:
        Eligible rows with a ``ret`` column.
    """
    month_end = panel["month"] + pd.offsets.MonthEnd(0)
    stale = (month_end - pd.to_datetime(panel["last_date"])).dt.days
    mask = (panel["security_id"].isin(members) & (panel["close_raw"] >= config.MIN_PRICE)
            & (panel["dollar_volume"] >= config.MIN_DOLLAR_VOLUME) & (panel["bars"] >= config.MIN_BARS)
            & (stale <= config.MAX_STALE_DAYS) & panel["ret_next"].notna())
    if "ret_vendor" in panel.columns:
        mask &= sources_agree(panel["ret_next"], panel["ret_vendor"])
    if "suspect" in panel.columns:
        mask &= ~panel["suspect"].astype(bool)
    rows = panel[mask].copy()
    by_kind = {kind: terminal_factor(kind, version) for kind in (config.LIVE, config.ORDINARY, config.DISTRESSED)}
    factors = rows["end_kind"].map(by_kind).fillna(1.0).astype(float)
    factors = factors.where(rows["next_terminal"].astype(bool), 1.0)
    rows["ret"] = (1.0 + rows["ret_next"].astype(float)) * factors - 1.0
    if clip:
        rows["ret"] = rows["ret"].clip(upper=config.CLIP_MONTHLY_RETURN)
    return rows


def _cost(weights: Dict[int, float], previous: Dict[int, float], prices: Dict[int, float], flat: Optional[float]) -> float:
    total = 0.0
    for sid in set(weights) | set(previous):
        change = abs(weights.get(sid, 0.0) - previous.get(sid, 0.0))
        if change == 0.0:
            continue
        if flat is not None:
            per_side = flat
        else:
            price = prices[sid] if sid in prices else config.MIN_PRICE
            per_side = _COSTS.per_side(price)
        total += change * per_side
    return total


def _book(rows: pd.DataFrame, picks: Dict[pd.Timestamp, List[int]], flat_cost: Optional[float]) -> pd.DataFrame:
    out = []
    previous: Dict[int, float] = {}
    by_month = {m: g.set_index("security_id") for m, g in rows.groupby("month")}
    for month in sorted(picks):
        names = picks[month]
        group = by_month[month]
        weights = {sid: 1.0 / len(names) for sid in names} if names else {}
        prices = {int(sid): float(px) for sid, px in zip(group.index, group["close_raw"])}
        cost = _cost(weights, previous, prices, flat_cost)
        gross = float(group.loc[names, "ret"].mean()) if names else 0.0
        out.append({"month": month, "ret": gross - cost, "cost": cost, "names": len(names)})
        previous = weights
    return pd.DataFrame(out, columns=["month", "ret", "cost", "names"])


def ew_universe(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Equal weight of every eligible stock, no costs.

    Args:
        rows: Output of ``eligible_rows``.

    Returns:
        Frame ``month, ret, cost, names``.
    """
    grouped = rows.groupby("month")["ret"]
    out = pd.DataFrame({"ret": grouped.mean(), "names": grouped.size()}).reset_index()
    out["cost"] = 0.0
    return out[["month", "ret", "cost", "names"]]


def momentum_decile(rows: pd.DataFrame) -> pd.DataFrame:
    """
    Top 10% by 12-1 return, equal weight, monthly, P24 cost tiers on turnover.

    Args:
        rows: Output of ``eligible_rows``.

    Returns:
        Frame ``month, ret, cost, names``.
    """
    picks: Dict[pd.Timestamp, List[int]] = {}
    for _, group in rows[rows["mom_ret"].notna()].groupby("month"):
        n = int(len(group) * config.DECILE)
        if n < 1:
            continue
        picks[group["month"].iloc[0]] = group.nlargest(n, "mom_ret")["security_id"].astype(int).tolist()
    return _book(rows, picks, flat_cost=None)


def momentum_p21(rows: pd.DataFrame) -> pd.DataFrame:
    """
    P21 core: 500 most liquid, risk-adjusted 12-1 signal, hold rank 60, entry rank 20, 20 names, 10 bps per side.

    Args:
        rows: Output of ``eligible_rows``.

    Returns:
        Frame ``month, ret, cost, names``.
    """
    picks: Dict[pd.Timestamp, List[int]] = {}
    held: List[int] = []
    for _, group in rows.groupby("month"):
        pool = group.nlargest(config.P21_POOL, "dollar_volume")
        pool = pool[pool["mom_ret"].notna() & (pool["mom_vol"] >= config.MIN_VOL)]
        if pool.empty:
            held = []
            continue
        ranked = (pool["mom_ret"] / pool["mom_vol"]).set_axis(pool["security_id"].astype(int)).sort_values(ascending=False)
        rank = {sid: i + 1 for i, sid in enumerate(ranked.index)}
        keep = [sid for sid in held if rank.get(sid, config.P21_HOLD_RANK + 1) <= config.P21_HOLD_RANK]
        for limit in (config.P21_ENTRY_RANK, 2 * config.P21_ENTRY_RANK):
            for sid in ranked.index:
                if len(keep) >= config.P21_TARGET or rank[sid] > limit:
                    break
                if sid not in keep:
                    keep.append(sid)
        picks[group["month"].iloc[0]] = keep
        held = keep
    return _book(rows, picks, flat_cost=config.P21_COST_PER_SIDE)


def _max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns).cumprod()
    return float((curve / curve.cummax() - 1.0).min()) if not curve.empty else float("nan")


def period_stats(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Statistics of a monthly strategy against a monthly benchmark, per period of ``config.PERIODS``.

    Args:
        strategy: ``month, ret, cost, names``.
        benchmark: ``month, ret`` of the same universe's equal-weight book.

    Returns:
        One dict per period: ``period, months, cagr, bench_cagr, excess_ann, t_excess, max_dd, names, cost_ann``.
    """
    joined = strategy.merge(benchmark[["month", "ret"]], on="month", suffixes=("", "_bench"))
    out = []
    for name, (start, end) in config.PERIODS.items():
        part = joined[(joined["month"] >= pd.Timestamp(start)) & (joined["month"] <= pd.Timestamp(end))]
        n = len(part)
        if n == 0:
            continue
        ret = part["ret"].to_numpy(dtype=float)
        bench = part["ret_bench"].to_numpy(dtype=float)
        excess = ret - bench
        sd = float(np.std(excess, ddof=1)) if n > 1 else float("nan")
        out.append({
            "period": name,
            "months": n,
            "cagr": float(np.prod(1.0 + ret) ** (12.0 / n) - 1.0),
            "bench_cagr": float(np.prod(1.0 + bench) ** (12.0 / n) - 1.0),
            "excess_ann": float(excess.mean() * 12.0),
            "t_excess": float(excess.mean() / sd * np.sqrt(n)) if sd and sd > 0 else float("nan"),
            "max_dd": _max_drawdown(part["ret"]),
            "names": float(part["names"].mean()),
            "cost_ann": float(part["cost"].mean() * 12.0),
        })
    return out
