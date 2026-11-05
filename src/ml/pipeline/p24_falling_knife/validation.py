"""
P24 Falling Knife — statistics, controls and the go/no-go gate (spec §7, §11).

Signals cluster in crashes, so every confidence interval is a block bootstrap over **calendar months of the signal
date**: whole months are resampled, never single trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.ml.pipeline.p24_falling_knife.config import CostModel

N_BOOT = 2000


def trade_metrics(trades: pd.DataFrame, column: str = "net_return") -> Dict[str, float]:
    """
    Trade-level summary statistics (spec §7.1).

    Args:
        trades: Trade rows.
        column: Return column.

    Returns:
        Dict of metrics (NaN where undefined).
    """
    r = trades[column].dropna().to_numpy(dtype=float) if column in trades.columns else np.array([])
    if r.size == 0:
        return {"count": 0.0}
    wins, losses = r[r > 0], r[r <= 0]
    trimmed = np.sort(r)[int(0.05 * r.size): max(int(0.05 * r.size) + 1, int(np.ceil(0.95 * r.size)))]
    bars = trades["bars_held"].to_numpy(dtype=float) if "bars_held" in trades.columns else np.full(r.size, np.nan)
    out = {
        "count": float(r.size),
        "win_rate": float((r > 0).mean()),
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "trimmed_mean_5": float(trimmed.mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "avg_win": float(wins.mean()) if wins.size else float("nan"),
        "avg_loss": float(losses.mean()) if losses.size else float("nan"),
        "mean_bars_held": float(np.nanmean(bars)) if bars.size else float("nan"),
        "share_le_minus_50": float((r <= -0.5).mean()),
    }
    mean_bars = out["mean_bars_held"]
    out["return_per_bar"] = out["mean"] / mean_bars if mean_bars and np.isfinite(mean_bars) else float("nan")
    if "excess_spy" in trades.columns:
        out["mean_excess_spy"] = float(trades["excess_spy"].mean())
    return out


def _month_keys(trades: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(trades["signal_date"]).dt.to_period("M").astype(str).to_numpy()


def bootstrap_ci(trades: pd.DataFrame, column: str = "net_return", stat: str = "mean", n_boot: int = N_BOOT,
                 seed: int = 24, level: float = 0.95) -> Dict[str, float]:
    """
    Month-block bootstrap confidence interval for the mean (or win rate) of a return column.

    Args:
        trades: Trade rows with ``signal_date``.
        column: Return column.
        stat: ``"mean"`` or ``"win_rate"``.
        n_boot: Bootstrap replications.
        seed: RNG seed.
        level: Confidence level.

    Returns:
        ``{"estimate", "lo", "hi", "months"}``.
    """
    frame = trades.dropna(subset=[column])
    if frame.empty:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "months": 0.0}
    values = frame[column].to_numpy(dtype=float)
    if stat == "win_rate":
        values = (values > 0).astype(float)
    keys = _month_keys(frame)
    months, inverse = np.unique(keys, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(months), size=(n_boot, len(months)))
    estimates = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    alpha = (1.0 - level) / 2.0
    return {"estimate": float(values.mean()), "lo": float(np.quantile(estimates, alpha)),
            "hi": float(np.quantile(estimates, 1.0 - alpha)), "months": float(len(months))}


def paired_difference_ci(system: pd.DataFrame, control: pd.DataFrame, column: str = "net_return",
                         n_boot: int = N_BOOT, seed: int = 24, level: float = 0.95) -> Dict[str, float]:
    """
    Month-block bootstrap CI of ``mean(system) − mean(control)``, resampling the same months for both.

    Args:
        system: Trades of the tested variant.
        control: Trades of the control variant.
        column: Return column.
        n_boot: Replications.
        seed: RNG seed.
        level: Confidence level.

    Returns:
        ``{"estimate", "lo", "hi"}``.
    """
    sys_f, ctl_f = system.dropna(subset=[column]), control.dropna(subset=[column])
    if sys_f.empty or ctl_f.empty:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan")}
    months = np.union1d(_month_keys(sys_f), _month_keys(ctl_f))
    index = {m: i for i, m in enumerate(months)}

    def sums(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        idx = np.array([index[m] for m in _month_keys(frame)])
        return (np.bincount(idx, weights=frame[column].to_numpy(dtype=float), minlength=len(months)),
                np.bincount(idx, minlength=len(months)).astype(float))

    s_sum, s_cnt = sums(sys_f)
    c_sum, c_cnt = sums(ctl_f)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(months), size=(n_boot, len(months)))
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = s_sum[picks].sum(1) / s_cnt[picks].sum(1) - c_sum[picks].sum(1) / c_cnt[picks].sum(1)
    diff = diff[np.isfinite(diff)]
    alpha = (1.0 - level) / 2.0
    estimate = float(sys_f[column].mean() - ctl_f[column].mean())
    return {"estimate": estimate, "lo": float(np.quantile(diff, alpha)), "hi": float(np.quantile(diff, 1 - alpha))}


def reprice_costs(trades: pd.DataFrame, costs: CostModel) -> pd.Series:
    """
    Net returns under another cost model, from gross returns and the as-traded entry/exit prices.

    Args:
        trades: Trades with ``gross_return``, ``entry_price_raw``, ``exit_price_raw``.
        costs: Cost model (e.g. with ``multiplier=2``).

    Returns:
        Net return series.
    """
    entry_cost = trades["entry_price_raw"].map(costs.per_side)
    exit_cost = trades["exit_price_raw"].map(costs.per_side)
    return (1.0 + trades["gross_return"]) * (1.0 - exit_cost) / (1.0 + entry_cost) - 1.0


def without_best_month(trades: pd.DataFrame, column: str = "net_return") -> pd.DataFrame:
    """
    Drop the calendar month (of the signal date) with the largest summed return (spec §11 item 5).

    Args:
        trades: Trade rows.
        column: Return column.

    Returns:
        Trades without that month.
    """
    if trades.empty:
        return trades
    keys = pd.Series(_month_keys(trades), index=trades.index)
    best = trades[column].groupby(keys).sum().idxmax()
    return trades[keys != best]


def random_control_p_value(draw_means: pd.Series, system_mean: float) -> float:
    """
    Share of matched-random draws whose mean return is at least the system's (one-sided).

    Args:
        draw_means: Per-draw mean returns of the random control.
        system_mean: Mean return of the full system on the same signals.

    Returns:
        p-value in [0, 1] (NaN without draws).
    """
    values = draw_means.dropna().to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(system_mean):
        return float("nan")
    return float((values >= system_mean).mean())


@dataclass
class GateResult:
    """One gate criterion (spec §11)."""

    name: str
    passed: Optional[bool]
    detail: str


def evaluate_gate(system: pd.DataFrame, control: pd.DataFrame, costs: CostModel, min_trades: int = 60) -> List[GateResult]:
    """
    Spec §11 go/no-go criteria on one timeframe's holdout trades (``suspect_gap`` excluded by the caller).

    Args:
        system: Full-system trades.
        control: Drop-only control trades of the same period.
        costs: The run's cost model (1×).
        min_trades: Minimum number of trades.

    Returns:
        One ``GateResult`` per criterion.
    """
    out: List[GateResult] = []
    n = len(system)
    out.append(GateResult("trade_count", n >= min_trades, f"{n} trades (need >= {min_trades})"))
    metrics = trade_metrics(system)
    ci = bootstrap_ci(system)
    ok2 = n > 0 and metrics["median"] > 0 and metrics["mean"] > 0 and ci["lo"] > 0
    out.append(GateResult("positive_returns", bool(ok2) if n else None,
                          f"mean {metrics.get('mean', float('nan')):.4f}, median {metrics.get('median', float('nan')):.4f}, "
                          f"95% CI of mean [{ci['lo']:.4f}, {ci['hi']:.4f}]"))
    diff = paired_difference_ci(system, control)
    out.append(GateResult("beats_drop_only", bool(diff["lo"] > 0) if np.isfinite(diff["lo"]) else None,
                          f"mean diff {diff['estimate']:.4f}, 95% CI [{diff['lo']:.4f}, {diff['hi']:.4f}]"))
    if n and {"entry_price_raw", "exit_price_raw", "gross_return"} <= set(system.columns):
        double = reprice_costs(system, CostModel(tiers=costs.tiers, multiplier=costs.multiplier * 2.0))
        out.append(GateResult("survives_2x_costs", bool(double.mean() > 0), f"mean at 2x costs {double.mean():.4f}"))
    else:
        out.append(GateResult("survives_2x_costs", None, "no trades or raw prices missing"))
    rest = without_best_month(system)
    rest_m = trade_metrics(rest)
    ok5 = len(rest) > 0 and rest_m["mean"] > 0 and rest_m["median"] > 0
    out.append(GateResult("not_one_month", bool(ok5) if len(rest) else None,
                          f"without best month: mean {rest_m.get('mean', float('nan')):.4f}, "
                          f"median {rest_m.get('median', float('nan')):.4f}"))
    out.append(GateResult("win_rate_vs_book", None, f"win rate {metrics.get('win_rate', float('nan')):.1%} (book claims >80%)"))
    return out
