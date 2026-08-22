"""
P21 Momentum backtest — stress windows (docs/pipeline-specification.md §14.6).

Table-driven from the spec's own nine-window list, so adding a tenth window
later is a one-line change to :data:`STRESS_WINDOWS`
(docs/implementation-plan.md §8.5). Each window is evaluated independently
against tracks A-E plus the realized ``regime_scalar`` path, per spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from backtest.p21_momentum.metrics import compute_risk_metrics

if TYPE_CHECKING:
    from backtest.p21_momentum.runner import BacktestResult

_TRACK_COLUMNS = ("nav_a", "nav_b", "nav_c", "nav_d", "nav_e")


@dataclass(slots=True)
class StressWindow:
    """One row of spec §14.6's window table."""

    name: str
    start: date
    end: date
    event: str
    question: str


STRESS_WINDOWS: List[StressWindow] = [
    StressWindow(
        "2009-03 -> 2009-05", date(2009, 3, 1), date(2009, 5, 31),
        "Momentum crash, ~-70% for the academic factor",
        "The decisive test. Does the overlay reduce the drawdown? If A-B is not strongly "
        "positive here, the overlay has no reason to exist.",
    ),
    StressWindow(
        "2008-09 -> 2009-02", date(2008, 9, 1), date(2009, 2, 28),
        "Bear market, slow decline",
        "Does the overlay de-risk with reasonable lag?",
    ),
    StressWindow(
        "2011-08 -> 2011-10", date(2011, 8, 1), date(2011, 10, 31),
        "Volatility spike, no sustained bear",
        "Whipsaw test - does the overlay incur cost for no benefit?",
    ),
    StressWindow(
        "2015-08 -> 2016-02", date(2015, 8, 1), date(2016, 2, 29),
        "Two corrections, rapid reversals",
        "Whipsaw test",
    ),
    StressWindow(
        "2018-10 -> 2018-12", date(2018, 10, 1), date(2018, 12, 31),
        "Fast drawdown, sharp recovery",
        "Overlay likely hurts here; quantify the cost",
    ),
    StressWindow(
        "2020-02 -> 2020-04", date(2020, 2, 1), date(2020, 4, 30),
        "COVID crash, 23 sessions peak-to-trough",
        "Overlay expected to fail (too fast). Confirm and size the failure.",
    ),
    StressWindow(
        "2020-11", date(2020, 11, 1), date(2020, 11, 30),
        "Vaccine rotation, ~-15% momentum month",
        "Overlay cannot help; measures raw factor fragility",
    ),
    StressWindow(
        "2022-01 -> 2022-10", date(2022, 1, 1), date(2022, 10, 31),
        "Slow bear market",
        "Second decisive test - the overlay's best-case scenario",
    ),
    StressWindow(
        "2023-01 -> 2023-12", date(2023, 1, 1), date(2023, 12, 31),
        "MTUM ~+9% vs S&P ~+26%",
        "Rebalance-timing failure after a bear year. Does the 20-name version with monthly "
        "rebalancing recover faster than the semi-annual ETF?",
    ),
]


@dataclass(slots=True)
class WindowMetrics:
    """One track's realized outcome within one stress window."""

    window_return: Optional[float]
    max_drawdown: Optional[float]


@dataclass(slots=True)
class StressWindowResult:
    """One window's full evaluation across all five tracks."""

    window: StressWindow
    in_range: bool  # False if the backtest's own date range doesn't cover this window
    track_metrics: dict  # {"nav_a": WindowMetrics, ...}
    a_minus_b: Optional[float]  # spec §14.9 B5/B6/B7's decisive comparison
    regime_scalar_min: Optional[float]
    regime_scalar_max: Optional[float]


def _slice_window(nav: pd.Series, start: date, end: date) -> pd.Series:
    ts_start, ts_end = pd.Timestamp(start), pd.Timestamp(end)
    return nav.loc[(nav.index >= ts_start) & (nav.index <= ts_end)].dropna()


def _window_return(nav_slice: pd.Series) -> Optional[float]:
    if len(nav_slice) < 2 or nav_slice.iloc[0] == 0:
        return None
    return float(nav_slice.iloc[-1] / nav_slice.iloc[0] - 1.0)


def evaluate_window(result: "BacktestResult", window: StressWindow) -> StressWindowResult:
    """Evaluate one stress window against a full backtest result's nav_daily + regime_history."""
    nav_daily = result.nav_daily
    if nav_daily.empty:
        in_range = False
    else:
        in_range = bool(
            nav_daily.index.min() <= pd.Timestamp(window.end) and nav_daily.index.max() >= pd.Timestamp(window.start)
        )

    track_metrics = {}
    for col in _TRACK_COLUMNS:
        if not in_range or col not in nav_daily.columns:
            track_metrics[col] = WindowMetrics(None, None)
            continue
        nav_slice = _slice_window(nav_daily[col], window.start, window.end)
        window_return = _window_return(nav_slice)
        max_dd = compute_risk_metrics(nav_slice).max_drawdown if len(nav_slice) >= 2 else None
        track_metrics[col] = WindowMetrics(window_return, max_dd)

    ret_a, ret_b = track_metrics["nav_a"].window_return, track_metrics["nav_b"].window_return
    a_minus_b = (ret_a - ret_b) if ret_a is not None and ret_b is not None else None

    scalars = [
        float(e["scalar_applied"])
        for e in result.regime_history
        if window.start.isoformat() <= e["date"] <= window.end.isoformat()
    ]
    scalar_min = min(scalars) if scalars else None
    scalar_max = max(scalars) if scalars else None

    return StressWindowResult(
        window=window,
        in_range=in_range,
        track_metrics=track_metrics,
        a_minus_b=a_minus_b,
        regime_scalar_min=scalar_min,
        regime_scalar_max=scalar_max,
    )


def evaluate_all_windows(result: "BacktestResult") -> List[StressWindowResult]:
    """Evaluate every window in :data:`STRESS_WINDOWS` against one backtest result."""
    return [evaluate_window(result, w) for w in STRESS_WINDOWS]


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v * 100:+.1f}%" if v is not None else "N/A (outside backtest range)"


def render_stress_windows_md(results: List[StressWindowResult]) -> str:
    """Render spec §14.10's ``stress_windows.md`` deliverable."""
    lines = [
        "# Stress Windows (spec §14.6)",
        "",
        "Each window evaluated independently against tracks A-E. `A-B` is the decisive overlay",
        "comparison for B5/B6/B7 of the §14.9 acceptance table.",
        "",
    ]
    for r in results:
        w = r.window
        lines.append(f"## {w.name} — {w.event}")
        lines.append("")
        lines.append(f"_{w.question}_")
        lines.append("")
        if not r.in_range:
            lines.append("Outside this backtest's date range — not evaluated.")
            lines.append("")
            continue
        lines.append("| Track | Return | Max Drawdown |")
        lines.append("|---|---|---|")
        for col, label in zip(_TRACK_COLUMNS, ("A", "B", "C", "D", "E")):
            m = r.track_metrics[col]
            lines.append(f"| {label} | {_fmt_pct(m.window_return)} | {_fmt_pct(m.max_drawdown)} |")
        lines.append("")
        lines.append(f"**A - B: {_fmt_pct(r.a_minus_b)}**")
        lines.append("")
        if r.regime_scalar_min is not None:
            lines.append(f"Regime scalar range during window: {r.regime_scalar_min:.2f} - {r.regime_scalar_max:.2f}")
        else:
            lines.append("Regime scalar: no regime history recorded during window.")
        lines.append("")
    return "\n".join(lines)
