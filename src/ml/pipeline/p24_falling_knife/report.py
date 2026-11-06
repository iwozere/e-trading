"""
P24 Falling Knife — Phase 1 report (spec §8, §11).

Reads one backtest run folder and writes ``report.md``: the survivorship warning first, then per timeframe the
variant table, controls, exit reasons, cost sensitivity, price buckets, concentration and — when holdout trades
exist — the go/no-go gate. Trades flagged ``suspect_gap`` or ``suspect_in_trade`` are excluded from every statistic and counted separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.ml.pipeline.p24_falling_knife.backtest import FULL, HOLDOUT, TRAIN, UPPER_BOUND
from src.ml.pipeline.p24_falling_knife.config import DAILY, WEEKLY, CostModel, PortfolioParams
from src.ml.pipeline.p24_falling_knife.portfolio import simulate_portfolio
from src.ml.pipeline.p24_falling_knife.validation import (
    bootstrap_ci,
    clean_trades,
    evaluate_gate,
    paired_difference_ci,
    random_control_p_value,
    reprice_costs,
    trade_metrics,
)

CONTROL = "drop_only"
PERIODS = (TRAIN, HOLDOUT, UPPER_BOUND)
COST_MULTIPLIERS = (0.5, 1.0, 2.0, 3.0)


def _fmt(value: float, pct: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.1%}" if pct else f"{value:.2f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _clean(trades: pd.DataFrame) -> pd.DataFrame:
    return clean_trades(trades)


def survivorship_section(run_dir: Path) -> str:
    """Markdown for the §7.4 survivorship table."""
    path = run_dir / "survivorship.csv"
    if not path.exists():
        return "## Survivorship\n\nNo survivorship.csv in this run.\n"
    table = pd.read_csv(path)
    rows = [[str(year), str(listed), str(delisted), _fmt(rate), _fmt(coverage), "yes" if thin_year else ""]
            for year, listed, delisted, rate, coverage, thin_year in zip(
                table["year"].astype(int), table["listed"].astype(int), table["delisted"].astype(int),
                table["delisting_rate"].astype(float), table["est_coverage"].astype(float),
                table["upper_bound_only"].astype(bool))]
    thin = table[table["upper_bound_only"]]["year"].astype(int).tolist()
    return ("## Survivorship (read first)\n\n"
            "Coverage is estimated as the year's delisting rate over the median rate of the reference years. Years "
            f"marked upper-bound-only ({thin[0] if thin else '—'}–{thin[-1] if thin else '—'}) miss most delisted "
            "names, so results there are optimistic (bankrupt knives are missing).\n\n"
            + _table(["year", "listed", "delisted", "rate", "est. coverage", "upper bound only"], rows) + "\n")


def variant_section(trades: pd.DataFrame) -> str:
    """Markdown table: every variant × period, net of 1× costs, suspect gaps excluded."""
    rows = []
    for (variant, period), part in sorted(_clean(trades).groupby(["variant", "period"])):
        m = trade_metrics(part)
        ci = bootstrap_ci(part)
        rows.append([variant, period, str(int(m["count"])), _fmt(m.get("win_rate", np.nan)), _fmt(m.get("mean", np.nan)),
                     _fmt(m.get("median", np.nan)), f"[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}]",
                     _fmt(m.get("profit_factor", np.nan), pct=False), _fmt(m.get("mean_excess_spy", np.nan)),
                     _fmt(m.get("mean_bars_held", np.nan), pct=False)])
    return _table(["variant", "period", "n", "win", "mean", "median", "95% CI mean", "PF", "excess SPY", "bars"], rows)


def control_section(trades: pd.DataFrame, draws: pd.DataFrame, timeframe: str) -> str:
    """Markdown: full vs drop-only (paired month bootstrap) and vs the matched-random control."""
    clean = _clean(trades)
    rows = []
    for period in PERIODS:
        full = clean[(clean["variant"] == FULL) & (clean["period"] == period)]
        ctl = clean[(clean["variant"] == CONTROL) & (clean["period"] == period)]
        if full.empty:
            continue
        diff = paired_difference_ci(full, ctl) if not ctl.empty else {"estimate": np.nan, "lo": np.nan, "hi": np.nan}
        part = draws[(draws["timeframe"] == timeframe) & (draws["period"] == period)] if not draws.empty else draws
        p_value = random_control_p_value(part["mean_net_return"], float(full["net_return"].mean())) if not part.empty else np.nan
        random_mean = float(part["mean_net_return"].mean()) if not part.empty else np.nan
        rows.append([period, _fmt(diff["estimate"]), f"[{_fmt(diff['lo'])}, {_fmt(diff['hi'])}]", _fmt(random_mean),
                     _fmt(p_value, pct=False)])
    return _table(["period", "full − drop-only", "95% CI", "random control mean", "p (random ≥ full)"], rows)


def exit_section(trades: pd.DataFrame) -> str:
    """Markdown: exit reasons of the full system by period."""
    full = _clean(trades)[_clean(trades)["variant"] == FULL]
    if full.empty:
        return "No full-system trades."
    rows = []
    for (period, reason), part in sorted(full.groupby(["period", "exit_reason"])):
        rows.append([period, reason, str(len(part)), _fmt(float(part["net_return"].mean()))])
    return _table(["period", "exit reason", "n", "mean net"], rows)


def cost_section(trades: pd.DataFrame, costs: CostModel) -> str:
    """Markdown: full-system mean net return at 0.5× … 3× costs."""
    full = _clean(trades)[_clean(trades)["variant"] == FULL]
    if full.empty or "exit_price_raw" not in full.columns:
        return "No full-system trades with raw prices."
    rows = []
    for period, part in sorted(full.groupby("period")):
        values = [reprice_costs(part, CostModel(tiers=costs.tiers, multiplier=m)).mean() for m in COST_MULTIPLIERS]
        rows.append([str(period)] + [_fmt(float(v)) for v in values])
    return _table(["period"] + [f"{m}× costs" for m in COST_MULTIPLIERS], rows)


def bucket_section(trades: pd.DataFrame) -> str:
    """Markdown: full system by as-traded price bucket."""
    full = _clean(trades)[_clean(trades)["variant"] == FULL]
    rows = []
    for (period, bucket), part in sorted(full.groupby(["period", "price_bucket"])):
        m = trade_metrics(part)
        rows.append([period, bucket, str(int(m["count"])), _fmt(m["win_rate"]), _fmt(m["mean"]), _fmt(m["median"])])
    return _table(["period", "price", "n", "win", "mean", "median"], rows) if rows else "No full-system trades."


def concentration_section(trades: pd.DataFrame) -> str:
    """Markdown: how much of the full system's trades come from its busiest months."""
    full = _clean(trades)[_clean(trades)["variant"] == FULL]
    if full.empty:
        return "No full-system trades."
    months = pd.to_datetime(full["signal_date"]).dt.to_period("M")
    counts = months.value_counts()
    top3 = counts.head(3)
    share = top3.sum() / counts.sum()
    listing = ", ".join(f"{p} ({n})" for p, n in top3.items())
    return f"Busiest months: {listing}; top-3 share of all trades {share:.1%}; max signals in one month {counts.max()}."


def portfolio_section(trades: pd.DataFrame, params: PortfolioParams) -> str:
    """Markdown: full system replayed through the slot-limited book (spec §6.3), per period."""
    full = _clean(trades)[_clean(trades)["variant"] == FULL]
    rows = []
    for period, part in sorted(full.groupby("period")):
        s = simulate_portfolio(part, params).summary()
        rows.append([period, str(int(s["taken"])), str(int(s["skipped"])), str(int(s["max_concurrent"])),
                     _fmt(s["total_return"]), _fmt(s["cagr"]), _fmt(s["max_drawdown"])])
    note = (f"{params.max_positions} slots, equal weight, liquidity cap {params.liquidity_cap_frac:.0%} of 20-bar dollar "
            "volume, cash earns 0; open positions carried at cost (realized drawdown).\n\n")
    return note + _table(["period", "taken", "no slot", "max open", "total", "CAGR", "max DD"], rows) if rows else "No full-system trades."


def gate_timeframes(run_dir: Path) -> List[str]:
    """Timeframes the owner gate is decided on (``_meta.gate_timeframes`` of the run's calibration; default both)."""
    path = run_dir / "calibration_frozen.json"
    if not path.exists():
        return [DAILY, WEEKLY]
    meta = json.loads(path.read_text(encoding="utf-8")).get("_meta", {})
    return list(meta.get("gate_timeframes", [DAILY, WEEKLY]))


def gate_section(trades: pd.DataFrame, costs: CostModel, info_only: bool = False) -> str:
    """Markdown: spec §11 gate on holdout trades, or a note that the holdout has not been run."""
    clean = _clean(trades)
    full = clean[(clean["variant"] == FULL) & (clean["period"] == HOLDOUT)]
    if full.empty:
        return "Holdout not evaluated in this run (needs `--holdout --calibration ...`)."
    control = clean[(clean["variant"] == CONTROL) & (clean["period"] == HOLDOUT)]
    rows = [[g.name, {True: "PASS", False: "FAIL", None: "info"}[g.passed], g.detail]
            for g in evaluate_gate(full, control, costs)]
    note = "**Information only:** this timeframe is outside the owner gate (calibration `gate_timeframes`).\n\n" if info_only else ""
    return note + _table(["criterion", "result", "detail"], rows)


def samples_section(run_dir: Path) -> str:
    """Markdown image list of the run's sample trade charts (``samples.write_sample_charts``)."""
    folder = run_dir / "charts" / "samples"
    images = sorted(folder.glob("*.png")) if folder.exists() else []
    if not images:
        return "## Sample trade charts\n\nNo sample charts in this run.\n"
    lines = [f"![{p.stem}](charts/samples/{p.name})" for p in images]
    return ("## Sample trade charts\n\nSeeded random sample of clean full-system trades (headline periods first).\n\n"
            + "\n\n".join(lines) + "\n")


def build_report(run_dir: Path, costs: CostModel, portfolio: PortfolioParams = PortfolioParams()) -> str:
    """
    Assemble ``report.md`` for a run folder.

    Args:
        run_dir: Backtest run folder.
        costs: The run's 1× cost model.
        portfolio: Portfolio-layer settings.

    Returns:
        Markdown text.
    """
    draws_path = run_dir / "random_control_draws.csv"
    draws = pd.read_csv(draws_path) if draws_path.exists() and draws_path.stat().st_size > 0 else pd.DataFrame()
    parts: List[str] = [f"# P24 Falling Knife — backtest report `{run_dir.name}`\n",
                        "Returns are per trade, net of tiered costs (as-traded price), split-adjusted, dividends "
                        "excluded. Entry next open after the signal bar. CIs are month-block bootstraps. Trades whose "
                        "drop (`suspect_gap`) or holding period (`suspect_in_trade`) contains a suspected unadjusted corporate event are excluded below.\n",
                        survivorship_section(run_dir)]
    gated = gate_timeframes(run_dir)
    for timeframe in (DAILY, WEEKLY):
        path = run_dir / f"trades_{timeframe}.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        trades = pd.read_csv(path)
        if trades.empty:
            continue
        suspect = len(trades) - len(_clean(trades))
        sections: Dict[str, str] = {
            "Variants (ablations)": variant_section(trades),
            "Controls": control_section(trades, draws, timeframe),
            "Exit reasons (full system)": exit_section(trades),
            "Cost sensitivity (full system, mean net)": cost_section(trades, costs),
            "Price buckets (full system)": bucket_section(trades),
            "Concentration (full system)": concentration_section(trades),
            "Portfolio (full system)": portfolio_section(trades, portfolio),
            "Go/no-go gate (holdout)": gate_section(trades, costs, info_only=timeframe not in gated),
        }
        parts.append(f"## {timeframe.capitalize()}\n\n{suspect} trade rows flagged `suspect_gap` or `suspect_in_trade` were excluded.\n")
        parts.extend(f"### {title}\n\n{body}\n" for title, body in sections.items())
    parts.append(samples_section(run_dir))
    return "\n".join(parts)


def write_report(run_dir: Path, costs: CostModel) -> Path:
    """
    Write ``report.md`` into the run folder.

    Args:
        run_dir: Backtest run folder.
        costs: The run's 1× cost model.

    Returns:
        Path of the report.
    """
    path = run_dir / "report.md"
    path.write_text(build_report(run_dir, costs), encoding="utf-8")
    return path
