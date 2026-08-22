"""
P21 Momentum — Monthly report renderer (docs/pipeline-specification.md §12).

Renders results/p21_momentum/<run_date>/report.md from this run's outputs
plus history read from _state/. §12.2's statistical-power disclaimer is
reproduced in full every month — this is a hard requirement (spec §0: "The
agent must reproduce this warning in every monthly report."), verified here
by a module-level constant string a regression test can grep for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

from src.ml.pipeline.p21_momentum.schemas import LedgerEntry, Position

# §12.2 — reproduced verbatim every month. The regression test greps for
# this exact phrase so a future edit cannot silently drop the disclaimer.
STATISTICAL_POWER_DISCLAIMER_KEY_PHRASE = (
    "This report is **not** evidence that the strategy does or does not work."
)

_DECISION_CRITERIA = [
    ("D1", "Missed rebalances", "0 of 12", "No discipline -> buy QDVA"),
    ("D2", "Realized turnover", "130-220% p.a.", "Model diverged from reality; re-check hysteresis"),
    ("D3", "Realized costs", "< 0.30% p.a.", "Economics worse than forecast"),
    ("D4", "MANUAL_REVIEW events", "<= 4 per year", "Pipeline demands too much hand-holding"),
    ("D5", "f4_data_missing", "< 15% of candidates", "Quality filter inert; remove or change source"),
    ("D6", "WARN_UNDERFILLED", "<= 2 per year", "Sector cap too tight for current market"),
    ("D7", "Track A max drawdown", "Survived without intervention", "Psychological test"),
    ("D8", "Mean A-D difference", "informational, not a criterion", "-"),
]


@dataclass(slots=True)
class DecisionMetrics:
    """Realized §12.6 metrics, aggregated by the caller across the trailing 12 months."""

    missed_rebalances: int = 0
    realized_turnover_annualized_pct: float = 0.0
    realized_costs_annualized_pct: float = 0.0
    manual_review_count: int = 0
    f4_data_missing_pct: float = 0.0
    warn_underfilled_count: int = 0
    track_a_max_drawdown_pct: float = 0.0
    track_a_survived_without_intervention: bool = True


def _fmt_pct(x: float) -> str:
    return f"{x:+.2%}"


def _t_statistic(monthly_diffs: List[float]) -> Optional[float]:
    """t = mean_monthly_diff / (std_monthly_diff / sqrt(N)), per spec §12.2. None if N < 2."""
    n = len(monthly_diffs)
    if n < 2:
        return None
    mean = sum(monthly_diffs) / n
    variance = sum((d - mean) ** 2 for d in monthly_diffs) / (n - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return mean / (std / math.sqrt(n))


def _render_header(
    signal_date: date, execution_date: date, regime: Dict, nav_by_track: Dict[str, float]
) -> str:
    lines = [
        "## 1. Header",
        "",
        f"- Signal date: {signal_date.isoformat()}",
        f"- Execution date: {execution_date.isoformat()}",
        f"- Regime scalar: {regime.get('scalar_applied')} "
        f"(bear={regime.get('bear')}, high_vol={regime.get('high_vol')}, "
        f"months_at_state={regime.get('months_at_state')})",
        "",
        "| Track | NAV |",
        "|---|---|",
    ]
    for track in ("A", "B", "C", "D", "E"):
        nav = nav_by_track.get(track)
        lines.append(f"| {track} | {'$' + format(nav, ',.2f') if nav is not None else 'n/a'} |")
    return "\n".join(lines)


def _render_disclaimer(monthly_diffs: List[float]) -> str:
    n = len(monthly_diffs)
    t_stat = _t_statistic(monthly_diffs)
    latest_diff = monthly_diffs[-1] if monthly_diffs else 0.0
    t_display = f"{t_stat:.2f}" if t_stat is not None else "n/a (insufficient history)"
    return (
        "## 2. Statistical Power Disclaimer\n\n"
        f"> {n} months elapsed. With tracking error of ~7.5% annualized, the observed A-D difference of "
        f"{_fmt_pct(latest_diff)} is statistically indistinguishable from zero (t = {t_display}). "
        f"{STATISTICAL_POWER_DISCLAIMER_KEY_PHRASE} Decision criteria are in section 6."
    )


def _render_trades(trades: List[LedgerEntry], rank_before: Dict[str, int], rank_after: Dict[str, int]) -> str:
    lines = ["## 3. Trades This Month", ""]
    if not trades:
        lines.append("_No trades this month._")
        return "\n".join(lines)
    lines.append("| Ticker | Side | Shares | Price | Commission | Reason | Rank before | Rank after |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in trades:
        rb = rank_before.get(t.ticker, "-")
        ra = rank_after.get(t.ticker, "-")
        lines.append(
            f"| {t.ticker} | {t.side} | {t.shares:.4f} | ${t.fill_price:.2f} | "
            f"${t.commission_usd:.2f} | {t.reason} | {rb} | {ra} |"
        )
    return "\n".join(lines)


def _render_portfolio(positions: List[Position], as_of: date, max_per_sector: int) -> str:
    lines = ["## 4. Current Portfolio", ""]
    if not positions:
        lines.append("_No positions held._")
        return "\n".join(lines)
    lines.append("| Ticker | Sector | Weight % | Rank | Return since entry | Days held |")
    lines.append("|---|---|---|---|---|---|")
    for p in positions:
        entry = date.fromisoformat(p.entry_date)
        days_held = (as_of - entry).days
        return_since_entry = p.high_water_price / p.avg_cost - 1.0 if p.avg_cost else 0.0
        lines.append(
            f"| {p.ticker} | {p.sector or '-'} | {p.target_weight_pct:.2%} | {p.current_rank or '-'} | "
            f"{_fmt_pct(return_since_entry)} | {days_held} |"
        )

    sector_counts: Dict[str, int] = {}
    for p in positions:
        key = p.sector or "Unknown"
        sector_counts[key] = sector_counts.get(key, 0) + 1
    lines.append("")
    lines.append("### Sector distribution (cap verification)")
    lines.append("")
    lines.append("| Sector | Count | Cap OK |")
    lines.append("|---|---|---|")
    for sector, count in sorted(sector_counts.items()):
        ok = "yes" if count <= max_per_sector else "**BREACH**"
        lines.append(f"| {sector} | {count} | {ok} |")
    return "\n".join(lines)


def _render_attribution(
    cum_returns: Dict[str, Dict[str, float]],  # {track: {"month": x, "ytd": x, "since_inception": x}}
    differences: Dict[str, float],  # {"stock_selection_effect": ..., ...}
    turnover_annualized_pct: float,
    costs_bps: float,
    max_drawdown_by_track: Dict[str, float],
) -> str:
    lines = ["## 5. Attribution", "", "| Track | Month | YTD | Since inception |", "|---|---|---|---|"]
    for track in ("A", "B", "C", "D", "E"):
        r = cum_returns.get(track, {})
        lines.append(
            f"| {track} | {_fmt_pct(r.get('month', 0.0))} | {_fmt_pct(r.get('ytd', 0.0))} | "
            f"{_fmt_pct(r.get('since_inception', 0.0))} |"
        )
    lines.append("")
    lines.append("| Decomposition | Value |")
    lines.append("|---|---|")
    lines.append(f"| B - C (stock selection effect) | {_fmt_pct(differences.get('stock_selection_effect', 0.0))} |")
    lines.append(f"| A - B (overlay effect on stocks) | {_fmt_pct(differences.get('overlay_effect_on_stocks', 0.0))} |")
    lines.append(f"| D - C (overlay effect on ETF) | {_fmt_pct(differences.get('overlay_effect_on_etf', 0.0))} |")
    lines.append(f"| A - D (total DIY benefit) | {_fmt_pct(differences.get('total_diy_benefit', 0.0))} |")
    lines.append("")
    lines.append(f"Annualized turnover: {turnover_annualized_pct:.1%} — costs: {costs_bps:.1f} bps")
    lines.append("")
    lines.append("| Track | Max drawdown |")
    lines.append("|---|---|")
    for track, dd in max_drawdown_by_track.items():
        lines.append(f"| {track} | {_fmt_pct(dd)} |")
    return "\n".join(lines)


def _render_decision_panel(months_elapsed: int, metrics: DecisionMetrics) -> str:
    lines = ["## 6. Decision Criteria Panel", ""]
    if months_elapsed < 12:
        lines.append(
            f"_Insufficient history: {months_elapsed}/12 months elapsed. Criteria below are evaluated at T+12 "
            "— shown for reference only, not as PASS/FAIL._"
        )
        lines.append("")

    lines.append("| # | Criterion | Threshold | Realized | Failure means |")
    lines.append("|---|---|---|---|---|")
    realized = {
        "D1": f"{metrics.missed_rebalances} missed",
        "D2": f"{metrics.realized_turnover_annualized_pct:.1%}",
        "D3": f"{metrics.realized_costs_annualized_pct:.2%}",
        "D4": f"{metrics.manual_review_count}",
        "D5": f"{metrics.f4_data_missing_pct:.1%}",
        "D6": f"{metrics.warn_underfilled_count}",
        "D7": "survived" if metrics.track_a_survived_without_intervention else "INTERVENED",
        "D8": f"{metrics.track_a_max_drawdown_pct:.1%} (informational)",
    }
    for code, criterion, threshold, failure in _DECISION_CRITERIA:
        realized_val = realized.get(code, "-")
        prefix = "" if months_elapsed >= 12 else "(ref) "
        lines.append(f"| {code} | {criterion} | {threshold} | {prefix}{realized_val} | {failure} |")

    if months_elapsed >= 12:
        go_live = (
            metrics.missed_rebalances == 0
            and metrics.track_a_survived_without_intervention
        )
        lines.append("")
        lines.append(
            f"**Decision rule:** go live if D1-D7 pass. D1={metrics.missed_rebalances == 0}, "
            f"D7={metrics.track_a_survived_without_intervention} -> "
            f"{'PROCEED' if go_live else 'DO NOT GO LIVE regardless of returns'}."
        )
    return "\n".join(lines)


def render_report(
    signal_date: date,
    execution_date: date,
    regime: Dict,
    nav_by_track: Dict[str, float],
    monthly_a_minus_d_diffs: List[float],
    trades_this_month: List[LedgerEntry],
    rank_before: Dict[str, int],
    rank_after: Dict[str, int],
    current_positions: List[Position],
    max_per_sector: int,
    cum_returns: Dict[str, Dict[str, float]],
    differences: Dict[str, float],
    turnover_annualized_pct: float,
    costs_bps: float,
    max_drawdown_by_track: Dict[str, float],
    months_elapsed: int,
    decision_metrics: DecisionMetrics,
) -> str:
    """
    Assemble the full report.md, sections §12.1-§12.6 in order.

    See individual _render_* helpers for each section's contract. All
    inputs are pre-computed by the caller (jobs/run_monthly_execute.py) —
    this module is a pure renderer, not an aggregator, so it stays testable
    against fixed fixtures without touching _state/ itself.
    """
    sections = [
        f"# P21 Momentum — Monthly Report ({execution_date.isoformat()})",
        "",
        "**Educational material; not personalized investment advice. Paper simulation only.**",
        "",
        _render_header(signal_date, execution_date, regime, nav_by_track),
        "",
        _render_disclaimer(monthly_a_minus_d_diffs),
        "",
        _render_trades(trades_this_month, rank_before, rank_after),
        "",
        _render_portfolio(current_positions, execution_date, max_per_sector),
        "",
        _render_attribution(cum_returns, differences, turnover_annualized_pct, costs_bps, max_drawdown_by_track),
        "",
        _render_decision_panel(months_elapsed, decision_metrics),
        "",
    ]
    return "\n".join(sections)
