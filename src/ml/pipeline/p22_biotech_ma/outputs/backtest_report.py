"""
P22 — the weekly backtest email (2026-11-09).

Lift of the learned ranking against the §0.3 baselines and the Phase 1 composite, over the
snapshots the learned model ranked (`metrics_learned_window`, the like-for-like table), with the
change since the previous weekly run and per-block attribution, and (2026-11-10) the portfolio
returns of each ranking's top k held 24 months or to the deal (`portfolio_learned_window`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.outputs.email_notify import html_table
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

MAIN_RANKINGS = ("learned", "naive_informed", "catalyst_proximity", "model")
BLOCK_PREFIX = "block:"
REPORT_KS = (25, 50, 200)
HEADLINE_K = 50
PORTFOLIO_HEADLINE_K = 25


def portfolio_cell(summary: Mapping[str, Any], ranking: str, k: Any, key: str) -> Optional[float]:
    """One value of the learned-window portfolio table (`k` is "all" for the universe row)."""
    for row in summary.get("portfolio_learned_window") or []:
        if row.get("ranking") == ranking and str(row.get("k")) == str(k):
            return row.get(key)
    return None


def lifts(metric_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[int, Optional[float]]]:
    """`{ranking: {k: lift}}` from a metrics table's rows."""
    out: Dict[str, Dict[int, Optional[float]]] = {}
    for row in metric_rows:
        out.setdefault(str(row["ranking"]), {})[int(row["k"])] = row.get("lift")
    return out


def previous_summary(results_root: Path, current_dir: Path) -> Optional[Dict[str, Any]]:
    """The newest `YYYY-MM-DD/backtest/summary.json` under `results_root` from before `current_dir`."""
    earlier = sorted(p for p in results_root.glob("*/backtest/summary.json") if p.parent.parent.name < current_dir.name)
    if not earlier:
        return None
    try:
        return json.loads(earlier[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _logger.warning("Cannot read previous backtest summary %s", earlier[-1])
        return None


def _fmt(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _pct(value: Optional[float], signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%"


def _portfolio_rows(summary: Mapping[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for r in summary.get("portfolio_learned_window") or []:
        rows.append([r.get("ranking"), r.get("k"), _pct(r.get("mean_return")), _pct(r.get("excess"), signed=True),
                     _pct(r.get("stressed_excess"), signed=True), _pct(r.get("beat_share")), _pct(r.get("deal_share")),
                     _pct(r.get("price_ended_share"))])
    return rows


def _delta(now: Optional[float], before: Optional[float]) -> Optional[float]:
    return now - before if now is not None and before is not None else None


def _ranking_rows(current: Mapping[str, Mapping[int, Optional[float]]], before: Mapping[str, Mapping[int, Optional[float]]],
                  names: Sequence[str]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for name in names:
        values = current.get(name, {})
        rows.append([name, *(_fmt(values.get(k)) for k in REPORT_KS),
                     _fmt(_delta(values.get(HEADLINE_K), before.get(name, {}).get(HEADLINE_K)), signed=True)])
    return rows


def build_backtest_report(summary: Mapping[str, Any], previous: Optional[Mapping[str, Any]],
                          model_info: Mapping[str, Any]) -> Tuple[str, str, str]:
    """
    `(title, plain text, html)` for the weekly email.

    Args:
        summary: This run's `summary.json` content (`backtest_cli.write_results`).
        previous: The previous run's summary, or None.
        model_info: What happened to the live model (`saved`, `train_rows`, `train_positives`, `path`).
    """
    current = lifts(summary.get("metrics_learned_window") or [])
    before = lifts((previous or {}).get("metrics_learned_window") or [])
    learned50 = current.get("learned", {}).get(HEADLINE_K)
    naive50 = current.get("naive_informed", {}).get(HEADLINE_K)
    beats = learned50 is not None and naive50 is not None and learned50 > naive50
    verdict = "beats naive" if beats else "BELOW naive"
    title = f"P22 weekly backtest: learned lift@{HEADLINE_K} {_fmt(learned50)} vs naive {_fmt(naive50)} ({verdict})"

    blocks = sorted(name for name in current if name.startswith(BLOCK_PREFIX))
    headers = ["ranking", *(f"lift@{k}" for k in REPORT_KS), f"Δ@{HEADLINE_K} vs last run"]
    facts: List[List[Any]] = [
        ["labels complete through", summary.get("labels_complete_through")],
        ["snapshots (all / learned window)", f"{summary.get('snapshots')} / {_window_snapshots(summary)}"],
        ["deals entered", summary.get("deals_entered")],
        ["previous run", (previous or {}).get("labels_complete_through", "none") if previous else "none"],
        ["live model", _model_line(model_info)],
    ]
    body = (
        f"<h3>{title}</h3>"
        + html_table(["", ""], facts)
        + "<h4>Like-for-like: snapshots ranked by the learned model</h4>"
        + html_table(headers, _ranking_rows(current, before, MAIN_RANKINGS))
        + "<h4>Each composite block alone (attribution)</h4>"
        + html_table(headers, _ranking_rows(current, before, blocks))
        + "<h4>Portfolio: top k bought at each snapshot, held 24 months or to the deal</h4>"
        + html_table(["ranking", "k", "mean return", "vs universe", "vs universe, stressed", "beat universe",
                      "exited by deal", "price ended"], _portfolio_rows(summary))
        + "<p>Equal weight, total holding-period return averaged over snapshots (they overlap). Stressed: every "
          "series that stopped trading without a deal counted at -100% instead of its last close.</p>"
        + "<p>Lift = precision in the top k divided by the base rate (1.00 = random). "
          "Full tables: <code>results/p22_biotech_ma/&lt;date&gt;/backtest/</code>.</p>"
    )
    text_lines = [title, ""] + [f"{name:24s} " + "  ".join(f"@{k} {_fmt(current.get(name, {}).get(k))}" for k in REPORT_KS)
                                for name in MAIN_RANKINGS]
    learned_excess = portfolio_cell(summary, "learned", PORTFOLIO_HEADLINE_K, "excess")
    text_lines += ["", f"Portfolio top {PORTFOLIO_HEADLINE_K}, learned: {_pct(portfolio_cell(summary, 'learned', PORTFOLIO_HEADLINE_K, 'mean_return'))} "
                       f"({_pct(learned_excess, signed=True)} vs universe)"]
    text_lines += ["", f"Labels complete through {summary.get('labels_complete_through')}; live model: {_model_line(model_info)}"]
    return title, "\n".join(text_lines), body


def _window_snapshots(summary: Mapping[str, Any]) -> Any:
    for row in summary.get("metrics_learned_window") or []:
        if row.get("ranking") == "learned":
            return row.get("snapshots")
    return 0


def _model_line(model_info: Mapping[str, Any]) -> str:
    if not model_info.get("saved"):
        return "not retrained (too few positives); scoring keeps the previous model"
    return f"retrained on {model_info.get('train_rows')} rows, {model_info.get('train_positives')} positives"
