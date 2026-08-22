"""Unit tests for src.ml.pipeline.p21_momentum.reporting.monthly_report."""

from __future__ import annotations

import unittest
from datetime import date
from typing import Any, Dict, Optional

from src.ml.pipeline.p21_momentum.reporting.monthly_report import (
    STATISTICAL_POWER_DISCLAIMER_KEY_PHRASE,
    DecisionMetrics,
    render_report,
)
from src.ml.pipeline.p21_momentum.schemas import LedgerEntry, Position


def _base_kwargs(months_elapsed: int = 2, monthly_diffs: Optional[list] = None) -> Dict[str, Any]:
    positions = [
        Position(
            ticker="NVDA",
            shares=14.2371,
            avg_cost=175.40,
            entry_date="2026-06-01",
            entry_rank=3,
            current_rank=7,
            sector="Information Technology",
            target_weight_pct=0.95,
            high_water_price=198.20,
        )
    ]
    trade = LedgerEntry(
        ts="2026-09-01T09:45:00-04:00",
        track="A",
        ticker="NVDA",
        side="BUY",
        shares=14.2371,
        ref_open=175.35,
        fill_price=175.40,
        slippage_bps=3.0,
        commission_usd=0.35,
        gross_usd=2496.00,
        net_usd=2496.35,
        reason="ENTRY_RANK_3",
    )
    return dict(
        signal_date=date(2026, 8, 31),
        execution_date=date(2026, 9, 1),
        regime={"scalar_applied": 1.0, "bear": False, "high_vol": False, "months_at_state": 3},
        nav_by_track={"A": 251000.0, "B": 250500.0, "C": 250200.0, "D": 250800.0, "E": 251500.0},
        monthly_a_minus_d_diffs=monthly_diffs if monthly_diffs is not None else [0.001, -0.002],
        trades_this_month=[trade],
        rank_before={"NVDA": 3},
        rank_after={"NVDA": 3},
        current_positions=positions,
        max_per_sector=4,
        cum_returns={
            "A": {"month": 0.004, "ytd": 0.03, "since_inception": 0.04},
            "B": {"month": 0.002, "ytd": 0.02, "since_inception": 0.03},
            "C": {"month": 0.001, "ytd": 0.01, "since_inception": 0.008},
            "D": {"month": 0.0015, "ytd": 0.012, "since_inception": 0.009},
            "E": {"month": 0.003, "ytd": 0.025, "since_inception": 0.035},
        },
        differences={
            "stock_selection_effect": 0.02,
            "overlay_effect_on_stocks": 0.01,
            "overlay_effect_on_etf": 0.003,
            "total_diy_benefit": 0.031,
        },
        turnover_annualized_pct=1.75,
        costs_bps=15.0,
        max_drawdown_by_track={"A": -0.12, "B": -0.15, "C": -0.10, "D": -0.09, "E": -0.20},
        months_elapsed=months_elapsed,
        decision_metrics=DecisionMetrics(),
    )


class TestRenderReport(unittest.TestCase):
    def test_disclaimer_present_verbatim(self):
        md = render_report(**_base_kwargs())
        self.assertIn(STATISTICAL_POWER_DISCLAIMER_KEY_PHRASE, md)

    def test_insufficient_history_note_before_t12(self):
        md = render_report(**_base_kwargs(months_elapsed=3))
        self.assertIn("Insufficient history", md)

    def test_full_decision_table_at_t12(self):
        md = render_report(**_base_kwargs(months_elapsed=12, monthly_diffs=[0.001] * 12))
        self.assertNotIn("Insufficient history", md)
        self.assertIn("Decision rule", md)

    def test_trades_section_lists_trade(self):
        md = render_report(**_base_kwargs())
        self.assertIn("NVDA", md)
        self.assertIn("ENTRY_RANK_3", md)

    def test_no_trades_renders_placeholder(self):
        kwargs = _base_kwargs()
        kwargs["trades_this_month"] = []
        md = render_report(**kwargs)
        self.assertIn("No trades this month", md)

    def test_sector_cap_breach_flagged(self):
        kwargs = _base_kwargs()
        kwargs["current_positions"] = [
            Position(f"T{i}", 1.0, 10.0, "2026-06-01", 1, 1, "Tech", 0.05, 10.0) for i in range(5)
        ]
        kwargs["max_per_sector"] = 4
        md = render_report(**kwargs)
        self.assertIn("BREACH", md)

    def test_t_statistic_insufficient_history_with_single_month(self):
        md = render_report(**_base_kwargs(monthly_diffs=[0.001]))
        self.assertIn("insufficient history", md)


if __name__ == "__main__":
    unittest.main()
