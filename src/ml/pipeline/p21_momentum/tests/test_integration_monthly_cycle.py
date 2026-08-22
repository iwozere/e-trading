"""
Full-cycle integration test: monthly_rebalance -> monthly_execute -> daily_mark.

Runs against synthetic, deterministic in-memory data with REAL file I/O
into a tmp directory (results_dir/state_dir threaded through each job's
run(), per docs/implementation-plan.md's testability design) — not mocked
writers. This is the one place in the test suite that exercises the actual
on-disk round trip across all three jobs in sequence, and it includes the
§14.9 B10 determinism check: two identical monthly_rebalance runs against
the same inputs must produce byte-identical targets.json.
"""

from __future__ import annotations

import json
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml.pipeline.p21_momentum.config import MIN_CONSTITUENTS
from src.ml.pipeline.p21_momentum.data.universe import UniverseConstituent
from src.ml.pipeline.p21_momentum.jobs import run_daily_mark, run_monthly_execute, run_monthly_rebalance
from src.ml.pipeline.p21_momentum.schemas import TargetPosition

_SECTORS = [f"Sector{i}" for i in range(15)]


def _make_ohlcv_df(n_days: int = 400, seed: int = 0, drift: float = 0.001, vol: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(loc=drift, scale=vol, size=n_days)
    close = 100.0 * np.cumprod(1 + rets)
    idx = pd.bdate_range(end=pd.Timestamp("2026-08-31"), periods=n_days)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [2_000_000] * n_days,
        }
    )


def _make_universe(n: int = MIN_CONSTITUENTS) -> list:
    return [UniverseConstituent(ticker=f"T{i:04d}", sector=_SECTORS[i % len(_SECTORS)]) for i in range(n)]


class TestIntegrationMonthlyCycle(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.already_processed", return_value=False)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.write_universe")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.next_earnings_date", return_value=None)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.fetch_fundamentals_cached")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.fetch_price_panel")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.read_current_positions", return_value=[])
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.fetch_universe")
    def test_full_cycle_and_rebalance_determinism(
        self,
        mock_fetch_universe,
        _mock_read_positions,
        mock_fetch_panel_rebalance,
        mock_fetch_fund,
        _mock_next_earnings,
        _mock_write_universe,
        _mock_already_processed,
    ):
        del _mock_read_positions, _mock_next_earnings, _mock_write_universe, _mock_already_processed
        constituents = _make_universe()
        mock_fetch_universe.return_value = constituents

        panel = {c.ticker: _make_ohlcv_df(seed=hash(c.ticker) % 1000) for c in constituents}
        panel["MTUM"] = _make_ohlcv_df(seed=9001)
        panel["SPY"] = _make_ohlcv_df(seed=9002)
        panel["^GSPC"] = _make_ohlcv_df(seed=9003, drift=0.0008, vol=0.01)
        panel["^VIX"] = _make_ohlcv_df(seed=9004, drift=0.0, vol=0.01)
        panel["^VIX"]["close"] = 15.0
        mock_fetch_panel_rebalance.return_value = panel
        mock_fetch_fund.return_value = {c.ticker: {"fcf_ttm": 100.0, "net_income_ttm": 50.0} for c in constituents}

        with TemporaryDirectory() as tmp:
            results_dir = Path(tmp) / "results"
            state_dir = results_dir / "_state"
            regime_history_path = state_dir / "regime_history.json"
            positions_path = state_dir / "current_positions.json"
            exclusions_path = Path(tmp) / "exclusions.json"
            exclusions_path.write_text('{"exclusions": []}', encoding="utf-8")

            signal_date = date(2026, 8, 31)
            result1 = run_monthly_rebalance.run(
                run_date=signal_date,
                results_dir=results_dir,
                regime_history_path=regime_history_path,
                current_positions_path=positions_path,
                exclusions_path=exclusions_path,
            )
            self.assertFalse(result1.get("skipped"))
            self.assertFalse(result1.get("aborted"))
            self.assertEqual(result1["targets_count"], 20)

            targets_path = results_dir / "2026-08-31" / "targets.json"
            self.assertTrue(targets_path.exists())
            first_run_bytes = targets_path.read_bytes()

            # --- Determinism check (spec §14.9 B10): rerun with --force, same inputs ---
            result2 = run_monthly_rebalance.run(
                run_date=signal_date,
                force=True,
                results_dir=results_dir,
                regime_history_path=regime_history_path,
                current_positions_path=positions_path,
                exclusions_path=exclusions_path,
            )
            self.assertFalse(result2.get("aborted"))
            second_run_bytes = targets_path.read_bytes()
            self.assertEqual(
                first_run_bytes, second_run_bytes, "monthly_rebalance is not deterministic (spec §14.9 B10)"
            )

            # --- monthly_execute against the targets just written ---
            execution_date = date(2026, 9, 1)
            open_panel = {t.ticker: _make_ohlcv_df(seed=1, n_days=5) for t in _read_targets(targets_path)}
            open_panel["MTUM"] = _make_ohlcv_df(seed=2, n_days=5)
            open_panel["SPY"] = _make_ohlcv_df(seed=3, n_days=5)
            for df in open_panel.values():
                df["timestamp"] = pd.bdate_range(end=pd.Timestamp(execution_date), periods=len(df))

            with patch(
                "src.ml.pipeline.p21_momentum.jobs.run_monthly_execute.fetch_price_panel", return_value=open_panel
            ):
                exec_result = run_monthly_execute.run(
                    run_date=execution_date,
                    results_dir=results_dir,
                    state_dir=state_dir,
                    current_positions_path=positions_path,
                )

            self.assertFalse(exec_result.get("skipped"), exec_result)
            self.assertFalse(exec_result.get("aborted"), exec_result)
            self.assertGreater(exec_result["trades_count"], 0)
            self.assertTrue((results_dir / "2026-09-01" / "positions.json").exists())
            self.assertTrue((results_dir / "2026-09-01" / "report.md").exists())
            self.assertTrue((state_dir / "ledger.jsonl").exists())
            self.assertTrue(positions_path.exists())

            # --- daily_mark against the positions just opened ---
            daily_panel = dict(open_panel)
            with patch(
                "src.ml.pipeline.p21_momentum.jobs.run_daily_mark.fetch_price_panel", return_value=daily_panel
            ):
                mark_result = run_daily_mark.run(
                    run_date=execution_date,
                    results_dir=results_dir,
                    state_dir=state_dir,
                    current_positions_path=positions_path,
                )

            self.assertFalse(mark_result.get("skipped"), mark_result)
            self.assertFalse(mark_result.get("aborted"), mark_result)
            self.assertTrue((results_dir / "2026-09-01" / "daily_mark.json").exists())
            self.assertTrue((state_dir / "nav_daily.csv").exists())


def _read_targets(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [TargetPosition.from_dict(t) for t in payload["targets"]]


if __name__ == "__main__":
    unittest.main()
