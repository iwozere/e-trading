"""Integration test for src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.

Exercises the full orchestration (wiring order, gate evaluation, target
writing) against synthetic in-memory data — no network, no real filesystem
writes under results/p21_momentum/. Individual strategy/quality modules
already have their own thorough unit tests; this test's job is to catch
wiring bugs (wrong argument order, wrong gate call, etc.), not to
re-verify strategy math.
"""

from __future__ import annotations

import unittest
from collections import Counter
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml.pipeline.p21_momentum.config import MIN_CONSTITUENTS
from src.ml.pipeline.p21_momentum.data.universe import UniverseConstituent
from src.ml.pipeline.p21_momentum.jobs import run_monthly_rebalance as job

_SECTORS = [f"Sector{i}" for i in range(15)]  # 15 sectors * 30 tickers = 450, >= MIN_CONSTITUENTS


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


class TestRunMonthlyRebalanceGuards(unittest.TestCase):
    def test_skips_when_not_last_trading_day(self):
        result = job.run(run_date=date(2026, 8, 3))  # arbitrary mid-month Monday
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "not_last_trading_day")

    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.already_processed", return_value=True)
    def test_skips_when_already_processed(self, _mock):
        del _mock  # unused: injected by @patch, only its return_value matters
        # 2026-08-31 is the last trading day of August 2026
        result = job.run(run_date=date(2026, 8, 31))
        self.assertTrue(result.get("skipped"))
        self.assertEqual(result.get("reason"), "already_processed")


class TestRunMonthlyRebalanceHappyPath(unittest.TestCase):
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.already_processed", return_value=False)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.write_targets")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.write_universe")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.next_earnings_date", return_value=None)
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.fetch_fundamentals_cached")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.fetch_price_panel")
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.read_current_positions", return_value=[])
    @patch("src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.fetch_universe")
    def test_full_cycle_produces_targets(
        self,
        mock_fetch_universe,
        _mock_read_positions,
        mock_fetch_panel,
        mock_fetch_fund,
        _mock_next_earnings,
        _mock_write_universe,
        mock_write_targets,
        _mock_already_processed,
    ):
        del _mock_read_positions, _mock_next_earnings, _mock_write_universe, _mock_already_processed
        constituents = _make_universe()
        mock_fetch_universe.return_value = constituents

        panel = {c.ticker: _make_ohlcv_df(seed=hash(c.ticker) % 1000) for c in constituents}
        panel["MTUM"] = _make_ohlcv_df(seed=9001)
        panel["SPY"] = _make_ohlcv_df(seed=9002)
        panel["^GSPC"] = _make_ohlcv_df(seed=9003, drift=0.0008, vol=0.01)
        panel["^VIX"] = _make_ohlcv_df(seed=9004, drift=0.0, vol=0.05)
        panel["^VIX"]["close"] = 15.0  # flat low VIX -> normal regime
        mock_fetch_panel.return_value = panel

        mock_fetch_fund.return_value = {c.ticker: {"fcf_ttm": 100.0, "net_income_ttm": 50.0} for c in constituents}

        with TemporaryDirectory() as tmp:
            regime_history_path = Path(tmp) / "regime_history.json"
            with patch(
                "src.ml.pipeline.p21_momentum.jobs.run_monthly_rebalance.REGIME_HISTORY_PATH", regime_history_path
            ):
                result = job.run(run_date=date(2026, 8, 31))

        self.assertFalse(result.get("skipped"))
        self.assertFalse(result.get("aborted"))
        self.assertEqual(result["targets_count"], 20)
        self.assertIn("regime_scalar", result)
        mock_write_targets.assert_called_once()
        written_targets = mock_write_targets.call_args[0][1]
        self.assertEqual(len(written_targets), 20)
        # Sector cap respected in the final target list
        counts = Counter(t.sector for t in written_targets)
        for count in counts.values():
            self.assertLessEqual(count, 4)


if __name__ == "__main__":
    unittest.main()
