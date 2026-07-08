"""
Unit tests for src/simulator.py.

Validates simulation mechanics using synthetic price data:
- Total cost bookkeeping
- Correct payoff when S_expiry << K
- No payoff when S_expiry > K (OTM at expiry)
- March 2020-like scenario: put purchased before crash pays out
"""

import numpy as np
import pandas as pd

from src.ml.pipeline.p16_taleb.src.simulator import simulate_barbell


def _make_master(
    n: int = 500,
    seed: int = 99,
    vix: float = 20.0,
    rate: float = 0.04,
) -> pd.DataFrame:
    """Synthetic master DataFrame for simulator tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 3000.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1e8,
            "vix": vix,
            "rate_3m": rate,
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class TestSimulateBarbell:
    def test_returns_dataframe(self):
        df = _make_master()
        result = simulate_barbell(df, moneyness=0.85, T_days=90, rebalance_days=21)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_required_output_columns(self):
        result = simulate_barbell(_make_master())
        required = [
            "S",
            "K",
            "T",
            "moneyness",
            "sigma_used",
            "put_price",
            "budget_spent",
            "num_contracts",
            "payoff",
            "pnl",
            "cum_cost",
            "cum_payoff",
            "cum_pnl",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_total_cost_equals_budget_times_periods(self):
        """
        Each period spends exactly put_budget_pct × initial_capital.
        Total cost = n_valid_periods × budget_per_period.
        """
        budget_pct = 0.02
        capital = 100_000.0
        result = simulate_barbell(_make_master(), put_budget_pct=budget_pct, initial_capital=capital)
        expected_per_period = budget_pct * capital
        # All budget_spent values should equal the per-period budget
        assert (result["budget_spent"] - expected_per_period).abs().max() < 1e-6

    def test_payoff_zero_when_otm_at_expiry(self):
        """
        If price rises, the OTM put expires worthless (payoff = 0).
        Construct a strongly uptrending series.
        """
        n = 300
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Strong uptrend: close rises 0.1% per day
        close = 4000.0 * np.cumprod(np.full(n, 1.001))
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e8, "vix": 15.0, "rate_3m": 0.04},
            index=dates,
        )
        df.index.name = "date"
        result = simulate_barbell(df, moneyness=0.85, T_days=90, rebalance_days=21)
        # With strong uptrend, 15% OTM puts should almost never pay out
        assert (result["payoff"] == 0).sum() > 0.8 * len(result)

    def test_payoff_positive_on_crash(self):
        """
        Construct a sudden crash: price drops 30% after position is established.
        The 15% OTM put should produce a large positive payoff.
        """
        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        # Flat then crash at row 80
        close = np.full(n, 4000.0)
        close[80:] = close[80:] * 0.65  # 35% crash
        df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1e8, "vix": 20.0, "rate_3m": 0.04},
            index=dates,
        )
        df.index.name = "date"
        result = simulate_barbell(df, moneyness=0.85, T_days=90, rebalance_days=21)
        # At least one period should have positive payoff
        assert result["payoff"].max() > 0

    def test_otm_pct_matches_moneyness(self):
        """otm_pct column = (1 - moneyness) × 100."""
        for moneyness in [0.80, 0.85, 0.90]:
            result = simulate_barbell(_make_master(), moneyness=moneyness)
            expected_otm = (1.0 - moneyness) * 100.0
            assert (result["otm_pct"] - expected_otm).abs().max() < 1e-9

    def test_cum_pnl_is_cumsum_of_pnl(self):
        result = simulate_barbell(_make_master())
        expected = result["pnl"].cumsum()
        pd.testing.assert_series_equal(result["cum_pnl"], expected, check_names=False)

    def test_empty_df_returns_empty(self):
        result = simulate_barbell(pd.DataFrame())
        assert result.empty

    def test_price_source_is_bs(self):
        result = simulate_barbell(_make_master())
        assert (result["price_source"] == "bs").all()

    def test_k_equals_moneyness_times_s(self):
        """K = moneyness × S for every row."""
        result = simulate_barbell(_make_master(), moneyness=0.85)
        diff = (result["K"] - result["moneyness"] * result["S"]).abs()
        assert diff.max() < 1e-6
