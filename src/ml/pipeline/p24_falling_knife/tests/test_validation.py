"""Tests for P24 statistics and the gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.pipeline.p24_falling_knife.config import CostModel
from src.ml.pipeline.p24_falling_knife.validation import (
    bootstrap_ci,
    evaluate_gate,
    paired_difference_ci,
    random_control_p_value,
    reprice_costs,
    trade_metrics,
    without_best_month,
)


def _trades(returns: list[float], months: list[str], price: float = 20.0) -> pd.DataFrame:
    gross = np.asarray(returns)
    return pd.DataFrame({
        "signal_date": pd.to_datetime([f"{m}-15" for m in months]),
        "net_return": gross,
        "gross_return": gross,
        "entry_price_raw": price,
        "exit_price_raw": price * (1 + gross),
        "bars_held": 10,
    })


def test_trade_metrics_basic() -> None:
    m = trade_metrics(_trades([0.1, -0.05, 0.2, -0.6], ["2022-01"] * 4))
    assert m["count"] == 4 and m["win_rate"] == 0.5
    assert m["mean"] == pytest.approx(-0.0875)
    assert m["profit_factor"] == pytest.approx(0.3 / 0.65)
    assert m["share_le_minus_50"] == 0.25
    assert trade_metrics(pd.DataFrame({"net_return": []}))["count"] == 0


def test_bootstrap_ci_brackets_estimate_and_uses_months() -> None:
    rng = np.random.default_rng(0)
    months = [f"2022-{m:02d}" for m in range(1, 13)] * 10
    trades = _trades(list(rng.normal(0.05, 0.1, 120)), months)
    ci = bootstrap_ci(trades)
    assert ci["lo"] < ci["estimate"] < ci["hi"]
    assert ci["months"] == 12


def test_paired_difference_detects_shift() -> None:
    months = [f"2022-{m:02d}" for m in range(1, 13)] * 5
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.05, 60)
    diff = paired_difference_ci(_trades(list(base + 0.10), months), _trades(list(base), months))
    assert diff["estimate"] == pytest.approx(0.10)
    assert diff["lo"] > 0


def test_reprice_costs_matches_one_x_and_hurts_at_two_x() -> None:
    trades = _trades([0.1, -0.1], ["2022-01", "2022-02"], price=3.0)
    one = reprice_costs(trades, CostModel())
    expected = 1.1 * (1 - CostModel().per_side(3.3)) / (1 + CostModel().per_side(3.0)) - 1
    assert one.iloc[0] == pytest.approx(expected)
    assert (reprice_costs(trades, CostModel(multiplier=2.0)) < one).all()


def test_without_best_month() -> None:
    trades = _trades([0.5, 0.4, 0.1, -0.1], ["2022-03", "2022-03", "2022-04", "2022-05"])
    rest = without_best_month(trades)
    assert len(rest) == 2 and "2022-03" not in rest["signal_date"].dt.strftime("%Y-%m").tolist()


def test_random_control_p_value() -> None:
    assert random_control_p_value(pd.Series([0.0, 0.01, 0.05, 0.10]), 0.06) == 0.25
    assert np.isnan(random_control_p_value(pd.Series([], dtype=float), 0.1))


def test_gate_passes_a_clear_edge_and_fails_a_loser() -> None:
    months = [f"20{y}-{m:02d}" for y in (24, 25) for m in range(1, 13)] * 4
    rng = np.random.default_rng(2)
    good = _trades(list(rng.normal(0.08, 0.05, len(months))), months)
    control = _trades(list(rng.normal(0.0, 0.05, len(months))), months)
    results = {g.name: g.passed for g in evaluate_gate(good, control, CostModel())}
    assert results["trade_count"] and results["positive_returns"] and results["beats_drop_only"]
    assert results["survives_2x_costs"] and results["not_one_month"]
    bad = _trades(list(rng.normal(-0.05, 0.05, len(months))), months)
    assert not {g.name: g.passed for g in evaluate_gate(bad, control, CostModel())}["positive_returns"]
