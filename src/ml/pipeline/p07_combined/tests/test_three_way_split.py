"""
Verifies that the OOS test set is never passed to Optuna.
The objective function must use pf_val exclusively.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ml.pipeline.p07_combined import optimize as p07_optimize
from src.ml.pipeline.p07_combined.evaluator import P07Evaluator


def _make_ohlcv(n: int = 600) -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2022-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.ones(n) * 5000,
        },
        index=dates,
    )


def test_objective_accesses_pf_val_not_pf_test():
    """
    The Optuna objective must call .sharpe_ratio() on pf_val, not pf_test.
    We mock run_evaluation to return a result dict and verify which key is used.
    """
    ohlcv = _make_ohlcv()

    mock_pf_val = type(
        "PF",
        (),
        {
            "sharpe_ratio": lambda self: 1.2,
            "trades": type("T", (), {"count": lambda self: type("C", (), {"sum": lambda self: 50})()})(),
        },
    )()
    mock_pf_test = type(
        "PF",
        (),
        {
            "sharpe_ratio": lambda self: 9.9,
            "trades": type("T", (), {"count": lambda self: type("C", (), {"sum": lambda self: 5})()})(),
        },
    )()

    fake_result = {
        "pf_val": mock_pf_val,
        "pf_test": mock_pf_test,
        "pf": mock_pf_test,
    }

    with patch.object(P07Evaluator, "run_evaluation", return_value=fake_result) as mock_eval:
        trial = type(
            "Trial",
            (),
            {
                "suggest_int": lambda self, *a, **kw: 14,
                "suggest_float": lambda self, *a, **kw: 0.5,
            },
        )()
        score = p07_optimize.objective(trial, ohlcv, timeframe="15m")

    # Score must be based on pf_val (sharpe=1.2), not pf_test (sharpe=9.9)
    assert score < 5.0, (
        f"Objective score {score} looks like it came from pf_test (sharpe=9.9) instead of pf_val (sharpe=1.2)"
    )


def test_objective_returns_negative_on_error():
    """Objective must return -1.0 when run_evaluation returns an error."""
    ohlcv = _make_ohlcv()

    with patch.object(P07Evaluator, "run_evaluation", return_value={"error": "test error"}):
        trial = type(
            "Trial",
            (),
            {
                "suggest_int": lambda self, *a, **kw: 14,
                "suggest_float": lambda self, *a, **kw: 0.5,
            },
        )()
        score = p07_optimize.objective(trial, ohlcv, timeframe="15m")

    assert score == -1.0
