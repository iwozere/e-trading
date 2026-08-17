import os

import numpy as np
import pandas as pd
import pytest

from src.vectorbt.pipeline.manager import StudyManager
from src.vectorbt.pipeline.objective import Objective


def test_optimization_dry_run():
    # 1. Setup StudyManager with a test DB
    test_db = "src/vectorbt/db/test_optimization.db"
    if os.path.exists(test_db):
        os.remove(test_db)

    manager = StudyManager(db_path=test_db)

    # 2. Run a minimal study (requires data in data/ folder or mocked data split)
    # If no data exists, we skip
    if not os.path.exists("data"):
        pytest.skip("Test data directory 'data' not found")

    try:
        study = manager.run_optimization(interval="1h", n_trials=5, n_jobs=1, study_name="test_dry_run")

        assert study is not None
        assert len(study.trials) >= 5
        assert study.best_trial is not None

        print("✅ Optimization dry-run test passed")
    except Exception as e:
        if "No files found" in str(e):
            pytest.skip(f"No CSV files found for pattern: {e}")
        else:
            raise e
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


def test_liquidation_penalty():
    # This test verifies that the objective function correctly penalizes
    # trials with excessive drawdowns (>60%)

    # 1. Create synthetic data split (MultiIndex)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")  # pandas 3.0 dropped the deprecated "H" alias
    # Simulate a crash: price goes from 100 to 30 (70% drawdown)
    prices = np.linspace(100, 30, 100)

    # Build MultiIndex DataFrame
    cols = pd.MultiIndex.from_tuples(
        [("BTC", "Open"), ("BTC", "High"), ("BTC", "Low"), ("BTC", "Close"), ("BTC", "Volume")],
        names=["symbol", "column"],
    )
    data = pd.DataFrame(index=dates, columns=cols)
    data[("BTC", "Open")] = prices  # type: ignore
    data[("BTC", "High")] = prices * 1.01  # type: ignore
    data[("BTC", "Low")] = prices * 0.99  # type: ignore
    data[("BTC", "Close")] = prices  # type: ignore
    data[("BTC", "Volume")] = 1000  # type: ignore

    # 2. Setup objective with this crash data.
    # A minimal single-indicator strategy: enter long whenever RSI < lower (rsi_main_lower=100
    # below makes this always true post-warmup, so the portfolio stays long through the whole
    # crash) and never exit (long_exit target of 999 is unreachable on a 0-100 RSI scale) --
    # this is what actually realizes the drawdown pf.max_drawdown() is asserted on below.
    # An empty strategy_config here (as this test previously had) skips indicator/logic
    # evaluation entirely, so the engine never enters a position at all: max_drawdown is then
    # trivially 0 on a flat portfolio, regardless of how severe the underlying price crash is.
    strategy_config = {
        "indicators": {
            "rsi_main": {
                "type": "RSI",
                "space": {
                    "window": {"type": "int", "min": 2, "max": 30},
                    "lower": {"type": "int", "min": 1, "max": 100},
                    "upper": {"type": "int", "min": 1, "max": 100},
                },
            },
        },
        "logic": {
            "long_entry": {"indicator": "rsi_main", "field": "rsi", "op": "<", "target": "rsi_main.lower"},
            "long_exit": {"indicator": "rsi_main", "field": "rsi", "op": ">", "target": 999},
        },
    }
    obj = Objective(data_splits=[data], strategy_config=strategy_config)

    # 3. Create a mock trial with high leverage (which will surely cause >60% DD)
    class MockTrial:
        def __init__(self, params):
            self.params = params
            self.number = 1
            self.user_attrs = {}

        def suggest_int(self, name, low, high, step=1):
            return self.params[name]

        def suggest_float(self, name, low, high, step=None):
            return self.params[name]

        def set_user_attr(self, name, value):
            self.user_attrs[name] = value

    # Strategy that's long all the time in a crash
    params = {
        "rsi_main_window": 14,
        "rsi_main_lower": 100,  # RSI always below this post-warmup -> always long
        "rsi_main_upper": 1,  # unused by long_entry/long_exit logic above, but suggest_int still needs a value
        "leverage": 1.0,  # Even 1x with 70% drop should trigger proxy penalty
    }

    trial = MockTrial(params)

    # 4. Run objective
    score = obj(trial)

    # 5. Verify penalty (score should be very low, e.g., -1e6 or similar)
    assert score < -100000
    assert trial.user_attrs.get("avg_max_drawdown", 0) > 0.6

    print("✅ Liquidation penalty logic verified")


if __name__ == "__main__":
    test_liquidation_penalty()
