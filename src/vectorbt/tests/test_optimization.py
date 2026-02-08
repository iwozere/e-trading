import pytest
import pandas as pd
import numpy as np
import os
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
        study = manager.run_optimization(
            interval="1h",
            n_trials=5,
            n_jobs=1,
            study_name="test_dry_run"
        )

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
    dates = pd.date_range("2024-01-01", periods=100, freq="1H")
    # Simulate a crash: price goes from 100 to 30 (70% drawdown)
    prices = np.linspace(100, 30, 100)

    # Build MultiIndex DataFrame
    cols = pd.MultiIndex.from_tuples([('BTC', 'Open'), ('BTC', 'High'), ('BTC', 'Low'), ('BTC', 'Close'), ('BTC', 'Volume')], names=['symbol', 'column'])
    data = pd.DataFrame(index=dates, columns=cols)
    data[('BTC', 'Open')] = prices
    data[('BTC', 'High')] = prices * 1.01
    data[('BTC', 'Low')] = prices * 0.99
    data[('BTC', 'Close')] = prices
    data[('BTC', 'Volume')] = 1000

    # 2. Setup objective with this crash data
    obj = Objective(data_splits=[data])

    # 3. Create a mock trial with high leverage (which will surely cause >60% DD)
    class MockTrial:
        def __init__(self, params):
            self.params = params
            self.number = 1
            self.user_attrs = {}
        def suggest_int(self, name, low, high): return self.params[name]
        def suggest_float(self, name, low, high): return self.params[name]
        def set_user_attr(self, name, value): self.user_attrs[name] = value

    # Strategy that's long all the time in a crash
    params = {
        'rsi_window': 14,
        'rsi_lower': 100, # Always below -> long
        'rsi_upper': 0,
        'bb_window': 20,
        'bb_std': 2.0,
        'leverage': 1.0 # Even 1x with 70% drop should trigger proxy penalty
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
