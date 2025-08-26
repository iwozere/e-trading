import numpy as np
import pandas as pd
import pytest
from src.optimizer.bb_volume_supertrend_optimizer import \
    BBSuperTrendVolumeBreakoutOptimizer
from src.optimizer.ichimoku_rsi_volume_optimizer import \
    IchimokuRSIATRVolumeOptimizer
from src.optimizer.rsi_bb_optimizer import MeanReversionRSBBATROptimizer
from src.optimizer.rsi_bb_volume_optimizer import RsiBBVolumeOptimizer
from src.optimizer.rsi_volume_supertrend_optimizer import \
    RsiVolumeSuperTrendOptimizer

"""
Unit tests for all optimizers in src/optimizer.

These tests ensure that each optimizer can be instantiated, run a minimal optimization on dummy data, and that the results contain expected keys ('metrics', 'trades').
The tests use pytest and pandas with randomly generated dummy data.
"""


@pytest.mark.parametrize(
    "optimizer_cls",
    [
        RsiVolumeSuperTrendOptimizer,
        RsiBBVolumeOptimizer,
        IchimokuRSIATRVolumeOptimizer,
        BBSuperTrendVolumeBreakoutOptimizer,
        MeanReversionRSBBATROptimizer,
    ],
)
def test_optimizer_runs_minimal(optimizer_cls, tmp_path):
    """
    Test that an optimizer can run a minimal optimization on dummy data and returns a result dict with 'metrics' and 'trades' keys (if not None).
    The test passes if no error is raised and the result structure is correct.
    """
    # Create dummy data
    idx = pd.date_range("2023-01-01", periods=50, freq="H")
    df = pd.DataFrame(
        {
            "open": np.random.rand(50) * 10 + 100,
            "high": np.random.rand(50) * 12 + 102,
            "low": np.random.rand(50) * 8 + 98,
            "close": np.random.rand(50) * 10 + 100,
            "volume": np.random.rand(50) * 1000 + 100,
        },
        index=idx,
    )
    df["high"] = np.maximum(df["high"], df["open"])
    df["high"] = np.maximum(df["high"], df["close"])
    df["low"] = np.minimum(df["low"], df["open"])
    df["low"] = np.minimum(df["low"], df["close"])
    # Save to CSV in tmp_path
    data_file = tmp_path / "DUMMYPAIR_1h_20230101_20230103.csv"
    df.to_csv(data_file)
    # Patch optimizer to use tmp_path as data_dir
    optimizer = optimizer_cls()
    optimizer.data_dir = str(tmp_path)
    optimizer.results_dir = str(tmp_path)
    optimizer.load_all_data()
    # Run a single file optimization (should not error)
    result = optimizer.optimize_single_file(data_file.name)
    assert result is None or isinstance(result, dict)
    # If result is dict, check for expected keys
    if isinstance(result, dict):
        assert "metrics" in result
        assert "trades" in result
