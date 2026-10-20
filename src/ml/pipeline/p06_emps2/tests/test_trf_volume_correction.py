"""
Regression tests for AccumulationAnalyzer._apply_trf_volume_correction.

Under pandas 3, writing the scaled (float) volumes back into an int64 column raised
TypeError, which silently dropped every ticker that had a TRF correction factor.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p06_emps2.accumulation_analyzer import AccumulationAnalyzer
from src.ml.pipeline.p06_emps2.config import EMPS2FilterConfig


def _make_analyzer(tmp_path: Path) -> AccumulationAnalyzer:
    return AccumulationAnalyzer(data_manager=MagicMock(), config=EMPS2FilterConfig(), results_dir=tmp_path)


def _make_int_volume_df() -> pd.DataFrame:
    today = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    timestamps = [today - timedelta(days=2), today - timedelta(days=1), today]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [1.0, 1.0, 1.0],
            "high": [1.1, 1.1, 1.1],
            "low": [0.9, 0.9, 0.9],
            "close": [1.0, 1.0, 1.0],
            "volume": pd.Series([1000, 2000, 3000], dtype="int64"),
        }
    )


def test_trf_correction_scales_int64_volume_without_error(tmp_path: Path) -> None:
    analyzer = _make_analyzer(tmp_path)

    # Non-integral products are what pandas 3 rejects for an int64 column.
    result = analyzer._apply_trf_volume_correction(_make_int_volume_df(), 1.3333)

    assert result["volume"].tolist() == [1000 * 1.3333, 2000 * 1.3333, 3000.0]


def test_trf_correction_leaves_input_unchanged(tmp_path: Path) -> None:
    analyzer = _make_analyzer(tmp_path)
    df = _make_int_volume_df()

    analyzer._apply_trf_volume_correction(df, 2.0)

    assert df["volume"].tolist() == [1000, 2000, 3000]
    assert df["volume"].dtype == "int64"
