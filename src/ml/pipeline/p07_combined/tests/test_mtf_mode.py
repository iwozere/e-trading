"""
Tests for Phase 3 — P07/P08 merge and MTF mode.

Verifies:
1. merge_mtf() is look-ahead safe: each execution bar only sees anchor bars
   whose close was fully committed *before* it opened (1-bar shift + merge_asof).
2. P07Pipeline(enable_mtf=True) uses get_mtf_dataset() instead of get_merged_dataset().
3. build_features(enable_mtf=True) adds anchor columns; enable_mtf=False does not.
4. P08Pipeline emits DeprecationWarning and delegates run_batch() to P07Pipeline.
"""

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + rng.standard_normal(n).cumsum()
    df = pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "volume": rng.integers(1000, 5000, size=n).astype(float),
    }, index=idx)
    return df


def _make_anchor(n: int = 30, freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(n).cumsum()
    df = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": rng.integers(2000, 8000, size=n).astype(float),
    }, index=idx)
    return df


# ---------------------------------------------------------------------------
# 1. merge_mtf() look-ahead safety
# ---------------------------------------------------------------------------

class TestMergeMtfLookaheadSafety:
    """merge_mtf must never let an execution bar see its own anchor bar's close."""

    def test_anchor_columns_are_present_after_merge(self):
        from src.ml.pipeline.p07_combined.data_loader import P07DataLoader
        loader = P07DataLoader()
        df_exec = _make_ohlcv(100, "15min")
        df_anchor = _make_anchor(30, "1h")

        merged = loader.merge_mtf(df_exec, df_anchor)

        assert "anchor_close" in merged.columns
        assert "anchor_volume" in merged.columns

    def test_first_execution_bar_has_nan_anchor(self):
        """Because anchor is shifted by 1 bar, the very first row must be NaN."""
        from src.ml.pipeline.p07_combined.data_loader import P07DataLoader
        loader = P07DataLoader()
        df_exec = _make_ohlcv(100, "15min")
        df_anchor = _make_anchor(30, "1h")

        merged = loader.merge_mtf(df_exec, df_anchor)

        # After 1-bar shift the first anchor row becomes NaN
        assert pd.isna(merged["anchor_close"].iloc[0]), (
            "First execution bar should have NaN anchor_close (look-ahead safe)"
        )

    def test_anchor_value_at_exec_bar_predates_that_bar(self):
        """
        For any execution bar at time T, the joined anchor_close must come from
        an anchor bar whose timestamp strictly < T (after 1-bar shift).
        """
        from src.ml.pipeline.p07_combined.data_loader import P07DataLoader
        loader = P07DataLoader()
        df_exec = _make_ohlcv(120, "15min")
        df_anchor = _make_anchor(40, "1h")

        merged = loader.merge_mtf(df_exec, df_anchor)
        merged_clean = merged.dropna(subset=["anchor_close"])

        # anchor_close values must come from *shifted* anchor: each value seen at
        # exec time T was the close of an anchor bar that *closed before* T.
        # We verify by checking that merged anchor_close values are in the
        # shifted anchor series (i.e., from df_anchor shifted by 1).
        shifted_anchor_closes = set(df_anchor["close"].shift(1).dropna().round(6))
        observed_closes = set(merged_clean["anchor_close"].round(6))

        # All observed anchor closes must be in the shifted set
        assert observed_closes.issubset(shifted_anchor_closes), (
            "Execution bars are seeing unshifted anchor closes (look-ahead leak!)"
        )


# ---------------------------------------------------------------------------
# 2. P07Pipeline uses get_mtf_dataset when enable_mtf=True
# ---------------------------------------------------------------------------

class TestP07PipelineMtfFlag:

    def test_load_dataset_calls_get_mtf_dataset_when_enabled(self):
        from src.ml.pipeline.p07_combined.pipeline import P07Pipeline

        with patch.object(P07Pipeline, "__init__", return_value=None):
            p = P07Pipeline.__new__(P07Pipeline)
            p.enable_mtf = True
            p.data_loader = MagicMock()
            p.data_loader.get_mtf_dataset.return_value = _make_ohlcv(50)
            p.data_loader.get_merged_dataset.return_value = _make_ohlcv(50)

        fake_path = Path("BTC_15m_20230101_20231231.csv")
        p._load_dataset(fake_path)

        p.data_loader.get_mtf_dataset.assert_called_once_with(fake_path)
        p.data_loader.get_merged_dataset.assert_not_called()

    def test_load_dataset_calls_get_merged_dataset_when_disabled(self):
        from src.ml.pipeline.p07_combined.pipeline import P07Pipeline

        with patch.object(P07Pipeline, "__init__", return_value=None):
            p = P07Pipeline.__new__(P07Pipeline)
            p.enable_mtf = False
            p.data_loader = MagicMock()
            p.data_loader.get_merged_dataset.return_value = _make_ohlcv(50)
            p.data_loader.get_mtf_dataset.return_value = _make_ohlcv(50)

        fake_path = Path("BTC_15m_20230101_20231231.csv")
        p._load_dataset(fake_path)

        p.data_loader.get_merged_dataset.assert_called_once_with(fake_path)
        p.data_loader.get_mtf_dataset.assert_not_called()


# ---------------------------------------------------------------------------
# 3. build_features enable_mtf flag
# ---------------------------------------------------------------------------

class TestBuildFeaturesEnableMtf:

    def _base_df(self, n: int = 80) -> pd.DataFrame:
        return _make_ohlcv(n)

    def test_mtf_disabled_produces_no_anchor_columns(self):
        from src.ml.pipeline.p07_combined.features import build_features
        df = self._base_df()
        X = build_features(df, {}, enable_mtf=False)

        anchor_cols = [c for c in X.columns if c.startswith("anchor_")]
        assert anchor_cols == [], f"Unexpected anchor columns with enable_mtf=False: {anchor_cols}"

    def test_mtf_enabled_without_anchor_data_uses_placeholders(self):
        """When anchor columns are absent, MTF adds placeholder zeros."""
        from src.ml.pipeline.p07_combined.features import build_features
        df = self._base_df()
        X = build_features(df, {}, enable_mtf=True)

        assert "anchor_trend" in X.columns
        assert "anchor_regime" in X.columns
        assert "mtf_divergence" in X.columns
        # placeholder values
        assert (X["anchor_trend"] == 0.0).all()
        assert (X["anchor_regime"] == 0).all()

    def test_mtf_enabled_with_anchor_data_computes_real_features(self):
        """When anchor columns exist, MTF features are non-trivial."""
        from src.ml.pipeline.p07_combined.features import build_features
        from src.ml.pipeline.p07_combined.data_loader import P07DataLoader

        df_exec = _make_ohlcv(120, "15min")
        df_anchor = _make_anchor(40, "1h")
        loader = P07DataLoader()
        merged = loader.merge_mtf(df_exec, df_anchor)

        X = build_features(merged.dropna(), {}, enable_mtf=True)

        assert "anchor_trend" in X.columns
        assert "anchor_rsi" in X.columns
        # Not all zeros — real computation happened
        assert not (X["anchor_trend"] == 0.0).all()


# ---------------------------------------------------------------------------
# 4. P08Pipeline deprecation shim
# ---------------------------------------------------------------------------

class TestP08PipelineDeprecationShim:

    def test_p08_pipeline_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="P08Pipeline is deprecated"):
            with patch("src.ml.pipeline.p08_mtf.pipeline.P07Pipeline"):
                from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
                P08Pipeline()

    def test_p08_pipeline_run_batch_delegates_to_p07(self):
        mock_delegate = MagicMock()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("src.ml.pipeline.p08_mtf.pipeline.P07Pipeline", return_value=mock_delegate):
                from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
                p = P08Pipeline()
                fake_files = [Path("BTC_15m_20230101_20231231.csv")]
                p.run_batch(fake_files, train_years=["2023"])

        mock_delegate.run_batch.assert_called_once_with(fake_files, train_years=["2023"])

    def test_p08_pipeline_delegates_with_enable_mtf_true(self):
        """P07Pipeline must always be constructed with enable_mtf=True."""
        captured_kwargs = {}

        def fake_p07(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("src.ml.pipeline.p08_mtf.pipeline.P07Pipeline", side_effect=fake_p07):
                from src.ml.pipeline.p08_mtf.pipeline import P08Pipeline
                P08Pipeline()

        assert captured_kwargs.get("enable_mtf") is True, (
            "P08Pipeline shim must construct P07Pipeline with enable_mtf=True"
        )
