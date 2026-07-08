"""
Tests for Phase 4 — multi-regime LSTM training in P01.

Verifies:
1. _train_multi_regime() produces one model file per HMM regime with >= min_regime_samples.
2. Regimes below min_regime_samples are skipped.
3. LSTMValidator.find_regime_models() discovers per-regime .pkl files.
4. _validate_multi_regime() evaluates only on each regime's own rows.
5. P00 run_pipeline.py emits DeprecationWarning.
"""

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labeled_df(n: int = 600, n_regimes: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": rng.integers(1000, 5000, size=n).astype(float),
            "log_return": np.concatenate([[0], np.diff(np.log(close))]),
            "regime": (np.arange(n) % n_regimes).astype(int),
            "regime_confidence": rng.random(n),
            "regime_duration": rng.integers(1, 50, size=n).astype(float),
        }
    )


def _minimal_config(multi_regime: bool = False, min_samples: int = 10) -> dict:
    return {
        "lstm": {
            "sequence_length": 10,
            "hidden_size": 16,
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 0.01,
            "validation_split": 0.2,
            "dropout": 0.1,
            "num_layers": 1,
            "multi_regime": multi_regime,
            "min_regime_samples": min_samples,
        },
        "hmm": {"n_components": 3},
        "evaluation": {"test_split": 0.1},
        "paths": {
            "data_labeled": "data/labeled",
            "models_lstm": "src/ml/pipeline/p01_hmm_lstm/models/lstm",
            "reports": "src/ml/pipeline/p01_hmm_lstm/models/results",
        },
    }


# ---------------------------------------------------------------------------
# 1. _train_multi_regime produces one result per qualifying regime
# ---------------------------------------------------------------------------


class TestTrainMultiRegime:
    def test_produces_result_per_regime(self, tmp_path):
        """_train_multi_regime returns one entry per regime with enough samples."""
        from src.ml.pipeline.p01_hmm_lstm.x_06_train_lstm import LSTMTrainer

        cfg = _minimal_config(multi_regime=True, min_samples=10)

        with patch.object(LSTMTrainer, "__init__", return_value=None):
            trainer = LSTMTrainer.__new__(LSTMTrainer)
            trainer.config = cfg
            trainer.models_dir = tmp_path
            trainer.labeled_data_dir = tmp_path

        df = _make_labeled_df(300, n_regimes=3)

        save_calls = []

        def fake_train_single(df_in, symbol, timeframe, lstm_params, regime_suffix=""):
            model_file = tmp_path / f"lstm_SYM_15m{regime_suffix}_20230101_000000.pkl"
            # Write a dummy pkl
            with open(model_file, "wb") as f:
                pickle.dump({"dummy": True}, f)
            save_calls.append(regime_suffix)
            return {
                "model_path": str(model_file),
                "best_val_loss": 0.01,
                "training_epochs": 2,
                "n_parameters": 100,
                "features_used": 5,
            }

        with patch.object(trainer, "_train_single", side_effect=fake_train_single):
            results = trainer._train_multi_regime(df, "SYM", "15m", cfg["lstm"])

        assert len(results) == 3, "Should have one result per regime"
        assert all(r["success"] for r in results.values())
        assert set(results.keys()) == {0, 1, 2}

    def test_skips_regimes_below_min_samples(self, tmp_path):
        """Regimes with fewer rows than min_regime_samples must be skipped."""
        from src.ml.pipeline.p01_hmm_lstm.x_06_train_lstm import LSTMTrainer

        cfg = _minimal_config(multi_regime=True, min_samples=200)

        with patch.object(LSTMTrainer, "__init__", return_value=None):
            trainer = LSTMTrainer.__new__(LSTMTrainer)
            trainer.config = cfg
            trainer.models_dir = tmp_path
            trainer.labeled_data_dir = tmp_path

        # 300 rows evenly split → 100 per regime → all below 200
        df = _make_labeled_df(300, n_regimes=3)

        with patch.object(trainer, "_train_single") as mock_train:
            results = trainer._train_multi_regime(df, "SYM", "15m", cfg["lstm"])

        mock_train.assert_not_called()
        assert results == {}

    def test_regime_suffix_in_filename(self, tmp_path):
        """_train_single is called with '_regime{id}' suffix for each regime."""
        from src.ml.pipeline.p01_hmm_lstm.x_06_train_lstm import LSTMTrainer

        cfg = _minimal_config(multi_regime=True, min_samples=5)

        with patch.object(LSTMTrainer, "__init__", return_value=None):
            trainer = LSTMTrainer.__new__(LSTMTrainer)
            trainer.config = cfg
            trainer.models_dir = tmp_path
            trainer.labeled_data_dir = tmp_path

        df = _make_labeled_df(60, n_regimes=3)  # 20 per regime > min_samples=5

        captured_suffixes = []

        def fake_train_single(df_in, symbol, timeframe, lstm_params, regime_suffix=""):
            captured_suffixes.append(regime_suffix)
            model_file = tmp_path / f"lstm_SYM_15m{regime_suffix}_20230101.pkl"
            with open(model_file, "wb") as f:
                pickle.dump({}, f)
            return {
                "model_path": str(model_file),
                "best_val_loss": 0.0,
                "training_epochs": 1,
                "n_parameters": 10,
                "features_used": 3,
            }

        with patch.object(trainer, "_train_single", side_effect=fake_train_single):
            trainer._train_multi_regime(df, "SYM", "15m", cfg["lstm"])

        assert "_regime0" in captured_suffixes
        assert "_regime1" in captured_suffixes
        assert "_regime2" in captured_suffixes


# ---------------------------------------------------------------------------
# 2. find_regime_models discovers per-regime .pkl files
# ---------------------------------------------------------------------------


class TestFindRegimeModels:
    def test_finds_per_regime_pkl_files(self, tmp_path):
        from src.ml.pipeline.p01_hmm_lstm.x_07_validate_lstm import LSTMValidator

        # Create dummy per-regime model files
        for regime in [0, 1, 2]:
            (tmp_path / f"lstm_BTC_15m_regime{regime}_20230101_000000.pkl").write_bytes(b"")

        with patch.object(LSTMValidator, "__init__", return_value=None):
            validator = LSTMValidator.__new__(LSTMValidator)
            validator.models_dir = tmp_path

        regime_map = validator.find_regime_models("BTC", "15m")

        assert set(regime_map.keys()) == {0, 1, 2}

    def test_returns_latest_file_per_regime(self, tmp_path):
        from src.ml.pipeline.p01_hmm_lstm.x_07_validate_lstm import LSTMValidator

        # Two files for regime 0, one older one newer
        (tmp_path / "lstm_BTC_15m_regime0_20230101_000000.pkl").write_bytes(b"")
        (tmp_path / "lstm_BTC_15m_regime0_20230601_000000.pkl").write_bytes(b"")

        with patch.object(LSTMValidator, "__init__", return_value=None):
            validator = LSTMValidator.__new__(LSTMValidator)
            validator.models_dir = tmp_path

        regime_map = validator.find_regime_models("BTC", "15m")

        assert "20230601" in regime_map[0].name

    def test_find_latest_model_excludes_regime_files(self, tmp_path):
        """find_latest_model(regime_id=None) must not return per-regime files."""
        from src.ml.pipeline.p01_hmm_lstm.x_07_validate_lstm import LSTMValidator

        (tmp_path / "lstm_BTC_15m_regime0_20230101.pkl").write_bytes(b"")
        (tmp_path / "lstm_BTC_15m_20230101_000000.pkl").write_bytes(b"")

        with patch.object(LSTMValidator, "__init__", return_value=None):
            validator = LSTMValidator.__new__(LSTMValidator)
            validator.models_dir = tmp_path

        result = validator.find_latest_model("BTC", "15m", regime_id=None)

        assert result is not None
        assert "_regime" not in result.name


# ---------------------------------------------------------------------------
# 3. P00 deprecation shim
# ---------------------------------------------------------------------------


class TestP00DeprecationShim:
    def test_p00_run_pipeline_emits_deprecation_warning(self):
        """Importing p00_hmm_3lstm/run_pipeline.py must emit DeprecationWarning."""
        import importlib
        import sys

        mod_name = "src.ml.pipeline.p00_hmm_3lstm.run_pipeline"
        # Ensure fresh import
        sys.modules.pop(mod_name, None)

        with pytest.warns(DeprecationWarning, match="p00_hmm_3lstm"):
            importlib.import_module(mod_name)
