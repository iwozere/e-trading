"""
Tests for Phase 5 — P02/P03 CNN variant merge.

Verifies:
1. build_cnn_model("simple") returns a CNN1D instance.
2. build_cnn_model("cnn_lstm") raises ImportError when HybridCNNLSTM unavailable (not a real failure path, tested via mock).
3. build_cnn_model with invalid variant raises ValueError.
4. CNN1D forward pass produces correct embedding shape.
5. ModelValidator.walk_forward_validate() returns per-fold metrics and averages.
6. P03 run_pipeline.py emits DeprecationWarning.
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 1. CNN1D model via build_cnn_model factory
# ---------------------------------------------------------------------------


class TestBuildCnnModel:
    def test_simple_variant_returns_cnn1d(self):
        from src.ml.pipeline.p02_cnn_lstm_xgboost.models import CNN1D, build_cnn_model

        model = build_cnn_model("simple", input_channels=5, sequence_length=60)
        assert isinstance(model, CNN1D)

    def test_unknown_variant_raises_value_error(self):
        from src.ml.pipeline.p02_cnn_lstm_xgboost.models import build_cnn_model

        with pytest.raises(ValueError, match="Unknown cnn_variant"):
            build_cnn_model("transformer")

    def test_cnn_lstm_variant_raises_import_error_when_module_absent(self):
        from src.ml.pipeline.p02_cnn_lstm_xgboost.models import build_cnn_model

        # Patch the import inside build_cnn_model to simulate missing module
        with patch.dict(sys.modules, {"x_03_optuna_cnn_lstm": None}):
            with pytest.raises(ImportError):
                build_cnn_model("cnn_lstm")


# ---------------------------------------------------------------------------
# 2. CNN1D forward pass
# ---------------------------------------------------------------------------


class TestCNN1DForward:
    def test_output_shape_is_embedding_dim(self):
        import torch

        from src.ml.pipeline.p02_cnn_lstm_xgboost.models import CNN1D

        model = CNN1D(input_channels=5, sequence_length=60, num_filters=[16, 32], kernel_sizes=[3, 5])
        model.eval()
        batch = torch.randn(4, 5, 60)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"

    def test_default_embedding_dim_is_128(self):
        import torch

        from src.ml.pipeline.p02_cnn_lstm_xgboost.models import CNN1D

        model = CNN1D(input_channels=5, sequence_length=120)
        model.eval()
        batch = torch.randn(2, 5, 120)
        with torch.no_grad():
            out = model(batch)

        assert out.shape[-1] == 128, f"Default embedding dim should be 128, got {out.shape[-1]}"


# ---------------------------------------------------------------------------
# 3. walk_forward_validate produces fold metrics
# ---------------------------------------------------------------------------


class TestWalkForwardValidate:
    def _make_validator(self):
        from src.ml.pipeline.p02_cnn_lstm_xgboost.x_08_validate_models import ModelValidator

        with patch.object(ModelValidator, "__init__", return_value=None):
            v = ModelValidator.__new__(ModelValidator)
            v.config = {}
            v.reports_dir = Path(".")
            v.predictions_dir = Path(".")
        return v

    def test_returns_n_folds_metrics(self):
        validator = self._make_validator()
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(200)
        y_pred = y_true + rng.standard_normal(200) * 0.1

        result = validator.walk_forward_validate(y_true, y_pred, n_splits=4)

        assert result["n_folds"] == 4
        assert len(result["fold_metrics"]) == 4

    def test_avg_mse_is_finite(self):
        validator = self._make_validator()
        rng = np.random.default_rng(1)
        y_true = rng.standard_normal(300)
        y_pred = y_true + rng.standard_normal(300) * 0.05

        result = validator.walk_forward_validate(y_true, y_pred, n_splits=5)

        assert np.isfinite(result["avg_mse"])
        assert np.isfinite(result["avg_directional_accuracy"])

    def test_directional_accuracy_between_0_and_1(self):
        validator = self._make_validator()
        rng = np.random.default_rng(2)
        y_true = np.cumsum(rng.standard_normal(400))
        y_pred = y_true * 0.9 + rng.standard_normal(400) * 0.5

        result = validator.walk_forward_validate(y_true, y_pred, n_splits=5)

        for fold in result["fold_metrics"]:
            da = fold["directional_accuracy"]
            assert 0.0 <= da <= 1.0, f"Fold {fold['fold']}: directional_accuracy={da} out of range"


# ---------------------------------------------------------------------------
# 4. P03 deprecation shim
# ---------------------------------------------------------------------------


class TestP03DeprecationShim:
    def test_p03_run_pipeline_emits_deprecation_warning(self):
        mod_name = "src.ml.pipeline.p03_cnn_xgboost.run_pipeline"
        sys.modules.pop(mod_name, None)

        with pytest.warns(DeprecationWarning, match="p03_cnn_xgboost"):
            importlib.import_module(mod_name)
