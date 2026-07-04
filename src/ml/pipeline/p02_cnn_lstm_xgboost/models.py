"""
CNN model variants for P02 (CNN-LSTM-XGBoost pipeline).

Provides:
- CNN1D  — simple 1D convolutional network (merged from P03); lower overfitting risk,
           better suited for shorter time series.
- build_cnn_model() — factory that returns the right model based on cnn_variant config.

HybridCNNLSTM (CNN + attention + LSTM) lives in x_03_optuna_cnn_lstm.py and is
selected when cnn_variant = "cnn_lstm".
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# CNN1D — merged from src/ml/pipeline/p03_cnn_xgboost/x_02_train_cnn.py
# ---------------------------------------------------------------------------


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for time series feature extraction.

    Designed for OHLCV financial data with configurable architecture.
    Produces a fixed-length embedding vector via global average pooling.
    """

    def __init__(
        self,
        input_channels: int = 5,
        sequence_length: int = 120,
        num_filters: List[int] | None = None,
        kernel_sizes: List[int] | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        """
        Args:
            input_channels: Number of input features (OHLCV = 5).
            sequence_length: Length of the input time series (informational only).
            num_filters: Filter counts per conv layer. Default [32, 64, 128].
            kernel_sizes: Kernel sizes per conv layer. Default [3, 5, 7].
            dropout_rate: Dropout rate applied after each conv block.
        """
        super().__init__()

        if num_filters is None:
            num_filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.embedding_dim = num_filters[-1]

        layers: list = []
        in_ch = input_channels
        for filters, ks in zip(num_filters, kernel_sizes):
            layers.extend(
                [
                    nn.Conv1d(in_ch, filters, ks, padding=ks // 2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            in_ch = filters

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, input_channels, sequence_length).
        Returns:
            Embedding tensor of shape (batch, embedding_dim).
        """
        out = self.conv_layers(x)
        out = self.global_pool(out)
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_cnn_model(cnn_variant: str = "cnn_lstm", **kwargs) -> nn.Module:
    """
    Instantiate the CNN backbone specified by cnn_variant.

    Args:
        cnn_variant: "simple" → CNN1D; "cnn_lstm" → HybridCNNLSTM.
        **kwargs: Forwarded to the model constructor.

    Returns:
        An nn.Module instance.
    """
    if cnn_variant == "simple":
        _logger.info("CNN variant: CNN1D (simple)")
        return CNN1D(
            input_channels=kwargs.get("input_channels", 5),
            sequence_length=kwargs.get("sequence_length", 120),
            num_filters=kwargs.get("num_filters"),
            kernel_sizes=kwargs.get("kernel_sizes"),
            dropout_rate=kwargs.get("dropout_rate", 0.3),
        )

    if cnn_variant == "cnn_lstm":
        _logger.info("CNN variant: HybridCNNLSTM")
        try:
            import sys
            from pathlib import Path

            p02_dir = Path(__file__).parent
            if str(p02_dir) not in sys.path:
                sys.path.insert(0, str(p02_dir))
            from x_03_optuna_cnn_lstm import HybridCNNLSTM  # type: ignore[import]

            return HybridCNNLSTM(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"HybridCNNLSTM requires x_03_optuna_cnn_lstm.py to be importable. Original error: {e}"
            ) from e

    raise ValueError(f"Unknown cnn_variant '{cnn_variant}'. Valid values: 'simple', 'cnn_lstm'.")
