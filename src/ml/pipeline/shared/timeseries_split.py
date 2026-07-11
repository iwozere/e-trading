"""
Leakage-safe time-series split and feature-scaling primitives (ML-2).

All 17 ML pipelines should use these helpers instead of rolling their own
train/val splits or scalers.  Using a shared implementation makes it
structurally impossible for pipelines to diverge on leakage-safety.

## Rules enforced here

1. **Split before scale** — the scaler is fit on the training fold only; the
   validation fold is transformed with the already-fitted scaler.  Fitting the
   scaler on the full dataset before splitting is the most common source of
   preprocessing leakage.

2. **No shuffle for time-series** — ``shuffle=False`` preserves temporal order.
   Shuffling time-series data breaks the forward-only causality guarantee and
   allows information from future bars to leak into the training set.

3. **Single split point** — for simple hold-out validation.  For walk-forward
   or k-fold validation, use ``sklearn.model_selection.TimeSeriesSplit`` directly
   with ``shuffle=False``.
"""

from __future__ import annotations

from typing import Tuple, Type, Union, cast

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SCALER_MAP: dict[str, Type[TransformerMixin]] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}

_SplitResult = Tuple[
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
    Union[pd.Series, np.ndarray],
    Union[pd.Series, np.ndarray],
]


def split_timeseries(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    test_size: float = 0.2,
) -> Tuple[
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
    Union[pd.Series, np.ndarray],
    Union[pd.Series, np.ndarray],
]:
    """
    Split time-series data into train and validation sets without shuffling.

    Args:
        X: Feature matrix (rows are observations ordered by time).
        y: Target vector aligned with ``X``.
        test_size: Fraction of observations to allocate to the validation set.
            The split is always from the *end* of the series (no shuffling).

    Returns:
        ``(X_train, X_val, y_train, y_val)`` — all with the same dtype as inputs.
    """
    # sklearn's stubs type the split results as plain lists; the runtime values
    # keep the input types (DataFrame/ndarray slices)
    X_train, X_val, y_train, y_val = cast(
        _SplitResult,
        tuple(
            train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=False,  # preserve temporal order — must never be True for time-series
            )
        ),
    )
    _logger.debug(
        "split_timeseries: train=%d rows, val=%d rows (test_size=%.2f)",
        len(X_train),
        len(X_val),
        test_size,
    )
    return X_train, X_val, y_train, y_val


def fit_scaler(
    X_train: Union[pd.DataFrame, np.ndarray],
    scaler_type: str = "standard",
) -> TransformerMixin:
    """
    Fit a scaler on the training fold only.

    Args:
        X_train: Training features.
        scaler_type: One of ``"standard"``, ``"minmax"``, or ``"robust"``.

    Returns:
        Fitted scaler instance.

    Raises:
        ValueError: If ``scaler_type`` is not recognised.
    """
    cls = _SCALER_MAP.get(scaler_type.lower())
    if cls is None:
        raise ValueError(f"Unknown scaler_type '{scaler_type}'. Supported: {list(_SCALER_MAP)}")
    scaler = cls()
    scaler.fit(X_train)  # type: ignore[attr-defined]
    _logger.debug("fit_scaler: fitted %s on %d samples", scaler_type, len(X_train))
    return scaler


def apply_scaler(
    scaler: TransformerMixin,
    X: Union[pd.DataFrame, np.ndarray],
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Transform features using an already-fitted scaler.

    Preserves the DataFrame index and column names if ``X`` is a DataFrame.

    Args:
        scaler: Previously fitted scaler (from :func:`fit_scaler`).
        X: Feature matrix to transform.

    Returns:
        Scaled feature matrix with the same type as ``X``.
    """
    scaled = scaler.transform(X)  # type: ignore[attr-defined]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)
    return scaled


def split_and_scale(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    *,
    test_size: float = 0.2,
    scaler_type: str = "standard",
    fitted_scaler: TransformerMixin | None = None,
) -> Tuple[
    Union[pd.DataFrame, np.ndarray],
    Union[pd.DataFrame, np.ndarray],
    Union[pd.Series, np.ndarray],
    Union[pd.Series, np.ndarray],
    TransformerMixin,
]:
    """
    Split a time-series dataset and scale features in a leakage-safe way.

    Order of operations (critical):
    1. Split (no shuffle) → train / val.
    2. Fit scaler on train only.
    3. Transform both train and val with the fitted scaler.

    Args:
        X: Feature matrix ordered by time.
        y: Target vector aligned with ``X``.
        test_size: Validation fraction (passed to :func:`split_timeseries`).
        scaler_type: Scaler variant (passed to :func:`fit_scaler`).
        fitted_scaler: Optional pre-fitted scaler.  When provided, ``scaler_type``
            is ignored and this scaler is used directly (useful for applying a
            training-fold scaler to a held-out test set).

    Returns:
        ``(X_train_scaled, X_val_scaled, y_train, y_val, fitted_scaler)``
    """
    X_train_raw, X_val_raw, y_train, y_val = split_timeseries(X, y, test_size=test_size)

    if fitted_scaler is None:
        fitted_scaler = fit_scaler(X_train_raw, scaler_type=scaler_type)

    X_train_scaled = apply_scaler(fitted_scaler, X_train_raw)
    X_val_scaled = apply_scaler(fitted_scaler, X_val_raw)

    return X_train_scaled, X_val_scaled, y_train, y_val, fitted_scaler
