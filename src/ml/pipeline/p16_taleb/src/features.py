"""
Feature engineering for the P16 Taleb barbell pipeline.

All features are computed on the master daily DataFrame produced by
data_loader.load_all(). This module is a pure function with no I/O.
"""

import logging

import pandas as pd

_logger = logging.getLogger(__name__)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the master daily DataFrame.

    Input columns required: close, vix, avgtone (optional).
    Returns the same DataFrame with additional columns appended in-place.

    Args:
        df: Master daily DataFrame with DatetimeIndex.

    Returns:
        Enriched DataFrame with all feature columns added.
    """
    out = df.copy()

    # --- Returns ---
    out["ret_1d"] = out["close"].pct_change(1)
    out["ret_5d"] = out["close"].pct_change(5)
    out["ret_21d"] = out["close"].pct_change(21)
    out["ret_63d"] = out["close"].pct_change(63)

    # --- Drawdown from all-time high ---
    cum_peak = out["close"].cummax()
    out["drawdown"] = (out["close"] - cum_peak) / cum_peak

    # --- VIX features ---
    out["vix_ma20"] = out["vix"].rolling(20).mean()
    out["vol_ratio"] = out["vix"] / out["vix_ma20"]

    vix_bins = [0, 15, 20, 30, 40, 200]
    vix_labels = ["low", "normal", "elevated", "high", "extreme"]
    out["vix_regime"] = pd.cut(
        out["vix"],
        bins=vix_bins,
        labels=vix_labels,
        right=False,
    )

    # --- Stress flag ---
    out["stress_flag"] = (out["vix"] > 30) | (out["drawdown"] < -0.10)

    # --- GDELT features (NaN when coverage starts 2015-02-18) ---
    if "avgtone" in out.columns:
        out["gdelt_tone_ma5"] = out["avgtone"].rolling(5).mean()
        out["gdelt_available"] = out["avgtone"].notna()
    else:
        out["gdelt_tone_ma5"] = float("nan")
        out["gdelt_available"] = False

    _logger.info(
        "build_features: %d rows, stress periods=%d, gdelt_available=%d",
        len(out),
        int(out["stress_flag"].sum()),
        int(out["gdelt_available"].sum()) if "gdelt_available" in out.columns else 0,
    )
    return out
