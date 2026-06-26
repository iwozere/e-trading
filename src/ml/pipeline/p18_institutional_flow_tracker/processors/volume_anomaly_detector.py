"""
Volume Anomaly Detector

Detects abnormal trading volume in stocks that are already flagged for
institutional distribution.  Uses the existing DataManager to fetch OHLCV data.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd

from src.data.data_manager import DataManager
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class VolumeAnomalyDetector:
    """
    Checks a list of tickers for volume spikes relative to their rolling average.

    Fetches OHLCV from DataManager and computes:
    - 20-day rolling average volume (baseline)
    - Average volume over the last N days (signal window)
    - Spike ratio = signal_window_avg / baseline_avg
    """

    def __init__(
        self,
        data_manager: Optional[DataManager] = None,
        lookback_days: int = 20,
        spike_recent_days: int = 5,
        spike_multiplier: float = 3.5,
    ):
        """
        Args:
            data_manager: DataManager instance. Created fresh if None.
            lookback_days: Rolling window for the baseline volume average.
            spike_recent_days: Number of recent days evaluated for the spike.
            spike_multiplier: Threshold — recent_avg / baseline_avg must exceed
                this value to flag the ticker.
        """
        self._dm = data_manager or DataManager()
        self._lookback = lookback_days
        self._recent = spike_recent_days
        self._threshold = spike_multiplier

    def detect(
        self,
        tickers: List[str],
        as_of_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Run volume anomaly detection on a list of tickers.

        Args:
            tickers: List of ticker symbols to check.
            as_of_date: Reference date for OHLCV end date. Defaults to yesterday UTC.

        Returns:
            DataFrame of tickers that triggered the spike threshold, with columns:
            ticker, volume_spike_ratio, price_change_5d_pct, above_spike_days.
            Sorted by volume_spike_ratio descending.
        """
        if not tickers:
            return pd.DataFrame()

        end = as_of_date or (datetime.now(timezone.utc) - timedelta(days=1))
        start = end - timedelta(days=self._lookback + self._recent + 5)

        results = []
        for ticker in tickers:
            try:
                df = self._dm.get_ohlcv(
                    symbol=ticker,
                    timeframe="1d",
                    start_date=start,
                    end_date=end,
                )
                if df is None or len(df) < self._lookback:
                    continue

                row = _compute_spike(df, self._lookback, self._recent, self._threshold)
                if row is not None:
                    row["ticker"] = ticker
                    results.append(row)
            except Exception:
                _logger.exception("Volume anomaly check failed for %s", ticker)

        if not results:
            return pd.DataFrame()

        out = pd.DataFrame(results)
        out.sort_values("volume_spike_ratio", ascending=False, inplace=True)
        out.reset_index(drop=True, inplace=True)
        _logger.info(
            "Volume anomaly: %d/%d tickers flagged (threshold=%.1fx)",
            len(out), len(tickers), self._threshold,
        )
        return out


def _compute_spike(
    df: pd.DataFrame,
    lookback: int,
    recent: int,
    threshold: float,
) -> Optional[dict]:
    """Compute spike metrics for a single ticker's OHLCV DataFrame."""
    if "volume" not in df.columns or df.empty:
        return None

    vol = df["volume"].dropna()
    if len(vol) < lookback + recent:
        return None

    baseline_avg = float(vol.iloc[-(lookback + recent): -recent].mean())
    recent_avg = float(vol.iloc[-recent:].mean())

    if baseline_avg == 0:
        return None

    spike_ratio = recent_avg / baseline_avg
    if spike_ratio < threshold:
        return None

    price_change = 0.0
    if "close" in df.columns and len(df) >= recent:
        close = df["close"].dropna()
        if len(close) >= recent + 1:
            price_change = float((close.iloc[-1] - close.iloc[-recent - 1]) / close.iloc[-recent - 1] * 100)

    above_days = int((vol.iloc[-recent:] > baseline_avg * threshold).sum())

    return {
        "volume_spike_ratio": round(spike_ratio, 2),
        "price_change_5d_pct": round(price_change, 2),
        "above_spike_days": above_days,
    }
