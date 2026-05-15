"""
P17 Technical Agent

Computes all technical indicators from daily OHLCV data:
  - Relative volume (rvol)
  - Price momentum (5d / 20d / 60d returns)
  - SMA20, SMA50, price-vs-SMA50
  - Breakout detection (above 20d / 50d high)
  - Bollinger Band squeeze
  - ATR (Average True Range) as % of price
  - OBV slope (accumulation proxy)
  - Accumulation day count (high-volume green days)

All indicators are computed using pure pandas/numpy — no TA-Lib dependency.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p17_penny_stocks.config import P17TechnicalConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate

_logger = setup_logger(__name__)


class TechnicalAgent:
    """
    Stage 3: Compute technical features for each candidate.

    Accepts the OHLCV dict from MarketAgent and populates technical fields
    on each Candidate object in-place.
    """

    def __init__(self, config: P17TechnicalConfig) -> None:
        self.config = config

    def run(
        self,
        candidates: List[Candidate],
        ohlcv: Dict[str, pd.DataFrame],
    ) -> List[Candidate]:
        """
        Enrich candidates with technical indicators.

        Args:
            candidates: List of Candidate objects (with market snapshot filled).
            ohlcv: Dict[ticker → daily OHLCV DataFrame].

        Returns:
            Same list with technical fields populated.
        """
        enriched, skipped = 0, 0
        for c in candidates:
            df = ohlcv.get(c.ticker)
            if df is None or df.empty:
                _logger.debug("No OHLCV for %s — skipping technical analysis", c.ticker)
                skipped += 1
                continue
            self._enrich(c, df)
            enriched += 1

        _logger.info("Technical agent: %d enriched, %d skipped (no OHLCV)", enriched, skipped)
        return candidates

    # ── Per-ticker enrichment ──────────────────────────────────────────────

    def _enrich(self, c: Candidate, df: pd.DataFrame) -> None:
        df = df.copy().sort_index()

        # Explicit Series extraction — avoids Pyright's ambiguous df[col] inference
        close: pd.Series = pd.Series(df["Close"].values, index=df.index)
        high: pd.Series = pd.Series(df["High"].values, index=df.index)
        low: pd.Series = pd.Series(df["Low"].values, index=df.index)
        volume: pd.Series = pd.Series(df["Volume"].values, index=df.index)
        n = len(df)

        # ── Relative volume ────────────────────────────────────────────────
        lookback = min(self.config.rvol_lookback_days, n - 1)
        avg_vol = float(volume.iloc[-lookback - 1 : -1].mean()) if lookback > 0 else float(volume.mean())
        today_vol = float(volume.iloc[-1])
        c.relative_volume = today_vol / avg_vol if avg_vol > 0 else 0.0

        # ── Price returns ──────────────────────────────────────────────────
        c.price_5d_return = self._return(close, 5)
        c.price_20d_return = self._return(close, 20)
        c.price_60d_return = self._return(close, 60)

        # Cap 20d return — above 300% signals late euphoric spike
        if c.price_20d_return > self.config.momentum_20d_max:
            c.price_20d_return = self.config.momentum_20d_max

        # ── Moving averages ────────────────────────────────────────────────
        close_arr = close.to_numpy(dtype=float)
        high_arr = high.to_numpy(dtype=float)
        volume_arr = volume.to_numpy(dtype=float)
        current_price = float(close_arr[-1])

        if n >= 20:
            c.sma20 = float(np.array(close.rolling(20).mean(), dtype=float)[-1])
        if n >= 50:
            c.sma50 = float(np.array(close.rolling(50).mean(), dtype=float)[-1])
            c.above_sma50 = current_price > c.sma50

        # ── Breakout detection ─────────────────────────────────────────────
        lb20 = min(self.config.breakout_lookback_20d, n - 1)
        lb50 = min(self.config.breakout_lookback_50d, n - 1)

        if lb20 > 1:
            c.breakout_20d = current_price > float(high_arr[-(lb20 + 1) : -1].max())

        if lb50 > 1:
            c.breakout_50d = current_price > float(high_arr[-(lb50 + 1) : -1].max())

        # ── Bollinger Band squeeze ─────────────────────────────────────────
        bb_len = min(self.config.bb_period, n)
        if bb_len >= 10:
            sma_arr = np.array(close.rolling(bb_len).mean(), dtype=float)
            std_arr = np.array(close.rolling(bb_len).std(), dtype=float)
            bb_mid = float(sma_arr[-1])
            if bb_mid > 0:
                bb_width = (2 * self.config.bb_std * float(std_arr[-1])) / bb_mid
                c.bb_squeeze = bb_width < self.config.bb_squeeze_width_pct

        # ── ATR as % of price ──────────────────────────────────────────────
        atr_len = min(self.config.atr_period, n - 1)
        if atr_len >= 5:
            tr = self._true_range(high, low, close)
            atr = float(np.array(tr.rolling(atr_len).mean(), dtype=float)[-1])
            c.atr_pct = atr / current_price if current_price > 0 else 0.0

        # ── OBV slope ─────────────────────────────────────────────────────
        if n >= 10:
            diff_arr = np.diff(np.concatenate([[close_arr[0]], close_arr]))
            direction_arr = np.sign(diff_arr)
            obv_arr = np.cumsum(direction_arr * volume_arr)
            c.obv_slope = self._linear_slope(obv_arr[-10:])

        # ── Accumulation days ──────────────────────────────────────────────
        lookback_acc = min(self.config.accumulation_lookback_days, n)
        if lookback_acc >= 5:
            open_arr = pd.Series(df["Open"].values, index=df.index).to_numpy(dtype=float)
            avg_vol_acc = float(volume_arr[-(lookback_acc + 1) : -1].mean())
            green = close_arr[-lookback_acc:] > open_arr[-lookback_acc:]
            high_vol = volume_arr[-lookback_acc:] > avg_vol_acc
            c.accumulation_days = int((green & high_vol).sum())

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _return(close: pd.Series, days: int) -> float:
        n = len(close)
        if n < days + 1:
            return 0.0
        start = float(close.iloc[-(days + 1)])
        end = float(close.iloc[-1])
        return (end - start) / start if start > 0 else 0.0

    @staticmethod
    def _true_range(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        prev_close = close.shift(1)
        hl = high - low
        hc = (high - prev_close).abs()
        lc = (low - prev_close).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1)

    @staticmethod
    def _linear_slope(values: np.ndarray) -> float:
        """Least-squares slope of an array, normalised by its mean."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        mean_val = values.mean()
        if mean_val == 0:
            return 0.0
        slope = float(np.polyfit(x, values, 1)[0])
        return slope / abs(mean_val)

    # ── Universe-level post-filter ─────────────────────────────────────────

    def apply_breakout_filter(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Optional hard filter: remove candidates with zero technical signals.
        Use only when the universe is too large to score fully.
        """
        return [
            c for c in candidates
            if c.breakout_20d or c.relative_volume >= self.config.rvol_strong_threshold
            or c.bb_squeeze or c.accumulation_days >= 3
        ]

    def build_snapshot(self, candidates: List[Candidate]) -> Optional[pd.DataFrame]:
        """Return a DataFrame summary of technical features for logging / CSV."""
        if not candidates:
            return None
        rows = [
            {
                "ticker": c.ticker,
                "price": c.price,
                "rvol": round(c.relative_volume, 2),
                "20d_ret": round(c.price_20d_return, 3),
                "above_sma50": c.above_sma50,
                "breakout_20d": c.breakout_20d,
                "breakout_50d": c.breakout_50d,
                "bb_squeeze": c.bb_squeeze,
                "atr_pct": round(c.atr_pct, 4),
                "obv_slope": round(c.obv_slope, 4),
                "accum_days": c.accumulation_days,
            }
            for c in candidates
        ]
        return pd.DataFrame(rows)
