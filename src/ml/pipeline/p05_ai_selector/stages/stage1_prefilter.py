"""Stage 1 — Liquidity & Momentum Pre-filter: ~3,020 → ~200 candidates."""

import json
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import List, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.ml.pipeline.p05_ai_selector.config import (
    MIN_PRICE,
    MIN_AVG_DAILY_VOLUME_USD,
    MIN_CRYPTO_DAILY_VOLUME,
    STAGE1_LOOKBACK_DAYS,
    STAGE1_TOP_N,
    STAGE1_CACHE_DIR,
    CRYPTO_TICKERS,
)
from src.data.data_manager import DataManager
from src.ml.pipeline.p05_ai_selector.signals.technical import score_technicals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_CRYPTO_SET = set(CRYPTO_TICKERS)


class Stage1Prefilter:
    """
    Applies hard liquidity filters and soft momentum scoring.

    Hard filters:
      - Last close > MIN_PRICE ($2)
      - Average daily volume × price > MIN_AVG_DAILY_VOLUME_USD ($5M) for equities
      - Average daily volume > MIN_CRYPTO_DAILY_VOLUME for crypto

    Soft scoring: SMA crossover, RSI, volume surge, momentum, ATR, 52-week proximity.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or STAGE1_CACHE_DIR

    def run(
        self,
        tickers: List[str],
        as_of_date: date,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Filter and score the universe.

        Args:
            tickers: Full universe ticker list.
            as_of_date: Reference date (used for OHLCV fetch and cache naming).
            force_refresh: Bypass cache and recompute.

        Returns:
            DataFrame sorted descending by momentum_score, capped at STAGE1_TOP_N.
            Columns: ticker, asset_type, last_price, avg_vol_usd, momentum_score,
                     volume_surge_ratio, signal_breakdown.
        """
        cache_file = self._cache_dir / f"{as_of_date}.csv.gz"
        if not force_refresh and cache_file.exists():
            _logger.info("Stage1: loading from cache %s", cache_file)
            return pd.read_csv(cache_file, compression="gzip")

        _logger.info("Stage1: processing %d symbols for %s", len(tickers), as_of_date)

        dm = DataManager()

        start_dt = datetime.combine(as_of_date, dt_time.min)
        # Request extra days to account for weekends/holidays
        from datetime import timedelta
        lookback_start = datetime.combine(
            as_of_date - timedelta(days=STAGE1_LOOKBACK_DAYS + 10), dt_time.min
        )

        rows = []
        n_price_fail = 0
        n_vol_fail = 0

        for ticker in tickers:
            is_crypto = ticker in _CRYPTO_SET
            try:
                ohlcv = dm.get_ohlcv(ticker, "1d", lookback_start, start_dt)
                if ohlcv is None or ohlcv.empty:
                    continue

                last_price = float(ohlcv["close"].iloc[-1])
                if last_price < MIN_PRICE:
                    n_price_fail += 1
                    continue

                avg_vol = float(ohlcv["volume"].tail(20).mean())

                if is_crypto:
                    if avg_vol < MIN_CRYPTO_DAILY_VOLUME:
                        n_vol_fail += 1
                        continue
                    avg_vol_usd = avg_vol * last_price
                else:
                    avg_vol_usd = avg_vol * last_price
                    if avg_vol_usd < MIN_AVG_DAILY_VOLUME_USD:
                        n_vol_fail += 1
                        continue

                momentum_score, breakdown = score_technicals(ohlcv)
                vol_surge = breakdown.get("volume_surge_ratio", 1.0)

                rows.append({
                    "ticker": ticker,
                    "asset_type": "crypto" if is_crypto else "equity",
                    "last_price": round(last_price, 4),
                    "avg_vol_usd": round(avg_vol_usd, 2),
                    "momentum_score": momentum_score,
                    "volume_surge_ratio": vol_surge,
                    "signal_breakdown": json.dumps(breakdown),
                })
            except Exception:
                _logger.exception("Stage1: error processing %s", ticker)

        df = pd.DataFrame(rows)
        if df.empty:
            _logger.warning("Stage1: no symbols passed filters")
            return df

        df = df.sort_values("momentum_score", ascending=False).head(STAGE1_TOP_N).reset_index(drop=True)

        _logger.info(
            "Stage1 complete: input=%d, price_fail=%d, vol_fail=%d, output=%d",
            len(tickers),
            n_price_fail,
            n_vol_fail,
            len(df),
        )

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False, compression="gzip")
        return df
