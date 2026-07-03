"""
P20 Kestrel — EOD ingest.

Reads OHLCV from the DataManager cache, computes technicals via TALib,
and upserts signal rows into k20_signals.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.common.technicals import calculate_technicals_talib
from src.data.data_manager import DataManager
from src.ml.pipeline.p20_kestrel.db.repos import (
    finish_job_run,
    get_active_tickers,
    start_job_run,
    upsert_signals,
)
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "ingest_eod"
_OHLCV_LOOKBACK_DAYS = 730  # 2 years


def _compute_signals_for_ticker(
    ticker: str,
    ohlcv: pd.DataFrame,
    as_of_date: date,
) -> List[Dict[str, Any]]:
    """
    Derive signal rows from an OHLCV DataFrame for a single ticker.

    Args:
        ticker: The ticker symbol.
        ohlcv: DataFrame with columns: open, high, low, close, volume.
        as_of_date: The date to attach signals to.

    Returns:
        List of signal dicts ready for k20_signals upsert.
    """
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 20:
        return []

    ohlcv = ohlcv.sort_index()
    close = ohlcv["close"]
    high = ohlcv["high"]
    volume = ohlcv["volume"]

    rows: List[Dict[str, Any]] = []

    def _signal(signal_type: str, value: Optional[float]) -> None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        rows.append({"ticker": ticker, "date": as_of_date, "signal_type": signal_type, "value": round(value, 6)})

    try:
        ta = calculate_technicals_talib(ohlcv)
        last_close = float(close.iloc[-1])

        _signal("close", last_close)
        _signal("sma_50", ta.sma_fast)
        _signal("sma_200", ta.sma_slow)
        _signal("rsi_14", ta.rsi)

        if ta.sma_fast is not None and last_close:
            _signal("price_vs_50dma", 1.0 if last_close > ta.sma_fast else 0.0)
        if ta.sma_slow is not None and last_close:
            _signal("price_vs_200dma", 1.0 if last_close > ta.sma_slow else 0.0)

        # 2-year high and drawdown
        window = min(504, len(high))
        two_yr_high = float(high.iloc[-window:].max())
        _signal("two_yr_high", two_yr_high)
        if two_yr_high and last_close:
            _signal("drawdown_from_2y_high", (last_close - two_yr_high) / two_yr_high)

        # Dollar ADV 20d
        dv = close * volume
        if len(dv) >= 20:
            _signal("adv_20d", float(dv.iloc[-20:].mean()))

        # Momentum returns
        if len(close) >= 63:
            _signal("return_3m", float(close.pct_change(63).iloc[-1]))
        if len(close) >= 126:
            _signal("return_6m", float(close.pct_change(126).iloc[-1]))

        # SMA-50 slope: rising if recent 5-day mean > prior 5-day mean
        if ta.sma_fast is not None and len(close) >= 60:
            sma_series: pd.Series = close.rolling(50).mean()  # type: ignore[assignment]
            recent_5 = float(sma_series.iloc[-5:].mean())
            prior_5 = float(sma_series.iloc[-10:-5].mean())
            _signal("sma_50_rising", 1.0 if recent_5 > prior_5 else 0.0)

    except Exception:
        _logger.exception("Error computing technicals for %s", ticker)

    return rows


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Ingest EOD data for all active universe tickers into k20_signals.

    Args:
        as_of_date: Date to ingest. Defaults to yesterday.

    Returns:
        Summary dict with tickers_processed, signals_upserted.
    """
    target_date = as_of_date or (date.today() - timedelta(days=1))
    _logger.info("Running EOD ingest for %s", target_date)
    start_job_run(_JOB_NAME, target_date)

    tickers = get_active_tickers()
    _logger.info("Processing %d universe tickers", len(tickers))

    dm = DataManager()
    end_dt = datetime.combine(target_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=_OHLCV_LOOKBACK_DAYS)

    all_signal_rows: List[Dict[str, Any]] = []
    tickers_ok = 0
    tickers_failed = 0

    try:
        try:
            ohlcv_batch = dm.get_ohlcv_batch(tickers, "1d", start_date=start_dt, end_date=end_dt)
        except Exception:
            _logger.exception("get_ohlcv_batch failed — falling back to per-ticker fetch")
            ohlcv_batch = {}

        for ticker in tickers:
            try:
                ohlcv = ohlcv_batch.get(ticker) if ohlcv_batch else None
                if ohlcv is None or ohlcv.empty:
                    ohlcv = dm.get_ohlcv(ticker, "1d", start_date=start_dt, end_date=end_dt)
                if ohlcv is None or ohlcv.empty:
                    _logger.debug("No OHLCV data for %s", ticker)
                    continue
                sig_rows = _compute_signals_for_ticker(ticker, ohlcv, target_date)
                all_signal_rows.extend(sig_rows)
                tickers_ok += 1
            except Exception:
                _logger.exception("Failed to process ticker %s", ticker)
                tickers_failed += 1

        upserted = upsert_signals(all_signal_rows)
        _logger.info(
            "EOD ingest done: %d tickers ok, %d failed, %d signals upserted",
            tickers_ok, tickers_failed, upserted,
        )
        finish_job_run(_JOB_NAME, target_date, status="ok", rows_out=upserted)
        return {"tickers_ok": tickers_ok, "tickers_failed": tickers_failed, "signals_upserted": upserted}

    except Exception as exc:
        _logger.exception("EOD ingest failed")
        finish_job_run(_JOB_NAME, target_date, status="failed", error=str(exc))
        raise
