"""
P20 Kestrel — EOD ingest.

Reads OHLCV from the DataManager cache, computes technicals via TALib,
and upserts signal rows into k20_signals.
"""

from __future__ import annotations

import functools
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.common.technicals import calculate_technicals_talib
from src.data.data_manager import DataManager
from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
finish_job_run = _kestrel.finish_job_run
get_active_tickers = _kestrel.get_active_tickers
start_job_run = _kestrel.start_job_run
upsert_signals = _kestrel.upsert_signals
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_JOB_NAME = "ingest_eod"
_OHLCV_LOOKBACK_DAYS = 730  # 2 years
_EOD_COMPUTE_WORKERS = 4    # parallel TALib compute threads (matches Pi 4 core count)


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
            _signal("return_3m", float(close.pct_change(63, fill_method=None).iloc[-1]))
        if len(close) >= 126:
            _signal("return_6m", float(close.pct_change(126, fill_method=None).iloc[-1]))

        # SMA-50 slope: rising if recent 5-day mean > prior 5-day mean
        if ta.sma_fast is not None and len(close) >= 60:
            sma_series: pd.Series = close.rolling(50).mean()  # type: ignore[assignment]
            recent_5 = float(sma_series.iloc[-5:].mean())
            prior_5 = float(sma_series.iloc[-10:-5].mean())
            _signal("sma_50_rising", 1.0 if recent_5 > prior_5 else 0.0)

    except Exception:
        _logger.exception("Error computing technicals for %s", ticker)

    return rows


def _process_ticker(
    ticker: str,
    *,
    ohlcv_batch: Dict[str, pd.DataFrame],
    dm: DataManager,
    start_dt: datetime,
    end_dt: datetime,
    target_date: date,
) -> Tuple[str, List[Dict[str, Any]], bool]:
    """
    Resolve OHLCV (batch hit or individual fallback) and compute signals.

    Designed as a module-level function so it can be called from a
    ThreadPoolExecutor worker thread.  All inputs are either read-only
    or thread-safe (DataManager cache reads hold no write lock).

    Args:
        ticker: Ticker symbol to process.
        ohlcv_batch: Pre-fetched batch keyed by ticker (may be empty dict).
        dm: Shared DataManager instance.
        start_dt: Start of the OHLCV window.
        end_dt: End of the OHLCV window.
        target_date: Signal date to attach to computed rows.

    Returns:
        (ticker, signal_rows, ok) — ok is False when no data or on error.
    """
    try:
        ohlcv = ohlcv_batch.get(ticker)
        if ohlcv is None or ohlcv.empty:
            # Batch missed this ticker (e.g. delisted mid-batch); try individually.
            ohlcv = dm.get_ohlcv(ticker, "1d", start_date=start_dt, end_date=end_dt)
        if ohlcv is None or ohlcv.empty:
            _logger.debug("No OHLCV data for %s", ticker)
            return ticker, [], False
        rows = _compute_signals_for_ticker(ticker, ohlcv, target_date)
        return ticker, rows, True
    except Exception:
        _logger.exception("Failed to process ticker %s", ticker)
        return ticker, [], False


def run(as_of_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Ingest EOD data for all active universe tickers into k20_signals.

    Phase 1 — batch OHLCV download  (single yf.download call, fast)
    Phase 2 — parallel signal compute (ThreadPoolExecutor, _EOD_COMPUTE_WORKERS)
    Phase 3 — bulk DB upsert          (single call, fast)

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
        # ── Phase 1: batch OHLCV download ────────────────────────────────
        try:
            ohlcv_batch = dm.get_ohlcv_batch(tickers, "1d", start_date=start_dt, end_date=end_dt)
        except Exception:
            _logger.exception("get_ohlcv_batch failed — falling back to per-ticker fetch")
            ohlcv_batch = {}

        # ── Phase 2: parallel TALib compute ──────────────────────────────
        worker = functools.partial(
            _process_ticker,
            ohlcv_batch=ohlcv_batch,
            dm=dm,
            start_dt=start_dt,
            end_dt=end_dt,
            target_date=target_date,
        )

        _logger.info(
            "Computing signals for %d tickers using %d workers",
            len(tickers), _EOD_COMPUTE_WORKERS,
        )
        with ThreadPoolExecutor(max_workers=_EOD_COMPUTE_WORKERS) as pool:
            for _ticker, rows, ok in pool.map(worker, tickers):
                if ok:
                    all_signal_rows.extend(rows)
                    tickers_ok += 1
                else:
                    tickers_failed += 1

        # ── Phase 3: bulk DB upsert ───────────────────────────────────────
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
