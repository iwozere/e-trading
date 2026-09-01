"""
P22 — yfinance daily price client (spec §2.0.7, M3, 2026-09-01).

Free, no-API-key source for the ONGOING/daily current-price ingest job
(`jobs/run_price_ingest.py`) — a different role from FMP's one-time deep
historical backfill (`ingest/fmp_backfill.py`). **Never used to backfill
history in bulk** — see the correctness trap below for why.

**Live-verified correctness trap, 2026-09-01**: yfinance's `Close` column
(even with `auto_adjust=False`, which is documented as "not adjusted") is
SPLIT-ADJUSTED RETROACTIVELY across a stock's entire history. Confirmed live
against NVDA's real 10-for-1 split (2024-06-10): fetching a wide range shows
`Close` running smoothly from ~$94 (2024-05-20) to ~$131 (2024-06-14) with no
discontinuity at the split date — but NVDA's real pre-split price in that
window was ~$940-1220 (10x). This is the SAME risk class flagged for IBKR
(`docs/Tasks.md` item 6) — confirmed here, not just suspected.
**Never call `fetch_recent_daily_bars` with a wide date range and land the
result as raw price data** — it would silently store split-adjusted values
under `p22_price_daily.close_raw`, corrupting every `as_of` before the split
once P22's own read-time adjustment (`ingest/price_archive.py`) is applied
on top (double-adjustment).

This IS safe for the narrow, incremental use this client is built for:
fetching only the last few days' bars, once per day, going forward. A bar
fetched shortly after its own trading day reflects the true as-traded price
for that day, because no FUTURE split has happened yet to retroactively
adjust it — the corruption only appears when a wide historical range is
fetched in one shot, long after intervening splits occurred.
`P22Repo.upsert_price_daily` is a second line of defense: it never rewrites
an existing `(company_id, trade_date, vendor)` row, so even an accidental
re-fetch of an old date can't silently overwrite an already-correct value
with a retroactively-adjusted one.

`Dividends`/`Stock Splits` columns ARE reliable as event flags (not price
levels) and feed `p22_corporate_action` — yfinance's split-ratio convention
(N for an N-for-1 forward split, e.g. `10.0`; a fraction for a reverse split)
already matches `ingest/price_archive.py`'s own documented convention
("4.0 for 4:1 fwd, 0.05 for 1:20 reverse"), so no translation is needed.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, cast

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.pipeline.p22_biotech_ma.config import YFINANCE_LOOKBACK_DAYS
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def fetch_recent_daily_bars(ticker: str, *, lookback_days: int = YFINANCE_LOOKBACK_DAYS) -> List[Dict[str, Any]]:
    """
    Fetch the last `lookback_days` calendar days of daily bars for `ticker` —
    OHLC, volume, dividends, stock splits. See module docstring for why this
    must stay a narrow window, never a deep historical range.

    Returns:
        JSON-serializable dicts, one per trading day in the window, oldest
        first: `{ticker, date, open, high, low, close, volume, dividends,
        stock_splits}`. Possibly empty (invalid/delisted ticker, no trading
        days in the window, or a data outage). Never raises — a single
        ticker's failure must not crash a multi-hundred-ticker daily job.
    """
    try:
        end = date.today() + timedelta(days=1)  # yfinance's `end` is exclusive of the day itself
        start = date.today() - timedelta(days=lookback_days)
        history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
    except Exception:
        _logger.exception("yfinance fetch failed for %s", ticker)
        return []

    if history.empty:
        return []

    bars: List[Dict[str, Any]] = []
    for idx, row in history.iterrows():
        trade_date = pd.Timestamp(cast(Any, idx)).date()
        bars.append({
            "ticker": ticker,
            "date": trade_date.isoformat(),
            "open": float(row["Open"]) if pd.notna(row.get("Open")) else None,
            "high": float(row["High"]) if pd.notna(row.get("High")) else None,
            "low": float(row["Low"]) if pd.notna(row.get("Low")) else None,
            "close": float(row["Close"]) if pd.notna(row.get("Close")) else None,
            "volume": int(row["Volume"]) if pd.notna(row.get("Volume")) else None,
            "dividends": float(row["Dividends"]) if pd.notna(row.get("Dividends")) else 0.0,
            "stock_splits": float(row["Stock Splits"]) if pd.notna(row.get("Stock Splits")) else 0.0,
        })

    _logger.debug("Fetched %d bar(s) for %s (lookback=%dd)", len(bars), ticker, lookback_days)
    return bars
