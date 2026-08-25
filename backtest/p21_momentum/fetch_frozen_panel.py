"""
P21 Momentum backtest — one-time frozen price-panel fetch (spec §14.2).

**Not part of runner.py's per-run path.** Run this once, manually, as an
operator step before Phase 0 begins:

    python -m backtest.p21_momentum.fetch_frozen_panel

Downloads Option A's universe (current S&P 500 constituents, applied
retroactively — see runner.py's module docstring) plus MTUM/SPY/^GSPC/^VIX
from 2005-01-01 through 2026-06-30 via ``YahooDataDownloader.get_ohlcv_batch()``
(spec §2), and writes the result to ``backtest/p21_momentum/data/prices.parquet``
with a fetch-timestamp sidecar JSON.

**Never re-run this mid-study.** Spec §14.2: "Download the full price panel
once ... write it to backtest/data/prices.parquet with a fetch timestamp,
and run every subsequent experiment against that frozen file. Otherwise two
runs of the same code a week apart produce different results and you will
not know why." ``runner.py`` and every script that consumes the panel reads
this parquet file, never ``YahooDataDownloader`` directly.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.data_downloader_factory import DataDownloaderFactory
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p21_momentum.data.universe import fetch_universe
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"
SIDECAR_PATH = DATA_DIR / "prices_fetch_metadata.json"
SECTORS_PATH = DATA_DIR / "constituents.json"

# Fetched 15 months before the official study start (2005-01-01, phase0_report.BACKTEST_START)
# so compute_signal's MIN_HISTORY=260-bar lookback (~13 months, confirmed by
# tests/test_runner.py's warmup comment) is already satisfied on day one of the evaluation
# window. Without this buffer the strategy holds zero positions for all of 2005 -- the exact
# cause of the P21 Phase 0 B3 acceptance failure (pct_months_below_12), diagnosed 2026-08-23.
BACKTEST_START = "2003-10-01"
BACKTEST_END = "2026-06-30"

_PROXY_TICKER = "MTUM"
_MARKET_TICKER = "SPY"
_REGIME_INDEX_TICKER = "^GSPC"
_REGIME_VOL_TICKER = "^VIX"


def fetch_and_freeze(start: str = BACKTEST_START, end: str = BACKTEST_END) -> Path:
    """
    Fetch the full Option A universe + benchmarks and write the frozen panel.

    Returns:
        Path to the written prices.parquet.
    """
    _logger.info("Fetching Option A universe (current S&P 500 constituents)...")
    constituents = fetch_universe()
    sector_by_ticker = {c.ticker: c.sector for c in constituents}
    tickers = sorted(
        {c.ticker for c in constituents} | {_PROXY_TICKER, _MARKET_TICKER, _REGIME_INDEX_TICKER, _REGIME_VOL_TICKER}
    )
    _logger.info("Fetching frozen panel for %d tickers, %s to %s", len(tickers), start, end)

    downloader = DataDownloaderFactory.create_downloader("yahoo")
    if downloader is None:
        raise RuntimeError("DataDownloaderFactory could not create a 'yahoo' downloader")
    downloader = cast(YahooDataDownloader, downloader)

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    panel = downloader.get_ohlcv_batch(tickers, interval="1d", start_date=start_dt, end_date=end_dt)

    frames = []
    for ticker, df in panel.items():
        if df is None or df.empty:
            _logger.warning("No data for %s — omitted from frozen panel", ticker)
            continue
        df = df.copy()
        df["ticker"] = ticker
        frames.append(df)

    if not frames:
        raise RuntimeError("No data fetched for any ticker — refusing to write an empty frozen panel")

    combined = pd.concat(frames, ignore_index=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PRICES_PATH, index=False)

    metadata = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "tickers_requested": len(tickers),
        "tickers_written": combined["ticker"].nunique(),
        "sectors_source": "src.util.tickers_list.get_sp500_constituents_with_sector (Option A, current membership)",
    }
    with SIDECAR_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Sectors are frozen alongside prices, not re-fetched at study time: a later Wikipedia
    # edit must never silently change which sector a ticker is attributed to mid-study
    # (same "snapshot once" discipline as the price panel itself, spec §14.2).
    with SECTORS_PATH.open("w", encoding="utf-8") as f:
        sectors_payload = {"fetched_at": metadata["fetched_at"], "sectors": sector_by_ticker}
        json.dump(sectors_payload, f, indent=2, sort_keys=True)

    _logger.info(
        "Wrote frozen panel: %s (%d tickers, %d rows)", PRICES_PATH, combined["ticker"].nunique(), len(combined)
    )
    return PRICES_PATH


def load_frozen_panel(path: Path = PRICES_PATH) -> Dict[str, pd.DataFrame]:
    """
    Load prices.parquet back into the {ticker: DataFrame} shape run_backtest() expects.

    Returns:
        {ticker: DataFrame[timestamp, open, high, low, close, volume]}.

    Raises:
        FileNotFoundError: if the frozen panel has not been fetched yet —
            run this module's fetch_and_freeze() (or the CLI entry point)
            first.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python -m backtest.p21_momentum.fetch_frozen_panel` once first."
        )
    combined = pd.read_parquet(path)
    panel = {}
    for ticker, group in combined.groupby("ticker"):
        panel[ticker] = group.drop(columns=["ticker"]).reset_index(drop=True)
    return panel


def load_frozen_sectors(path: Path = SECTORS_PATH) -> Dict[str, str]:
    """
    Load ``constituents.json`` back into the {ticker: sector} shape run_backtest() expects.

    Frozen alongside the price panel by fetch_and_freeze() — never re-fetched from Wikipedia
    at study time (see fetch_and_freeze()'s docstring note on this).

    Raises:
        FileNotFoundError: if the frozen panel has not been fetched yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `python -m backtest.p21_momentum.fetch_frozen_panel` once first."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload["sectors"])


if __name__ == "__main__":
    fetch_and_freeze()
