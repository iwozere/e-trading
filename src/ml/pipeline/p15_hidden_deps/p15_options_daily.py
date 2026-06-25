"""
P15 Options Pipeline — Daily NASDAQ Universe Snapshot

Downloads yesterday's full options chain for every optionable NASDAQ-listed
security and stores the results in the shared options cache.

Scheduled cron: 0 6 * * 1-5   (06:00 UTC, Mon–Fri)
    → 7 hours before p15_daily.py (13:00 UTC), so the 10 NASDAQ-listed P15
      tickers (QQQ, TLT, IBB, SOXX, SMH, IEF, SHY, EMB, MBB, PDBC) are
      already cached when p15_daily.py runs and will be skipped there.

Ticker source:
    https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt
    Downloaded fresh on every run.  Test issues, delinquent/halted securities,
    and non-standard symbols (warrants, rights, units) are filtered out,
    leaving clean 1–5 letter tickers (~3 000–4 000 symbols).  Of those,
    ~1 500–2 000 typically have listed options; the rest return empty chains
    and are skipped quickly.

Parallelism:
    ThreadPoolExecutor with _WORKER_COUNT = 3 threads.
    Each ticker is independent (separate cache files), so no locking needed.

Cache layout (shared with p15_daily.py options job):
    DATA_CACHE_DIR/options/
      chains/{TICKER}/{YYYY-MM-DD}.csv.gz   ← full chain: type, expiration,
                                                strike, volume, OI, IV, …
      putcall/{TICKER}_putcall.csv.gz        ← growing daily P/C summary

Logs: results/p15_hidden_deps/p15_options_pipeline.log (TimedRotatingFileHandler,
      daily rotation to p15_options_pipeline.log.YYYY-MM-DD, 30-day retention)
"""

import json
import logging
import logging.handlers
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from config.donotshare.donotshare import DATA_CACHE_DIR as _cache_root
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_WORKER_COUNT = 3                    # parallel threads — tune with care
_NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
_NASDAQ_REQUEST_TIMEOUT = 20         # seconds


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_file_logging() -> None:
    """Attach a daily-rotating file handler to the root logger."""
    log_dir = PROJECT_ROOT / "results" / "p15_hidden_deps"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Rotate at midnight so each day's run lands in its own dated file
    # (p15_options_pipeline.log.YYYY-MM-DD); keep 30 days, then auto-prune.
    handler = logging.handlers.TimedRotatingFileHandler(
        log_dir / "p15_options_pipeline.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-40s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(handler)


# ---------------------------------------------------------------------------
# NASDAQ ticker list
# ---------------------------------------------------------------------------

def _load_nasdaq_tickers(date_str: str) -> List[str]:
    """
    Return filtered NASDAQ tickers for date_str, using a local cache when available.

    Cache path: DATA_CACHE_DIR/nasdaq/{date_str}.csv.gz
      - If the file exists for today, it is loaded directly (no HTTP call).
      - If not, the NASDAQ FTP file is downloaded, filtered, and saved.

    The raw file columns are preserved in the cache so the full record is
    available for inspection; only the Symbol column is returned.

    Filters applied to the raw NASDAQ file:
      - Test issues excluded          (TestIssue = Y)
      - Delinquent / halted excluded  (FinStatus in {D, H, E, Q})
      - Only clean tickers: 1–5 uppercase letters — removes warrants (W),
        rights (R), units (U), preferred shares (P), and similar derivatives
        that never have standard exchange-listed options chains.

    Args:
        date_str: ISO date string for today, e.g. ``"2026-05-15"``.

    Returns:
        Sorted list of clean NASDAQ ticker symbols.
    """
    nasdaq_dir  = Path(_cache_root) / "nasdaq"
    cache_path  = nasdaq_dir / f"{date_str}.csv.gz"

    if cache_path.exists():
        _logger.info("NASDAQ list: loading from cache %s", cache_path)
        df = pd.read_csv(cache_path, compression="gzip", dtype=str).fillna("")
        tickers = sorted(df["Symbol"].str.strip().tolist())
        _logger.info("NASDAQ list: %d symbols from cache", len(tickers))
        return tickers

    _logger.info("NASDAQ list: downloading from %s", _NASDAQ_URL)
    resp = requests.get(_NASDAQ_URL, timeout=_NASDAQ_REQUEST_TIMEOUT)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    data_lines = [l for l in lines[1:] if not l.startswith("File Creation")]

    df = pd.DataFrame(
        [l.split("|") for l in data_lines],
        columns=["Symbol", "Name", "MarketCat", "TestIssue",
                 "FinStatus", "RoundLot", "ETF", "NextShares"],
    )

    df = df[df["TestIssue"] != "Y"]
    df = df[~df["FinStatus"].isin(["D", "H", "E", "Q"])]
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$")]
    df = df.sort_values("Symbol").reset_index(drop=True)

    nasdaq_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False, compression="gzip")
    _logger.info(
        "NASDAQ list: %d symbols saved to %s", len(df), cache_path
    )

    return sorted(df["Symbol"].str.strip().tolist())


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _options_append_summary(
    putcall_dir: Path,
    ticker: str,
    row: Dict[str, Any],
) -> None:
    """
    Append one daily summary row to the per-ticker putcall CSV.gz.

    Reads the existing file (if any), appends, deduplicates on date (keeping
    the newest row), sorts chronologically, and writes back atomically.

    Args:
        putcall_dir: Directory that holds per-ticker putcall CSV files.
        ticker:      Ticker symbol used as filename stem.
        row:         Dict containing a ``date`` key plus the metrics returned
                     by ``YahooDataDownloader.compute_options_summary()``.
    """
    path = putcall_dir / f"{ticker}_putcall.csv.gz"
    new_row = pd.DataFrame([row]).set_index("date")
    new_row.index = pd.to_datetime(new_row.index)

    if path.exists():
        try:
            existing = pd.read_csv(
                path, index_col=0, parse_dates=True, compression="gzip"
            )
            combined = pd.concat([existing, new_row])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        except Exception:
            _logger.warning("%s: corrupt putcall cache — overwriting with new row", ticker)
            combined = new_row
    else:
        combined = new_row

    combined.to_csv(path, compression="gzip")


# ---------------------------------------------------------------------------
# Per-ticker worker (runs in thread pool)
# ---------------------------------------------------------------------------

def _process_ticker(
    ticker: str,
    date_str: str,
    chains_dir: Path,
    putcall_dir: Path,
) -> Dict[str, Any]:
    """
    Download and cache one ticker's options chain for date_str.

    Each call instantiates its own YahooDataDownloader so threads do not
    share yfinance session state.

    Args:
        ticker:      Ticker symbol.
        date_str:    ISO date string, e.g. ``"2026-05-14"``.
        chains_dir:  Root chains cache directory.
        putcall_dir: Root putcall cache directory.

    Returns:
        Dict with ``status``: ``"ok"`` | ``"skipped"`` | ``"empty"`` | ``"error"``
        plus optional diagnostic fields.
    """
    chain_path = chains_dir / ticker / f"{date_str}.csv.gz"

    if chain_path.exists():
        return {"ticker": ticker, "status": "skipped"}

    try:
        dl = YahooDataDownloader()
        chain_df = dl.get_options_chain_full(ticker)

        if chain_df.empty:
            _logger.debug("options: %s — no chain data", ticker)
            return {"ticker": ticker, "status": "empty"}

        chain_path.parent.mkdir(parents=True, exist_ok=True)
        chain_df.to_csv(chain_path, index=False, compression="gzip")

        summary = dl.compute_options_summary(chain_df)
        summary["date"] = pd.Timestamp(date_str)
        _options_append_summary(putcall_dir, ticker, summary)

        _logger.debug(
            "options: %s ok  pc_vol=%.3f  pc_oi=%.3f  n_exp=%d",
            ticker,
            summary.get("pc_volume_ratio") or 0.0,
            summary.get("pc_oi_ratio") or 0.0,
            summary.get("n_expirations", 0),
        )
        return {
            "ticker": ticker,
            "status": "ok",
            "pc_vol": summary.get("pc_volume_ratio"),
            "n_exp":  summary.get("n_expirations", 0),
        }

    except Exception:
        _logger.exception("options: %s — unexpected error", ticker)
        return {"ticker": ticker, "status": "error"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the NASDAQ options snapshot job and emit a structured scheduler result.

    Exit codes: 0 on success (including partial failures), non-zero only on
    catastrophic failures such as being unable to fetch the ticker list.
    """
    _setup_file_logging()

    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    date_str  = yesterday.strftime("%Y-%m-%d")
    _logger.info("=== P15 Options Daily  date=%s  workers=%d ===", date_str, _WORKER_COUNT)
    t0 = time.monotonic()

    options_dir = Path(_cache_root) / "options"
    chains_dir  = options_dir / "chains"
    putcall_dir = options_dir / "putcall"
    putcall_dir.mkdir(parents=True, exist_ok=True)

    try:
        tickers = _load_nasdaq_tickers(date_str)
    except Exception:
        _logger.exception("Failed to load NASDAQ ticker list — aborting")
        print('__SCHEDULER_RESULT__:{"success": false, "error": "nasdaq_fetch_failed"}')
        sys.exit(1)

    counters: Dict[str, int] = {"ok": 0, "skipped": 0, "empty": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=_WORKER_COUNT) as pool:
        futures = {
            pool.submit(_process_ticker, ticker, date_str, chains_dir, putcall_dir): ticker
            for ticker in tickers
        }
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            done += 1
            try:
                res = future.result()
                status = res.get("status", "error")
                counters[status] = counters.get(status, 0) + 1
            except Exception:
                counters["error"] += 1
                _logger.exception("Unhandled future error for %s", futures[future])

            if done % 100 == 0 or done == total:
                elapsed_so_far = round(time.monotonic() - t0, 0)
                _logger.info(
                    "Progress: %d/%d  ok=%d skipped=%d empty=%d error=%d  %.0fs",
                    done, total,
                    counters["ok"], counters["skipped"],
                    counters["empty"], counters["error"],
                    elapsed_so_far,
                )

    elapsed = round(time.monotonic() - t0, 1)
    _logger.info(
        "=== P15 Options Daily done: ok=%d skipped=%d empty=%d error=%d / %d  %.1fs ===",
        counters["ok"], counters["skipped"], counters["empty"],
        counters["error"], len(tickers), elapsed,
    )

    result = {
        "success":           counters["error"] == 0,
        "bundle":            "p15_options_daily",
        "date":              date_str,
        "tickers_attempted": len(tickers),
        "ok":                counters["ok"],
        "skipped":           counters["skipped"],
        "empty":             counters["empty"],
        "error":             counters["error"],
        "elapsed_s":         elapsed,
        "run_at":            datetime.now(timezone.utc).isoformat(),
    }
    print(f"__SCHEDULER_RESULT__:{json.dumps(result)}")


if __name__ == "__main__":
    main()
