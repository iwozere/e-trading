"""
P15 Pipeline — Weekly Bundle Runner

Runs all heavy / full-rebuild downloaders for the P15 signal research pipeline.
Replaces individual per-downloader Friday DB job entries with a single run.

Scheduled via public.job_schedules:
    cron: 30 13 * * 5   (Friday 13:30 UTC, 30 min after the daily bundle)

Jobs executed (in order, failures are isolated):
    1. aaii              — AAII investor sentiment full download (XLS, ~1 MB)
    2. fear_greed        — CNN Fear & Greed full rebuild (archive + CNN API)
    3. fred_weekly       — FRED weekly series incremental update
    4. fred_monthly      — FRED monthly series incremental update
    5. fred_quarterly    — FRED quarterly series incremental update
    6. russell3000       — Russell 3000 constituent list refresh
    7. fred_combined     — Rebuild the combined FRED wide parquet
    8. nasdaq_screener   — Nasdaq all-stocks screener CSV for P20 universe loader
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from config.donotshare.donotshare import DATA_CACHE_DIR as _cache_root
from src.data.downloader.aaii_downloader import AaiiDownloader
from src.data.downloader.fear_greed_downloader import FearGreedDownloader
from src.data.downloader.fred_downloader import FredDownloader
from src.data.downloader.russell3000_downloader import Russell3000Downloader
from src.notification.logger import setup_logger

_NASDAQ_SCREENER_CSV = Path(_cache_root) / "universe" / "nasdaq_screener.csv"
_NASDAQ_API_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=25000&download=true"
)

_logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_job(name: str, fn: Callable[[], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Run a single job function, catching all exceptions.

    Args:
        name: Human-readable job name used in log messages.
        fn:   Zero-argument callable that returns an optional result dict.

    Returns:
        Dict with at least 'success' (bool) and 'elapsed_s' (float).
        Any keys returned by fn() are merged in on success.
    """
    t0 = time.monotonic()
    try:
        extra = fn() or {}
        elapsed = round(time.monotonic() - t0, 1)
        _logger.info("%-28s OK   %.1fs", name, elapsed)
        return {"success": True, "elapsed_s": elapsed, **extra}
    except Exception:
        elapsed = round(time.monotonic() - t0, 1)
        _logger.exception("%-28s FAIL %.1fs", name, elapsed)
        return {"success": False, "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# Job functions
# ---------------------------------------------------------------------------

def _job_russell3000_refresh() -> Optional[Dict[str, Any]]:
    dl = Russell3000Downloader()
    if not dl.is_stale():
        _logger.info("russell3000: cache fresh — skipping")
        return {"skipped": True}
    df = dl.load(force=True)
    return {"rows": len(df), "source": dl.last_source_used}


def _job_aaii() -> Optional[Dict[str, Any]]:
    path = AaiiDownloader().download(force=True)
    if path and path.exists():
        return {"path": str(path)}
    return None


def _job_fear_greed_full() -> Optional[Dict[str, Any]]:
    path = FearGreedDownloader().download(full_rebuild=True)
    if path and path.exists():
        return {"path": str(path)}
    return None


def _job_fred_freq(dl: FredDownloader, freqs: list) -> Optional[Dict[str, Any]]:
    return dl.update_by_freq(freqs)  # type: ignore[return-value]


def _job_fred_combined(dl: FredDownloader) -> Optional[Dict[str, Any]]:
    df = dl.build_combined()
    return {"rows": len(df)}


def _job_nasdaq_screener() -> Optional[Dict[str, Any]]:
    """
    Download the Nasdaq all-stocks screener CSV and save to the shared data cache.

    Destination: DATA_CACHE_DIR/universe/nasdaq_screener.csv
    P20 Kestrel universe_loader reads from the same path (via config.NASDAQ_TICKERS_CSV).

    Returns:
        Dict with rows count and file path.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nasdaq.com/",
    }
    resp = requests.get(_NASDAQ_API_URL, headers=headers, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data", {}).get("rows", [])
    if not rows:
        raise ValueError("Nasdaq screener API returned no rows — response: %s" % str(payload)[:200])

    df = pd.DataFrame(rows)
    rename = {
        "symbol": "Symbol",
        "name": "Name",
        "marketCap": "Market Cap",
        "sector": "Sector",
        "industry": "Industry",
        "lastsale": "Last Sale",
        "country": "Country",
        "ipoyear": "IPO Year",
        "volume": "Volume",
        "netchange": "Net Change",
        "pctchange": "% Change",
        "url": "url",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "Symbol" not in df.columns:
        raise ValueError("Nasdaq screener response missing 'symbol' field; columns: %s" % list(df.columns))

    _NASDAQ_SCREENER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_NASDAQ_SCREENER_CSV, index=False)
    _logger.info("nasdaq_screener: saved %d tickers → %s", len(df), _NASDAQ_SCREENER_CSV)
    return {"rows": len(df), "path": str(_NASDAQ_SCREENER_CSV)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all weekly P15 jobs and emit a structured scheduler result."""
    _logger.info("=== P15 Weekly Bundle ===")

    fred_dl = FredDownloader()
    results: Dict[str, Dict[str, Any]] = {}

    results["aaii"]                = _run_job("aaii",                _job_aaii)
    results["fear_greed_full"]     = _run_job("fear_greed_full",     _job_fear_greed_full)
    results["fred_weekly"]         = _run_job("fred_weekly",         lambda: _job_fred_freq(fred_dl, ["weekly"]))
    results["fred_monthly"]        = _run_job("fred_monthly",        lambda: _job_fred_freq(fred_dl, ["monthly"]))
    results["fred_quarterly"]      = _run_job("fred_quarterly",      lambda: _job_fred_freq(fred_dl, ["quarterly"]))
    results["russell3000_refresh"] = _run_job("russell3000_refresh", _job_russell3000_refresh)
    results["fred_combined"]       = _run_job("fred_combined",       lambda: _job_fred_combined(fred_dl))
    results["nasdaq_screener"]     = _run_job("nasdaq_screener",     _job_nasdaq_screener)

    n_ok   = sum(1 for r in results.values() if r["success"])
    n_fail = len(results) - n_ok
    _logger.info("=== P15 Weekly Bundle done: %d ok / %d failed ===", n_ok, n_fail)

    summary = {
        "success":      n_fail == 0,
        "bundle":       "p15_weekly",
        "jobs_ok":      n_ok,
        "jobs_failed":  n_fail,
        "jobs":         results,
        "run_at":       datetime.now(timezone.utc).isoformat(),
    }
    print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
