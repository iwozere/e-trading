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
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, cast

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
_NASDAQ_API_URL = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=25000&download=true"
# nasdaqtrader.com is the official programmatic mirror — no geo-blocking.
# Two pipe-delimited files cover all US exchanges.
_NASDAQTRADER_LISTED = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
_NASDAQTRADER_OTHER = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

_logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_job(name: str, fn: Callable[[], Dict[str, Any] | None]) -> Dict[str, Any]:
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


def _job_russell3000_refresh() -> Dict[str, Any] | None:
    dl = Russell3000Downloader()
    if not dl.is_stale():
        _logger.info("russell3000: cache fresh — skipping")
        return {"skipped": True}
    df = dl.load(force=True)
    return {"rows": len(df), "source": dl.last_source_used}


def _job_aaii() -> Dict[str, Any] | None:
    path = AaiiDownloader().download(force=True)
    if path and path.exists():
        return {"path": str(path)}
    return None


def _job_fear_greed_full() -> Dict[str, Any] | None:
    path = FearGreedDownloader().download(full_rebuild=True)
    if path and path.exists():
        return {"path": str(path)}
    return None


def _job_fred_freq(dl: FredDownloader, freqs: list) -> Dict[str, Any] | None:
    return dl.update_by_freq(freqs)  # type: ignore[return-value]


def _job_fred_combined(dl: FredDownloader) -> Dict[str, Any] | None:
    df = dl.build_combined()
    return {"rows": len(df)}


def _job_nasdaq_screener() -> Dict[str, Any] | None:
    """
    Download the Nasdaq all-stocks screener CSV and save to the shared data cache.

    Tries api.nasdaq.com first (rich data: sector, industry, market cap).
    Falls back to nasdaqtrader.com pipe-delimited files (ticker + exchange only,
    no geo-blocking) when the primary API times out or returns an error.

    Destination: DATA_CACHE_DIR/universe/nasdaq_screener.csv
    P20 Kestrel universe_loader reads from the same path (via config.NASDAQ_TICKERS_CSV).

    Returns:
        Dict with rows count, file path, and source used.
    """
    df: pd.DataFrame | None = None
    source = "api.nasdaq.com"

    # --- Primary: Nasdaq JSON API (rich data, geo-blocked on some servers) ---
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nasdaq.com/",
        }
        resp = requests.get(_NASDAQ_API_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload.get("data", {}).get("rows", [])
        if rows:
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
                df = None
                _logger.warning("nasdaq_screener primary: 'symbol' field missing, falling back")
        else:
            _logger.warning("nasdaq_screener primary: empty rows, falling back")
    except Exception as exc:
        _logger.warning("nasdaq_screener primary failed (%s), falling back to nasdaqtrader.com", exc)

    # --- Fallback: nasdaqtrader.com pipe-delimited files (no geo-blocking) ---
    if df is None:
        source = "nasdaqtrader.com"
        parts: List[pd.DataFrame] = []
        for url, exch_col in [
            (_NASDAQTRADER_LISTED, None),
            (_NASDAQTRADER_OTHER, "Exchange"),
        ]:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            lines = r.text.splitlines()
            # Last line is a File Creation timestamp — drop it
            data_lines = [ln for ln in lines if not ln.startswith("File Creation")]
            part = pd.read_csv(StringIO("\n".join(data_lines)), sep="|", dtype=str)
            # nasdaqlisted.txt uses "Symbol"; otherlisted.txt uses "ACT Symbol"
            if "ACT Symbol" in part.columns:
                part = part.rename(columns={"ACT Symbol": "Symbol"})
            if exch_col and "Exchange" in part.columns:
                # Replace single-letter codes: A=NYSE American, N=NYSE, P=NYSE Arca, Z=BATS, V=IEX
                _exch_map = {"A": "NYSE MKT", "N": "NYSE", "P": "NYSE ARCA", "Z": "BATS", "V": "IEX"}
                part["Exchange"] = part["Exchange"].replace(_exch_map)
            else:
                part["Exchange"] = "NASDAQ"
            part = part.rename(columns={"Security Name": "Name"})
            # Drop file-footer rows (Symbol == "Symbol" header duplicates or NaN)
            part = part[part["Symbol"].notna() & (part["Symbol"] != "Symbol")]
            # Both files carry ETF and Test Issue Y/N flags — the P20 universe
            # wants operating companies only.
            if "ETF" in part.columns:
                part = part[part["ETF"] != "Y"]
            if "Test Issue" in part.columns:
                part = part[part["Test Issue"] != "Y"]
            parts.append(cast(pd.DataFrame, part[["Symbol", "Name", "Exchange"]]))

        merged: pd.DataFrame = cast(pd.DataFrame, pd.concat(parts, ignore_index=True))
        # Dollar-sign suffixes denote warrants/units — exclude them
        merged = cast(pd.DataFrame, merged[~merged["Symbol"].str.endswith("$", na=False)])
        df = merged.drop_duplicates(subset=["Symbol"])
        _logger.info("nasdaq_screener fallback: %d tickers from nasdaqtrader.com", len(df))

    _NASDAQ_SCREENER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_NASDAQ_SCREENER_CSV, index=False)
    _logger.info("nasdaq_screener: saved %d tickers → %s (source: %s)", len(df), _NASDAQ_SCREENER_CSV, source)
    return {"rows": len(df), "path": str(_NASDAQ_SCREENER_CSV), "source": source}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all weekly P15 jobs and emit a structured scheduler result."""
    _logger.info("=== P15 Weekly Bundle ===")

    fred_dl = FredDownloader()
    results: Dict[str, Dict[str, Any]] = {}

    results["aaii"] = _run_job("aaii", _job_aaii)
    results["fear_greed_full"] = _run_job("fear_greed_full", _job_fear_greed_full)
    results["fred_weekly"] = _run_job("fred_weekly", lambda: _job_fred_freq(fred_dl, ["weekly"]))
    results["fred_monthly"] = _run_job("fred_monthly", lambda: _job_fred_freq(fred_dl, ["monthly"]))
    results["fred_quarterly"] = _run_job("fred_quarterly", lambda: _job_fred_freq(fred_dl, ["quarterly"]))
    results["russell3000_refresh"] = _run_job("russell3000_refresh", _job_russell3000_refresh)
    results["fred_combined"] = _run_job("fred_combined", lambda: _job_fred_combined(fred_dl))
    results["nasdaq_screener"] = _run_job("nasdaq_screener", _job_nasdaq_screener)

    n_ok = sum(1 for r in results.values() if r["success"])
    n_fail = len(results) - n_ok
    _logger.info("=== P15 Weekly Bundle done: %d ok / %d failed ===", n_ok, n_fail)

    summary = {
        "success": n_fail == 0,
        "bundle": "p15_weekly",
        "jobs_ok": n_ok,
        "jobs_failed": n_fail,
        "jobs": results,
        "run_at": datetime.now(UTC).isoformat(),
    }
    print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
