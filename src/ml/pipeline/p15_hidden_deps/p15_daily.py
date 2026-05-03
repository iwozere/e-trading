"""
P15 Pipeline — Daily Bundle Runner

Runs all daily incremental downloaders for the P15 signal research pipeline.
Self-healing: each job detects gaps back to its effective start date and
backfills up to _GAP_CAP_DAYS calendar days per run, so transient failures
repair themselves automatically over the following nights.

Scheduled via public.job_schedules:
    cron: 0 13 * * 1-5   (Mon–Fri 13:00 UTC — all previous-day data is complete)

Jobs executed (in order, failures are isolated):
    1. yfinance_prices    — Full OHLCV for all 57 P15 tickers via DataManager;
                            cached per-ticker at DATA_CACHE_DIR/ohlcv/{TICKER}/1d/YYYY.csv.gz
    2. cboe               — CBOE put/call ratio; single file fully replaced each run
                            → DATA_CACHE_DIR/cboe/cboe_putcall.csv.gz
    3. fear_greed         — CNN Fear & Greed incremental append+dedup in-place
                            → DATA_CACHE_DIR/fear_greed/cnn_fear_greed.csv.gz
    4. gdelt_gkg          — GDELT v2 GKG aggregated parquet; range fill via
                            download_gkg_range() which skips already-cached days
    5. gdelt_events       — GDELT v2 Events aggregated parquet; same range-fill pattern
    6. fred_daily         — FRED daily series incremental update;
                            per-series → DATA_CACHE_DIR/fred/{SERIES_ID}.csv.gz
    7. fred_combined      — Rebuild DATA_CACHE_DIR/fred/fred_combined.csv.gz
    8. edgar_submissions  — SEC EDGAR submissions refresh for all tracked CIKs
                            (lightweight ~KB files; used for 8-K event tracking)
    9. edgar_facts        — SEC EDGAR XBRL company facts full refresh
                            (quarterly: first weekday on or after the 15th of
                            March/May/August/November)
   10. finra_trf          — FINRA TRF short-sale volume; weekday range fill;
                            per-day → DATA_CACHE_DIR/trf/YYYY-MM-DD.csv.gz
                            (skipped silently if FINRA credentials are absent)

Gap-fill policy:
    Cutoff date : 2010-01-01 (data before this is never requested)
    Cap per run : 60 calendar days (prevents nightly timeout on long gaps)
    Fill order  : most-recent 60 days first; older gaps heal across subsequent runs

Logs: results/p15_hidden_deps/pipeline.log (RotatingFileHandler, 10 MB × 5 backups)
"""

import json
import logging
import logging.handlers
import sys
import time
from datetime import date as _Date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader.cboe_downloader import CboeDownloader
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.data.downloader.fear_greed_downloader import FearGreedDownloader
from src.data.downloader.fred_downloader import FredDownloader
from src.data.downloader.gdelt_downloader import GdeltDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CUTOFF_DATE = _Date(2010, 1, 1)        # absolute historical floor for all sources
_GAP_CAP_DAYS = 60                       # max calendar days backfilled per run

_YFINANCE_START  = _Date(2010, 1, 1)
_GDELT_V2_START  = _Date(2015, 2, 18)   # GDELT 2.0 launch date
_FINRA_TRF_START = _Date(2014, 4, 1)    # approximate Reg SHO API availability


# ---------------------------------------------------------------------------
# File logging
# ---------------------------------------------------------------------------

def _setup_file_logging() -> None:
    """Attach a rotating file handler to the root logger (pipeline.log)."""
    log_dir = PROJECT_ROOT / "results" / "p15_hidden_deps"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "pipeline.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)-40s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(handler)


# ---------------------------------------------------------------------------
# Gap-detection helpers
# ---------------------------------------------------------------------------


def _gap_window(
    watermark: Optional[_Date],
    source_start: _Date,
    yesterday: _Date,
    cap_days: int = _GAP_CAP_DAYS,
) -> Tuple[_Date, _Date]:
    """
    Compute (download_start, download_end) for a self-healing gap fill.

    Rules applied in order:
      1. download_end   = yesterday (always; ensures yesterday is never missed)
      2. natural_start  = watermark + 1 day  (or source_start if no cache)
      3. hard_floor     = yesterday - (cap_days - 1)  (prevents nightly timeouts)
      4. download_start = max(natural_start, hard_floor, source_start)

    When the cache is already current (watermark >= yesterday) the returned
    window is (yesterday, yesterday); downloaders with skip-if-cached logic
    will resolve it in O(1).

    Args:
        watermark:    Most recent date already in cache, or None if cache is empty.
        source_start: Earliest date this data source provides useful data.
        yesterday:    Target end date (UTC yesterday from the caller).
        cap_days:     Maximum calendar-day window per run (default: _GAP_CAP_DAYS).

    Returns:
        Tuple (start_date, end_date) where start_date <= end_date = yesterday.
    """
    natural_start = (watermark + timedelta(days=1)) if watermark is not None else source_start
    natural_start = max(natural_start, source_start, _CUTOFF_DATE)
    hard_floor    = yesterday - timedelta(days=cap_days - 1)
    start = max(natural_start, hard_floor)
    start = min(start, yesterday)   # safety: never overshoot yesterday
    return start, yesterday


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _check_finra_available() -> bool:
    """Return True if FINRA TRF credentials are importable."""
    try:
        import src.data.downloader.finra_trf_downloader  # noqa: F401
        return True
    except Exception:
        return False


def _yesterday_utc() -> datetime:
    """Return yesterday at midnight, timezone-naive (UTC)."""
    dt = datetime.now(timezone.utc) - timedelta(days=1)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)


_EDGAR_FACTS_MONTHS = frozenset([3, 5, 8, 11])


def _is_edgar_facts_day(today: datetime) -> bool:
    """
    Return True on the first weekday on or after the 15th of each quarter-end month.

    Trigger months: March (10-K season), May (Q1 10-Qs), August (Q2 10-Qs),
    November (Q3 10-Qs).  Handles the case where the 15th falls on a weekend by
    advancing to the following Monday (16th or 17th).
    """
    if today.month not in _EDGAR_FACTS_MONTHS:
        return False
    if today.day not in (15, 16, 17):
        return False
    weekday_of_15 = today.replace(day=15).weekday()  # 0=Mon … 6=Sun
    if weekday_of_15 < 5:
        return today.day == 15
    elif weekday_of_15 == 5:    # Saturday → trigger Monday the 17th
        return today.day == 17
    else:                       # Sunday   → trigger Monday the 16th
        return today.day == 16


def _gdelt_watermark(gdelt_dir: Path, suffix: str) -> Optional[_Date]:
    """
    Return the most recent cached date in gdelt_dir by scanning YYYYMMDD{suffix} filenames.

    Args:
        gdelt_dir: Directory containing per-day YYYYMMDD{suffix} files.
        suffix:    File suffix, e.g. ``.gkg.csv.gz`` or ``.events.csv.gz``.

    Returns:
        Most recent date present, or None if the directory is absent or empty.
    """
    if not gdelt_dir.exists():
        return None
    dates = []
    for f in gdelt_dir.glob(f"????????{suffix}"):
        try:
            dates.append(_Date(int(f.name[:4]), int(f.name[4:6]), int(f.name[6:8])))
        except ValueError:
            pass
    return max(dates) if dates else None


def _trf_watermark(trf_dir: Path) -> Optional[_Date]:
    """
    Return the most recent trading date cached in trf_dir.

    Scans for YYYY-MM-DD.csv.gz filenames and returns the maximum date found.

    Args:
        trf_dir: Directory containing per-day YYYY-MM-DD.csv.gz files.

    Returns:
        Most recent date present, or None if the directory is absent or empty.
    """
    if not trf_dir.exists():
        return None
    dates = []
    for f in trf_dir.glob("????-??-??.csv.gz"):
        try:
            dates.append(_Date.fromisoformat(f.name.removesuffix(".csv.gz")))
        except ValueError:
            pass
    return max(dates) if dates else None


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
# Job functions — each returns a plain dict or None
# ---------------------------------------------------------------------------

def _job_cboe() -> Optional[Dict[str, Any]]:
    dl = CboeDownloader()
    directory = dl.download()
    if directory is not None:
        df = dl.load()
        return {"path": str(directory), "rows": len(df)}
    return None


def _job_fear_greed() -> Optional[Dict[str, Any]]:
    # _append_recent fetches from last cached date forward, healing end-gaps naturally.
    # Deep historical gaps (> CNN API window) are covered by the weekly full rebuild.
    path = FearGreedDownloader().download(full_rebuild=False)
    if path and path.exists():
        return {"path": str(path)}
    return None


def _job_gdelt_gkg(yesterday: _Date) -> Optional[Dict[str, Any]]:
    try:
        from config.donotshare.donotshare import DATA_CACHE_DIR as _cache_root
    except ImportError:
        _cache_root = "c:/data-cache"

    gkg_dir = Path(_cache_root) / "gdelt" / "gkg"
    watermark = _gdelt_watermark(gkg_dir, ".gkg.csv.gz")
    start, end = _gap_window(watermark, _GDELT_V2_START, yesterday)
    _logger.info("gdelt_gkg: %s → %s (watermark=%s)", start, end, watermark)

    dl = GdeltDownloader()
    summary = dl.download_gkg_range(
        datetime(start.year, start.month, start.day),
        datetime(end.year,   end.month,   end.day),
    )
    return dict(summary)


def _job_gdelt_events(yesterday: _Date) -> Optional[Dict[str, Any]]:
    try:
        from config.donotshare.donotshare import DATA_CACHE_DIR as _cache_root
    except ImportError:
        _cache_root = "c:/data-cache"

    events_dir = Path(_cache_root) / "gdelt" / "events"
    watermark = _gdelt_watermark(events_dir, ".events.csv.gz")
    start, end = _gap_window(watermark, _GDELT_V2_START, yesterday)
    _logger.info("gdelt_events: %s → %s (watermark=%s)", start, end, watermark)

    dl = GdeltDownloader()
    summary = dl.download_events_range(
        datetime(start.year, start.month, start.day),
        datetime(end.year,   end.month,   end.day),
    )
    return dict(summary)


def _job_fred_daily(dl: FredDownloader) -> Optional[Dict[str, Any]]:
    return dl.update_by_freq(["daily"])  # type: ignore[return-value]


def _job_fred_combined(dl: FredDownloader) -> Optional[Dict[str, Any]]:
    df = dl.build_combined()
    return {"rows": len(df)}


# Full P15 price universe from library.md §4
_P15_TICKERS = [
    # Sector ETFs
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLB", "XLU", "XLP", "XLRE", "XLY", "XLC",
    # Sub-sector ETFs
    "KRE", "KBE", "XBI", "IBB", "ITB", "JETS", "XOP", "OIH", "SOXX", "SMH", "IYR", "XHB",
    # Broad market
    "SPY", "QQQ", "IWM", "DIA", "MDY",
    # Commodities (ETFs + continuous futures)
    "GLD", "SLV", "USO", "BNO", "PDBC", "DBA", "CPER", "UNG", "WEAT",
    "CL=F", "BZ=F", "GC=F", "NG=F",
    # Bonds
    "TLT", "IEF", "SHY", "TIP", "HYG", "LQD", "EMB", "MBB",
    # Currencies & international
    "UUP", "FXE", "FXY", "EEM", "EFA", "FXI", "EWJ", "EWG",
    "EURUSD=X", "USDJPY=X", "GBPUSD=X",
    # Volatility & sentiment
    "^VIX", "^VXN", "^SKEW", "VIXY",
]


def _job_yfinance_prices(yesterday: _Date) -> Optional[Dict[str, Any]]:
    """
    Download P15 OHLCV data for all tickers via DataManager.

    DataManager handles gap detection and caches each ticker under
    DATA_CACHE_DIR/ohlcv/{TICKER}/1d/YYYY.csv.gz.

    Args:
        yesterday: UTC date to fill up to (inclusive).

    Returns:
        Dict with symbols_ok and rows_total counts.
    """
    from src.data.data_manager import DataManager

    start_dt = datetime(_YFINANCE_START.year, _YFINANCE_START.month, _YFINANCE_START.day)
    end_dt   = datetime(yesterday.year, yesterday.month, yesterday.day)

    dm = DataManager()
    batch = dm.get_ohlcv_batch(_P15_TICKERS, "1d", start_dt, end_dt)

    symbols_ok  = sum(1 for df in batch.values() if df is not None and not df.empty)
    rows_total  = sum(len(df) for df in batch.values() if df is not None and not df.empty)
    _logger.info(
        "yfinance OHLCV: %d/%d symbols cached, %d rows total",
        symbols_ok, len(_P15_TICKERS), rows_total,
    )
    return {"symbols_ok": symbols_ok, "rows_total": rows_total}


def _job_edgar_submissions() -> Optional[Dict[str, Any]]:
    summary = EdgarDownloader().download_all_submissions(force=True)
    return summary  # type: ignore[return-value]


def _job_edgar_facts() -> Optional[Dict[str, Any]]:
    summary = EdgarDownloader().download_all_company_facts(force=True)
    return summary  # type: ignore[return-value]


def _job_finra_trf(yesterday: _Date) -> Optional[Dict[str, Any]]:
    """
    Download FINRA TRF short-sale volume for the gap window ending at yesterday.

    Iterates weekdays only (FINRA has no data for weekends/holidays). A FinraTRFDownloader
    is created per day; each saves to DATA_CACHE_DIR/trf/YYYY-MM-DD.csv.gz.

    Args:
        yesterday: UTC date to fill up to (inclusive).

    Returns:
        Dict with rows (total rows across all days) and days_downloaded.
    """
    from src.data.downloader.finra_trf_downloader import FinraTRFDownloader

    try:
        from config.donotshare.donotshare import DATA_CACHE_DIR as _cache_root
    except ImportError:
        _cache_root = "c:/data-cache"

    trf_cache_dir = Path(_cache_root) / "trf"
    watermark = _trf_watermark(trf_cache_dir)
    start, end = _gap_window(watermark, _FINRA_TRF_START, yesterday)
    _logger.info("finra_trf: %s → %s (watermark=%s)", start, end, watermark)

    total_rows = 0
    days_downloaded = 0
    current = start
    while current <= end:
        if current.weekday() < 5:   # Mon–Fri only
            dl = FinraTRFDownloader(date=current.strftime("%Y-%m-%d"))
            df = dl.run()
            if df is not None and not df.empty:
                total_rows += len(df)
                days_downloaded += 1
        current += timedelta(days=1)

    return {"rows": total_rows, "days_downloaded": days_downloaded}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all daily P15 jobs and emit a structured scheduler result."""
    _setup_file_logging()

    yesterday_dt = _yesterday_utc()
    yesterday    = yesterday_dt.date()
    _logger.info("=== P15 Daily Bundle  date=%s ===", yesterday)

    fred_dl = FredDownloader()
    results: Dict[str, Dict[str, Any]] = {}

    results["yfinance_prices"]   = _run_job("yfinance_prices",   lambda: _job_yfinance_prices(yesterday))
    results["cboe"]              = _run_job("cboe",              _job_cboe)
    results["fear_greed"]        = _run_job("fear_greed",        _job_fear_greed)
    results["gdelt_gkg"]         = _run_job("gdelt_gkg",         lambda: _job_gdelt_gkg(yesterday))
    results["gdelt_events"]      = _run_job("gdelt_events",      lambda: _job_gdelt_events(yesterday))
    results["fred_daily"]        = _run_job("fred_daily",        lambda: _job_fred_daily(fred_dl))
    results["fred_combined"]     = _run_job("fred_combined",     lambda: _job_fred_combined(fred_dl))
    results["edgar_submissions"] = _run_job("edgar_submissions", _job_edgar_submissions)
    if _is_edgar_facts_day(yesterday_dt + timedelta(days=1)):
        results["edgar_facts"]   = _run_job("edgar_facts",       _job_edgar_facts)

    if _check_finra_available():
        results["finra_trf"]     = _run_job("finra_trf",         lambda: _job_finra_trf(yesterday))
    else:
        _logger.debug("FINRA TRF skipped — credentials not available")

    n_ok   = sum(1 for r in results.values() if r["success"])
    n_fail = len(results) - n_ok
    _logger.info("=== P15 Daily Bundle done: %d ok / %d failed ===", n_ok, n_fail)

    summary = {
        "success":     n_fail == 0,
        "bundle":      "p15_daily",
        "date":        str(yesterday),
        "jobs_ok":     n_ok,
        "jobs_failed": n_fail,
        "jobs":        results,
        "run_at":      datetime.now(timezone.utc).isoformat(),
    }
    print(f"__SCHEDULER_RESULT__:{json.dumps(summary)}")


if __name__ == "__main__":
    main()
