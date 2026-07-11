"""
Data loading and merging for the P16 Taleb barbell pipeline.

OHLCV data (SPY, ^VIX) is fetched through the project's DataManager, which
handles cache-first retrieval, gap detection, and provider failover.
FRED rates and GDELT events are read directly from the shared DATA_CACHE_DIR.

Cache layout (relative to DATA_CACHE_DIR):
    ohlcv/{ticker}/1d/{YYYY}.csv.gz        <- managed by DataManager / P15
    fred/{SERIES_ID}.csv.gz                <- FRED series (index=date)
    gdelt/events/YYYYMMDD.events.csv.gz    <- GDELT v2 Events per day
    gdelt/gdelt_p16_daily.csv.gz           <- built by load_gdelt() on first run
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def _norm_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame's DatetimeIndex to date-only (midnight, tz-naive)."""
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()  # type: ignore[attr-defined]
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Project-level imports (src/ lives outside this package)
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve()
_project_root = _here.parents[5]  # src/ml/pipeline/p16_taleb/src/ -> project root
sys.path.insert(0, str(_project_root))

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR as _DEFAULT_CACHE_DIR
except ImportError:
    _DEFAULT_CACHE_DIR = "c:/data-cache"
    logging.getLogger(__name__).warning(
        "Could not import DATA_CACHE_DIR from project config; using default: %s",
        _DEFAULT_CACHE_DIR,
    )

from src.data.data_manager import DataManager

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def load_sp500(
    ticker: str = "SPY",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load S&P 500 daily OHLCV via DataManager (cache-first, gap-filling).

    Args:
        ticker: Ticker symbol, default "SPY".
        start:  ISO start date (inclusive).
        end:    ISO end date (inclusive).

    Returns:
        DataFrame with DatetimeIndex and columns: open, high, low, close, volume.
    """
    dm = DataManager()
    df = dm.get_ohlcv(
        ticker,
        "1d",
        start_date=datetime.fromisoformat(start),
        end_date=datetime.fromisoformat(end),
    )
    if df.empty:
        _logger.warning("No SP500 data returned for %s", ticker)
        return df
    df = _norm_index(df)
    _logger.info("SP500 (%s): %d rows, %s to %s", ticker, len(df), df.index[0].date(), df.index[-1].date())  # type: ignore[union-attr]
    return df


def load_vix(
    ticker: str = "^VIX",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load VIX daily close via DataManager (cache-first, gap-filling).

    Args:
        ticker: VIX ticker, default "^VIX".
        start:  ISO start date (inclusive).
        end:    ISO end date (inclusive).

    Returns:
        DataFrame with DatetimeIndex and a single column named "vix".
    """
    dm = DataManager()
    df = dm.get_ohlcv(
        ticker,
        "1d",
        start_date=datetime.fromisoformat(start),
        end_date=datetime.fromisoformat(end),
    )
    if df.empty:
        _logger.warning("No VIX data returned for %s", ticker)
        return df
    df = _norm_index(df)
    vix = df[["close"]].rename(columns={"close": "vix"})  # type: ignore[call-overload]
    _logger.info("VIX (%s): %d rows, %s to %s", ticker, len(vix), vix.index[0].date(), vix.index[-1].date())  # type: ignore[union-attr]
    return vix


def load_rates(
    cache_dir: Path,
    series_id: str = "DGS3MO",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """
    Load risk-free rate from the P15 FRED cache.

    FRED values are stored as percentages (e.g. 5.33 = 5.33%).
    This function converts to decimal (÷ 100) for Black-Scholes.

    Args:
        cache_dir: DATA_CACHE_DIR root path.
        series_id: FRED series, default "DGS3MO".
        start:     ISO start date (inclusive).
        end:       ISO end date (inclusive).

    Returns:
        DataFrame with DatetimeIndex and a single column named "rate_3m".
    """
    path = cache_dir / "fred" / f"{series_id}.csv.gz"
    if not path.exists():
        _logger.warning("FRED %s not found: %s — will use constant rate", series_id, path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
    except Exception:
        _logger.exception("Failed to read FRED %s", series_id)
        return pd.DataFrame()

    df = _norm_index(df)

    if series_id not in df.columns:
        _logger.warning("Column %s not found in FRED file; columns: %s", series_id, list(df.columns))
        return pd.DataFrame()

    rates = df[[series_id]].copy()
    rates.columns = pd.Index(["rate_3m"])
    rates["rate_3m"] = rates["rate_3m"] / 100.0
    rates = rates.loc[start:end].dropna()  # type: ignore[misc]
    _logger.info("Rates (%s): %d rows, %s to %s", series_id, len(rates), rates.index[0].date(), rates.index[-1].date())  # type: ignore[union-attr]
    return rates


def load_gdelt(
    cache_dir: Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Aggregate GDELT 2.0 Events to daily scalar signals.

    Reads per-day files from gdelt/events/YYYYMMDD.events.csv.gz, computes
    article-weighted mean AvgTone and GoldsteinScale, and caches the result
    at gdelt/gdelt_p16_daily.csv.gz (plain CSV for human readability).

    Incremental: only processes files not already in the cache.

    Args:
        cache_dir:     DATA_CACHE_DIR root path.
        force_rebuild: If True, ignore existing cache and rebuild from scratch.

    Returns:
        DataFrame with DatetimeIndex (date) and columns:
        avgtone, goldstein_scale, num_articles, num_events.
        Coverage starts 2015-02-18; rows before that date are not present.
    """
    events_dir = cache_dir / "gdelt" / "events"
    output_path = cache_dir / "gdelt" / "gdelt_p16_daily.csv.gz"

    existing = pd.DataFrame()
    if output_path.exists() and not force_rebuild:
        try:
            existing = pd.read_csv(output_path, index_col=0, parse_dates=True, compression="gzip")
            existing = _norm_index(existing)
        except Exception:
            _logger.warning("Could not read existing GDELT cache; rebuilding")
            existing = pd.DataFrame()

    last_cached = existing.index[-1].date() if not existing.empty else None  # type: ignore[union-attr]

    if not events_dir.exists():
        _logger.warning("GDELT events dir not found: %s", events_dir)
        return existing

    event_files = sorted(events_dir.glob("*.events.csv.gz"))
    if not event_files:
        _logger.warning("No GDELT events files found in %s", events_dir)
        return existing

    if last_cached is not None and not force_rebuild:
        event_files = [f for f in event_files if datetime.strptime(f.name[:8], "%Y%m%d").date() > last_cached]

    if not event_files:
        _logger.info("GDELT cache is up to date: %d days", len(existing))
        return existing

    _logger.info("Processing %d new GDELT events files...", len(event_files))

    new_rows = []
    for path in event_files:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
            if df.empty or "num_articles" not in df.columns:
                continue

            total_articles = float(df["num_articles"].sum())
            if total_articles <= 0:
                continue

            avgtone = float((df["avg_tone"] * df["num_articles"]).sum() / total_articles)
            goldstein = float((df["goldstein_scale_avg"] * df["num_articles"]).sum() / total_articles)
            file_date = datetime.strptime(path.name[:8], "%Y%m%d").date()

            new_rows.append(
                {
                    "date": pd.Timestamp(file_date),
                    "avgtone": avgtone,
                    "goldstein_scale": goldstein,
                    "num_articles": int(total_articles),
                    "num_events": int(df["num_events"].sum()),
                }
            )
        except Exception:
            _logger.warning("Failed to process GDELT file: %s", path.name)

    if not new_rows:
        _logger.info("No new GDELT rows aggregated")
        return existing

    new_df = pd.DataFrame(new_rows).set_index("date").sort_index()

    if not existing.empty:
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, compression="gzip")
    _logger.info("GDELT cache saved: %d days (added %d) -> %s", len(combined), len(new_rows), output_path)
    return combined  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_master(df: pd.DataFrame, start: str, end: str) -> None:
    """Warn if the master DataFrame has gaps longer than 5 business days."""
    if df.empty:
        _logger.warning("Master DataFrame is empty")
        return

    bday_index = pd.bdate_range(start, end)
    missing = bday_index.difference(pd.DatetimeIndex(df.index))
    if len(missing) == 0:
        _logger.info("Master validation: no missing business days")
        return

    gaps = []
    run_start = missing[0]
    prev = missing[0]
    for d in missing[1:]:
        if (d - prev).days > 3:
            gaps.append((run_start, prev))
            run_start = d
        prev = d
    gaps.append((run_start, prev))

    long_gaps = [(s, e) for s, e in gaps if (e - s).days >= 5]
    if long_gaps:
        _logger.warning(
            "Master has %d gap(s) longer than 5 business days: %s",
            len(long_gaps),
            [(str(s.date()), str(e.date())) for s, e in long_gaps],
        )
    else:
        _logger.info("Master validation: %d missing days, all short gaps", len(missing))


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------


def load_all(
    config: dict,
    cache_dir: Path | None = None,
    force_gdelt_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Load all data sources and merge into master_daily.csv.gz.

    OHLCV data is fetched via DataManager (cache-first, gap-filling, provider
    failover). FRED rates and GDELT are read directly from DATA_CACHE_DIR.

    Merge strategy: left join on S&P 500 trading days.
    Rates are forward-filled across weekends/holidays.
    GDELT is left-joined (NaN for pre-2015 dates).

    Output is written to results/p16_taleb/master_daily.csv.gz under the project root.

    Args:
        config:              Parsed config.yaml as a dict.
        cache_dir:           DATA_CACHE_DIR override; uses project default if None.
        force_gdelt_rebuild: Rebuild GDELT daily cache from scratch.

    Returns:
        Master DataFrame with DatetimeIndex and all feature-ready columns.
    """
    if cache_dir is None:
        cache_dir = Path(_DEFAULT_CACHE_DIR)

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    # --- Load individual sources ---
    sp500 = load_sp500(config["data"]["sp500_ticker"], start, end)
    if sp500.empty:
        raise RuntimeError("SP500 data is empty — check DataManager / P15 cache")

    vix = load_vix(config["data"]["vix_ticker"], start, end)
    rates = load_rates(cache_dir, config["data"]["rate_source"], start, end)
    gdelt = load_gdelt(cache_dir, force_rebuild=force_gdelt_rebuild)

    # --- Merge on SP500 trading days ---
    master = sp500.copy()

    if not vix.empty:
        master = master.join(vix, how="left")
    else:
        master["vix"] = float("nan")

    if not rates.empty:
        master = master.join(rates, how="left")
        master["rate_3m"] = master["rate_3m"].ffill()
    else:
        _logger.warning("No FRED rates loaded; using constant risk-free rate")
        master["rate_3m"] = config["pricing"]["risk_free_rate_const"]

    if not gdelt.empty:
        master = master.join(gdelt[["avgtone", "goldstein_scale"]], how="left")
    else:
        master["avgtone"] = float("nan")
        master["goldstein_scale"] = float("nan")

    # --- Validate ---
    _validate_master(master, start, end)

    # --- Save master ---
    master_path = _project_root / config["data"]["master_path"]
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(master_path, compression="gzip")

    _logger.info(
        "Master dataset: %d rows x %d cols, %s to %s -> %s",
        len(master),
        len(master.columns),
        master.index.min().date(),
        master.index.max().date(),
        master_path,  # type: ignore[union-attr]
    )
    return master
