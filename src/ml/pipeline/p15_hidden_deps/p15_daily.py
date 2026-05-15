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
                            NOTE: CBOE CDN frozen at 2019-10-04; kept for historical ML features
    3. options_putcall    — Per-ticker daily options chain snapshot via yfinance (pre-market);
                            volume = previous session, openInterest = previous EOD.
                            Raw chains → DATA_CACHE_DIR/options/chains/{TICKER}/{YYYY-MM-DD}.csv.gz
                            Daily P/C  → DATA_CACHE_DIR/options/putcall/{TICKER}_putcall.csv.gz
                            Skips: continuous futures, forex pairs, VIX index series.
    4. fear_greed         — CNN Fear & Greed incremental append+dedup in-place
                            → DATA_CACHE_DIR/fear_greed/cnn_fear_greed.csv.gz
    4. gdelt_gkg          — GDELT v2 GKG aggregated parquet; range fill via
                            download_gkg_range() which skips already-cached days
    5. gdelt_events       — GDELT v2 Events aggregated parquet; same range-fill pattern
    6. fred_daily         — FRED daily series incremental update;
                            per-series → DATA_CACHE_DIR/fred/{SERIES_ID}.csv.gz
    7. fred_combined      — Rebuild DATA_CACHE_DIR/fred/fred_combined.csv.gz
    8. edgar_submissions  — SEC EDGAR submissions refresh for Tier-1 watchlist CIKs
                            (~160 sector-bellwether stocks; lightweight ~KB files;
                            used for 8-K event tracking)
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
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from config.donotshare.donotshare import DATA_CACHE_DIR as _cache_root
from src.data.downloader.finra_trf_downloader import FinraTRFDownloader
from src.data.downloader.cboe_downloader import CboeDownloader
from src.data.downloader.edgar_downloader import EdgarDownloader
from src.data.downloader.fear_greed_downloader import FearGreedDownloader
from src.data.downloader.fred_downloader import FredDownloader
from src.data.downloader.gdelt_downloader import GdeltDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
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
_P15_TICKERS: List[str] = [
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

# Options-eligible subset of _P15_TICKERS: excludes continuous futures (CL=F …),
# forex pairs (EURUSD=X …), and VIX index series (^VIX, ^VXN, ^SKEW) because
# yfinance does not provide standard exchange-listed options chains for those.
_OPTIONS_TICKERS: List[str] = [
    t for t in _P15_TICKERS
    if not (t.endswith("=F") or t.endswith("=X") or t.startswith("^"))
]

# Tier-1 EDGAR watchlist: sector-bellwether stocks whose 8-K filings move sector ETFs.
# Submissions are refreshed daily for these ~160 companies only (not the full SEC universe).
# Organized by the P15 sector ETF each company primarily drives.
_TIER1_WATCHLIST: Dict[str, List[str]] = {

    # ── XLF: Financials ─────────────────────────────────────────────────────
    # BRK-B (11.75%), JPM (11.23%), V (7.20%), MA (5.76%), BAC (4.58%),
    # GS (3.59%), WFC (3.49%), C (2.81%), MS (2.80%), AXP (2.29%)
    # + key sub-sector representatives: BLK (asset mgmt), SPGI (ratings), CME (exchange)
    "XLF": [
        "BRK-B",  # Berkshire — systemic bellwether
        "JPM",    # largest US bank, earnings move entire sector
        "V",      # payments — rate-insensitive growth within financials
        "MA",     # payments duopoly
        "BAC",    # rate-sensitive commercial bank
        "GS",     # capital markets / investment banking cycle
        "WFC",    # consumer banking, mortgage exposure
        "C",      # global wholesale bank, EM exposure
        "MS",     # wealth management + capital markets
        "AXP",    # consumer credit, spending indicator
        "BLK",    # largest asset manager, AUM as market sentiment proxy
        "SPGI",   # S&P Global — credit ratings, financial data
        "CME",    # exchange — volatility → volume → revenue
        "USB",    # US Bancorp — large regional bank proxy
        "PGR",    # Progressive — insurance cycle indicator
    ],

    # ── XLE: Energy ──────────────────────────────────────────────────────────
    # XOM (22%), CVX (17%), COP (7%), WMB (4.4%), SLB (4.5%),
    # EOG (4%), PSX (3.8%), VLO (3.7%), KMI (3.7%), MPC (3.6%)
    # + oilfield services (HAL), E&P pure-play (OXY), midstream (OKE)
    "XLE": [
        "XOM",    # largest US oil major, global integrated
        "CVX",    # second major, Permian + international
        "COP",    # pure E&P, price-sensitive upstream
        "WMB",    # Williams Cos — natural gas midstream/pipeline
        "SLB",    # oilfield services leader, capex cycle proxy
        "EOG",    # Permian pure-play E&P, shale bellwether
        "PSX",    # Phillips 66 — refining margin indicator
        "VLO",    # Valero — crack spread proxy
        "KMI",    # Kinder Morgan — gas pipeline infrastructure
        "MPC",    # Marathon Petroleum — refining
        "OXY",    # Occidental — Buffett holding, Permian
        "HAL",    # Halliburton — oilfield services #2
        "BKR",    # Baker Hughes — services + LNG equipment
        "OKE",    # ONEOK — gas gathering/processing
        "FANG",   # Diamondback Energy — Permian pure-play
    ],

    # ── XLK: Technology ──────────────────────────────────────────────────────
    # NVDA (14.78%), AAPL (12.14%), MSFT (9.23%), AVGO (6.03%),
    # MU (4.32%) — weights shift frequently due to AI rally
    # + enterprise software (CRM, ORCL), semis (AMD, QCOM, TXN), infra (ACN)
    "XLK": [
        "NVDA",   # AI/GPU — dominant signal for AI capex cycle
        "AAPL",   # largest market cap, consumer hardware + services
        "MSFT",   # cloud (Azure) + enterprise software
        "AVGO",   # Broadcom — networking chips, AI accelerators
        "MU",     # Micron — memory cycle, leading indicator for semis
        "AMD",    # CPU/GPU competitor to Intel/NVDA
        "ORCL",   # Oracle — enterprise cloud, database
        "CRM",    # Salesforce — enterprise SaaS spending indicator
        "ACN",    # Accenture — IT services, consulting capex
        "QCOM",   # Qualcomm — mobile chips, handset cycle
        "TXN",    # Texas Instruments — analog semis, industrial demand
        "AMAT",   # Applied Materials — semiconductor equipment
        "PLTR",   # Palantir — government AI/defense tech
        "NOW",    # ServiceNow — enterprise workflow automation
        "PANW",   # Palo Alto Networks — cybersecurity cycle
    ],

    # ── XLV: Health Care ─────────────────────────────────────────────────────
    # LLY (14%), JNJ (10%), ABBV (7%), UNH (6%), MRK (5%),
    # AMGN (3.5%), TMO (3.5%), ISRG (3.2%), ABT (3.1%), GILD (3.1%)
    "XLV": [
        "LLY",    # Eli Lilly — GLP-1/obesity drugs, dominant weight
        "JNJ",    # Johnson & Johnson — diversified, defensive
        "ABBV",   # AbbVie — Humira/Skyrizi, income proxy
        "UNH",    # UnitedHealth — managed care, insurance cycle
        "MRK",    # Merck — Keytruda oncology, vaccines
        "AMGN",   # Amgen — biotech large-cap
        "TMO",    # Thermo Fisher — lab equipment, biotech capex proxy
        "ISRG",   # Intuitive Surgical — robotic surgery, procedure volume
        "ABT",    # Abbott Labs — devices + diagnostics
        "GILD",   # Gilead — HIV/oncology, cash flow story
        "BSX",    # Boston Scientific — cardiac devices
        "SYK",    # Stryker — orthopedic implants, elective surgery
        "BMY",    # Bristol-Myers Squibb — oncology pipeline
        "CVS",    # CVS Health — PBM + retail pharmacy + insurance
        "HCA",    # HCA Healthcare — hospital utilization indicator
    ],

    # ── XLI: Industrials ─────────────────────────────────────────────────────
    # CAT (7%), GE (6.6%), RTX (5%), GEV (4.4%), BA (3.3%),
    # UBER (3%), UNP (2.9%), DE (2.9%), HON (2.9%), ETN (2.6%)
    "XLI": [
        "CAT",    # Caterpillar — global construction/mining capex
        "GE",     # GE Aerospace — commercial aviation cycle
        "RTX",    # RTX Corp — defense + aircraft engines
        "GEV",    # GE Vernova — power generation, energy transition
        "BA",     # Boeing — commercial/defense aviation
        "UNP",    # Union Pacific — rail freight, economic activity
        "DE",     # Deere — agricultural capex, commodity cycle
        "HON",    # Honeywell — industrial conglomerate
        "ETN",    # Eaton — electrical components, data center power
        "LMT",    # Lockheed Martin — defense budget indicator
        "NOC",    # Northrop Grumman — defense
        "UPS",    # UPS — parcel volume, consumer + B2B indicator
        "EMR",    # Emerson Electric — automation/process control
        "CSX",    # CSX — eastern rail freight
        "PWR",    # Quanta Services — grid infrastructure buildout
    ],

    # ── XLB: Materials ───────────────────────────────────────────────────────
    # LIN (17%), NEM (7.3%), SHW (6.2%), FCX (5.3%), CRH (5%),
    # ECL (4.8%), APD (4.7%), CTVA (4.8%), MLM (4.4%), NUE (3.5%)
    "XLB": [
        "LIN",    # Linde — industrial gases, largest weight
        "NEM",    # Newmont — gold mining, safe-haven proxy
        "SHW",    # Sherwin-Williams — housing/construction indicator
        "FCX",    # Freeport-McMoRan — copper, China demand proxy
        "CRH",    # CRH — cement/construction materials
        "APD",    # Air Products — industrial gases, hydrogen
        "ECL",    # Ecolab — water treatment, specialty chemicals
        "CTVA",   # Corteva — agricultural chemicals/seeds
        "MLM",    # Martin Marietta — aggregates, construction
        "NUE",    # Nucor — steel, manufacturing demand
        "DOW",    # Dow Inc — commodity chemicals
        "PPG",    # PPG Industries — coatings, auto/industrial
        "VMC",    # Vulcan Materials — aggregates, infrastructure
        "ALB",    # Albemarle — lithium, EV battery supply chain
        "CF",     # CF Industries — nitrogen fertilizers, nat gas spread
    ],

    # ── XLU: Utilities ───────────────────────────────────────────────────────
    # NEE (13.8%), SO (7.3%), DUK (6.9%), CEG (6.5%), AEP (5.1%),
    # SRE (4.2%), VST (3.8%), D (3.7%), XEL (3.4%), EXC (3.4%)
    "XLU": [
        "NEE",    # NextEra — renewable energy leader, rate-sensitive
        "SO",     # Southern Company — regulated, nuclear
        "DUK",    # Duke Energy — large regulated utility
        "CEG",    # Constellation Energy — nuclear, AI power demand
        "AEP",    # American Electric Power — transmission grid
        "SRE",    # Sempra Energy — gas utility + LNG export
        "VST",    # Vistra — power generation, merchant energy
        "D",      # Dominion Energy — regulated, rate-sensitive
        "XEL",    # Xcel Energy — renewables transition
        "EXC",    # Exelon — nuclear + regulated distribution
        "PCG",    # PG&E — California utility, wildfire risk proxy
        "ED",     # Consolidated Edison — NYC utility, stable
        "EIX",    # Edison International — California utility
        "ETR",    # Entergy — nuclear + regulated South
        "FE",     # FirstEnergy — mid-Atlantic regulated
    ],

    # ── XLP: Consumer Staples ────────────────────────────────────────────────
    # WMT (11.9%), COST (9.4%), PG (7.4%), KO (6.5%), PM (5.5%),
    # CL (4.75%), PEP (4.7%), MO (4.7%), MDLZ (4.4%), MNST (3.7%)
    "XLP": [
        "WMT",    # Walmart — consumer spending bellwether
        "COST",   # Costco — membership model, affluent consumer
        "PG",     # Procter & Gamble — household staples pricing power
        "KO",     # Coca-Cola — global beverage, defensive
        "PM",     # Philip Morris — international tobacco, EM exposure
        "PEP",    # PepsiCo — beverages + snacks (Frito-Lay)
        "MO",     # Altria — domestic tobacco, high yield
        "CL",     # Colgate-Palmolive — oral/personal care
        "MDLZ",   # Mondelez — global snack foods
        "MNST",   # Monster Beverage — energy drinks growth story
        "TGT",    # Target — discretionary/staples overlap
        "KR",     # Kroger — grocery chain, food inflation proxy
        "GIS",    # General Mills — packaged food
        "KHC",    # Kraft Heinz — packaged food, pricing pressure
        "STZ",    # Constellation Brands — beer/wine/spirits
    ],

    # ── XLRE: Real Estate ────────────────────────────────────────────────────
    # WELL (10.3%), PLD (9.1%), EQIX (7.25%), AMT (5.8%), DLR (4.8%),
    # SPG (4.6%), CBRE (4.5%), VTR (4.4%), O (4.4%), PSA (3.5%)
    "XLRE": [
        "WELL",   # Welltower — healthcare REIT, aging demographics
        "PLD",    # Prologis — industrial/warehouse REIT, e-commerce
        "EQIX",   # Equinix — data center REIT, AI infrastructure
        "AMT",    # American Tower — cell tower REIT
        "DLR",    # Digital Realty — data center REIT
        "SPG",    # Simon Property Group — retail/mall REIT
        "CBRE",   # CBRE Group — CRE services, transaction volume
        "VTR",    # Ventas — senior housing + medical office
        "O",      # Realty Income — net lease, monthly dividend
        "PSA",    # Public Storage — self-storage
        "CCI",    # Crown Castle — cell tower infrastructure
        "EQR",    # Equity Residential — apartment REIT, rents
        "AVB",    # AvalonBay — apartment REIT, coastal markets
        "ARE",    # Alexandria RE — life science lab REIT
        "VICI",   # VICI Properties — gaming/entertainment REIT
    ],

    # ── XLY: Consumer Discretionary ─────────────────────────────────────────
    # AMZN (21-28% depending on date), TSLA (15-20%), HD (5-7%),
    # TJX (4%), MCD (4%), BKNG (4%), LOW (3%), SBUX (2.3%), ORLY (2.1%)
    "XLY": [
        "AMZN",   # Amazon — e-commerce + AWS, dominant weight
        "TSLA",   # Tesla — EV cycle, volatile weight
        "HD",     # Home Depot — housing/renovation indicator
        "MCD",    # McDonald's — consumer spending health
        "BKNG",   # Booking Holdings — travel demand
        "TJX",    # TJX Companies — value retail, trade-down indicator
        "LOW",    # Lowe's — housing/DIY, rate sensitivity
        "SBUX",   # Starbucks — discretionary spending indicator
        "ORLY",   # O'Reilly Auto — auto aftermarket, aging fleet
        "NKE",    # Nike — global consumer brand, China exposure
        "CMG",    # Chipotle — fast casual restaurant cycle
        "ABNB",   # Airbnb — short-term rental, travel
        "LVS",    # Las Vegas Sands — Macau/Singapore gaming
        "GM",     # General Motors — auto cycle, EV transition
        "F",      # Ford — auto + EV, labor cost indicator
    ],

    # ── XLC: Communication Services ──────────────────────────────────────────
    # META (14-23%), GOOGL (8.6%), GOOG (6.9%), DIS (4.6%), CMCSA (4.5%),
    # NFLX (5.9%), T (5%), VZ (4.7%), TMUS (4.6%), WBD (4.7%)
    "XLC": [
        "META",   # Meta — digital advertising, AI investment cycle
        "GOOGL",  # Alphabet A — search + cloud + YouTube
        "GOOG",   # Alphabet C — same economic exposure
        "NFLX",   # Netflix — streaming, subscriber growth
        "T",      # AT&T — telecom, high yield, capex
        "VZ",     # Verizon — wireless, dividend yield proxy
        "TMUS",   # T-Mobile — wireless subscriber growth
        "DIS",    # Disney — streaming + parks + content
        "CMCSA",  # Comcast — cable + NBC + streaming
        "WBD",    # Warner Bros Discovery — media/streaming
        "EA",     # Electronic Arts — gaming cycle
        "TTWO",   # Take-Two Interactive — GTA cycle
        "LYV",    # Live Nation — live events, concert demand
        "CHTR",   # Charter Communications — cable broadband
        "OMC",    # Omnicom — advertising industry indicator
    ],
}

# Flat sorted list of all Tier-1 tickers for CIK resolution
_EDGAR_WATCHLIST_TICKERS: List[str] = sorted(set(
    ticker
    for tickers in _TIER1_WATCHLIST.values()
    for ticker in tickers
))


# ---------------------------------------------------------------------------
# Options put/call helpers
# ---------------------------------------------------------------------------

def _options_append_summary(putcall_dir: Path, ticker: str, row: Dict[str, Any]) -> None:
    """
    Append one daily summary row to the per-ticker putcall CSV.gz, deduplicating on date.

    Creates the file if it does not exist.

    Args:
        putcall_dir: Directory for putcall CSV files (DATA_CACHE_DIR/options/putcall).
        ticker:      Ticker symbol.
        row:         Dict containing 'date' plus metric keys from compute_options_summary().
    """
    path = putcall_dir / f"{ticker}_putcall.csv.gz"
    new_row = pd.DataFrame([row]).set_index("date")
    new_row.index = pd.to_datetime(new_row.index)

    if path.exists():
        try:
            existing = pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
            combined = pd.concat([existing, new_row])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        except Exception:
            _logger.warning("%s: corrupt putcall cache — overwriting with new row", ticker)
            combined = new_row
    else:
        combined = new_row

    combined.to_csv(path, compression="gzip")


def _job_options_putcall(yesterday: _Date) -> Optional[Dict[str, Any]]:
    """
    Snapshot yesterday's options chain for all options-eligible P15 tickers.

    Called at 13:00 UTC (9 AM ET, before market open) so that yfinance reports:
      - volume        = previous session's traded contracts
      - openInterest  = previous EOD settlement

    Per-ticker cache layout:
      DATA_CACHE_DIR/options/chains/{TICKER}/{YYYY-MM-DD}.csv.gz   ← raw full chain
      DATA_CACHE_DIR/options/putcall/{TICKER}_putcall.csv.gz       ← growing daily summary

    Args:
        yesterday: UTC date representing the trading session to capture.

    Returns:
        Dict with symbols_ok, symbols_skipped, symbols_empty, date.
    """
    options_dir  = Path(_cache_root) / "options"
    chains_dir   = options_dir / "chains"
    putcall_dir  = options_dir / "putcall"
    putcall_dir.mkdir(parents=True, exist_ok=True)

    dl = YahooDataDownloader()
    date_str = yesterday.strftime("%Y-%m-%d")
    symbols_ok = symbols_skipped = symbols_empty = 0

    for ticker in _OPTIONS_TICKERS:
        chain_path = chains_dir / ticker / f"{date_str}.csv.gz"

        if chain_path.exists():
            symbols_skipped += 1
            _logger.debug("options: %s %s already cached — skipping", ticker, date_str)
            continue

        chain_df = dl.get_options_chain_full(ticker)

        if chain_df.empty:
            symbols_empty += 1
            _logger.warning("options: %s returned empty chain — skipping", ticker)
            time.sleep(0.3)
            continue

        chain_path.parent.mkdir(parents=True, exist_ok=True)
        chain_df.to_csv(chain_path, index=False, compression="gzip")

        summary = dl.compute_options_summary(chain_df)
        summary["date"] = pd.Timestamp(yesterday)
        _options_append_summary(putcall_dir, ticker, summary)

        symbols_ok += 1
        _logger.debug(
            "options: %s %s  pc_vol=%.3f  pc_oi=%.3f  n_exp=%d",
            ticker, date_str,
            summary.get("pc_volume_ratio") or 0,
            summary.get("pc_oi_ratio") or 0,
            summary.get("n_expirations", 0),
        )
        time.sleep(0.3)

    _logger.info(
        "options_putcall %s: ok=%d  skipped=%d  empty=%d / %d tickers",
        date_str, symbols_ok, symbols_skipped, symbols_empty, len(_OPTIONS_TICKERS),
    )
    return {
        "symbols_ok":      symbols_ok,
        "symbols_skipped": symbols_skipped,
        "symbols_empty":   symbols_empty,
        "date":            date_str,
    }


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
    dl = EdgarDownloader()
    cik_list = dl.resolve_tickers_to_ciks(_EDGAR_WATCHLIST_TICKERS)
    _logger.info(
        "edgar_submissions: %d watchlist tickers → %d CIKs",
        len(_EDGAR_WATCHLIST_TICKERS), len(cik_list),
    )
    summary = dl.download_all_submissions(cik_list=cik_list, force=True)
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

    trf_cache_dir = Path(_cache_root) / "trf"
    watermark = _trf_watermark(trf_cache_dir)
    start, end = _gap_window(watermark, _FINRA_TRF_START, yesterday)
    _logger.info("finra_trf: %s → %s (watermark=%s)", start, end, watermark)

    total_rows = 0
    days_downloaded = 0
    current = start
    while current <= end:
        if current.weekday() < 5:   # Mon–Fri only
            dl = FinraTRFDownloader(date=current.strftime("%Y-%m-%d"), fetch_yfinance_data=False)
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
    results["options_putcall"]   = _run_job("options_putcall",   lambda: _job_options_putcall(yesterday))
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
