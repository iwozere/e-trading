"""
Constants and locations of the EODHD pipeline.

EODHD's terms let a private user keep what was downloaded, so this data root is a kept dataset (unlike the Sharadar
research window). It still lives outside the repository: the repository is public and the data is licensed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

API_BASE = "https://eodhd.com/api"
API_KEY_NAME = "EODHD_API_KEY"
DATA_DIR_NAME = "eodhd-data"
EXCHANGE = "US"

#: Listing venues kept by default; OTC venues (PINK, OTCQB, OTCGREY, …) and mutual-fund codes (NMFQS) are left out.
MAIN_EXCHANGES = ("NYSE", "NASDAQ", "NYSE MKT", "AMEX", "BATS", "NYSE ARCA")
#: EODHD ``Type`` values downloaded by default.
DEFAULT_TYPES = ("Common Stock",)
#: EODHD ``Type`` -> vintage ``instrument_type``. EODHD does not separate ADRs from common stock.
INSTRUMENT_TYPES = {"Common Stock": "stock", "Preferred Stock": "preferred", "ETF": "etf", "FUND": "fund",
                    "Mutual Fund": "fund", "Unit": "unit", "Warrant": "warrant", "Notes": "note", "BOND": "bond", "ETC": "etc"}

#: Our security ids start here, far above the FMP vintage ids (about 20,000), so the two are never confused.
ID_OFFSET = 1_000_000
#: API calls per stock: daily bars, splits, dividends.
CALLS_PER_STOCK = 3
#: Paid plans allow 1,000 calls a minute; stay below it.
CALLS_PER_MINUTE = 900
#: Calls left unused of the daily quota so that the key stays usable for the day.
DAILY_RESERVE = 200
#: EODHD documents ``volume`` as split-adjusted while OHLC is as traded; the build converts volume back to as traded.
#: The vendor cross-check (volume ratio against the FMP vintage) verifies this on the paid key.
VOLUME_SPLIT_ADJUSTED = True

SYMBOLS_DIR = "raw/symbols"
HISTORY_DIR = "raw/history"
PROGRESS_FILE = "raw/progress.jsonl"
UNIVERSE_FILE = "universe.parquet"
IDS_FILE = "ids.json"
STORE_DIR = "store"
REPORTS_DIR = "reports"

SECURITIES = "reference/securities.parquet"
SPLITS = "events/splits.parquet"
DIVIDENDS = "events/dividends.parquet"
PRICES_DIR = "prices/1d_raw"
BUILD_INFO = "_meta/build_info.json"
#: Daily price files: one per ``security_id % PRICE_BUCKETS``.
PRICE_BUCKETS = 64


def default_data_root() -> Path:
    """Return ``DATA_CACHE_DIR/eodhd-data``."""
    from config.donotshare.donotshare import DATA_CACHE_DIR

    return Path(DATA_CACHE_DIR) / DATA_DIR_NAME


def api_key() -> Optional[str]:
    """Return the EODHD API key from the project configuration, or None."""
    from src.config.provider_config import get_api_key

    return get_api_key(API_KEY_NAME) or None
