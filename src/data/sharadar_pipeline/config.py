"""
Constants and locations of the Sharadar pipeline.

The data root lives outside the repository and outside the FMP data root, because the Sharadar personal licence
requires deleting every copy of the data within ``RETENTION_DAYS`` of the subscription ending (docs/Findings.md §3).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

API_BASE = "https://api.sharadar.com/v1.0"
#: Public key of the free sample: full ``tickers`` and ``descriptions``, about five years of a few large caps.
SAMPLE_KEY = "test-api-key"
API_KEY_ENV = "SHARADAR_API_KEY"
DATA_DIR_NAME = "sharadar-data"
#: Days after the end of the subscription within which all data must be deleted (terms §10).
RETENTION_DAYS = 30

#: Every table of the Bundle, in download order (small first).
BULK_TABLES = ("descriptions", "tickers", "actions", "sp500", "events", "fundamentals", "daily", "metrics",
               "stocks", "funds", "insiders", "holdings_ticker", "holdings_investor", "holdings")
#: Tables the store build needs; the others are converted as they are.
CORE_TABLES = ("tickers", "actions", "stocks", "fundamentals")
#: ``tickers.table`` values, pulled one by one for the free sample (paging needs a stable order).
TICKER_TABLES = ("stocks", "fundamentals", "funds", "insiders", "holdings_investor")
#: Tables pulled per ticker for the free sample.
SAMPLE_TICKER_TABLES = ("stocks", "actions", "fundamentals", "daily")
#: Large caps tried for the free sample; those outside the free tier are skipped.
SAMPLE_TICKERS = ("AAPL", "MSFT", "AMZN", "NVDA", "WMT", "JPM", "KO", "GOOGL", "BA", "JNJ", "PG", "HD", "DIS", "MCD",
                  "IBM", "CVX", "MRK", "V", "UNH", "CSCO")

RAW_DIR = "raw"
STORE_DIR = "store"
REPORTS_DIR = "reports"
MANIFEST_FILE = "manifest.json"

SECURITIES = "reference/securities.parquet"
SYMBOL_CHANGES = "reference/symbol_changes.parquet"
SPLITS = "events/splits.parquet"
DIVIDENDS = "events/dividends.parquet"
DELISTINGS = "events/delistings.parquet"
FUNDAMENTALS = "fundamentals/fundamentals_ar.parquet"
PRICES_DIR = "prices/1d_raw"
BUILD_INFO = "_meta/build_info.json"
#: Daily price files: one per ``security_id % PRICE_BUCKETS``.
PRICE_BUCKETS = 64


def default_data_root() -> Path:
    """Return ``DATA_CACHE_DIR/sharadar-data``."""
    from config.donotshare.donotshare import DATA_CACHE_DIR

    return Path(DATA_CACHE_DIR) / DATA_DIR_NAME


def api_key() -> Optional[str]:
    """Return the paid API key from the environment (``.env`` is loaded by ``config.donotshare``), or None."""
    try:
        importlib.import_module("config.donotshare.donotshare")  # loads .env
    except ImportError:
        pass
    return os.environ.get(API_KEY_ENV) or None
