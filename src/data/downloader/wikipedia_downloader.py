"""
Wikipedia Index Changes Downloader

Downloads and caches S&P 500 and Nasdaq-100 constituent changes.
Cache: DATA_CACHE_DIR/index_changes/YYYY-MM-DD.csv.gz (cached daily)
"""

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, List, cast

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import requests

from src.data.downloader.base_data_downloader import BaseDataDownloader
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

try:
    from config.donotshare.donotshare import DATA_CACHE_DIR
except ImportError:
    DATA_CACHE_DIR = "c:/data-cache"


class WikipediaDownloader(BaseDataDownloader):
    """
    Downloads and caches S&P 500 and Nasdaq-100 constituent changes from Wikipedia.
    """

    def __init__(self):
        super().__init__()
        self._cache_dir = Path(DATA_CACHE_DIR) / "index_changes"

    def get_provider_name(self) -> str:
        return "wikipedia"

    def get_supported_intervals(self) -> List[str]:
        return []

    def get_ohlcv(self, symbol, interval, start_date, end_date, **kwargs):
        _logger.warning("WikipediaDownloader does not provide OHLCV data")
        return pd.DataFrame()

    def download_index_changes(self, as_of_date: date) -> pd.DataFrame:
        """
        Download and merge S&P 500 and Nasdaq-100 changes from Wikipedia,
        caching the output as a gzip-compressed CSV.
        """
        from io import StringIO

        dest = self._cache_dir / f"{as_of_date.isoformat()}.csv.gz"
        headers = {"User-Agent": "Mozilla/5.0"}

        # 1. S&P 500
        r_sp = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=15)
        r_sp.raise_for_status()
        tables_sp = pd.read_html(StringIO(r_sp.text))
        df_sp = tables_sp[1]
        df_sp.columns = [f"{col[0]}_{col[1]}" if col[0] != col[1] else col[0] for col in df_sp.columns]
        df_sp = df_sp.rename(columns={"Effective Date": "date"})
        df_sp["index_name"] = "SP500"

        # 2. Nasdaq-100
        r_nd = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers, timeout=15)
        r_nd.raise_for_status()
        tables_nd = pd.read_html(StringIO(r_nd.text))
        df_nd = None
        for df in tables_nd:
            if df is None or not hasattr(df, "columns"):
                continue
            cols = [str(c).lower() for c in df.columns.tolist()]
            if any("added" in c or "removed" in c or "date" in c for c in cols):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [f"{col[0]}_{col[1]}" if col[0] != col[1] else col[0] for col in df.columns]
                if "Date" in df.columns or "Effective Date" in df.columns:
                    df_nd = df
                    break

        if df_nd is not None:
            df_nd = df_nd.rename(columns={"Date": "date"})
            df_nd["index_name"] = "NASDAQ100"
            merged = pd.concat([df_sp, df_nd], ignore_index=True)
        else:
            merged = df_sp

        def clean_date(date_str):
            try:
                clean_str = re.sub(r"\[\d+\]", "", str(date_str)).strip()
                return pd.to_datetime(clean_str).strftime("%Y-%m-%d")
            except Exception:
                return None

        merged["date"] = merged["date"].apply(clean_date)
        merged = merged.dropna(subset=["date"])
        merged = merged.sort_values(by="date", ascending=False).reset_index(drop=True)

        final_cols = [
            "date",
            "index_name",
            "Added_Ticker",
            "Added_Security",
            "Removed_Ticker",
            "Removed_Security",
            "Reason",
        ]
        for col in final_cols:
            if col not in merged.columns:
                merged[col] = None

        merged = merged[final_cols]
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(dest, index=False, compression="gzip")
        _logger.info("wikipedia_downloader: cached %d constituent changes -> %s", len(merged), dest)
        return cast(pd.DataFrame, merged)
