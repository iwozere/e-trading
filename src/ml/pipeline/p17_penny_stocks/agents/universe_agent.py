"""
P17 Universe Agent

Downloads the full NASDAQ/NYSE-American universe and applies P17 hard filters:
price range, market cap, float, liquidity, and exchange.

Uses yfinance.Ticker.info for per-ticker market snapshot data with
checkpoint/resume support to survive interruptions on large universes.
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p17_penny_stocks.config import P17FilterConfig
from src.ml.pipeline.shared.config import UniverseConfig
from src.ml.pipeline.shared.universe_downloader import NasdaqUniverseDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.cache.fundamentals_cache import FundamentalsCache

_logger = setup_logger(__name__)

_FUNDAMENTALS_CACHE_TTL_DAYS = 1

# yfinance exchange code → normalised exchange name
_EXCHANGE_MAP: Dict[str, str] = {
    "NMS": "NASDAQ",        # NASDAQ Global Select Market
    "NGM": "NASDAQ",        # NASDAQ Global Market
    "NCM": "NASDAQ",        # NASDAQ Capital Market
    "ASE": "NYSE_AMERICAN", # NYSE American (formerly AMEX)
    "NYQ": "NYSE",
    "PCX": "NYSE_ARCA",
    "PNK": "OTC",
    "OTC": "OTC",
    "GREY": "OTC",
    "BATS": "BATS",
}
_ALLOWED_EXCHANGES = {"NASDAQ", "NYSE_AMERICAN"}


def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
    """Coerce value to float, returning default on failure or NaN."""
    try:
        v = float(value)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


class UniverseAgent:
    """
    Stage 1: Download NASDAQ universe and apply P17 hard filters.

    Returns a DataFrame with one row per passing ticker and columns:
    ticker, company_name, exchange, sector, industry, price, market_cap,
    float_shares, shares_outstanding, avg_volume, volume, short_ratio,
    short_pct_float.
    """

    def __init__(
        self,
        config: P17FilterConfig,
        universe_config: UniverseConfig,
        results_dir: Path,
        target_date: str,
    ) -> None:
        self.config = config
        self.results_dir = results_dir
        self.target_date = target_date

        self._universe_downloader = NasdaqUniverseDownloader(
            universe_config,
            results_dir=results_dir,
            target_date=target_date,
        )
        self._downloader = YahooDataDownloader()
        self._fundamentals_cache = FundamentalsCache()

    def run(
        self,
        force_refresh: bool = False,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run universe download and hard-filter pass.

        Args:
            force_refresh: Bypass all caches.
            tickers: Optional explicit list (skips NASDAQ FTP download).

        Returns:
            Filtered DataFrame ready for market_agent enrichment.
        """
        if tickers:
            _logger.info("Using %d explicitly provided tickers", len(tickers))
            all_tickers = tickers
        else:
            all_tickers = self._universe_downloader.download_universe(force_refresh)

        if not all_tickers:
            _logger.warning("Empty universe — aborting universe agent")
            return pd.DataFrame()

        _logger.info("Fetching market snapshot for %d tickers", len(all_tickers))

        checkpoint_path = self.results_dir / "00_universe_checkpoint.csv"
        rows = self._fetch_with_checkpoint(all_tickers, checkpoint_path, force_refresh)

        if not rows:
            _logger.warning("No market data returned for any ticker")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        _logger.info("Snapshot fetched for %d tickers; applying hard filters", len(df))

        df = self._apply_hard_filters(df)

        out = self.results_dir / "01_universe_filtered.csv"
        df.to_csv(out, index=False)
        _logger.info("Universe agent complete: %d tickers passed → %s", len(df), out)

        return df

    # ── Internal: fetch with checkpoint ────────────────────────────────────

    def _fetch_with_checkpoint(
        self,
        tickers: List[str],
        checkpoint_path: Path,
        force_refresh: bool,
    ) -> List[dict]:
        processed: Dict[str, dict] = {}

        if not force_refresh and checkpoint_path.exists():
            try:
                cp_df = pd.read_csv(checkpoint_path)
                processed = {r["ticker"]: r for r in cp_df.to_dict("records")}
                _logger.info("Checkpoint loaded: %d tickers already processed", len(processed))
            except Exception:
                _logger.warning("Could not load checkpoint — starting fresh")

        remaining = [t for t in tickers if t not in processed]
        _logger.info("%d tickers remaining after checkpoint", len(remaining))

        chunk_size = self.config.ohlcv_chunk_size
        n_chunks = (len(remaining) + chunk_size - 1) // chunk_size

        for i in range(0, len(remaining), chunk_size):
            chunk = remaining[i : i + chunk_size]
            chunk_num = i // chunk_size + 1
            _logger.info("Chunk %d/%d — fetching %d tickers", chunk_num, n_chunks, len(chunk))

            chunk_results = self._fetch_chunk(chunk)
            processed.update({r["ticker"]: r for r in chunk_results})

            try:
                pd.DataFrame(list(processed.values())).to_csv(checkpoint_path, index=False)
            except Exception:
                _logger.warning("Could not save checkpoint after chunk %d", chunk_num)

        return list(processed.values())

    def _fetch_chunk(self, tickers: List[str]) -> List[dict]:
        results: List[dict] = []
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(self._fetch_ticker_info, t): t for t in tickers}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        return results

    def _fetch_ticker_info(self, ticker: str, max_retries: int = 3) -> Optional[dict]:
        cache_meta = self._fundamentals_cache.find_latest_json(
            ticker, provider="yahoo", max_age_days=_FUNDAMENTALS_CACHE_TTL_DAYS
        )
        if cache_meta:
            cached = self._fundamentals_cache.read_json(cache_meta.file_path)
            if cached:
                return self._fundamentals_to_universe_record(ticker, cached)

        for attempt in range(max_retries):
            try:
                f = self._downloader.get_fundamentals(ticker)
                if f is None:
                    return None
                f_dict = dataclasses.asdict(f)
                self._fundamentals_cache.write_json(ticker, "yahoo", f_dict)
                return self._fundamentals_to_universe_record(ticker, f_dict)
            except Exception as e:
                err = str(e)
                if "Too Many Requests" in err or "Rate limited" in err:
                    delay = 2 ** attempt
                    _logger.debug("Rate limited on %s (attempt %d/%d), retrying in %ds", ticker, attempt + 1, max_retries, delay)
                    time.sleep(delay)
                else:
                    _logger.debug("Failed to fetch info for %s: %s", ticker, e)
                    return None

        _logger.debug("Giving up on %s after %d attempts (rate limited)", ticker, max_retries)
        return None

    def _fundamentals_to_universe_record(self, ticker: str, f: dict) -> Optional[dict]:
        """Map a cached Fundamentals dict to the universe record format, applying equity/price filters."""
        if f.get("quote_type") not in ("EQUITY", None):
            return None
        price = _safe_float(f.get("current_price"))
        if price is None or price <= 0:
            return None
        return {
            "ticker": ticker,
            "company_name": f.get("company_name") or "",
            "exchange": f.get("exchange") or "",
            "sector": f.get("sector") or "",
            "industry": f.get("industry") or "",
            "price": price,
            "market_cap": _safe_float(f.get("market_cap"), 0.0),
            "float_shares": _safe_float(f.get("float_shares"), 0.0),
            "shares_outstanding": _safe_float(f.get("shares_outstanding"), 0.0),
            "avg_volume": _safe_float(f.get("avg_volume"), 0.0),
            "volume": _safe_float(f.get("volume"), 0.0),
            "short_ratio": _safe_float(f.get("short_ratio")),
            "short_pct_float": _safe_float(f.get("short_pct_float")),
            "high_52w": _safe_float(f.get("fifty_two_week_high"), 0.0),
            "low_52w": _safe_float(f.get("fifty_two_week_low"), 0.0),
            "institutional_pct": _safe_float(f.get("institutional_pct")),
        }

    # ── Internal: hard filters ─────────────────────────────────────────────

    def _apply_hard_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)

        # Exchange
        df = df.copy()
        df["exchange_norm"] = df["exchange"].map(
            lambda x: _EXCHANGE_MAP.get(str(x), "OTHER")
        )
        df = df.loc[df["exchange_norm"].isin(list(_ALLOWED_EXCHANGES))].copy()
        _logger.info("Exchange filter: %d → %d", n0, len(df))

        # Price range
        n = len(df)
        df = df.loc[
            (df["price"] >= self.config.min_price) & (df["price"] <= self.config.max_price)
        ].copy()
        _logger.info("Price $%.2f–$%.2f: %d → %d", self.config.min_price, self.config.max_price, n, len(df))

        # Market cap
        n = len(df)
        df = df.loc[
            (df["market_cap"] >= self.config.min_market_cap)
            & (df["market_cap"] <= self.config.max_market_cap)
        ].copy()
        _logger.info("Market cap $%dM–$%dB: %d → %d",
                     self.config.min_market_cap // 1_000_000,
                     self.config.max_market_cap // 1_000_000_000,
                     n, len(df))

        # Float
        n = len(df)
        df = df.loc[
            (df["float_shares"] >= self.config.min_float)
            & (df["float_shares"] <= self.config.max_float)
        ].copy()
        _logger.info("Float %dM–%dM shares: %d → %d",
                     self.config.min_float // 1_000_000,
                     self.config.max_float // 1_000_000,
                     n, len(df))

        # Average volume
        n = len(df)
        df = df.loc[df["avg_volume"] >= self.config.min_avg_volume].copy()
        _logger.info("Avg volume ≥%d: %d → %d", self.config.min_avg_volume, n, len(df))

        # Average dollar volume
        n = len(df)
        df["avg_dollar_volume"] = df["avg_volume"] * df["price"]
        df = df.loc[df["avg_dollar_volume"] >= self.config.min_avg_dollar_volume].copy()
        _logger.info("Avg dollar volume ≥$%.0f: %d → %d",
                     self.config.min_avg_dollar_volume, n, len(df))

        return df.reset_index(drop=True)
