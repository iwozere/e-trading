"""
P17 Market Agent

Downloads 90-day OHLCV history and quarterly fundamental data for the
filtered universe. OHLCV data is read from / written to the shared
UnifiedCache (DATA_CACHE_DIR/ohlcv/<symbol>/1d/), so it is available
to other pipelines and avoids redundant downloads on re-runs.

Fundamentals are served from FundamentalsCache (per-symbol JSON files),
which is also shared across pipelines.
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p17_penny_stocks.config import P17FilterConfig
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.data.cache.fundamentals_cache import FundamentalsCache
from src.data.cache.unified_cache import get_unified_cache

_logger = setup_logger(__name__)


def _safe_float(value, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(value)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


class MarketAgent:
    """
    Stage 2: OHLCV history and fundamental enrichment.

    For each ticker from the universe agent this class produces:
      - ohlcv: Dict[ticker → daily OHLCV DataFrame (90 days)]
      - fundamentals: Dict[ticker → dict of fundamental metrics]
    """

    def __init__(
        self,
        config: P17FilterConfig,
        results_dir: Path,
        target_date: str,
    ) -> None:
        self.config = config
        self.results_dir = results_dir
        self.target_date = target_date
        self._downloader = YahooDataDownloader()
        self._fundamentals_cache = FundamentalsCache()

    def run(
        self,
        tickers: List[str],
        force_refresh: bool = False,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
        """
        Download OHLCV and fundamentals for all tickers.

        Args:
            tickers: Tickers that passed universe hard filters.
            force_refresh: Bypass file caches.

        Returns:
            (ohlcv, fundamentals) — dicts keyed by ticker.
        """
        ohlcv = self._download_ohlcv(tickers, force_refresh)
        fundamentals = self._download_fundamentals(tickers, force_refresh)
        return ohlcv, fundamentals

    # ── OHLCV ──────────────────────────────────────────────────────────────

    def _download_ohlcv(
        self,
        tickers: List[str],
        force_refresh: bool,
    ) -> Dict[str, pd.DataFrame]:
        """
        Serve 90-day daily OHLCV per ticker.

        Cache-first: reads from UnifiedCache (DATA_CACHE_DIR/ohlcv/<symbol>/1d/).
        Cache misses are batch-downloaded and written back to the cache.
        """
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self.config.ohlcv_lookback_days + 10)
        # Data is considered fresh when it reaches within 5 calendar days of target_date
        min_fresh = pd.Timestamp(self.target_date) - timedelta(days=5)

        cache = get_unified_cache()
        result: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []

        if not force_refresh:
            for ticker in tickers:
                cached = cache.get(ticker, "1d", start_dt, end_dt)
                if (
                    cached is not None
                    and not cached.empty
                    and len(cached) >= 20
                    and cached.index[-1] >= min_fresh  # type: ignore[operator]
                ):
                    result[ticker] = self._to_technical_format(cached)
                else:
                    missing.append(ticker)
        else:
            missing = list(tickers)

        if missing:
            _logger.info(
                "Batch OHLCV download: %d cache misses (cache hits: %d)",
                len(missing), len(result),
            )
            batch = self._downloader.get_ohlcv_batch(missing, "1d", start_dt, end_dt)
            for ticker, df in batch.items():
                if df is None or df.empty:
                    continue
                df_indexed = self._set_datetime_index(df)
                if len(df_indexed) < 20:
                    continue
                result[ticker] = self._to_technical_format(df_indexed)
                try:
                    cache.put(df_indexed, ticker, "1d", start_dt, end_dt, provider="yahoo")
                except Exception:
                    _logger.debug("Could not write %s to unified cache", ticker)
        else:
            _logger.info("OHLCV: all %d tickers served from unified cache", len(result))

        _logger.info("OHLCV ready: %d/%d tickers have sufficient history", len(result), len(tickers))
        return result

    @staticmethod
    def _set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert get_ohlcv_batch() output (timestamp column + lowercase) to
        a DataFrame with DatetimeIndex and lowercase columns, ready for UnifiedCache.
        """
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).set_index("timestamp")
            if hasattr(df.index, "tz") and df.index.tz is not None:  # type: ignore[union-attr]
                df.index = df.index.tz_localize(None)  # type: ignore[union-attr]
        return df

    @staticmethod
    def _to_technical_format(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize any OHLCV DataFrame to the format TechnicalAgent expects:
        DatetimeIndex + capitalized column names (Close, High, Low, Volume, Open).
        """
        rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        cols = {c: rename[c.lower()] for c in df.columns if c.lower() in rename}
        df = df.rename(columns=cols)
        df = df.dropna(how="all")
        return df

    # ── Fundamentals ───────────────────────────────────────────────────────

    def _download_fundamentals(
        self,
        tickers: List[str],
        force_refresh: bool,
    ) -> Dict[str, dict]:
        """
        Fetch fundamentals per ticker from FundamentalsCache (shared per-symbol
        JSON cache). Falls back to live download when cache is stale or missing.
        """
        _logger.info("Fetching fundamentals for %d tickers...", len(tickers))
        results: Dict[str, dict] = {}

        chunk_size = self.config.ohlcv_chunk_size
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = {
                    pool.submit(self._fetch_ticker_fundamentals, t, force_refresh): t
                    for t in chunk
                }
                for future in as_completed(futures):
                    ticker = futures[future]
                    data = future.result()
                    if data:
                        results[ticker] = data

        _logger.info("Fundamentals fetched: %d/%d tickers", len(results), len(tickers))
        return results

    def _fetch_ticker_fundamentals(self, ticker: str, force_refresh: bool = False) -> Optional[dict]:
        if not force_refresh:
            cache_meta = self._fundamentals_cache.find_latest_json(
                ticker, provider="yahoo", max_age_days=1
            )
            if cache_meta:
                cached = self._fundamentals_cache.read_json(cache_meta.file_path)
                if cached:
                    return self._fundamentals_to_market_record(ticker, cached)

        try:
            f = self._downloader.get_fundamentals(ticker)
            if f is None:
                return None
            f_dict = dataclasses.asdict(f)
            f_dict["revenue_growth_yoy"] = self._compute_revenue_growth(
                ticker, revenue_growth_fallback=f_dict.get("revenue_growth")
            )
            self._fundamentals_cache.write_json(ticker, "yahoo", f_dict)
            return self._fundamentals_to_market_record(ticker, f_dict)
        except Exception:
            _logger.debug("Failed to fetch fundamentals for %s", ticker)
            return None

    def _fundamentals_to_market_record(self, ticker: str, f: dict) -> Optional[dict]:
        """Map a cached Fundamentals dict to the market agent record format."""
        operating_cf = _safe_float(f.get("operating_cashflow"))
        total_cash = _safe_float(f.get("total_cash"))

        cash_runway = None
        if total_cash is not None and operating_cf is not None and operating_cf < 0:
            monthly_burn = abs(operating_cf) / 12
            if monthly_burn > 0:
                cash_runway = total_cash / monthly_burn

        return {
            "ticker": ticker,
            "revenue_growth_yoy": f.get("revenue_growth_yoy"),
            "gross_margin": _safe_float(f.get("gross_margin")),
            "total_cash": total_cash,
            "total_debt": _safe_float(f.get("total_debt")),
            "operating_cashflow": operating_cf,
            "cash_runway_months": cash_runway,
            "short_ratio": _safe_float(f.get("short_ratio")),
            "short_pct_float": _safe_float(f.get("short_pct_float")),
            "institutional_pct": _safe_float(f.get("institutional_pct")),
            "high_52w": _safe_float(f.get("fifty_two_week_high")),
            "low_52w": _safe_float(f.get("fifty_two_week_low")),
            "data_as_of": f.get("last_updated") or datetime.now(timezone.utc).isoformat(),
        }

    def _compute_revenue_growth(
        self, ticker: str, revenue_growth_fallback: Optional[float] = None
    ) -> Optional[float]:
        """
        Compute YoY revenue growth from quarterly income statements.
        Falls back to the TTM revenueGrowth value already in the Fundamentals object.
        """
        try:
            qf = yf.Ticker(ticker).quarterly_income_stmt
            if qf is None or qf.empty:
                raise ValueError("no quarterly income stmt")

            rev_row = next(
                (row for row in qf.index if "Revenue" in str(row) or "Total Revenue" in str(row)),
                None,
            )
            if rev_row is None:
                raise ValueError("revenue row not found")

            rev = qf.loc[rev_row].dropna()
            if len(rev) < 5:
                raise ValueError("insufficient quarters")

            ttm_current = rev.iloc[:4].sum()
            ttm_prior = rev.iloc[4:8].sum() if len(rev) >= 8 else rev.iloc[4:].sum()

            if ttm_prior and ttm_prior != 0:
                return float((ttm_current - ttm_prior) / abs(ttm_prior))
        except Exception:
            pass

        return revenue_growth_fallback

    def apply_survival_filter(
        self,
        tickers: List[str],
        fundamentals: Dict[str, dict],
        config: P17FilterConfig,
    ) -> List[str]:
        """
        Apply financial survival hard-stops from fundamentals data.
        Returns subset of tickers that pass.
        """
        passed = []
        for ticker in tickers:
            f = fundamentals.get(ticker, {})

            runway = f.get("cash_runway_months")
            if runway is not None and runway < config.min_cash_runway_months:
                _logger.debug("%s failed cash runway: %.1f months", ticker, runway)
                continue

            cash = f.get("total_cash") or 0.0
            debt = f.get("total_debt") or 0.0
            if cash > 0 and (debt / cash) > config.max_debt_to_cash:
                _logger.debug("%s failed debt/cash ratio: %.1f", ticker, debt / cash)
                continue

            passed.append(ticker)

        _logger.info("Survival filter: %d → %d tickers", len(tickers), len(passed))
        return passed
