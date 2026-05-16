"""
P17 Market Agent

Downloads 90-day OHLCV history and quarterly fundamental data for the
filtered universe using yfinance. Results are cached as parquet files
to avoid redundant downloads on reruns.
"""

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
        self._ohlcv_cache_dir = results_dir / "ohlcv_cache"
        self._ohlcv_cache_dir.mkdir(parents=True, exist_ok=True)
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
        cache_file = self._ohlcv_cache_dir / f"ohlcv_{self.target_date}.parquet"

        if not force_refresh and cache_file.exists():
            try:
                raw = pd.read_parquet(cache_file)
                ohlcv = self._split_ohlcv(raw, tickers)
                _logger.info("OHLCV loaded from cache: %d tickers", len(ohlcv))
                return ohlcv
            except Exception:
                _logger.warning("OHLCV cache unreadable — re-downloading")

        _logger.info("Batch OHLCV download for %d tickers...", len(tickers))
        period = f"{self.config.ohlcv_lookback_days}d"

        try:
            raw = yf.download(
                tickers=tickers,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            _logger.exception("Batch OHLCV download failed")
            return {}

        if raw is None or raw.empty:
            _logger.warning("yf.download returned empty result")
            return {}

        try:
            raw.to_parquet(cache_file)
        except Exception:
            _logger.warning("Could not cache OHLCV to parquet")

        ohlcv = self._split_ohlcv(raw, tickers)
        _logger.info("OHLCV downloaded: %d/%d tickers have data", len(ohlcv), len(tickers))
        return ohlcv

    def _split_ohlcv(
        self,
        raw: pd.DataFrame,
        tickers: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Extract per-ticker DataFrames from the yfinance multi-ticker download."""
        result: Dict[str, pd.DataFrame] = {}

        if raw.empty:
            return result

        # Single ticker: yf.download returns a simple DataFrame (no MultiIndex)
        if len(tickers) == 1:
            ticker = tickers[0]
            df = raw.dropna(how="all")
            if not df.empty:
                result[ticker] = df
            return result

        for ticker in tickers:
            try:
                ticker_data = raw[ticker]
                if not isinstance(ticker_data, pd.DataFrame):
                    continue
                df = ticker_data.dropna(how="all")
                if not df.empty and len(df) >= 20:
                    result[ticker] = df
            except (KeyError, TypeError):
                pass

        return result

    # ── Fundamentals ───────────────────────────────────────────────────────

    def _download_fundamentals(
        self,
        tickers: List[str],
        force_refresh: bool,
    ) -> Dict[str, dict]:
        cache_file = self._ohlcv_cache_dir / f"fundamentals_{self.target_date}.parquet"

        if not force_refresh and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                data = {r["ticker"]: r for r in df.to_dict("records")}
                _logger.info("Fundamentals loaded from cache: %d tickers", len(data))
                return data
            except Exception:
                _logger.warning("Fundamentals cache unreadable — re-downloading")

        _logger.info("Fetching fundamentals for %d tickers...", len(tickers))
        results: Dict[str, dict] = {}

        chunk_size = self.config.ohlcv_chunk_size
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = {
                    pool.submit(self._fetch_ticker_fundamentals, t): t for t in chunk
                }
                for future in as_completed(futures):
                    ticker = futures[future]
                    data = future.result()
                    if data:
                        results[ticker] = data

        try:
            pd.DataFrame(list(results.values())).to_parquet(cache_file, index=False)
        except Exception:
            _logger.warning("Could not cache fundamentals to parquet")

        _logger.info("Fundamentals fetched: %d/%d tickers", len(results), len(tickers))
        return results

    def _fetch_ticker_fundamentals(self, ticker: str) -> Optional[dict]:
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
                ticker, revenue_growth_fallback=f.revenue_growth
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
