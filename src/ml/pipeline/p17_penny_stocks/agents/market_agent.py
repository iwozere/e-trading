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
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pandas_market_calendars as mcal
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cache.fundamentals_cache import FundamentalsCache
from src.data.cache.unified_cache import get_unified_cache
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader
from src.ml.pipeline.p17_penny_stocks.config import P17FilterConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_XNYS = mcal.get_calendar("XNYS")


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        v = float(value)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _naive_dates(index: pd.Index) -> pd.DatetimeIndex:
    """DatetimeIndex normalised to midnight, timezone dropped."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def last_session_on_or_before(day: pd.Timestamp) -> pd.Timestamp:
    """
    Most recent NYSE trading session on or before ``day``.

    Args:
        day: Calendar date (time and timezone ignored).

    Returns:
        Tz-naive midnight timestamp of that session (weekends and exchange holidays roll back).
    """
    day = pd.Timestamp(day).normalize()
    if day.tz is not None:
        day = day.tz_localize(None)
    sessions = _XNYS.valid_days(start_date=day - timedelta(days=14), end_date=day)
    if len(sessions) == 0:
        return day
    return _naive_dates(sessions)[-1]


def last_bar_date(df: pd.DataFrame) -> pd.Timestamp:
    """Date (tz-naive midnight) of the last bar in a DatetimeIndex-ed OHLCV frame."""
    return _naive_dates(df.index)[-1]


def truncate_to(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Drop bars dated after ``day`` (no look-ahead for historical target dates)."""
    return df.loc[_naive_dates(df.index) <= pd.Timestamp(day).normalize()]


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

        Download-first: the post-filter universe (~150 names) is always batch-downloaded,
        because the shared UnifiedCache (DATA_CACHE_DIR/ohlcv/<symbol>/1d/) cannot tell a
        complete daily bar from a partial intraday one written by another job during market
        hours, and the old 5-calendar-day freshness tolerance let ~half of the rows be scored
        on 1–3-session-old bars. Fresh downloads are written back (overwriting partial bars).
        The cache is only a fallback for tickers whose download fails, and then only if it
        reaches the most recent NYSE session ≤ target_date. Bars after target_date are
        dropped so historical ``--target-date`` runs see no future data.

        Args:
            tickers: Tickers that passed the universe hard filters.
            force_refresh: Kept for interface compatibility; OHLCV is always re-downloaded.
        """
        end_dt = datetime.now(UTC)
        start_dt = end_dt - timedelta(days=self.config.ohlcv_lookback_days + 10)
        target = pd.Timestamp(self.target_date).normalize()
        expected_session = last_session_on_or_before(target)

        cache = get_unified_cache()
        result: Dict[str, pd.DataFrame] = {}

        _logger.info("Batch OHLCV download: %d tickers (expected last session %s)", len(tickers), expected_session.date())
        batch = self._downloader.get_ohlcv_batch(tickers, "1d", start_dt, end_dt) if tickers else {}
        for ticker, df in batch.items():
            if df is None or df.empty:
                continue
            df_indexed = self._set_datetime_index(df)
            try:
                cache.put(df_indexed, ticker, "1d", start_dt, end_dt, provider="yahoo")
            except Exception:
                _logger.debug("Could not write %s to unified cache", ticker)
            df_indexed = truncate_to(df_indexed, target)
            if len(df_indexed) >= 20:
                result[ticker] = self._to_technical_format(df_indexed)

        fallback = [t for t in tickers if t not in result]
        from_cache = 0
        for ticker in fallback:
            cached = cache.get(ticker, "1d", start_dt, end_dt)
            if cached is None or cached.empty:
                continue
            cached = truncate_to(cached, target)
            if len(cached) >= 20 and last_bar_date(cached) >= expected_session:
                result[ticker] = self._to_technical_format(cached)
                from_cache += 1
        if fallback:
            _logger.info("OHLCV: %d tickers failed to download; %d served from a fresh cache", len(fallback), from_cache)

        # Halted names / Yahoo lag: still usable, but say so — their RVOL/breakouts describe an older day.
        lagging = sorted(t for t, df in result.items() if last_bar_date(df) < expected_session)
        if lagging:
            _logger.warning(
                "OHLCV: %d tickers still end before %s after download: %s",
                len(lagging),
                expected_session.date(),
                ", ".join(lagging[:20]),
            )

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
                futures = {pool.submit(self._fetch_ticker_fundamentals, t, force_refresh): t for t in chunk}
                for future in as_completed(futures):
                    ticker = futures[future]
                    data = future.result()
                    if data:
                        results[ticker] = data

        _logger.info("Fundamentals fetched: %d/%d tickers", len(results), len(tickers))
        return results

    def _fetch_ticker_fundamentals(self, ticker: str, force_refresh: bool = False) -> dict | None:
        if not force_refresh:
            cache_meta = self._fundamentals_cache.find_latest_json(ticker, provider="yahoo", max_age_days=1)
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

    def _fundamentals_to_market_record(self, ticker: str, f: dict) -> dict | None:
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
            "data_as_of": f.get("last_updated") or datetime.now(UTC).isoformat(),
        }

    def _compute_revenue_growth(self, ticker: str, revenue_growth_fallback: float | None = None) -> float | None:
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
