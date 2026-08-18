"""
P19 Layer 0 — Structural Integrity Profiler orchestrator (spec v2 §4.0).

Runs **pre-market only** (decision #5 — never intraday). For each watchlist
entry: cache check (weekly TTL + daily delta) → fetch (``EdgarDownloader``
companyfacts/submissions, the widened Form 4 daily cache, the daily 13D/G
cache, EFTS text search, yfinance splits + short-interest) → grade
(``grading.grade_ticker``) → cache write. No new EDGAR network surface for
Form 4 or 13D/G — both read an already-scheduled daily bulk cache
(design-v2.md §3.1; P18's daily scan for 13D/G) rather than issuing
per-ticker calls. EFTS text search (N5/N6/N16) and yfinance short-interest
(P11) *are* per-ticker network calls — cheap (a handful of small requests per
name, weekly cadence via the profile cache TTL) but real, unlike Form4/13D-G.
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p19_penny_intraday.config import P19Config
from src.ml.pipeline.p19_penny_intraday.models.structural_profile import StructuralProfile
from src.ml.pipeline.p19_penny_intraday.models.watchlist_entry import WatchlistEntry
from src.ml.pipeline.p19_penny_intraday.structural import xbrl_facts
from src.ml.pipeline.p19_penny_intraday.structural.cache import StructuralProfileCache
from src.ml.pipeline.p19_penny_intraday.structural.grading import GradingInputs, grade_ticker
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Covers the widest Phase-1.5 Form 4 window (P2/N14, 90 days) with headroom for
# weekends/holidays and a few missing cache days without losing coverage.
_FORM4_LOOKBACK_DAYS = 100

# P8's trailing-2-quarters window (config.py's default p8_lookback_quarters=2),
# expressed in calendar days for the daily-cache walk (design-v2.md §3.1 pattern).
_DG_LOOKBACK_DAYS = 183

_ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A")
_INTERIM_FORMS = ("10-Q", "10-Q/A", "6-K")
_FPI_FORMS = frozenset({"20-F", "20-F/A", "6-K"})
_DOMESTIC_PERIODIC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A"})


def _parse_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(str(s)[:10])
    except ValueError:
        return None


def _latest_form_date(filings: List[Dict[str, Any]], forms: Tuple[str, ...]) -> Optional[Tuple[date, str]]:
    """Latest (date, actual_form_type) among ``forms`` — the specific form type
    is kept so callers can scope an EFTS query to exactly one form (EFTS
    ``forms`` is an exact match, not an OR; see edgar_downloader.py) instead of
    re-querying every candidate type in ``forms`` at that date."""
    hits = [
        (d, str(f.get("form", "")).upper().strip())
        for f in filings
        if str(f.get("form", "")).upper().strip() in forms
        for d in [_parse_date(str(f.get("filingDate", "")))]
        if d is not None
    ]
    return max(hits, key=lambda h: h[0]) if hits else None


def _earliest_filing_date(filings: List[Dict[str, Any]]) -> Optional[date]:
    dates = [d for f in filings for d in [_parse_date(str(f.get("filingDate", "")))] if d is not None]
    return min(dates) if dates else None


def _infer_is_fpi(filings: List[Dict[str, Any]]) -> Optional[bool]:
    """
    N15 + StructuralSignals.md §2 — FPI reporting inferred from which form
    types appear, not a separate lookup. None (unresolved) when neither an FPI
    nor a domestic periodic-report form has ever appeared — e.g. a brand-new
    registrant with only an S-1 on file, genuinely undetermined either way.
    """
    forms_seen = {str(f.get("form", "")).upper().strip() for f in filings}
    has_fpi_forms = bool(forms_seen & _FPI_FORMS)
    has_domestic_forms = bool(forms_seen & _DOMESTIC_PERIODIC_FORMS)
    if has_domestic_forms:
        return False
    if has_fpi_forms:
        return True
    return None


class StructuralProfiler:
    """Refreshes and caches ``StructuralProfile``s for a watchlist."""

    def __init__(
        self,
        config: Optional[P19Config] = None,
        edgar: Optional[EdgarDownloader] = None,
        cache: Optional[StructuralProfileCache] = None,
    ) -> None:
        self.cfg = config or P19Config.create_default()
        self._edgar = edgar or EdgarDownloader()
        self._cache = cache or StructuralProfileCache(ttl_days=self.cfg.structural_config.profile_cache_ttl_days)
        self._cik_map: Optional[Dict[str, str]] = None

    # ── Public API ────────────────────────────────────────────────────────

    def refresh_watchlist(
        self,
        entries: List[WatchlistEntry],
        as_of: Optional[date] = None,
        force: bool = False,
    ) -> Dict[str, StructuralProfile]:
        """Refresh (or reuse cached) profiles for every entry, return ticker → profile."""
        as_of = as_of or datetime.now().date()
        form4_df = self._load_form4_window(as_of)
        dg_df = self._load_dg_window(as_of)
        out: Dict[str, StructuralProfile] = {}
        refreshed = 0
        for entry in entries:
            profile, was_refreshed = self._refresh_one(entry, as_of, form4_df, dg_df, force)
            out[entry.ticker] = profile
            refreshed += int(was_refreshed)
        _logger.info("Structural profiler: %d/%d names refreshed (rest served from cache)", refreshed, len(entries))
        return out

    # ── Per-ticker orchestration ─────────────────────────────────────────

    def _refresh_one(
        self,
        entry: WatchlistEntry,
        as_of: date,
        form4_df: pd.DataFrame,
        dg_df: pd.DataFrame,
        force: bool,
    ) -> Tuple[StructuralProfile, bool]:
        ticker = entry.ticker
        cik = self._resolve_cik(ticker)
        latest_filing_date = self._latest_filing_date(cik) if cik else None

        if not force and self._cache.is_fresh(ticker, as_of, latest_filing_date):
            cached = self._cache.load(ticker)
            if cached is not None:
                return cached, False

        inputs = self._build_inputs(entry, cik, as_of, form4_df, dg_df)
        profile = grade_ticker(inputs, self.cfg.structural_config)
        self._cache.save(profile)
        return profile, True

    def _build_inputs(
        self,
        entry: WatchlistEntry,
        cik: Optional[str],
        as_of: date,
        form4_df: pd.DataFrame,
        dg_df: pd.DataFrame,
    ) -> GradingInputs:
        company_facts = None
        filings: Optional[List[Dict[str, Any]]] = None
        if cik is not None:
            try:
                # force_refresh=True: EdgarDownloader's own cache has no TTL
                # (design-v2.md §0.1) — Layer 0 owns freshness, and this branch
                # only runs when the profiler already decided a refresh is due.
                company_facts = self._edgar.load_company_facts(cik, force_refresh=True)
            except Exception:
                _logger.warning("companyfacts fetch failed for %s (CIK %s)", entry.ticker, cik)
            try:
                filings = self._edgar.get_recent_filings(cik, force_refresh=True)
            except Exception:
                _logger.warning("submissions fetch failed for %s (CIK %s)", entry.ticker, cik)

        splits = self._fetch_splits(entry.ticker)
        form4_rows = self._filter_form4(form4_df, entry.ticker)

        listing_date: Optional[date] = None
        is_fpi: Optional[bool] = None
        floating_convert_hit: Optional[bool] = None
        going_concern_hit: Optional[bool] = None
        auditor_name: Optional[str] = None
        if filings is not None:
            listing_date = _earliest_filing_date(filings)
            is_fpi = _infer_is_fpi(filings)
            if cik is not None:
                floating_convert_hit, going_concern_hit, auditor_name = self._fetch_text_signals(cik, filings, as_of)

        debt_maturity_near_term = xbrl_facts.has_near_term_debt_maturity(company_facts) if company_facts else None
        dg_activity_2q = self._filter_dg(dg_df, cik) if cik is not None else None
        si_pct_float, days_to_cover = self._fetch_short_interest(entry.ticker)

        return GradingInputs(
            ticker=entry.ticker,
            cik=cik,
            as_of=as_of,
            company_facts=company_facts,
            filings=filings,
            splits=splits,
            form4_rows=form4_rows,
            market_cap=entry.market_cap or None,
            float_shares=entry.float_shares or None,
            prior_close=entry.prior_close or None,
            floating_convert_hit=floating_convert_hit,
            going_concern_hit=going_concern_hit,
            auditor_name=auditor_name,
            listing_date=listing_date,
            is_fpi=is_fpi,
            dg_activity_2q=dg_activity_2q,
            debt_maturity_near_term=debt_maturity_near_term,
            short_interest_pct_float=si_pct_float,
            days_to_cover=days_to_cover,
        )

    # ── CIK resolution ────────────────────────────────────────────────────

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        if self._cik_map is None:
            self._cik_map = self._build_cik_map()
        return self._cik_map.get(ticker.upper())

    def _build_cik_map(self) -> Dict[str, str]:
        try:
            raw = self._edgar.load_company_tickers()
        except Exception:
            _logger.warning("company_tickers.json load failed — CIK resolution disabled this run")
            return {}
        out: Dict[str, str] = {}
        for entry in raw.values():
            t = str(entry.get("ticker", "")).upper()
            cik = entry.get("cik_str")
            if t and cik:
                out[t] = str(cik)
        return out

    def _latest_filing_date(self, cik: str) -> Optional[date]:
        try:
            filings = self._edgar.get_recent_filings(cik)
        except Exception:
            return None
        dates = [d for d in (_parse_date(f.get("filingDate", "")) for f in filings) if d is not None]
        return max(dates) if dates else None

    # ── Splits (yfinance, direct call — design-v2.md §3.3) ──────────────────

    def _fetch_splits(self, ticker: str) -> Optional[List[Tuple[date, float]]]:
        try:
            import yfinance as yf

            raw = yf.Ticker(ticker).splits
        except Exception:
            _logger.warning("yfinance splits fetch failed for %s", ticker)
            return None
        if raw is None:
            return []
        # yfinance silently returns an empty Series on most failures too, so an
        # empty result here is treated as "resolved: no splits" — a known
        # limitation (yfinance doesn't distinguish "no data" from "no splits").
        out: List[Tuple[date, float]] = []
        for ts, ratio in raw.items():
            ts_any: Any = ts
            d = ts_any.date() if hasattr(ts_any, "date") else ts_any
            out.append((cast(date, d), float(cast(Any, ratio))))
        return out

    # ── Form 4 — read the already-scheduled daily cache, no new fetch surface ──

    def _load_form4_window(self, as_of: date) -> pd.DataFrame:
        frames = []
        d = as_of
        start = as_of - timedelta(days=_FORM4_LOOKBACK_DAYS)
        while d >= start:
            if d.weekday() < 5:
                try:
                    # force=False reads the on-disk cache written by the daily
                    # P15/P18 job; only a genuinely missing day triggers a
                    # (self-healing) live EDGAR call.
                    day_df = self._edgar.download_form4_filings(as_of_date=d, force=False)
                    if day_df is not None and not day_df.empty:
                        frames.append(day_df)
                except Exception:
                    _logger.debug("Form 4 cache read failed for %s", d)
            d -= timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _filter_form4(form4_df: pd.DataFrame, ticker: str) -> List[Dict[str, Any]]:
        if form4_df.empty or "ticker" not in form4_df.columns:
            return []
        mask = form4_df["ticker"].astype(str).str.upper() == ticker.upper()
        return cast(List[Dict[str, Any]], form4_df[mask].to_dict("records"))

    # ── 13D/G — read P18's already-scheduled daily cache, no new fetch surface ──

    def _load_dg_window(self, as_of: date) -> pd.DataFrame:
        frames = []
        d = as_of
        start = as_of - timedelta(days=_DG_LOOKBACK_DAYS)
        while d >= start:
            if d.weekday() < 5:
                try:
                    # force=False reads the on-disk cache P18's daily scan
                    # already writes (design-v2.md §3.1's Form4 pattern, same
                    # reuse for 13D/G); a missing day self-heals via one live
                    # EDGAR quarterly-index fetch.
                    day_df = self._edgar.download_13dg_filings(as_of_date=d, force=False)
                    if day_df is not None and not day_df.empty:
                        frames.append(day_df)
                except Exception:
                    _logger.debug("13D/G cache read failed for %s", d)
            d -= timedelta(days=1)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _filter_dg(dg_df: pd.DataFrame, cik: str) -> Optional[bool]:
        """
        P8 proxy — was any 13D/G filed against this CIK in the window.
        ``None`` only when the window produced no data at all (couldn't
        resolve, as opposed to genuinely resolving to "no activity").
        """
        if dg_df.empty or "cik" not in dg_df.columns:
            return None
        cik_norm = cik.lstrip("0") or "0"
        mask = dg_df["cik"].astype(str).str.lstrip("0") == cik_norm
        return bool(mask.any())

    # ── EFTS text signals — N5/N6/N16 (design-v2.md §3.2, real per-ticker calls) ─

    def _fetch_text_signals(
        self,
        cik: str,
        filings: List[Dict[str, Any]],
        as_of: date,
    ) -> Tuple[Optional[bool], Optional[bool], Optional[str]]:
        """Returns (floating_convert_hit [N5], going_concern_hit [N6], auditor_name [N16])."""
        del as_of  # window is anchored on the filings' own dates, not today
        cfg = self.cfg.structural_config
        annual = _latest_form_date(filings, _ANNUAL_FORMS)
        interim = _latest_form_date(filings, _INTERIM_FORMS)

        # N5: latest annual AND latest interim (StructuralSignals.md N5). N6:
        # latest annual only — going-concern is an annual-audit-opinion signal.
        convert_hit = self._phrase_hit_on(cik, cfg.n5_convert_phrases, [w for w in (annual, interim) if w])
        going_concern_hit = self._phrase_hit_on(cik, [cfg.n6_going_concern_phrase], [annual] if annual else [])

        auditor_name: Optional[str] = None
        if annual is not None:
            annual_date, _ = annual
            try:
                # A window around the annual filing date, not an exact-day
                # match — get_auditor_name issues its own per-form EFTS calls
                # (comma-list forms don't OR in EFTS, see edgar_downloader.py).
                auditor_name = self._edgar.get_auditor_name(
                    cik,
                    start_dt=(annual_date - timedelta(days=5)).isoformat(),
                    end_dt=(annual_date + timedelta(days=30)).isoformat(),
                )
            except Exception:
                _logger.warning("get_auditor_name failed for CIK %s", cik)

        return convert_hit, going_concern_hit, auditor_name

    def _phrase_hit_on(self, cik: str, phrases: List[str], windows: List[Tuple[date, str]]) -> Optional[bool]:
        """
        True/False once at least one (date, form) window was successfully
        queried; None only if there was nothing to scope the search to at all
        (StructuralSignals.md's recency-scoping requirement means an unscoped
        search is not a safer fallback, it's a different and worse signal).
        One EFTS request per window — the form type is already known from
        ``_latest_form_date``, not re-guessed across every candidate type.
        """
        if not windows:
            return None
        any_hit = False
        for d, form in windows:
            try:
                hits = self._edgar.efts_text_search(
                    cik=cik, phrases=phrases, forms=form, start_dt=d.isoformat(), end_dt=d.isoformat()
                )
            except Exception:
                _logger.warning("efts_text_search failed for CIK %s form %s on %s", cik, form, d)
                continue
            if hits:
                any_hit = True
        return any_hit

    # ── Short interest (yfinance, direct call — same pattern as _fetch_splits) ──

    def _fetch_short_interest(self, ticker: str) -> Tuple[Optional[float], Optional[float]]:
        """P11 evidence: (short_interest_pct_float, days_to_cover)."""
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info or {}
        except Exception:
            _logger.warning("yfinance short-interest fetch failed for %s", ticker)
            return None, None
        si_pct = info.get("shortPercentOfFloat")
        dtc = info.get("shortRatio")
        return (float(si_pct) if si_pct is not None else None), (float(dtc) if dtc is not None else None)
