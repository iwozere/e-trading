"""
P17 Catalyst Agent

Detects bullish catalysts from recent SEC EDGAR 8-K filings and produces a
normalized ``catalyst_score`` (0–100) plus human-readable ``catalyst_signals``.

Catalysts are the natural driver of explosive penny-stock moves (spec §8.6), yet
were a Phase-1 placeholder (always 0). This agent fills that gap.

Data source: the universe-wide **daily 8-K index cache** written by the P15 daily
bundle (``EdgarDownloader.download_8k_filings`` →
``DATA_CACHE_DIR/edgar/8k/index/{date}.csv.gz``). The agent reads the cached index
for its lookback window — no per-candidate EDGAR calls. If no index file exists for
the window yet (e.g. before P15 has backfilled), it falls back to per-CIK
``get_recent_filings``.

Signal model
------------
8-K filings are classified by their structured ``items`` codes and refined by
``primaryDocDescription`` keywords:

  Tier 1 (strong):
    - item 1.01  Entry into a Material Definitive Agreement (contract / partnership)
    - keywords:  FDA / clearance / approval, defense / DoD / DARPA, contract award,
                 merger / acquisition / definitive agreement, guidance raise,
                 nuclear, rare earth(s)
  Tier 2 (moderate):
    - item 2.02  Results of Operations (earnings event)
    - item 7.01 / 8.01  Reg-FD / Other Events (generic press release)
    - keywords:  AI, patent, milestone, launch, uplisting

Bearish / dilutive items (1.02 termination, 1.03 bankruptcy, 3.01 delisting,
4.02 restatement) are never treated as catalysts — dilution is handled by the
dilution agent. Catalysts decay quickly, so only filings within
``lookback_days`` are considered and more recent ones are weighted higher.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p17_penny_stocks.config import P17CatalystConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_CATALYST_FORMS = {"8-K", "8-K/A"}

# Structured 8-K item codes
_MATERIAL_AGREEMENT_ITEMS = {"1.01"}  # contract / partnership — Tier 1
_EARNINGS_ITEMS = {"2.02"}  # results of operations — Tier 2
_PRESS_RELEASE_ITEMS = {"7.01", "8.01"}  # Reg-FD / other events — Tier 2
_BEARISH_ITEMS = {"1.02", "1.03", "3.01", "4.02"}  # never a catalyst

# Keyword refinement on primaryDocDescription (lower-cased)
_TIER1_KEYWORDS = {
    "fda",
    "clearance",
    "approval",
    "breakthrough",
    "phase 3",
    "phase iii",
    "defense",
    "department of defense",
    "darpa",
    "contract award",
    "awarded a contract",
    "definitive agreement",
    "merger",
    "acquisition",
    "acquire",
    "letter of intent",
    "guidance",
    "record revenue",
    "nuclear",
    "rare earth",
}
_TIER2_KEYWORDS = {
    "artificial intelligence",
    " ai ",
    "patent",
    "milestone",
    "launch",
    "partnership",
    "collaboration",
    "uplist",
    "grant",
}


def _cik_key(cik: Any) -> str:
    """Normalise a CIK (int or zero-padded/plain string) to a leading-zero-free key."""
    try:
        return str(int(cik))
    except (TypeError, ValueError):
        return str(cik).strip().lstrip("0")


class CatalystAgent:
    """
    Stage 5.5: Detect bullish catalysts from SEC EDGAR 8-K filings.

    Populates on each Candidate:
      - catalyst_score   (0–100, used by the scoring agent with weight_catalyst)
      - catalyst_signals (list of human-readable signal names)
    """

    def __init__(
        self,
        catalyst_config: P17CatalystConfig,
        results_dir: Path,
        target_date: str,
    ) -> None:
        self.cfg = catalyst_config
        self.results_dir = results_dir
        self.target_date = target_date
        self.lookback = timedelta(days=catalyst_config.lookback_days)
        self._edgar = EdgarDownloader()

    def run(
        self,
        candidates: List[Candidate],
        force_refresh: bool = False,
    ) -> List[Candidate]:
        """
        Populate catalyst_score and catalyst_signals on each candidate.

        Args:
            candidates: Candidate list to enrich.
            force_refresh: Bypass the EDGAR submissions cache.

        Returns:
            Same list with catalyst fields set.
        """
        cik_map = self._build_cik_map(candidates)
        if not cik_map:
            _logger.warning("CIK map empty — catalyst detection skipped")
            return candidates

        today = datetime.strptime(self.target_date, "%Y-%m-%d")
        since = today - self.lookback

        # Prefer the universe-wide daily 8-K index cache (written by the P15 daily
        # bundle). Fall back to per-candidate EDGAR submissions only when no index
        # file exists for the window (e.g. before the P15 8-K job has backfilled).
        index_by_cik = self._load_index_window(today, since)
        source = "8-K index cache" if index_by_cik is not None else "per-CIK EDGAR (fallback)"
        _logger.info("Catalyst agent: analysing %d candidates via %s", len(candidates), source)

        flagged, skipped = 0, 0
        for c in candidates:
            cik = cik_map.get(c.ticker.upper())
            if cik is None:
                skipped += 1
                continue
            if index_by_cik is not None:
                rows = index_by_cik.get(_cik_key(cik), [])
            else:
                rows = self._fetch_edgar_rows(cik, since, force_refresh)
            if self._score_candidate(c, rows, today):
                flagged += 1

        _logger.info(
            "Catalyst agent: %d candidates with a catalyst, %d skipped (no CIK)",
            flagged,
            skipped,
        )
        return candidates

    # ── Filing analysis ────────────────────────────────────────────────────

    def _load_index_window(
        self,
        today: datetime,
        since: datetime,
    ) -> Dict[str, List[Dict[str, Any]]] | None:
        """
        Load cached daily 8-K index rows for [since, today], grouped by CIK key.

        Returns None when no index file exists anywhere in the window, signalling
        the caller to fall back to the per-CIK EDGAR path. Returns a (possibly
        empty) dict when at least one daily index file is present.
        """
        index_dir = self._edgar._8k_index_dir
        if not index_dir.exists():
            return None

        by_cik: Dict[str, List[Dict[str, Any]]] = {}
        found_file = False
        current = since.date()
        end = today.date()
        while current <= end:
            path = index_dir / f"{current.isoformat()}.csv.gz"
            if path.exists():
                found_file = True
                try:
                    df = pd.read_csv(path, compression="gzip", dtype=str)
                except Exception:
                    _logger.warning("Could not read 8-K index %s", path)
                    df = None
                if df is not None and not df.empty:
                    for _, r in df.iterrows():
                        key = _cik_key(r.get("cik", ""))
                        if not key:
                            continue
                        by_cik.setdefault(key, []).append(
                            {
                                "items": str(r.get("items", "") or ""),
                                "description": str(r.get("description", "") or ""),
                                "filing_date": self._parse_date(str(r.get("filed_date", "") or "")),
                            }
                        )
            current += timedelta(days=1)

        return by_cik if found_file else None

    def _fetch_edgar_rows(
        self,
        cik: Union[int, str],
        since: datetime,
        force_refresh: bool,
    ) -> List[Dict[str, Any]]:
        """Legacy fallback: fetch a candidate's recent 8-K filings directly from EDGAR."""
        try:
            filings = self._edgar.get_recent_filings(cik, form_type=None, since=since, force_refresh=force_refresh)
        except Exception:
            _logger.debug("Could not fetch filings for CIK %s", cik)
            return []

        rows: List[Dict[str, Any]] = []
        for filing in filings:
            form = str(filing.get("form") or "").upper().strip()
            if form not in _CATALYST_FORMS:
                continue
            rows.append(
                {
                    "items": str(filing.get("items") or ""),
                    "description": str(filing.get("primaryDocDescription") or ""),
                    "filing_date": self._parse_date(str(filing.get("filingDate") or "")),
                }
            )
        return rows

    def _score_candidate(
        self,
        c: Candidate,
        rows: List[Dict[str, Any]],
        today: datetime,
    ) -> bool:
        """
        Score a candidate from normalised 8-K rows (items, description, filing_date).

        Returns True when a catalyst was found and catalyst_score/signals were set.
        """
        best = 0.0
        categories: set = set()
        signals: List[str] = []

        for row in rows:
            filing_date = row.get("filing_date")
            if filing_date is None:
                continue
            age_days = (today - filing_date).days
            if age_days < 0 or age_days > self.cfg.lookback_days:
                continue

            category, base = self._classify(
                str(row.get("items") or ""),
                str(row.get("description") or ""),
            )
            if category is None or base <= 0:
                continue

            points = base * self._recency_multiplier(age_days)
            if points <= 0:
                continue

            categories.add(category)
            signals.append(f"catalyst_{category}_{filing_date.date().isoformat()}")
            best = max(best, points)

        if best <= 0:
            return False

        score = best
        if len(categories) >= 2:
            score = min(100.0, score + self.cfg.multi_catalyst_bonus)

        c.catalyst_score = round(min(100.0, score), 1)
        c.catalyst_signals = signals[:5]
        _logger.debug("%s catalyst_score=%.1f signals=%s", c.ticker, c.catalyst_score, c.catalyst_signals)
        return True

    def _classify(self, items: str, description: str) -> Tuple[str | None, float]:
        """
        Map an 8-K's item codes + description to (category, base_points).

        Keyword matches take precedence over item codes because they identify the
        catalyst type; a bare item code is a weaker, generic signal.
        """
        item_set = {i.strip() for i in items.split(",") if i.strip()}
        desc = f" {description.lower()} "

        # Purely bearish/dilutive filings are not catalysts.
        if item_set and item_set <= _BEARISH_ITEMS:
            return (None, 0.0)

        if any(kw in desc for kw in _TIER1_KEYWORDS):
            return ("tier1_news", self.cfg.points_tier1)
        if item_set & _MATERIAL_AGREEMENT_ITEMS:
            return ("material_agreement", self.cfg.points_tier1)
        if any(kw in desc for kw in _TIER2_KEYWORDS):
            return ("tier2_news", self.cfg.points_tier2)
        if item_set & _PRESS_RELEASE_ITEMS:
            return ("press_release", self.cfg.points_tier2)
        if item_set & _EARNINGS_ITEMS:
            return ("earnings", self.cfg.points_tier2)
        return (None, 0.0)

    def _recency_multiplier(self, age_days: int) -> float:
        """More recent catalysts weigh more; linear-ish decay over the lookback."""
        if age_days <= 3:
            return 1.0
        if age_days <= 7:
            return 0.85
        if age_days <= 14:
            return 0.6
        if age_days <= self.cfg.lookback_days:
            return 0.35
        return 0.0

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_cik_map(self, candidates: List[Candidate]) -> Dict[str, Union[int, str]]:
        """Resolve candidate tickers to CIK numbers via EDGAR company_tickers.json."""
        try:
            tickers = [c.ticker for c in candidates]
            raw_map: Dict[str, Any] = self._edgar.load_company_tickers()
            ticker_to_cik: Dict[str, Union[int, str]] = {}
            for entry in raw_map.values():
                t = str(entry.get("ticker", "")).upper()
                c_str = entry.get("cik_str")
                if t and c_str:
                    ticker_to_cik[t] = int(c_str) if str(c_str).isdigit() else c_str

            result = {t: ticker_to_cik[t] for t in tickers if t in ticker_to_cik}
            _logger.info("CIK map: resolved %d/%d tickers", len(result), len(tickers))
            return result
        except Exception:
            _logger.exception("Failed to build CIK map")
            return {}

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
