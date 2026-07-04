"""
P17 Dilution Agent

Detects dilution risk signals from SEC EDGAR filing history.

Checks (in the filing's recent submissions JSON):
  - S-3 / S-3ASR: shelf registration → penalty_shelf_offering
  - 424B*: prospectus supplement (ATM offering) → penalty_atm_offering
  - 8-K with reverse-split keywords → penalty_reverse_split
  - 8-K with convertible/ATM keywords → penalty_convertible_debt
  - 8-K with warrant keywords → penalty_warrant_issuance

Total penalty is the SUM of all applicable penalties.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Union

PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p17_penny_stocks.config import P17ScoringConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

_SHELF_FORMS = {"S-3", "S-3ASR", "S-1", "S-1/A"}
_ATM_FORMS = {"424B1", "424B2", "424B3", "424B4", "424B5", "424B8", "FWP"}
_REVERSE_SPLIT_KEYWORDS = {"reverse split", "reverse stock split", "rs effective"}
_CONVERTIBLE_KEYWORDS = {
    "convertible note",
    "convertible debenture",
    "at-the-market",
    "atm agreement",
    "equity distribution",
}
_WARRANT_KEYWORDS = {"warrant", "warrants issued", "warrant exercise"}


class DilutionAgent:
    """
    Stage 5: Detect dilution risk from SEC EDGAR filings.

    Populates on each Candidate:
      - dilution_penalty (sum of applicable penalties)
      - dilution_signals (list of human-readable signal names)
    """

    def __init__(
        self,
        scoring_config: P17ScoringConfig,
        results_dir: Path,
        target_date: str,
        lookback_days_shelf: int = 365,
        lookback_days_atm: int = 180,
        lookback_days_8k: int = 90,
    ) -> None:
        self.cfg = scoring_config
        self.results_dir = results_dir
        self.target_date = target_date
        self.lookback_shelf = timedelta(days=lookback_days_shelf)
        self.lookback_atm = timedelta(days=lookback_days_atm)
        self.lookback_8k = timedelta(days=lookback_days_8k)
        self._edgar = EdgarDownloader()

    def run(
        self,
        candidates: List[Candidate],
        force_refresh: bool = False,
    ) -> List[Candidate]:
        """
        Populate dilution_penalty and dilution_signals on each candidate.

        Args:
            candidates: Candidate list to enrich.
            force_refresh: Bypass EDGAR cache.

        Returns:
            Same list with dilution fields set.
        """
        _logger.info("Dilution agent: analysing %d candidates via EDGAR", len(candidates))

        cik_map = self._build_cik_map(candidates)
        if not cik_map:
            _logger.warning("CIK map empty — dilution detection skipped")
            return candidates

        today = datetime.strptime(self.target_date, "%Y-%m-%d")
        since_shelf = today - self.lookback_shelf

        enriched, skipped = 0, 0
        for c in candidates:
            cik = cik_map.get(c.ticker.upper())
            if cik is None:
                skipped += 1
                continue
            self._analyse_filings(c, cik, since_shelf, force_refresh)
            enriched += 1

        _logger.info("Dilution agent: %d analysed, %d skipped (no CIK)", enriched, skipped)
        return candidates

    # ── Filing analysis ────────────────────────────────────────────────────

    def _analyse_filings(
        self,
        c: Candidate,
        cik: Union[int, str],
        since: datetime,
        force_refresh: bool,
    ) -> None:
        try:
            filings = self._edgar.get_recent_filings(cik, since=since, force_refresh=force_refresh)
        except Exception:
            _logger.debug("Could not fetch filings for %s (CIK %s)", c.ticker, cik)
            return

        today = datetime.strptime(self.target_date, "%Y-%m-%d")
        penalty = 0.0
        signals: List[str] = []

        for filing in filings:
            form = str(filing.get("form") or "").upper().strip()
            date_str = str(filing.get("filingDate") or "")
            description = str(filing.get("description") or "").lower()

            filing_date = self._parse_date(date_str)
            if filing_date is None:
                continue

            age = today - filing_date

            if form in _SHELF_FORMS and age <= self.lookback_shelf:
                penalty += self.cfg.penalty_shelf_offering
                signals.append(f"shelf_{form}_{date_str[:7]}")

            elif form in _ATM_FORMS and age <= self.lookback_atm:
                penalty += self.cfg.penalty_atm_offering
                signals.append(f"atm_offering_{date_str[:7]}")

            elif form == "8-K" and age <= self.lookback_8k:
                if any(kw in description for kw in _REVERSE_SPLIT_KEYWORDS):
                    penalty += self.cfg.penalty_reverse_split
                    signals.append(f"reverse_split_{date_str[:7]}")
                if any(kw in description for kw in _CONVERTIBLE_KEYWORDS):
                    penalty += self.cfg.penalty_convertible_debt
                    signals.append(f"convertible_{date_str[:7]}")
                if any(kw in description for kw in _WARRANT_KEYWORDS):
                    penalty += self.cfg.penalty_warrant_issuance
                    signals.append(f"warrants_{date_str[:7]}")

        if penalty > 0:
            c.dilution_penalty = round(penalty, 1)
            c.dilution_signals = signals
            _logger.debug("%s dilution_penalty=%.0f signals=%s", c.ticker, penalty, signals)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_cik_map(self, candidates: List[Candidate]) -> Dict[str, Union[int, str]]:
        """
        Resolve candidate tickers to CIK numbers using EDGAR company_tickers.json.
        Returns Dict[ticker → cik].
        """
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
