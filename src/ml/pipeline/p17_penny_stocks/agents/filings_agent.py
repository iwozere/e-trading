"""
P17 Filings Agent

Stage 5.6: annotate candidates with recent insider and ownership filings, read
**cache-only** from the per-day EDGAR files the P15 daily bundle writes at 06:00 UTC
(two hours before P17), so this stage adds no EDGAR traffic:

  * Form 4  — DATA_CACHE_DIR/edgar/13f/form4/{date}.csv.gz (one row per transaction,
              issuer ticker from the XML). Open-market buys = code P / acquired,
              sells = code S.
  * 13D/G   — DATA_CACHE_DIR/edgar/13f/13dg/{date}.csv.gz (EDGAR form.idx lines; each
              filing is listed under both the filer and the subject company, so the
              subject is matched by the candidate's CIK).

Window: filings dated in (target_date − lookback_days, target_date].

Event study 2026-11-07 (CR-2026-11-06 §8, 314 tickers, May–Nov 2026): an initial
13G in the last 5 days → fwd10 mean +6.8% / median +1.8% / win 54% vs baseline
−2.0% / −4.0% / 37% (107 episodes, bootstrap p = 0.004). Insider buys, 13D and
insider sells showed no reliable edge — they are recorded for future evaluation
and shown as signals only. Nothing here changes the score.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.downloader.edgar_downloader import EdgarDownloader
from src.ml.pipeline.p17_penny_stocks.config import P17FilingsConfig
from src.ml.pipeline.p17_penny_stocks.models.candidate import Candidate
from src.ml.pipeline.shared.edgar_cik import build_cik_map
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def _cik_key(cik: object) -> str:
    """CIK (int or zero-padded string) → leading-zero-free string key."""
    try:
        return str(int(str(cik).strip()))
    except (TypeError, ValueError):
        return str(cik).strip().lstrip("0")


class FilingsAgent:
    """Populate Form 4 and 13D/G fields on each candidate from the P15 EDGAR cache."""

    def __init__(
        self,
        config: P17FilingsConfig,
        target_date: str,
        edgar: EdgarDownloader | None = None,
    ) -> None:
        self.cfg = config
        self.target = date.fromisoformat(target_date)
        self._edgar = edgar or EdgarDownloader()

    def window_days(self) -> List[date]:
        """Calendar days in (target − lookback, target]."""
        return [self.target - timedelta(days=i) for i in range(self.cfg.lookback_days)]

    def run(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Enrich candidates in place.

        Args:
            candidates: Candidate list (ticker set).

        Returns:
            The same list.
        """
        if not candidates:
            return candidates
        days = self.window_days()
        form4 = self._load_form4(days)
        dg = self._load_13dg(days)
        self._apply_form4(candidates, form4)
        self._apply_13dg(candidates, dg)
        _logger.info(
            "Filings: %d Form 4 rows / %d 13D/G rows over %s..%s | buys on %d, sells on %d, fresh 13G %d, fresh 13D %d",
            len(form4),
            len(dg),
            days[-1],
            days[0],
            sum(1 for c in candidates if c.insider_buy_count),
            sum(1 for c in candidates if c.insider_sell_count),
            sum(1 for c in candidates if c.fresh_13g),
            sum(1 for c in candidates if c.fresh_13d),
        )
        return candidates

    # ── Loading (cache only) ───────────────────────────────────────────────

    def _load_form4(self, days: List[date]) -> pd.DataFrame:
        frames = []
        missing = []
        for d in days:
            if d.weekday() >= 5:
                continue
            try:
                df = self._edgar.download_form4_filings(as_of_date=d, allow_network=False)
            except Exception:
                _logger.exception("Could not read Form 4 cache for %s", d)
                continue
            if df is None or df.empty:
                if not (self._edgar._form4_dir / f"{d}.csv.gz").exists():
                    missing.append(str(d))
                continue
            frames.append(df)
        if missing:
            _logger.warning("Form 4 cache missing for %s (P15 edgar_form4 not run yet?)", ", ".join(missing))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_13dg(self, days: List[date]) -> pd.DataFrame:
        frames = []
        missing = []
        for d in days:
            if d.weekday() >= 5:
                continue
            path = self._edgar._13dg_dir / f"{d}.csv.gz"
            if not path.exists():
                missing.append(str(d))
                continue
            try:
                frames.append(pd.read_csv(path, compression="gzip", dtype=str))
            except Exception:
                _logger.exception("Could not read 13D/G cache %s", path)
        if missing:
            _logger.warning("13D/G cache missing for %s (P15 edgar_13dg not run yet?)", ", ".join(missing))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── Applying ───────────────────────────────────────────────────────────

    def _apply_form4(self, candidates: List[Candidate], form4: pd.DataFrame) -> None:
        if form4.empty or "ticker" not in form4.columns:
            return
        f4 = form4.copy()
        f4["ticker"] = f4["ticker"].astype(str).str.upper().str.strip()
        if "is_derivative" in f4.columns:
            f4 = f4.loc[~f4["is_derivative"].astype(str).str.lower().eq("true")]
        if "total_value_usd" in f4.columns:
            f4["total_value_usd"] = pd.to_numeric(f4["total_value_usd"], errors="coerce").fillna(0.0)
        else:
            f4["total_value_usd"] = 0.0
        buys = f4.loc[(f4["transaction_code"] == "P") & (f4["acquired_disposed_code"] == "A")]
        sells = f4.loc[f4["transaction_code"] == "S"]
        buy_groups = {t: g for t, g in buys.groupby("ticker")}
        sell_groups = {t: g for t, g in sells.groupby("ticker")}
        for c in candidates:
            g = buy_groups.get(c.ticker.upper())
            if g is not None:
                c.insider_buy_count = len(g)
                c.insider_buy_value = float(g["total_value_usd"].sum())
                c.insider_buyers = int(g["insider_name"].nunique())
            s = sell_groups.get(c.ticker.upper())
            if s is not None:
                c.insider_sell_count = len(s)
                c.insider_sell_value = float(s["total_value_usd"].sum())

    def _apply_13dg(self, candidates: List[Candidate], dg: pd.DataFrame) -> None:
        if dg.empty or "cik" not in dg.columns:
            return
        cik_map = build_cik_map(self._edgar, [c.ticker for c in candidates])
        by_cik: Dict[str, Set[str]] = {}
        for cik, form in zip(dg["cik"].map(_cik_key), dg["form_type"].map(_norm_form)):
            by_cik.setdefault(cik, set()).add(form)
        for c in candidates:
            cik = cik_map.get(c.ticker.upper())
            if cik is None:
                continue
            forms = by_cik.get(_cik_key(cik), set())
            c.fresh_13g = "SC 13G" in forms
            c.fresh_13d = "SC 13D" in forms
            c.fresh_13dg_amend = bool(forms & {"SC 13G/A", "SC 13D/A"})


def _norm_form(form: object) -> str:
    """Normalise a 13D/G form type: form.idx uses `SC 13G`, EFTS/new EDGAR uses `SCHEDULE 13G`."""
    text = " ".join(str(form).upper().split())
    return "SC " + text[len("SCHEDULE "):] if text.startswith("SCHEDULE ") else text
